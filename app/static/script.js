// ── Constants ────────────────────────────────────────────────────────────────
const WS_URL           = `ws://${location.host}/ws/new`;
const CHUNK_INTERVAL   = 250;
const SILENCE_TIMEOUT  = 1800;
const VOLUME_THRESHOLD = 18;
const BARGE_MUTE_MS    = 300;
const BARGE_MIN_MS     = 500;
const NUM_BARS         = 24;

// ── State ────────────────────────────────────────────────────────────────────
let ws             = null;
let sessionId      = null;
let mediaStream    = null;
let mediaRecorder  = null;
let audioCtx       = null;
let analyser       = null;
let animFrameId    = null;

let isRecording     = false;
let isAgentSpeaking = false;
let agentAudioQueue = [];
let isPlayingAudio  = false;
let isFirstAgentChunk = false;   

let silenceTimer    = null;
let bargeStartTime  = null;
let bargeMuteUntil  = 0;
let bargeSilenceFrames = 0;

// ── Logger ────────────────────────────────────────────────────────────────────
// Centralised logger — prefix every log with timestamp and session ID
function log(level, ...args) {
  const ts      = new Date().toISOString().substring(11, 23);  // HH:MM:SS.mmm
  const sid     = sessionId ? `[${sessionId.substring(0, 8)}]` : '[no-session]';
  const prefix  = `[${ts}] ${sid}`;

  switch (level) {
    case 'debug': console.debug(prefix, ...args); break;
    case 'warn':  console.warn(prefix, ...args);  break;
    case 'error': console.error(prefix, ...args); break;
    default:      console.log(prefix, ...args);
  }
}

// ── DOM ──────────────────────────────────────────────────────────────────────
const conversation = document.getElementById('conversation');
const emptyState   = document.getElementById('emptyState');
const statusPill   = document.getElementById('statusPill');
const statusText   = document.getElementById('statusText');
const sessionInfo  = document.getElementById('sessionInfo');
const btnStart     = document.getElementById('btnStart');
const btnEnd       = document.getElementById('btnEnd');
const barsEl       = document.getElementById('bars');

// ── Build visualizer bars ─────────────────────────────────────────────────────
for (let i = 0; i < NUM_BARS; i++) {
  const b = document.createElement('div');
  b.className = 'bar';
  barsEl.appendChild(b);
}
const bars = Array.from(barsEl.querySelectorAll('.bar'));

// ── Status helpers ────────────────────────────────────────────────────────────
function setStatus(state, label) {
  statusPill.className = `status-pill ${state}`;
  statusText.textContent = label;
  log('debug', `Status → ${state} — "${label}"`);
}

// ── Add message to UI ─────────────────────────────────────────────────────────
function addMessage(role, text) {
  if (emptyState) emptyState.remove();

  const msg    = document.createElement('div');
  msg.className = `msg ${role}`;

  const label  = document.createElement('div');
  label.className = 'msg-label';
  label.textContent = role === 'customer' ? 'You' : 'Priya';

  const bubble = document.createElement('div');
  bubble.className = 'msg-bubble';
  bubble.textContent = text;

  msg.appendChild(label);
  msg.appendChild(bubble);
  conversation.appendChild(msg);
  conversation.scrollTop = conversation.scrollHeight;

  log('info', `Message added — role=${role} — "${text.substring(0, 60)}${text.length > 60 ? '...' : ''}"`);
}

// ── Start call ────────────────────────────────────────────────────────────────
btnStart.addEventListener('click', async () => {
  log('info', 'Start button clicked — requesting microphone');
  btnStart.disabled = true;

  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation : true,
        noiseSuppression : true,
        autoGainControl  : true,
      }
    });
    log('info', 'Microphone access granted');
  } catch (err) {
    log('error', `Microphone access denied — ${err.message}`);
    alert('Microphone access denied. Please allow microphone access and try again.');
    btnStart.disabled = false;
    return;
  }

  // Set up Web Audio API
  audioCtx  = new AudioContext({ sampleRate: 16000 });
  analyser  = audioCtx.createAnalyser();
  analyser.fftSize = 256;
  const source = audioCtx.createMediaStreamSource(mediaStream);
  source.connect(analyser);
  log('debug', `AudioContext created — sampleRate=${audioCtx.sampleRate}Hz`);

  // Connect to WebSocket
  log('info', `Connecting to WebSocket — ${WS_URL}`);
  ws = new WebSocket(WS_URL);
  ws.binaryType = 'arraybuffer';

  ws.onopen = () => {
    log('info', 'WebSocket connected');
    btnEnd.disabled = false;
    setStatus('active', 'Connected');
    sessionInfo.innerHTML = `session — <span>connecting...</span>`;
    startVolumeLoop();
  };

  ws.onmessage = (event) => {
    const msg = JSON.parse(event.data);
    log('debug', `WebSocket message received — type=${msg.type}`);
    handleServerMessage(msg);
  };

  ws.onclose = (event) => {
    log('info', `WebSocket closed — code=${event.code} reason="${event.reason || 'none'}"`);
    setStatus('idle', 'Idle');
    btnStart.disabled = false;
    btnEnd.disabled   = true;
    stopRecording();
    stopVolumeLoop();
  };

  ws.onerror = (err) => {
    log('error', `WebSocket error — ${err.message || 'unknown error'}`);
  };
});

// ── End call ──────────────────────────────────────────────────────────────────
btnEnd.addEventListener('click', () => {
  log('info', 'End button clicked — closing WebSocket');
  if (ws) ws.close();
});

// ── Handle server messages ────────────────────────────────────────────────────
function handleServerMessage(msg) {
  switch (msg.type) {

    case 'session_id':
      sessionId = msg.session_id;
      sessionInfo.innerHTML = `session — <span>${sessionId}</span>`;
      log('info', `Session ID assigned — ${sessionId}`);
      break;

    case 'listening':
      log('info', 'Server ready — starting recording');
      isAgentSpeaking = false;
      setStatus('active', 'Listening');
      if (!isPlayingAudio) {
        startRecording();   // ← only starts if nothing is playing
      }
      // if isPlayingAudio=true, playNextChunk() will send "ready"
      // server responds with "listening" → hits this case again
      // by then isPlayingAudio=false → startRecording() runs
      break;

    case 'transcript':
      log('info', `Transcript received — "${msg.text}"`);
      addMessage('customer', msg.text);
      stopRecording();
      setStatus('thinking', 'Thinking');
      break;

    case 'audio_chunk':
      const binary  = atob(msg.data);
      const bytes   = new Uint8Array(binary.length);
      for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);

      // Raw PCM from Cartesia — wrap in WAV header before decoding
      const wavBuffer = buildWavBuffer(bytes.buffer);
      log('debug', `Audio chunk — pcm=${(bytes.length / 1024).toFixed(1)}KB — wav=${(wavBuffer.byteLength / 1024).toFixed(1)}KB — queue=${agentAudioQueue.length}`);

      agentAudioQueue.push(wavBuffer);
      log('debug', `Audio chunk queued — size=${(bytes.length / 1024).toFixed(1)}KB — queue=${agentAudioQueue.length}`);
      if (!isPlayingAudio){
        isFirstAgentChunk = true; 
        playNextChunk();
      } 
      break;

    case 'audio_end':
      log('info', 'Audio end received — agent finished speaking');
      break;

    case 'reply_text':
      log('info', `Reply text received — "${msg.text.substring(0, 60)}${msg.text.length > 60 ? '...' : ''}"`);
      addMessage('agent', msg.text);
      break;

    default:
      log('warn', `Unknown message type received — ${msg.type}`);
  }
}

// ── Audio recording ───────────────────────────────────────────────────────────
function startRecording() {
  if (isRecording || !mediaStream) {
    log('debug', `startRecording() skipped — isRecording=${isRecording} mediaStream=${!!mediaStream}`);
    return;
  }
  isRecording = true;
  log('info', 'Recording started');

  const chunks = [];

  mediaRecorder = new MediaRecorder(mediaStream, { mimeType: 'audio/webm;codecs=opus' });

  mediaRecorder.ondataavailable = (e) => {
    if (e.data.size > 0) {
      chunks.push(e.data);
      log('debug', `Audio data available — chunk size=${(e.data.size / 1024).toFixed(1)}KB — total chunks=${chunks.length}`);
    }
  };

  mediaRecorder.onstop = () => {
    log('info', `Recording stopped — building blob from ${chunks.length} chunks`);
    const blob   = new Blob(chunks, { type: 'audio/webm;codecs=opus' });
    log('info', `Blob created — size=${(blob.size / 1024).toFixed(1)}KB — sending to server`);
    const reader = new FileReader();
    reader.onload = () => {
      const base64 = reader.result.split(',')[1];
      wsSend({ type: 'audio_chunk', data: base64 });
      wsSend({ type: 'audio_end' });
      log('info', 'audio_chunk + audio_end sent to server');
    };
    reader.readAsDataURL(blob);
  };

  mediaRecorder.start();
  resetSilenceTimer();
}

function stopRecording() {
  if (!isRecording) {
    log('debug', 'stopRecording() called but not recording — skipped');
    return;
  }
  isRecording = false;
  clearTimeout(silenceTimer);
  if (mediaRecorder && mediaRecorder.state !== 'inactive') {
    log('info', 'Stopping MediaRecorder');
    mediaRecorder.stop();
  }
}

function resetSilenceTimer() {
  clearTimeout(silenceTimer);
  silenceTimer = setTimeout(() => {
    if (isRecording) {
      log('info', `Silence timeout reached (${SILENCE_TIMEOUT}ms) — stopping recording`);
      stopRecording();
    }
  }, SILENCE_TIMEOUT);
}

// ── Audio playback ────────────────────────────────────────────────────────────
async function playNextChunk() {
  if (agentAudioQueue.length === 0) {
    log('debug', 'Audio queue empty — playback complete');
    isPlayingAudio  = false;
    isAgentSpeaking = false;
    wsSend({ type: 'ready' });   // ← tells server browser finished playing
    return;
  }

  isPlayingAudio  = true;
  isAgentSpeaking = true;
  setStatus('speaking', 'Speaking');

  if (isFirstAgentChunk) {
    bargeMuteUntil    = Date.now() + BARGE_MUTE_MS;  // ← only set once
    bargeStartTime    = null;                          // ← reset sustain
    isFirstAgentChunk = false;                         // ← never set again this turn
    bargeSilenceFrames = 0;
  }

  const buffer = agentAudioQueue.shift();
  log('debug', `Playing audio chunk — size=${(buffer.byteLength / 1024).toFixed(1)}KB — remaining in queue=${agentAudioQueue.length}`);

  // Inspect raw bytes before decode attempt
  const view = new DataView(buffer);
  const firstBytes = [];
  for (let i = 0; i < Math.min(16, buffer.byteLength); i++) {
    firstBytes.push(view.getUint8(i).toString(16).padStart(2, '0'));
  }
  log('debug', `buffer inspect — size: ${buffer.byteLength} bytes | first 16 bytes: ${firstBytes.join(' ')}`);

  try {
    const decoded = await audioCtx.decodeAudioData(buffer);
    log('debug', `Audio decoded — duration=${decoded.duration.toFixed(2)}s`);
    const source  = audioCtx.createBufferSource();
    source.buffer = decoded;
    source.connect(audioCtx.destination);
    source.onended = () => {
      log('debug', 'Audio chunk playback ended — moving to next');
      playNextChunk();
    };
    source.start();
  } catch (err) {
    log('error', `Audio decode error — ${err.message}`);
    playNextChunk();
  }
}

function clearAudioQueue() {
  const cleared = agentAudioQueue.length;
  agentAudioQueue = [];
  isPlayingAudio  = false;
  isAgentSpeaking = false;
  log('info', `Audio queue cleared — ${cleared} chunks discarded`);
}

// ── Volume monitor + barge-in ─────────────────────────────────────────────────
function startVolumeLoop() {
  log('info', 'Volume monitoring loop started');
  const dataArray = new Uint8Array(analyser.frequencyBinCount);

  function loop() {
    animFrameId = requestAnimationFrame(loop);
    analyser.getByteFrequencyData(dataArray);

    let sum = 0;
    for (let i = 0; i < dataArray.length; i++) sum += dataArray[i];
    const avg = sum / dataArray.length;

    updateBars(dataArray);

    if (isAgentSpeaking) {
      const now = Date.now();

      if (now < bargeMuteUntil) return;

      if (avg < VOLUME_THRESHOLD) {
        if (bargeStartTime !== null) {
          bargeSilenceFrames++;
          if (bargeSilenceFrames > 5) {
            bargeStartTime     = null;
            bargeSilenceFrames = 0;
            log('debug', 'Barge-in sustain reset — silence exceeded tolerance');
          }
        }
        return;
      }
      bargeSilenceFrames = 0;

      if (!bargeStartTime) {
        log('debug', `Barge-in voice detected — volume=${avg.toFixed(1)} — waiting for sustain (${BARGE_MIN_MS}ms)`);
        bargeStartTime = now;
        return;
      }

      if (now - bargeStartTime >= BARGE_MIN_MS) {
        log('info', `Barge-in sustained for ${BARGE_MIN_MS}ms — triggering interrupt`);
        triggerBargeIn();
      }
    } else {
      if (isRecording && avg > VOLUME_THRESHOLD) {
        resetSilenceTimer();
      }
    }
  }

  loop();
}

function stopVolumeLoop() {
  if (animFrameId) {
    cancelAnimationFrame(animFrameId);
    animFrameId = null;
    log('info', 'Volume monitoring loop stopped');
  }
}

function updateBars(dataArray) {
  const step = Math.floor(dataArray.length / NUM_BARS);
  bars.forEach((bar, i) => {
    const val    = dataArray[i * step] || 0;
    const height = Math.max(4, (val / 255) * 28);
    bar.style.height = `${height}px`;
    bar.classList.toggle('active', val > VOLUME_THRESHOLD * 2);
  });
}

function triggerBargeIn() {
  log('info', 'Barge-in triggered — sending interrupt to server');
  bargeStartTime = null;
  bargeMuteUntil = 0;
  bargeSilenceFrames = 0; 
  clearAudioQueue();
  stopRecording();
  wsSend({ type: 'interrupt' });
  setStatus('active', 'Listening');
}

// ── WebSocket send ────────────────────────────────────────────────────────────
function wsSend(data) {
  if (ws && ws.readyState === WebSocket.OPEN) {
    // Don't log audio_chunk — too noisy (large base64 payload)
    if (data.type !== 'audio_chunk') {
      log('debug', `WebSocket send — type=${data.type}`);
    }
    ws.send(JSON.stringify(data));
  } else {
    log('warn', `wsSend() failed — WebSocket not open — type=${data.type} readyState=${ws?.readyState}`);
  }
}


// ── WAV header builder ────────────────────────────────────────────────────────
// Cartesia WebSocket sends raw PCM (pcm_s16le, 16000Hz, mono).
// decodeAudioData() needs a proper WAV container — so we build the header here.
function buildWavBuffer(pcmBuffer) {
  const numChannels  = 1;       // mono
  const sampleRate   = 16000;   // 16kHz — must match Cartesia output_format
  const bitsPerSample = 16;     // pcm_s16le
  const byteRate     = sampleRate * numChannels * bitsPerSample / 8;
  const blockAlign   = numChannels * bitsPerSample / 8;
  const dataSize     = pcmBuffer.byteLength;
  const headerSize   = 44;
  const totalSize    = headerSize + dataSize;

  const wav    = new ArrayBuffer(totalSize);
  const view   = new DataView(wav);

  // RIFF chunk
  writeString(view, 0,  'RIFF');
  view.setUint32(4,  totalSize - 8,  true);  // file size minus RIFF header
  writeString(view, 8,  'WAVE');

  // fmt chunk
  writeString(view, 12, 'fmt ');
  view.setUint32(16, 16,           true);   // chunk size = 16 for PCM
  view.setUint16(20, 1,            true);   // audio format = 1 (PCM)
  view.setUint16(22, numChannels,  true);
  view.setUint32(24, sampleRate,   true);
  view.setUint32(28, byteRate,     true);
  view.setUint16(32, blockAlign,   true);
  view.setUint16(34, bitsPerSample,true);

  // data chunk
  writeString(view, 36, 'data');
  view.setUint32(40, dataSize, true);

  // Copy raw PCM bytes after the header
  new Uint8Array(wav, headerSize).set(new Uint8Array(pcmBuffer));

   // ── Temporary size check ──
  console.log('buildWavBuffer — pcmSize:', pcmBuffer.byteLength, 'wavSize:', wav.byteLength, 'diff:', wav.byteLength - pcmBuffer.byteLength);


  return wav;
}

function writeString(view, offset, str) {
  for (let i = 0; i < str.length; i++) {
    view.setUint8(offset + i, str.charCodeAt(i));
  }
}