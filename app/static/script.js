// ══════════════════════════════════════════════════════════════════════════════
//  CONSTANTS
// ══════════════════════════════════════════════════════════════════════════════

const WS_URL        = `ws://${location.host}/ws/new`;
const NUM_BARS      = 32;
const VOLUME_THRESH = 18;   // used only for visualizer bar colour

// ══════════════════════════════════════════════════════════════════════════════
//  STATE
// ══════════════════════════════════════════════════════════════════════════════

let ws              = null;
let sessionId       = null;
let mediaStream     = null;
let audioCtx        = null;
let analyser        = null;
let animFrameId     = null;
let vadInstance     = null;        // @ricky0123/vad-web MicVAD instance

let isAgentSpeaking = false;
let isPlayingAudio  = false;
let agentAudioQueue = [];

// ══════════════════════════════════════════════════════════════════════════════
//  LOGGER
// ══════════════════════════════════════════════════════════════════════════════

function log(level, ...args) {
  const ts     = new Date().toISOString().substring(11, 23);
  const sid    = sessionId ? `[${sessionId.substring(0, 8)}]` : '[no-session]';
  const prefix = `[${ts}] ${sid}`;
  switch (level) {
    case 'debug': console.debug(prefix, ...args); break;
    case 'warn':  console.warn(prefix, ...args);  break;
    case 'error': console.error(prefix, ...args); break;
    default:      console.log(prefix, ...args);
  }
}

// ══════════════════════════════════════════════════════════════════════════════
//  DOM REFS
// ══════════════════════════════════════════════════════════════════════════════

const conversation = document.getElementById('conversation');
const emptyState   = document.getElementById('emptyState');
const statusPill   = document.getElementById('statusPill');
const statusText   = document.getElementById('statusText');
const sessionInfo  = document.getElementById('sessionInfo');
const btnStart     = document.getElementById('btnStart');
const btnEnd       = document.getElementById('btnEnd');
const barsEl       = document.getElementById('bars');
const vadStatusEl  = document.getElementById('vadStatus');

// Build visualizer bars
for (let i = 0; i < NUM_BARS; i++) {
  const b = document.createElement('div');
  b.className = 'bar';
  barsEl.appendChild(b);
}
const bars = Array.from(barsEl.querySelectorAll('.bar'));

// ══════════════════════════════════════════════════════════════════════════════
//  STATUS HELPERS
// ══════════════════════════════════════════════════════════════════════════════

function setStatus(state, label) {
  statusPill.className = `status-pill ${state}`;
  statusText.textContent = label;
  log('debug', `Status → ${state} — "${label}"`);
}

function setVadStatus(cls, label) {
  vadStatusEl.className = `vad-status ${cls}`;
  vadStatusEl.textContent = label;
}

// ══════════════════════════════════════════════════════════════════════════════
//  MESSAGES
// ══════════════════════════════════════════════════════════════════════════════

function addMessage(role, text) {
  if (emptyState && emptyState.parentNode) emptyState.remove();

  const msg    = document.createElement('div');
  msg.className = `msg ${role}`;

  const label  = document.createElement('div');
  label.className   = 'msg-label';
  label.textContent = role === 'customer' ? 'You' : 'Priya';

  const bubble = document.createElement('div');
  bubble.className   = 'msg-bubble';
  bubble.textContent = text;

  msg.appendChild(label);
  msg.appendChild(bubble);
  conversation.appendChild(msg);
  conversation.scrollTop = conversation.scrollHeight;

  log('info', `Message — role=${role} — "${text.substring(0, 60)}${text.length > 60 ? '...' : ''}"`);
}

// ══════════════════════════════════════════════════════════════════════════════
//  START CALL
// ══════════════════════════════════════════════════════════════════════════════

btnStart.addEventListener('click', async () => {
  log('info', 'Start button clicked');
  btnStart.disabled = true;

  // 1. Request microphone
  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation    : true,
        noiseSuppression    : true,
        autoGainControl     : true,
        channelCount        : 1,
        sampleRate          : 16000,
        googNoiseSuppression: true,
        googHighpassFilter  : true,
      }
    });
    log('info', 'Microphone access granted');
  } catch (err) {
    log('error', `Microphone denied — ${err.message}`);
    alert('Microphone access denied. Please allow microphone access and try again.');
    btnStart.disabled = false;
    return;
  }

  // 2. Web Audio API — analyser node for visualizer bars only
  audioCtx = new AudioContext({ sampleRate: 16000 });
  analyser = audioCtx.createAnalyser();
  analyser.fftSize = 256;
  const source = audioCtx.createMediaStreamSource(mediaStream);
  source.connect(analyser);
  log('debug', `AudioContext ready — sampleRate=${audioCtx.sampleRate}Hz`);

  // 3. Open WebSocket
  log('info', `Connecting WebSocket — ${WS_URL}`);
  ws = new WebSocket(WS_URL);

  ws.onopen = () => {
    log('info', 'WebSocket connected');
    btnEnd.disabled = false;
    setStatus('active', 'Connected');
    sessionInfo.innerHTML = `session — <span>connecting…</span>`;
    startVolumeLoop();
    initVAD();   // start VAD once WS is open
  };

  ws.onmessage = (event) => {
    const msg = JSON.parse(event.data);
    log('debug', `WS message — type=${msg.type}`);
    handleServerMessage(msg);
  };

  ws.onclose = (event) => {
    log('info', `WebSocket closed — code=${event.code}`);
    setStatus('idle', 'Idle');
    setVadStatus('', 'vad inactive');
    btnStart.disabled = false;
    btnEnd.disabled   = true;
    stopVolumeLoop();
    destroyVAD();
  };

  ws.onerror = (err) => {
    log('error', `WebSocket error — ${err.message || 'unknown'}`);
  };
});

// ══════════════════════════════════════════════════════════════════════════════
//  END CALL
// ══════════════════════════════════════════════════════════════════════════════

btnEnd.addEventListener('click', () => {
  log('info', 'End button clicked — closing WebSocket');
  if (ws) ws.close();
});

// ══════════════════════════════════════════════════════════════════════════════
//  VAD — @ricky0123/vad-web (Silero ML model)
//
//  Replaces MediaRecorder + silence timer entirely.
//
//  onSpeechStart:
//    Agent speaking  → barge-in: clear queue + send interrupt to server
//    Agent silent    → update UI to show listening state
//
//  onSpeechEnd(audio):
//    audio = Float32Array (16kHz mono, range -1..1) — full speech segment
//    Convert Float32 → int16 PCM → WAV header → base64 → send to server
//
//  onVADMisfire:
//    Silero detected something too short — noise, breath, cough. Ignored.
// ══════════════════════════════════════════════════════════════════════════════

async function initVAD() {
  log('info', 'Initialising VAD…');
  setVadStatus('', 'vad loading…');

  try {
    vadInstance = await vad.MicVAD.new({
      onnxWASMBasePath : 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/',
      baseAssetPath    : 'https://cdn.jsdelivr.net/npm/@ricky0123/vad-web@0.0.29/dist/',
      stream           : mediaStream,

      positiveSpeechThreshold : 0.75,  // was 0.8 — slightly lower so light speech triggers
      negativeSpeechThreshold : 0.65,  // was 0.6 — higher so background noise stops faster
      minSpeechFrames         : 4,     // was 5 — trigger slightly sooner on light speech
      redemptionFrames        : 10,    // was 6 — more patience before cutting off speech
      preSpeechPadFrames      : 5,     // was 3 — more pre-speech context captured

      onSpeechStart: () => {
        log('info', 'VAD — speech start');
        if (isAgentSpeaking || isPlayingAudio) {
          log('info', 'Barge-in — agent interrupted by customer');
          setVadStatus('barge', 'barge-in detected');
          triggerBargeIn();
        } else {
          setVadStatus('speech', 'speech detected');
          setStatus('active', 'Listening');
        }
      },

      onSpeechEnd: (audio) => {
        log('info', `VAD — speech end — ${audio.length} samples (${(audio.length / 16000).toFixed(2)}s)`);
        if (isAgentSpeaking || isPlayingAudio) {
          log('debug', 'onSpeechEnd discarded — barge-in already sent');
          setVadStatus('', 'vad listening');
          return;
        }
        setVadStatus('', 'vad listening');
        setStatus('thinking', 'Thinking');
        const wavBuffer = float32ToWav(audio);
        const base64    = arrayBufferToBase64(wavBuffer);
        wsSend({ type: 'audio_chunk', data: base64 });
        wsSend({ type: 'audio_end' });
        log('info', `Sent audio_chunk + audio_end — wav=${(wavBuffer.byteLength / 1024).toFixed(1)}KB`);
      },

      onVADMisfire: () => {
        log('debug', 'VAD misfire — ignored');
        setVadStatus('', 'vad listening');
      },
    });

    await vadInstance.start();
    setVadStatus('', 'vad listening');
    log('info', 'VAD started and listening');

  } catch (err) {
    log('error', `VAD init failed — ${err.message}`);
    setVadStatus('error', 'vad error');
  }
}

async function destroyVAD() {
  if (vadInstance) {
    try {
      await vadInstance.destroy();
      log('info', 'VAD destroyed');
    } catch (e) {
      log('warn', `VAD destroy error — ${e.message}`);
    }
    vadInstance = null;
  }
}

// ══════════════════════════════════════════════════════════════════════════════
//  SERVER MESSAGE HANDLER
// ══════════════════════════════════════════════════════════════════════════════

function handleServerMessage(msg) {
  switch (msg.type) {

    case 'session_id':
      sessionId = msg.session_id;
      sessionInfo.innerHTML = `session — <span>${sessionId}</span>`;
      log('info', `Session ID — ${sessionId}`);
      break;

    case 'listening':
      // Server is ready — VAD is already running continuously, just update UI
      log('info', 'Server ready — VAD already listening');
      isAgentSpeaking = false;
      setStatus('active', 'Listening');
      break;

    case 'transcript':
      log('info', `Transcript — "${msg.text}"`);
      addMessage('customer', msg.text);
      setStatus('thinking', 'Thinking');
      break;

    case 'audio_chunk': {
      // Cartesia sends raw PCM (pcm_s16le, 16kHz mono) — wrap in WAV header
      const binary = atob(msg.data);
      const bytes  = new Uint8Array(binary.length);
      for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);

      const wavBuffer = buildWavBuffer(bytes.buffer);
      log('debug', `Audio chunk — pcm=${(bytes.length / 1024).toFixed(1)}KB — queue=${agentAudioQueue.length}`);

      agentAudioQueue.push(wavBuffer);
      if (!isPlayingAudio) playNextChunk();
      break;
    }

    case 'audio_end':
      log('info', 'audio_end — agent finished speaking');
      isAgentSpeaking = false;
      break;

    case 'reply_text':
      log('info', `Reply — "${msg.text.substring(0, 60)}…"`);
      addMessage('agent', msg.text);
      break;

    default:
      log('warn', `Unknown message type — ${msg.type}`);
  }
}

// ══════════════════════════════════════════════════════════════════════════════
//  AUDIO PLAYBACK
// ══════════════════════════════════════════════════════════════════════════════

async function playNextChunk() {
  if (agentAudioQueue.length === 0) {
    log('debug', 'Queue empty — playback complete');
    isPlayingAudio  = false;
    isAgentSpeaking = false;
    wsSend({ type: 'ready' });
    setStatus('active', 'Listening');
    return;
  }

  isPlayingAudio  = true;
  isAgentSpeaking = true;
  setStatus('speaking', 'Speaking');

  const buffer = agentAudioQueue.shift();
  log('debug', `Playing — ${(buffer.byteLength / 1024).toFixed(1)}KB — queue=${agentAudioQueue.length}`);

  try {
    const decoded = await audioCtx.decodeAudioData(buffer);
    const src     = audioCtx.createBufferSource();
    src.buffer    = decoded;
    src.connect(audioCtx.destination);
    src.onended   = () => playNextChunk();
    src.start();
    log('debug', `Playing — duration=${decoded.duration.toFixed(2)}s`);
  } catch (err) {
    log('error', `Audio decode error — ${err.message}`);
    playNextChunk();   // skip bad chunk, continue queue
  }
}

function clearAudioQueue() {
  const n = agentAudioQueue.length;
  agentAudioQueue = [];
  isPlayingAudio  = false;
  isAgentSpeaking = false;
  log('info', `Audio queue cleared — ${n} chunks discarded`);
}

// ══════════════════════════════════════════════════════════════════════════════
//  BARGE-IN
// ══════════════════════════════════════════════════════════════════════════════

function triggerBargeIn() {
  clearAudioQueue();
  wsSend({ type: 'interrupt' });
  setStatus('active', 'Listening');
  log('info', 'Barge-in: queue cleared + interrupt sent to server');
}

// ══════════════════════════════════════════════════════════════════════════════
//  VISUALIZER — analyser node drives bars only, no barge-in logic here
// ══════════════════════════════════════════════════════════════════════════════

function startVolumeLoop() {
  const dataArray = new Uint8Array(analyser.frequencyBinCount);

  function loop() {
    animFrameId = requestAnimationFrame(loop);
    analyser.getByteFrequencyData(dataArray);
    updateBars(dataArray);
  }

  loop();
  log('info', 'Volume loop started');
}

function stopVolumeLoop() {
  if (animFrameId) {
    cancelAnimationFrame(animFrameId);
    animFrameId = null;
    log('info', 'Volume loop stopped');
  }
}

function updateBars(dataArray) {
  const step = Math.floor(dataArray.length / NUM_BARS);
  bars.forEach((bar, i) => {
    const val    = dataArray[i * step] || 0;
    const height = Math.max(4, (val / 255) * 28);
    bar.style.height = `${height}px`;
    bar.classList.toggle('active', val > VOLUME_THRESH * 2);
  });
}

// ══════════════════════════════════════════════════════════════════════════════
//  AUDIO CONVERSION HELPERS
// ══════════════════════════════════════════════════════════════════════════════

/**
 * float32ToWav
 * Converts Silero's Float32Array output (-1..1, 16kHz mono)
 * into a complete WAV ArrayBuffer ready to send to the server.
 */
function float32ToWav(float32Array) {
  const numSamples  = float32Array.length;
  const sampleRate  = 16000;
  const numChannels = 1;
  const bitDepth    = 16;
  const byteRate    = sampleRate * numChannels * bitDepth / 8;
  const blockAlign  = numChannels * bitDepth / 8;
  const dataSize    = numSamples * blockAlign;
  const buffer      = new ArrayBuffer(44 + dataSize);
  const view        = new DataView(buffer);

  writeString(view, 0,  'RIFF');
  view.setUint32(4,  buffer.byteLength - 8, true);
  writeString(view, 8,  'WAVE');
  writeString(view, 12, 'fmt ');
  view.setUint32(16, 16,           true);
  view.setUint16(20, 1,            true);   // PCM
  view.setUint16(22, numChannels,  true);
  view.setUint32(24, sampleRate,   true);
  view.setUint32(28, byteRate,     true);
  view.setUint16(32, blockAlign,   true);
  view.setUint16(34, bitDepth,     true);
  writeString(view, 36, 'data');
  view.setUint32(40, dataSize, true);

  // Float32 → Int16
  const pcm = new Int16Array(buffer, 44);
  for (let i = 0; i < numSamples; i++) {
    const s = Math.max(-1, Math.min(1, float32Array[i]));
    pcm[i]  = s < 0 ? s * 0x8000 : s * 0x7FFF;
  }

  return buffer;
}

/**
 * buildWavBuffer
 * Wraps raw PCM bytes from Cartesia Sonic (pcm_s16le, 16kHz mono)
 * in a WAV header so decodeAudioData() can play them.
 */
function buildWavBuffer(pcmBuffer) {
  const sampleRate    = 16000;
  const numChannels   = 1;
  const bitsPerSample = 16;
  const byteRate      = sampleRate * numChannels * bitsPerSample / 8;
  const blockAlign    = numChannels * bitsPerSample / 8;
  const dataSize      = pcmBuffer.byteLength;
  const wav           = new ArrayBuffer(44 + dataSize);
  const view          = new DataView(wav);

  writeString(view, 0,  'RIFF');
  view.setUint32(4,  wav.byteLength - 8, true);
  writeString(view, 8,  'WAVE');
  writeString(view, 12, 'fmt ');
  view.setUint32(16, 16,            true);
  view.setUint16(20, 1,             true);
  view.setUint16(22, numChannels,   true);
  view.setUint32(24, sampleRate,    true);
  view.setUint32(28, byteRate,      true);
  view.setUint16(32, blockAlign,    true);
  view.setUint16(34, bitsPerSample, true);
  writeString(view, 36, 'data');
  view.setUint32(40, dataSize, true);

  new Uint8Array(wav, 44).set(new Uint8Array(pcmBuffer));
  return wav;
}

function writeString(view, offset, str) {
  for (let i = 0; i < str.length; i++) {
    view.setUint8(offset + i, str.charCodeAt(i));
  }
}

function arrayBufferToBase64(buffer) {
  const bytes = new Uint8Array(buffer);
  let   bin   = '';
  for (let i = 0; i < bytes.byteLength; i++) bin += String.fromCharCode(bytes[i]);
  return btoa(bin);
}

// ══════════════════════════════════════════════════════════════════════════════
//  WEBSOCKET SEND
// ══════════════════════════════════════════════════════════════════════════════

function wsSend(data) {
  if (ws && ws.readyState === WebSocket.OPEN) {
    if (data.type !== 'audio_chunk') log('debug', `WS send — type=${data.type}`);
    ws.send(JSON.stringify(data));
  } else {
    log('warn', `wsSend failed — WS not open — type=${data.type}`);
  }
}