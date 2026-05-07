// ── Constants ────────────────────────────────────────────────────────────────
const WS_URL           = `ws://${location.host}/ws/new`;
const CHUNK_INTERVAL   = 250;    // ms — send audio chunk every 250ms
const SILENCE_TIMEOUT  = 1800;   // ms — stop recording after 1.8s of silence
const VOLUME_THRESHOLD = 18;     // 0-255 — minimum volume to count as speech
const BARGE_MUTE_MS    = 300;    // ms — ignore mic after agent starts speaking
const BARGE_MIN_MS     = 500;    // ms — voice must sustain this long to interrupt
const NUM_BARS         = 24;     // number of visualizer bars

// ── State ────────────────────────────────────────────────────────────────────
let ws             = null;
let sessionId      = null;
let mediaStream    = null;
let mediaRecorder  = null;
let audioCtx       = null;
let analyser       = null;
let animFrameId    = null;

let isRecording    = false;
let isAgentSpeaking = false;
let agentAudioQueue = [];       // queued WAV byte arrays
let isPlayingAudio  = false;

let silenceTimer    = null;
let bargeStartTime  = null;     // when sustained voice started
let bargeMuteUntil  = 0;        // timestamp — ignore mic until this time

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
}

// ── Add message to UI ─────────────────────────────────────────────────────────
function addMessage(role, text) {
  if (emptyState) emptyState.remove();

  const msg = document.createElement('div');
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
}

// ── Start call ────────────────────────────────────────────────────────────────
btnStart.addEventListener('click', async () => {
  btnStart.disabled = true;

  try {
    // Get mic with echo cancellation
    mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation : true,
        noiseSuppression : true,
        autoGainControl  : true,
      }
    });
  } catch (err) {
    alert('Microphone access denied. Please allow microphone access and try again.');
    btnStart.disabled = false;
    return;
  }

  // Set up Web Audio API for volume monitoring
  audioCtx  = new AudioContext();
  analyser  = audioCtx.createAnalyser();
  analyser.fftSize = 256;
  const source = audioCtx.createMediaStreamSource(mediaStream);
  source.connect(analyser);

  // Connect to WebSocket
  ws = new WebSocket(WS_URL);
  ws.binaryType = 'arraybuffer';

  ws.onopen = () => {
    btnEnd.disabled = false;
    setStatus('active', 'Connected');
    sessionInfo.innerHTML = `session — <span>connecting...</span>`;
    startVolumeLoop();
  };

  ws.onmessage = (event) => handleServerMessage(JSON.parse(event.data));

  ws.onclose = () => {
    setStatus('idle', 'Idle');
    btnStart.disabled = false;
    btnEnd.disabled   = true;
    stopRecording();
    stopVolumeLoop();
  };

  ws.onerror = (err) => {
    console.error('WebSocket error:', err);
  };
});

// ── End call ──────────────────────────────────────────────────────────────────
btnEnd.addEventListener('click', () => {
  if (ws) ws.close();
});

// ── Handle server messages ────────────────────────────────────────────────────
function handleServerMessage(msg) {
  switch (msg.type) {

    case 'session_id':
      sessionId = msg.session_id;
      sessionInfo.innerHTML = `session — <span>${sessionId}</span>`;
      break;

    case 'listening':
      isAgentSpeaking = false;
      setStatus('active', 'Listening');
      startRecording();
      break;

    case 'transcript':
      addMessage('customer', msg.text);
      stopRecording();
      setStatus('thinking', 'Thinking');
      break;

    case 'audio_chunk':
      // Decode base64 WAV bytes and queue for playback
      const binary  = atob(msg.data);
      const bytes   = new Uint8Array(binary.length);
      for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
      agentAudioQueue.push(bytes.buffer);
      if (!isPlayingAudio) playNextChunk();
      break;

    case 'audio_end':
      isAgentSpeaking = false;
      break;

    case 'reply_text':
      addMessage('agent', msg.text);
      break;
  }
}

// ── Audio recording ───────────────────────────────────────────────────────────
function startRecording() {
  if (isRecording || !mediaStream) return;
  isRecording = true;

  const chunks = [];  // ← collect all chunks locally

  mediaRecorder = new MediaRecorder(mediaStream, { mimeType: 'audio/webm;codecs=opus' });

  mediaRecorder.ondataavailable = (e) => {
    if (e.data.size > 0) chunks.push(e.data);  // ← just collect, don't send yet
  };

  mediaRecorder.onstop = () => {
    // ← send complete WebM blob ONCE when recording stops
    const blob = new Blob(chunks, { type: 'audio/webm;codecs=opus' });
    const reader = new FileReader();
    reader.onload = () => {
      const base64 = reader.result.split(',')[1];
      wsSend({ type: 'audio_chunk', data: base64 });
      wsSend({ type: 'audio_end' });
    };
    reader.readAsDataURL(blob);
  };

  mediaRecorder.start();  // ← no timeslice
  resetSilenceTimer();
}

function stopRecording() {
  if (!isRecording) return;
  isRecording = false;
  clearTimeout(silenceTimer);
  if (mediaRecorder && mediaRecorder.state !== 'inactive') {
    mediaRecorder.stop();
  }
}

function resetSilenceTimer() {
  clearTimeout(silenceTimer);
  silenceTimer = setTimeout(() => {
    if (isRecording) {
      stopRecording();
    }
  }, SILENCE_TIMEOUT);
}

// ── Audio playback ────────────────────────────────────────────────────────────
async function playNextChunk() {
  if (agentAudioQueue.length === 0) {
    isPlayingAudio  = false;
    isAgentSpeaking = false;
    return;
  }

  isPlayingAudio  = true;
  isAgentSpeaking = true;
  setStatus('speaking', 'Speaking');

  // Set mute window — ignore barge-in for first 300ms
  bargeMuteUntil = Date.now() + BARGE_MUTE_MS;
  bargeStartTime = null;

  const buffer = agentAudioQueue.shift();

  try {
    const decoded = await audioCtx.decodeAudioData(buffer);
    const source  = audioCtx.createBufferSource();
    source.buffer = decoded;
    source.connect(audioCtx.destination);
    source.onended = () => playNextChunk();
    source.start();
  } catch (err) {
    console.error('Audio decode error:', err);
    playNextChunk();
  }
}

function clearAudioQueue() {
  agentAudioQueue = [];
  isPlayingAudio  = false;
  isAgentSpeaking = false;
}

// ── Volume monitor + barge-in ─────────────────────────────────────────────────
function startVolumeLoop() {
  const dataArray = new Uint8Array(analyser.frequencyBinCount);

  function loop() {
    animFrameId = requestAnimationFrame(loop);
    analyser.getByteFrequencyData(dataArray);

    // Average volume
    let sum = 0;
    for (let i = 0; i < dataArray.length; i++) sum += dataArray[i];
    const avg = sum / dataArray.length;

    // Update visualizer bars
    updateBars(dataArray);

    // Barge-in detection — only when agent is speaking
    if (isAgentSpeaking) {
      const now = Date.now();

      // Filter 1 — 300ms mute window at start of playback
      if (now < bargeMuteUntil) return;

      // Filter 2 — volume threshold
      if (avg < VOLUME_THRESHOLD) {
        bargeStartTime = null;
        return;
      }

      // Filter 3 — sustained voice for 500ms+
      if (!bargeStartTime) {
        bargeStartTime = now;
        return;
      }

      if (now - bargeStartTime >= BARGE_MIN_MS) {
        // All filters passed — trigger barge-in
        triggerBargeIn();
      }
    } else {
      // Not agent speaking — use volume to extend silence timer
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
  bargeStartTime  = null;
  bargeMuteUntil  = 0;
  clearAudioQueue();
  stopRecording();
  wsSend({ type: 'interrupt' });
  setStatus('active', 'Listening');
}

// ── WebSocket send ────────────────────────────────────────────────────────────
function wsSend(data) {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify(data));
  }
}
