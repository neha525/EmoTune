/**
 * app.js — EmoTune frontend logic
 * Handles: WAV upload, mic recording (MediaRecorder → WAV blob),
 *          waveform visualisation, API calls, and result rendering.
 */

"use strict";

// ── Constants ────────────────────────────────────────────────────────────────
const EMOTIONS = ["happy", "sad", "angry", "neutral", "fear", "surprise"];
const EMOJI = {
  happy: "😄", sad: "😢", angry: "😠",
  neutral: "😐", fear: "😨", surprise: "😲"
};
const COLOR = {
  happy: "#ffd166", sad: "#6eb5ff", angry: "#ff6b6b",
  neutral: "#80c9a4", fear: "#c77dff", surprise: "#ff9f43"
};

// ── DOM refs ──────────────────────────────────────────────────────────────────
const dropZone = document.getElementById("dropZone");
const fileInput = document.getElementById("fileInput");
const fileNameLabel = document.getElementById("fileNameLabel");
const micBtn = document.getElementById("micBtn");
const stopBtn = document.getElementById("stopBtn");
const analyseBtn = document.getElementById("analyseBtn");
const analyseAgainBtn = document.getElementById("analyseAgainBtn");
const waveformContainer = document.getElementById("waveformContainer");
const waveformCanvas = document.getElementById("waveformCanvas");
const recordControls = document.getElementById("recordControls");
const timerLabel = document.getElementById("timerLabel");
const recordingBadge = document.getElementById("recordingBadge");
const inputCard = document.getElementById("inputCard");
const loadingCard = document.getElementById("loadingCard");
const resultsSection = document.getElementById("resultsSection");
const emotionCard = document.getElementById("emotionCard");
const emotionEmoji = document.getElementById("emotionEmoji");
const emotionLabel = document.getElementById("emotionLabel");
const probGrid = document.getElementById("probGrid");
const songsList = document.getElementById("songsList");
const historyList = document.getElementById("historyList");
const clearHistoryBtn = document.getElementById("clearHistoryBtn");
const audioPreview = document.getElementById("audioPreview");
const audioPlayer = document.getElementById("audioPlayer");
const confettiCanvas = document.getElementById("confettiCanvas");

// ── State ────────────────────────────────────────────────────────────────────
let selectedFile = null;     // File object from upload
let mediaRecorder = null;
let audioChunks = [];
let recordedBlob = null;
let analyserNode = null;
let animFrameId = null;
let timerInterval = null;
let recordSeconds = 0;

// ── Init ─────────────────────────────────────────────────────────────────────
window.addEventListener("DOMContentLoaded", () => {
  loadHistory();
  bindEvents();
});

function bindEvents() {
  // Drag and drop
  dropZone.addEventListener("dragover", e => { e.preventDefault(); dropZone.classList.add("drag-over"); });
  dropZone.addEventListener("dragleave", () => dropZone.classList.remove("drag-over"));
  dropZone.addEventListener("drop", e => {
    e.preventDefault();
    dropZone.classList.remove("drag-over");
    const f = e.dataTransfer.files[0];
    if (f) handleFileSelected(f);
  });
  dropZone.addEventListener("click", e => {
    if (e.target === dropZone) fileInput.click();
  });

  fileInput.addEventListener("change", () => {
    if (fileInput.files[0]) handleFileSelected(fileInput.files[0]);
  });

  micBtn.addEventListener("click", toggleRecording);
  stopBtn.addEventListener("click", stopRecording);
  analyseBtn.addEventListener("click", runAnalysis);
  analyseAgainBtn.addEventListener("click", resetUI);
  clearHistoryBtn.addEventListener("click", clearHistory);
}

// ── File selection ────────────────────────────────────────────────────────────
function handleFileSelected(file) {
  const allowed = [".wav", ".mp3", ".ogg", ".m4a", ".flac"];
  const ext = file.name.slice(file.name.lastIndexOf(".")).toLowerCase();
  if (!allowed.includes(ext)) {
    alert("Please select a WAV, MP3, OGG, M4A, or FLAC file.");
    return;
  }
  selectedFile = file;
  recordedBlob = null;
  fileNameLabel.textContent = `\ud83d\udcc4 ${file.name}`;
  // Show audio preview
  const url = URL.createObjectURL(file);
  audioPlayer.src = url;
  audioPreview.style.display = "block";
  // Show the Analyse button so user can click when ready
  analyseBtn.style.display = "block";
}

// ── Mic Recording ─────────────────────────────────────────────────────────────
async function toggleRecording() {
  if (mediaRecorder && mediaRecorder.state === "recording") {
    stopRecording(); return;
  }
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    fileNameLabel.textContent = "\u274c Microphone not supported in this browser.";
    return;
  }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    startRecording(stream);
  } catch (err) {
    const msg = err.name === "NotAllowedError"
      ? "Microphone permission denied. Please allow microphone access and try again."
      : "Microphone error: " + err.message;
    fileNameLabel.textContent = "\u274c " + msg;
  }
}

function startRecording(stream) {
  audioChunks = [];
  recordedBlob = null;
  selectedFile = null;
  fileNameLabel.textContent = "";
  analyseBtn.style.display = "none";

  mediaRecorder = new MediaRecorder(stream);
  mediaRecorder.ondataavailable = e => audioChunks.push(e.data);
  mediaRecorder.onstop = onRecordingStop;
  mediaRecorder.start(100); // collect every 100ms

  micBtn.classList.add("recording");
  micBtn.textContent = "🔴 Recording…";
  waveformContainer.style.display = "block";
  recordControls.style.display = "flex";
  recordingBadge.style.display = "flex";

  // Timer
  recordSeconds = 0;
  timerLabel.textContent = "0s";
  timerInterval = setInterval(() => {
    recordSeconds++;
    timerLabel.textContent = recordSeconds + "s";
  }, 1000);

  // Waveform
  const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  const source = audioCtx.createMediaStreamSource(stream);
  analyserNode = audioCtx.createAnalyser();
  analyserNode.fftSize = 256;
  source.connect(analyserNode);
  drawWaveform();
}

function stopRecording() {
  if (mediaRecorder && mediaRecorder.state === "recording") {
    mediaRecorder.stop();
    mediaRecorder.stream.getTracks().forEach(t => t.stop());
  }
  micBtn.classList.remove("recording");
  micBtn.textContent = "🎤 Record";
  clearInterval(timerInterval);
  cancelAnimationFrame(animFrameId);
  recordingBadge.style.display = "none";
  recordControls.style.display = "none";
}

function onRecordingStop() {
  // MediaRecorder captures WebM/Opus — give preview immediately, then convert to real WAV
  const nativeBlob = new Blob(audioChunks, { type: mediaRecorder.mimeType || "audio/webm" });
  waveformContainer.style.display = "none";

  const url = URL.createObjectURL(nativeBlob);
  audioPlayer.src = url;
  audioPreview.style.display = "block";

  // Decode → re-encode as proper 16-bit PCM WAV so librosa can read it
  nativeBlob.arrayBuffer().then(buffer => {
    const audioCtx = new AudioContext();
    return audioCtx.decodeAudioData(buffer);
  }).then(audioBuffer => {
    const wavBuffer = encodeWAV(audioBuffer);
    recordedBlob = new Blob([wavBuffer], { type: "audio/wav" });
    runAnalysis();
  }).catch(err => {
    console.error("WAV conversion failed, sending native format:", err);
    recordedBlob = nativeBlob;
    runAnalysis();
  });
}

/** Encode an AudioBuffer as a 16-bit PCM WAV ArrayBuffer (mono, original sample rate). */
function encodeWAV(audioBuffer) {
  const sampleRate = audioBuffer.sampleRate;
  const bitsPerSample = 16;
  const bytesPerSample = bitsPerSample / 8;

  // Mix down to mono
  let samples;
  if (audioBuffer.numberOfChannels === 1) {
    samples = audioBuffer.getChannelData(0);
  } else {
    const ch0 = audioBuffer.getChannelData(0);
    const ch1 = audioBuffer.getChannelData(1);
    samples = new Float32Array(ch0.length);
    for (let i = 0; i < ch0.length; i++) samples[i] = (ch0[i] + ch1[i]) / 2;
  }

  const dataLen = samples.length * bytesPerSample;
  const buf = new ArrayBuffer(44 + dataLen);
  const v = new DataView(buf);

  // RIFF header
  writeStr(v, 0, "RIFF");
  v.setUint32(4, 36 + dataLen, true);
  writeStr(v, 8, "WAVE");
  writeStr(v, 12, "fmt ");
  v.setUint32(16, 16, true);           // chunk size
  v.setUint16(20, 1, true);           // PCM
  v.setUint16(22, 1, true);           // mono
  v.setUint32(24, sampleRate, true);
  v.setUint32(28, sampleRate * bytesPerSample, true);
  v.setUint16(32, bytesPerSample, true);
  v.setUint16(34, bitsPerSample, true);
  writeStr(v, 36, "data");
  v.setUint32(40, dataLen, true);

  // PCM samples: float32 → int16
  let off = 44;
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    v.setInt16(off, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
    off += 2;
  }
  return buf;
}

function writeStr(view, offset, str) {
  for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
}

function drawWaveform() {
  const ctx = waveformCanvas.getContext("2d");
  const W = waveformCanvas.width = waveformCanvas.offsetWidth * devicePixelRatio;
  const H = waveformCanvas.height = waveformCanvas.offsetHeight * devicePixelRatio;

  const data = new Uint8Array(analyserNode.frequencyBinCount);
  analyserNode.getByteTimeDomainData(data);

  ctx.clearRect(0, 0, W, H);
  ctx.lineWidth = 2.5 * devicePixelRatio;
  ctx.strokeStyle = "#a875ff";
  ctx.shadowColor = "#a875ff";
  ctx.shadowBlur = 8;
  ctx.beginPath();

  const sliceWidth = W / data.length;
  let x = 0;
  for (let i = 0; i < data.length; i++) {
    const v = data[i] / 128.0;
    const y = (v * H) / 2;
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    x += sliceWidth;
  }
  ctx.lineTo(W, H / 2);
  ctx.stroke();

  animFrameId = requestAnimationFrame(drawWaveform);
}

// ── API call ──────────────────────────────────────────────────────────────────
async function runAnalysis() {
  const blob = recordedBlob || (selectedFile ? selectedFile : null);
  if (!blob) { alert("Please upload or record audio first."); return; }

  // Show loading
  inputCard.style.display = "none";
  resultsSection.style.display = "none";
  loadingCard.style.display = "flex";

  try {
    const formData = new FormData();
    formData.append("audio", blob, "audio.wav");

    const res = await fetch("/predict", { method: "POST", body: formData });
    const data = await res.json();

    if (data.error) throw new Error(data.error);

    renderResults(data);
    loadHistory(); // refresh sidebar

  } catch (err) {
    loadingCard.style.display = "none";
    inputCard.style.display = "flex";
    alert("Error: " + err.message);
  }
}

// ── Render results ─────────────────────────────────────────────────────────────
function renderResults(data) {
  const { emotion, probs, songs } = data;

  // Emotion card
  emotionCard.className = "card emotion-card " + emotion;
  emotionEmoji.textContent = EMOJI[emotion] || "🎵";
  emotionLabel.textContent = emotion;
  emotionLabel.style.color = COLOR[emotion] || "#fff";

  // Probability bars
  probGrid.innerHTML = "";
  EMOTIONS.forEach(emo => {
    const p = probs[emo] ?? 0;
    const row = document.createElement("div");
    row.className = "prob-row";
    row.innerHTML = `
      <span class="prob-name">${emo}</span>
      <div class="prob-bar-bg">
        <div class="prob-bar-fill ${emo}" style="width:0%" data-pct="${(p * 100).toFixed(1)}%"></div>
      </div>
      <span class="prob-val">${(p * 100).toFixed(1)}%</span>`;
    probGrid.appendChild(row);
  });

  // Songs
  songsList.innerHTML = "";
  (songs || []).forEach((s, i) => {
    const item = document.createElement("div");
    item.className = "song-item";
    item.style.animationDelay = `${i * 0.08}s`;
    const linkHtml = s.url
      ? `<a class="song-link" href="${s.url}" target="_blank" rel="noopener">▶ Spotify</a>`
      : "";
    const artHtml = s.album_art
      ? `<img class="song-art" src="${s.album_art}" alt="" loading="lazy" />`
      : `<div class="song-art"></div>`;
    item.innerHTML = `
      <span class="song-num">${i + 1}</span>
      ${artHtml}
      <div class="song-meta">
        <div class="song-title">${s.title}</div>
        <div class="song-artist">${s.artist}</div>
      </div>
      ${linkHtml}`;
    songsList.appendChild(item);
  });

  loadingCard.style.display = "none";
  resultsSection.style.display = "flex";

  // Confetti for happy / surprise
  if (emotion === "happy" || emotion === "surprise") {
    setTimeout(launchConfetti, 300);
  }

  // Animate bars slightly after paint
  requestAnimationFrame(() => setTimeout(() => {
    document.querySelectorAll(".prob-bar-fill").forEach(el => {
      el.style.width = el.dataset.pct;
    });
  }, 80));
}

// ── Reset ─────────────────────────────────────────────────────────────────────
function resetUI() {
  selectedFile = null;
  recordedBlob = null;
  fileNameLabel.textContent = "";
  analyseBtn.style.display = "none";
  waveformContainer.style.display = "none";
  recordControls.style.display = "none";
  resultsSection.style.display = "none";
  loadingCard.style.display = "none";
  inputCard.style.display = "flex";
  audioPreview.style.display = "none";
  audioPlayer.src = "";
  fileInput.value = "";
}

// ── History ────────────────────────────────────────────────────────────────────
async function loadHistory() {
  try {
    const res = await fetch("/history?n=10");
    const sessions = await res.json();
    renderHistory(sessions);
  } catch { /* non-critical */ }
}

function renderHistory(sessions) {
  historyList.innerHTML = "";
  if (!sessions.length) {
    historyList.innerHTML = '<p class="empty-msg">No sessions yet.</p>';
    return;
  }
  sessions.forEach(s => {
    const el = document.createElement("div");
    el.className = "history-item";
    const dt = new Date(s.timestamp);
    const timeStr = dt.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
    const topSongs = (s.songs || []).slice(0, 2).map(x => x.title).join(", ");
    el.innerHTML = `
      <div class="h-item-top">
        <span class="h-emoji">${s.emoji}</span>
        <span class="h-emotion">${s.emotion}</span>
        <span class="h-time">${timeStr}</span>
      </div>
      ${topSongs ? `<div class="h-songs">🎵 ${topSongs}</div>` : ""}
    `;
    historyList.appendChild(el);
  });
}

async function clearHistory() {
  if (!confirm("Clear all session history?")) return;
  await fetch("/clear-history", { method: "POST" });
  loadHistory();
}

// ── Confetti ─────────────────────────────────────────────────────────────────────
function launchConfetti() {
  const canvas = confettiCanvas;
  const ctx = canvas.getContext("2d");
  canvas.width = window.innerWidth;
  canvas.height = window.innerHeight;
  canvas.style.display = "block";

  const COLORS = ["#ffd166", "#ff6b6b", "#a875ff", "#6eb5ff", "#80c9a4", "#ff9f43"];
  const N = 160;
  const pieces = Array.from({ length: N }, () => ({
    x: Math.random() * canvas.width,
    y: Math.random() * -canvas.height,
    r: Math.random() * 8 + 4,
    d: Math.random() * N,
    color: COLORS[Math.floor(Math.random() * COLORS.length)],
    tilt: Math.random() * 10 - 10,
    tiltAngle: 0,
    tiltSpeed: Math.random() * 0.07 + 0.05,
  }));

  let frame = 0;
  const maxFrames = 220;

  function draw() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    pieces.forEach(p => {
      ctx.beginPath();
      ctx.lineWidth = p.r / 2;
      ctx.strokeStyle = p.color;
      ctx.moveTo(p.x + p.tilt + p.r / 4, p.y);
      ctx.lineTo(p.x + p.tilt, p.y + p.tilt + p.r / 3);
      ctx.stroke();

      p.tiltAngle += p.tiltSpeed;
      p.y += (Math.cos(p.d + frame / 50) + 1.5) * 1.5;
      p.tilt = Math.sin(p.tiltAngle) * 12;

      if (p.y > canvas.height) {
        p.x = Math.random() * canvas.width;
        p.y = -10;
      }
    });
    frame++;
    if (frame < maxFrames) requestAnimationFrame(draw);
    else canvas.style.display = "none";
  }
  draw();
}
