/**
 * DuckLive Client UI — Browser-based capture + preview + engine control
 *
 * Flow:
 *   1. getUserMedia() → capture camera + mic
 *   2. Send raw frames to server via WebSocket /feed
 *   3. Receive processed frames back → render preview
 *   4. Control face/voice engines via REST API
 *
 * Protocol: 13-byte header (type:1 + timestamp:8 + size:4) + payload
 *   Upstream:   0x10 = RAW_VIDEO (JPEG), 0x11 = RAW_AUDIO (PCM s16le)
 *   Downstream: 0x01 = PROCESSED_VIDEO (JPEG), 0x02 = PROCESSED_AUDIO (PCM)
 */

const HEADER_SIZE = 13;
const FRAME_RAW_VIDEO = 0x10;
const FRAME_RAW_AUDIO = 0x11;
const FRAME_PROCESSED_VIDEO = 0x01;
const TARGET_FPS = 30;
const TARGET_AUDIO_RATE = 16000;
const JPEG_QUALITY = 0.85;
const STATUS_POLL_MS = 2000;
const ENGINE_POLL_MS = 3000;

// ─── State ───

let isCapturing = false;
let feedWs = null;
let mediaStream = null;
let audioContext = null;
let audioWorkletNode = null;
let captureInterval = null;
let processedCtx = null;
let framesSent = 0;
let framesReceived = 0;
let fpsTimestamps = [];
let serverFeedUrl = null;

// Engine state cache
let engineState = {
    face_swap: { available: false, enabled: false, current_face: '' },
    voice_change: { available: false, enabled: false, current_voice: '', pitch_shift: 0 },
};
let availableFaces = [];
let availableVoices = [];

// ─── Device Enumeration ───

async function enumerateDevices() {
    const devices = await navigator.mediaDevices.enumerateDevices();

    const camSelect = document.getElementById('camera-select');
    const micSelect = document.getElementById('mic-select');
    camSelect.innerHTML = '';
    micSelect.innerHTML = '';

    devices.filter(d => d.kind === 'videoinput').forEach((d, i) => {
        const opt = document.createElement('option');
        opt.value = d.deviceId;
        opt.textContent = d.label || `Camera ${i + 1}`;
        camSelect.appendChild(opt);
    });

    devices.filter(d => d.kind === 'audioinput').forEach((d, i) => {
        const opt = document.createElement('option');
        opt.value = d.deviceId;
        opt.textContent = d.label || `Microphone ${i + 1}`;
        micSelect.appendChild(opt);
    });
}

// ─── Engine Control ───

async function loadEngineState() {
    try {
        const [enginesRes, facesRes, voicesRes] = await Promise.all([
            fetch('/api/engines'),
            fetch('/api/faces'),
            fetch('/api/voices'),
        ]);
        engineState = await enginesRes.json();
        const facesData = await facesRes.json();
        const voicesData = await voicesRes.json();

        availableFaces = facesData.faces || [];
        availableVoices = voicesData.voices || [];

        renderFaceGrid();
        renderVoiceList();
        syncEngineUI();
    } catch (e) {
        console.warn('[DuckLive] Failed to load engine state:', e);
    }
}

function syncEngineUI() {
    const fs = engineState.face_swap;
    const vc = engineState.voice_change;

    // Face swap toggle
    document.getElementById('face-toggle').checked = fs.enabled;
    const faceStatus = document.getElementById('face-engine-status');
    if (!fs.available) {
        faceStatus.textContent = '不可用';
        faceStatus.className = 'engine-status unavailable';
    } else if (fs.enabled && fs.current_face) {
        faceStatus.textContent = '运行中';
        faceStatus.className = 'engine-status active';
    } else if (fs.enabled) {
        faceStatus.textContent = '未选人脸';
        faceStatus.className = 'engine-status';
    } else {
        faceStatus.textContent = '已关闭';
        faceStatus.className = 'engine-status';
    }

    // Voice change toggle
    document.getElementById('voice-toggle').checked = vc.enabled;
    const voiceStatus = document.getElementById('voice-engine-status');
    if (!vc.available) {
        voiceStatus.textContent = '不可用';
        voiceStatus.className = 'engine-status unavailable';
    } else if (vc.enabled && vc.current_voice) {
        voiceStatus.textContent = '运行中';
        voiceStatus.className = 'engine-status active';
    } else if (vc.enabled) {
        voiceStatus.textContent = '未选声音';
        voiceStatus.className = 'engine-status';
    } else {
        voiceStatus.textContent = '已关闭';
        voiceStatus.className = 'engine-status';
    }

    // Pitch slider
    document.getElementById('pitch-slider').value = vc.pitch_shift || 0;
    document.getElementById('pitch-value').textContent = formatPitch(vc.pitch_shift || 0);

    // Highlight selected face/voice
    document.querySelectorAll('.face-item').forEach(el => {
        el.classList.toggle('selected', el.dataset.name === fs.current_face);
    });
    document.querySelectorAll('.voice-item').forEach(el => {
        el.classList.toggle('selected', el.dataset.name === vc.current_voice);
    });
}

function renderFaceGrid() {
    const grid = document.getElementById('face-grid');
    if (availableFaces.length === 0) {
        grid.innerHTML = '<div class="face-empty">服务器无可用人脸</div>';
        return;
    }

    grid.innerHTML = '';
    for (const name of availableFaces) {
        const item = document.createElement('div');
        item.className = 'face-item';
        item.dataset.name = name;
        if (engineState.face_swap.current_face === name) {
            item.classList.add('selected');
        }

        const img = document.createElement('img');
        img.src = `/api/faces/${encodeURIComponent(name)}/thumbnail`;
        img.alt = name;
        img.loading = 'lazy';

        const label = document.createElement('div');
        label.className = 'face-name';
        label.textContent = name.replace(/\.[^.]+$/, '');

        item.appendChild(img);
        item.appendChild(label);
        item.onclick = () => selectFace(name);
        grid.appendChild(item);
    }
}

function renderVoiceList() {
    const list = document.getElementById('voice-list');
    if (availableVoices.length === 0) {
        list.innerHTML = '<div class="face-empty">服务器无可用声音模型</div>';
        return;
    }

    list.innerHTML = '';
    for (const name of availableVoices) {
        const item = document.createElement('div');
        item.className = 'voice-item';
        item.dataset.name = name;
        if (engineState.voice_change.current_voice === name.replace(/\.[^.]+$/, '')) {
            item.classList.add('selected');
        }
        item.textContent = name.replace(/\.pth$/, '');
        item.onclick = () => selectVoice(name);
        list.appendChild(item);
    }
}

async function toggleEngine(engine, enabled) {
    try {
        const body = {};
        body[engine + '_enabled'] = enabled;
        const res = await fetch('/api/engines/configure', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });
        if (res.ok) {
            const data = await res.json();
            engineState.face_swap.enabled = data.face_swap_enabled;
            engineState.voice_change.enabled = data.voice_change_enabled;
            syncEngineUI();
        }
    } catch (e) {
        console.error('[DuckLive] Toggle engine failed:', e);
    }
}

async function selectFace(name) {
    // If already selected, deselect
    const filename = (engineState.face_swap.current_face === name) ? '' : name;
    try {
        const res = await fetch('/api/faces/select', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ filename }),
        });
        if (res.ok) {
            const data = await res.json();
            engineState.face_swap.current_face = data.face;
            syncEngineUI();
        } else {
            const err = await res.json();
            console.error('[DuckLive] Select face failed:', err.detail);
        }
    } catch (e) {
        console.error('[DuckLive] Select face error:', e);
    }
}

async function selectVoice(name) {
    const currentName = engineState.voice_change.current_voice;
    const stemName = name.replace(/\.pth$/, '');
    const filename = (currentName === stemName) ? '' : name;
    try {
        const res = await fetch('/api/voices/select', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ filename }),
        });
        if (res.ok) {
            const data = await res.json();
            engineState.voice_change.current_voice = data.voice;
            syncEngineUI();
        }
    } catch (e) {
        console.error('[DuckLive] Select voice error:', e);
    }
}

function updatePitchLabel(val) {
    document.getElementById('pitch-value').textContent = formatPitch(parseInt(val));
}

async function setPitch(val) {
    const pitch = parseInt(val);
    try {
        await fetch('/api/engines/configure', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ voice_pitch_shift: pitch }),
        });
        engineState.voice_change.pitch_shift = pitch;
    } catch (e) {
        console.error('[DuckLive] Set pitch failed:', e);
    }
}

function formatPitch(val) {
    if (val > 0) return '+' + val;
    return '' + val;
}

// ─── Capture Control ───

async function toggleCapture() {
    if (isCapturing) {
        stopCapture();
    } else {
        await startCapture();
    }
}

async function startCapture() {
    const btn = document.getElementById('start-btn');
    btn.disabled = true;
    btn.textContent = '⏳ 连接中...';

    try {
        // 1. Get server info
        console.log('[DuckLive] Fetching server info...');
        const infoRes = await fetch('/api/server-info');
        const info = await infoRes.json();
        if (!info.feed_url) {
            alert('未找到服务器，请确保 DuckLive Server 已启动');
            btn.disabled = false;
            btn.textContent = '▶ 开始';
            return;
        }
        serverFeedUrl = info.feed_url;
        document.getElementById('server-addr').textContent = info.server_address || '--';
        console.log('[DuckLive] Server feed URL:', serverFeedUrl);

        // 2. Get camera + mic
        const camId = document.getElementById('camera-select').value;
        const micId = document.getElementById('mic-select').value;

        console.log('[DuckLive] Calling getUserMedia({video:true, audio:false})...');
        const gumPromise = navigator.mediaDevices.getUserMedia({ video: true, audio: false });
        const timeoutPromise = new Promise((_, reject) =>
            setTimeout(() => reject(new Error('getUserMedia timed out after 10s')), 10000)
        );
        mediaStream = await Promise.race([gumPromise, timeoutPromise]);
        console.log('[DuckLive] Video OK:', mediaStream.getVideoTracks().map(t => t.label));

        // Try to add audio track
        try {
            const audioStream = await navigator.mediaDevices.getUserMedia({
                video: false,
                audio: micId ? { deviceId: micId } : true,
            });
            for (const track of audioStream.getAudioTracks()) {
                mediaStream.addTrack(track);
            }
            console.log('[DuckLive] Audio OK:', mediaStream.getAudioTracks().map(t => t.label));
        } catch (audioErr) {
            console.warn('[DuckLive] Audio capture failed (video-only mode):', audioErr.message);
        }

        // Show local preview
        const localVideo = document.getElementById('local-video');
        localVideo.srcObject = mediaStream;
        document.getElementById('overlay-local').classList.add('hidden');

        // 3. Connect to server /feed
        console.log('[DuckLive] Connecting feed WebSocket...');
        connectFeedWs();

        // 4. Start video frame capture
        captureInterval = setInterval(captureVideoFrame, 1000 / TARGET_FPS);

        // 5. Start audio capture (non-blocking)
        startAudioCapture().catch(e => {
            console.warn('[DuckLive] Audio capture failed (video still works):', e);
        });

        isCapturing = true;
        btn.textContent = '⏹ 停止';
        btn.disabled = false;
        btn.classList.add('btn-danger');
        btn.classList.remove('btn-primary');

        // Refresh engine state now that we're connected
        loadEngineState();

    } catch (e) {
        console.error('[DuckLive] Start failed:', e);
        alert('启动失败: ' + e.message);
        stopCapture();
        btn.disabled = false;
        btn.textContent = '▶ 开始';
    }
}

function stopCapture() {
    isCapturing = false;

    if (captureInterval) {
        clearInterval(captureInterval);
        captureInterval = null;
    }

    if (audioWorkletNode) {
        audioWorkletNode.disconnect();
        audioWorkletNode = null;
    }

    if (audioContext) {
        audioContext.close();
        audioContext = null;
    }

    if (mediaStream) {
        mediaStream.getTracks().forEach(t => t.stop());
        mediaStream = null;
    }

    if (feedWs) {
        feedWs.close();
        feedWs = null;
    }

    const localVideo = document.getElementById('local-video');
    localVideo.srcObject = null;

    document.getElementById('overlay-local').classList.remove('hidden');
    document.getElementById('overlay-processed').classList.remove('hidden');

    const btn = document.getElementById('start-btn');
    btn.textContent = '▶ 开始';
    btn.classList.remove('btn-danger');
    btn.classList.add('btn-primary');

    updateFeedStatus(false);
}

// ─── WebSocket Feed Connection ───

function connectFeedWs() {
    if (!serverFeedUrl) return;

    feedWs = new WebSocket(serverFeedUrl);
    feedWs.binaryType = 'arraybuffer';

    feedWs.onopen = () => {
        console.log('[DuckLive] Feed WebSocket connected');
        updateFeedStatus(true);
        document.getElementById('overlay-processed').textContent = '等待服务器处理...';
    };

    feedWs.onmessage = (event) => {
        if (!(event.data instanceof ArrayBuffer) || event.data.byteLength < HEADER_SIZE) return;

        const frameType = new DataView(event.data).getUint8(0);

        if (frameType === FRAME_PROCESSED_VIDEO) {
            renderProcessedFrame(event.data);
            framesReceived++;
            fpsTimestamps.push(performance.now());
            const cutoff = performance.now() - 1000;
            fpsTimestamps = fpsTimestamps.filter(t => t > cutoff);
        }
    };

    feedWs.onclose = () => {
        console.log('[DuckLive] Feed WebSocket disconnected');
        updateFeedStatus(false);
        if (isCapturing) {
            setTimeout(connectFeedWs, 2000);
        }
    };

    feedWs.onerror = (e) => {
        console.error('[DuckLive] Feed WebSocket error:', e);
    };
}

function updateFeedStatus(connected) {
    const badge = document.getElementById('feed-status');
    if (connected) {
        badge.textContent = '🟢 已连接';
        badge.className = 'status-badge connected';
    } else {
        badge.textContent = '未连接';
        badge.className = 'status-badge disconnected';
    }
}

// ─── Video Frame Capture ───

const captureCanvas = document.createElement('canvas');
const captureCtx = captureCanvas.getContext('2d');

function captureVideoFrame() {
    if (!feedWs || feedWs.readyState !== WebSocket.OPEN) return;

    const video = document.getElementById('local-video');
    if (video.videoWidth === 0) return;

    captureCanvas.width = video.videoWidth;
    captureCanvas.height = video.videoHeight;
    captureCtx.drawImage(video, 0, 0);

    captureCanvas.toBlob((blob) => {
        if (!blob || !feedWs || feedWs.readyState !== WebSocket.OPEN) return;

        blob.arrayBuffer().then(buf => {
            const frame = packFrame(FRAME_RAW_VIDEO, new Uint8Array(buf));
            feedWs.send(frame);
            framesSent++;
        });
    }, 'image/jpeg', JPEG_QUALITY);
}

// ─── Audio Capture ───

async function startAudioCapture() {
    if (!mediaStream) return;

    const audioTrack = mediaStream.getAudioTracks()[0];
    if (!audioTrack) return;

    audioContext = new AudioContext({ sampleRate: 48000 });

    try {
        await audioContext.audioWorklet.addModule('/static/js/audio-worklet.js');
    } catch (e) {
        console.warn('[DuckLive] AudioWorklet not supported:', e);
        return;
    }

    const source = audioContext.createMediaStreamSource(
        new MediaStream([audioTrack])
    );

    audioWorkletNode = new AudioWorkletNode(audioContext, 'ducklive-audio-processor');

    audioWorkletNode.port.onmessage = (event) => {
        const { samples, sourceSampleRate } = event.data;
        if (!feedWs || feedWs.readyState !== WebSocket.OPEN) return;

        const resampled = resampleAudio(samples, sourceSampleRate, TARGET_AUDIO_RATE);
        const int16 = float32ToInt16(resampled);
        const frame = packFrame(FRAME_RAW_AUDIO, new Uint8Array(int16.buffer));
        feedWs.send(frame);
    };

    source.connect(audioWorkletNode);
    audioWorkletNode.connect(audioContext.destination);
}

// ─── Audio Helpers ───

function resampleAudio(samples, srcRate, targetRate) {
    if (srcRate === targetRate) return samples;
    const ratio = srcRate / targetRate;
    const outputLen = Math.round(samples.length / ratio);
    const output = new Float32Array(outputLen);
    for (let i = 0; i < outputLen; i++) {
        const srcIdx = i * ratio;
        const idx0 = Math.floor(srcIdx);
        const idx1 = Math.min(idx0 + 1, samples.length - 1);
        const frac = srcIdx - idx0;
        output[i] = samples[idx0] * (1 - frac) + samples[idx1] * frac;
    }
    return output;
}

function float32ToInt16(float32) {
    const int16 = new Int16Array(float32.length);
    for (let i = 0; i < float32.length; i++) {
        const s = Math.max(-1, Math.min(1, float32[i]));
        int16[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
    }
    return int16;
}

// ─── Protocol Helpers ───

function packFrame(type, payload) {
    const timestamp = BigInt(Date.now()) * 1000n;
    const buf = new ArrayBuffer(HEADER_SIZE + payload.byteLength);
    const view = new DataView(buf);

    view.setUint8(0, type);
    view.setBigUint64(1, timestamp, false);
    view.setUint32(9, payload.byteLength, false);

    new Uint8Array(buf, HEADER_SIZE).set(payload);
    return buf;
}

// ─── Processed Frame Rendering ───

function renderProcessedFrame(data) {
    const jpegData = new Uint8Array(data, HEADER_SIZE);
    const blob = new Blob([jpegData], { type: 'image/jpeg' });
    const url = URL.createObjectURL(blob);

    if (!processedCtx) {
        processedCtx = document.getElementById('processed-canvas').getContext('2d');
    }
    const canvas = document.getElementById('processed-canvas');

    const img = new Image();
    img.onload = () => {
        canvas.width = img.width;
        canvas.height = img.height;
        processedCtx.drawImage(img, 0, 0);
        URL.revokeObjectURL(url);
        document.getElementById('overlay-processed').classList.add('hidden');
    };
    img.src = url;
}

// ─── Status Polling ───

async function pollStatus() {
    try {
        const res = await fetch('/api/status');
        if (!res.ok) return;
        const s = await res.json();

        const cam = s.devices.virtual_camera;
        document.getElementById('vcam-status').textContent =
            cam.running ? cam.name : (cam.enabled ? '未启动' : '未安装');
    } catch (e) {}

    // Local stats
    document.getElementById('frames-sent').textContent = framesSent;
    document.getElementById('frames-received').textContent = framesReceived;
    document.getElementById('fps').textContent = fpsTimestamps.length;
}

// ─── Camera Test ───

async function testCamera() {
    const btn = document.getElementById('test-btn');
    btn.textContent = '🔍 ...';
    btn.disabled = true;

    const results = [];

    results.push('1. mediaDevices API: ' + (navigator.mediaDevices ? '✅' : '❌'));
    results.push('   getUserMedia: ' + (navigator.mediaDevices?.getUserMedia ? '✅' : '❌'));

    try {
        const devices = await navigator.mediaDevices.enumerateDevices();
        const cams = devices.filter(d => d.kind === 'videoinput');
        const mics = devices.filter(d => d.kind === 'audioinput');
        results.push('2. 设备: ✅ ' + cams.length + ' 摄像头, ' + mics.length + ' 麦克风');
    } catch (e) {
        results.push('2. 设备枚举: ❌ ' + e.message);
    }

    try {
        const p = navigator.mediaDevices.getUserMedia({ video: true, audio: false });
        const timeout = new Promise((_, rej) => setTimeout(() => rej(new Error('TIMEOUT 5s')), 5000));
        const stream = await Promise.race([p, timeout]);
        const track = stream.getVideoTracks()[0];
        results.push('3. 摄像头: ✅ ' + track.label + ' (' + track.getSettings().width + 'x' + track.getSettings().height + ')');
        stream.getTracks().forEach(t => t.stop());
    } catch (e) {
        results.push('3. 摄像头: ❌ ' + e.message);
    }

    alert(results.join('\n'));
    btn.textContent = '🔍 测试';
    btn.disabled = false;
}

// ─── Init ───

enumerateDevices();
loadEngineState();
setInterval(pollStatus, STATUS_POLL_MS);
setInterval(loadEngineState, ENGINE_POLL_MS);
pollStatus();
