/**
 * DuckLive Dashboard — realtime preview + state polling
 *
 * This is the server monitoring dashboard. It displays engine status,
 * preview streams, and allows toggling engines on/off.
 *
 * Face/voice selection is handled by the client WebUI, not here.
 * The dashboard only shows which face/voice is currently active (read-only).
 *
 * Connects to ws://host:port/dashboard to receive:
 *   0x01 = processed video, 0x02 = processed audio,
 *   0x04 = original video,  0x05 = original audio,
 *   0x06 = audio levels (JSON)
 */

const STATE_POLL_MS = 1000;
const FRAME_TYPE = { VIDEO: 0x01, AUDIO: 0x02, CONTROL: 0x03,
                     ORIG_VIDEO: 0x04, ORIG_AUDIO: 0x05, AUDIO_LEVELS: 0x06 };
const HEADER_SIZE = 13;

let ctxOriginal = null;
let ctxProcessed = null;

// --- State Polling ---

async function pollState() {
    try {
        const res = await fetch('/api/state');
        if (!res.ok) return;
        const s = await res.json();
        updateDashboard(s);
    } catch (e) {
        // silently retry
    }
}

function updateDashboard(s) {
    // Server status badge
    const badge = document.getElementById('server-status');
    badge.textContent = s.status === 'running' ? 'LIVE' : s.status.toUpperCase();
    badge.className = 'status-badge ' + s.status;

    // Cards
    document.getElementById('uptime').textContent = fmtUptime(s.uptime_seconds);
    document.getElementById('fps').textContent = (s.stream?.fps ?? 0).toFixed(1);
    document.getElementById('client-count').textContent = s.clients?.length ?? 0;

    // GPU
    document.getElementById('gpu-name').textContent = s.gpu_name || 'GPU';
    document.getElementById('gpu-util').textContent = (s.gpu_utilization_pct ?? 0).toFixed(0);
    document.getElementById('gpu-temp').textContent = (s.gpu_temperature_c ?? 0).toFixed(0);
    document.getElementById('gpu-mem').textContent =
        `${(s.gpu_memory_used_mb ?? 0).toFixed(0)}/${(s.gpu_memory_total_mb ?? 0).toFixed(0)}`;

    // Feed source
    document.getElementById('feed-status').textContent = s.feed_connected ? 'Online' : 'Offline';
    document.getElementById('feed-source').textContent = s.feed_source || 'Waiting for client connection...';

    // Engine stats
    document.getElementById('face-model').textContent = s.face_swap?.model_name || 'Not loaded';
    document.getElementById('face-latency').textContent = (s.face_swap?.avg_latency_ms ?? 0).toFixed(1);
    document.getElementById('voice-model').textContent = s.voice_change?.model_name || 'Not loaded';
    document.getElementById('voice-latency').textContent = (s.voice_change?.avg_latency_ms ?? 0).toFixed(1);

    // Read-only display of currently active face and voice
    const currentFace = s.current_face || 'None';
    document.getElementById('current-face').textContent = currentFace;
    // Highlight active face with accent color
    document.getElementById('current-face').classList.toggle('active', !!s.current_face);

    const currentVoice = s.current_voice || 'None';
    document.getElementById('current-voice').textContent = currentVoice;
    document.getElementById('current-voice').classList.toggle('active', !!s.current_voice);
}

function fmtUptime(sec) {
    const h = Math.floor(sec / 3600);
    const m = Math.floor((sec % 3600) / 60);
    const s = Math.floor(sec % 60);
    return `${String(h).padStart(2,'0')}:${String(m).padStart(2,'0')}:${String(s).padStart(2,'0')}`;
}

// --- WebSocket Preview (dashboard endpoint) ---

function connectPreview() {
    const canvasOrig = document.getElementById('preview-original');
    const canvasProc = document.getElementById('preview-processed');
    ctxOriginal = canvasOrig.getContext('2d');
    ctxProcessed = canvasProc.getContext('2d');

    // Dashboard connects to /dashboard path on WS port
    fetch('/api/preview/url')
        .then(r => r.json())
        .then(data => {
            const wsUrl = data.ws_url.replace('{host}', window.location.hostname);
            startWs(wsUrl, canvasOrig, canvasProc);
        })
        .catch(e => console.warn('Preview URL error:', e));
}

function startWs(url, canvasOrig, canvasProc) {
    const ws = new WebSocket(url);
    ws.binaryType = 'arraybuffer';

    ws.onopen = () => {
        document.getElementById('overlay-original').classList.add('hidden');
        document.getElementById('overlay-processed').classList.add('hidden');
    };

    ws.onmessage = (event) => {
        if (!(event.data instanceof ArrayBuffer) || event.data.byteLength < HEADER_SIZE) return;

        const view = new DataView(event.data);
        const frameType = view.getUint8(0);
        const payload = new Uint8Array(event.data, HEADER_SIZE);

        switch (frameType) {
            case FRAME_TYPE.ORIG_VIDEO:
                renderJpeg(payload, canvasOrig, ctxOriginal);
                break;
            case FRAME_TYPE.VIDEO:
                renderJpeg(payload, canvasProc, ctxProcessed);
                break;
            case FRAME_TYPE.AUDIO_LEVELS:
                try {
                    const text = new TextDecoder().decode(payload);
                    const levels = JSON.parse(text);
                    updateAudioMeters(levels);
                } catch (e) {}
                break;
        }
    };

    ws.onclose = () => {
        document.getElementById('overlay-original').classList.remove('hidden');
        document.getElementById('overlay-processed').classList.remove('hidden');
        setTimeout(() => startWs(url, canvasOrig, canvasProc), 2000);
    };
}

function renderJpeg(jpegData, canvas, ctx) {
    const blob = new Blob([jpegData], { type: 'image/jpeg' });
    const url = URL.createObjectURL(blob);
    const img = new Image();
    img.onload = () => {
        canvas.width = img.width;
        canvas.height = img.height;
        ctx.drawImage(img, 0, 0);
        URL.revokeObjectURL(url);
    };
    img.src = url;
}

function updateAudioMeters(levels) {
    // Convert dBFS (-100..0) to percentage (0..100)
    const origPct = Math.max(0, Math.min(100, (levels.original_db + 60) / 60 * 100));
    const procPct = Math.max(0, Math.min(100, (levels.processed_db + 60) / 60 * 100));

    document.getElementById('meter-original').style.width = origPct + '%';
    document.getElementById('meter-processed').style.width = procPct + '%';
    document.getElementById('meter-original-db').textContent =
        levels.original_db <= -100 ? '-\u221E dB' : levels.original_db.toFixed(1) + ' dB';
    document.getElementById('meter-processed-db').textContent =
        levels.processed_db <= -100 ? '-\u221E dB' : levels.processed_db.toFixed(1) + ' dB';
}

// --- Engine Toggle Controls (enable/disable only, no asset selection) ---

document.getElementById('face-swap-toggle')?.addEventListener('change', (e) => {
    fetch('/api/engines/configure', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ face_swap_enabled: e.target.checked }),
    });
});

document.getElementById('voice-change-toggle')?.addEventListener('change', (e) => {
    fetch('/api/engines/configure', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ voice_change_enabled: e.target.checked }),
    });
});

// --- Init ---

setInterval(pollState, STATE_POLL_MS);
pollState();
connectPreview();
