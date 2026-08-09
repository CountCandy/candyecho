const form = document.getElementById('generateForm');
const textArea = document.getElementById('text');
const favoriteList = document.getElementById('favoriteList');
const otherVoices = document.getElementById('otherVoices');
const otherPreview = document.getElementById('otherPreview');
const otherRename = document.getElementById('otherRename');
const otherFavorite = document.getElementById('otherFavorite');
const otherDelete = document.getElementById('otherDelete');
const voicePreview = document.getElementById('voicePreview');
const normalizeVolume = document.getElementById('normalizeVolume');
const cleanAudio = document.getElementById('cleanAudio');
const selectedVoiceBadge = document.getElementById('selectedVoiceBadge');
const otherPicker = document.getElementById('otherPicker');

// Voice panel state
let voices = [];
let durations = {};          // voice name -> seconds (or null)
let selectedVoice = null;
const FAVORITES_KEY = 'candyecho.favorites';
let favorites = new Set(loadFavorites());

function loadFavorites() {
    try { return JSON.parse(localStorage.getItem(FAVORITES_KEY) || '[]'); }
    catch (e) { return []; }
}
function saveFavorites() {
    localStorage.setItem(FAVORITES_KEY, JSON.stringify([...favorites]));
}
const generateBtn = document.getElementById('generateBtn');
const stopBtn = document.getElementById('stopBtn');
const progressDiv = document.getElementById('progress');
const errorDiv = document.getElementById('error');
const audioPlayerDiv = document.getElementById('audioPlayer');
const themeToggle = document.getElementById('themeToggle');

// --- Generation loading bar (candy flavor text + rock-candy crust) ---
const genProgress = document.getElementById('genProgress');
const genTrack = document.getElementById('genTrack');
const genReveal = document.getElementById('genReveal');
const genFlavor = document.getElementById('genFlavor');
const genPercent = document.getElementById('genPercent');

// --- Rock-candy crust: a deterministic crystal field drawn once as inline SVG,
// then revealed left->right by growing the clip window (see setGenPercent). ---
function mulberry32(a) {
    return function () {
        a |= 0; a = a + 0x6D2B79F5 | 0;
        let t = Math.imul(a ^ a >>> 15, 1 | a);
        t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
        return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
}
const _cClamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
const _cLerp = (a, b, t) => a + (b - a) * t;
// Colour flows along the stick: bubblegum pink -> orchid -> grape.
const _CRUST_STOPS = [[0, [255, 99, 176]], [0.5, [201, 92, 238]], [1, [167, 89, 245]]];
function _colorAt(t) {
    t = _cClamp(t, 0, 1);
    for (let i = 1; i < _CRUST_STOPS.length; i++) {
        if (t <= _CRUST_STOPS[i][0]) {
            const [t0, c0] = _CRUST_STOPS[i - 1], [t1, c1] = _CRUST_STOPS[i];
            const k = (t - t0) / (t1 - t0);
            return [_cLerp(c0[0], c1[0], k), _cLerp(c0[1], c1[1], k), _cLerp(c0[2], c1[2], k)];
        }
    }
    return _CRUST_STOPS[_CRUST_STOPS.length - 1][1];
}
const _cRgba = (c, a) => 'rgba(' + (c[0] | 0) + ',' + (c[1] | 0) + ',' + (c[2] | 0) + ',' + a + ')';
const _cShade = (c, f) => [_cClamp(c[0] * f, 0, 255), _cClamp(c[1] * f, 0, 255), _cClamp(c[2] * f, 0, 255)];
const _cMix = (c, d, t) => [_cLerp(c[0], d[0], t), _cLerp(c[1], d[1], t), _cLerp(c[2], d[2], t)];

// One faceted quartz-like crystal: body + lit facet + shadow facet + tip glint.
function _crystal(cx, cy, size, rot, base, rng) {
    const w = size * (0.46 + 0.34 * rng()), h = size * (1.05 + 0.75 * rng());
    const jg = (rng() - 0.5) * 0.18;
    const bottom = [w * 0.10 * jg, h / 2];
    const P = [[0, -h / 2], [w / 2, -h * 0.20], [w * 0.40, h * 0.30], bottom, [-w * 0.40, h * 0.28], [-w / 2, -h * 0.18]];
    const fmt = (p) => p[0].toFixed(1) + ',' + p[1].toFixed(1);
    const body = P.map(fmt).join(' ');
    const litFace = [P[0], P[5], P[4], bottom].map(fmt).join(' ');
    const shFace = [P[0], P[1], P[2], bottom].map(fmt).join(' ');
    const lit = _cShade(base, 1.28), dk = _cShade(base, 0.72);
    let g = '<g transform="translate(' + cx.toFixed(1) + ',' + cy.toFixed(1) + ') rotate(' + rot.toFixed(1) + ')">';
    g += '<polygon points="' + body + '" fill="' + _cRgba(base, 0.60) + '" stroke="' + _cRgba(_cShade(base, 1.5), 0.7) + '" stroke-width="0.5"/>';
    g += '<polygon points="' + litFace + '" fill="' + _cRgba(lit, 0.55) + '"/>';
    g += '<polygon points="' + shFace + '" fill="' + _cRgba(dk, 0.5) + '"/>';
    if (rng() > 0.45) {
        const gl = (w * 0.16).toFixed(1);
        g += '<polygon points="0,' + (-h / 2).toFixed(1) + ' ' + gl + ',' + (-h * 0.24).toFixed(1) + ' ' + (-gl) + ',' + (-h * 0.24).toFixed(1) + '" fill="rgba(255,255,255,0.75)"/>';
    }
    return g + '</g>';
}

function buildCrustSVG(W, H) {
    const rng = mulberry32(0x1F5A2C);       // fixed seed -> stable crust for a given width
    const cy = H / 2, stickH = 5;
    let s = '<svg class="gen-crust" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ' + W + ' ' + H
        + '" preserveAspectRatio="none" style="width:' + W + 'px;height:100%">'
        + '<defs>'
        + '<linearGradient id="genCore" x1="0" x2="1"><stop offset="0" stop-color="#ff63b0"/>'
        + '<stop offset="0.5" stop-color="#c95cee"/><stop offset="1" stop-color="#a759f5"/></linearGradient>'
        + '<filter id="genSoft" x="-20%" y="-60%" width="140%" height="220%"><feGaussianBlur stdDeviation="2.6"/></filter>'
        + '</defs>';
    // Coated spine (a soft glow copy + the solid core) spanning the full stick;
    // the outer clip window decides how much of it shows.
    s += '<rect x="0" y="' + (cy - 8) + '" width="' + W + '" height="16" rx="8" fill="url(#genCore)" opacity="0.4" filter="url(#genSoft)"/>';
    s += '<rect x="0" y="' + (cy - stickH / 2) + '" width="' + W + '" height="' + stickH + '" rx="' + (stickH / 2) + '" fill="url(#genCore)"/>';

    // Dense crystal field across the whole stick; small grains first, big last.
    const crystals = [];
    const STEP = 6.2;
    for (let x = 2; x <= W; x += STEP) {
        const per = 2 + Math.floor(rng() * 2);
        for (let k = 0; k < per; k++) {
            const cxj = x + (rng() - 0.5) * STEP * 1.7;
            const size = 3 + Math.pow(rng(), 1.9) * 12;
            const spread = 3 + (size / 15) * 7 + rng() * 5;
            const cyj = cy + (rng() < 0.5 ? -1 : 1) * rng() * spread;
            const rot = (rng() - 0.5) * 80;
            let base = _colorAt(cxj / W);
            base = _cShade(base, 0.82 + rng() * 0.4);
            if (rng() < 0.10) base = _cMix(base, [235, 255, 248], 0.55);  // occasional white sparkle grain
            crystals.push({ x: cxj, y: cyj, size, rot, base });
        }
    }
    crystals.sort((a, b) => a.size - b.size);
    for (const c of crystals) s += _crystal(c.x, c.y, c.size, c.rot, c.base, rng);

    // Larger turquoise milestone gem at every 5% mark, on top with a soft glow.
    const TURQ = [64, 224, 208];
    for (let i = 1; i <= 20; i++) {
        const px = W * (i * 5) / 100;
        const mr = mulberry32(0x7EA1 + i * 997);
        const size = 18 + mr() * 4;          // a little bigger than the crust crystals
        const rot = (mr() - 0.5) * 34;
        const y = cy - 2 - mr() * 3;
        s += '<ellipse cx="' + px.toFixed(1) + '" cy="' + cy + '" rx="' + (size * 0.72).toFixed(1) + '" ry="' + (size * 0.85).toFixed(1) + '" fill="rgba(64,224,208,0.30)" filter="url(#genSoft)"/>';
        s += _crystal(px, y, size, rot, TURQ, mr);
    }
    return s + '</svg>';
}

function buildGenCrust() {
    if (!genTrack || !genReveal) return;
    const W = Math.max(1, Math.round(genTrack.clientWidth));
    genReveal.innerHTML = buildCrustSVG(W, 42);
}

// Rotating flavor text shown while a batch is cooking. The first five are the
// headline messages; the rest are extra puns to keep long batches sweet.
const FLAVOR_MESSAGES = [
    'Melting the raw data into a workable syrup…',
    'Sorting the soundbytes into flavor profiles…',
    'Tempering the tone for that perfect chocolatey smoothness…',
    'Balancing the sweet highs and sour lows…',
    'Putting a cherry on top of the final output…',
    'Pulling the taffy until every vowel stretches just right…',
    'Dusting the consonants with a little powdered sugar…',
    'Letting the syllables set in the candy mould…',
];

let flavorTimer = null;
let flavorIdx = 0;
let genTotalChunks = 0;

function setGenFlavor(msg, animate = true) {
    if (!genFlavor) return;
    if (!animate) { genFlavor.textContent = msg; return; }
    genFlavor.classList.add('swap');           // fade out
    setTimeout(() => {
        genFlavor.textContent = msg;
        genFlavor.classList.remove('swap');    // fade back in
    }, 300);
}

function setGenPercent(pct) {
    const clamped = Math.max(0, Math.min(100, pct));
    if (genReveal) genReveal.style.width = `${clamped}%`;   // grow the crust's clip window
    if (genPercent) genPercent.textContent = `${Math.round(clamped)}%`;
}

function startGenProgress(totalChunks) {
    genTotalChunks = totalChunks > 0 ? totalChunks : 0;
    flavorIdx = 0;
    if (genProgress) {
        genProgress.classList.add('visible');
        genProgress.setAttribute('aria-hidden', 'false');
    }
    buildGenCrust();                 // needs the track to be visible to measure its width
    setGenFlavor(FLAVOR_MESSAGES[0], false);
    setGenPercent(0);
    if (flavorTimer) clearInterval(flavorTimer);
    flavorTimer = setInterval(() => {
        flavorIdx = (flavorIdx + 1) % FLAVOR_MESSAGES.length;
        setGenFlavor(FLAVOR_MESSAGES[flavorIdx]);
    }, 2600);
}

// Rebuild the crust at the new width if the window resizes mid-generation.
let _crustResizeTimer = null;
window.addEventListener('resize', () => {
    if (!genProgress || !genProgress.classList.contains('visible')) return;
    clearTimeout(_crustResizeTimer);
    _crustResizeTimer = setTimeout(buildGenCrust, 150);
});

function updateGenProgressFromChunk(index) {
    if (genTotalChunks > 0) setGenPercent(((index + 1) / genTotalChunks) * 100);
}

function stopFlavorRotation() {
    if (flavorTimer) { clearInterval(flavorTimer); flavorTimer = null; }
}

function finishGenProgress(message) {
    stopFlavorRotation();
    setGenPercent(100);
    setGenFlavor(message);
}

function hideGenProgress() {
    stopFlavorRotation();
    if (genProgress) {
        genProgress.classList.remove('visible');
        genProgress.setAttribute('aria-hidden', 'true');
    }
    if (genReveal) genReveal.style.width = '0%';
    if (genPercent) genPercent.textContent = '0%';
}

// Custom audio player control refs
const playPauseBtn = document.getElementById('playPauseBtn');
const playPauseIcon = document.getElementById('playPauseIcon');
const timeDisplay = document.getElementById('timeDisplay');
const progressContainer = document.getElementById('progressContainer');
const progressBar = document.getElementById('progressBar');
const volumeIcon = document.getElementById('volumeIcon');
const volumeSlider = document.getElementById('volumeSlider');

const toastContainer = document.getElementById('toastContainer');

function showToast(message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    toastContainer.appendChild(toast);

    // Remove after animation completes
    setTimeout(() => {
        toast.remove();
    }, 4000);
}

// --- Voice panel: "Sweet Treats" (favorites, expanded) + "Unwrapped Candy" (dropdown) ---
function fmtDuration(sec) {
    if (sec == null || !isFinite(sec)) return '';
    const m = Math.floor(sec / 60);
    const s = Math.floor(sec % 60);
    return `${m}:${String(s).padStart(2, '0')}`;
}

const alpha = (list) => [...list].sort((a, b) => a.localeCompare(b));
const favoriteVoices = () => alpha(voices.filter((v) => favorites.has(v)));
const otherVoiceNames = () => alpha(voices.filter((v) => !favorites.has(v)));

function renderVoiceList() {
    renderFavorites();
    renderOthers();
    updatePreviewButtons();
}

function renderFavorites() {
    favoriteList.textContent = '';
    const favs = favoriteVoices();
    if (favs.length === 0) {
        const empty = document.createElement('div');
        empty.className = 'voice-empty';
        empty.textContent = 'Candy jar’s empty — pick a voice below and tap ☆ to sweeten it up.';
        favoriteList.appendChild(empty);
        return;
    }
    for (const name of favs) {
        favoriteList.appendChild(makeVoiceRow(name));
    }
}

function makeVoiceRow(name) {
    const row = document.createElement('div');
    row.className = 'voice-row' + (name === selectedVoice ? ' selected' : '');
    row.dataset.voice = name;
    row.setAttribute('role', 'option');
    row.setAttribute('aria-selected', name === selectedVoice ? 'true' : 'false');

    const fav = document.createElement('button');
    fav.type = 'button';
    fav.className = 'voice-star on';
    fav.title = 'Remove from Sweet Treats';
    fav.textContent = '★';
    fav.addEventListener('click', (e) => { e.stopPropagation(); toggleFavorite(name); });

    const label = document.createElement('span');
    label.className = 'voice-name';
    label.textContent = name;

    const dur = document.createElement('span');
    dur.className = 'voice-duration';
    dur.textContent = fmtDuration(durations[name]);

    const play = document.createElement('button');
    play.type = 'button';
    play.className = 'voice-btn voice-preview';
    play.title = 'Preview this voice';
    play.textContent = '▶';
    play.addEventListener('click', (e) => { e.stopPropagation(); previewVoice(name); });

    const rename = document.createElement('button');
    rename.type = 'button';
    rename.className = 'voice-btn voice-rename';
    rename.title = 'Rename this voice';
    rename.textContent = '✎';
    rename.addEventListener('click', (e) => { e.stopPropagation(); promptRename(name); });

    const del = document.createElement('button');
    del.type = 'button';
    del.className = 'voice-btn voice-delete';
    del.title = 'Delete this voice';
    del.textContent = '🗑';
    del.addEventListener('click', (e) => { e.stopPropagation(); deleteVoice(name); });

    row.append(fav, label, dur, play, rename, del);
    row.addEventListener('click', () => selectVoice(name));
    return row;
}

function renderOthers() {
    const others = otherVoiceNames();
    otherVoices.textContent = '';
    if (others.length === 0) {
        const opt = document.createElement('option');
        opt.value = '';
        opt.textContent = voices.length ? 'Every flavor’s a favorite ✨' : 'Jar’s empty — add a flavor below';
        otherVoices.appendChild(opt);
    } else {
        const placeholder = document.createElement('option');
        placeholder.value = '';
        placeholder.textContent = 'Select a voice…';
        otherVoices.appendChild(placeholder);
        for (const name of others) {
            const opt = document.createElement('option');
            opt.value = name;
            const d = fmtDuration(durations[name]);
            opt.textContent = d ? `${name}  (${d})` : name;
            otherVoices.appendChild(opt);
        }
    }
    syncOtherControls();
}

function syncOtherControls() {
    // The dropdown reflects the selected voice only when it is a non-favorite.
    const isOther = selectedVoice && !favorites.has(selectedVoice) && voices.includes(selectedVoice);
    otherVoices.value = isOther ? selectedVoice : '';
    const hasSel = !!otherVoices.value;
    otherPreview.disabled = !hasSel;
    otherRename.disabled = !hasSel;
    otherFavorite.disabled = !hasSel;
    otherDelete.disabled = !hasSel;
    updateSelectedIndicator();
}

// Make it unmistakable which voice is active and which section it lives in.
function updateSelectedIndicator() {
    const hasSel = !!(selectedVoice && voices.includes(selectedVoice));
    const inFav = hasSel && favorites.has(selectedVoice);

    if (selectedVoiceBadge) {
        selectedVoiceBadge.textContent = '';
        if (hasSel) {
            const strong = document.createElement('strong');
            strong.textContent = selectedVoice;
            selectedVoiceBadge.append('🎙 ', strong,
                document.createTextNode(inFav ? ' · Sweet Treats' : ' · Unwrapped Candy'));
            selectedVoiceBadge.classList.add('has-voice');
        } else {
            const none = document.createElement('span');
            none.className = 'selected-voice-none';
            none.textContent = 'none selected';
            selectedVoiceBadge.appendChild(none);
            selectedVoiceBadge.classList.remove('has-voice');
        }
    }

    const favSection = favoriteList && favoriteList.closest('.voice-section');
    const otherSection = otherPicker && otherPicker.closest('.voice-section');
    if (favSection) favSection.classList.toggle('active-section', hasSel && inFav);
    if (otherSection) otherSection.classList.toggle('active-section', hasSel && !inFav);
    if (otherPicker) otherPicker.classList.toggle('active', hasSel && !inFav);
}

function selectVoice(name) {
    selectedVoice = name;
    favoriteList.querySelectorAll('.voice-row').forEach((r) => {
        const on = r.dataset.voice === name;
        r.classList.toggle('selected', on);
        r.setAttribute('aria-selected', on ? 'true' : 'false');
    });
    syncOtherControls();
    updatePreviewButtons();
}

function addVoice(name) {
    if (!voices.includes(name)) voices.push(name);
    if (!selectedVoice) selectedVoice = name;
    renderVoiceList();
}

function removeVoice(name) {
    voices = voices.filter((v) => v !== name);
    delete durations[name];
    if (favorites.delete(name)) saveFavorites();
    if (selectedVoice === name) selectedVoice = alpha(voices)[0] || null;
    renderVoiceList();
}

function renameVoiceInList(oldName, newName) {
    if (!voices.includes(oldName)) return;  // already applied (e.g. via SSE)
    voices = voices.map((v) => (v === oldName ? newName : v));
    if (oldName in durations) { durations[newName] = durations[oldName]; delete durations[oldName]; }
    if (favorites.delete(oldName)) { favorites.add(newName); saveFavorites(); }
    if (selectedVoice === oldName) selectedVoice = newName;
    renderVoiceList();
}

function toggleFavorite(name) {
    if (favorites.has(name)) favorites.delete(name);
    else favorites.add(name);
    saveFavorites();
    renderVoiceList();
}

// Voice preview via a hidden <audio> element
let previewingVoice = null;

function previewVoice(name) {
    if (!name) return;
    if (previewingVoice === name && !voicePreview.paused) {
        voicePreview.pause();
        return;
    }
    previewingVoice = name;
    voicePreview.src = `/voices/${encodeURIComponent(name)}/audio`;
    voicePreview.play().catch((err) => {
        console.error('Preview failed:', err);
        showToast(`Could not play '${name}'`, 'error');
        previewingVoice = null;
        updatePreviewButtons();
    });
}

function updatePreviewButtons() {
    const playingName = (previewingVoice && !voicePreview.paused) ? previewingVoice : null;
    favoriteList.querySelectorAll('.voice-row').forEach((r) => {
        const btn = r.querySelector('.voice-preview');
        if (!btn) return;
        const on = r.dataset.voice === playingName;
        btn.textContent = on ? '⏸' : '▶';
        btn.classList.toggle('playing', on);
    });
    const otherOn = otherVoices.value && otherVoices.value === playingName;
    otherPreview.textContent = otherOn ? '⏸' : '▶';
    otherPreview.classList.toggle('playing', !!otherOn);
}

if (voicePreview) {
    voicePreview.addEventListener('play', updatePreviewButtons);
    voicePreview.addEventListener('pause', updatePreviewButtons);
    voicePreview.addEventListener('ended', () => { previewingVoice = null; updatePreviewButtons(); });
}

async function promptRename(name) {
    if (!name) return;
    const input = window.prompt(`Rename voice "${name}" to:`, name);
    if (input === null) return;
    const newName = input.trim();
    if (!newName || newName === name) return;
    try {
        const resp = await fetch(`/voices/${encodeURIComponent(name)}/rename`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ new_name: newName }),
        });
        const data = await resp.json().catch(() => ({}));
        if (!resp.ok) throw new Error(data.detail || `Rename failed (${resp.status})`);
        renameVoiceInList(name, data.new);
        showToast(`Renamed to '${data.new}'`, 'success');
    } catch (err) {
        console.error('Rename failed:', err);
        showToast(err.message || 'Rename failed', 'error');
    }
}

// "Unwrapped Candy" dropdown + action buttons
otherVoices.addEventListener('change', () => {
    if (otherVoices.value) selectVoice(otherVoices.value);
    else syncOtherControls();
});
otherPreview.addEventListener('click', () => previewVoice(otherVoices.value));
otherRename.addEventListener('click', () => { if (otherVoices.value) promptRename(otherVoices.value); });
otherFavorite.addEventListener('click', () => { if (otherVoices.value) toggleFavorite(otherVoices.value); });
otherDelete.addEventListener('click', () => { if (otherVoices.value) deleteVoice(otherVoices.value); });

async function deleteVoice(name) {
    if (!name) return;
    if (!window.confirm(`Delete voice "${name}"? This removes its .wav for good.`)) return;
    try {
        const resp = await fetch(`/voices/${encodeURIComponent(name)}`, { method: 'DELETE' });
        const data = await resp.json().catch(() => ({}));
        if (!resp.ok) throw new Error(data.detail || `Delete failed (${resp.status})`);
        removeVoice(name);
        showToast(`Deleted '${name}'`, 'info');
    } catch (err) {
        console.error('Delete failed:', err);
        showToast(err.message || 'Delete failed', 'error');
    }
}

// Voice event source for cleanup
let voiceEventSource = null;

// Subscribe to voice events
function subscribeToVoiceEvents() {
    if (voiceEventSource) {
        voiceEventSource.close();
    }
    voiceEventSource = new EventSource('/voice-events');

    voiceEventSource.onmessage = (event) => {
        const data = JSON.parse(event.data);

        if (data.type === 'processing') {
            showToast(`Processing '${data.voice}'...`, 'info');
        } else if (data.type === 'ready') {
            showToast(`Voice '${data.voice}' ready`, 'success');
            addVoice(data.voice);
        } else if (data.type === 'error') {
            showToast(`Failed to load '${data.voice}': ${data.reason}`, 'error');
        } else if (data.type === 'removed') {
            showToast(`Voice '${data.voice}' removed`, 'info');
            removeVoice(data.voice);
        } else if (data.type === 'renamed') {
            renameVoiceInList(data.old, data.new);
        }
    };

    voiceEventSource.onerror = (error) => {
        console.error('Voice events connection error:', error);
        voiceEventSource.close();
        // Reconnect after 5 seconds
        setTimeout(subscribeToVoiceEvents, 5000);
    };
}

let audioBuffers = [];
let audioContext = null;
let processingChunk = false;
let chunkQueue = [];
let currentAbortController = null;
let currentGenerationId = null;  // Track generation ID for stop requests

// AudioBufferSourceNode scheduling state
let nextStartTime = 0;
let gainNode = null;
let isStreaming = false;

// Playback state management
let isPlaying = false;
let playbackStartTime = 0;        // audioContext.currentTime when play started
let playbackOffset = 0;           // where in the audio we started from
let totalDuration = 0;            // sum of all buffer durations
let currentSource = null;         // current AudioBufferSourceNode
let animationFrameId = null;      // for requestAnimationFrame loop
let previousVolume = 100;         // for mute/unmute toggle

// Theme handling (cycles dark \u2192 light \u2192 candy)
const THEMES = ['dark', 'light', 'candy'];
const THEME_ICONS = { dark: '\uD83C\uDF19', light: '\u2600\uFE0F', candy: '\uD83C\uDF6C' };

function setTheme(theme) {
    if (!THEMES.includes(theme)) theme = 'dark';
    document.documentElement.setAttribute('data-theme', theme);
    themeToggle.textContent = THEME_ICONS[theme];
    themeToggle.title = `Theme: ${theme} \u2014 click to switch`;
    localStorage.setItem('theme', theme);
}

function toggleTheme() {
    const current = document.documentElement.getAttribute('data-theme') || 'dark';
    const next = THEMES[(THEMES.indexOf(current) + 1) % THEMES.length];
    setTheme(next);
}

// Load saved theme or default to dark
const savedTheme = localStorage.getItem('theme') || 'dark';
setTheme(savedTheme);

themeToggle.addEventListener('click', toggleTheme);

// Format time in MM:SS or H:MM:SS for times >= 1 hour
function formatTime(seconds) {
    if (!isFinite(seconds) || seconds < 0) seconds = 0;
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);

    if (hours > 0) {
        return `${hours}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    }
    return `${minutes}:${secs.toString().padStart(2, '0')}`;
}

// Get current playback time
function getCurrentTime() {
    if (!audioContext) return 0;
    if (isPlaying) {
        return playbackOffset + (audioContext.currentTime - playbackStartTime);
    }
    return playbackOffset;
}

// Update time display and progress bar
function updateTimeDisplay() {
    const current = getCurrentTime();
    const duration = totalDuration;

    timeDisplay.textContent = `${formatTime(current)} / ${formatTime(duration)}`;

    if (duration > 0) {
        const percentage = Math.min((current / duration) * 100, 100);
        progressBar.style.width = `${percentage}%`;
        progressContainer.setAttribute('aria-valuenow', Math.round(percentage));
    } else {
        progressBar.style.width = '0%';
        progressContainer.setAttribute('aria-valuenow', 0);
    }
}

// Start the playback animation loop
function startPlaybackLoop() {
    function loop() {
        updateTimeDisplay();

        // Check if we've reached the end
        const current = getCurrentTime();
        if (current >= totalDuration && totalDuration > 0 && !isStreaming) {
            // Playback ended naturally
            handlePlaybackEnded();
            return;
        }

        if (isPlaying) {
            animationFrameId = requestAnimationFrame(loop);
        }
    }
    animationFrameId = requestAnimationFrame(loop);
}

// Stop the playback animation loop
function stopPlaybackLoop() {
    if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
        animationFrameId = null;
    }
}

// Handle when playback reaches the end naturally
function handlePlaybackEnded() {
    isPlaying = false;
    playbackOffset = 0;
    currentSource = null;
    playPauseIcon.textContent = '\u25B6';
    playPauseIcon.dataset.state = 'play';
    stopPlaybackLoop();
    updateTimeDisplay();
}

// Reset all playback state for a new generation
function resetPlaybackState() {
    // Stop current playback if playing
    if (isPlaying) {
        stopPlayback();
    }

    // Stop any existing source
    if (currentSource) {
        try {
            currentSource.onended = null;
            currentSource.stop();
        } catch (e) {
            // Source may already be stopped
        }
        currentSource = null;
    }

    // Cancel animation frames
    stopPlaybackLoop();

    // Reset playback state variables
    isPlaying = false;
    playbackStartTime = 0;
    playbackOffset = 0;
    totalDuration = 0;

    // Reset UI
    playPauseIcon.textContent = '\u25B6';
    playPauseIcon.dataset.state = 'play';
    timeDisplay.textContent = '0:00 / 0:00';
    progressBar.style.width = '0%';
    progressContainer.setAttribute('aria-valuenow', 0);

    // Hide audio player
    audioPlayerDiv.classList.remove('visible');
}

// Start playback from a given offset
function startPlayback(fromOffset = playbackOffset) {
    if (audioBuffers.length === 0) return;

    // Stop any existing source
    if (currentSource) {
        try {
            currentSource.onended = null;
            currentSource.stop();
        } catch (e) {
            // Source may already be stopped
        }
        currentSource = null;
    }

    // Concatenate all buffers
    const concatenatedBuffer = concatenateAudioBuffers(audioBuffers);
    if (!concatenatedBuffer) return;

    // Clamp offset to valid range
    fromOffset = Math.max(0, Math.min(fromOffset, concatenatedBuffer.duration));

    // Create new AudioBufferSourceNode
    const source = audioContext.createBufferSource();
    source.buffer = concatenatedBuffer;
    source.connect(gainNode);

    // Start at the specified offset
    source.start(0, fromOffset);

    // Update state
    currentSource = source;
    isPlaying = true;
    playbackStartTime = audioContext.currentTime;
    playbackOffset = fromOffset;

    // Update UI
    playPauseIcon.textContent = '\u23F8';
    playPauseIcon.dataset.state = 'pause';
    audioPlayerDiv.classList.add('visible');

    // Start animation loop
    startPlaybackLoop();

    // Handle natural end of playback
    source.onended = () => {
        // Only handle if this is still the current source and we're still "playing"
        if (currentSource === source && isPlaying) {
            // Calculate where we should be in the audio
            const elapsedTime = audioContext.currentTime - playbackStartTime;
            const endedAtPosition = fromOffset + elapsedTime;

            // Check if more audio has arrived since we started
            // (totalDuration may have grown while we were playing)
            if (isStreaming && endedAtPosition < totalDuration) {
                // More audio available - restart from where we left off
                console.log(`Continuing playback: ended at ${endedAtPosition.toFixed(2)}s, total now ${totalDuration.toFixed(2)}s`);
                startPlayback(endedAtPosition);
            } else if (!isStreaming && endedAtPosition < totalDuration - 0.1) {
                // Not streaming but still more audio (edge case: final chunks arrived just before end)
                console.log(`Final continuation: ended at ${endedAtPosition.toFixed(2)}s, total ${totalDuration.toFixed(2)}s`);
                startPlayback(endedAtPosition);
            } else {
                // Truly finished
                handlePlaybackEnded();
            }
        }
    };
}

// Stop/pause playback
function stopPlayback() {
    if (!isPlaying) return;

    // Record current position
    playbackOffset = getCurrentTime();

    // Stop the source
    if (currentSource) {
        try {
            currentSource.onended = null;
            currentSource.stop();
        } catch (e) {
            // Source may already be stopped
        }
        currentSource = null;
    }

    // Update state
    isPlaying = false;
    playPauseIcon.textContent = '\u25B6';
    playPauseIcon.dataset.state = 'play';

    // Stop animation loop
    stopPlaybackLoop();
    updateTimeDisplay();
}

// Toggle play/pause
function togglePlayPause() {
    if (isPlaying) {
        stopPlayback();
    } else {
        startPlayback();
    }
}

// Seek to a position (0-1 ratio)
function seekTo(ratio) {
    const targetTime = ratio * totalDuration;
    playbackOffset = targetTime;

    if (isPlaying) {
        // Restart playback from new position
        startPlayback(targetTime);
    } else {
        // Just update display
        updateTimeDisplay();
    }
}

// Event listeners for audio controls
playPauseBtn.addEventListener('click', togglePlayPause);

// Volume slider control
volumeSlider.addEventListener('input', () => {
    const volume = volumeSlider.value / 100;
    if (gainNode) {
        gainNode.gain.value = volume;
    }

    // Update volume icon
    if (volume === 0) {
        volumeIcon.textContent = '\uD83D\uDD07';
    } else if (volume < 0.5) {
        volumeIcon.textContent = '\uD83D\uDD09';
    } else {
        volumeIcon.textContent = '\uD83D\uDD0A';
    }
});

// Volume icon click to toggle mute
volumeIcon.addEventListener('click', () => {
    if (volumeSlider.value > 0) {
        // Mute: store current volume and set to 0
        previousVolume = volumeSlider.value;
        volumeSlider.value = 0;
    } else {
        // Unmute: restore previous volume
        volumeSlider.value = previousVolume || 100;
    }
    // Trigger input event to update gainNode
    volumeSlider.dispatchEvent(new Event('input'));
});

// Progress bar click to seek
progressContainer.addEventListener('click', (e) => {
    if (totalDuration <= 0) return;

    const rect = progressContainer.getBoundingClientRect();
    const ratio = (e.clientX - rect.left) / rect.width;
    seekTo(Math.max(0, Math.min(1, ratio)));
});

// Progress bar drag to seek
let isDraggingProgress = false;
let dragRatio = 0;

function updateDragVisual(ratio) {
    // Update visual without actually seeking (to avoid creating many AudioBufferSourceNodes)
    const clampedRatio = Math.max(0, Math.min(1, ratio));
    const percentage = clampedRatio * 100;
    progressBar.style.width = `${percentage}%`;
    progressContainer.setAttribute('aria-valuenow', Math.round(percentage));

    // Update time display to show where we'd seek to
    const targetTime = clampedRatio * totalDuration;
    timeDisplay.textContent = `${formatTime(targetTime)} / ${formatTime(totalDuration)}`;
}

progressContainer.addEventListener('mousedown', (e) => {
    if (totalDuration <= 0) return;

    isDraggingProgress = true;
    const rect = progressContainer.getBoundingClientRect();
    dragRatio = (e.clientX - rect.left) / rect.width;
    updateDragVisual(dragRatio);

    // Prevent text selection while dragging
    e.preventDefault();
});

document.addEventListener('mousemove', (e) => {
    if (!isDraggingProgress) return;

    const rect = progressContainer.getBoundingClientRect();
    dragRatio = (e.clientX - rect.left) / rect.width;
    updateDragVisual(dragRatio);
});

document.addEventListener('mouseup', (e) => {
    if (!isDraggingProgress) return;

    isDraggingProgress = false;

    // Finalize seek to the drag position
    const finalRatio = Math.max(0, Math.min(1, dragRatio));
    seekTo(finalRatio);
});

// Keyboard support for progress bar
progressContainer.addEventListener('keydown', (e) => {
    if (totalDuration <= 0) return;

    const currentRatio = getCurrentTime() / totalDuration;
    let newRatio = currentRatio;

    switch (e.key) {
        case 'ArrowLeft':
            newRatio = Math.max(0, currentRatio - 0.05);
            break;
        case 'ArrowRight':
            newRatio = Math.min(1, currentRatio + 0.05);
            break;
        case 'Home':
            newRatio = 0;
            break;
        case 'End':
            newRatio = 1;
            break;
        default:
            return;
    }

    e.preventDefault();
    seekTo(newRatio);
});

// Load voices on page load
async function loadVoices() {
    try {
        const response = await fetch('/voices');
        const data = await response.json();
        const list = Array.isArray(data.voices) ? data.voices : [];
        voices = list.map((v) => v.name);
        durations = {};
        for (const v of list) durations[v.name] = v.duration_seconds;

        // Drop favorites for voices that no longer exist
        let changed = false;
        for (const f of [...favorites]) {
            if (!voices.includes(f)) { favorites.delete(f); changed = true; }
        }
        if (changed) saveFavorites();

        if (!selectedVoice || !voices.includes(selectedVoice)) {
            selectedVoice = alpha(voices)[0] || null;
        }
        renderVoiceList();
    } catch (error) {
        console.error('Failed to load voices:', error);
        favoriteList.textContent = '';
        const err = document.createElement('div');
        err.className = 'voice-empty';
        err.textContent = 'Error loading voices';
        favoriteList.appendChild(err);
    }
}

// --- Voice upload (click to browse or drag-and-drop a .wav) ---
const voiceDrop = document.getElementById('voiceDrop');
const voiceFileInput = document.getElementById('voiceFile');
const voiceDropText = document.getElementById('voiceDropText');
let uploadingVoice = false;

async function uploadVoiceFile(file) {
    if (!file || uploadingVoice) return;

    if (!file.name.toLowerCase().endsWith('.wav')) {
        showToast('Only .wav files are supported', 'error');
        return;
    }

    uploadingVoice = true;
    voiceDrop.classList.add('uploading');
    const originalText = voiceDropText.innerHTML;
    voiceDropText.textContent = `Uploading ${file.name}...`;

    try {
        const formData = new FormData();
        formData.append('file', file);

        const response = await fetch('/voices', { method: 'POST', body: formData });
        const data = await response.json().catch(() => ({}));

        if (!response.ok) {
            throw new Error(data.detail || `Upload failed (${response.status})`);
        }

        // Add + select immediately (the /voice-events stream may also deliver a
        // 'ready' event; addVoice is idempotent so this is safe).
        if (data.voice) {
            addVoice(data.voice);
            selectVoice(data.voice);
            showToast(`Voice '${data.voice}' added`, 'success');
        }
    } catch (error) {
        console.error('Voice upload failed:', error);
        showToast(error.message || 'Voice upload failed', 'error');
    } finally {
        uploadingVoice = false;
        voiceDrop.classList.remove('uploading');
        voiceDropText.innerHTML = originalText;
        voiceFileInput.value = '';  // allow re-selecting the same file
    }
}

voiceDrop.addEventListener('click', () => {
    if (!uploadingVoice) voiceFileInput.click();
});

voiceDrop.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        if (!uploadingVoice) voiceFileInput.click();
    }
});

voiceFileInput.addEventListener('change', () => {
    if (voiceFileInput.files.length > 0) {
        uploadVoiceFile(voiceFileInput.files[0]);
    }
});

['dragenter', 'dragover'].forEach((evt) => {
    voiceDrop.addEventListener(evt, (e) => {
        e.preventDefault();
        e.stopPropagation();
        voiceDrop.classList.add('dragover');
    });
});

voiceDrop.addEventListener('dragleave', (e) => {
    e.preventDefault();
    e.stopPropagation();
    // Ignore leave events fired while moving over child elements
    if (voiceDrop.contains(e.relatedTarget)) return;
    voiceDrop.classList.remove('dragover');
});

voiceDrop.addEventListener('drop', (e) => {
    e.preventDefault();
    e.stopPropagation();
    voiceDrop.classList.remove('dragover');
    const files = e.dataTransfer && e.dataTransfer.files;
    if (files && files.length > 0) {
        uploadVoiceFile(files[0]);
    }
});

// --- Load text from a .txt / .epub file ---
const loadTextBtn = document.getElementById('loadTextBtn');
const textFile = document.getElementById('textFile');

if (loadTextBtn && textFile) {
    loadTextBtn.addEventListener('click', () => textFile.click());
    textFile.addEventListener('change', async () => {
        const file = textFile.files && textFile.files[0];
        if (!file) return;
        const name = file.name.toLowerCase();
        if (!name.endsWith('.txt') && !name.endsWith('.epub')) {
            showToast('Only .txt and .epub files are supported', 'error');
            textFile.value = '';
            return;
        }
        const original = loadTextBtn.textContent;
        loadTextBtn.disabled = true;
        loadTextBtn.textContent = 'Reading…';
        try {
            const fd = new FormData();
            fd.append('file', file);
            const resp = await fetch('/extract-text', { method: 'POST', body: fd });
            const data = await resp.json().catch(() => ({}));
            if (!resp.ok) throw new Error(data.detail || `Import failed (${resp.status})`);
            textArea.value = data.text || '';
            textArea.dispatchEvent(new Event('input'));
            const chars = (data.chars || (data.text || '').length).toLocaleString();
            showToast(`Loaded ${chars} characters from ${file.name}`, 'success');
        } catch (err) {
            console.error('Text import failed:', err);
            showToast(err.message || 'Could not read file', 'error');
        } finally {
            loadTextBtn.disabled = false;
            loadTextBtn.textContent = original;
            textFile.value = '';   // allow re-selecting the same file
        }
    });
}

// --- Advanced generation controls (Echo-TTS sampler knobs) ---
const advSteps = document.getElementById('advSteps');
const advCfgText = document.getElementById('advCfgText');
const advCfgSpeaker = document.getElementById('advCfgSpeaker');
const advSeed = document.getElementById('advSeed');
const advStepsVal = document.getElementById('advStepsVal');
const advCfgTextVal = document.getElementById('advCfgTextVal');
const advCfgSpeakerVal = document.getElementById('advCfgSpeakerVal');
const advSeedLast = document.getElementById('advSeedLast');
const advReset = document.getElementById('advReset');
const ADV_DEFAULTS = { steps: 40, cfgText: 3, cfgSpeaker: 8 };

function syncAdvOutputs() {
    if (advStepsVal) advStepsVal.textContent = String(advSteps.value);
    if (advCfgTextVal) advCfgTextVal.textContent = Number(advCfgText.value).toFixed(1);
    if (advCfgSpeakerVal) advCfgSpeakerVal.textContent = Number(advCfgSpeaker.value).toFixed(1);
}
[advSteps, advCfgText, advCfgSpeaker].forEach((el) => el && el.addEventListener('input', syncAdvOutputs));
syncAdvOutputs();

if (advReset) {
    advReset.addEventListener('click', () => {
        advSteps.value = ADV_DEFAULTS.steps;
        advCfgText.value = ADV_DEFAULTS.cfgText;
        advCfgSpeaker.value = ADV_DEFAULTS.cfgSpeaker;
        advSeed.value = '';
        syncAdvOutputs();
    });
}

if (advSeedLast) {
    advSeedLast.addEventListener('click', () => {
        const s = advSeedLast.dataset.seed;
        if (s) { advSeed.value = s; showToast(`Seed ${s} locked in — next run will match`, 'info'); }
    });
}

// Show the seed a run actually used, so a lucky random take can be reproduced.
function showUsedSeed(seed) {
    if (!advSeedLast || seed == null) return;
    advSeedLast.dataset.seed = String(seed);
    advSeedLast.textContent = `🎲 last run used seed ${seed} — click to reuse`;
    advSeedLast.hidden = false;
}

function readAdvancedParams() {
    const p = {
        steps: advSteps ? parseInt(advSteps.value, 10) : undefined,
        cfg_text: advCfgText ? parseFloat(advCfgText.value) : undefined,
        cfg_speaker: advCfgSpeaker ? parseFloat(advCfgSpeaker.value) : undefined,
    };
    if (advSeed && advSeed.value.trim() !== '') {
        const s = parseInt(advSeed.value, 10);
        if (Number.isFinite(s) && s >= 0) p.seed = s;
    }
    return p;
}

// --- Textbook cleaning panel ---
const cleanEnabled = document.getElementById('cleanEnabled');
const cleanPreset = document.getElementById('cleanPreset');
const cleanRulesBox = document.getElementById('cleanRules');
const cleanDict = document.getElementById('cleanDict');
const cleanPreviewBtn = document.getElementById('cleanPreview');
const cleanApplyBtn = document.getElementById('cleanApply');
const cleanReport = document.getElementById('cleanReport');
const CLEAN_KEY = 'candyecho.cleaning';

let cleaningPresets = {};   // preset name -> [rule names]
let cleaningRuleSpecs = []; // [{name, label, description}]
let lastCleanedText = null;

function saveCleaningSettings() {
    try {
        localStorage.setItem(CLEAN_KEY, JSON.stringify(readCleaningOptions()));
    } catch (e) { /* storage full or blocked - settings just won't persist */ }
}

function loadCleaningSettings() {
    try { return JSON.parse(localStorage.getItem(CLEAN_KEY) || 'null'); }
    catch (e) { return null; }
}

// One `word = replacement` per line.
function parseDictionary(raw) {
    const out = {};
    for (const line of (raw || '').split('\n')) {
        const idx = line.indexOf('=');
        if (idx <= 0) continue;
        const key = line.slice(0, idx).trim();
        const value = line.slice(idx + 1).trim();
        if (key) out[key] = value;
    }
    return out;
}

function dictionaryToText(dict) {
    return Object.entries(dict || {}).map(([k, v]) => `${k} = ${v}`).join('\n');
}

// Rules are sent explicitly rather than relying on the preset, so what the
// checkboxes show is exactly what the server applies.
function readCleaningOptions() {
    const rules = {};
    cleanRulesBox.querySelectorAll('input[type=checkbox][data-rule]').forEach((cb) => {
        rules[cb.dataset.rule] = cb.checked;
    });
    return {
        enabled: !!(cleanEnabled && cleanEnabled.checked),
        preset: cleanPreset ? cleanPreset.value : 'textbook',
        rules,
        substitutions: parseDictionary(cleanDict ? cleanDict.value : ''),
    };
}

function applyPresetToCheckboxes(preset) {
    const active = new Set(cleaningPresets[preset] || []);
    cleanRulesBox.querySelectorAll('input[type=checkbox][data-rule]').forEach((cb) => {
        cb.checked = active.has(cb.dataset.rule);
    });
}

function renderCleaningRules(saved) {
    cleanRulesBox.textContent = '';
    const active = new Set(
        saved && saved.rules
            ? Object.keys(saved.rules).filter((k) => saved.rules[k])
            : (cleaningPresets[cleanPreset.value] || [])
    );

    for (const spec of cleaningRuleSpecs) {
        const row = document.createElement('label');
        row.className = 'clean-rule';
        row.title = spec.description;

        const cb = document.createElement('input');
        cb.type = 'checkbox';
        cb.dataset.rule = spec.name;
        cb.checked = active.has(spec.name);
        cb.addEventListener('change', saveCleaningSettings);

        const label = document.createElement('span');
        label.className = 'clean-rule-name';
        label.textContent = spec.label;

        const hint = document.createElement('small');
        hint.className = 'clean-rule-desc';
        hint.textContent = spec.description;

        row.append(cb, label, hint);
        cleanRulesBox.appendChild(row);
    }
}

async function loadCleaningRules() {
    try {
        const resp = await fetch('/cleaning-rules');
        const data = await resp.json();
        cleaningRuleSpecs = data.rules || [];
        cleaningPresets = data.presets || {};

        const saved = loadCleaningSettings();
        if (saved) {
            if (cleanEnabled) cleanEnabled.checked = saved.enabled !== false;
            if (cleanPreset && saved.preset) cleanPreset.value = saved.preset;
            if (cleanDict) cleanDict.value = dictionaryToText(saved.substitutions);
        }
        renderCleaningRules(saved);
    } catch (error) {
        console.error('Failed to load cleaning rules:', error);
        cleanRulesBox.textContent = 'Could not load cleaning rules.';
    }
}

function renderCleanReport(data) {
    cleanReport.textContent = '';
    cleanReport.hidden = false;

    const removed = (data.chars_before || 0) - (data.chars_after || 0);
    const summary = document.createElement('p');
    summary.className = 'clean-summary';
    summary.textContent = removed > 0
        ? `Trimmed ${removed.toLocaleString()} characters (${(data.chars_before || 0).toLocaleString()} → ${(data.chars_after || 0).toLocaleString()}).`
        : 'Nothing needed removing — the text is already clean.';
    cleanReport.appendChild(summary);

    for (const item of data.report || []) {
        const row = document.createElement('div');
        row.className = 'clean-report-row';

        const head = document.createElement('div');
        head.className = 'clean-report-head';
        const count = document.createElement('strong');
        count.textContent = `${item.count}×`;
        head.append(count, document.createTextNode(' ' + item.label));
        row.appendChild(head);

        if (item.samples && item.samples.length) {
            const list = document.createElement('ul');
            list.className = 'clean-samples';
            for (const sample of item.samples) {
                const li = document.createElement('li');
                li.textContent = sample;
                list.appendChild(li);
            }
            row.appendChild(list);
        }
        cleanReport.appendChild(row);
    }
}

async function previewCleaning() {
    const text = textArea.value;
    if (!text.trim()) {
        showToast('Nothing to clean — add some text first', 'error');
        return;
    }
    const opts = readCleaningOptions();
    const original = cleanPreviewBtn.textContent;
    cleanPreviewBtn.disabled = true;
    cleanPreviewBtn.textContent = 'Checking…';
    try {
        const resp = await fetch('/clean-text', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                text,
                preset: opts.preset,
                rules: opts.rules,
                substitutions: opts.substitutions,
            }),
        });
        const data = await resp.json().catch(() => ({}));
        if (!resp.ok) throw new Error(data.detail || `Preview failed (${resp.status})`);
        lastCleanedText = data.text || '';
        renderCleanReport(data);
        cleanApplyBtn.hidden = lastCleanedText === text;
    } catch (error) {
        console.error('Cleaning preview failed:', error);
        showToast(error.message || 'Cleaning preview failed', 'error');
    } finally {
        cleanPreviewBtn.disabled = false;
        cleanPreviewBtn.textContent = original;
    }
}

if (cleanPreviewBtn) cleanPreviewBtn.addEventListener('click', previewCleaning);

if (cleanApplyBtn) {
    cleanApplyBtn.addEventListener('click', () => {
        if (lastCleanedText === null) return;
        textArea.value = lastCleanedText;
        textArea.dispatchEvent(new Event('input'));
        cleanApplyBtn.hidden = true;
        showToast('Cleaned text applied', 'success');
    });
}

if (cleanPreset) {
    cleanPreset.addEventListener('change', () => {
        applyPresetToCheckboxes(cleanPreset.value);
        saveCleaningSettings();
    });
}
if (cleanEnabled) cleanEnabled.addEventListener('change', saveCleaningSettings);
if (cleanDict) cleanDict.addEventListener('change', saveCleaningSettings);

// Handle form submission
form.addEventListener('submit', async (e) => {
    e.preventDefault();

    const text = textArea.value.trim();
    const voice = selectedVoice;

    if (!text || !voice) {
        showError('Please enter text and select a voice');
        return;
    }

    // Reset all playback state before starting new generation
    resetPlaybackState();

    // Close existing AudioContext if present
    if (audioContext) {
        try {
            audioContext.close();
        } catch (e) {}
    }

    // Reset audio data state
    audioBuffers = [];
    chunkQueue = [];
    processingChunk = false;

    // Create fresh AudioContext
    audioContext = new (window.AudioContext || window.webkitAudioContext)();

    // Create GainNode for volume control and connect to destination
    gainNode = audioContext.createGain();
    gainNode.connect(audioContext.destination);
    // Apply current volume slider value
    gainNode.gain.value = volumeSlider.value / 100;

    // Reset scheduling state
    nextStartTime = 0;
    isStreaming = true;

    // Clear UI messages
    progressDiv.classList.remove('visible');
    errorDiv.classList.remove('visible');
    hideGenProgress();
    generateBtn.disabled = true;

    // Start generation
    try {
        currentAbortController = new AbortController();
        const response = await fetch('/generate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                text,
                voice,
                normalize_volume: !!(normalizeVolume && normalizeVolume.checked),
                clean_audio: !!(cleanAudio && cleanAudio.checked),
                cleaning: readCleaningOptions(),
                ...readAdvancedParams(),
            }),
            signal: currentAbortController.signal,
        });
        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.detail || `Server error: ${response.status}`);
        }
        stopBtn.disabled = false;

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop();

            for (const line of lines) {
                if (!line.startsWith('data: ')) continue;
                const data = JSON.parse(line.slice(6));

                if (data.type === 'start') {
                    currentGenerationId = data.generation_id;
                    if (typeof data.seed !== 'undefined') showUsedSeed(data.seed);
                    startGenProgress(data.chunks);
                } else if (data.type === 'progress') {
                    // Flavor text carries the running commentary now; the bar
                    // itself advances on each decoded chunk below.
                } else if (data.type === 'chunk') {
                    if (typeof data.index === 'number') updateGenProgressFromChunk(data.index);
                    handleAudioChunk(data.data);
                } else if (data.type === 'complete') {
                    currentGenerationId = null;
                    isStreaming = false;
                    finishGenProgress('Fresh batch ready — unwrap it below! 🍬');
                    generateBtn.disabled = false;
                    stopBtn.disabled = true;
                } else if (data.type === 'error') {
                    currentGenerationId = null;
                    isStreaming = false;
                    hideGenProgress();
                    showError(data.message);
                    generateBtn.disabled = false;
                    stopBtn.disabled = true;
                }
            }
        }
    } catch (error) {
        if (error.name !== 'AbortError') {
            console.error('Generation error:', error);
            hideGenProgress();
            showError(error.message || 'Failed to start generation');
        }
        currentAbortController = null;
        currentGenerationId = null;
        isStreaming = false;
        generateBtn.disabled = false;
        stopBtn.disabled = true;
    }
});

// Stop generation
function stopGeneration() {
    if (currentAbortController) {
        // Request backend to stop this specific generation with timeout
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 5000);
        const stopUrl = currentGenerationId !== null
            ? `/stop?generation_id=${currentGenerationId}`
            : '/stop';
        fetch(stopUrl, { method: 'POST', signal: controller.signal })
            .catch(e => console.error('Failed to send stop request:', e))
            .finally(() => clearTimeout(timeoutId));
        currentAbortController.abort();
        currentAbortController = null;
        currentGenerationId = null;
        isStreaming = false;
        hideGenProgress();
        showProgress('Batch paused — your half-made candy is saved below 🍬');
        generateBtn.disabled = false;
        stopBtn.disabled = true;

        // Stop audio playback but keep buffers for replay
        if (isPlaying) {
            stopPlayback();
        }

        // Reset playback position to beginning for replay
        playbackOffset = 0;
        updateTimeDisplay();

        // Note: We don't clear audioBuffers or hide the player,
        // so the user can still replay what was generated
    }
}

// Stop button click handler
stopBtn.addEventListener('click', stopGeneration);

function showProgress(message) {
    progressDiv.textContent = message;
    progressDiv.classList.add('visible');
}

function showError(message) {
    errorDiv.textContent = message;
    errorDiv.classList.add('visible');
}

async function handleAudioChunk(base64Data) {
    // Add to queue
    chunkQueue.push(base64Data);

    // Process queue if not already processing
    if (!processingChunk) {
        await processChunkQueue();
    }
}

async function processChunkQueue() {
    if (chunkQueue.length === 0) {
        processingChunk = false;
        return;
    }

    processingChunk = true;
    const base64Data = chunkQueue.shift();

    console.log(`Processing chunk ${audioBuffers.length + 1}, ${chunkQueue.length} remaining in queue`);

    // Decode base64 to array buffer
    const binaryString = atob(base64Data);
    const bytes = new Uint8Array(binaryString.length);
    for (let i = 0; i < binaryString.length; i++) {
        bytes[i] = binaryString.charCodeAt(i);
    }

    // Decode audio data using Web Audio API
    let audioBuffer;
    try {
        audioBuffer = await audioContext.decodeAudioData(bytes.buffer);
    } catch (error) {
        console.error('Failed to decode audio chunk:', error);
        showError('Failed to decode audio chunk. The stream may be corrupted.');
        processingChunk = false;
        return;
    }
    audioBuffers.push(audioBuffer);

    // Update total duration
    totalDuration += audioBuffer.duration;

    console.log(`Decoded chunk ${audioBuffers.length}: ${audioBuffer.duration.toFixed(2)}s, total duration: ${totalDuration.toFixed(2)}s`);

    // Show audio player and update time display
    audioPlayerDiv.classList.add('visible');
    updateTimeDisplay();

    // Auto-start playback on first chunk
    if (audioBuffers.length === 1 && !isPlaying) {
        startPlayback(0);
    }

    // Process next chunk in queue
    await processChunkQueue();
}

function concatenateAudioBuffers(buffers) {
    if (buffers.length === 0) return null;
    if (buffers.length === 1) return buffers[0];

    const sampleRate = buffers[0].sampleRate;
    const numberOfChannels = buffers[0].numberOfChannels;

    // Calculate total length
    const totalLength = buffers.reduce((sum, buf) => sum + buf.length, 0);

    // Create new buffer
    const result = audioContext.createBuffer(numberOfChannels, totalLength, sampleRate);

    // Copy data from each buffer
    let offset = 0;
    for (const buffer of buffers) {
        for (let channel = 0; channel < numberOfChannels; channel++) {
            result.getChannelData(channel).set(buffer.getChannelData(channel), offset);
        }
        offset += buffer.length;
    }

    return result;
}

// Encode AudioBuffer as WAV (PCM 16-bit)
function audioBufferToWav(buffer) {
    const numChannels = buffer.numberOfChannels;
    const sampleRate = buffer.sampleRate;
    const bitsPerSample = 16;
    const bytesPerSample = bitsPerSample / 8;
    const blockAlign = numChannels * bytesPerSample;
    const numSamples = buffer.length;
    const dataSize = numSamples * blockAlign;
    const headerSize = 44;
    const totalSize = headerSize + dataSize;

    const arrayBuffer = new ArrayBuffer(totalSize);
    const view = new DataView(arrayBuffer);

    // WAV header
    writeString(view, 0, 'RIFF');
    view.setUint32(4, totalSize - 8, true);
    writeString(view, 8, 'WAVE');
    writeString(view, 12, 'fmt ');
    view.setUint32(16, 16, true);              // fmt chunk size
    view.setUint16(20, 1, true);               // PCM format
    view.setUint16(22, numChannels, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * blockAlign, true); // byte rate
    view.setUint16(32, blockAlign, true);
    view.setUint16(34, bitsPerSample, true);
    writeString(view, 36, 'data');
    view.setUint32(40, dataSize, true);

    // Interleave channels and convert Float32 to Int16
    let offset = headerSize;
    for (let i = 0; i < numSamples; i++) {
        for (let ch = 0; ch < numChannels; ch++) {
            const sample = buffer.getChannelData(ch)[i];
            const clamped = Math.max(-1, Math.min(1, sample));
            view.setInt16(offset, clamped < 0 ? clamped * 0x8000 : clamped * 0x7FFF, true);
            offset += 2;
        }
    }

    return new Blob([arrayBuffer], { type: 'audio/wav' });
}

function writeString(view, offset, string) {
    for (let i = 0; i < string.length; i++) {
        view.setUint8(offset + i, string.charCodeAt(i));
    }
}

// Encode AudioBuffer as MP3 using lamejs
function audioBufferToMp3(buffer) {
    const numChannels = buffer.numberOfChannels;
    const sampleRate = buffer.sampleRate;
    const kbps = 128;

    // Convert Float32 to Int16 arrays
    const left = floatTo16BitPCM(buffer.getChannelData(0));
    const right = numChannels > 1 ? floatTo16BitPCM(buffer.getChannelData(1)) : null;

    const encoder = new lamejs.Mp3Encoder(numChannels, sampleRate, kbps);
    const mp3Data = [];
    const blockSize = 1152;

    for (let i = 0; i < left.length; i += blockSize) {
        const leftChunk = left.subarray(i, i + blockSize);
        let mp3buf;
        if (numChannels === 1) {
            mp3buf = encoder.encodeBuffer(leftChunk);
        } else {
            const rightChunk = right.subarray(i, i + blockSize);
            mp3buf = encoder.encodeBuffer(leftChunk, rightChunk);
        }
        if (mp3buf.length > 0) {
            mp3Data.push(mp3buf);
        }
    }

    const end = encoder.flush();
    if (end.length > 0) {
        mp3Data.push(end);
    }

    return new Blob(mp3Data, { type: 'audio/mpeg' });
}

function floatTo16BitPCM(float32Array) {
    const int16 = new Int16Array(float32Array.length);
    for (let i = 0; i < float32Array.length; i++) {
        const s = Math.max(-1, Math.min(1, float32Array[i]));
        int16[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
    }
    return int16;
}

// Trigger file download from a Blob
function downloadBlob(blob, filename) {
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

// Generate filename: longecho-{voice}-{YYYYMMDD-HHmmss}.{ext}
function generateFilename(ext) {
    const voice = selectedVoice || 'audio';
    const now = new Date();
    const pad = (n) => String(n).padStart(2, '0');
    const timestamp = `${now.getFullYear()}${pad(now.getMonth() + 1)}${pad(now.getDate())}-${pad(now.getHours())}${pad(now.getMinutes())}${pad(now.getSeconds())}`;
    return `longecho-${voice}-${timestamp}.${ext}`;
}

// Download button refs
const downloadBtn = document.getElementById('downloadBtn');
const downloadBtnText = document.getElementById('downloadBtnText');
const downloadFormat = document.getElementById('downloadFormat');

downloadBtn.addEventListener('click', async () => {
    if (audioBuffers.length === 0) return;

    const format = downloadFormat.value;
    const concatenated = concatenateAudioBuffers(audioBuffers);
    if (!concatenated) return;

    if (format === 'wav') {
        const blob = audioBufferToWav(concatenated);
        downloadBlob(blob, generateFilename('wav'));
    } else if (format === 'mp3') {
        downloadBtn.disabled = true;
        downloadBtnText.textContent = 'Encoding...';
        // Yield to UI so the button text updates
        await new Promise(r => setTimeout(r, 0));
        try {
            const blob = audioBufferToMp3(concatenated);
            downloadBlob(blob, generateFilename('mp3'));
        } finally {
            downloadBtn.disabled = false;
            downloadBtnText.textContent = 'Download';
        }
    }
});

// Load voices on page load
loadVoices();
loadCleaningRules();
subscribeToVoiceEvents();

// Fill the API-instructions panel with this page's actual base URL
(() => {
    const origin = window.location.origin;
    const base = document.getElementById('apiBaseUrl');
    const speech = document.getElementById('apiSpeechUrl');
    if (base) base.textContent = origin;
    if (speech) speech.textContent = origin + '/v1/audio/speech';
    document.querySelectorAll('.api-instructions .u').forEach((el) => { el.textContent = origin; });
})();
