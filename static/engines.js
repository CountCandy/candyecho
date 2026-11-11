// Engine picker: Echo (built in) plus the optional Drama 3 and Breeze engines.
//
// Loaded after app.js. app.js calls into window.CandyEngines when a generation
// starts and announces voice changes with a "candy:voice" event; everything
// else here is self-contained.
(function () {
    'use strict';

    const $ = (id) => document.getElementById(id);
    const engineSelect = $('engineSelect');
    if (!engineSelect) return;
    const textArea = $('text');
    const showToast = (message, type) => {
        if (typeof window.showToast === 'function') window.showToast(message, type);
    };

    const engineBadge = $('engineBadge');
    const engineCard = $('engineCard');
    const engineBlurb = $('engineBlurb');
    const engineModelState = $('engineModelState');
    const engineServerState = $('engineServerState');
    const engineDownloadBtn = $('engineDownloadBtn');
    const engineStartBtn = $('engineStartBtn');
    const engineStopBtn = $('engineStopBtn');
    const engineBar = $('engineBar');
    const engineBarFill = $('engineBarFill');
    const engineError = $('engineError');
    const engineLicense = $('engineLicense');
    const engineRefresh = $('engineRefresh');
    const echoCard = $('echoCard');
    const echoState = $('echoState');
    const echoToggle = $('echoToggle');
    const extraRefsBox = $('extraRefsBox');
    const extraRefsList = $('extraRefsList');
    const extraRefsNote = $('extraRefsNote');
    const advTopK = $('advTopK');

    const transcriptBox = $('transcriptBox');
    const voiceTranscript = $('voiceTranscript');
    const transcriptAuto = $('transcriptAuto');
    const transcriptSave = $('transcriptSave');
    const transcriptState = $('transcriptState');
    const transcriptSeconds = $('transcriptSeconds');

    const enhanceRow = $('enhanceRow');
    const enhanceBtn = $('enhanceBtn');
    const enhanceUndo = $('enhanceUndo');
    const enhancerDownload = $('enhancerDownload');
    const enhanceState = $('enhanceState');

    const advEchoOnly = $('advEchoOnly');
    const advDrama = $('advDrama');
    const advBreeze = $('advBreeze');
    const advTemperature = $('advTemperature');
    const advTopP = $('advTopP');
    const advCfgScale = $('advCfgScale');
    const advInstruction = $('advInstruction');
    const takesNoteEcho = $('advTakesNoteEcho');
    const takesNoteRemote = $('advTakesNoteRemote');

    const ENGINE_KEY = 'candyecho.engine';
    const EXTRA_REFS_KEY = 'candyecho.extraRefs';
    // Status is only polled while something is in progress (a download or an
    // engine loading). Otherwise it is checked on demand: the "Check status"
    // button, switching engine, and after a generation.
    const BUSY_POLL_MS = 2000;

    let engines = {};            // key -> status object from GET /engines
    let enhancer = null;         // enhancer section of GET /engines
    let selected = localStorage.getItem(ENGINE_KEY) || 'echo';
    let pollTimer = null;
    let wasDownloading = {};     // model key -> true while a download was seen running
    let transcriptVoice = null;  // voice whose transcript is in the box
    let transcriptSaved = '';    // last text known to be on the server
    let textBeforeEnhance = null;

    const current = () => engines[selected] || null;
    const isRemote = () => selected !== 'echo';

    function gb(bytes) {
        return (bytes / 1073741824).toFixed(bytes >= 10737418240 ? 0 : 1) + ' GB';
    }

    async function api(url, options) {
        const resp = await fetch(url, options);
        const data = await resp.json().catch(() => ({}));
        if (!resp.ok) throw new Error(data.detail || `Server error: ${resp.status}`);
        return data;
    }

    // ---- status -----------------------------------------------------------

    async function refresh() {
        try {
            const data = await api('/engines');
            const next = {};
            for (const e of data.engines || []) next[e.key] = e;
            engines = next;
            enhancer = data.enhancer || null;
            if (transcriptSeconds && data.reference_seconds) {
                transcriptSeconds.textContent = Math.round(data.reference_seconds);
            }
            if (!engines[selected]) selected = 'echo';
            noticeFinishedDownloads();
            render();
        } catch (e) {
            console.error('Failed to load engines:', e);
        }
        schedulePoll();
    }

    function busy() {
        const models = Object.values(engines).map((e) => e.model).filter(Boolean);
        if (enhancer && enhancer.model) models.push(enhancer.model);
        if (models.some((m) => m.state === 'downloading')) return true;
        return Object.values(engines).some((e) => e.server && e.server.state === 'starting');
    }

    function schedulePoll() {
        clearTimeout(pollTimer);
        if (busy()) pollTimer = setTimeout(refresh, BUSY_POLL_MS);
    }

    // When a download that was running in this tab completes, bring the engine
    // up straight away so it is usable without any further clicks or a restart.
    function noticeFinishedDownloads() {
        for (const e of Object.values(engines)) {
            if (!e.model) continue;
            const key = e.model.key;
            if (e.model.state === 'downloading') {
                wasDownloading[key] = true;
            } else if (wasDownloading[key]) {
                wasDownloading[key] = false;
                if (e.model.state === 'ready') {
                    showToast(`${e.label} downloaded`, 'success');
                    if (e.server.state === 'stopped' && e.server.can_start) startEngine(e.key);
                } else if (e.model.state === 'error') {
                    showToast(`${e.label} download failed`, 'error');
                }
            }
        }
        if (enhancer && enhancer.model) {
            const m = enhancer.model;
            if (m.state === 'downloading') {
                wasDownloading[m.key] = true;
            } else if (wasDownloading[m.key]) {
                wasDownloading[m.key] = false;
                showToast(m.state === 'ready' ? 'Tag model downloaded' : 'Tag model download failed',
                    m.state === 'ready' ? 'success' : 'error');
            }
        }
    }

    // ---- rendering --------------------------------------------------------

    function setBadge(text, kind) {
        engineBadge.textContent = text;
        engineBadge.dataset.kind = kind || '';
    }

    function render() {
        engineSelect.value = selected;
        const e = current();
        const remote = isRemote() && e;

        engineCard.hidden = !remote;
        renderEcho();
        renderExtraRefs();
        transcriptBox.hidden = !remote;
        enhanceRow.hidden = !(remote && e.supports_tags);
        if (advEchoOnly) advEchoOnly.hidden = !!remote;
        if (advDrama) advDrama.hidden = selected !== 'drama3';
        if (advBreeze) advBreeze.hidden = selected !== 'breeze';
        if (takesNoteEcho) takesNoteEcho.hidden = !!remote;
        if (takesNoteRemote) takesNoteRemote.hidden = !remote;
        for (const [id, key] of [['deliveryEcho', 'echo'], ['deliveryDrama', 'drama3'], ['deliveryBreeze', 'breeze']]) {
            const el = $(id);
            if (el) el.hidden = selected !== key;
        }

        if (!remote) {
            setBadge('ready', 'ready');
            return;
        }

        engineBlurb.textContent = e.blurb;
        const model = e.model;
        const server = e.server;

        // 1 · model files
        engineBar.hidden = true;
        engineDownloadBtn.hidden = true;
        engineDownloadBtn.disabled = false;
        if (model.state === 'ready') {
            engineModelState.textContent = `✓ downloaded (${gb(model.bytes_total)})`;
        } else if (model.state === 'downloading') {
            const pct = model.bytes_total ? Math.floor((model.bytes_done / model.bytes_total) * 100) : 0;
            engineModelState.textContent = model.bytes_total
                ? `downloading… ${gb(model.bytes_done)} of ${gb(model.bytes_total)} (${pct}%)`
                : 'starting download…';
            engineBar.hidden = false;
            engineBar.setAttribute('aria-valuenow', String(pct));
            engineBarFill.style.width = pct + '%';
        } else if (model.state === 'partial' || model.state === 'error') {
            engineModelState.textContent = model.state === 'error'
                ? 'download stopped'
                : `incomplete (${gb(model.bytes_done)} on disk)`;
            engineDownloadBtn.textContent = 'Resume download';
            engineDownloadBtn.hidden = false;
        } else {
            engineModelState.textContent = `not downloaded (about ${Math.round(model.approx_gb)} GB)`;
            engineDownloadBtn.textContent = 'Download';
            engineDownloadBtn.hidden = false;
        }

        // 2 · engine server
        engineStartBtn.hidden = true;
        engineStopBtn.hidden = true;
        if (server.state === 'ready') {
            engineServerState.textContent = server.managed ? '✓ running' : '✓ running (started outside CandyEcho)';
            engineStopBtn.hidden = !server.managed;
        } else if (server.state === 'starting') {
            engineServerState.textContent = `loading the model… ${server.seconds_starting || 0}s`;
            engineStopBtn.hidden = false;
        } else {
            engineServerState.textContent = server.state === 'failed' ? 'failed to start' : 'stopped';
            const startable = server.can_start && model.state === 'ready';
            engineStartBtn.hidden = !startable;
            engineStartBtn.textContent = server.state === 'failed' ? 'Try again' : 'Start';
        }

        const problem = model.error || server.error;
        engineError.hidden = !problem;
        engineError.textContent = problem || '';

        engineLicense.textContent = '';
        engineLicense.append(`Weights: ${model.license}. `);
        const link = document.createElement('a');
        link.href = model.license_url;
        link.target = '_blank';
        link.rel = 'noopener';
        link.textContent = 'Model page';
        engineLicense.append(link, ` · Stored in ${model.path}`);

        if (server.state === 'ready') setBadge('ready', 'ready');
        else if (server.state === 'starting') setBadge('loading…', 'busy');
        else if (model.state === 'downloading') setBadge('downloading…', 'busy');
        else if (model.state === 'ready') setBadge('starts when you generate', 'idle');
        else setBadge('needs download', 'warn');

        renderEnhancer();
    }

    function renderEcho() {
        const echo = engines.echo;
        echoCard.hidden = isRemote() || !echo;
        if (!echo) return;
        const loaded = echo.loaded !== false;
        echoState.textContent = loaded ? 'loaded on the GPU' : 'unloaded (VRAM freed)';
        echoToggle.textContent = loaded ? 'Unload' : 'Load now';
    }

    // ---- extra reference clips (engines that accept several) ---------------

    function savedExtras() {
        try {
            return JSON.parse(localStorage.getItem(EXTRA_REFS_KEY) || '{}') || {};
        } catch (e) {
            return {};
        }
    }

    function extrasFor(voice) {
        const list = savedExtras()[voice];
        return Array.isArray(list) ? list : [];
    }

    function setExtrasFor(voice, list) {
        const all = savedExtras();
        if (list.length) all[voice] = list;
        else delete all[voice];
        localStorage.setItem(EXTRA_REFS_KEY, JSON.stringify(all));
    }

    let extraRefsVoice = null;      // voice the checklist was built for
    let extraRefsNames = [];        // every voice in the library

    function renderExtraRefs() {
        const e = current();
        const limit = (isRemote() && e) ? (e.max_extra_references || 0) : 0;
        const voice = activeVoice();
        extraRefsBox.hidden = !(limit > 0 && voice);
        if (extraRefsBox.hidden) return;

        const chosen = extrasFor(voice).filter((n) => extraRefsNames.includes(n) && n !== voice);
        const others = extraRefsNames.filter((n) => n !== voice);
        extraRefsList.textContent = '';
        if (!others.length) {
            extraRefsNote.textContent = 'Add more recordings of this speaker to the voice library to use them here.';
            return;
        }
        for (const name of others) {
            const label = document.createElement('label');
            label.className = 'extra-ref';
            const box = document.createElement('input');
            box.type = 'checkbox';
            box.checked = chosen.includes(name);
            box.disabled = !box.checked && chosen.length >= limit;
            box.addEventListener('change', () => {
                const now = extrasFor(voice).filter((n) => n !== name);
                if (box.checked) now.push(name);
                setExtrasFor(voice, now.slice(0, limit));
                renderExtraRefs();
            });
            label.append(box, ' ' + name);
            extraRefsList.appendChild(label);
        }
        extraRefsNote.textContent = chosen.length
            ? `${chosen.length} extra clip${chosen.length === 1 ? '' : 's'} selected (up to ${limit}).`
            : `None selected. Up to ${limit}.`;
        extraRefsVoice = voice;
    }

    async function loadVoiceNames() {
        try {
            const data = await api('/voices');
            extraRefsNames = (data.voices || []).map((v) => (typeof v === 'string' ? v : v.name));
        } catch (e) {
            extraRefsNames = [];
        }
        renderExtraRefs();
    }

    function renderEnhancer() {
        if (!enhancer || enhanceRow.hidden) return;
        const m = enhancer.model;
        enhancerDownload.hidden = true;
        enhanceBtn.disabled = false;
        if (!m) return;                       // external endpoint: nothing to download
        if (m.state === 'ready') return;
        enhanceBtn.disabled = true;
        if (m.state === 'downloading') {
            const pct = m.bytes_total ? Math.floor((m.bytes_done / m.bytes_total) * 100) : 0;
            enhanceState.textContent = `Downloading tag model… ${pct}%`;
        } else {
            enhancerDownload.hidden = false;
            enhancerDownload.textContent = m.state === 'missing'
                ? `⬇ Download tag model (about ${Math.round(m.approx_gb)} GB)`
                : '⬇ Resume tag model download';
            enhanceState.textContent = m.error || 'Needs a one-time download.';
        }
    }

    // ---- actions ----------------------------------------------------------

    async function downloadModel(key) {
        try {
            await api(`/models/${encodeURIComponent(key)}/download`, { method: 'POST' });
            wasDownloading[key] = true;
        } catch (e) {
            showToast(e.message, 'error');
        }
        refresh();
    }

    async function startEngine(key) {
        try {
            await api(`/engines/${encodeURIComponent(key)}/start`, { method: 'POST' });
        } catch (e) {
            showToast(e.message, 'error');
        }
        refresh();
    }

    async function stopEngine(key) {
        try {
            await api(`/engines/${encodeURIComponent(key)}/stop`, { method: 'POST' });
        } catch (e) {
            showToast(e.message, 'error');
        }
        refresh();
    }

    engineSelect.addEventListener('change', () => {
        selected = engineSelect.value;
        localStorage.setItem(ENGINE_KEY, selected);
        render();
        refresh();
        loadTranscript(true);
        loadVoiceNames();
    });
    engineRefresh.addEventListener('click', async () => {
        engineRefresh.disabled = true;
        await refresh();
        engineRefresh.disabled = false;
    });
    echoToggle.addEventListener('click', async () => {
        const loaded = !engines.echo || engines.echo.loaded !== false;
        echoToggle.disabled = true;
        echoState.textContent = loaded ? 'unloading…' : 'loading…';
        try {
            await api(loaded ? '/engines/echo/unload' : '/engines/echo/load', { method: 'POST' });
        } catch (e) {
            showToast(e.message, 'error');
        }
        echoToggle.disabled = false;
        refresh();
    });
    engineDownloadBtn.addEventListener('click', () => {
        const e = current();
        if (!e) return;
        engineDownloadBtn.disabled = true;
        downloadModel(e.model.key);
    });
    engineStartBtn.addEventListener('click', () => { if (current()) startEngine(selected); });
    engineStopBtn.addEventListener('click', () => { if (current()) stopEngine(selected); });
    enhancerDownload.addEventListener('click', () => {
        enhancerDownload.hidden = true;
        downloadModel('enhancer');
    });

    // ---- reference transcript ---------------------------------------------

    function activeVoice() {
        return document.body.dataset.voice || null;
    }

    async function loadTranscript(force) {
        const voice = activeVoice();
        if (!isRemote() || !voice) return;
        if (!force && voice === transcriptVoice) return;
        transcriptVoice = voice;
        voiceTranscript.value = '';
        transcriptSaved = '';
        transcriptState.textContent = '';
        try {
            const data = await api(`/voices/${encodeURIComponent(voice)}/transcript`);
            if (voice !== transcriptVoice) return;      // selection moved on
            transcriptSaved = data.transcript || '';
            voiceTranscript.value = transcriptSaved;
            transcriptState.textContent = transcriptSaved ? '' : 'Not transcribed yet.';
        } catch (e) {
            transcriptState.textContent = e.message;
        }
    }

    async function saveTranscript(quiet) {
        const voice = transcriptVoice;
        const text = voiceTranscript.value.trim();
        if (!voice || text === transcriptSaved) return;
        try {
            const data = await api(`/voices/${encodeURIComponent(voice)}/transcript`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text }),
            });
            transcriptSaved = data.transcript || '';
            if (!quiet) transcriptState.textContent = 'Saved.';
        } catch (e) {
            transcriptState.textContent = e.message;
        }
    }

    transcriptSave.addEventListener('click', () => saveTranscript(false));
    voiceTranscript.addEventListener('change', () => saveTranscript(true));
    transcriptAuto.addEventListener('click', async () => {
        const voice = transcriptVoice || activeVoice();
        if (!voice) return;
        transcriptAuto.disabled = true;
        transcriptState.textContent = 'Listening… (the first run downloads a speech-recognition model)';
        try {
            const data = await api(`/voices/${encodeURIComponent(voice)}/transcribe`, { method: 'POST' });
            if (voice === transcriptVoice) {
                transcriptSaved = data.transcript || '';
                voiceTranscript.value = transcriptSaved;
                transcriptState.textContent = 'Transcribed. Check it against the clip and fix any mistakes.';
            }
        } catch (e) {
            transcriptState.textContent = e.message;
        }
        transcriptAuto.disabled = false;
    });
    document.addEventListener('candy:voice', () => {
        loadTranscript(false);
        if (isRemote() && activeVoice() !== extraRefsVoice) loadVoiceNames();
    });

    // ---- tag enhancer -----------------------------------------------------

    enhanceBtn.addEventListener('click', async () => {
        const text = textArea.value;
        if (!text.trim()) {
            enhanceState.textContent = 'Write something first.';
            return;
        }
        enhanceBtn.disabled = true;
        enhanceState.textContent = 'Writing directions… (the first run loads the language model)';
        try {
            const data = await api('/enhance-text', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text }),
            });
            if (data.enhanced > 0) {
                textBeforeEnhance = text;
                textArea.value = data.text;
                enhanceUndo.hidden = false;
            }
            const parts = [];
            if (data.enhanced > 0) parts.push(`Added ${data.tags_added} tag${data.tags_added === 1 ? '' : 's'}.`);
            if (data.unchanged > 0) {
                parts.push(`${data.unchanged} passage${data.unchanged === 1 ? ' was' : 's were'} left as written.`);
            }
            enhanceState.textContent = parts.join(' ') || 'Nothing to tag.';
            enhanceState.title = (data.notes || []).join('\n');
        } catch (e) {
            enhanceState.textContent = e.message;
            refresh();
        }
        enhanceBtn.disabled = false;
    });
    enhanceUndo.addEventListener('click', () => {
        if (textBeforeEnhance !== null) textArea.value = textBeforeEnhance;
        textBeforeEnhance = null;
        enhanceUndo.hidden = true;
        enhanceState.textContent = 'Tags removed.';
    });

    // ---- per-engine controls ----------------------------------------------

    function bindOutput(input, outputId, digits) {
        const out = $(outputId);
        if (!input || !out) return;
        const sync = () => { out.textContent = parseFloat(input.value).toFixed(digits); };
        input.addEventListener('input', sync);
        sync();
    }
    bindOutput(advTemperature, 'advTemperatureVal', 2);
    bindOutput(advTopP, 'advTopPVal', 2);
    bindOutput(advCfgScale, 'advCfgScaleVal', 1);
    bindOutput(advTopK, 'advTopKVal', 0);

    // ---- hooks for app.js -------------------------------------------------

    window.CandyEngines = {
        // A reason generation cannot start, or null when it can.
        blockedReason() {
            if (!isRemote()) return null;
            const e = current();
            if (!e) return 'That engine is not available.';
            if (e.server.state === 'ready') return null;
            if (e.model.state === 'downloading') return `${e.label} is still downloading.`;
            if (e.model.state !== 'ready') {
                return `${e.label} is not downloaded yet. Press Download in the Engine section.`;
            }
            return null;    // downloaded: the server is started on demand
        },
        // Flush an edited transcript so the run uses what is on screen.
        async beforeGenerate() {
            if (isRemote()) await saveTranscript(true);
        },
        // A run may have transcribed the reference or started the server.
        afterGenerate() {
            if (isRemote()) loadTranscript(true);
            refresh();
        },
        payload() {
            if (!isRemote()) return { engine: 'echo' };
            const p = { engine: selected };
            if (selected === 'drama3') {
                if (advTemperature) p.temperature = parseFloat(advTemperature.value);
                if (advTopP) p.top_p = parseFloat(advTopP.value);
                if (advTopK) p.top_k = parseInt(advTopK.value, 10);
            } else if (selected === 'breeze') {
                const instruction = advInstruction ? advInstruction.value.trim() : '';
                if (instruction) p.instruction = instruction;
                // 1.0 is "leave it to Breeze" (which uses 4 with an instruction).
                if (advCfgScale && parseFloat(advCfgScale.value) > 1.0) {
                    p.cfg_scale = parseFloat(advCfgScale.value);
                }
            }
            const e = current();
            const voice = activeVoice();
            if (e && e.max_extra_references > 0 && voice) {
                const extras = extrasFor(voice)
                    .filter((n) => n !== voice && extraRefsNames.includes(n))
                    .slice(0, e.max_extra_references);
                if (extras.length) p.extra_voices = extras;
            }
            return p;
        },
        refresh,
    };

    render();
    refresh();
    loadVoiceNames();
})();
