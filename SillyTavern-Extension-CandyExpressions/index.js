/*
 * CandyExpressions — a SillyTavern UI extension.
 *
 * Adds sticky, switchable expression "variants" (outfits / forms / art styles)
 * on top of SillyTavern's built-in Character Expressions "costume" system.
 *
 * A variant is just a sub-folder of a character's sprite folder, e.g.
 *   <SpriteFolder>/<Character>/swimsuit/joy.png
 * which SillyTavern already knows how to render via the `/costume` slash command
 * (it sets an `expressionOverrides` path for the character). This extension is a
 * thin *controller* over that mechanism and adds:
 *
 *   1. A manual variant picker (wand menu + a dropdown in the settings drawer)
 *      so you can switch outfit/form directly from chat.
 *   2. Sticky persistence — the chosen variant is remembered per-chat (or as a
 *      per-character global default) and re-applied automatically, so it survives
 *      emotion changes, swipes and reloads until you explicitly change it.
 *   3. An opt-in, deliberately conservative auto-switch classifier that only
 *      changes the variant when the narration *explicitly* describes such a
 *      change and it maps onto a known variant. Otherwise it stays put.
 *
 * The whole extension talks to SillyTavern through the public
 * `SillyTavern.getContext()` surface — no fragile relative imports — so it is
 * resilient across app versions.
 */

'use strict';

(function () {
    const MODULE_NAME = 'candyexpressions';
    const LOG_PREFIX = '[CandyExpressions]';

    /** Default, deep-clonable settings. */
    const DEFAULT_SETTINGS = {
        enabled: true,
        // Sticky state is stored per-chat when true, otherwise as a global
        // per-character default.
        perChat: true,
        // Opt-in automatic switching on explicit narrative cues.
        autoSwitch: false,
        // 'strict'  -> requires a change-verb AND a variant name to even ask the model.
        // 'balanced'-> either a change-verb OR a variant name is enough to ask.
        autoStrictness: 'strict',
        // Per-character declared variants: { [avatarKey]: string[] }.
        variants: {},
        // Per-character global default variant: { [avatarKey]: string }.
        defaults: {},
        // Optional custom classifier prompt (supports {{variants}}, {{current}}, {{text}}).
        classifierPrompt: '',
        debug: false,
    };

    // Verbs / phrases that hint an on-screen wardrobe or form change.
    const CHANGE_CUES = [
        'change', 'changed', 'changes', 'changing', 'changes into', 'changed into',
        'put on', 'puts on', 'putting on', 'slip into', 'slips into', 'slipped into',
        'pull on', 'pulls on', 'throw on', 'throws on', 'wear', 'wears', 'wearing',
        'dressed in', 'gets dressed', 'get dressed', 'dresses', 'undress', 'undresses',
        'take off', 'takes off', 'took off', 'strip', 'strips', 'stripped',
        'remove', 'removes', 'removed', 'transform', 'transforms', 'transformed',
        'transforming', 'morph', 'morphs', 'shift into', 'shifts into', 'shifted into',
        'now in', 'now wearing', 'emerges in', 'steps out in', 'appears in',
        'reappears in', 'outfit', 'costume', 'uniform', 'clothes', 'clothing', 'naked',
        'nude',
    ];

    // ---------------------------------------------------------------------------
    //  Small utilities
    // ---------------------------------------------------------------------------

    function ctx() {
        return globalThis.SillyTavern.getContext();
    }

    function log(...args) {
        if (getSettings().debug) console.log(LOG_PREFIX, ...args);
    }

    function warn(...args) {
        console.warn(LOG_PREFIX, ...args);
    }

    function toast(kind, message) {
        try {
            globalThis.toastr?.[kind]?.(message, 'CandyExpressions');
        } catch { /* toastr may not be ready */ }
    }

    function deepClone(value) {
        try {
            return structuredClone(value);
        } catch {
            return JSON.parse(JSON.stringify(value));
        }
    }

    function escapeRegExp(value) {
        return String(value).replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    }

    function stripExtension(value) {
        return String(value || '').replace(/\.[^/.]+$/, '');
    }

    /** Turn arbitrary user input into a safe single-segment folder name. */
    function sanitizeVariantName(value) {
        return String(value || '')
            .trim()
            .replace(/[\\/]+/g, '')      // no path separators — variants are single sub-folders
            .replace(/\s+/g, ' ')
            .trim();
    }

    function htmlEscape(value) {
        return String(value ?? '')
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#39;');
    }

    // ---------------------------------------------------------------------------
    //  Settings + state
    // ---------------------------------------------------------------------------

    function getSettings() {
        const c = ctx();
        const store = c.extensionSettings;
        if (!store[MODULE_NAME] || typeof store[MODULE_NAME] !== 'object') {
            store[MODULE_NAME] = deepClone(DEFAULT_SETTINGS);
        }
        const settings = store[MODULE_NAME];
        for (const [key, value] of Object.entries(DEFAULT_SETTINGS)) {
            if (settings[key] === undefined) {
                settings[key] = deepClone(value);
            }
        }
        return settings;
    }

    function saveSettings() {
        ctx().saveSettingsDebounced();
    }

    /** Live per-chat state object: { active: { [avatarKey]: variantName } }. */
    function getChatState() {
        const metadata = ctx().chatMetadata;
        if (!metadata) return { active: {} };
        if (!metadata[MODULE_NAME] || typeof metadata[MODULE_NAME] !== 'object') {
            metadata[MODULE_NAME] = { active: {} };
        }
        if (!metadata[MODULE_NAME].active || typeof metadata[MODULE_NAME].active !== 'object') {
            metadata[MODULE_NAME].active = {};
        }
        return metadata[MODULE_NAME];
    }

    function saveChatState() {
        try {
            ctx().saveMetadataDebounced();
        } catch (err) {
            warn('failed to persist chat metadata', err);
        }
    }

    function getVariants(avatarKey) {
        const list = getSettings().variants[avatarKey];
        return Array.isArray(list) ? list.slice() : [];
    }

    function setVariants(avatarKey, list) {
        const cleaned = [];
        for (const raw of list) {
            const name = sanitizeVariantName(raw);
            if (name && !cleaned.includes(name)) cleaned.push(name);
        }
        getSettings().variants[avatarKey] = cleaned;
        saveSettings();
        return cleaned;
    }

    /**
     * Resolve the variant that *should* be active for a character right now.
     * Per-chat state wins (including an explicit "" meaning "default"), then the
     * per-character global default, otherwise "" (base sprite folder).
     */
    function resolveVariant(avatarKey) {
        const settings = getSettings();
        if (settings.perChat) {
            const active = ctx().chatMetadata?.[MODULE_NAME]?.active;
            if (active && Object.prototype.hasOwnProperty.call(active, avatarKey)) {
                return active[avatarKey] || '';
            }
        }
        return settings.defaults[avatarKey] || '';
    }

    // ---------------------------------------------------------------------------
    //  Character resolution
    // ---------------------------------------------------------------------------

    /**
     * The character the picker should currently act on.
     * In solo chats this is the selected character; in groups it is the author of
     * the most recent character message.
     * @returns {{ name: string, avatarKey: string } | null}
     */
    function getActiveCharacter() {
        const c = ctx();

        if (c.groupId) {
            const chat = Array.isArray(c.chat) ? c.chat : [];
            for (let i = chat.length - 1; i >= 0; i--) {
                const message = chat[i];
                if (!message || message.is_user || message.is_system) continue;
                const name = message.name;
                const character = (c.characters || []).find(x => x && x.name === name);
                if (character) return { name: character.name, avatarKey: stripExtension(character.avatar) };
                if (name) return { name, avatarKey: name };
                break;
            }
            return null;
        }

        const id = c.characterId;
        if (id === undefined || id === null) return null;
        const character = (c.characters || [])[id];
        if (!character) return null;
        return { name: character.name, avatarKey: stripExtension(character.avatar) };
    }

    function lastCharacterMessageText() {
        const chat = Array.isArray(ctx().chat) ? ctx().chat : [];
        for (let i = chat.length - 1; i >= 0; i--) {
            const message = chat[i];
            if (message && !message.is_user && !message.is_system) return message.mes || '';
        }
        return '';
    }

    // ---------------------------------------------------------------------------
    //  SillyTavern integration
    // ---------------------------------------------------------------------------

    /**
     * Ask SillyTavern whether a given variant sub-folder actually has sprites.
     * There is no "list sub-folders" endpoint, so validation is done by probing.
     */
    async function probeVariant(characterName, variant) {
        if (!characterName || !variant) return false;
        const folder = `${characterName}/${variant}`;
        try {
            const response = await fetch(`/api/sprites/get?name=${encodeURIComponent(folder)}`, {
                headers: ctx().getRequestHeaders?.() ?? {},
            });
            if (!response.ok) return false;
            const sprites = await response.json();
            return Array.isArray(sprites) && sprites.length > 0;
        } catch (err) {
            warn('probeVariant failed', folder, err);
            return false;
        }
    }

    /** Filter a character's declared variants down to those that actually exist. */
    async function getValidVariants(character) {
        const declared = getVariants(character.avatarKey);
        const valid = [];
        await Promise.all(declared.map(async (variant) => {
            if (await probeVariant(character.name, variant)) valid.push(variant);
        }));
        // Preserve the user's declared order.
        return declared.filter(v => valid.includes(v));
    }

    /**
     * Apply a variant by driving SillyTavern's public `/costume` slash command.
     * An empty variant resets the character to their base sprite folder.
     */
    async function applyCostume(characterName, variant) {
        const c = ctx();
        const nameArg = characterName ? `name="${String(characterName).replace(/"/g, '')}" ` : '';
        const subFolder = variant ? `/${variant}` : '';
        const command = `/costume ${nameArg}${subFolder}`.trim();
        try {
            await c.executeSlashCommandsWithOptions(command, {
                handleExecutionErrors: true,
            });
            log('applied costume', command);
            return true;
        } catch (err) {
            warn('costume command failed; is the Character Expressions extension enabled?', err);
            return false;
        }
    }

    /**
     * Persist a variant choice for a character and apply it immediately.
     * @param {{name:string, avatarKey:string}} character
     * @param {string} variant "" clears back to the base folder.
     * @param {{ persist?: boolean, asDefault?: boolean }} [options]
     */
    async function setVariant(character, variant, { persist = true, asDefault = false } = {}) {
        const settings = getSettings();
        const value = variant || '';

        if (persist) {
            if (asDefault || !settings.perChat) {
                settings.defaults[character.avatarKey] = value;
                saveSettings();
            }
            if (settings.perChat && !asDefault) {
                getChatState().active[character.avatarKey] = value;
                saveChatState();
            }
        }

        await applyCostume(character.name, value);
        refreshUi();
    }

    /**
     * Re-apply whatever variant should be active for the current character.
     * Only acts when a non-empty variant is stored, so it never fights a costume
     * the user set through vanilla SillyTavern.
     */
    async function reconcile(reason) {
        refreshUi();
        const settings = getSettings();
        if (!settings.enabled) return;
        const character = getActiveCharacter();
        if (!character) return;
        const variant = resolveVariant(character.avatarKey);
        if (variant) {
            log('reconcile', reason, character.name, '->', variant);
            await applyCostume(character.name, variant);
        }
    }

    // ---------------------------------------------------------------------------
    //  Auto-switch classifier
    // ---------------------------------------------------------------------------

    function hasChangeCue(text, variants, settings) {
        const lower = text.toLowerCase();
        const nameHit = variants.some(v => new RegExp(`\\b${escapeRegExp(v.toLowerCase())}\\b`).test(lower));
        const verbHit = CHANGE_CUES.some(cue => lower.includes(cue));
        if (settings.autoStrictness === 'balanced') return nameHit || verbHit;
        return nameHit && verbHit;
    }

    function buildClassifierPrompt(variants, text, current) {
        const list = variants.join(', ');
        return [
            'You are a strict wardrobe/appearance-change detector for a roleplay scene.',
            'You are given the latest in-character message. Decide whether the character has JUST, explicitly, within this message, changed into one of the known appearance variants.',
            `Known variants: ${list}.`,
            `Current variant: ${current || 'default'}.`,
            'Rules:',
            '- Only pick a variant if the text explicitly narrates changing into it, now wearing it, or transforming into it.',
            '- A mere mention, memory, plan, wish, or question is NOT a change. When unsure, answer NONE.',
            '- Never pick the current variant.',
            'Message:',
            '"""',
            text,
            '"""',
            `Reply with EXACTLY one word: one of [${list}] or NONE.`,
        ].join('\n');
    }

    function renderCustomPrompt(template, variants, text, current) {
        return String(template)
            .replaceAll('{{variants}}', variants.join(', '))
            .replaceAll('{{current}}', current || 'default')
            .replaceAll('{{text}}', text);
    }

    /** Parse the model reply into a single unambiguous variant, or null. */
    function parseClassifierReply(reply, variants) {
        if (!reply) return null;
        const lower = String(reply).toLowerCase();
        const matches = variants.filter(v => new RegExp(`\\b${escapeRegExp(v.toLowerCase())}\\b`).test(lower));
        return matches.length === 1 ? matches[0] : null;
    }

    async function classify(text, variants, current, settings) {
        try {
            const prompt = settings.classifierPrompt && settings.classifierPrompt.trim()
                ? renderCustomPrompt(settings.classifierPrompt, variants, text, current)
                : buildClassifierPrompt(variants, text, current);
            // quietToLoud=false, skipWIAN=true — keep the classifier lean and side-effect free.
            const reply = await ctx().generateQuietPrompt(prompt, false, true);
            log('classifier reply:', reply);
            const pick = parseClassifierReply(reply, variants);
            if (pick && pick.toLowerCase() === String(current).toLowerCase()) return null;
            return pick;
        } catch (err) {
            warn('classify failed', err);
            return null;
        }
    }

    let autoBusy = false;

    async function maybeAutoSwitch() {
        const settings = getSettings();
        if (!settings.enabled || !settings.autoSwitch || autoBusy) return;

        const c = ctx();
        if (c.streamingProcessor && !c.streamingProcessor.isFinished) return;

        const character = getActiveCharacter();
        if (!character) return;

        const variants = getVariants(character.avatarKey);
        if (variants.length === 0) return;

        const text = lastCharacterMessageText();
        if (!text || !hasChangeCue(text, variants, settings)) return;

        autoBusy = true;
        try {
            const current = resolveVariant(character.avatarKey);
            const pick = await classify(text, variants, current, settings);
            if (pick && variants.includes(pick) && pick !== current) {
                await setVariant(character, pick, { persist: true });
                toast('success', `Auto-switched ${character.name} to “${pick}”.`);
            }
        } finally {
            autoBusy = false;
        }
    }

    let autoTimer = null;
    function debouncedAutoSwitch() {
        clearTimeout(autoTimer);
        autoTimer = setTimeout(() => { maybeAutoSwitch().catch(warn); }, 600);
    }

    // ---------------------------------------------------------------------------
    //  Quick chooser (wand menu / in-chat overlay)
    // ---------------------------------------------------------------------------

    function closeChooser() {
        document.getElementById('candy_chooser_backdrop')?.remove();
    }

    async function openVariantChooser() {
        closeChooser();

        const character = getActiveCharacter();
        if (!character) {
            toast('warning', 'No active character to switch.');
            return;
        }

        const backdrop = document.createElement('div');
        backdrop.id = 'candy_chooser_backdrop';
        backdrop.className = 'candy-chooser-backdrop';

        const menu = document.createElement('div');
        menu.className = 'candy-chooser';
        menu.innerHTML = `<div class="candy-chooser-title">🍬 ${htmlEscape(character.name)} — variant</div>
            <div class="candy-chooser-body"><div class="candy-chooser-loading">Scanning sprite folders…</div></div>`;
        backdrop.appendChild(menu);
        backdrop.addEventListener('click', (event) => {
            if (event.target === backdrop) closeChooser();
        });
        document.body.appendChild(backdrop);

        const valid = await getValidVariants(character);
        const current = resolveVariant(character.avatarKey);
        const body = menu.querySelector('.candy-chooser-body');

        const options = [{ value: '', label: 'Default (base outfit)' }]
            .concat(valid.map(v => ({ value: v, label: v })));

        if (valid.length === 0) {
            body.innerHTML = `<div class="candy-chooser-empty">No valid variant sub-folders found for
                <b>${htmlEscape(character.name)}</b>.<br>Add sprite sub-folders like
                <code>${htmlEscape(character.name)}/swimsuit/</code> and declare them in the
                CandyExpressions settings panel.</div>`;
        } else {
            body.innerHTML = '';
        }

        for (const option of options) {
            const button = document.createElement('div');
            const isActive = (option.value || '') === (current || '');
            button.className = 'candy-chooser-item interactable' + (isActive ? ' candy-active' : '');
            button.innerHTML = `<span class="candy-dot"></span><span>${htmlEscape(option.label)}</span>`;
            button.addEventListener('click', async () => {
                closeChooser();
                await setVariant(character, option.value, { persist: true });
                toast('info', option.value
                    ? `${character.name}: “${option.value}”.`
                    : `${character.name}: default outfit.`);
            });
            body.appendChild(button);
        }
    }

    // ---------------------------------------------------------------------------
    //  Wand menu button
    // ---------------------------------------------------------------------------

    function injectWandMenu(attempt = 0) {
        const menu = document.getElementById('extensionsMenu');
        if (!menu) {
            if (attempt < 20) setTimeout(() => injectWandMenu(attempt + 1), 1000);
            return;
        }
        if (document.getElementById('candy_wand_item')) return;

        const item = document.createElement('div');
        item.id = 'candy_wand_item';
        item.className = 'list-group-item flex-container flexGap5 interactable';
        item.tabIndex = 0;
        item.title = 'Switch character outfit / variant (CandyExpressions)';
        item.innerHTML = '<div class="fa-solid fa-shirt extensionsMenuExtensionButton"></div><span>Switch Variant</span>';
        item.addEventListener('click', (event) => {
            event.stopPropagation();
            openVariantChooser().catch(warn);
        });
        menu.appendChild(item);
        log('wand menu item added');
    }

    // ---------------------------------------------------------------------------
    //  Settings panel
    // ---------------------------------------------------------------------------

    function buildSettingsHtml() {
        return `
        <div class="candy-expressions-settings">
            <div class="inline-drawer">
                <div class="inline-drawer-toggle inline-drawer-header">
                    <b>🍬 CandyExpressions</b>
                    <div class="inline-drawer-icon fa-solid fa-circle-chevron-down down"></div>
                </div>
                <div class="inline-drawer-content">
                    <small class="candy-muted">Sticky, switchable expression variants (outfits / forms) layered on top of Character Expressions.</small>

                    <label class="checkbox_label" for="candy_enabled">
                        <input type="checkbox" id="candy_enabled">
                        <span>Enable CandyExpressions</span>
                    </label>
                    <label class="checkbox_label" for="candy_perchat">
                        <input type="checkbox" id="candy_perchat">
                        <span>Remember variant per-chat (off = one global default per character)</span>
                    </label>

                    <hr>
                    <div class="candy-section-title">Current character</div>
                    <div id="candy_char_name" class="candy-muted">No active character.</div>
                    <div class="flex-container flexGap5 candy-row">
                        <select id="candy_active_select" class="text_pole flex1"></select>
                        <div id="candy_set_default" class="menu_button" title="Save the selected variant as this character's global default">Set default</div>
                    </div>

                    <div class="candy-section-title">Declared variants for this character</div>
                    <small class="candy-muted">One sub-folder name per line. These must match sprite sub-folders, e.g. <code>swimsuit</code> → <code>&lt;Character&gt;/swimsuit/</code>.</small>
                    <textarea id="candy_variants_text" class="text_pole" rows="4" placeholder="swimsuit&#10;school-uniform&#10;casual"></textarea>
                    <div class="flex-container flexGap5 candy-row">
                        <div id="candy_save_variants" class="menu_button">Save variants</div>
                        <div id="candy_validate_variants" class="menu_button" title="Check which declared variants have sprites">Validate</div>
                    </div>
                    <div id="candy_validate_result" class="candy-muted"></div>

                    <hr>
                    <div class="candy-section-title">Automatic switching</div>
                    <label class="checkbox_label" for="candy_autoswitch">
                        <input type="checkbox" id="candy_autoswitch">
                        <span>Auto-switch on explicit narrative cues (uses your text model)</span>
                    </label>
                    <label for="candy_strictness">Sensitivity</label>
                    <select id="candy_strictness" class="text_pole">
                        <option value="strict">Strict — needs a change verb and a variant name (rare)</option>
                        <option value="balanced">Balanced — a change verb or a variant name is enough</option>
                    </select>

                    <div class="inline-drawer candy-advanced">
                        <div class="inline-drawer-toggle inline-drawer-header">
                            <b>Advanced</b>
                            <div class="inline-drawer-icon fa-solid fa-circle-chevron-down down"></div>
                        </div>
                        <div class="inline-drawer-content">
                            <label for="candy_prompt">Custom classifier prompt (optional)</label>
                            <small class="candy-muted">Placeholders: <code>{{variants}}</code>, <code>{{current}}</code>, <code>{{text}}</code>. Leave empty for the built-in prompt.</small>
                            <textarea id="candy_prompt" class="text_pole" rows="4" placeholder="Leave empty to use the default prompt"></textarea>
                            <label class="checkbox_label" for="candy_debug">
                                <input type="checkbox" id="candy_debug">
                                <span>Verbose console logging</span>
                            </label>
                        </div>
                    </div>
                </div>
            </div>
        </div>`;
    }

    function injectSettingsUi(attempt = 0) {
        const container = document.getElementById('extensions_settings2') || document.getElementById('extensions_settings');
        if (!container) {
            if (attempt < 20) setTimeout(() => injectSettingsUi(attempt + 1), 1000);
            return;
        }
        if (document.querySelector('.candy-expressions-settings')) return;

        container.insertAdjacentHTML('beforeend', buildSettingsHtml());
        bindSettingsHandlers();
        refreshUi();
        log('settings panel added');
    }

    function bindSettingsHandlers() {
        const settings = getSettings();

        const enabled = document.getElementById('candy_enabled');
        enabled.checked = settings.enabled;
        enabled.addEventListener('change', () => {
            settings.enabled = enabled.checked;
            saveSettings();
            if (settings.enabled) reconcile('enabled-toggle').catch(warn);
        });

        const perChat = document.getElementById('candy_perchat');
        perChat.checked = settings.perChat;
        perChat.addEventListener('change', () => {
            settings.perChat = perChat.checked;
            saveSettings();
            refreshUi();
        });

        const autoSwitch = document.getElementById('candy_autoswitch');
        autoSwitch.checked = settings.autoSwitch;
        autoSwitch.addEventListener('change', () => {
            settings.autoSwitch = autoSwitch.checked;
            saveSettings();
        });

        const strictness = document.getElementById('candy_strictness');
        strictness.value = settings.autoStrictness;
        strictness.addEventListener('change', () => {
            settings.autoStrictness = strictness.value;
            saveSettings();
        });

        const prompt = document.getElementById('candy_prompt');
        prompt.value = settings.classifierPrompt || '';
        prompt.addEventListener('change', () => {
            settings.classifierPrompt = prompt.value;
            saveSettings();
        });

        const debug = document.getElementById('candy_debug');
        debug.checked = settings.debug;
        debug.addEventListener('change', () => {
            settings.debug = debug.checked;
            saveSettings();
        });

        document.getElementById('candy_active_select').addEventListener('change', async (event) => {
            const character = getActiveCharacter();
            if (!character) return;
            await setVariant(character, event.target.value, { persist: true });
        });

        document.getElementById('candy_set_default').addEventListener('click', async () => {
            const character = getActiveCharacter();
            if (!character) { toast('warning', 'No active character.'); return; }
            const value = document.getElementById('candy_active_select').value;
            await setVariant(character, value, { persist: true, asDefault: true });
            toast('success', `Default for ${character.name}: ${value || 'base outfit'}.`);
        });

        document.getElementById('candy_save_variants').addEventListener('click', () => {
            const character = getActiveCharacter();
            if (!character) { toast('warning', 'No active character.'); return; }
            const lines = document.getElementById('candy_variants_text').value.split('\n');
            const saved = setVariants(character.avatarKey, lines);
            toast('success', `Saved ${saved.length} variant(s) for ${character.name}.`);
            refreshUi();
        });

        document.getElementById('candy_validate_variants').addEventListener('click', async () => {
            const character = getActiveCharacter();
            if (!character) { toast('warning', 'No active character.'); return; }
            const result = document.getElementById('candy_validate_result');
            const declared = getVariants(character.avatarKey);
            if (declared.length === 0) { result.textContent = 'No variants declared.'; return; }
            result.textContent = 'Checking…';
            const rows = await Promise.all(declared.map(async (variant) => {
                const ok = await probeVariant(character.name, variant);
                return `${ok ? '✅' : '❌'} ${variant}`;
            }));
            result.innerHTML = rows.map(htmlEscape).join('<br>');
        });
    }

    /** Repopulate the per-character portion of the settings panel. */
    function refreshUi() {
        const nameEl = document.getElementById('candy_char_name');
        const select = document.getElementById('candy_active_select');
        const variantsText = document.getElementById('candy_variants_text');
        if (!nameEl || !select) return;

        const character = getActiveCharacter();
        if (!character) {
            nameEl.textContent = 'No active character.';
            select.innerHTML = '<option value="">Default (base outfit)</option>';
            select.value = '';
            if (variantsText) variantsText.value = '';
            return;
        }

        const settings = getSettings();
        nameEl.innerHTML = `Acting on <b>${htmlEscape(character.name)}</b> · `
            + (settings.perChat ? 'per-chat memory' : 'global default');

        const variants = getVariants(character.avatarKey);
        const current = resolveVariant(character.avatarKey);
        select.innerHTML = '';
        const optDefault = document.createElement('option');
        optDefault.value = '';
        optDefault.textContent = 'Default (base outfit)';
        select.appendChild(optDefault);
        for (const variant of variants) {
            const option = document.createElement('option');
            option.value = variant;
            option.textContent = variant;
            select.appendChild(option);
        }
        select.value = current || '';

        if (variantsText && document.activeElement !== variantsText) {
            variantsText.value = variants.join('\n');
        }
    }

    // ---------------------------------------------------------------------------
    //  Slash commands
    // ---------------------------------------------------------------------------

    function registerSlashCommands() {
        const c = ctx();
        const { SlashCommandParser, SlashCommand, SlashCommandArgument, ARGUMENT_TYPE } = c;
        if (!SlashCommandParser || !SlashCommand) {
            warn('slash command API unavailable; skipping registration');
            return;
        }

        // Register each command independently so a name collision with another
        // extension can't take the rest down with it.
        try {
            SlashCommandParser.addCommandObject(SlashCommand.fromProps({
                name: 'candyvariant',
                aliases: ['cvar'],
                helpString: 'Set the active CandyExpressions variant for the current character. '
                    + 'Provide a variant sub-folder name, or leave empty to reset to the base outfit.',
                returns: 'the variant that was applied',
                unnamedArgumentList: [
                    new SlashCommandArgument('variant name (empty = default)', [ARGUMENT_TYPE.STRING], false),
                ],
                callback: async (_args, variant) => {
                    const character = getActiveCharacter();
                    if (!character) {
                        toast('warning', 'No active character.');
                        return '';
                    }
                    const name = sanitizeVariantName(Array.isArray(variant) ? variant.join(' ') : variant);
                    await setVariant(character, name, { persist: true });
                    return name;
                },
            }));
        } catch (err) {
            warn('failed to register /candyvariant', err);
        }

        try {
            SlashCommandParser.addCommandObject(SlashCommand.fromProps({
                name: 'candyvariants',
                helpString: 'List the declared CandyExpressions variants for the current character.',
                returns: 'a comma-separated list of variant names',
                callback: async () => {
                    const character = getActiveCharacter();
                    if (!character) return '';
                    return getVariants(character.avatarKey).join(', ');
                },
            }));
        } catch (err) {
            warn('failed to register /candyvariants', err);
        }

        log('slash commands registered');
    }

    // ---------------------------------------------------------------------------
    //  Events + boot
    // ---------------------------------------------------------------------------

    function wireEvents() {
        const { eventSource, eventTypes } = ctx();
        if (!eventSource || !eventTypes) {
            warn('event system unavailable');
            return;
        }
        if (eventTypes.CHAT_CHANGED) {
            eventSource.on(eventTypes.CHAT_CHANGED, () => { reconcile('chat-changed').catch(warn); });
        }
        if (eventTypes.CHARACTER_MESSAGE_RENDERED) {
            eventSource.on(eventTypes.CHARACTER_MESSAGE_RENDERED, debouncedAutoSwitch);
        }
        if (eventTypes.GROUP_UPDATED) {
            eventSource.on(eventTypes.GROUP_UPDATED, () => { refreshUi(); });
        }
    }

    function boot() {
        if (!globalThis.SillyTavern?.getContext) {
            setTimeout(boot, 300);
            return;
        }
        try {
            getSettings();          // materialise defaults
            injectSettingsUi();
            injectWandMenu();
            registerSlashCommands();
            wireEvents();
            setTimeout(() => { reconcile('boot').catch(warn); }, 1500);
            console.log(LOG_PREFIX, 'ready');
        } catch (err) {
            warn('boot failed', err);
        }
    }

    if (globalThis.jQuery) {
        globalThis.jQuery(boot);
    } else if (document.readyState === 'complete' || document.readyState === 'interactive') {
        boot();
    } else {
        window.addEventListener('DOMContentLoaded', boot);
    }
})();
