# 🍬 CandyExpressions

A [SillyTavern](https://github.com/SillyTavern/SillyTavern) UI extension that adds
**sticky, switchable expression _variants_** — outfits, forms, art styles — on top
of the built-in **Character Expressions** system.

SillyTavern already shows an emotion sprite for the character that is speaking.
CandyExpressions lets each character have **multiple sprite sets** (e.g. `casual`,
`swimsuit`, `school-uniform`, `battle-form`) and gives you control over which set is
active:

- **Pick a variant from chat** — a *Switch Variant* button in the wand (🪄) menu and a
  dropdown in the settings drawer.
- **Sticky** — the chosen variant is remembered and re-applied automatically, so it
  survives emotion changes, swipes, and reloads. It stays put until *you* change it.
- **Conservative auto-switch (opt-in)** — an optional classifier that changes the
  variant **only** when the narration explicitly describes such a change (e.g. "she
  changed into her swimsuit") and it maps onto one of that character's known variants.
  Otherwise it never touches your selection.

It is a thin controller over SillyTavern's own `/costume` sprite-folder-override
mechanism, so it stays compatible with the stock Expressions extension instead of
replacing it.

---

## How variants map to sprites

A "variant" is simply a **sub-folder** of a character's sprite folder. If your
character `Seraphina` normally loads sprites from:

```
<sprite root>/Seraphina/
    joy.png
    sadness.png
    anger.png
    ...
```

then a `swimsuit` variant is just:

```
<sprite root>/Seraphina/
    swimsuit/
        joy.png
        sadness.png
        anger.png
        ...
    school-uniform/
        joy.png
        ...
```

Each variant sub-folder should contain the **same expression labels** as the base
folder (`joy`, `sadness`, etc.). This is exactly the layout SillyTavern's
`/costume /swimsuit` command already expects — CandyExpressions just makes it
discoverable, persistent, and switchable from the UI.

> You can upload sprites into a sub-folder from SillyTavern itself using the stock
> Expressions panel, or the `/uploadsprite folder=/swimsuit label=joy <image>` slash
> command. See the [example layout](example/README.md).

---

## Installation

### Option A — Install from URL (recommended)

1. In SillyTavern, open **Extensions** (the stacked-blocks icon) → **Install extension**.
2. Paste the repository URL:
   ```
   https://github.com/CountCandy/SillyTavern-Extension-CandyExpressions
   ```
3. Choose to install for yourself (or all users) and confirm.
4. Reload SillyTavern if prompted.

### Option B — Manual install

Clone or copy this folder into your SillyTavern third-party extensions directory:

```
SillyTavern/data/<your-user>/extensions/CandyExpressions/
```

(or `SillyTavern/public/scripts/extensions/third-party/CandyExpressions/` on older,
single-user setups), then reload.

The **Character Expressions** extension must be enabled — CandyExpressions drives it.

---

## Usage

### 1. Declare a character's variants

Open **Extensions → CandyExpressions**. With a character selected, list one variant
sub-folder name per line under **Declared variants**, e.g.:

```
swimsuit
school-uniform
casual
```

Click **Save variants**, then **Validate** to confirm each one has sprites
(✅ / ❌).

### 2. Switch variants from chat

- Click the wand (🪄) menu → **Switch Variant**, then pick from the list, **or**
- Use the **Current character** dropdown in the settings drawer, **or**
- Run a slash command:
  ```
  /candyvariant swimsuit      // switch to the "swimsuit" variant
  /candyvariant               // reset to the base outfit
  /candyvariants              // list declared variants
  ```
  (alias: `/cvar`)

Your choice is **sticky** — it persists until you change it again.

### 3. (Optional) Automatic switching

Enable **Auto-switch on explicit narrative cues** in settings. After each character
message, if the text contains a wardrobe/appearance cue **and** references one of the
character's declared variants, CandyExpressions asks your text model to confirm the
change and switches only on a confident, unambiguous match.

- **Strict** (default): requires both a change verb *and* a variant name in the
  message before the model is even consulted — deliberately rare.
- **Balanced**: either a change verb or a variant name is enough to consult the model.

Auto-switch never fires while a response is still streaming, and never picks the
already-active variant.

---

## Settings reference

| Setting | Description |
| --- | --- |
| **Enable CandyExpressions** | Master switch. When off, nothing is applied or auto-detected. |
| **Remember variant per-chat** | On: the active variant is stored in the chat's metadata (each chat can differ). Off: a single global default per character. |
| **Declared variants** | The variant sub-folder names offered for the current character. |
| **Set default** | Save the currently selected variant as this character's global default. |
| **Auto-switch on narrative cues** | Opt-in automatic switching (see above). |
| **Sensitivity** | `Strict` or `Balanced` gate before the classifier is consulted. |
| **Custom classifier prompt** | Advanced. Override the built-in prompt; supports `{{variants}}`, `{{current}}`, `{{text}}`. |
| **Verbose console logging** | Extra `[CandyExpressions]` logs for debugging. |

---

## How it works (for the curious)

- Variants are applied through SillyTavern's public `/costume` slash command, which
  sets the character's `expressionOverrides` path and re-renders the sprite. No
  internal functions are imported.
- Validation probes `GET /api/sprites/get?name=<Character>/<variant>` and treats a
  variant as valid when it returns at least one sprite.
- Sticky state lives in the chat metadata (`per-chat`) or in the extension's global
  settings (`defaults`), and is reconciled on `CHAT_CHANGED` and at boot.
- The auto-switch classifier uses `generateQuietPrompt`, gated by a keyword pre-filter
  so it stays cheap and rare.

---

## Compatibility & limitations

- Requires the stock **Character Expressions** extension to be enabled.
- Group chats are supported best-effort: the picker acts on the character who sent the
  most recent message.
- There is no server API to *enumerate* sprite sub-folders, so variants are declared
  by name and validated by probing. This is intentional — it keeps you in control.

---

## License

MIT © CountCandy. See [LICENSE](LICENSE).

Sprite art and model weights are **not** covered by this license and remain the
property of their respective creators.
