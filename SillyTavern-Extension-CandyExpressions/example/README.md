# Example sprite layout

CandyExpressions variants are just **sub-folders** inside a character's sprite
folder. Below is a complete example for a character named **Seraphina** with three
variants: the base outfit plus `swimsuit` and `school-uniform`.

```
<SillyTavern sprite root>/
└── Seraphina/                 ← base outfit (default)
    ├── joy.png
    ├── sadness.png
    ├── anger.png
    ├── surprise.png
    ├── neutral.png
    ├── ...                     ← the rest of your expression labels
    │
    ├── swimsuit/               ← variant "swimsuit"
    │   ├── joy.png
    │   ├── sadness.png
    │   ├── anger.png
    │   ├── neutral.png
    │   └── ...
    │
    └── school-uniform/         ← variant "school-uniform"
        ├── joy.png
        ├── sadness.png
        ├── anger.png
        ├── neutral.png
        └── ...
```

## Rules of thumb

1. **Same labels everywhere.** Each variant sub-folder should contain the same
   expression labels (`joy`, `sadness`, …) as the base folder. If a label is missing
   from a variant, SillyTavern falls back to your configured fallback expression.
2. **Sub-folder name = variant name.** The folder name (`swimsuit`,
   `school-uniform`) is exactly what you type into the CandyExpressions
   **Declared variants** box and what `/candyvariant swimsuit` expects.
3. **Single level only.** Variants are one sub-folder deep. `Seraphina/swimsuit` is
   valid; `Seraphina/beach/swimsuit` is not.

## Uploading sprites into a variant

From SillyTavern you can populate a variant folder without touching the filesystem:

```
/uploadsprite folder=/swimsuit label=joy {{pipe}}
```

or use the stock **Character Expressions** upload panel after switching to the
variant with `/costume /swimsuit`.

## Where is the sprite root?

Depending on your SillyTavern version and setup, character sprite folders live under
something like:

- `SillyTavern/data/<user>/user/images/<Character>/` (multi-user), or
- `SillyTavern/public/characters/<Character>/` / the configured expressions path.

Use the **Validate** button in the CandyExpressions settings panel to confirm a
variant is being found — it probes SillyTavern's own sprite API, so a ✅ means the
app can see it regardless of the exact path.
