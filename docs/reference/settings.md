---
title: Settings Reference
description: Reference for every settings page and its persistence behavior.
---

# Settings Reference

Open **File -> Settings** or the settings button on the project screen. Pipeline, provider, translation, and typesetting edits are saved automatically after a short delay.

## Appearance

- **Theme** — Light, Dark, or System.
- **Language** — Changes the desktop interface language.

Appearance is interface-only. It does not change page rendering or translation output.

## Pipeline

Select independent models for detection, OCR, bubble inpainting, artwork inpainting, and decensoring. Model-specific fields appear with the processor that owns them:

- detection text, bubble, and panel thresholds;
- generative inpainting positive and negative prompts;
- censor detection enable, mosaic detection, threshold, mask grow, and mask blur, which belong to the censor detection stage rather than to a model.

Bubble inpainting selects LaMa or AOT; uniform bubble backgrounds are filled without model inference. Artwork inpainting defaults to FLUX.2 Klein 4B for lettering and sound effects outside bubbles. Its prompt, negative prompt, sampling settings, and endpoint are independent of decensoring, even when both use the same model. Switching models and returning restores that route's saved profile.

Choose the bubble model under `[pipeline.inpainting] bubbles`; select the artwork model under `[pipeline.inpainting.artwork] model`. Profiles live under `[pipeline.inpainting.artwork.<model>]` and `[pipeline.decensor.<model>]`; `[pipeline.processor]` contains detection settings only. The old flat inpainting tables are not read. Re-select the bubble model and restore text-cleanup profiles under the artwork namespace; do not reuse a decensor prompt there. Previously processed pages need a whole-page **Detection** pass to create the split masks before inpainting.

Manual **Inpainting (FLUX)** has a separate `[pipeline.inpainting.manual]` profile. Selecting **Fast**, **Inpainting (FLUX)**, or **Decensor** opens that tool's model controls in the editor's Properties panel. The 9B controls include prompt, inference steps, guidance scale, seed, denoise strength, and optional LoRA path and strength. Apply changes before drawing: already accepted masks keep their captured settings.

## Providers

Configure Local, Atlas Cloud, OpenAI, Gemini, Claude, Grok, Cursor, DeepSeek, OpenAI-compatible, OpenRouter, LM Studio, DeepL, Google Cloud Translation, and Caiyun connections.

Credentials are stored in the operating system credential service. Provider URLs and non-secret options are stored in the shared configuration file. Changing a provider causes the model picker to refresh its catalog.

## Translation

- provider and model;
- local-model quantization when available;
- target language;
- project-wide translation instructions;
- content profile — Hentai, Manga, or Comic — and that profile's system prompt;
- temperature, top P, top K, and min P;
- repeat, frequency, and presence penalties;
- maximum tokens;
- thinking mode;
- vision input when the selected model supports source-page images.

The processing selector exposes the most frequently changed model and output fields without opening the full settings page.

Each content profile keeps its own system prompt. An empty prompt uses the profile's built-in template, shown as the field placeholder; **Edit built-in prompt** copies it for editing. `{source}` and `{target}` are replaced with the run's languages, so one saved prompt serves every language pair. Editor runs use the selected profile. Automation runs, including Telegram orders, name their order's profile and use the matching saved prompt. Prompts are stored under `[pipeline.translation.system_prompts]` as `hentai`, `manga`, and `comic`; the former single `system_prompt` entry is not read.

Each translation request includes valid source/translation pairs from preceding project pages in the selected target language. Selected pages translate in project order, using the latest committed scene rather than a separate chat history. Editing source text invalidates its generated translation; a manually authored translation is preserved. This supplies continuity within the model's context window, not training or guaranteed improvement with every page.

## Typesetting

Configure the ordered default font-family stack. When a text layer has no usable preferred family, Koharu tries these families in order and then uses system fallback.

This is a fallback policy; individual text layers can still choose their own font, weight, style, size, colors, alignment, and writing mode.

## Shortcuts

Assign one character to Select, Text, Brush, Eraser, Color picker, Fast, Inpainting (FLUX), Decensor, Pan, and Fit Window. The current shortcut editor updates the running UI session; custom tool bindings are not part of `~/.koharu/config.toml`.

## Storage

Pipeline, provider, translation, typesetting, and agent configuration use sections of:

```text
~/.koharu/config.toml
```

The order-management workspace is hidden unless `orders = true` is set in the `[ui]` section; the order store keeps working either way. Restart Koharu after changing it.

Do not put credentials in this file. See [Formats and data locations](/reference/formats-and-data/) for all storage boundaries.

### Isolated profiles

Two environment variables relocate everything Koharu stores for a user, which is useful for a portable install, a second profile, or reproducible screenshots:

| Variable | Replaces | Also moves |
|---|---|---|
| `KOHARU_CONFIG_DIR` | `~/.koharu` | `config.toml`, the token-usage journal `usage.sqlite3`, `stage-timings.jsonl` |
| `KOHARU_PROJECTS_DIR` | `Documents/Koharu` | `.khrproj` projects and the orders store `orders.sqlite3` |

Both must be absolute paths. Credentials stay in the operating system credential service and are shared between profiles. A second profile running next to the first also needs its own bundle identifier, because Koharu allows one window per identifier.

`KOHARU_REMOTE_DEBUGGING_PORT=<port>` additionally opens the Chromium DevTools protocol on `127.0.0.1:<port>` for UI automation. It gives the connecting program full control of the app; leave it unset in normal use.
