---
title: Install Koharu
description: Build and install Koharu locally.
---

# Install Koharu

This fork is distributed as a local source build. Install the prerequisites from [Development Setup](/development/setup/), then run:

```powershell
bun install
cargo tauri build --bundles nsis
```

The Windows installer is written to `target/release/bundle/nsis`.

## First launch

Koharu opens a project browser after the native runtime is ready. The first launch can take longer because Koharu may need to download native runtime packages. Individual model files are resolved when the selected model is first used.

Downloads require access to GitHub release assets and, for model weights, Hugging Face. Progress appears in the activity center. Do not close the application while a package is being published to the local cache.

## Next step

Create a project and process a page in [Translate your first project](/getting-started/first-project/). For hardware selection and cache behavior, see [Runtimes, models, and hardware](/getting-started/runtime-models-and-hardware/).
