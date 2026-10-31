---
title: Generative Inpainting Integrations
description: Native FLUX.2 Klein 9B with an uncensored quantized encoder and local DiffusionCreamPy-compatible ComfyUI inpainting profiles.
---

# Generative Inpainting Integrations

Status: the native FLUX.2 Klein 9B uncensored profile and the ComfyUI-backed Animagine XL 4.0 Opt and Illustrious XL 2.0 profiles are selectable in Pipeline settings and run through Koharu's in-process stage lifecycle.

## Scope

Koharu exposes two distinct generative repair paths, each selectable for ordinary inpainting and for decensoring:

1. a DiffusionCreamPy-compatible SDXL workflow executed by a loopback ComfyUI service, with separate Animagine XL 4.0 Opt and Illustrious XL 2.0 profiles;
2. a native FLUX.2 Klein 9B image-editing profile using a Q8_0 GGUF denoiser and an uncensored Q4_K_M text encoder.

**Inpainting** has two independent routes: speech-bubble text defaults to LaMa (AOT is also selectable), while lettering over artwork, including onomatopoeia and isolated glyphs, defaults to FLUX.2 Klein 4B with a text-removal prompt; the uncensored 9B profile remains selectable. Uniform bubble interiors are filled without model inference; textured interiors use the selected fast model. Artwork repair runs after bubble cleanup and sees the cleaned image. **Decensor** retains its own model, prompt, sampling, and endpoint settings; sharing a model never shares conditioning or turns text cleanup into censorship reconstruction.

These paths share image cropping, masks, result composition, progress, cancellation, and scene history. Animagine is an SDXL checkpoint used with ControlNet Union ProMax; FLUX.2 Klein is a separate transformer architecture and must not be substituted into the SDXL workflow.

## DiffusionCreamPy with Animagine XL 4.0 Opt

[DiffusionCreamPy](https://github.com/Schwi-AI/DiffusionCreamPy) is a collection of ComfyUI workflow templates rather than a Python package or inference server. Its bar-removal, mosaic-removal, and basic-inpainting templates combine mask-aware cropping, an SDXL checkpoint, ControlNet Union ProMax, sampling, color matching, and masked composition.

The supported profiles use:

- workflow: [DiffusionCreamPy Inpaint CNet++](https://github.com/Schwi-AI/DiffusionCreamPy/blob/main/workflows/DiffusionCreamPy-Inpaint-CNet%2B%2B.json);
- Animagine checkpoint: [cagliostrolab/animagine-xl-4.0](https://huggingface.co/cagliostrolab/animagine-xl-4.0), file `animagine-xl-4.0-opt.safetensors`;
- Illustrious checkpoint: [OnomaAIResearch/Illustrious-XL-v2.0](https://huggingface.co/OnomaAIResearch/Illustrious-XL-v2.0), file `Illustrious-XL-v2.0.safetensors`;
- ControlNet: [xinsir/controlnet-union-sdxl-1.0](https://huggingface.co/xinsir/controlnet-union-sdxl-1.0), ProMax weights installed as `controlnet-union-sdxl-1.0.safetensors`.

Both profiles pin the same API-format graph but retain independent settings. The default sampling values mirror the upstream DiffusionCreamPy inpaint graph: Euler Ancestral, beta scheduler, 10 steps, CFG 7, denoise 0.5, 128-pixel crop padding, and a 16-pixel mask blur. The checkpoint, prompts, seed, sampling, mask, color-match, endpoint, and timeout values remain editable per profile.

### Required ComfyUI environment

The integration targets a separately installed, loopback-only ComfyUI service. The reference workflow requires these custom-node packages:

- `comfyui-advanced-controlnet`;
- `comfyui_controlnet_aux`;
- `comfyui-inpaint-nodes`;
- `comfyui-inpainteasy`.

Koharu does not silently install or update ComfyUI or custom nodes. Settings hold the loopback endpoint, model filenames, and workflow parameters. Stage loading reports missing nodes, checkpoints, or ControlNet assets before an image is uploaded.

The imported DiffusionCreamPy editor workflow is not directly submitted by the application. Koharu pins an API-format graph at `crates/koharu-pipeline/workflows/diffusioncreampy-sdxl-api.json`; mutable request values are bound only to its named image, checkpoint, prompt, seed, sampling, mask, crop, and color-match inputs. Arbitrary user-supplied workflow execution is outside this integration.

### ComfyUI request lifecycle

The adapter uses the local server contract:

1. upload a uniquely named RGBA source whose alpha channel carries the inpainting mask with `POST /upload/image`;
2. queue the pinned API-format workflow with `POST /prompt`;
3. poll only the matching `prompt_id` through `GET /history/{prompt_id}`;
4. surface the matching execution error or read output metadata from the pinned output node;
5. download that output through `GET /view`.

Each request carries a unique client ID and filename prefix. Koharu never overwrites a shared fixed input name. Cancellation outside the adapter discards the uncommitted stage result, so a late ComfyUI result cannot publish a scene patch.

### Workflow modes

Basic inpainting is the only ComfyUI mode implemented for the translation pipeline. It consumes the removal mask produced by detection or the transient mask painted with the Remove tool.

Bar and mosaic restoration remain manual editor operations until Koharu has explicit censorship-region inputs. A text-removal mask is not evidence of a bar or mosaic region and must not select those modes automatically.

### Shared result invariants

Koharu retains ownership of the durable result even when inference runs in ComfyUI:

- output dimensions match the input crop;
- pixels outside the final feathered composition mask remain unchanged;
- one invocation produces one named scene-history step;
- rerunning replaces derived cleanup without modifying authored raster repair;
- progress and downloads use the existing Activity Center;
- ComfyUI paths and temporary filenames never enter the project document.

DiffusionCreamPy is licensed under CC BY 4.0. Distributions that include derived workflow templates must retain attribution to the upstream project.

## FLUX.2 Klein 9B with uncensored text encoder

The FLUX profile is separate from DiffusionCreamPy and ComfyUI. It extends Koharu's native `stable-diffusion.cpp` path with a 9B denoiser, the existing FLUX.2 VAE, source-image conditioning, and FLUX.2 sampling parameters.

### Pinned assets

- denoiser repository: [unsloth/FLUX.2-klein-9B-GGUF](https://huggingface.co/unsloth/FLUX.2-klein-9B-GGUF);
- denoiser file: `flux-2-klein-9b-Q8_0.gguf`;
- text encoder repository: [ponpoke/flux2-klein-9b-uncensored-text-encoder](https://huggingface.co/ponpoke/flux2-klein-9b-uncensored-text-encoder);
- text encoder file: `flux2-klein-9b-uncensored-q4_k_m.gguf`;
- text encoder size: 5,027,783,648 bytes (downloaded pinned file; SHA-256 `df33c8bef6f75c00dd282cd9850e2bf40afb363c6bbe464e977b225882550a14`);
- VAE file: `full_encoder_small_decoder.safetensors`.

The previous 15.3 GB `qwen_3_8b.safetensors` encoder, base-9B denoiser, and line-art reference-control LoRA are not part of this profile. Koharu exposes one 9B configuration and never substitutes the removed assets.

### Native inference contract

The selected crop conditions native FLUX.2 editing as a reference image. At full strength, the native call omits `init_image` and `mask_image`, avoiding an unnecessary second VAE encode. Partial-strength calls retain native IMG2IMG and mask conditioning. The distilled 9B denoiser defaults to Euler with the Flux2 scheduler, CFG 1.0, eight inference steps, and full-strength editing. Four steps left black censor bars visible in a real masked-crop smoke; existing profiles explicitly set to four steps retain that setting and should be adjusted in Pipeline settings when reconstructing censorship. The prompt, CFG, and step count remain editable.

The generated crop is composited only inside the selected stage mask. Page dimensions, pixels outside the composition mask, cancellation semantics, and scene history remain owned by the existing stage commit path.

### Optional local LoRA

The 9B profile accepts `lora_path` and `lora_strength` independently for each stage. In **Settings → Pipeline**, set the local `.safetensors` path and strength on **Decensor**, leaving the artwork text-cleanup profile's path empty. An empty path or strength `0` disables the adapter. Model weights remain shared, but every generation supplies its own adapter list; disabling it clears the previous request's LoRA rather than retaining it for text removal. Quantized weights use native runtime adapters, not a full-model merge.

For the supplied `uncensorMangaKlein_Monochrome.safetensors`, the inspected metadata identifies `flux2_klein_9b`, rank 32, and 224 BF16 tensors. The file is 165,704,424 bytes with SHA-256 `913613a95e72da845f8f113d521af66cc2a7d4f3ad3caac4e207d5bc01a45d51`. This is not the archived `uncensorMangaKlein_V2.safetensors`; do not substitute its checksum. Keep the local adapter in the runtime store's `loras/<sha256>/` directory instead of Downloads, and select it with strength `1.0`. Weights and machine-specific configuration are not committed.

On a real 352×336 crop, a fixed-seed CUDA smoke confirmed that enabling this adapter changes only selected pixels; clearing the path and setting strength to zero both reproduce the no-LoRA baseline exactly. A separate pipeline smoke committed **Decensor → Inpainting → Decensor** with the adapter enabled only for Decensor.


### Lettering masks and settings

Detection writes disjoint page-sized `bubble-text-mask` and `artwork-text-mask` assets and their `text-mask` union. Bubble ownership uses instance-mask containment rather than bounding-box corners. Bubble text growth is clipped to the bubble interior. Artwork uses smaller dilation without closing gaps between neighbouring glyphs; enclosed interiors of outlined lettering are included so opaque glyph fills can be reconstructed. Bubble and panel boundaries constrain added growth, but genuine sound effects crossing those boundaries remain artwork. These rules refine predicted masks; missed or inaccurate detections and generated texture artifacts still require manual review.

Settings use `[pipeline.inpainting] bubbles = "lama"` or `"aot-inpainting"`, `[pipeline.inpainting.artwork] model = "flux2-klein"` (or `"flux2-klein-9b-uncensored"`), and `[pipeline.inpainting.artwork.<model>]` for artwork profiles. Decensor profiles remain under `[pipeline.decensor.<model>]`. The old flat inpainting schema is not read: choose the bubble model again and restore only text-cleanup settings into the artwork profile, never a censorship-reconstruction prompt.

Existing `text-mask`-only pages require **Detection** on the whole page before inpainting. A regional rerun does not silently convert incomplete legacy masks. Once both masks exist, regional detection preserves each route outside the selected window. Inpainting and decensoring commit only selected mask pixels into one shared cleanup layer under one pipeline producer, so either automatic stage may rerun over the other's generated pixels while preserving cleanup elsewhere. Neither automatic stage writes a cleanup layer whose entity, layer, or pixels the user authored; such pages are skipped until the Cleanup layer is deleted.

Manual **Fast** uses the selected bubble LaMa/AOT backend; **Inpainting (FLUX)** uses `[pipeline.inpainting.manual]`; **Decensor** uses its own selected backend and profile. The manual FLUX prompt, seed, denoise strength, sampling settings, and optional LoRA are independent of automatic artwork cleanup and decensoring. Their model controls are available in the editor's Properties panel.

Native admission freezes the accepted mask's model settings before PNG preparation and orders all manual targets in one FIFO queue. Each job takes the latest scene when it starts, preserving the preceding job's edits without changing the accepted prompt. Same-page frame activation preserves an unfinished browser stroke, and pending masks remain visible by job ID. Project-session changes stop outstanding work, retain bounded terminal API results, and prevent stale commits even when reopening the same project. The transient mask is neither persisted nor forwarded to a later operation.


### Automatic censorship mask

**Censor detection**, **Inpainting**, and **Decensor** are separate pipeline stages. Detection combines Er0mangaSeg black-bar scores on the original page with white-stroke stack geometry and, when **Detect mosaics** is enabled, pixel-mosaic heuristics. It commits a grayscale `censor-mask` asset. Its settings — enabled, mosaic detection, threshold, mask grow, and mask blur — belong to the stage rather than a generative model profile. The threshold affects the checkpoint's black-bar scores, not the white-stroke or mosaic heuristics. The editor's **Censor mask** button overlays the saved mask in red; run only **Censor detection** to inspect it without loading a generative model.

Black-bar refinement uses confident model scores as seeds, follows weakly supported dark ink to recover bar ends, and groups nearby parallel or fanned bars. Thin connecting strokes and collinear fragments of a panel border do not form a censorship stack. Accepted stacks can include touching rings or ladder-shaped bar networks; dark anti-aliased rims are restored before the configured grow and blur. The existing conservative requirement of at least three neighbouring bars remains: isolated one- or two-bar censorship is not guaranteed.

White-stroke detection does not invert the page. It searches multiple directions for straight, constant-width white bands interrupting textured ink, rejects bands next to large white regions or continuous outlines, and reports only stacks of at least three spaced, roughly parallel strokes. Speech bubbles, gutters, lettering rows and bare paper are not evidence of censorship. Isolated strokes, pairs, thin or crossing marks, and ends disappearing into white paper can remain unmarked. Inspect the mask before reconstruction; this heuristic is not complete censorship segmentation.

Mosaic candidates must sit among flat-colour tile interiors at full resolution before they contribute to the periodic-grid estimate. This rejects fine dotted screentones whose downsampled pattern otherwise aliases into a mosaic grid, including large connected false masks at panel edges. Coarse dots, noisy tiles and other periodic textures remain possible sources of misses or false positives.

**Inpainting** removes lettering using the bubble and artwork routes described above. **Decensor** reconstructs only the saved censor mask with its own selected model. It does not rerun detection or remove text, and it preserves existing cleanup pixels outside the mask. Running **Decensor** alone changes only the masked area. After updating the detector or changing threshold, dilation, blur, or mosaic detection, rerun **Censor detection**: previously saved masks are not recomputed automatically. Turning censor detection off skips mask generation while keeping text inpainting available; decensoring then records no change unless the page already carries a saved censor mask.

The checkpoint is not committed. Convert the official `convnext_1024_iter_400.pth` with `scripts/export_eromangaseg.py --config <Er0mangaSeg source>/configs/convnext/convnext_h.py --checkpoint <checkpoint>`. The script requires PyTorch, mmengine, mmcv-lite, mmsegmentation, and mmpretrain in its Python environment. Both the config and checkpoint must come from trusted upstream: the official checkpoint includes serialized training metadata, so the exporter must deserialize it with `torch.load(weights_only=False)`. The exporter verifies SHA-256 `9c0e7ae74d043d6ff94e08408bf8a1c032158d2a86d6f7feb6738fb8843b31a6` before loading.

It writes the frozen TorchScript model to `target/release/store/eromanga-seg/2f1d5210459c751ed716b4887be906dffbb4f034/model.pt`; release bundling copies that store directory, including the upstream Apache-2.0 notice. The bundled MIT notice for the mosaic detector is `LICENSE-MOSAIC-DETECTOR`. Prepare the model before bundling; without it, censor detection fails with an explicit installation error instead of committing a partial page.

### Residency and failure handling

The Q8_0 denoiser remains a high-memory profile even though the Q4_K_M encoder reduces host storage and parameter pressure. Koharu loads only the active generative backend and uses the existing accelerator lane. FLUX parameters stay memory-mapped in RAM and stream to the accelerator through stable-diffusion.cpp's budgeted graph-cut path: at every compute the budget is clamped to the VRAM free at that moment, so other resident models reduce speed gradually instead of oversubscribing the card. Staging the whole 9.5 GB denoiser per tile caused Windows to page GPU allocations through system memory on a 16 GB card, where the same 352x336 tile slowed from about 3 s to over 4 minutes.

Missing gated assets, rejected model terms, an incompatible encoder, or a native loader error must fail before a scene commit and leave the original page unchanged. FLUX.2 Klein 9B and derived encoder weights remain subject to their respective model terms; changing the encoder does not relax the denoiser's license.

Decision: keep only the distilled 9B profile with the uncensored quantized encoder. The base-9B profile and reference-control LoRA are removed rather than retained as hidden or compatibility paths.

## Implementation order

1. The native 9B denoiser and uncensored encoder are pinned by repository revision and filename.
2. The pipeline profile, settings controls, masked composition, cancellation, and scene history integration are complete.
3. The DiffusionCreamPy API graph, loopback adapter, custom-node preflight, and Animagine and Illustrious profiles are complete.
4. Shared Koharu crop tiling and final masked composition remain authoritative around every generative backend.
5. Manual bar or mosaic modes remain out of scope until their masks and preprocessing have explicit editor-owned inputs.

## Acceptance checks

- Animagine and Illustrious output is produced through the pinned DiffusionCreamPy-derived API workflow, not a visually similar replacement graph.
- Missing ComfyUI custom nodes and model files are named before image upload.
- FLUX uses the distilled 9B Q8_0 denoiser and the pinned uncensored Q4_K_M encoder without a reference-control LoRA.
- The removed base-9B and line-art profiles are not selectable or accepted by configuration parsing.
- Fixed-seed runs preserve image dimensions and leave pixels outside the composition mask unchanged.
- Cancellation never publishes a late output.
- Each successful operation creates one undoable history entry.
- Switching one stage between SDXL and FLUX does not retain both heavyweight backends concurrently.
- Inpainting and decensoring keep independent model selections, prompts, negative prompts, and sampling settings across save/reload and model switches.
