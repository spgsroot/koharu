---
title: Vision and Inpainting Models
description: Choose detection, OCR, and inpainting processors and understand their trade-offs.
---

# Vision and Inpainting Models

Open **Settings -> Pipeline** to choose a model for each processing stage. **Inpainting** and **Decensor** each keep their own selection. Every model owns its preprocessing and options; switching models does not create a second workflow.

## Detection

**Koharu Layout RF-DETR Seg 2XL** is the current detection processor. It finds text, speech bubbles, and panels and produces segmentation data used by later stages.

Its optional text, bubble, and panel thresholds control how much evidence is required for each class. Lower thresholds retain more uncertain regions and can increase false positives. Raise a threshold only after checking several representative pages.

## OCR

Current OCR choices are:

- **PaddleOCR-VL 1.6** — the default general vision-language OCR path;
- **Manga OCR** — specialized for Japanese manga text;
- **Baberu OCR** — an alternative manga-oriented recognizer;
- **Hayai OCR** — a Chinese, Korean, Japanese, English manga-oriented recognizer.

OCR runs on detected text regions. A recognizer cannot recover text that detection omitted, so inspect regions before treating an empty OCR result as a language-model problem.

## Inpainting

Inpainting rebuilds the artwork behind removed source lettering. Current choices are:

- **LaMa** — the default direct inpainting model;
- **AOT Inpainting** — an alternative direct model;
- **FLUX.2 Klein** — generative inpainting with a prompt;
- **RORem Mixed** — manga-focused generative inpainting with positive and negative prompts.

Generative choices generally need larger runtime packages, more memory, and more time. Keep prompts about reconstructing the surrounding artwork and excluding letters; do not ask the inpainting model to typeset the translation.

## Decensoring

**Decensor** reconstructs the areas that **Censor detection** recorded, and selects its model separately from inpainting. Direct models are not offered here because they spread surrounding pixels rather than inventing obscured anatomy. The choices are:

- **FLUX.2 Klein 9B Uncensored** — the default decensoring model;
- **FLUX.2 Klein** — native generative inpainting with a prompt;
- **DiffusionCreamPy · Animagine XL 4.0 Opt** and **DiffusionCreamPy · Illustrious XL 2.0** — local ComfyUI SDXL profiles;
- **RORem Mixed** — manga-focused generative inpainting with positive and negative prompts.

Decensoring runs only where a saved censor mask exists, keeps cleanup pixels outside that mask, and uses the prompt profile of the model it selects. See [Censor detection and decensoring](/development/generative-inpainting-integrations/) for the mask settings.

## Model profiles

Koharu remembers one profile per model **within each stage**. Returning to a model restores that stage's prompts and sampling settings; selecting the same model for inpainting and decensoring does not share their configuration. Native loaded weights can be shared without sharing prompts.

The default decensor instruction is: “Remove the black censor bars and reconstruct the original artwork. Match surrounding line art, shading, anatomy and composition exactly.” FLUX text inpainting instead defaults to “Remove the text and reconstruct the background.” Existing custom prompts remain editable.

Models load lazily. The first run includes download, loading, and profiling costs that later runs may not have. Compare quality on identical pages after warm-up before changing defaults for a project.
