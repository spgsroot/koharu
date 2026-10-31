---
title: Cleanup and Inpainting
description: Remove source lettering automatically or with an authored mask, then repair remaining artifacts.
---

# Cleanup and Inpainting

Inpainting reconstructs artwork where source lettering is removed. Good output depends more on a precise removal region than on using the largest model.

## Automatic cleanup

Detection produces two disjoint lettering masks: text inside speech bubbles and text over artwork, including sound effects and isolated glyphs. Inpainting fills uniform bubble interiors without inference, cleans remaining bubble text with LaMa/AOT, then reconstructs artwork with its independently selected model (FLUX.2 Klein 4B by default). Only masked pixels are committed. Without masked lettering, no model is loaded. Pages carrying only the old combined text mask need a whole-page **Detection** pass first.

Review automatic output at both reading size and high zoom. Look for:

- fragments of source glyphs;
- erased speech-bubble borders;
- repeated texture or smeared screentone;
- changes outside the lettering;
- missed small punctuation or furigana.

## Manual cleanup brushes

Choose **Fast** (`J`) for lettering cleanup with the selected bubble LaMa/AOT model, **Inpainting (FLUX)** (`P`) for manual FLUX repair, or **Decensor** (`D`) for censorship reconstruction. Manual FLUX has its own prompt and sampling settings, independent of both automatic artwork cleanup and Decensor. Selecting a repair tool exposes its model controls in the right-hand Properties panel; apply changes before drawing.

Set the brush diameter and paint only the marks that need reconstruction. Releasing a stroke immediately submits its mask to one FIFO queue shared by all three tools; no separate Run action is needed. Continue drawing while another mask is processed, including across same-page canvas updates. Accepted masks keep their settings and run on the preceding job's committed result. A stroke replaces the saved mask for its invocation only; manual FLUX and Decensor do not require saved automatic masks.

Use several controlled strokes instead of covering a large bubble. Include enough of each glyph for the model to remove it, but avoid neighboring line art whenever possible.

Cancel an unfinished gesture if its mask is wrong. Stop a queued or running job through the activity controls. Switching or closing projects cancels their outstanding masks rather than applying them to another project. Temporary masks are not retained for a later run.

## Authored raster repair

Use **Brush** and **Eraser** when a deterministic manual touch-up is faster than another model run. Sample nearby color first, work on the smallest useful brush, and check the page at normal zoom afterward.

## Choosing an inpainting model

Select LaMa or AOT for bubble inpainting. Artwork inpainting has its own model row, prompt, and sampling settings; FLUX.2 Klein 4B is the default for sound effects over drawing. Direct LaMa/AOT remain optional artwork choices: they are faster and handle blurred or softly shaded backgrounds well, but leave outline ghosts on flat decorative plates. Decensoring has a separate generative-model row and never supplies its prompt to text cleanup. See [Cleanup evidence](/fork/cleanup-evidence/) and [Vision and inpainting models](/models/vision-and-inpainting/) before switching.

Rerunning automatic **Inpainting** or **Decensor** updates only masked pixels in the selected scope and preserves cleanup elsewhere, including the other stage's generated pixels. Both automatic stages skip a page whose cleanup the user edited with any manual repair brush, or with Brush and Eraser on the Cleanup layer. To regenerate such a page automatically, delete its Cleanup layer, then run Inpainting and Decensor again. Precise masks limit changes outside lettering, but detection misses and generative artifacts still need review.
