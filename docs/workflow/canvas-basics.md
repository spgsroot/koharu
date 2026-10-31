---
title: Canvas Basics
description: Navigate the canvas and use selection, text, paint, erase, color, remove, decensor, and pan tools.
---

# Canvas Basics

Koharu renders the page through WebGPU in the desktop webview. The toolbar changes how pointer input is interpreted; the inspector changes the selected layer's persistent properties.

## Navigate

- **Pan** moves the view without changing project geometry.
- The status bar controls zoom.
- **View -> Fit Window** or the `0` shortcut fits the active page.
- Selecting another page resets the canvas to a fitted view.

## Select and transform

Use **Select** to pick editable layers. Modifier keys allow multi-selection. Selection controls can move, resize, and rotate supported layers. A transform is previewed while dragging and committed when the gesture finishes; canceling restores the previous presentation.

## Add text

Use **Text** to create text presentation. A click creates point text, while a dragged frame creates a paragraph text box. Enter the translation in the inspector, then configure fitting and typography.

## Paint and erase

**Brush** paints an authored raster layer with the selected color and diameter. **Eraser** removes pixels from an authored raster target. These tools create durable artwork edits; they are different from the model-generated inpainting result.

## Sample a color

Use **Color picker** to sample the composited canvas. The chosen color can be applied to brush strokes or text styling without estimating the page color by eye.

## Manual cleanup

**Fast** (`J`) runs the selected bubble LaMa/AOT model over a painted stroke. **Inpainting (FLUX)** (`P`) uses its independent manual repair prompt and settings; **Decensor** (`D`) uses its separate censorship-reconstruction profile. Releasing a stroke queues its mask while drawing remains available. Pending masks stay visible until their jobs finish, fail, or stop. Temporary masks do not become paint layers or alter automatic lettering masks. See [Cleanup and inpainting](/workflow/cleanup-and-inpainting/).

## Keyboard focus

Single-key tool shortcuts are disabled while a text field is focused. See [Keyboard shortcuts](/reference/keyboard-shortcuts/) for defaults and customization.
