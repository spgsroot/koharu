---
title: Keyboard Shortcuts
description: Default canvas, history, selection, and gesture shortcuts.
---

# Keyboard Shortcuts

Single-key tool shortcuts work when a project is open and a text input, textarea, or editable text region is not focused.

## Tools

| Action | Default |
| --- | --- |
| Select | `V` |
| Text | `T` |
| Brush | `B` |
| Eraser | `E` |
| Color picker | `I` |
| Fast | `J` |
| Inpainting (FLUX) | `P` |
| Decensor | `D` |
| Pan | `H` |
| Fit Window | `0` |

Change these characters in **Settings -> Shortcuts**.

## Navigation and gestures

| Action | Shortcut |
| --- | --- |
| Temporarily pan | Hold `Space` and drag |
| Cancel the current gesture or clear selection | `Escape` |
| Fit active page | `0` by default |

Releasing Space restores the previously selected tool. Switching away from the application cancels an unfinished canvas gesture.

## Editing

| Action | Windows/Linux | macOS |
| --- | --- | --- |
| Undo | `Ctrl+Z` | `Command+Z` |
| Redo | `Ctrl+Shift+Z` or `Ctrl+Y` | `Command+Shift+Z` |
| Select all editable layers on the page | `Ctrl+A` | `Command+A` |
| Delete selected layers | `Delete` or `Backspace` | `Delete` or `Backspace` |

Undo and redo commit the focused text draft before navigating document history. Other editing shortcuts remain inactive while an editable text field is focused.

## Shortcut conflicts

The shortcut editor accepts one character per tool. Avoid assigning the same character to multiple tools: the first matching tool in the toolbar order wins. Appearance and tool shortcuts are desktop UI settings and do not change exported pages.
