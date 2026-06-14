# Mask templates — provenance

The mask template PNGs in this directory and their anchor points
(`anchors.json`) are vendored from the **MaskTheFace** project:

- Repository: https://github.com/aqeelanwar/MaskTheFace
- Paper: Anwar & Raychowdhury, *"Masked Face Recognition for Secure Authentication"*
  (arXiv:2008.11104) / *"A realistic approach to generate masked faces…"*
  (arXiv:2109.01745).
- License: **MIT** © 2020 Aqeel Anwar.

Vendored templates: `surgical`, `surgical_blue`, `surgical_green`, `cloth`,
`KN95`, `N95` (frontal templates only).

`anchors.json` holds, per template, the six MaskTheFace key points
(`a`–`f`) in template-pixel space:

```
        b (nose bridge, top-centre)
   a                     c            (upper cheeks)
   d                     f            (jaw sides)
        e (chin, bottom-centre)
```

The landmark mask-fitter (`src/face_occlusion/data/synthetic_mask_occluder.py`)
warps these points onto MediaPipe face landmarks. A few anchor coordinates in the
upstream `masks.cfg` slightly exceed the template bounds; the fitter clamps them.

## MIT License (MaskTheFace)

```
MIT License

Copyright (c) 2020 Aqeel Anwar

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
