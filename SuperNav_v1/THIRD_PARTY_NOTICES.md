# Third-party notices

SuperNav's project license is the [Project Registration License (PRL) v1.0](LICENSE).
The third-party code and assets below retain their original licenses. The project
license does not replace these notices or relicense those components.

## Meta-licensed source files — MIT

The following files retain their Meta copyright and MIT license headers:

- `src/supernav/backends/habitat/http_server.py`
- `src/supernav/backends/habitat/stdio_adapter.py`
- `src/supernav/backends/habitat/simulator/api.py`
- `src/supernav/backends/habitat/simulator/core.py`
- `src/supernav/backends/habitat/simulator/motion.py`
- `src/supernav/backends/habitat/simulator/session_scene.py`
- `src/supernav/backends/habitat/simulator/types.py`
- `src/supernav/backends/habitat/simulator/visual_media.py`

The MIT notice retained from the previous root license is reproduced below.

```text
MIT License

Copyright (c) Meta Platforms, Inc. and its affiliates.

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

## Bundled fonts — SIL Open Font License 1.1

The full copyright and license texts accompany the fonts at these source paths,
also preserved under `.dist-info/licenses/` in Python distributions:

- Space Grotesk: `src/supernav/web/static/fonts/SpaceGrotesk-OFL.txt`.
- IBM Plex Mono: `src/supernav/web/static/fonts/IBMPlexMono-OFL.txt`.
- The release documentation includes a [copy of the Space Grotesk license](docs/release/assets/SpaceGrotesk-OFL.txt).

## Go2 deployment and DimOS — Apache-2.0

The independent Go2 deployment under `robots/go2/` retains its
[license](robots/go2/LICENSE) and [notice](robots/go2/NOTICE). Its public
`dimensionalOS/dimos` submodule is pinned to
`79c9ed163ec6b1abd278a9161f0c74f4562f1231`. The overlays and patches include
modified and extracted DimOS code with original notices retained. The submodule
is assembled with those changes into a local runtime; upstream source remains
unmodified.

## External components

External simulator backends, datasets, model weights, and separately installed
dependencies retain their own licenses. Installing or using them does not change
their licensing terms.

## HabitatGS-Benchmark evaluation contract

`src/supernav/evaluation/objectnav/_habitatgs_v1.py` preserves the scoring functions
from [HabitatGS-Benchmark](https://huggingface.co/datasets/yyh929/HabitatGS-Benchmark),
revision `4a8fe5cf654c1d19f0c3c992720d3c4f57844f12`, `evaluation/score.py`
(SHA256 `37bb98a11141eae72fa68b464a8c1a9ea32773b83e74ac5745a4211992b6716e`).
Copyright (c) 2026 ZJU3DV Group at Zhejiang University. The release LICENSE is
Project Registration License (PRL) v1.0, reproduced in this repository’s LICENSE.
No scene assets or benchmark episode annotations are bundled here.
