# Changelog

All notable changes to PPAI are documented here.

## [0.3.0] — 2026-03-15

### Added
- **Sketch verification mode** — fast probabilistic verification using random
  linear projections. O(k·d_in) per step instead of O(d_out·d_in). Default k=30
  gives ~10⁻⁹ miss probability.
- **Bias serialization in traces** — matmul steps with bias are now correctly
  recorded and verified in both JSON and NPZ formats.
- **NPZ roundtrip test** — dedicated test for binary trace serialization.
- **Qwen2.5-3B-Instruct validation** — verified lossless on 3B-parameter model
  (253 layers, 506 steps, 5/5 prompts identical) on NVIDIA L4 GPU.
- **Dual licensing** — AGPL-3.0 for open source, commercial license available.
- **`pyproject.toml`** — proper Python packaging with `pip install` support.
- **SPDX license headers** in every source file.
- **P matrix caching** — PPAILinear shares projection matrices across layers
  with identical specs, critical for lossless mode on large models.
- **TF32 guidance** — documented requirement to disable TF32 for GPU inference
  when traces will be CPU-verified.

### Fixed
- NPZ trace format now saves and loads `bias_values` for matmul steps.
  Previously, bias was lost during NPZ serialization, causing verification
  failures on models with biased attention projections (q_proj, k_proj, v_proj).

### Changed
- Version bumped to 0.3.0.
- Package renamed from `ppai_v2` to `ppai_v3` (publish-ready).
- README fully rewritten with accurate import paths, v3 features, and
  Qwen 3B verified results.

## [0.2.0] — 2026-03-14

### Added
- Sketch verification mode (initial implementation).
- P matrix caching in PPAILinear.
- Qwen2.5-3B-Instruct GCloud pipeline.

## [0.1.0] — 2026-03-12

### Added
- Initial release.
- PPT projection math (core/projection.py).
- SVD-guided angle optimization (core/optimize.py).
- Bit-exact activation specs (core/spec.py).
- PPAILinear drop-in layer replacement.
- Model conversion pipeline (compress/convert.py).
- Knowledge distillation calibration (compress/calibrate.py).
- Trace recording, JSON/NPZ serialization, independent verification.
- HuggingFace integration (convert, infer, extract weights).
- CLI (convert, verify, inspect).
- GPT-2 Small verified: 96/96 steps PASS, 5/5 prompts identical (lossless).
- 37 unit + integration tests.
