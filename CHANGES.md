# Changes from the published recipes

- Replaced the TabbyAPI / ExLlamaV3 recommendations with TensorFold and FP8 KV plus bundled, batch-aware DSpark.
- Recomputed branch, parallelism and per-request contexts for desktop and non-desktop card classes; 4.0bpw desktop and all capacity limits are projections. Added the 1 GiB Spark desktop scenario.
- Moved historical TabbyAPI configs to `configs/legacy-exllamav3/`; the old helper scripts remain historical.
- Replaced the measurement file with TensorFold S1–S6 results; dropped the previous engine tables. Added upstream credits, build and launch instructions, and revised the separate model-card draft.
- Added 16 GB plain-decoding options: 106,496 tokens headless and 45,056 with a desktop reserve; the headless DSpark option remains available.
- These files are a local draft. Edge fits, several concurrency counts and the final engine commit remain unqualified.
