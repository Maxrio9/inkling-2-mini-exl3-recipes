# Inkling 2 Mini on your GPU

Text-only TensorFold recipes for [our Inkling 2 Mini EXL3 quants](https://huggingface.co/Maxrio/Inkling-2-Mini-exl3). They use FP8 KV, with the bundled DSpark drafter enabled where memory allows. [Engine branch](https://github.com/Maxrio9/TensorFold/tree/inkling2-mini-exl3) · [Measurements and calculations](data/MEASUREMENTS.md)

| Card | Branch | Desktop reserve | Parallel requests | Context per request | DSpark | Measured decode |
|---|---|---|---:|---:|---|---|
| 12 GB | No branch fits | — | — | — | — | — |
| 16 GB | 3.3bpw | None | 1 | [53,248](configs/16gb.toml) | On | RTX PRO 4500, 3.3bpw, c=1: code 126.7, prose 81.8, plain 66.5 tok/s |
| 16 GB | 3.3bpw | None | 1 | [106,496](configs/16gb-long.toml) | Off | Nearest: RTX PRO 4500, 3.3bpw, c=1, plain 66.5 tok/s |
| 16 GB | 3.3bpw | 1 GiB | 1 | [45,056](configs/16gb-desktop.toml) | Off | Nearest: RTX PRO 4500, 3.3bpw, c=1, plain 66.5 tok/s |
| 24 GB | 4.0bpw | None | 4 | [69,632](configs/24gb.toml) | On | Nearest: RTX PRO 4500, 3.3bpw, c=4, DSpark on: 140 aggregate tok/s |
| 24 GB | 4.0bpw | 1 GiB | 6 | [32,768](configs/24gb-desktop.toml) | On | Nearest: RTX PRO 4500, 3.3bpw, c=4, DSpark on: 140 aggregate tok/s |
| 32 GB | 4.0bpw | None | 9 | [65,536](configs/32gb.toml) | On | Nearest: RTX PRO 4500, 3.3bpw, c=4, DSpark on: 140 aggregate tok/s |
| 32 GB | 4.0bpw | 1 GiB | 14 | [32,768](configs/32gb-desktop.toml) | On | Nearest: RTX PRO 4500, 3.3bpw, c=4, DSpark on: 140 aggregate tok/s |
| DGX Spark | 4.0bpw | None | 16 | [221,184](configs/dgx-spark.toml) | On | DGX Spark, 4.0bpw, c=1: code 59.2, prose 42.2, plain 36.3; c=4 plain 74.4 aggregate tok/s |
| DGX Spark | 4.0bpw | 1 GiB | 16 | [217,088](configs/dgx-spark-desktop.toml) | On | Same measurement as above |

**Every capacity is computed**, including the 4.0bpw desktop estimates derived from measured 3.3bpw desktop allocations and the measured weight difference. Desktop card sizes use nominal GiB budgets; actual usable memory varies. On 16 GB, DSpark needs the headless setup; choose the plain configs for more context or a desktop session. The Spark projection assumes 116.48 GiB initially available, a 24 GiB system reserve and a 100 GiB allocation cap. Context includes prompt and output. These projected limits and the listed concurrency settings have not been tested at their edges. Concurrency above eight has not been tested; Spark c=4 was tested with drafting disabled. Speed numbers are measured at 8,192 prompt tokens, 400 output tokens, greedy sampling, no prompt cache, median of three; aggregate is shared decode, not per-request speed. They do not predict performance at the listed context limits.

1. Install Zig 0.17.0 and a CUDA toolkit with `nvcc` (the tests used CUDA 13.4.92; drivers 615.78.08 on RTX PRO 4500 and 595.99.02 on DGX Spark). Build the branch on the target machine:

   ```sh
   git clone --branch inkling2-mini-exl3 https://github.com/Maxrio9/TensorFold.git
   cd TensorFold
   git checkout 70d6031b52d158e0ce21eb4b54f454ce714370e8
   zig build -Dgpu=sm_120 -Dnvcc=/usr/local/cuda/bin/nvcc
   ```

   On DGX Spark, use `zig build -Dgpu=sm_121 -Dnvcc=/usr/local/cuda/bin/nvcc` instead. The binary is `zig-out/native/bin/tensorfold-native`. See the engine's `RUNBOOK.md` for prerequisites and optional structured-output support. 
2. Download the appropriate model branch into a local directory, including its bundled drafter. Clone this recipes repository and run, replacing both example paths:

   ```sh
   zig-out/native/bin/tensorfold-native serve /path/to/model --config /path/to/recipes/configs/24gb.toml
   ```

   TensorFold automatically loads the bundled drafter except in the two 16 GB plain configs, which set `no-drafts = true`. `--no-drafts` turns it off; `--kv-cache bf16` selects BF16 KV. Each config disables retained prompt caching and sets the per-request window and maximum parallel requests. Check startup admission and lower context or concurrency if necessary.

DSpark's gain depends on content: the measured code gain was large, the prose gain smaller, and at four parallel desktop requests there was essentially no gain. The engine currently serves text only for this model. Five-fact retrieval returned 5/5 with a 130,808-token prompt on RTX PRO 4500 and a 262,144-token prompt on DGX Spark; these are individual checks, not validation of every projected limit. See [measurements](data/MEASUREMENTS.md) for settings and limitations. The [legacy configs](configs/legacy-exllamav3/) and `tools/` belong to the previous ExLlamaV3 release.

## Credits

TensorFold by [@ashhart](https://github.com/ashhart) and contributors; model by [Thinking Machines](https://huggingface.co/thinkingmachines/Inkling-2-Mini). This engine uses upstream TensorFold work from [#3612](https://github.com/ashhart/TensorFold/pull/3612) @francip; [#4127](https://github.com/ashhart/TensorFold/pull/4127), [#4128](https://github.com/ashhart/TensorFold/pull/4128), [#4746](https://github.com/ashhart/TensorFold/pull/4746), [#4841](https://github.com/ashhart/TensorFold/pull/4841), [#4124](https://github.com/ashhart/TensorFold/pull/4124) @ajensenwaud; [#4697](https://github.com/ashhart/TensorFold/pull/4697) @jschmied (Jürgen Schmied); [#4786](https://github.com/ashhart/TensorFold/pull/4786) @jschmied and @ashhart; [#4337](https://github.com/ashhart/TensorFold/pull/4337), [#4704](https://github.com/ashhart/TensorFold/pull/4704) @BlivionIaG; [#4471](https://github.com/ashhart/TensorFold/pull/4471) @grantoverton; and [#4784](https://github.com/ashhart/TensorFold/pull/4784) @cwschroeder.
