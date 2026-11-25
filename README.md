# Inkling 2 Mini on your GPU

These TensorFold rc1 recipes run Inkling 2 Mini locally on NVIDIA GPUs through an OpenAI-compatible API. They pair Maxrio9's EXL3 weights with Inco AI's DFlash2 drafter. [EXL3 weights](https://huggingface.co/Maxrio/Inkling-2-Mini-exl3) · [Original DFlash2 checkpoint](https://huggingface.co/incoai/Inkling-2-Mini-DFlash2)

## Pick your card

Use the desktop row if this GPU also drives your display. The shared KV pool holds prompt and reply tokens for all active requests. One request can use most of the pool while other requests stay short. A request waits when its reservation exceeds the space available. The per-request cap bounds prompt plus reply length. All rows use FP8 KV. Each row comes from TensorFold's planner using the target weights and, where enabled, the packed DFlash2 checkpoint. [Planner receipts](data/planner)

<!-- planner-table:start -->
| GPU memory | Model branch | Parallel requests | KV pool, text | KV pool, with vision | Per-request cap (text / vision) | DFlash2 | Configs |
|---|---|---:|---:|---:|---:|---|---|
| 12 GB | No fit | — | — | — | — | — | — |
| 16 GB | `3.3bpw` | 1 | 40,192 | — | 39,936 | On | [Text](configs/16gb.toml) |
| 16 GB, desktop | `3.3bpw` | 1 | 40,192 | — | 39,936 | Off | [Text](configs/16gb-desktop.toml) |
| 16 GB, long context | `3.3bpw` | 1 | 77,056 | — | 76,800 | Off | [Text](configs/16gb-long.toml) |
| 24 GB | `4.0bpw` | 2 | 342,272 | 263,936 | 340,992 / 263,168 | On | [Text](configs/24gb.toml) · [Vision](configs/24gb-vision.toml) |
| 24 GB, desktop | `4.0bpw` | 2 | 283,136 | 206,592 | 282,624 / 205,824 | On | [Text](configs/24gb-desktop.toml) · [Vision](configs/24gb-desktop-vision.toml) |
| 32 GB | `4.0bpw` | 4 | 762,880 | 683,776 | 761,856 / 681,984 | On | [Text](configs/32gb.toml) · [Vision](configs/32gb-vision.toml) |
| 32 GB, desktop | `4.0bpw` | 4 | 705,280 | 624,896 | 703,488 / 623,616 | On | [Text](configs/32gb-desktop.toml) · [Vision](configs/32gb-desktop-vision.toml) |
| DGX Spark | `4.0bpw` | 8 | 5,348,608 | 5,257,728 | 1,048,576 / 1,048,576 | On | [Text](configs/dgx-spark.toml) · [Vision](configs/dgx-spark-vision.toml) |
| DGX Spark, desktop | `4.0bpw` | 8 | 5,285,120 | 5,194,240 | 1,048,576 / 1,048,576 | On | [Text](configs/dgx-spark-desktop.toml) · [Vision](configs/dgx-spark-desktop-vision.toml) |
<!-- planner-table:end -->

The desktop rows leave 1 GiB for the display session. All 16 GB rows use a 0.5 GiB engine reserve. The headless 16 GB default uses DFlash2. The 16 GB desktop recipe uses plain decode and disables the lazy-head proxy. The 16 GB long-context recipe uses plain decode to leave more room for KV.

Prompt caching is on by default. Repeated text and image prefixes can reuse the same global KV pages. When the engine retains a prefix, it takes additional pages from the same pool to save the convolution, sliding attention and drafter state needed to resume. The full pool is available to live requests, which can reclaim retained prefixes and their state in least-recently-used order. Use `--prompt-cache-gib N` to cap retention or set `--prompt-cache-gib 0` or `TF_KV_SHARE=0` to disable it.

The listed caps round the planner's maximum down to a multiple of 1,024 tokens. Each text and vision pair uses the same stream count. The launcher checks available memory before serving and lowers the selected cap if needed.

## Start

1. From the rc1 recipes checkout, run `./start.sh --engine-archive /path/to/paged-archive.tar.gz`.
2. Once the server is ready, open [the model endpoint](http://127.0.0.1:8080/v1/models).

The launcher uses Linux, Python 3.11 or newer, and the NVIDIA driver. Use the prepared local paged archive. `./start.sh --dry-run` shows the selected recipe. Add `--vision` before `--` to select the matching vision preset on a 24 GB or larger card. To start offline, supply `--engine-archive FILE` and `--model-dir DIR`. Drafted recipes also need a verified `dflash2/` bundle in that model directory or `-- --drafter DIR`. Run `./start.sh --help` for card, display, and model overrides. Pass TensorFold flags after `--`, as in `./start.sh -- --port 8081`.

On Blackwell, the launcher checks the 8-bit drafter repository [Maxrio/Inkling-2-Mini-DFlash2-FP8](https://huggingface.co/Maxrio/Inkling-2-Mini-DFlash2-FP8) and verifies `SHA256SUMS` before installing the bundle. If the package is absent, it downloads Inco AI's pinned BF16 checkpoint and checks each file against a pinned SHA-256 digest. Older GPUs use that BF16 checkpoint directly. The launcher exposes the verified bundle under the model's `dflash2/` path and uses it for both planning and serving.

The vision presets need the CUDA 13 cuBLAS runtime so TensorFold can find `libcublasLt.so.13`. PNG input uses the built-in decoder. JPEG and WebP input need the matching libraries listed in the archive's `RUNTIME.md`.

### Manual install

You need Linux with glibc 2.28 or newer, an NVIDIA driver that supports your GPU and CUDA 13, and the [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/en/guides/cli).

1. **Verify and extract the prepared paged archive.** Use Linux x86_64 for RTX 30 (sm_86), RTX 40 (sm_89), RTX 50 and RTX PRO Blackwell (sm_120). Use Linux aarch64 for DGX Spark (GB10, sm_121). Check the archive against its accompanying SHA-256 digest.

   From the directory where you extracted the archive, enter its folder.

   ```sh
   cd tensorfold-1.3.4-linux-x86_64
   ```

   On DGX Spark, use the folder ending in `linux-aarch64`. The shared-pool presets apply to sm_120 and sm_121. The x86_64 archive also includes the legacy CUDA path for sm_86 and sm_89. On those GPUs, set a context you have checked for the chosen stream count with `./start.sh -- --context N` or supply your own config.

2. **Download the model branch for your GPU.** This command uses the 24 GB example. Use the launcher to install the matching drafter, or place a verified DFlash2 bundle in `inkling-mini/dflash2`.

   ```sh
   hf download Maxrio/Inkling-2-Mini-exl3 --revision 4.0bpw --local-dir ./inkling-mini
   (cd inkling-mini && sha256sum --check --strict SHA256SUMS)
   ```

3. **Save the config from the table and start TensorFold.** Save the raw TOML in the current folder as `recipe.toml`.

   ```sh
   ./bin/tensorfold-native serve ./inkling-mini --config ./recipe.toml
   ```

   For the **16 GB** configs, prefix the command with `TENSORFOLD_MEMORY_RESERVE_GIB=0.5`. For the **16 GB desktop** config, also set `TF_INKLING2_LAZY_HEAD=0`. The other configs use the engine's default reserve. If startup reports insufficient memory, free memory or lower `context` in the TOML.

The OpenAI-compatible API is at `http://127.0.0.1:8080/v1`. Open `http://127.0.0.1:8080/v1/models` to see the model ID for your client.

### Build from source

Install Zig 0.17.0 and a CUDA toolkit with `nvcc` that supports your GPU. From the prepared TensorFold paged source checkout, set `-Dgpu` to the architecture in step 1.

```sh
zig build -Dgpu=sm_120 -Dnvcc="$(command -v nvcc)"
```

Use `zig-out/native/bin/tensorfold-native` in step 3. The TensorFold checkout's `RUNBOOK.md` describes how to build the grammar library alongside the native binary.

## RC1 speed measurements

The final paged build passed the 3% desktop throughput gate for single-request code, single-request prose, and four mixed requests. [Paged build results](data/MEASUREMENTS.md)

On RTX PRO 4500, the rc1 head-precision bracket served the 3.3bpw target with FP8 KV and FP8 DFlash2 layers. Each phase used three greedy repeats, 8,192 input tokens, 400 output tokens and disabled prompt caching.

| Workload | BF16 head control A | FP8 head | BF16 head control B | Pooled BF16 head median |
|---|---:|---:|---:|---:|
| Code | 186.49 | 184.28 | 177.58 | 181.02 |
| Prose | 99.71 | 100.48 | 98.98 | 99.34 |

Rates are median decode tokens per second. FP8 heads are the Blackwell default.

### DGX Spark comparison

The RC1 Spark tests used DFlash2 with FP8 layers and head, disabled prompt caching, 8,192 input tokens and 400 greedy output tokens per request. Each rate is the median of three runs in decode tokens per second. The 4- and 8-request rates aggregate a fixed mix of code and prose over their common decode interval.

| EXL3 branch | Target KV | 1 code | 1 prose | 4 mixed | 8 mixed |
|---|---|---:|---:|---:|---:|
| `3.3bpw` | FP8 | 86.59 | 47.15 | 75.97 | 97.18 |
| `4.0bpw` | FP8 | 85.49 | 50.29 | 75.63 | 96.38 |
| `3.3bpw` | BF16 | 84.78 | 51.06 | 78.69 | 97.72 |
| `4.0bpw` | BF16 | 87.38 | 48.46 | 76.11 | 97.15 |

The Spark recipe uses 4.0bpw with FP8 KV. At four and eight requests, the two FP8 quant branches were within 1% of each other. The 4.0bpw model had lower KL against the official checkpoint in the weight-release comparison (0.0084 vs 0.0121 nats). [EXL3 model card](https://huggingface.co/Maxrio/Inkling-2-Mini-exl3)

### V3 development measurements

These are development measurements from the V3 decode build. Each sample used FP8 KV, 8,192 input tokens, 400 output tokens, greedy sampling, and disabled prompt caching. Parallel figures count total decode tokens per second while all requests were decoding.

| GPU and branch | Requests | Plain | With DFlash2 |
|---|---|---:|---:|
| RTX PRO 4500 · `3.3bpw` | 1, code / prose | 70.64 / 70.67 | 175.60 / 97.67 |
| RTX PRO 4500 · `3.3bpw` | 2, code / prose | 111.70 / 111.56 | 131.65 / 117.80 |
| RTX PRO 4500 · `3.3bpw` | 4, code / prose | 160.10 / 160.21 | 159.20 / 158.81 |
| DGX Spark · `4.0bpw` | 1, code / prose | 38.75 / 38.67 | 86.68 / 49.24 |
| DGX Spark · `4.0bpw` | 4, code / prose | 77.95 / 78.93 | 81.89 / 80.04 |

The V3 desktop settings used FP8 drafter layers and a BF16 drafter head. The Spark development trial also used an FP8 drafter head. [Measurement settings and historical results](data/MEASUREMENTS.md)

## KV precision and quality

On the RTX PRO 4500, the teacher-forced gate compared the same 3.3bpw target weights over fixed token histories. The FP8 KV arm used the FP8 DFlash2 drafter and certified lazy head. The reference used BF16 KV, the BF16 drafter and the plain head.

| Prompt length | Prompts | Top-1 agreement | Mean KL (nats) |
|---|---:|---:|---:|
| 8,192 | 16 | 990 / 1,024 | 0.01247 |
| 32,768 | 4 | 251 / 256 | 0.00363 |

The previously identified 8k prompt reproduced a mean KL of 0.11368. The certified head matched the dense greedy winner at all 1,260 checked decode positions within the FP8 arm.

FP8 KV is the default on every card. The original 8k prompt accounts for most of the overall 0.01247 nats at 8k. The other 15 prompts average 0.00572 nats. Use `--kv-cache bf16` for the closest match to the unquantized cache. Expect about half the context when KV memory sets the limit. With the launcher, pass it as `./start.sh -- --kv-cache bf16`.

A 300-item paired IFBench run at the model’s default thinking effort scored 73.00% with FP8 KV and 72.33% with BF16 KV on the loose measure. The paired difference was +0.67 percentage points (95% interval −3.67 to +5.00), with no detectable loss. Observed aggregate output rates were 175.98 tok/s with FP8 and 181.95 tok/s with BF16. [Evaluation details](data/MEASUREMENTS.md).

## Good to know

- DFlash2 sped up code more than prose in the single-request tests.
- Parallel requests share the GPU's generation speed.
- The vision-enabled configs accept image input through the OpenAI-compatible chat endpoint.
- The card table estimates context from measured memory use, and the exact limits depend on free memory and the driver.

## Credits

Inkling 2 Mini is by [Thinking Machines](https://huggingface.co/thinkingmachines/Inkling-2-Mini). EXL3 weights and recipes are by [Maxrio9](https://github.com/Maxrio9). TensorFold is by [@ashhart](https://github.com/ashhart) and contributors.

These contributors supplied upstream work used in the port.

- Inco AI supplied the DFlash2 design and Mini checkpoint. [DFlash2](https://inco.ai/blog/dflash2/) · [Checkpoint](https://huggingface.co/incoai/Inkling-2-Mini-DFlash2)
- z-lab, Jian Chen, Yesheng Liang and Zhijian Liu supplied the original DFlash implementation. [DFlash](https://github.com/z-lab/dflash)
- MiaAI-Lab supplied the structured-output, draft storage, lazy head, router, attention prefetch, CUDA graph, prefill and vision patches adapted for these recipes. [MiaAI-Lab](https://github.com/MiaAI-Lab)
- @francip — [#3612](https://github.com/ashhart/TensorFold/pull/3612)
- @ajensenwaud — [#4127](https://github.com/ashhart/TensorFold/pull/4127), [#4128](https://github.com/ashhart/TensorFold/pull/4128), [#4746](https://github.com/ashhart/TensorFold/pull/4746), [#4841](https://github.com/ashhart/TensorFold/pull/4841), [#4124](https://github.com/ashhart/TensorFold/pull/4124)
- @jschmied (Jürgen Schmied) — [#4697](https://github.com/ashhart/TensorFold/pull/4697)
- @jschmied and @ashhart — [#4786](https://github.com/ashhart/TensorFold/pull/4786)
- @BlivionIaG — [#4337](https://github.com/ashhart/TensorFold/pull/4337), [#4704](https://github.com/ashhart/TensorFold/pull/4704)
- @grantoverton — [#4471](https://github.com/ashhart/TensorFold/pull/4471)
- @cwschroeder — [#4784](https://github.com/ashhart/TensorFold/pull/4784)

## For agents and advanced use

| Serve flag | What it does |
|---|---|
| `--config FILE` | Reads a TOML config. Command-line values override the file. |
| `--context N` | Sets the prompt-plus-output limit for each request. |
| `--parallel N` | Sets the concurrency limit. The model supports up to 16 requests sharing the pool. |
| `--kv-cache fp8\|bf16` | Sets the target KV cache format. BF16 uses more memory. |
| `--no-drafts` | Runs plain decoding. Remove `no-drafts = true` from the file to enable drafts. |
| `--drafter DIR` | Chooses a drafter directory. The launcher normally passes the installed `dflash2/` bundle. |
| `--prompt-cache-gib N` | Optionally caps the pool pages held for prefixes and their restart state. Zero disables prefix caching. |
| `--vision` | Enables image input and reserves memory for the vision encoder and prompt rows. |
| `--host HOST`, `--port PORT` | Set the listen address and port. |
| `--name NAME` | Sets the model ID clients use. |

The rc1 CUDA backend supports automatic, required and named tool calls and schema-constrained replies through `response_format` with `type: "json_schema"`. Required and schema-constrained replies use the grammar library in the release archive.

### Make your own config

The calculator asks TensorFold's planner (`tensorfold-native plan`) and prints TOML. The planner reads the downloaded model branch and needs no GPU. Use the paged Mini build of the planner.

```sh
python3 tools/recipes.py --card 24 --model ./inkling-mini --drafter ./inkling-mini/dflash2 > recipe.toml
```

| Calculator flag | What it does |
|---|---|
| `--card 16\|24\|32\|spark` | Chooses the card class, its default branch, request count and engine reserve. |
| `--model DIR` | The downloaded model branch whose checkpoint the planner reads. |
| `--desktop` | Leaves 1 GiB for a desktop session on the same GPU. |
| `--parallel N` | Sets the number of simultaneous requests, up to 16. |
| `--drafter DIR`, `--no-drafts` | Plans with DFlash2 or plain decode. |
| `--draft-min-context N` | Plans plain decode if drafting fits fewer than N tokens per request. |
| `--plain-head` | Budgets the output head without the lazy-head proxy, as in the 16 GB desktop recipe. |
| `--vision`, `--no-vision` | Includes or excludes vision memory. Vision is off by default and available from 24 GB up. |
| `--prompt-cache`, `--no-prompt-cache` | Uses the engine's default shared prefix cache or disables it. |
| `--prompt-cache-gib N` | Caps the pool pages held for prefixes and their restart state. Zero disables caching. |
| `--kv fp8\|bf16` | Chooses the target KV format. BF16 uses more memory. |
| `--reserve-gib R` | Sets the engine reserve, as `TENSORFOLD_MEMORY_RESERVE_GIB` would. |
| `--binary PATH`, `--json` | Picks the engine binary; prints the planner's full report with bytes per component. |

Use `tensorfold-native plan` to check a specific context against available memory. For this 24 GB example, supply the free MiB available at CUDA startup:

```sh
./bin/tensorfold-native plan ./inkling-mini --device-mib 24564 --free-mib 24229 \
  --parallel 2 --context 65536 --kv-cache fp8 --vision --drafter ./inkling-mini/dflash2
```

See the [measurement settings, results and memory model](data/MEASUREMENTS.md).
