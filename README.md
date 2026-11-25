# Inkling 2 Mini on your GPU

These TensorFold rc1 recipes run Inkling 2 Mini locally on NVIDIA GPUs through an OpenAI-compatible API. They pair Maxrio9's EXL3 weights with Inco AI's DFlash2 drafter. [EXL3 weights](https://huggingface.co/Maxrio/Inkling-2-Mini-exl3) · [Original DFlash2 checkpoint](https://huggingface.co/incoai/Inkling-2-Mini-DFlash2)

## Pick your card

Use the desktop row if this GPU also drives your display. Context includes both the prompt and the reply. Each row comes from TensorFold's planner using the target and the packed DFlash2 checkpoint. [Planner receipts](data/planner)

| GPU memory | Model branch | KV | Parallel requests | Tokens per request | DFlash2 | Vision | Prompt cache | Config |
|---|---|---|---:|---:|---|---|---:|---|
| 12 GB | No fit | — | — | — | — | — | — | — |
| 16 GB | `3.3bpw` | FP8 | 1 | 39,936 | On | Off | 0 GiB | [Config](configs/16gb.toml) |
| 16 GB, desktop | `3.3bpw` | FP8 | 1 | 40,960 | Off | Off | 0 GiB | [Config](configs/16gb-desktop.toml) |
| 16 GB, long context | `3.3bpw` | FP8 | 1 | 76,800 | Off | Off | 0 GiB | [Config](configs/16gb-long.toml) |
| 24 GB | `4.0bpw` | FP8 | 2 | 116,736 | On | Off | 2 GiB | [Config](configs/24gb.toml) |
| 24 GB, vision | `4.0bpw` | FP8 | 2 | 76,800 | On | On | 2 GiB | [Config](configs/24gb-vision.toml) |
| 24 GB, desktop | `4.0bpw` | FP8 | 1 | 184,320 | On | Off | 2 GiB | [Config](configs/24gb-desktop.toml) |
| 24 GB, desktop, vision | `4.0bpw` | FP8 | 1 | 108,544 | On | On | 2 GiB | [Config](configs/24gb-desktop-vision.toml) |
| 32 GB | `4.0bpw` | FP8 | 4 | 172,032 | On | Off | 2 GiB | [Config](configs/32gb.toml) |
| 32 GB, vision | `4.0bpw` | FP8 | 4 | 150,528 | On | On | 2 GiB | [Config](configs/32gb-vision.toml) |
| 32 GB, desktop | `4.0bpw` | FP8 | 4 | 156,672 | On | Off | 2 GiB | [Config](configs/32gb-desktop.toml) |
| 32 GB, desktop, vision | `4.0bpw` | FP8 | 4 | 135,168 | On | On | 2 GiB | [Config](configs/32gb-desktop-vision.toml) |
| DGX Spark | `4.0bpw` | FP8 | 8 | 609,280 | On | Off | 8 GiB | [Config](configs/dgx-spark.toml) |
| DGX Spark, vision | `4.0bpw` | FP8 | 8 | 598,016 | On | On | 8 GiB | [Config](configs/dgx-spark-vision.toml) |
| DGX Spark, desktop | `4.0bpw` | FP8 | 8 | 602,112 | On | Off | 8 GiB | [Config](configs/dgx-spark-desktop.toml) |
| DGX Spark, desktop, vision | `4.0bpw` | FP8 | 8 | 589,824 | On | On | 8 GiB | [Config](configs/dgx-spark-desktop-vision.toml) |

The desktop rows leave 1 GiB for the display session. All 16 GB rows use a 0.5 GiB engine reserve. The headless 16 GB default uses DFlash2. The 16 GB desktop recipe uses plain decode and disables the lazy-head proxy to fit its 40,960-token window. Use the 16 GB long-context row for longer documents.

Each text and vision pair uses the same stream count. The 24 GB desktop pair uses one stream because two streams with vision leave only 1.38 GiB for prompt caching at 65,536 tokens per request. The text-only 24 GB desktop plan can serve two streams at 87,040 tokens with a 2 GiB cache when selected with the calculator.

The planner keeps at least 65,536 tokens per request in every cached preset. It reserves 2 GiB on 24 GB and 32 GB cards and 8 GiB on Spark, then takes the largest context rounded down to 1,024 tokens. The launcher checks current free memory and adjusts context and cache before starting.

## Start

1. From the rc1 recipes checkout, run `./start.sh --engine-archive /path/to/rc1-archive.tar.gz`.
2. Once the server is ready, open [the model endpoint](http://127.0.0.1:8080/v1/models).

The launcher uses Linux, Python 3.11 or newer, and the NVIDIA driver. Use the prepared local archive for rc1. `./start.sh --dry-run` shows the selected recipe. Add `--vision` before `--` to select the matching vision preset on a 24 GB or larger card. To start offline, supply `--engine-archive FILE` and `--model-dir DIR`. Drafted recipes also need a verified `dflash2/` bundle in that model directory or `-- --drafter DIR`. Run `./start.sh --help` for card, display, and model overrides. Pass TensorFold flags after `--`, as in `./start.sh -- --port 8081`.

On Blackwell, the launcher checks the model repository's `dflash2-fp8` branch and verifies `SHA256SUMS` before installing the bundle. If the package is absent, it downloads Inco AI's pinned BF16 checkpoint and checks each file against a pinned SHA-256 digest. Older GPUs use that BF16 checkpoint directly. The launcher exposes the verified bundle under the model's `dflash2/` path and uses it for both planning and serving.

The vision presets need the CUDA 13 cuBLAS runtime so TensorFold can find `libcublasLt.so.13`. PNG input uses the built-in decoder. JPEG and WebP input need the matching libraries listed in the archive's `RUNTIME.md`.

### Manual install

You need Linux with glibc 2.28 or newer, an NVIDIA driver that supports your GPU and CUDA 13, and the [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/en/guides/cli).

1. **Verify and extract the prepared rc1 archive.** Use Linux x86_64 for RTX 30 (sm_86), RTX 40 (sm_89), RTX 50 and RTX PRO Blackwell (sm_120). Use Linux aarch64 for DGX Spark (GB10, sm_121). Check the archive against its accompanying SHA-256 digest.

   From the directory where you extracted the archive, enter its folder.

   ```sh
   cd tensorfold-1.3.4-linux-x86_64
   ```

   On DGX Spark, use the folder ending in `linux-aarch64`.

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

Install Zig 0.17.0 and a CUDA toolkit with `nvcc` that supports your GPU. From the prepared TensorFold rc1 source checkout, set `-Dgpu` to the architecture in step 1.

```sh
zig build -Dgpu=sm_120 -Dnvcc="$(command -v nvcc)"
```

Use `zig-out/native/bin/tensorfold-native` in step 3. The TensorFold checkout's `RUNBOOK.md` describes how to build the grammar library alongside the native binary.

## Speed you can expect

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
| `--parallel N` | Reserves memory for N simultaneous requests. The model supports up to 16. |
| `--kv-cache fp8\|bf16` | Sets the target KV cache format. BF16 uses more memory. |
| `--no-drafts` | Runs plain decoding. Remove `no-drafts = true` from the file to enable drafts. |
| `--drafter DIR` | Chooses a drafter directory. The launcher normally passes the installed `dflash2/` bundle. |
| `--prompt-cache-gib N` | Sets the memory cap for retained prompt state. The planner uses the free memory after loading all request slots. |
| `--vision` | Enables image input and reserves memory for the vision encoder and prompt rows. |
| `--host HOST`, `--port PORT` | Set the listen address and port. |
| `--name NAME` | Sets the model ID clients use. |

The rc1 CUDA backend supports automatic, required and named tool calls and schema-constrained replies through `response_format` with `type: "json_schema"`. Required and schema-constrained replies use the grammar library in the release archive.

### Make your own config

The calculator asks TensorFold's planner (`tensorfold-native plan`) and prints TOML. The planner reads the downloaded model branch and needs no GPU. It needs an engine build that includes the planner.

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
| `--prompt-cache`, `--no-prompt-cache` | Budgets 2 GiB on 24 GB and 32 GB cards and up to 8 GiB on Spark while keeping at least 65,536 tokens per request. |
| `--kv fp8\|bf16` | Chooses the target KV format. BF16 uses more memory. |
| `--reserve-gib R` | Sets the engine reserve, as `TENSORFOLD_MEMORY_RESERVE_GIB` would. |
| `--binary PATH`, `--json` | Picks the engine binary; prints the planner's full report with bytes per component. |

Use `tensorfold-native plan` to check a specific context against available memory. For this 24 GB example, supply the free MiB available at CUDA startup:

```sh
./bin/tensorfold-native plan ./inkling-mini --device-mib 24564 --free-mib 24229 \
  --parallel 2 --context 65536 --kv-cache fp8 --vision --drafter ./inkling-mini/dflash2
```

See the [measurement settings, results and memory model](data/MEASUREMENTS.md).
