# Inkling 2 Mini on your GPU

These TensorFold recipes run Inkling 2 Mini locally on NVIDIA GPUs. They use [our EXL3 quants](https://huggingface.co/Maxrio/Inkling-2-Mini-exl3) and serve text through an OpenAI-compatible API.

## Pick your card

Use the desktop row if this GPU also drives your display. Context is per request and includes both the prompt and the reply. TensorFold's own planner sized every row from the model branch's checkpoint, and real servers started at the planner's limit on emulated 16 GB and 24 GB cards and refused one token above it.

| GPU memory | Model branch | Parallel requests | Tokens per request | DSpark | Engine reserve | Config |
|---|---|---:|---:|---|---|---|
| 12 GB | No fit | — | — | — | — | — |
| 16 GB | `3.3bpw` | 1 | 102,400 | Off | 0.5 GiB | [Long context](configs/16gb-long.toml) |
| 16 GB | `3.3bpw` | 1 | 46,080 | On | 0.5 GiB | [Faster](configs/16gb.toml) |
| 16 GB, desktop | `3.3bpw` | 1 | 41,984 | Off | 0.5 GiB | [Desktop](configs/16gb-desktop.toml) |
| 24 GB | `4.0bpw` | 2 | 151,552 | On | default | [Config](configs/24gb.toml) |
| 24 GB, desktop | `4.0bpw` | 2 | 126,976 | On | default | [Desktop](configs/24gb-desktop.toml) |
| 32 GB | `4.0bpw` | 4 | 168,960 | On | default | [Config](configs/32gb.toml) |
| 32 GB, desktop | `4.0bpw` | 4 | 155,648 | On | default | [Desktop](configs/32gb-desktop.toml) |
| DGX Spark | `4.0bpw` | 8 | 544,768 | On | default | [Config](configs/dgx-spark.toml) |
| DGX Spark, desktop | `4.0bpw` | 8 | 538,624 | On | default | [Desktop](configs/dgx-spark-desktop.toml) |

The request counts keep a long context per request. We timed up to eight concurrent requests on the RTX PRO 4500 and four on DGX Spark, where eight is a cap until a larger run exists. More requests fit at the cost of context per request; size them with the calculator below. A 16 GB card with DSpark on holds one request of about 46k tokens; for longer prompts use the long-context row. The desktop rows leave 1 GiB for the display session. The launcher checks free memory at startup and lowers context if needed.

## Install

1. `git clone https://github.com/Maxrio9/inkling-2-mini-exl3-recipes.git`
2. `cd inkling-2-mini-exl3-recipes && ./start.sh`
3. Once the server is ready, open [the model endpoint](http://127.0.0.1:8080/v1/models).

The launcher uses Linux, Python 3.11 or newer, and the NVIDIA driver. Run `./start.sh` each time to check for updates and start the server. `./start.sh --dry-run` shows the selected recipe. To start offline, supply both `--engine-archive FILE` and `--model-dir DIR`. Run `./start.sh --help` for card, display, and model overrides. Pass TensorFold flags after `--`, as in `./start.sh -- --port 8081`.

### Manual install

You need Linux with glibc 2.28 or newer, an NVIDIA driver that supports your GPU and CUDA 13, and the [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/en/guides/cli).

1. **Download and extract TensorFold.** Choose [Linux x86_64](https://github.com/Maxrio9/TensorFold/releases/download/inkling2-mini-exl3-v1/tensorfold-1.3.4-linux-x86_64.tar.gz) for RTX 30 (sm_86), RTX 40 (sm_89) or RTX 50 and RTX PRO Blackwell (sm_120). Choose [Linux aarch64](https://github.com/Maxrio9/TensorFold/releases/download/inkling2-mini-exl3-v1/tensorfold-1.3.4-linux-aarch64.tar.gz) for DGX Spark (GB10, sm_121). The x86_64 build was smoke-tested on RTX PRO 4500. The two older GPU architectures have build coverage. [Checksums](https://github.com/Maxrio9/TensorFold/releases/download/inkling2-mini-exl3-v1/SHA256SUMS)

   From the directory where you extracted the archive, enter its folder.

   ```sh
   cd tensorfold-1.3.4-linux-x86_64
   ```

   On DGX Spark, use the folder ending in `linux-aarch64`.

2. **Download the model branch for your GPU.** This command uses the 24 GB example and includes the bundled DSpark file.

   ```sh
   hf download Maxrio/Inkling-2-Mini-exl3 --revision 4.0bpw --local-dir ./inkling-mini
   ```

3. **Save the config from the table and start TensorFold.** Save the raw TOML in the current folder as `recipe.toml`.

   ```sh
   ./bin/tensorfold-native serve ./inkling-mini --config ./recipe.toml
   ```

   For the **16 GB** configs, prefix that command with `TENSORFOLD_MEMORY_RESERVE_GIB=0.5`. The other configs use the engine's default reserve. If startup reports insufficient memory, another program holds GPU memory: close it or lower `context` in the TOML.

The OpenAI-compatible API is at `http://127.0.0.1:8080/v1`. Open `http://127.0.0.1:8080/v1/models` to see the model ID for your client.

### Build from source

Install Zig 0.17.0 and a CUDA toolkit with `nvcc` that supports your GPU. The archives used CUDA 13.4.92. Set `-Dgpu` to match the architecture in step 1.

```sh
git clone --branch inkling2-mini-exl3 https://github.com/Maxrio9/TensorFold.git
cd TensorFold
git checkout 664053f8f7cc9132985fc749bd3261519511b536
zig build -Dgpu=sm_120 -Dnvcc="$(command -v nvcc)"
```

Use `zig-out/native/bin/tensorfold-native` in step 3. [TensorFold runbook](https://github.com/Maxrio9/TensorFold/blob/664053f8f7cc9132985fc749bd3261519511b536/RUNBOOK.md)

## Speed you can expect

These are measured decode rates from the development builds on the way to this release. Each run used FP8 KV, 8,192 input tokens, 400 output tokens, greedy sampling, and an empty prompt cache. Values are medians of three repeats, in tokens per second.

| GPU and branch | Requests | Plain | With DSpark |
|---|---|---:|---:|
| RTX PRO 4500 · `3.3bpw` | 1 code / 1 prose, run separately | 66.56 / 66.44 | 126.68 / 81.77 |
| RTX PRO 4500 · `3.3bpw` | 4 mixed, total while all decode | 140.32 / 140.54¹ | 140.14 |
| DGX Spark · `4.0bpw` | 1 code / 1 prose, run separately | 36.31 / 36.31 | 59.20 / 42.16 |
| DGX Spark · `4.0bpw` | 4 mixed, total while all decode | 74.41 | — |

¹ Plain runs before and after the DSpark run. [Full test results](data/MEASUREMENTS.md)

## Good to know

- TensorFold uses FP8 for the model's KV cache by default.
- DSpark sped up code more than prose in the single-request tests.
- Parallel requests share the GPU's generation speed.
- This engine serves text for Inkling 2 Mini.
- The card table estimates context from measured memory use, and the exact limits depend on free memory and the driver.

## Credits

Inkling 2 Mini is by [Thinking Machines](https://huggingface.co/thinkingmachines/Inkling-2-Mini). EXL3 weights and recipes are by [Maxrio9](https://github.com/Maxrio9). TensorFold is by [@ashhart](https://github.com/ashhart) and contributors.

These contributors supplied upstream work used in the port.

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
| `--no-drafts` | Skips DSpark. If the file already sets it, remove that line to turn DSpark back on. |
| `--drafter DIR` | Chooses a drafter directory. The bundled file is picked up automatically when drafting is on. |
| `--prompt-cache-gib N` | Sets the memory cap for retaining prompt state between requests. The recipes set it to zero. |
| `--host HOST`, `--port PORT` | Set the listen address and port. |
| `--name NAME` | Sets the model ID clients use. |

The release archives include the grammar library. Tool calling works (the model picks tools with `tool_choice` "auto", and multi-turn tool results are handled). Forcing a tool (`tool_choice` "required") and schema-constrained replies (`response_format` json_schema) are not supported yet on the Inkling 2 Mini CUDA backend. [Source build instructions](https://github.com/Maxrio9/TensorFold/blob/664053f8f7cc9132985fc749bd3261519511b536/RUNBOOK.md)

### Make your own config

The calculator asks TensorFold's planner (`tensorfold-native plan`) and prints TOML. The planner reads the downloaded model branch and needs no GPU. It needs an engine build that includes the planner.

```sh
python3 tools/recipes.py --card 24 --parallel 3 --model ./inkling-mini > recipe.toml
```

| Calculator flag | What it does |
|---|---|
| `--card 16\|24\|32\|spark` | Chooses the card class, its default branch, request count and engine reserve. |
| `--model DIR` | The downloaded model branch whose checkpoint the planner reads. |
| `--desktop` | Leaves 1 GiB for a desktop session on the same GPU. |
| `--parallel N` | Sets the number of simultaneous requests, up to 16. |
| `--no-drafts` | Plans plain decoding without DSpark, which frees memory for context. |
| `--kv fp8\|bf16` | Chooses the target KV format. BF16 uses more memory. |
| `--reserve-gib R` | Sets the engine reserve, as `TENSORFOLD_MEMORY_RESERVE_GIB` would. |
| `--binary PATH`, `--json` | Picks the engine binary; prints the planner's full report with bytes per component. |

Use `tensorfold-native plan` to check a specific context against available memory. For this 24 GB example, supply the free MiB available at CUDA startup:

```sh
./bin/tensorfold-native plan ./inkling-mini --device-mib 24564 --free-mib 24229 --parallel 2 --context 151552
```

See the [measurement settings, results and memory model](data/MEASUREMENTS.md).
