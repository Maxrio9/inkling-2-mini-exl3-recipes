# Inkling 2 Mini on your GPU

Text-only TabbyAPI recipes for [Inkling 2 Mini EXL3](https://huggingface.co/Maxrio/Inkling-2-Mini-exl3). [Engine](https://github.com/Maxrio9/exllamav3-inkling2) · [Measurements](data/MEASUREMENTS.md)

| GPU | Branch | KV | Pool without desktop | Pool with desktop | Measured decode / prefill tok/s (hardware) |
|---|---|---|---:|---:|---|
| 12 GB | No current branch fits | — | — | — | — |
| 16 GB | 3.3bpw | Q8 | [155,648](configs/tabby-16gb-no-dspark.yml) | [94,208](configs/tabby-16gb-desktop-no-dspark.yml) | 47.5 / 2,200 (RTX PRO 4500, 8k prompt) |
| 24 GB | 4.0bpw | Q8 | [483,328](configs/tabby-24gb-no-dspark.yml) | [417,792](configs/tabby-24gb-desktop-no-dspark.yml) | not yet measured on a desktop card with this engine |
| 32 GB | 4.0bpw | Q8 | [897,024](configs/tabby-32gb-no-dspark.yml) | [835,584](configs/tabby-32gb-desktop-no-dspark.yml) | not yet measured on a desktop card with this engine |
| DGX Spark | 4.0bpw | Q8 | [1,048,576](configs/tabby-dgx-spark-4.0bpw-q8.yml) | — | 26.7 / 745.9 (DGX Spark, 8k) |

Pool sizes are **projections** from measured Q8 memory, rounded down to 4,096 tokens and capped at 1,048,576. They have not been tested at their limits. The corresponding FP16 setups passed end-to-end five-fact retrieval checks at their limits; the 24/32 GB checks used the 4.75bpw branch. Desktop card classes were simulated on the RTX PRO 4500. Speeds are measured through TabbyAPI at an 8,192-token prompt, not at the pool limits: decode / prefill tok/s, median of three requests, one request at a time. With the 16-bit KV cache the same 16 GB setup decodes at 61.5 tok/s (2,230 tok/s prefill); the 8-bit cache trades that speed for about twice the context. Card classes are simulated with memory caps on an RTX PRO 4500; details are in [Measurements](data/MEASUREMENTS.md).

1. Install a matching CUDA toolkit and PyTorch, then build the [engine](https://github.com/Maxrio9/exllamav3-inkling2), choosing `12.0` (sm_120), `12.1` (sm_121), or `8.6` (sm_86):

   ```sh
   git clone https://github.com/Maxrio9/exllamav3-inkling2.git engine
   cd engine
   TORCH_CUDA_ARCH_LIST=12.0 pip install --no-build-isolation .
   cd ..
   ```

2. Check out TabbyAPI at `47281587fafbd69e50aaf77c7626f390811126b6`, apply the patch, and install its dependencies according to its README:

   ```sh
   git clone https://github.com/theroyallab/tabbyAPI.git tabbyAPI
   cd tabbyAPI
   git checkout 47281587fafbd69e50aaf77c7626f390811126b6
   git apply ../engine/patches/tabby-inkling2.patch
   ```

3. Download the model branch into a folder named `3.3bpw` or `4.0bpw` under your model directory. Copy the linked recipe to TabbyAPI's `config.yml`, then start from that checkout, replacing the example paths:

   ```sh
   cp /path/to/recipes/configs/tabby-24gb-no-dspark.yml config.yml
   PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 main.py --model-dir /path/to/models --host 127.0.0.1 --port 5027 --disable-auth true --web-ui false --allowed-origins http://127.0.0.1:5027
   ```

The DGX Spark limit is also a projection: the million-token FP16 check used 4.75bpw, not this Q8 config. Its tested launch used a PyTorch memory fraction of 0.55 and a 24 GiB host-memory guard. Ensure sufficient free host memory; pool includes prompt and output.

Serve one text request at a time. Image inputs have not been qualified for these recipes. DSpark is disabled: the engine refuses Q8 with DSpark, and DSpark showed no qualified prose speedup through TabbyAPI.
