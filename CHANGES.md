# Changes

- Prepared the rc1 recipes with DFlash2, text-only defaults, optional vision presets from 24 GB, and planner-sized prompt caches.
- Added checksum-verified DFlash2 installation under the model directory, with the pinned Inco AI BF16 checkpoint as fallback.
- Sized the 16 GB headless recipe for FP8 DFlash2 and the 16 GB desktop recipe for plain decode with a smaller memory footprint.
- Selected FP8 KV across the presets and kept the BF16 KV override.

- Added prebuilt Linux TensorFold archives and a three-step install.
- Updated the card configs and calculator to account for TensorFold's CUDA memory reserve.
- Capped the recommended concurrency using the release guidance and measured throughput.
- Linked measured speed to the hardware, model branch and request count used in each test.
- Sized every config with TensorFold's planner (`tensorfold-native plan`), checked against real servers at the edge; the calculator now wraps the planner.
- Raised DGX Spark to eight requests; the 16 GB configs start with a 0.5 GiB engine reserve.
