# Changes

- Added prebuilt Linux TensorFold archives and a three-step install.
- Updated the card configs and calculator to account for TensorFold's CUDA memory reserve.
- Capped the recommended concurrency using the release guidance and measured throughput.
- Linked measured speed to the hardware, model branch and request count used in each test.
- Sized every config with TensorFold's planner (`tensorfold-native plan`), checked against real servers at the edge; the calculator now wraps the planner.
- Raised DGX Spark to eight requests; the 16 GB configs start with a 0.5 GiB engine reserve.
