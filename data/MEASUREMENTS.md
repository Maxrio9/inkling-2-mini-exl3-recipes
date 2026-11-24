# TensorFold measurements

Slices S1–S6 used development builds of TensorFold leading to commit `70d6031b52d158e0ce21eb4b54f454ce714370e8` on `inkling2-mini-exl3`. S1 was based on `42bbc8e35`, S4 on `a1c1f8d53` and S6 on `cb54db840`. The release archive was checked with a short served request on RTX PRO 4500. RTX PRO 4500 Blackwell used sm_120 and NVIDIA driver 615.78.08. DGX Spark GB10 used sm_121 and driver 595.99.02. The benchmark builds used Zig 0.17.0 and CUDA nvcc 13.4.92. Decode rates exclude the first generated token.

## S1: initial server

RTX PRO 4500, 3.3bpw, one stream, BF16 KV, no drafts, no retained prompt cache, greedy, 128-token output cap. Rates and times are medians of three.

| Measure | Result |
|---|---:|
| 2,048 prompt decode | 66.57 tok/s |
| 8,192 prompt decode | 66.49 tok/s |
| Cold TTFT, 2,048 / 8,192 prompt | 0.778 / 3.193 s |

## S2: graph and BF16 KV drafter

RTX PRO 4500, 3.3bpw, one stream, BF16 KV, 8,192 input tokens, 400 output tokens, warm server and zero cached prompt tokens. Rates are medians of three. The requests set reasoning effort to `none` and the output still included reasoning.

| Workload | Temperature | Plain tok/s | DSpark tok/s | Accepted/proposed |
|---|---:|---:|---:|---:|
| Code | 0 | 66.80 | 122.14 | 826/1,613 |
| Code | 0.6 | 67.19 | 105.03 | 776/1,781 |
| Prose | 0 | 66.91 | 82.47 | 558/1,506 |
| Prose | 0.6 | 67.60 | 73.85 | 480/1,363 |

At a 2,048-token prompt and 128-token output cap, graph candidate decode was 68.07 tok/s versus 66.88 and 66.83 for bracketing controls. With BF16 KV, DSpark verification top-1 matched its plain target at all 4,408 positions tested across widths 1–8, with zero measured KL.

## S3: FP8 KV

RTX PRO 4500, 3.3bpw, one stream, 8,192 input tokens, 400 output tokens, greedy sampling, warm server, zero cached prompt tokens and DSpark. Rates are medians of three.

| Workload | BF16 control A | FP8 | BF16 control B | FP8 accepted/proposed |
|---|---:|---:|---:|---:|
| Code | 118.35 | 123.64 | 117.62 | 847/1,546 |
| Prose | 79.88 | 78.33 | 79.82 | 576/1,555 |

FP8 versus BF16 KV teacher-forced top-1 matched 1,080/1,088 positions (99.26%); mean KL was 0.008890 nats. FP8 DSpark verification matched 4,408/4,408 plain FP8 positions, with zero measured KL. Changing the target KV format from BF16 to FP8 changed the greedy completions on both timing prompts.

## S4: retrieval

RTX PRO 4500, 3.3bpw, one stream, DSpark, zero cached prompt tokens and requested effort `none`. Each row is one five-fact test that finished normally with a 256-token output cap.

| KV | Prompt tokens | Correct facts | Output tokens | Cold TTFT (s) | Decode tok/s | Sampled process peak MiB |
|---|---:|---:|---:|---:|---:|---:|
| FP8 | 32,502 | 5/5 | 191 | 13.975 | 140.72 | 17,472 |
| FP8 | 130,808 | 5/5 | 208 | 62.913 | 98.93 | 17,472 |
| BF16 | 130,808 | 5/5 | 203 | 60.745 | 95.27 | 19,688 |

The 131,072-token server window was preallocated for all rows. The 128-token output cap initially exhausted itself in reasoning; a 256-token cap yielded final answers.

## S5: concurrency before batch-aware policy

RTX PRO 4500, 3.3bpw, FP8 KV, 8,704-token window, eight allocated slots, 8,192-token input, 400-token output, greedy sampling, zero cached prompt tokens and three repeats. Shared aggregate measures the interval when all requests decode. Delivery aggregate includes prefill overlap and the time after the first request finishes.

| Concurrent | Mode | Shared aggregate tok/s | Delivery aggregate tok/s |
|---:|---|---:|---:|
| 2 | Plain / DSpark | 99.29 / 130.08 | 69.95 / 76.87 |
| 4 | Plain / DSpark | 140.11 / 133.02 | 73.97 / 69.72 |
| 8 | Plain / DSpark | 177.52 / 172.11 | 76.22 / 73.23 |

All 84 concurrent completions matched their solo control. Four-stream shared verification matched 256/256 width-one and 280/280 mixed-width positions to solo FP8 controls, zero measured KL.

## S6: batch-aware policy and DGX Spark

Both machines used the S5 prompts, 8,192 input tokens, 400 output tokens, greedy sampling, zero cached prompt tokens and three repeats. RTX PRO 4500 used 3.3bpw with eight allocated slots. DGX Spark used 4.0bpw with four allocated slots. Both used FP8 KV and an 8,704-token window. The four-request Spark run used plain decoding. Rates are shared decode medians.

| Hardware | Concurrency, workload | Plain tok/s | DSpark tok/s |
|---|---|---:|---:|
| RTX PRO 4500 | 1, code | 66.56 | 126.68 |
| RTX PRO 4500 | 1, prose | 66.44 | 81.77 |
| RTX PRO 4500 | 2, aggregate | 99.54 / 99.52 controls | 101.06 |
| RTX PRO 4500 | 4, aggregate | 140.32 / 140.54 controls | 140.14 |
| DGX Spark | 1, code | 36.31 | 59.20 |
| DGX Spark | 1, prose | 36.31 | 42.16 |
| DGX Spark | 4, mixed aggregate | 74.41 | Not tested |

The Spark c=4 delivery aggregate was 26.07 tok/s. All 144 desktop and 24 Spark timed completions matched their corresponding target token IDs. A single Spark retrieval test with a 262,144-token prompt and 227 output tokens returned 5/5 facts, cold TTFT 512.36 s. Spark peak process RSS was 29.37 GiB at 8,704 × 4 and 29.49 GiB at 262,656 × 1; minimum system available memory was 97.67 and 93.15 GiB respectively. These RSS figures include loading and measure process memory on the unified-memory Spark.

## Capacity: TensorFold's planner

Every config's context comes from `tensorfold-native plan`, recorded per config in [planner/](planner/). The planner reads the model branch's checkpoint headers and runs the engine's own sizing and startup admission without a GPU: target and drafter weights, working buffers, KV cache and per-request state, then the weight, first-request and parallel-request checks the server makes at startup. It reports bytes per component and the largest admitted context per request.

| Planner input | Value |
|---|---|
| Card memory | 16,303 / 24,564 / 32,607 MiB (the smallest common card of each class, as `nvidia-smi` reports it) |
| DGX Spark | 124,608 MiB total, 119,280 MiB available before launch (measured on an idle Spark), unified |
| Driver and CUDA context outside the engine's allocations | 335 MiB (measured on the RTX PRO 4500: 290 to 361 MiB at peak) |
| Desktop rows | 1 GiB less free memory at startup |
| Engine reserve | 0.5 GiB on 16 GB cards (the engine's minimum); default elsewhere: 1 GiB discrete, 12.2 GiB on Spark |
| KV, prompt cache | FP8, none |
| Rounding | Down to a multiple of 1,024 tokens |

### Checks against real servers

On the RTX PRO 4500 with the 3.3bpw branch, `TENSORFOLD_CUDA_MEMORY_LIMIT_GB` emulated the startup budget of a 16 GB card (14.59 GiB) and a 24 GB card (22.66 GiB). In every case the server started at the planner's maximum, served a request filling the window, and refused to start one token above it. The engine's tracked allocation matched the planner to the byte.

| Emulated card | Requests | DSpark | Planner maximum | At maximum | One token more | Request near the edge | Tracked bytes, planner = server |
|---|---:|---|---:|---|---|---|---:|
| 16 GB | 1 | On | 21,440 | Started | Refused | 21,251 + 128 tokens, completed | 15,669,206,488 |
| 16 GB | 1 | Off | 73,856 | Started | Refused | 73,667 + 128 tokens, completed | 15,669,583,832 |
| 24 GB | 2 | On | 204,672 | Started | Refused | 2 × 204,483 + 128 tokens, concurrent, completed | 24,331,956,192 |

These checks used the default 1 GiB reserve. The 4.0bpw plan reproduces the DGX Spark server's tracked allocation at both measured points exactly: 19,134,608,880 bytes at 8,704 tokens × 4 requests and 23,356,319,192 bytes at 262,656 tokens × 1 request. Device memory outside the engine's allocations peaked at 361 MiB, inside the reserve. Other programs on the GPU lower the free memory the engine starts from; leave room for them or use the desktop rows.
