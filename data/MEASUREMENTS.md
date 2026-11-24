# TensorFold measurements

These are served measurements, not speed projections. Slices S1–S6 used development builds leading to TensorFold branch `inkling2-mini-exl3`, final source commit `70d6031b52d158e0ce21eb4b54f454ce714370e8`; the individual builds predate that commit. S1 used source based on `42bbc8e35`, S4 `a1c1f8d53`, and S6 used builds based on `cb54db840`. The final commit has not itself been benchmarked. RTX PRO 4500 Blackwell: sm_120, NVIDIA driver 615.78.08. DGX Spark GB10: sm_121, driver 595.99.02. Builds used Zig 0.17.0 and CUDA nvcc 13.4.92. Speeds are decode unless otherwise stated; prompt and output counts and other settings appear below. Earlier ExLlamaV3 measurement tables were dropped from this file; legacy configs remain in `configs/legacy-exllamav3/`.

## S1: initial server and parity

RTX PRO 4500, 3.3bpw, one stream, BF16 KV, no drafts, no retained prompt cache, greedy, 128-token output cap; speed median of three. Reference used the same pack in ExLlamaV3, FP16 KV and FP16 output head, versus BF16 KV and head here.

| Measure | Result |
|---|---:|
| 2,048 prompt decode | 66.57 tok/s |
| 8,192 prompt decode | 66.49 tok/s |
| Cold TTFT, 2,048 / 8,192 prompt | 0.778 / 3.193 s |
| Exact greedy first 64 tokens | 10/16 prompts |
| Teacher-forced top-1 agreement | 1,015/1,024 (99.12%) |

## S2: graph and BF16 KV drafter

RTX PRO 4500, 3.3bpw, one stream, BF16 KV, 8,192 input tokens, 400 output tokens, warm server, no cached prompt tokens; median of three. Requested effort `none`, though the model generated reasoning. Temperature is shown below.

| Workload | Temperature | Plain tok/s | DSpark tok/s | Accepted/proposed |
|---|---:|---:|---:|---:|
| Code | 0 | 66.80 | 122.14 | 826/1,613 |
| Code | 0.6 | 67.19 | 105.03 | 776/1,781 |
| Prose | 0 | 66.91 | 82.47 | 558/1,506 |
| Prose | 0.6 | 67.60 | 73.85 | 480/1,363 |

At a 2,048-token prompt and 128-token output cap, graph candidate decode was 68.07 tok/s versus 66.88 and 66.83 for bracketing controls. With BF16 KV, DSpark verification top-1 matched its plain target at all 4,408 positions tested across widths 1–8, with zero measured KL.

## S3: FP8 KV

RTX PRO 4500, 3.3bpw, one stream, 8,192 input tokens, 400 output tokens, greedy, warm, no prompt cache, DSpark; median of three.

| Workload | BF16 control A | FP8 | BF16 control B | FP8 accepted/proposed |
|---|---:|---:|---:|---:|
| Code | 118.35 | 123.64 | 117.62 | 847/1,546 |
| Prose | 79.88 | 78.33 | 79.82 | 576/1,555 |

FP8 versus BF16 KV teacher-forced top-1 matched 1,080/1,088 positions (99.26%); mean KL was 0.008890 nats. FP8 DSpark verification matched 4,408/4,408 plain FP8 positions, zero measured KL. These checks do not establish identical output between KV formats.

## S4: retrieval

RTX PRO 4500, 3.3bpw, one stream, DSpark, no prompt cache, requested effort `none`; normal completion with a 256-token output cap. Each is one five-fact test.

| KV | Prompt tokens | Correct facts | Output tokens | Cold TTFT (s) | Decode tok/s | Sampled process peak MiB |
|---|---:|---:|---:|---:|---:|---:|
| FP8 | 32,502 | 5/5 | 191 | 13.975 | 140.72 | 17,472 |
| FP8 | 130,808 | 5/5 | 208 | 62.913 | 98.93 | 17,472 |
| BF16 | 130,808 | 5/5 | 203 | 60.745 | 95.27 | 19,688 |

The 131,072-token server window was preallocated for all rows. The 128-token output cap initially exhausted itself in reasoning; a 256-token cap yielded final answers.

## S5: concurrency before batch-aware policy

RTX PRO 4500, 3.3bpw, FP8 KV, 8,704-token window, eight allocated slots, 8,192-token input, 400-token output, greedy, no prompt cache, three repeats. Shared aggregate counts decode only over the common interval; delivery aggregate includes prefill overlap and tail.

| Concurrent | Mode | Shared aggregate tok/s | Delivery aggregate tok/s |
|---:|---|---:|---:|
| 2 | Plain / DSpark | 99.29 / 130.08 | 69.95 / 76.87 |
| 4 | Plain / DSpark | 140.11 / 133.02 | 73.97 / 69.72 |
| 8 | Plain / DSpark | 177.52 / 172.11 | 76.22 / 73.23 |

All 84 concurrent completions matched their solo control. Four-stream shared verification matched 256/256 width-one and 280/280 mixed-width positions to solo FP8 controls, zero measured KL.

## S6: batch-aware policy and DGX Spark

Same 8k prompts and sampling as S5, three repeats. RTX PRO 4500 used 3.3bpw; DGX Spark used 4.0bpw. Both used FP8 KV; the Spark c=4 run used plain decoding. Listed rates are shared decode medians.

| Hardware | Concurrency, workload | Plain tok/s | DSpark tok/s |
|---|---|---:|---:|
| RTX PRO 4500 | 1, code | 66.56 | 126.68 |
| RTX PRO 4500 | 1, prose | 66.44 | 81.77 |
| RTX PRO 4500 | 2, aggregate | 99.54 / 99.52 controls | 101.06 |
| RTX PRO 4500 | 4, aggregate | 140.32 / 140.54 controls | 140.14 |
| DGX Spark | 1, code | 36.31 | 59.20 |
| DGX Spark | 1, prose | 36.31 | 42.16 |
| DGX Spark | 4, mixed aggregate | 74.41 | Not tested |

The Spark c=4 delivery aggregate was 26.07 tok/s. All 144 desktop and 24 Spark timed completions matched their corresponding target token IDs. A single Spark retrieval test with a 262,144-token prompt and 227 output tokens returned 5/5 facts, cold TTFT 512.36 s. Spark peak process RSS was 29.37 GiB at 8,704 × 4 and 29.49 GiB at 262,656 × 1; minimum system available memory was 97.67 and 93.15 GiB respectively. These RSS figures include loading and are not discrete GPU memory.

## Capacity projections

The README configurations are projections, not tested fits. Model context is capped at 1,048,576 tokens. FP8 target KV costs 16,896 bytes per global token and drafter KV 4,096 bytes per token; target KV rounds up in 64-token steps. Estimates include context-dependent target and drafter attention scratch, fixed local rings, and stream state. The 3.3bpw desktop target weights measured 12,428,430,336 bytes; 4.0bpw target weights measured 14,721,666,048 bytes on Spark. The difference, 2,293,235,712 bytes, is added to the desktop 3.3bpw estimate for 4.0bpw. Drafter weights measured 739,122,944 bytes.

The single-stream 16 GB projection uses the S3 FP8 process peak of 14,936 MiB at 8,704 tokens, with measured fixed overhead 2,311,263,488 bytes. Multi-stream discrete estimates use the S5 eight-slot process peak 18,708 MiB, adjusted by 479,944,712 bytes per stream at 8,704; 294,912 bytes per stream is added for S6's new buffer. Spark uses its measured four-slot tracked anchor 19,134,608,880 bytes, 480,239,624 bytes per stream and 1,979,980,328 bytes for the largest observed additional available-memory drop. All estimates add 0.3 GiB allocator slack plus 243,531,776 bytes, the observed S3 fixed-overhead variation between BF16 and FP8. Desktop rows additionally reserve 1 GiB. Spark uses a budget of 92.48 GiB derived from 116.48 GiB available before launch minus 24 GiB, subject to a 100 GiB cap; the desktop row subtracts another 1 GiB. Other workloads and drivers can change these footprints.

The maximum supported count considered is 16 slots. For each row the largest count that allows at least 65,536 tokens without a desktop or 32,768 with a desktop was chosen; if none qualifies, one DSpark request is used. Contexts are rounded down to 4,096 tokens. At 16 GB without desktop the one-stream DSpark result is 53,248 tokens. With DSpark disabled, the same single-stream FP8 anchor and margins, less 739,122,944 bytes of drafter weights and 4,096 bytes per token of drafter KV, give 107,904 tokens without desktop and 47,360 with desktop before rounding: 106,496 and 45,056 respectively. Context-dependent drafter scratch growth is also removed; the anchor retains fixed drafter overhead, making this estimate conservative. The 24/32 GB desktop-card numbers use the desktop 3.3bpw anchor plus the measured weight difference and have not been measured with 4.0bpw. The Spark 16-slot and discrete 9/14-slot configurations exceed measured concurrency; allocation scaling and long-context scratch are extrapolations. No edge-fit or 128k speed test exists for these exact configurations. Usable VRAM, free unified memory, retained caches and other applications can reduce fit.
