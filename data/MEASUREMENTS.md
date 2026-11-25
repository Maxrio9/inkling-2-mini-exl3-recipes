# TensorFold measurements

## Paged CUDA release candidate

The [planner receipts](planner/) report the shared global KV pool, per-request context cap and fixed bytes per stream. Prefix retention takes pages on demand from that pool for both the global KV rows and restart state. The planner reports zero reserved snapshot slots and zero snapshot bytes. All 15 recipes use the same 3.3 bpw and 4.0 bpw checkpoint headers and FP8 DFlash2 bundle as the earlier build. The 24 GB, 32 GB and Spark recipes plan for 2, 4 and 8 requests, with the same counts for text and vision.

The 16 GB headless recipe admits DFlash2 with a 40,192-token global pool and a 39,936-token per-request cap. The planner accounts for 16,194,582,160 device bytes under the existing 0.5 GiB reserve. Default caching, `TF_KV_SHARE=0`, `--prompt-cache-gib 0`, and a 0.2 GiB retention cap all give the same capacity and allocation. With the additional 1 GiB display allowance, DFlash2 exceeds the startup budget even with the plain head. The desktop preset keeps plain decoding and the plain head.

On the RTX PRO 4500 with a 15.09375 GiB engine cap and 0.5 GiB reserve, the final release binary matched the 16 GB planner's 16,194,582,160 tracked bytes. The GPU process was sampled at 15,786 MiB. It served 8,192 input tokens and 400 output tokens. DFlash2 accepted 318 of 505 proposals in 81 rounds. The engine retained the prompt at its cache mark in 62 pool pages.

### Prefix reuse with reclaimable state

The RTX PRO 4500 served 3.3 bpw with FP8 KV and DFlash2 at `--parallel 4 --context 8704 --vision`. Greedy agreement compares 400 generated tokens per repeated prompt at the same positions.

| Prompt | Reused tokens | Additional pool pages held | Cold TTFT | Hit TTFT | Greedy agreement |
|---|---:|---:|---:|---:|---:|
| Same 8k text | 7,936/8,192 | 62 | 3.176 s | 0.157 s | 400/400 |
| Same PNG and text | 768/1,013 | 29 | 0.640 s | 0.136 s | 400/400 |

The text mark held 31 global KV pages and 31 state pages. The image mark held three global KV pages and 26 state pages. A pool page can hold 256 global KV rows. Four simultaneous requests each resumed the same 2,560-token system prefix. Live leases occupied 4,608 token rows at peak during their common decode interval, compared with 12,288 rows if each request held its own pages.

On a 39,680-token pool, a 23,808-token live reservation evicted the older of two retained 8k prefixes. Eviction released 62 pages, split evenly between global KV and restart state. The recently used prefix still resumed at 7,936 tokens. The evicted prompt then prefilled from token zero, matched 400/400 greedy tokens, and resumed at 7,936 tokens on its next request.

### DFlash2 width policy on the desktop

With prefix retention disabled, a timed RC1 and paged repeat first diverged at round 45, at position 8,393. RC1 proposed and accepted six drafts, and the paged build proposed and accepted seven. The first six proposal IDs and the preceding acceptance history matched. The estimated time to verify seven drafts was 31.22 ms on RC1 and 25.37 ms on the paged build. With the same frozen costs and timing adaptation disabled, all three repeats on each build produced identical round decisions and proposals, with 83 rounds, 499 proposals and 316 accepted tokens in each repeat.

The updated DFlash2 warmup captures the eligible greedy target graph at verification widths 1 through 8 and uses the median of three warmed target and three warmed drafter replays to price each width. During generation, per-stream timing and acceptance still inform the policy. The frozen-cost traces isolate drafter behavior from timing for this comparison.

### Desktop throughput after the warmup change

The RTX PRO 4500 ran 3.3 bpw with FP8 KV and DFlash2. All three arms used 8,192 input tokens, 400 greedy output tokens, three repeats per workload and disabled prefix retention. Rates are median decode tokens per second. C4 uses the common decode interval.

| Workload | RC1 A | Paged | RC1 B | Pooled RC1 | Change | 3% floor |
|---|---:|---:|---:|---:|---:|---|
| C1 code | 190.51 | 190.79 | 177.95 | 186.03 | +2.56% | Pass |
| C1 prose | 101.59 | 99.53 | 97.89 | 99.39 | +0.15% | Pass |
| C4 mixed | 163.08 | 154.40 | 155.05 | 158.47 | −2.57% | Pass |

Output token IDs matched across all three arms in each workload. For C1 code the candidate used 81, 81 and 79 verification rounds. RC1 A used 88, 83 and 81, and RC1 B used 85, 82 and 82. The candidate peaked at 16,924 MiB of GPU memory.

### Spark throughput after the warmup change

The final aarch64 archive ran on Spark1 with the same 4.0 bpw target, FP8 settings, container image, and 100 GiB caps as both RC1 controls. Three repeats per arm returned matching output IDs.

| Workload | RC1 A | Fixed paged | RC1 B | Pooled RC1 | Change | 3% floor |
|---|---:|---:|---:|---:|---:|---|
| Spark1 C1 code | 84.50 | 84.95 | 86.82 | 84.67 | +0.34% | Pass |
| Spark1 C8 mixed | 97.77 | 97.77 | 97.07 | 97.50 | +0.28% | Pass |

The candidate's minimum sampled host MemAvailable was 92.95 GiB. The final bracket and raw logs are in `paged-build/fix/spark/` in the run folder.

### Initial paged build

The following results predate the removal of reserved snapshot slots.

The desktop fidelity check matched all 512 teacher-forced positions at 8k and all 128 at 32k. Mean KL was 0 for both, matching the RC1 self-comparison. Each repeated prompt matched all 400 greedy output tokens. The same 8k text prompt reused 7,936 tokens and reduced TTFT from 3.086 to 0.139 seconds. The same image prompt reused 768 of 1,013 tokens and reduced TTFT from 0.687 to 0.120 seconds. Four concurrent requests reused one 2,560-token system prefix. Their shared occupancy was 4,608 tokens, versus 12,288 with independent reservations.

The pool-pressure test ran a 10,176-token request beside three 512-token prompts on a 39,680-token pool. Two 20,224-token reservations queued for pool room despite a free lane, and both completed. The planner and runtime tracked 17,179,293,620 bytes for that configuration.

The throughput bracket used 8,192 input tokens, 400 greedy output tokens, three repetitions per arm, and FP8 KV and DFlash2. Prefix caching was disabled for all throughput arms.

| GPU and workload | RC1 A | Paged | RC1 B | Pooled RC1 | Change |
|---|---:|---:|---:|---:|---:|
| RTX PRO 4500, C1 code | 182.94 | 174.85 | 186.01 | 183.03 | −4.47% |
| RTX PRO 4500, C1 prose | 96.83 | 97.69 | 97.73 | 97.39 | +0.31% |
| RTX PRO 4500, C4 mixed | 155.46 | 155.80 | 155.87 | 155.66 | +0.09% |
| DGX Spark, C1 code | 84.79 | 90.37 | 84.77 | 84.78 | +6.59% |
| DGX Spark, C8 mixed | 97.23 | 97.95 | 98.72 | 98.22 | −0.28% |

Rates are median decode tokens per second. Concurrent rates use the common decode interval. All output token IDs matched across each bracket. The desktop C1 code-only repeat measured 184.78 tok/s against a 196.09 tok/s pooled control (−5.77%). In an Nsight trace with three requests, the candidate used 88–98 verification rounds against 81–84 for RC1. The FP8 prefetch kernel averaged 37.62 μs per launch on the candidate and 38.79 μs on RC1. The desktop C1 code cell remains outside the 3% acceptance floor.

The Spark C1 code gain came from fewer verification rounds. Each RC1 repeat used 89 rounds and accepted 310 of 455 proposals. The paged repeats used 78–81 rounds and accepted 318–321 of 508–513 proposals. Median time per round increased from about 52.9 to 56.6 ms. All three arms used the same local container image, model and drafter, KV precision, context and parallel settings, reserves and caps. The output tokens matched.

The 16 GB default served an 8,192-token prompt and 400-token reply under a 15.09375 GiB tracked cap and a 0.5 GiB reserve. The planner and engine both tracked 16,199,745,952 bytes. The GPU process used 15,818 MiB, and DFlash2 accepted 306 of 420 draft tokens. The 16 GB desktop allowance still requires plain decoding.

## RC1 configuration sizing

The archived RC1 planner receipts used the integrated RC1 planner, staged 3.3bpw and 4.0bpw target headers, and the FP8 DFlash2 package. Each receipt has the unconstrained admission boundary in `maximum` and the selected config's fixed-window capacity report in `recipe.result`. The latter includes budget for the cache after the model and every request slot. On 16 GB desktop, the planner admitted 13,632 tokens with the lazy head and 40,960 with the plain head, using a 1 GiB display allowance and a 0.5 GiB engine reserve. The default presets are text-only. Each 24 GB, 32 GB and Spark profile has a separate vision preset with the same stream count. The 24 GB desktop pair uses one stream because the two-stream vision plan leaves 1.38 GiB for prompt caching at the 65,536-token floor. The planner reserves 2 GiB for prompt cache on 24 GB and 32 GB cards and 8 GiB on Spark. Both Spark vision rows still exceed 580,000 tokens per request.

## RC1 desktop IFBench

RTX PRO 4500 with identical staged 3.3bpw target weights and FP8 DFlash2 layers and head. All 300 items used seed 0, default high reasoning effort, a 33,792-token window, six streams, temperature 1.0, top-p 0.95, top-k 20 and a 32,768-token completion cap. Prompt caching was disabled. After the desktop changed from graphical to compute mode, the FP8 arm resumed from 177 completed items with the same generation policy. Aggregate output rates include request scheduling and prefill and use active time from each arm.

| Target KV | Loose | Strict | Output caps | Mean completion tokens | Median request time | Aggregate output tok/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BF16 | 217/300 (72.33%) | 209/300 (69.67%) | 11 | 5,603.8 | 85.67 s | 181.95 |
| FP8 | 219/300 (73.00%) | 212/300 (70.67%) | 9 | 5,308.7 | 93.98 s | 175.98 |

The paired FP8-minus-BF16 difference was +0.67 percentage points on the loose score (95% bootstrap interval -3.67 to +5.00) and +1.00 points on the strict score (95% interval -3.67 to +5.33). Neither score detected a loss. Both arms completed every item with reasoning present and zero request or grader errors. The capped responses had empty final answers and remain in the primary comparison. Excluding the 18 items capped by either arm gave 0.00 percentage points on the loose measure across 282 pairs (95% interval -4.26 to +4.26).

## S1–S6 development measurements

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

## Original planner validation

The planner reads the model branch's checkpoint headers and runs the engine's sizing and startup admission without a GPU. This validation used the original recipe settings and verifies the planner against live startup before the DFlash2 and vision update.

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
