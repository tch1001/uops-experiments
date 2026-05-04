# uops-style latency experiment (Linux port)

Date: 2026-05-04

Machine: AMD Ryzen 9 5950X (Zen 3, family 25), 16 cores / 32 logical CPUs, Ubuntu 22.04 (kernel 6.8).

## Method

The harness in `src/uops_latency_linux.cpp` is a straight Linux port of `src/uops_latency.cpp`. Same dependent-chain methodology: 64 unrolled identical instructions per loop iteration, 100,000 iterations × 31 samples = 6,400,000 measured ops per sample, with the median empty-loop cost subtracted.

Differences from the Windows version:

1. Affinity: `pthread_setaffinity_np` instead of `SetThreadAffinityMask`.
2. Primary timer: `clock_gettime(CLOCK_THREAD_CPUTIME_ID)` (thread on-CPU nanoseconds) instead of `QueryThreadCycleTime`.
3. RDPMC fixed-counter probe is removed. Linux user-mode `rdpmc` requires `perf_event_open` to arm a counter, which is gated by `kernel.perf_event_paranoid` (=4 on this host) and needs `CAP_PERFMON`/sudo.
4. No priority elevation (`SetPriorityClass`/`THREAD_PRIORITY_HIGHEST`); raising priority on Linux needs `CAP_SYS_NICE`.

## Build

```sh
g++ -O3 -std=c++17 -Wall -Wextra -mcrc32 -mlzcnt -mpopcnt -msse4.2 \
    -o uops_latency_linux src/uops_latency_linux.cpp -lpthread
```

## Run

```sh
./uops_latency_linux 0 100000 31
```

## Result: logical CPU 0

```
features: invariant_tsc=yes popcnt=yes lzcnt=yes sse4.2=yes
empty_loop_median: thread_ns=45440 tsc=151878
```

| Benchmark | Median ns/op | Median tsc/op | Estimated cycles/op |
| --- | ---: | ---: | ---: |
| add r64, imm | 0.212 | 0.722 | 1.00 |
| add r64, reg | 0.212 | 0.722 | 1.00 |
| imul r64, imm | 0.651 | 2.216 | 3.07 |
| shl r64, 1 | 0.212 | 0.723 | 1.00 |
| ror r64, 7 | 0.212 | 0.723 | 1.00 |
| popcnt r64 | 0.212 | 0.722 | 1.00 |
| lzcnt r64 | 0.212 | 0.724 | 1.00 |
| crc32 r64,r64 | 0.651 | 2.214 | 3.07 |
| addsd xmm | 0.651 | 2.214 | 3.07 |
| mulsd xmm | 0.651 | 2.215 | 3.07 |
| divsd xmm | 2.952 | 10.047 | 13.92 |
| sqrtsd xmm | 4.488 | 15.273 | 21.17 |
| L1 ptr chase | 0.902 | 3.070 | 4.25 |

Normalization: median of `add r64,imm/reg`, `shl r64,1`, `ror r64,7` = **0.212 ns/op ≈ 1 cycle**, implying an effective core clock of ~4.71 GHz (matches the 5950X single-core boost).

The TSC runs at ~3.40 GHz (base clock) — the ratio `0.722 / 0.212 ≈ 3.4` confirms TSC ticks at base while the thread-cputime clock measures wall-time on a boosted core. Both agree on the latency, just in different units.

## Cross-reference: Zen 3 (uops.info)

| Instruction | Local est. cycles | Zen 3 reference | Notes |
| --- | ---: | ---: | --- |
| `add r64, r64` | 1.00 | 1 | match |
| `imul r64, r64, imm` | 3.07 | 3 | match |
| `shl r64, 1` | 1.00 | 1 | match |
| `popcnt r64, r64` | 1.00 | 1 | Zen 3 is 1c (Intel through ADL is 3c) |
| `lzcnt r64, r64` | 1.00 | 1 | Zen 3 |
| `crc32 r64, r64` | 3.07 | 3 | match |
| `addsd xmm, xmm` | 3.07 | 3 | match |
| `mulsd xmm, xmm` | 3.07 | 3 | match |
| `divsd xmm, xmm` | 13.92 | 13.5 (range) | within range |
| `sqrtsd xmm, xmm` | 21.17 | 20 | close (Zen 3 sqrtsd ≈ 20c) |
| L1-hit dependent load | 4.25 | 4 | matches AMD's documented L1 latency |

Where the original Intel-Raptor-Lake run normalized timer units against 1-cycle probes and got everything within a few percent of uops.info Alder Lake values, the Linux/Zen 3 run does the same and lands within a few percent of uops.info Zen 3 values. The methodology transfers cleanly across vendors.

## What was NOT measured

Same gaps the original write-up flags, plus a few new ones specific to this host:

- True unhalted core cycles. Blocked by `perf_event_paranoid=4` (needs sudo or `setcap cap_perfmon+ep` on `perf`).
- Uop counts and execution-port pressure. Same gate; AMD-equivalent events would also need different umasks than the Alder Lake P-core list in `docs/WINDOWS_PMU.md`.
- Intel PT / `magic-trace`. Hardware does not exist on AMD; AMD's analog is IBS (`/sys/bus/event_source/devices/ibs_op` is present), but `perf` access is also gated.
- Throughput (independent ops, not dependent chain). The Linux port currently runs latency only; throughput would be an easy extension by removing the dependency between unrolled ops.
