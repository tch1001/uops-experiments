# uops-style latency experiment

Date: 2026-05-04

Machine: 13th Gen Intel(R) Core(TM) i9-13950HX, 24 cores / 32 logical CPUs.

## Method

The harness in `uops_latency.cpp` measures latency with a uops.info-style dependent chain:

1. Pin the benchmark thread to one logical CPU.
2. Raise process/thread priority to reduce scheduling noise.
3. Emit an unrolled chain of 64 identical dependent instructions in inline assembly.
4. Repeat the chain enough times to create 6,400,000 measured operations per sample.
5. Time each sample with `rdtsc/rdtscp` and Windows `QueryThreadCycleTime`.
6. Attempt Intel `RDPMC` fixed counter 1 (`CPU_CLK_UNHALTED.THREAD`) for true core cycles; this host blocks user-mode `RDPMC`, so the reported primary units are `QueryThreadCycleTime`/TSC-like scaled cycle units.
7. Subtract the median empty-loop cost and report median/min/p90 over repeated samples.

Because direct core-cycle counters were unavailable, the final estimate normalizes the primary units against the median of the one-cycle sanity checks: `add r64, imm`, `add r64, reg`, `shl r64, 1`, and `ror r64, 7`. That normalization factor was `0.626 primary units ~= 1 estimated core cycle` for the CPU 16 run.

## Build

```powershell
$env:Path='C:\ProgramData\chocolatey\lib\mingw\tools\install\mingw64\bin;' + $env:Path
g++ -O3 -std=c++17 -Wall -Wextra -o uops_latency.exe uops_latency.cpp
```

## Run

```powershell
$env:Path='C:\ProgramData\chocolatey\lib\mingw\tools\install\mingw64\bin;' + $env:Path
.\uops_latency.exe 16 100000 31
```

## Result: logical CPU 16

Command: `.\uops_latency.exe 16 100000 31`

`RDPMC` fixed core-cycle counter: unavailable. Primary units: `QueryThreadCycleTime`.

| Benchmark | Median primary units/op | Estimated core cycles/op |
| --- | ---: | ---: |
| add r64, imm | 0.614 | 0.98 |
| add r64, reg | 0.667 | 1.07 |
| imul r64, imm | 3.112 | 4.97 |
| shl r64, 1 | 0.628 | 1.00 |
| ror r64, 7 | 0.624 | 1.00 |
| popcnt r64 | 1.875 | 3.00 |
| lzcnt r64 | 1.874 | 2.99 |
| crc32 r64,r64 | 1.904 | 3.04 |
| addsd xmm | 1.920 | 3.07 |
| mulsd xmm | 2.503 | 4.00 |
| divsd xmm | 8.032 | 12.83 |
| sqrtsd xmm | 10.484 | 16.75 |
| L1 ptr chase | 1.848 | 2.95 |

## Notes

Logical CPU 0 was also tested, but the fallback timer showed an immediate-add anomaly and more scaling noise. Logical CPU 16 gave consistent one-cycle sanity checks, so it is the better run to interpret. On this hybrid Intel CPU, logical CPU 16 is likely a different core class than CPU 0.

For publication-grade uops.info-style data, rerun this harness on a system that allows user-mode `RDPMC` or with a small driver/perf subsystem that exposes fixed-function core-cycle counters.
