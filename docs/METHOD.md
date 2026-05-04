# Method: measuring micro-operation latency

This document explains how we arrived at the numbers in `RESULTS.md`, starting from first principles and then connecting our local experiment to the style of work represented by uops.info.

## 1. What is being measured?

Modern x86 CPUs do not execute most x86 instructions directly as written. The front end fetches and decodes architectural instructions into one or more internal micro-operations, usually called uops. Those uops are scheduled onto backend execution resources such as integer ALUs, vector units, load/store units, and divider/square-root units.

Three related measurements matter:

| Measurement | Meaning | Typical experiment |
| --- | --- | --- |
| Latency | How many cycles before the result of one operation can feed the next dependent operation. | Build a dependency chain: `x = op(x)`, repeated many times. |
| Throughput | How many independent operations can execute per cycle once the pipeline is full. | Run many independent copies of the instruction in parallel. |
| Port usage | Which backend execution resources the uops can use. | Combine performance counters, blocking instructions, and throughput experiments. |

Our experiment measured latency, not throughput or port usage.

## 2. Why a dependency chain reveals latency

Suppose an instruction has true latency `L` cycles. If we run this:

```asm
add rax, r8
add rax, r8
add rax, r8
...
```

each `add` needs the previous value of `rax`. Out-of-order execution cannot overlap the data-producing part of those operations, because operation `n + 1` cannot start until operation `n` has produced its result. If we execute `N` dependent operations and the total time is `T`, then:

```text
latency ~= (T - measurement_overhead) / N
```

That is the core idea behind the harness.

The same idea works for `imul`, `popcnt`, `addsd`, `mulsd`, `divsd`, and a pointer-chasing load:

```asm
mov rax, [rax]
mov rax, [rax]
mov rax, [rax]
...
```

The pointer chase measures load-to-use latency when every load address depends on the previous load result.

## 3. Why the experiment needs care

A short instruction sequence is too small to time directly. The timer overhead, branch overhead, interrupts, CPU migration, and power-management state can be as large as the thing being measured. The harness deals with this in several ways:

1. It pins the thread to a logical CPU with `SetThreadAffinityMask`.
2. It raises process and thread priority.
3. It emits 64 dependent instructions per loop iteration with inline assembly.
4. It runs 100,000 iterations for the final run, producing 6,400,000 operations per sample.
5. It measures an empty loop and subtracts the median empty-loop cost.
6. It repeats the sample 31 times and reports median, minimum, and p90.
7. It checks CPU features before running instructions such as `popcnt`, `lzcnt`, and `crc32`.
8. It disassembles the binary with `objdump` to confirm the intended instructions survived compilation.

## 4. Timing source

The harness tries three timing paths:

| Source | Purpose | Result on this machine |
| --- | --- | --- |
| `RDPMC` fixed counter 1 | Desired source: unhalted core cycles. | Blocked in user mode. |
| `QueryThreadCycleTime` | Windows thread-cycle accounting fallback. | Available. |
| `RDTSC/RDTSCP` | Timestamp counter fallback/check. | Available and invariant. |

The best uops.info-style latency number would come from unhalted core cycles via hardware performance counters. The harness attempted this with Intel `RDPMC` fixed counter 1, which corresponds to unhalted core cycles on Intel systems, but the instruction faulted in user mode. Because of that, the result table uses `QueryThreadCycleTime`/TSC-like units, then normalizes them against known one-cycle sanity checks.

This is why `RESULTS.md` calls the output a strong local experiment rather than publication-grade uops.info data.

## 5. What we actually built

The harness is `uops_latency.cpp`. It defines one benchmark function per operation:

| Function | Instruction chain |
| --- | --- |
| `bench_add_r64_imm` | `addq $1, reg` |
| `bench_add_r64_reg` | `addq reg, reg` |
| `bench_imul_r64_imm` | `imulq $3, reg, reg` |
| `bench_shl_r64_1` | `shlq $1, reg` |
| `bench_ror_r64_7` | `rorq $7, reg` |
| `bench_popcnt_r64` | `popcntq reg, reg` |
| `bench_lzcnt_r64` | `lzcntq reg, reg` |
| `bench_crc32_r64` | `crc32q reg, reg` |
| `bench_addsd_xmm` | scalar double add |
| `bench_mulsd_xmm` | scalar double multiply |
| `bench_divsd_xmm` | scalar double divide |
| `bench_sqrtsd_xmm` | scalar double square root |
| `bench_l1_pointer_chase` | dependent L1-ish load chain |

Each chain uses a single value that is both input and output, forcing a real data dependency.

## 6. How we got from source to result

The path was:

1. Checked the machine: 13th Gen Intel Core i9-13950HX, 24 cores / 32 logical CPUs.
2. Found MinGW GCC, CMake, and Ninja locally.
3. Wrote `uops_latency.cpp` with inline assembly chains and timing code.
4. Initial compilation failed silently because the MinGW shim was on PATH, but the real MinGW runtime DLL directory was not.
5. Rebuilt with:

```powershell
$env:Path='C:\ProgramData\chocolatey\lib\mingw\tools\install\mingw64\bin;' + $env:Path
g++ -O3 -std=c++17 -Wall -Wextra -o uops_latency.exe uops_latency.cpp
```

6. Added `_WIN32_WINNT 0x0600` so `QueryThreadCycleTime` was declared by the Windows headers.
7. Confirmed with `objdump` that the benchmark functions contain repeated `add`, `imul`, `popcnt`, `addsd`, `mulsd`, `divsd`, and `sqrtsd` instructions.
8. Ran CPU 0 first:

```powershell
.\uops_latency.exe 0 100000 11
```

CPU 0 showed more scaling noise and an impossible-looking immediate-add result, so it was not used as the final interpretation point.

9. Added a register-source `add` sanity check and an attempted `RDPMC` fixed-counter path.
10. Ran CPU 16:

```powershell
.\uops_latency.exe 16 100000 31
```

CPU 16 produced consistent one-cycle sanity checks:

| Sanity check | Primary units/op |
| --- | ---: |
| `add r64, imm` | 0.614 |
| `add r64, reg` | 0.667 |
| `shl r64, 1` | 0.628 |
| `ror r64, 7` | 0.624 |

The median of the central sanity checks was approximately:

```text
0.626 primary units ~= 1 estimated core cycle
```

So the final estimates were normalized by dividing primary units by `0.626`.

## 7. Final interpreted results

These are the normalized CPU 16 estimates:

| Instruction/test | Estimated latency |
| --- | ---: |
| `add r64, imm` | 0.98 cycles |
| `add r64, reg` | 1.07 cycles |
| `imul r64, imm` | 4.97 cycles |
| `shl r64, 1` | 1.00 cycles |
| `ror r64, 7` | 1.00 cycles |
| `popcnt r64` | 3.00 cycles |
| `lzcnt r64` | 2.99 cycles |
| `crc32 r64,r64` | 3.04 cycles |
| `addsd xmm` | 3.07 cycles |
| `mulsd xmm` | 4.00 cycles |
| `divsd xmm` | 12.83 cycles |
| `sqrtsd xmm` | 16.75 cycles |
| L1 pointer chase | 2.95 cycles |

The values are plausible for a dependent-chain latency experiment. The weak point is not the dependency-chain method; it is the inability to read direct unhalted core cycles from user mode on this Windows host.

## 8. How this compares to uops.info

uops.info is broader and more systematic than our local harness. It provides instruction-level latency, throughput, uop-count, and port-usage data across many x86 microarchitectures. It also distinguishes operand-to-operand latency cases, because the latency from operand 1 to the destination can differ from the latency from operand 2 to the destination for some instructions.

The uops.info paper describes automatically generating microbenchmarks from machine-readable instruction descriptions and then running them across processors. The surrounding research ecosystem includes:

| Work | Role |
| --- | --- |
| uops.info | Empirical instruction tables: latency, throughput, uops, ports. |
| nanoBench | A low-overhead framework for running tiny x86 microbenchmarks with hardware performance counters. |
| uiCA | A throughput analyzer that predicts basic-block throughput using detailed microarchitectural models and instruction data. |
| FACILE | A tool/modeling line that uses instruction characterizations for static throughput estimation. |

Typical research questions in this area are:

1. How many uops does an instruction decode into on a given microarchitecture?
2. Which execution ports can issue those uops?
3. What is the latency from each source operand to each destination operand?
4. What is the reciprocal throughput of an instruction in isolation?
5. How accurately can a model predict throughput of a full basic block?
6. How do instruction properties change across Intel, AMD, and hybrid-core generations?

Our harness answers a smaller question:

```text
For a few selected instructions on this machine, how long does a strict dependent chain appear to take?
```

It does not answer port usage, uop count, operand-specific latency, front-end limits, loop-stream behavior, or full basic-block throughput.

## 9. Magic Trace check

Magic Trace records Intel Processor Trace data and turns it into a timeline-style trace. The important requirements are:

1. A Linux environment.
2. Intel PT-capable hardware.
3. Kernel/perf support exposing Intel PT, usually visible through `perf` and `/sys/bus/event_source/devices/intel_pt`.
4. Sufficient permissions for tracing.

Local checks in this Codex session:

```powershell
where.exe magic-trace
# INFO: Could not find files for the given pattern(s).

wsl --status
# Access is denied.
# Error code: Wsl/EnumerateDistros/Service/E_ACCESSDENIED

wsl sh -lc "uname -a; command -v perf || true; grep -m1 intel_pt /proc/cpuinfo || true; ls /sys/bus/event_source/devices/intel_pt 2>/dev/null || true"
# Access is denied.
# Error code: Wsl/Service/CreateInstance/E_ACCESSDENIED
```

Conclusion: this machine cannot run `magic-trace` directly from the current Windows/Codex session. That is because the tool is Linux/perf/Intel-PT oriented, `magic-trace` is not installed on Windows PATH, and WSL is currently inaccessible from this session.

This does not prove the CPU lacks Intel PT. A 13th Gen Intel Core CPU is plausibly capable. It means the current operating environment is the blocker. To make `magic-trace` viable, use a native Linux boot or a Linux environment where Intel PT is exposed, then check:

```bash
command -v magic-trace
command -v perf
grep -m1 intel_pt /proc/cpuinfo
ls /sys/bus/event_source/devices/intel_pt
cat /proc/sys/kernel/perf_event_paranoid
```

If those checks pass, the CPU and kernel side are much more promising.

## Sources consulted

- [uops.info](https://uops.info/)
- [uops.info instruction tables](https://uops.info/table.html)
- [uops.info paper page](https://uops.info/paper.html)
- [uops.info paper: Characterizing Latency, Throughput, and Port Usage of Instructions on Intel Microarchitectures](https://arxiv.org/abs/1810.04610)
- [nanoBench: A Low-Overhead Tool for Running Microbenchmarks on x86 Systems](https://arxiv.org/abs/1911.03282)
- [uiCA project page](https://uica.uops.info/)
- [FACILE: Fast, Accurate, and Interpretable Basic-Block Throughput Prediction](https://arxiv.org/abs/2310.13212)
- [Magic Trace GitHub repository](https://github.com/janestreet/magic-trace)
- [Magic Trace supported platforms wiki](https://github.com/janestreet/magic-trace/wiki/Supported-platforms%2C-programming-languages%2C-and-runtimes)
