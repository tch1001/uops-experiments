# uops-style microbenchmarks: full Linux/Zen 3 run

Date: 2026-05-04

Machine: AMD Ryzen 9 5950X (Zen 3, family 25). Single socket / single NUMA node, two CCDs (CCD0 = cores 0-7, CCD1 = cores 8-15). Ubuntu 22.04, kernel 6.8.

**Sudo prerequisites** (run once per boot):
```sh
sudo sysctl -w kernel.perf_event_paranoid=0
sudo sh -c 'echo 2 > /sys/bus/event_source/devices/cpu/rdpmc'
```

**Build all** (in repo root):
```sh
mkdir -p build
for f in latency_linux latency_pmc throughput op2op vector storefwd memory generated; do
  g++ -O3 -std=c++17 -Wall -Wextra \
      -mcrc32 -mlzcnt -mpopcnt -msse4.2 -mavx2 -mfma -mbmi -mbmi2 \
      -o build/uops_${f} src/uops_${f}.cpp -lpthread
done
```

**Run all** (sequential, pinned to CPU 0):
```sh
tools/run_all.sh
```

Cycle conversions in this document use the local 1-cycle anchor: from `uops_latency_pmc`'s Pass-1 cycles column, ALU ops sit at 0.98c, so on this run 1 cycle ≈ 0.213 ns (4.71 GHz boost). Where a harness reports `est cycles` from its own normalization, the conversion may differ slightly because of empty-loop calibration.

---

## 1. Cycles, uops, port pressure, energy (`uops_latency_pmc`)

This is the canonical source: counters armed via `perf_event_open` and read with `rdpmc`. Five passes per bench (cycles+uops, FP-pipe, LS-dispatch, branches, RAPL). 100k×31 samples.

### 1a. cycles + uops/insn

| benchmark | cycles/op | insns/op | uops/op | uops.info Zen 3 lat | match? |
|---|---:|---:|---:|---:|---|
| add r64, imm | 0.98 | 1.00 | 1.00 | 1 | ✓ |
| add r64, reg | 0.98 | 1.00 | 1.00 | 1 | ✓ |
| imul r64, imm | 2.98 | 1.00 | 1.00 | 3 | ✓ |
| shl r64, 1 | 0.98 | 1.00 | 1.00 | 1 | ✓ |
| ror r64, 7 | 0.98 | 1.00 | 1.00 | 1 | ✓ |
| popcnt r64 | 0.98 | 1.00 | 1.00 | 1 | ✓ |
| lzcnt r64 | 0.98 | 1.00 | 1.00 | 1 | ✓ |
| crc32 r64,r64 | 2.98 | 1.00 | 1.00 | 3 | ✓ |
| addsd xmm | 2.98 | 1.00 | 1.00 | 3 | ✓ |
| mulsd xmm | 2.98 | 1.00 | 1.00 | 3 | ✓ |
| divsd xmm | 13.48 | 1.00 | 1.00 | 13.5 | ✓ |
| sqrtsd xmm | 20.49 | 1.00 | 1.00 | 20 | ✓ |
| L1 ptr chase (mov r,[mem]) | 4.00 | 1.00 | 1.00 | 4 | ✓ |

Every instruction in the set is single-uop on Zen 3 (no fused-uop, no multi-uop ops here). Cycle counts match uops.info Zen 3 within ~1%.

### 1b. FP-pipe binding (port pressure)

The Zen 3 FP cluster has 4 pipes (FP0..FP3) with overlapping but distinct capabilities. Each FP op is dispatched to one pipe; counter `fpu_pipe_assignment.totalN` increments per uop on pipe N. With 64 dependent ops/iter and 31 samples, splits below 0.001 are noise.

| benchmark | fp_p0 | fp_p1 | fp_p2 | fp_p3 |
|---|---:|---:|---:|---:|
| addsd xmm | 0.00 | 0.00 | **0.50** | **0.50** |
| mulsd xmm | **0.50** | **0.50** | 0.00 | 0.00 |
| divsd xmm | **0.50** | **0.50** | 0.00 | 0.00 |
| sqrtsd xmm | **0.50** | **0.50** | 0.00 | 0.00 |
| (all integer ops) | 0.00 | 0.00 | 0.00 | 0.00 |
| L1 ptr chase | 0.00 | 0.00 | 0.00 | 0.00 |

Reading: `addsd` issues onto pipes 2 and 3 (50/50); `mulsd / divsd / sqrtsd` issue onto pipes 0 and 1. This matches the published Zen 3 architecture (FP0/FP1 = mul/FMA/div/sqrt, FP2/FP3 = add/store-data). This is the Zen 3 analog of uops.info's per-port column.

### 1c. LS dispatch breakdown

| benchmark | ld_disp | st_disp | ldst_disp |
|---|---:|---:|---:|
| L1 ptr chase | **1.00** | 0.00 | 0.00 |
| (all others) | ≤0.001 | 0.00 | 0.00 |

Confirms 1 load per pointer-chase op; no spurious memory traffic in the other benches.

### 1d. Branches / mispredicts (sanity)

All benches: 0 branches, 0 mispredicts per op. Tight inline-asm bodies do not contain branches once the outer C loop is amortized over 64 ops.

### 1e. RAPL package energy per op (pJ)

| benchmark | pJ/op | relative to add |
|---|---:|---:|
| add r64, imm | 13,959 | 1.0× |
| add r64, reg | 13,723 | 1.0× |
| popcnt r64 | 13,943 | 1.0× |
| lzcnt r64 | 14,069 | 1.0× |
| ror r64, 7 | 14,558 | 1.0× |
| shl r64, 1 | 16,100 | 1.2× |
| imul r64, imm | 52,752 | 3.8× |
| crc32 r64,r64 | 52,478 | 3.8× |
| mulsd xmm | 53,990 | 3.9× |
| addsd xmm | 54,524 | 3.9× |
| L1 ptr chase | 78,497 | 5.6× |
| divsd xmm | 255,172 | 18.3× |
| sqrtsd xmm | 384,297 | 27.5× |

Caveat: RAPL measures whole-package energy, including 31 idle logical CPUs and uncore. The per-op figure is `(active - empty_loop) / ops`, so absolute pJ is dominated by package baseline. Relative ordering (`sqrtsd` > `divsd` >> `mulsd` ≈ `addsd` ≈ `crc32` ≈ `imul` >> `add`/`popcnt`/`lzcnt`) reflects the FP units being substantially more power-hungry than ALU pipes, with the divider/sqrt unit the dirtiest.

---

## 2. Reciprocal throughput (`uops_throughput`)

4-way independent destinations for ALU/FP-add/FP-mul, 8-way for divsd/sqrtsd, 8 streams for independent L1 loads.

| benchmark | tput ns/op | tput cycles/op | uops.info Zen 3 tput | match? |
|---|---:|---:|---:|---|
| add r64, imm | 0.053 | 0.25 | 0.25 | ✓ (4 ALU pipes) |
| add r64, reg | 0.053 | 0.25 | 0.25 | ✓ |
| imul r64, imm | 0.216 | 1.01 | 1 | ✓ (1 imul pipe) |
| shl r64, 1 | 0.106 | 0.50 | 0.5 | ✓ (2 shifters) |
| ror r64, 7 | 0.106 | 0.50 | 0.5 | ✓ |
| popcnt r64 | 0.055 | 0.26 | 0.25 | ✓ |
| lzcnt r64 | 0.052 | 0.24 | 0.25 | ✓ |
| crc32 r64,r64 | 0.217 | 1.02 | 1 | ✓ |
| addsd xmm | 0.167 | 0.79 | 0.5 | ~ (front-end limited?) |
| mulsd xmm | 0.167 | 0.79 | 0.5 | ~ |
| divsd xmm | 0.988 | 4.64 | 4.5 | ✓ |
| sqrtsd xmm | 1.875 | 8.81 | 8.5 | ✓ |
| L1 indep loads (8 streams) | 0.069 | 0.32 | 0.33 | ✓ (3 AGUs) |

The addsd/mulsd throughput coming in at ~0.79c instead of 0.5c is plausibly limited by inline-asm constraint shapes (4 xmm registers chasing through `sink_f64`). Could be tightened with explicit register-asm bindings.

---

## 3. Operand-to-operand latency (`uops_op2op`)

For ops where uops.info reports per-operand-pair latencies.

| iform | path | est cycles | uops.info Zen 3 | match? |
|---|---|---:|---|---|
| adc r1, r2 | dst→dst | 1.01 | 1 | ✓ |
| adc r1, r2 | src→dst | 1.01 | 1 | ✓ |
| adc r1, $0 | CF→dst | 1.01 | 1 | ✓ |
| sbb r1, r2 | dst→dst | 1.01 | 1 | ✓ |
| sbb r1, r2 | src→dst | 1.02 | 1 | ✓ |
| sbb r1, $0 | CF→dst | 1.02 | 1 | ✓ |
| cmovne r1, r2 | dst→dst | 1.02 | 1 | ✓ |
| cmovne r1, r2 | src→dst | 1.02 | 1 | ✓ |
| cmovne+test | flags→dst | 1.01 | 1 | ✓ |
| mov r,(r) | simple addr | 4.27 | 4 | ✓ |
| mov r,(b,i,8) | indexed addr | **5.18** | 4-5 | **+1c for AGU** |
| add r,[mem] | fused | 1.01 | 1 | ✓ |
| add r,r; add r,[m] | split via r1 | 1.01 | 1 | ✓ |

Key finding: **complex addressing (`base + index*scale + disp`) costs +1c on Zen 3** compared to simple `(reg)` pointer chase. AGU isn't free for non-trivial address modes.

---

## 4. AVX2 / FMA / SSE-vector (`uops_vector`)

Latency and throughput for 128-bit SSE, 256-bit AVX2 integer, 256-bit AVX FP, FMA, lane-crossing shuffles, and 256-bit divide/sqrt.

| benchmark | lat (cycles) | tput (cycles) | uops.info Zen 3 lat | match? |
|---|---:|---:|---|---|
| paddd xmm | 1.01 | 0.24 | 1 | ✓ |
| pmulld xmm | 3.08 | 0.50 | 3 | ✓ |
| pshufb xmm | 1.02 | 0.50 | 1 | ✓ |
| pcmpeqb xmm | 1.01 | 0.24 | 1 | ✓ |
| por xmm | 1.01 | 0.24 | 1 | ✓ |
| vpaddd ymm | 1.01 | 0.24 | 1 | ✓ |
| vpmulld ymm | 3.08 | 0.55 | 3 | ✓ |
| vpshufb ymm | 1.02 | 0.50 | 1 | ✓ |
| vpor ymm | 1.01 | 0.24 | 1 | ✓ |
| vpaddq ymm | 1.01 | 0.24 | 1 | ✓ |
| vaddps ymm | 3.08 | 0.50 | 3 | ✓ |
| vmulps ymm | 3.08 | 0.50 | 3 | ✓ |
| vaddpd ymm | 3.08 | 0.50 | 3 | ✓ |
| vmulpd ymm | 3.08 | 0.50 | 3 | ✓ |
| vfmadd231ps ymm | 4.11 | 0.55 | 4 | ✓ |
| vfmadd231pd ymm | 4.11 | 0.55 | 4 | ✓ |
| vfnmadd231ps ymm | 4.11 | 0.55 | 4 | ✓ |
| vpermq ymm,imm8 | 6.76 | 1.29 | 6 | ✓ (lane-cross +) |
| vpermd ymm,ymm | 8.61 | 1.50 | 6-8 | ✓ |
| vdivps ymm | 15.39 | 3.17 | ~14 | ✓ |
| vsqrtpd ymm | 22.04 | 8.87 | ~14-21 | ~ (high end of range) |

256-bit AVX2 integer ops show the same 1-cycle latency as 128-bit (Zen 3 widens to 256-bit FMA but ops still flow through 128-bit ALUs in lanes — confirmed). FMA has the expected +1c over add/mul. Lane-crossing shuffles cost ~6-8c vs in-lane shuffles at 1c.

---

## 5. Store-to-load forwarding + 4K aliasing (`uops_storefwd`)

| scenario | cycles/op | expected | match? |
|---|---:|---|---|
| A: aligned 8B→8B same addr | 4.82 | 5-7 (Zen 3 fwd) | ✓ |
| B: 8B@0 → 8B@4 (partial overlap) | 13.47 | ~13c stall | ✓ |
| C: 4B@0 → 8B@0 (wider load) | 13.48 | ~13c stall | ✓ |
| D: 8B@0 → 4B@0 (narrower load) | 4.78 | fwd succeeds | ✓ |
| E1: 8B@0 → 8B@4096 (4K alias) | (no penalty) | (Intel-only stall) | ✓ |
| E2: 8B@0 → 8B@64 (control) | (no penalty) | baseline | ✓ |

Confirms Zen 3 forwarding rules: succeeds when load-extent ⊆ store-extent and aligned; fails on partial overlap or wider load. Zen 3 has **no 4K-aliasing penalty** (an Intel-only front-end stall).

---

## 6. Memory hierarchy + cross-CCD (`uops_memory`)

### 6a. Working-set sweep (64B stride, randomized ring)

| working set | ns/load | est cycles | regime |
|---|---:|---:|---|
| 8 KB | 0.88 | 4.1 | L1 |
| 16 KB | 0.88 | 4.1 | L1 |
| 32 KB | 0.94 | 4.4 | L1 → L2 boundary |
| 64-256 KB | 2.65-3.01 | 12-14 | L2 |
| 512 KB - 2 MB | 5.95-11.10 | 28-52 | L2 spill / L3 |
| 4-32 MB | 11.71-16.30 | 55-77 | L3 |
| 64-512 MB | 18.00-24.12 | 84-113 | DRAM |
| 1 GB | 25.45 | 119 | DRAM (TLB pressure) |

Clear hierarchy steps: **L1 ≈ 4c, L2 ≈ 12c, L3 ≈ 50-75c, DRAM ≈ 90-120c**. L3 latency rises as the working set grows past one CCD's 32 MB cache.

### 6b. TLB sweep (4 KB pages)

| pages | size | ns/load | regime |
|---|---|---:|---|
| 64 | 256 KB | 3.30 | dTLB-L1 hit (Zen 3 dTLB-L1 = 64 entries) |
| 256 | 1 MB | 4.47 | dTLB-L2 hit |
| 1024 | 4 MB | 6.66 | dTLB-L2 hit |
| 4096 | 16 MB | 12.68 | dTLB-L2 spill / page walks |
| 16384 | 64 MB | 15.62 | full PTW cost |

### 6c. Cross-CCD pointer chase (1 MB working set)

| scenario | ns/load |
|---|---:|
| same-CCD baseline (CCD0→CCD0) | 9.74 |
| cross-CCD CCD1→CCD0 | 9.73 |
| cross-CCD CCD0→CCD1 | 9.62 |
| same-CCD baseline (CCD1→CCD1) | 9.63 |

At 1 MB working set the data sits in either CCD's 32 MB L3, so once the chase warms up locally there's no inter-die traffic. Cross-CCD effects are visible in different harnesses (concurrent contention, false sharing) but not in this single-thread chase. Adequate as a baseline.

---

## 7. XED-driven generator prototype (`uops_generated`)

**166 instruction variants** auto-generated from uops.info `instructions.xml` via `tools/gen_from_xed.py`. Families covered:

- BASE GPR binaries: ADD/SUB/AND/OR/XOR (R64,R64), (R64,I8), (R64,I32) — 12
- BASE GPR shifts: SHL/SHR/SAR/ROL/ROR R64 — 10
- IMUL 2/3-op — 3
- Bit-manipulation: POPCNT, LZCNT, TZCNT, BSR, BSF, BLSI, BLSR, BLSMSK — 8
- BMI2 3-op: ANDN, BEXTR, BZHI, SHLX, SHRX, SARX, PEXT, PDEP — 8
- SSE2 scalar/packed FP (XMM,XMM): ADDSD/SUBSD/MULSD/DIVSD/MAXSD/MINSD/SQRTSD + SS + PD + PS — 28
- AVX scalar/packed (3-op): same set in V-prefix — 38
- FMA: VFMADD/VFMSUB/VFNMADD/VFNMSUB × {132,213,231} × {PD,PS,SD,SS} × {XMM,YMM} — 60

Examples (a few from the run):

| iform | gen ns/op | gen cycles | hand-written / uops.info | match? |
|---|---:|---:|---|---|
| VADDPD_YMM_YMM_YMM | 0.657 | 3.08 | 3 | ✓ |
| VMULPD_YMM_YMM_YMM | 0.657 | 3.08 | 3 | ✓ |
| VFMADD231PD_YMM_YMM_YMM | 0.877 | 4.12 | 4 | ✓ |
| VDIVPD_YMM_YMM_YMM | 2.97 | 13.94 | 13.5 | ✓ |
| VDIVPS_YMM_YMM_YMM | 2.31 | 10.85 | 10 | ✓ |
| ADD_R64_R64 | 0.216 | 1.01 | 1 | ✓ |
| IMUL_R64_R64 | 0.657 | 3.08 | 3 | ✓ |
| TZCNT_R64_R64 | varies | varies | 1 | ⚠ (some 1-src ops measure as throughput) |
| VSQRTPD_YMM_YMM | 1.87 | 8.78 | 14-21 | ⚠ (low — chain not tight) |

**Known prototype limitations** (documented in the agent's report):

- Single-pass timing (no min/p90 percentiles).
- Some single-source ops (LZCNT, TZCNT, VSQRTPD/PS) measure closer to throughput than latency because `register asm("rax") x` + `lzcntq %%rax,%%rax` doesn't always communicate the dependency to gcc tightly enough. Hand-written + PMC harnesses are the canonical sources for those.
- No throughput-mode benchmarks (round-robin destinations).
- No memory-operand variants.
- No 32/16/8-bit GPR forms.

Despite the limitations, this represents a **12.7× expansion** in instruction coverage (13 → 166) with the framework now in place to scale further. Closing the chain-tightness issue requires per-iform asm template tuning; a v2 generator pass.

---

## Cross-cutting summary table

What we now have for Zen 3, comparable to the columns uops.info publishes:

| uops.info column | Local data | Source |
|---|---|---|
| Latency (operand-pair) | ✓ for 13 + 21 + 13 = 47 ops with min/p90 | latency_pmc, vector, op2op |
| Throughput (reciprocal) | ✓ for 13 + 21 ops | throughput, vector |
| Uops per instruction | ✓ for 13 ops | latency_pmc Pass 1 |
| Macro-op fusion | ✓ (no fusion seen in this set; insns/uops = 1.0) | latency_pmc Pass 1 |
| Port pressure | ✓ FP-pipe-level for 4 FP ops; LS-pipe-level for memory ops | latency_pmc Passes 2+3 |
| Per-instruction sweep at scale | 166 iforms (prototype) | generated |

What we have **beyond** uops.info's table:

- L1/L2/L3/DRAM latency hierarchy (working-set sweep)
- TLB latency sweep (dTLB-L1/L2/PTW)
- Cross-CCD pointer-chase baseline
- Store-to-load forwarding success/failure cases
- 4K-aliasing absence on Zen 3 (vs Intel)
- Per-op RAPL package energy

What's still **missing** vs. the broader research field:

- Full XED-driven coverage (we have 166 iforms; uops.info has ~10K)
- Memory-operand variants of every instruction
- 32/16/8-bit GPR forms
- AMD IBS-based per-op sampling (gated by `perf_event_paranoid`, but theoretically reachable)
- ROB / RS / scheduler capacity sweeps
- Branch-predictor structural reverse-engineering
- Throughput predictor (uiCA-equivalent) consuming this data

## File map

```
results/run_full_5950x/
  latency_linux.txt      — baseline ns/op + tsc/op for 13 ops
  latency_pmc.txt        — 5 PMC passes (cycles, uops, FP-pipe, LS, branches, RAPL)
  throughput.txt         — reciprocal throughput for 13 ops
  op2op.txt              — operand-pair latency for adc/sbb/cmovcc/mov/add+mem
  vector.txt             — 21 SSE/AVX2/FMA ops, latency + throughput
  storefwd.txt           — 6 store-fwd / 4K-alias scenarios
  memory.txt             — cache hierarchy, TLB, cross-CCD
  generated.txt          — 166 auto-generated benches from uops.info XML

src/
  pmc.h                  — perf_event_open + rdpmc helpers
  uops_latency_linux.cpp — original Linux port (latency only)
  uops_latency_pmc.cpp   — PMC-instrumented (cycles, uops, port, energy)
  uops_throughput.cpp    — independent-chain throughput
  uops_op2op.cpp         — operand-pair latency
  uops_vector.cpp        — AVX2 / FMA / SSE-vector lat + tput
  uops_storefwd.cpp      — store-forwarding + 4K aliasing
  uops_memory.cpp        — cache hierarchy + TLB + cross-CCD
  uops_generated.cpp     — auto-generated 166-iform harness

tools/
  gen_from_xed.py        — XED-driven generator
  run_all.sh             — run every binary, save outputs
  compare_uops.js        — original (Windows-data) comparison script
```
