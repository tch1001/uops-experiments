# Extended uops.info-style comparison

uops.info current XML date: 2026-03-29. Direct XML search found no Raptor Lake entry, so this compares the local Raptor Lake-HX P-core against Alder Lake-P and the local E-core against Alder Lake-E.

Normalization: P-core 0.529610 primary units ~= 1 cycle; E-core 0.667077 primary units ~= 1 cycle. The scale uses the median of local one-cycle dependency probes: ADD reg/reg, SHL r64,1, and ROR r64,7.

When uops.info reports latency as a min/max range rather than a single cycle count, the reference table below shows the midpoint. That mainly affects divide and square-root instructions.

## Selected Latency Comparison

| benchmark | local P est | ADL-P ref | local E est | ADL-E ref |
| --- | ---: | ---: | ---: | ---: |
| ADD_R64_R64 | 1.00 | 1.00 | 1.00 | 1.00 |
| IMUL_R64_R64_I32 | 3.03 | 3.00 | 4.97 | 5.00 |
| POPCNT_R64_R64 | 2.96 | 3.00 | 2.88 | 3.00 |
| ADDSD_XMM_XMM | 1.89 | 2.00 | 2.90 | 3.00 |
| MULSD_XMM_XMM | 3.93 | 4.00 | 3.85 | 4.00 |
| DIVSD_XMM_XMM | 13.63 | 14.00 | 12.43 | 14.00 |
| SQRTSD_XMM_XMM | 12.25 | 16.00 | 15.50 | 18.00 |

## Selected Throughput Comparison

| benchmark | local P est | ADL-P ref | local E est | ADL-E ref |
| --- | ---: | ---: | ---: | ---: |
| ADD_R64_R64 | 0.18 | 0.20 | 0.23 | 0.25 |
| IMUL_R64_R64_I32 | 0.88 | 1.00 | 0.64 | 0.50 |
| POPCNT_R64_R64 | 0.88 | 1.00 | 0.91 | 1.00 |
| ADDSD_XMM_XMM | 0.45 | 0.50 | 0.46 | 0.50 |
| MULSD_XMM_XMM | 0.44 | 0.50 | 0.55 | 0.50 |
| DIVSD_XMM_XMM | 3.49 | 4.00 | 7.29 | 8.00 |
| SQRTSD_XMM_XMM | 3.92 | 4.50 | 10.96 | 12.00 |

Full machine-readable comparison is in `comparison_pcore_adlp.csv` and `comparison_ecore_adle.csv`.

Important limitation: local values are normalized from Windows timing units because user-mode RDPMC is blocked. uops.info values are hardware-counter-backed measurements, so close agreement is more meaningful than small deltas.
