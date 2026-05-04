# Windows PMU notes

The short answer is: yes, Windows can do more than rough TSC timing. It has real PMU access paths. The catch is that the useful ones move through ETW/WPR/xperf, VTune, Intel PCM, or a driver, not through ordinary user-mode `rdpmc` in this shell.

## What this machine exposes

`wpr -pmcsources` and `xperf -pmcsources` both work here. The available sources include:

- `InstructionRetired`
- `InstructionsRetiredFixed`
- `TotalCycles`
- `UnhaltedCoreCycles`
- `UnhaltedCoreCyclesFixed`
- `UnhaltedReferenceCyclesFixed`
- `TotalIssues`
- branches, branch mispredicts, LLC references/misses, and `LbrInserts`

So the PMU surface exists. This is not a dead end.

## What failed locally

This Codex shell is running at medium integrity, not elevated Administrator. It can enumerate counters, but starting the kernel logger failed:

- `xperf -PmcProfile InstructionRetired` failed with `Failed to configure counters`.
- `xperf -pmc InstructionRetired,TotalCycles CSWITCH` failed with `Access is denied. (0x5)`.

That means the next step is an Administrator PowerShell run, not a rewrite of the benchmark.

## Windows routes

1. WPR/xperf ETW PMU mode: Microsoft documents two modes. One logs PMC values on selected ETW events such as `CSwitch`; the other samples when a PMU counter overflows. This is real PMU data, but it is trace/sampling oriented rather than nanoBench-style exact per-region counter deltas.

2. WPR custom profile extension: Windows can define extra microarchitectural PMU events in a `.wprp` profile or registry. This matters for port-like, model-specific, non-architectural events that are not in the default HAL list.

3. Intel VTune: VTune supports hardware event-based sampling on Windows. It uses PMU counter overflow sampling and has Microarchitecture Exploration, but it is still mostly a profiler view unless we build a very deliberate harness around one region at a time.

4. Intel PCM: PCM supports Windows and can program/read counters through an MSR driver. This is closer to what we would want for exact before/after counter deltas, but it needs Administrator rights, driver install/signing, and careful setup.

5. Custom driver / RDPMC enablement: closest to uops.info/nanoBench control. A driver can program fixed/generic counters and enable user-mode `rdpmc`, or expose read/reset ioctls. This is the serious route if we want publishable port/uop-count data on Windows.

6. `QueryThreadCycleTime`: useful for timing, and already used as one of our primary local signals, but it is not a general PMU API. Microsoft explicitly warns not to convert it blindly to elapsed time because the underlying CPU timer behavior varies.

## Port-counter angle

Intel's public PerfMon tables for Alder Lake list Golden Cove/Raptor Cove P-core execution-port events. The useful family for uops.info-style port pressure is:

- `UOPS_DISPATCHED.PORT_0`: EventSel `0xB2`, UMask `0x01`
- `UOPS_DISPATCHED.PORT_1`: EventSel `0xB2`, UMask `0x02`
- `UOPS_DISPATCHED.PORT_2_3_10`: EventSel `0xB2`, UMask `0x04`
- `UOPS_DISPATCHED.PORT_4_9`: EventSel `0xB2`, UMask `0x10`
- `UOPS_DISPATCHED.PORT_5_11`: EventSel `0xB2`, UMask `0x20`
- `UOPS_DISPATCHED.PORT_6`: EventSel `0xB2`, UMask `0x40`
- `UOPS_DISPATCHED.PORT_7_8`: EventSel `0xB2`, UMask `0x80`

Those are not in this machine's default `wpr -pmcsources` list, but Microsoft documents custom Intel PMU sources using `Event` and `Unit` attributes in WPRP. The open question is how cleanly Windows applies model-specific custom events on a hybrid CPU when the P-core and E-core PMUs differ. Pinning the workload to CPU 0 helps the benchmark side, but the trace provider may still try to program the source system-wide.

## Files added

- `windows_pmu_sampling.wprp`: WPR profile with `CSwitch` counter logging and retired-instruction overflow sampling.
- `run_windows_pmu_admin.ps1`: Administrator runner that records PMU traces around `uops_research.exe`.
- `windows_pmu_ports_pcore_experimental.wprp`: custom WPR profile for Raptor Cove/Golden Cove P-core port-dispatch counters. It validates with `wpr -profiles`, but it still needs an Administrator run to prove Windows will program those events on this hybrid CPU.
- `run_windows_pmu_ports_admin.ps1`: Administrator runner for the experimental P-core port profile.

## Magic-trace check

`magic-trace` is still a no for native Windows today. Its own documentation says it only supports Linux because it relies on Linux `perf`; the underlying Intel PT hardware is not the blocker, the OS integration/tooling is.

## Current conclusion

Windows matters, but it does not kill the project. For latency/throughput, our current benchmark is fine. For uops.info-grade uop count and port pressure, Windows needs one of these:

- Administrator WPR/xperf traces for approximate PMU-backed profiling.
- VTune or Intel PCM for richer hardware-counter access.
- A small local driver for exact benchmark-region counter programming/reading.

The Administrator WPR path is ready to try first.

Sources:

- Microsoft, Recording Hardware Performance (PMU) Events: https://learn.microsoft.com/en-us/windows-hardware/test/wpt/recording-pmu-events
- Microsoft, QueryThreadCycleTime: https://learn.microsoft.com/en-us/windows/win32/api/realtimeapiset/nf-realtimeapiset-querythreadcycletime
- Intel VTune, Hardware Event-based Sampling Collection: https://www.intel.com/content/www/us/en/docs/vtune-profiler/user-guide/2023-0/hw-event-based-sampling-collection.html
- Intel PCM: https://github.com/intel/pcm
- Intel PCM Windows HOWTO: https://github.com/intel/pcm/blob/master/doc/WINDOWS_HOWTO.md
- Intel PerfMon, Alder Lake P-core events: https://perfmon-events.intel.com/platforms/alderlake/core-events/p-core/
- magic-trace: https://github.com/janestreet/magic-trace
