// PMC-backed extension of uops_latency_linux.cpp. Reports per-instruction:
//   - true core cycles (PERF_COUNT_HW_CPU_CYCLES, fixed counter)
//   - retired instructions (PERF_COUNT_HW_INSTRUCTIONS)
//   - retired uops (Zen 3 raw event 0xC1)
//   - FP pipe distribution (FpuPipeAssignment.Total{0..3}, raw event 0x000)
//   - LS dispatch distribution (LsDispatch.{ld,st,ld_st}_dispatch, raw 0x029)
//   - package energy (power/energy-pkg/ via RAPL PMU, system-wide)
//
// Multiple passes because Zen 3 has 6 general-purpose counters per core.

#define _GNU_SOURCE

#include "pmc.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include <pthread.h>
#include <sched.h>
#include <time.h>
#include <unistd.h>

#include <cpuid.h>
#include <x86intrin.h>

#define REP2(x) x x
#define REP4(x) REP2(x) REP2(x)
#define REP8(x) REP4(x) REP4(x)
#define REP16(x) REP8(x) REP8(x)
#define REP32(x) REP16(x) REP16(x)
#define REP64(x) REP32(x) REP32(x)
static constexpr int OPS_PER_ITER = 64;

static volatile uint64_t sink_u64 = 0x9e3779b97f4a7c15ull;
static volatile double sink_f64 = 1.0000001;
static std::vector<uintptr_t> pointer_ring;

using BenchFn = uint64_t (*)(int);

__attribute__((noinline)) static uint64_t bench_empty(int iters) {
    uint64_t x = sink_u64;
    for (int i = 0; i < iters; ++i) {
        asm volatile("" : [x] "+r"(x) : : "cc");
    }
    sink_u64 = x;
    return x;
}

__attribute__((noinline)) static uint64_t bench_add_r64_imm(int iters) {
    uint64_t x = sink_u64 | 1ull;
    for (int i = 0; i < iters; ++i) {
        asm volatile(REP64("addq $1, %[x]\n\t") : [x] "+r"(x) : : "cc");
    }
    sink_u64 = x;
    return x;
}

__attribute__((noinline)) static uint64_t bench_add_r64_reg(int iters) {
    uint64_t x = sink_u64 | 1ull;
    const uint64_t y = 0x9e3779b97f4a7c15ull;
    for (int i = 0; i < iters; ++i) {
        asm volatile(REP64("addq %[y], %[x]\n\t") : [x] "+&r"(x) : [y] "r"(y) : "cc");
    }
    sink_u64 = x;
    return x;
}

__attribute__((noinline)) static uint64_t bench_imul_r64_imm(int iters) {
    uint64_t x = sink_u64 | 1ull;
    for (int i = 0; i < iters; ++i) {
        asm volatile(REP64("imulq $3, %[x], %[x]\n\t") : [x] "+r"(x) : : "cc");
    }
    sink_u64 = x;
    return x;
}

__attribute__((noinline)) static uint64_t bench_shl_r64_1(int iters) {
    uint64_t x = sink_u64 | 1ull;
    for (int i = 0; i < iters; ++i) {
        asm volatile(REP64("shlq $1, %[x]\n\t") : [x] "+r"(x) : : "cc");
    }
    sink_u64 = x;
    return x;
}

__attribute__((noinline)) static uint64_t bench_ror_r64_7(int iters) {
    uint64_t x = sink_u64 | 1ull;
    for (int i = 0; i < iters; ++i) {
        asm volatile(REP64("rorq $7, %[x]\n\t") : [x] "+r"(x) : : "cc");
    }
    sink_u64 = x;
    return x;
}

__attribute__((noinline)) static uint64_t bench_popcnt_r64(int iters) {
    uint64_t x = sink_u64 | 0xf0f0f0f0f0f0f0f1ull;
    for (int i = 0; i < iters; ++i) {
        asm volatile(REP64("popcntq %[x], %[x]\n\t") : [x] "+r"(x) : : "cc");
    }
    sink_u64 = x;
    return x;
}

__attribute__((noinline)) static uint64_t bench_lzcnt_r64(int iters) {
    uint64_t x = sink_u64 | 0x0102030405060708ull;
    for (int i = 0; i < iters; ++i) {
        asm volatile(REP64("lzcntq %[x], %[x]\n\t") : [x] "+r"(x) : : "cc");
    }
    sink_u64 = x;
    return x;
}

__attribute__((noinline)) static uint64_t bench_crc32_r64(int iters) {
    uint64_t x = sink_u64 | 0x123456789abcdef1ull;
    for (int i = 0; i < iters; ++i) {
        asm volatile(REP64("crc32q %[x], %[x]\n\t") : [x] "+r"(x) : : "cc");
    }
    sink_u64 = x;
    return x;
}

__attribute__((noinline)) static uint64_t bench_addsd_xmm(int iters) {
    double v = sink_f64;
    const double c = 1.0;
    for (int i = 0; i < iters; ++i) {
        asm volatile(REP64("addsd %[c], %[v]\n\t") : [v] "+&x"(v) : [c] "x"(c));
    }
    sink_f64 = v;
    return static_cast<uint64_t>(v);
}

__attribute__((noinline)) static uint64_t bench_mulsd_xmm(int iters) {
    double v = sink_f64;
    const double c = 1.0000000001;
    for (int i = 0; i < iters; ++i) {
        asm volatile(REP64("mulsd %[c], %[v]\n\t") : [v] "+&x"(v) : [c] "x"(c));
    }
    sink_f64 = v;
    return static_cast<uint64_t>(v);
}

__attribute__((noinline)) static uint64_t bench_divsd_xmm(int iters) {
    double v = sink_f64 + 1000.0;
    const double c = 1.0000000001;
    for (int i = 0; i < iters; ++i) {
        asm volatile(REP64("divsd %[c], %[v]\n\t") : [v] "+&x"(v) : [c] "x"(c));
    }
    sink_f64 = v;
    return static_cast<uint64_t>(v);
}

__attribute__((noinline)) static uint64_t bench_sqrtsd_xmm(int iters) {
    double v = sink_f64 + 123.0;
    for (int i = 0; i < iters; ++i) {
        asm volatile(REP64("sqrtsd %[v], %[v]\n\t") : [v] "+x"(v));
    }
    sink_f64 = v;
    return static_cast<uint64_t>(v);
}

__attribute__((noinline)) static uint64_t bench_l1_pointer_chase(int iters) {
    uintptr_t p = pointer_ring.empty() ? 0 : reinterpret_cast<uintptr_t>(&pointer_ring[0]);
    for (int i = 0; i < iters; ++i) {
        asm volatile(REP64("movq (%[p]), %[p]\n\t") : [p] "+r"(p) : : "memory");
    }
    sink_u64 = p;
    return p;
}

struct Benchmark {
    std::string name;
    BenchFn fn;
};

static const std::vector<Benchmark> g_benches = {
    {"add r64, imm", bench_add_r64_imm},
    {"add r64, reg", bench_add_r64_reg},
    {"imul r64, imm", bench_imul_r64_imm},
    {"shl r64, 1", bench_shl_r64_1},
    {"ror r64, 7", bench_ror_r64_7},
    {"popcnt r64", bench_popcnt_r64},
    {"lzcnt r64", bench_lzcnt_r64},
    {"crc32 r64,r64", bench_crc32_r64},
    {"addsd xmm", bench_addsd_xmm},
    {"mulsd xmm", bench_mulsd_xmm},
    {"divsd xmm", bench_divsd_xmm},
    {"sqrtsd xmm", bench_sqrtsd_xmm},
    {"L1 ptr chase", bench_l1_pointer_chase},
};

static double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    size_t n = values.size();
    if (n == 0) return 0.0;
    if ((n & 1u) != 0) return values[n / 2];
    return (values[n / 2 - 1] + values[n / 2]) / 2.0;
}

static void init_pointer_ring(size_t count) {
    pointer_ring.assign(count, 0);
    std::vector<size_t> perm(count);
    std::iota(perm.begin(), perm.end(), 0);
    uint64_t state = 0x4d595df4d0f33173ull;
    for (size_t i = count - 1; i > 0; --i) {
        state = state * 2862933555777941757ull + 3037000493ull;
        size_t j = static_cast<size_t>(state % (i + 1));
        std::swap(perm[i], perm[j]);
    }
    for (size_t i = 0; i < count; ++i) {
        size_t cur = perm[i];
        size_t next = perm[(i + 1) % count];
        pointer_ring[cur] = reinterpret_cast<uintptr_t>(&pointer_ring[next]);
    }
}

static bool pin_to_logical_cpu(unsigned cpu) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    return pthread_setaffinity_np(pthread_self(), sizeof(set), &set) == 0;
}

// Run a single bench `samples` times with the given counters open. Returns the
// per-op median for each counter. Subtracts an empty-loop baseline.
struct CounterSet {
    std::vector<PmcCounter*> counters;
    std::vector<std::string> names;
};

static std::vector<double> measure_per_op(BenchFn fn, int iters, int samples,
                                          CounterSet& set,
                                          const std::vector<double>& empty_medians) {
    const double ops = static_cast<double>(iters) * OPS_PER_ITER;

    // Warmup
    for (int i = 0; i < 8; ++i) fn(iters);

    std::vector<std::vector<double>> per_counter(set.counters.size());
    for (auto& v : per_counter) v.reserve(samples);

    for (int s = 0; s < samples; ++s) {
        std::vector<uint64_t> before(set.counters.size());
        std::vector<uint64_t> after(set.counters.size());
        for (size_t i = 0; i < set.counters.size(); ++i) {
            before[i] = pmc_read(*set.counters[i]);
        }
        _mm_lfence();
        fn(iters);
        _mm_lfence();
        for (size_t i = 0; i < set.counters.size(); ++i) {
            after[i] = pmc_read(*set.counters[i]);
        }
        for (size_t i = 0; i < set.counters.size(); ++i) {
            double delta = static_cast<double>(after[i] - before[i]);
            per_counter[i].push_back((delta - empty_medians[i]) / ops);
        }
    }

    std::vector<double> medians(set.counters.size());
    for (size_t i = 0; i < set.counters.size(); ++i) {
        medians[i] = median(per_counter[i]);
    }
    return medians;
}

static std::vector<double> measure_empty_baselines(int iters, int samples,
                                                   CounterSet& set) {
    for (int i = 0; i < 8; ++i) bench_empty(iters);
    std::vector<std::vector<double>> per_counter(set.counters.size());
    for (auto& v : per_counter) v.reserve(samples);
    for (int s = 0; s < samples; ++s) {
        std::vector<uint64_t> before(set.counters.size());
        std::vector<uint64_t> after(set.counters.size());
        for (size_t i = 0; i < set.counters.size(); ++i) {
            before[i] = pmc_read(*set.counters[i]);
        }
        _mm_lfence();
        bench_empty(iters);
        _mm_lfence();
        for (size_t i = 0; i < set.counters.size(); ++i) {
            after[i] = pmc_read(*set.counters[i]);
        }
        for (size_t i = 0; i < set.counters.size(); ++i) {
            per_counter[i].push_back(static_cast<double>(after[i] - before[i]));
        }
    }
    std::vector<double> medians(set.counters.size());
    for (size_t i = 0; i < set.counters.size(); ++i) {
        medians[i] = median(per_counter[i]);
    }
    return medians;
}

static void run_pass(const std::string& title, CounterSet& set,
                     int iters, int samples) {
    std::cout << "\n=== " << title << " ===\n";
    auto empties = measure_empty_baselines(iters, samples, set);

    std::cout << std::left << std::setw(20) << "benchmark";
    for (const auto& n : set.names) {
        std::cout << std::right << std::setw(16) << n;
    }
    std::cout << "\n" << std::string(20 + 16 * set.names.size(), '-') << "\n";

    for (const auto& b : g_benches) {
        auto m = measure_per_op(b.fn, iters, samples, set, empties);
        std::cout << std::left << std::setw(20) << b.name;
        for (double v : m) {
            std::cout << std::right << std::setw(16) << std::fixed
                      << std::setprecision(4) << v;
        }
        std::cout << "\n";
    }
}

int main(int argc, char** argv) {
    unsigned logical_cpu = (argc > 1) ? std::strtoul(argv[1], nullptr, 10) : 0;
    int iters = (argc > 2) ? std::max(1, std::atoi(argv[2])) : 100000;
    int samples = (argc > 3) ? std::max(3, std::atoi(argv[3])) : 31;

    pin_to_logical_cpu(logical_cpu);
    usleep(10000);
    init_pointer_ring(4096);

    std::cout << "uops PMC harness (linux, perf_event_open + rdpmc)\n";
    std::cout << "logical_cpu=" << logical_cpu
              << " current=" << sched_getcpu()
              << " iters=" << iters << " samples=" << samples
              << " ops_per_sample=" << (static_cast<uint64_t>(iters) * OPS_PER_ITER)
              << "\n";

    // ----- Pass 1: cycles + instructions + retired uops --------------------
    PmcCounter c_cycles, c_insns, c_uops;
    bool ok1 = pmc_open(c_cycles, PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES);
    bool ok2 = pmc_open(c_insns, PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS);
    bool ok3 = pmc_open(c_uops, PERF_TYPE_RAW, amd_raw(0xC1, 0x00));
    if (ok1 && ok2 && ok3) {
        CounterSet s;
        s.counters = {&c_cycles, &c_insns, &c_uops};
        s.names = {"cycles/op", "insns/op", "uops/op"};
        run_pass("Pass 1: cycles, instructions, retired uops", s, iters, samples);
    } else {
        std::cerr << "Pass 1 setup failed; check perf_event_paranoid and rdpmc.\n";
    }
    pmc_close(c_cycles);
    pmc_close(c_insns);
    pmc_close(c_uops);

    // ----- Pass 2: FP pipe assignment per pipe -----------------------------
    // FpuPipeAssignment event=0x000 with umasks 0x01,0x02,0x04,0x08 for pipes 0..3.
    PmcCounter cyc2, fp0, fp1, fp2, fp3;
    bool ok = pmc_open(cyc2, PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES);
    ok &= pmc_open(fp0, PERF_TYPE_RAW, amd_raw(0x000, 0x01));
    ok &= pmc_open(fp1, PERF_TYPE_RAW, amd_raw(0x000, 0x02));
    ok &= pmc_open(fp2, PERF_TYPE_RAW, amd_raw(0x000, 0x04));
    ok &= pmc_open(fp3, PERF_TYPE_RAW, amd_raw(0x000, 0x08));
    if (ok) {
        CounterSet s;
        s.counters = {&cyc2, &fp0, &fp1, &fp2, &fp3};
        s.names = {"cycles", "fp_p0", "fp_p1", "fp_p2", "fp_p3"};
        run_pass("Pass 2: FP pipe assignment", s, iters, samples);
    } else {
        std::cerr << "Pass 2 setup failed.\n";
    }
    pmc_close(cyc2); pmc_close(fp0); pmc_close(fp1); pmc_close(fp2); pmc_close(fp3);

    // ----- Pass 3: LS dispatch breakdown -----------------------------------
    // LsDispatch event=0x029 umasks 0x01=ld, 0x02=store, 0x04=ld_st_op.
    PmcCounter cyc3, ls_ld, ls_st, ls_ldst;
    ok = pmc_open(cyc3, PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES);
    ok &= pmc_open(ls_ld, PERF_TYPE_RAW, amd_raw(0x029, 0x01));
    ok &= pmc_open(ls_st, PERF_TYPE_RAW, amd_raw(0x029, 0x02));
    ok &= pmc_open(ls_ldst, PERF_TYPE_RAW, amd_raw(0x029, 0x04));
    if (ok) {
        CounterSet s;
        s.counters = {&cyc3, &ls_ld, &ls_st, &ls_ldst};
        s.names = {"cycles", "ld_disp", "st_disp", "ldst_disp"};
        run_pass("Pass 3: LS dispatch", s, iters, samples);
    } else {
        std::cerr << "Pass 3 setup failed.\n";
    }
    pmc_close(cyc3); pmc_close(ls_ld); pmc_close(ls_st); pmc_close(ls_ldst);

    // ----- Pass 4: Mispredicts + branches (sanity) -------------------------
    PmcCounter cyc4, brins, brmis;
    ok = pmc_open(cyc4, PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES);
    ok &= pmc_open(brins, PERF_TYPE_HARDWARE, PERF_COUNT_HW_BRANCH_INSTRUCTIONS);
    ok &= pmc_open(brmis, PERF_TYPE_HARDWARE, PERF_COUNT_HW_BRANCH_MISSES);
    if (ok) {
        CounterSet s;
        s.counters = {&cyc4, &brins, &brmis};
        s.names = {"cycles", "branches", "br_misses"};
        run_pass("Pass 4: branches + mispredicts (sanity)", s, iters, samples);
    } else {
        std::cerr << "Pass 4 setup failed.\n";
    }
    pmc_close(cyc4); pmc_close(brins); pmc_close(brmis);

    // ----- Pass 5: RAPL package energy -------------------------------------
    // RAPL is a separate PMU type with the energy-pkg event.
    uint32_t rapl_type = 0;
    uint64_t rapl_config = 0;
    bool rapl_ok = sysfs_read_u32(
        "/sys/bus/event_source/devices/power/type", &rapl_type);
    rapl_ok &= sysfs_read_perf_event(
        "/sys/bus/event_source/devices/power/events/energy-pkg", &rapl_config);
    if (rapl_ok) {
        PmcCounter rapl;
        // RAPL needs a per-cpu open (cpu = our pinned cpu).
        if (pmc_open_system(rapl, rapl_type, rapl_config,
                            static_cast<int>(logical_cpu))) {
            std::cout << "\n=== Pass 5: RAPL package energy (per op) ===\n";
            // Read scale factor from sysfs (units: Joules per count).
            FILE* f = std::fopen(
                "/sys/bus/event_source/devices/power/events/energy-pkg.scale", "r");
            double scale = 2.3283064365386963e-10; // 2^-32 J default
            if (f) {
                if (std::fscanf(f, "%lf", &scale) != 1) {
                    scale = 2.3283064365386963e-10;
                }
                std::fclose(f);
            }

            const double ops_per = static_cast<double>(iters) * OPS_PER_ITER;
            std::cout << std::left << std::setw(20) << "benchmark"
                      << std::right << std::setw(20) << "energy_pJ_per_op"
                      << "\n" << std::string(40, '-') << "\n";

            // Warmup
            for (int i = 0; i < 4; ++i) bench_empty(iters);

            // Empty-loop baseline
            std::vector<double> empties;
            empties.reserve(samples);
            for (int s = 0; s < samples; ++s) {
                uint64_t a = pmc_read(rapl);
                bench_empty(iters);
                uint64_t b = pmc_read(rapl);
                empties.push_back(static_cast<double>(b - a));
            }
            double empty_med = median(empties);

            for (const auto& b : g_benches) {
                std::vector<double> samps;
                samps.reserve(samples);
                for (int s = 0; s < samples; ++s) {
                    uint64_t a = pmc_read(rapl);
                    b.fn(iters);
                    uint64_t bb = pmc_read(rapl);
                    samps.push_back(static_cast<double>(bb - a) - empty_med);
                }
                double counts = median(samps);
                double joules = counts * scale;
                double pj_per_op = (joules * 1e12) / ops_per;
                std::cout << std::left << std::setw(20) << b.name
                          << std::right << std::setw(20) << std::fixed
                          << std::setprecision(2) << pj_per_op << "\n";
            }
            pmc_close(rapl);
        } else {
            std::cerr << "RAPL open failed (system-wide event may need higher cap).\n";
        }
    } else {
        std::cerr << "RAPL sysfs not readable; skipping energy pass.\n";
    }

    return 0;
}
