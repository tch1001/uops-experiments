#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0600
#endif

#include <windows.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

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
static volatile LONG rdpmc_faulted = 0;
static bool rdpmc_fixed_core_available = false;
static constexpr uint32_t RDPMC_FIXED_CORE_CYCLES = (1u << 30) | 1u;

using BenchFn = uint64_t (*)(int);

struct CpuFeatures {
    bool invariant_tsc = false;
    bool sse42 = false;
    bool popcnt = false;
    bool lzcnt = false;
};

static bool cpuid_leaf(uint32_t leaf, uint32_t subleaf,
                       uint32_t& eax, uint32_t& ebx,
                       uint32_t& ecx, uint32_t& edx) {
    uint32_t max_leaf = __get_cpuid_max(leaf & 0x80000000u, nullptr);
    if (max_leaf < leaf) {
        return false;
    }
    __cpuid_count(leaf, subleaf, eax, ebx, ecx, edx);
    return true;
}

static CpuFeatures get_cpu_features() {
    CpuFeatures f;
    uint32_t eax = 0, ebx = 0, ecx = 0, edx = 0;
    if (cpuid_leaf(1, 0, eax, ebx, ecx, edx)) {
        f.sse42 = (ecx & (1u << 20)) != 0;
        f.popcnt = (ecx & (1u << 23)) != 0;
    }
    if (cpuid_leaf(0x80000001u, 0, eax, ebx, ecx, edx)) {
        f.lzcnt = (ecx & (1u << 5)) != 0;
    }
    if (cpuid_leaf(0x80000007u, 0, eax, ebx, ecx, edx)) {
        f.invariant_tsc = (edx & (1u << 8)) != 0;
    }
    return f;
}

static uint64_t read_tsc_begin() {
    _mm_lfence();
    return __rdtsc();
}

static uint64_t read_tsc_end() {
    unsigned aux = 0;
    uint64_t t = __rdtscp(&aux);
    _mm_lfence();
    return t;
}

static uint64_t read_thread_cycles() {
    ULONG64 cycles = 0;
    QueryThreadCycleTime(GetCurrentThread(), &cycles);
    return static_cast<uint64_t>(cycles);
}

static LONG CALLBACK rdpmc_exception_handler(EXCEPTION_POINTERS* ep) {
    DWORD code = ep->ExceptionRecord->ExceptionCode;
    if (code == EXCEPTION_ILLEGAL_INSTRUCTION || code == EXCEPTION_PRIV_INSTRUCTION) {
        InterlockedExchange(&rdpmc_faulted, 1);
#if defined(__x86_64__) || defined(_M_X64)
        ep->ContextRecord->Rip += 2;  // rdpmc is 0f 33.
        return EXCEPTION_CONTINUE_EXECUTION;
#endif
    }
    return EXCEPTION_CONTINUE_SEARCH;
}

static uint64_t read_rdpmc(uint32_t counter) {
    uint32_t lo = 0;
    uint32_t hi = 0;
    InterlockedExchange(&rdpmc_faulted, 0);
    asm volatile(".byte 0x0f, 0x33"
                 : "=a"(lo), "=d"(hi)
                 : "c"(counter));
    if (rdpmc_faulted) {
        return 0;
    }
    return (static_cast<uint64_t>(hi) << 32) | lo;
}

static uint64_t read_rdpmc_ordered(uint32_t counter) {
    _mm_lfence();
    uint64_t v = read_rdpmc(counter);
    _mm_lfence();
    return v;
}

static bool detect_rdpmc_fixed_core() {
    AddVectoredExceptionHandler(1, rdpmc_exception_handler);
    uint64_t a = read_rdpmc_ordered(RDPMC_FIXED_CORE_CYCLES);
    if (rdpmc_faulted) {
        return false;
    }
    volatile uint64_t x = 1;
    for (int i = 0; i < 10000; ++i) {
        x += static_cast<uint64_t>(i);
    }
    sink_u64 ^= x;
    uint64_t b = read_rdpmc_ordered(RDPMC_FIXED_CORE_CYCLES);
    return !rdpmc_faulted && b > a;
}

static bool pin_to_logical_cpu(unsigned cpu) {
    if (cpu >= sizeof(DWORD_PTR) * 8) {
        return false;
    }
    DWORD_PTR mask = static_cast<DWORD_PTR>(1) << cpu;
    return SetThreadAffinityMask(GetCurrentThread(), mask) != 0;
}

static void raise_priority() {
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);
}

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
        asm volatile(REP64("addq $1, %[x]\n\t")
                     : [x] "+r"(x)
                     :
                     : "cc");
    }
    sink_u64 = x;
    return x;
}

__attribute__((noinline)) static uint64_t bench_add_r64_reg(int iters) {
    uint64_t x = sink_u64 | 1ull;
    const uint64_t y = 0x9e3779b97f4a7c15ull;
    for (int i = 0; i < iters; ++i) {
        asm volatile(REP64("addq %[y], %[x]\n\t")
                     : [x] "+&r"(x)
                     : [y] "r"(y)
                     : "cc");
    }
    sink_u64 = x;
    return x;
}

__attribute__((noinline)) static uint64_t bench_imul_r64_imm(int iters) {
    uint64_t x = sink_u64 | 1ull;
    for (int i = 0; i < iters; ++i) {
        asm volatile(REP64("imulq $3, %[x], %[x]\n\t")
                     : [x] "+r"(x)
                     :
                     : "cc");
    }
    sink_u64 = x;
    return x;
}

__attribute__((noinline)) static uint64_t bench_shl_r64_1(int iters) {
    uint64_t x = sink_u64 | 1ull;
    for (int i = 0; i < iters; ++i) {
        asm volatile(REP64("shlq $1, %[x]\n\t")
                     : [x] "+r"(x)
                     :
                     : "cc");
    }
    sink_u64 = x;
    return x;
}

__attribute__((noinline)) static uint64_t bench_ror_r64_7(int iters) {
    uint64_t x = sink_u64 | 1ull;
    for (int i = 0; i < iters; ++i) {
        asm volatile(REP64("rorq $7, %[x]\n\t")
                     : [x] "+r"(x)
                     :
                     : "cc");
    }
    sink_u64 = x;
    return x;
}

__attribute__((noinline)) static uint64_t bench_popcnt_r64(int iters) {
    uint64_t x = sink_u64 | 0xf0f0f0f0f0f0f0f1ull;
    for (int i = 0; i < iters; ++i) {
        asm volatile(REP64("popcntq %[x], %[x]\n\t")
                     : [x] "+r"(x)
                     :
                     : "cc");
    }
    sink_u64 = x;
    return x;
}

__attribute__((noinline)) static uint64_t bench_lzcnt_r64(int iters) {
    uint64_t x = sink_u64 | 0x0102030405060708ull;
    for (int i = 0; i < iters; ++i) {
        asm volatile(REP64("lzcntq %[x], %[x]\n\t")
                     : [x] "+r"(x)
                     :
                     : "cc");
    }
    sink_u64 = x;
    return x;
}

__attribute__((noinline)) static uint64_t bench_crc32_r64(int iters) {
    uint64_t x = sink_u64 | 0x123456789abcdef1ull;
    for (int i = 0; i < iters; ++i) {
        asm volatile(REP64("crc32q %[x], %[x]\n\t")
                     : [x] "+r"(x)
                     :
                     : "cc");
    }
    sink_u64 = x;
    return x;
}

__attribute__((noinline)) static uint64_t bench_addsd_xmm(int iters) {
    double v = sink_f64;
    const double c = 1.0;
    for (int i = 0; i < iters; ++i) {
        asm volatile(REP64("addsd %[c], %[v]\n\t")
                     : [v] "+&x"(v)
                     : [c] "x"(c));
    }
    sink_f64 = v;
    return static_cast<uint64_t>(v);
}

__attribute__((noinline)) static uint64_t bench_mulsd_xmm(int iters) {
    double v = sink_f64;
    const double c = 1.0000000001;
    for (int i = 0; i < iters; ++i) {
        asm volatile(REP64("mulsd %[c], %[v]\n\t")
                     : [v] "+&x"(v)
                     : [c] "x"(c));
    }
    sink_f64 = v;
    return static_cast<uint64_t>(v);
}

__attribute__((noinline)) static uint64_t bench_divsd_xmm(int iters) {
    double v = sink_f64 + 1000.0;
    const double c = 1.0000000001;
    for (int i = 0; i < iters; ++i) {
        asm volatile(REP64("divsd %[c], %[v]\n\t")
                     : [v] "+&x"(v)
                     : [c] "x"(c));
    }
    sink_f64 = v;
    return static_cast<uint64_t>(v);
}

__attribute__((noinline)) static uint64_t bench_sqrtsd_xmm(int iters) {
    double v = sink_f64 + 123.0;
    for (int i = 0; i < iters; ++i) {
        asm volatile(REP64("sqrtsd %[v], %[v]\n\t")
                     : [v] "+x"(v));
    }
    sink_f64 = v;
    return static_cast<uint64_t>(v);
}

__attribute__((noinline)) static uint64_t bench_l1_pointer_chase(int iters) {
    uintptr_t p = pointer_ring.empty() ? 0 : reinterpret_cast<uintptr_t>(&pointer_ring[0]);
    for (int i = 0; i < iters; ++i) {
        asm volatile(REP64("movq (%[p]), %[p]\n\t")
                     : [p] "+r"(p)
                     :
                     : "memory");
    }
    sink_u64 = p;
    return p;
}

struct Timing {
    uint64_t tsc = 0;
    uint64_t thread_cycles = 0;
    uint64_t pmc_core_cycles = 0;
};

static Timing measure_once(BenchFn fn, int iters) {
    uint64_t tc0 = read_thread_cycles();
    uint64_t pc0 = rdpmc_fixed_core_available
                       ? read_rdpmc_ordered(RDPMC_FIXED_CORE_CYCLES)
                       : 0;
    uint64_t t0 = read_tsc_begin();
    fn(iters);
    uint64_t t1 = read_tsc_end();
    uint64_t pc1 = rdpmc_fixed_core_available
                       ? read_rdpmc_ordered(RDPMC_FIXED_CORE_CYCLES)
                       : 0;
    uint64_t tc1 = read_thread_cycles();
    return Timing{t1 - t0, tc1 - tc0, pc1 - pc0};
}

static double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    size_t n = values.size();
    if (n == 0) {
        return 0.0;
    }
    if ((n & 1u) != 0) {
        return values[n / 2];
    }
    return (values[n / 2 - 1] + values[n / 2]) / 2.0;
}

static double percentile(std::vector<double> values, double p) {
    std::sort(values.begin(), values.end());
    if (values.empty()) {
        return 0.0;
    }
    double idx = p * static_cast<double>(values.size() - 1);
    size_t lo = static_cast<size_t>(idx);
    size_t hi = std::min(lo + 1, values.size() - 1);
    double frac = idx - static_cast<double>(lo);
    return values[lo] * (1.0 - frac) + values[hi] * frac;
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

struct Benchmark {
    std::string name;
    BenchFn fn;
    bool enabled;
};

static void run_benchmark(const Benchmark& b, int iters, int samples,
                          double empty_tsc_median, double empty_thread_median,
                          double empty_pmc_median) {
    const double ops = static_cast<double>(iters) * OPS_PER_ITER;
    for (int i = 0; i < 8; ++i) {
        measure_once(b.fn, iters);
    }

    std::vector<double> tsc_raw;
    std::vector<double> tsc_adj;
    std::vector<double> thread_raw;
    std::vector<double> thread_adj;
    std::vector<double> pmc_adj;
    tsc_raw.reserve(samples);
    tsc_adj.reserve(samples);
    thread_raw.reserve(samples);
    thread_adj.reserve(samples);
    pmc_adj.reserve(samples);

    for (int i = 0; i < samples; ++i) {
        Timing t = measure_once(b.fn, iters);
        tsc_raw.push_back(static_cast<double>(t.tsc) / ops);
        tsc_adj.push_back((static_cast<double>(t.tsc) - empty_tsc_median) / ops);
        thread_raw.push_back(static_cast<double>(t.thread_cycles) / ops);
        thread_adj.push_back((static_cast<double>(t.thread_cycles) - empty_thread_median) / ops);
        if (rdpmc_fixed_core_available) {
            pmc_adj.push_back((static_cast<double>(t.pmc_core_cycles) - empty_pmc_median) / ops);
        }
    }

    double primary = rdpmc_fixed_core_available ? median(pmc_adj) : median(thread_adj);
    double primary_min = rdpmc_fixed_core_available
                             ? *std::min_element(pmc_adj.begin(), pmc_adj.end())
                             : *std::min_element(thread_adj.begin(), thread_adj.end());
    double primary_p90 = rdpmc_fixed_core_available
                             ? percentile(pmc_adj, 0.90)
                             : percentile(thread_adj, 0.90);

    std::cout << std::left << std::setw(20) << b.name
              << std::right << std::fixed << std::setprecision(3)
              << std::setw(12) << primary
              << std::setw(12) << primary_min
              << std::setw(12) << primary_p90
              << std::setw(12) << median(thread_adj)
              << std::setw(12) << median(tsc_adj)
              << "\n";
}

int main(int argc, char** argv) {
    unsigned logical_cpu = 0;
    int iters = 2048;
    int samples = 101;
    if (argc > 1) {
        logical_cpu = static_cast<unsigned>(std::strtoul(argv[1], nullptr, 10));
    }
    if (argc > 2) {
        iters = std::max(1, std::atoi(argv[2]));
    }
    if (argc > 3) {
        samples = std::max(3, std::atoi(argv[3]));
    }

    raise_priority();
    bool pinned = pin_to_logical_cpu(logical_cpu);
    Sleep(10);
    rdpmc_fixed_core_available = detect_rdpmc_fixed_core();

    CpuFeatures features = get_cpu_features();
    init_pointer_ring(4096);

    std::vector<Benchmark> benchmarks = {
        {"add r64, imm", bench_add_r64_imm, true},
        {"add r64, reg", bench_add_r64_reg, true},
        {"imul r64, imm", bench_imul_r64_imm, true},
        {"shl r64, 1", bench_shl_r64_1, true},
        {"ror r64, 7", bench_ror_r64_7, true},
        {"popcnt r64", bench_popcnt_r64, features.popcnt},
        {"lzcnt r64", bench_lzcnt_r64, features.lzcnt},
        {"crc32 r64,r64", bench_crc32_r64, features.sse42},
        {"addsd xmm", bench_addsd_xmm, true},
        {"mulsd xmm", bench_mulsd_xmm, true},
        {"divsd xmm", bench_divsd_xmm, true},
        {"sqrtsd xmm", bench_sqrtsd_xmm, true},
        {"L1 ptr chase", bench_l1_pointer_chase, true},
    };

    for (int i = 0; i < 16; ++i) {
        measure_once(bench_empty, iters);
    }

    std::vector<double> empty_tsc;
    std::vector<double> empty_thread;
    std::vector<double> empty_pmc;
    empty_tsc.reserve(samples);
    empty_thread.reserve(samples);
    empty_pmc.reserve(samples);
    for (int i = 0; i < samples; ++i) {
        Timing t = measure_once(bench_empty, iters);
        empty_tsc.push_back(static_cast<double>(t.tsc));
        empty_thread.push_back(static_cast<double>(t.thread_cycles));
        if (rdpmc_fixed_core_available) {
            empty_pmc.push_back(static_cast<double>(t.pmc_core_cycles));
        }
    }
    double empty_tsc_median = median(empty_tsc);
    double empty_thread_median = median(empty_thread);
    double empty_pmc_median = rdpmc_fixed_core_available ? median(empty_pmc) : 0.0;

    std::cout << "uops-style dependent-chain latency probe\n";
    std::cout << "logical_cpu_requested=" << logical_cpu
              << " pinned=" << (pinned ? "yes" : "no")
              << " current_cpu=" << GetCurrentProcessorNumber()
              << " ops_per_sample=" << (static_cast<uint64_t>(iters) * OPS_PER_ITER)
              << " samples=" << samples << "\n";
    std::cout << "features: invariant_tsc=" << (features.invariant_tsc ? "yes" : "no")
              << " popcnt=" << (features.popcnt ? "yes" : "no")
              << " lzcnt=" << (features.lzcnt ? "yes" : "no")
              << " sse4.2=" << (features.sse42 ? "yes" : "no") << "\n";
    std::cout << "timing: rdpmc_fixed_core_cycles="
              << (rdpmc_fixed_core_available ? "yes" : "no")
              << " primary_column="
              << (rdpmc_fixed_core_available ? "rdpmc_core_cycles" : "QueryThreadCycleTime")
              << "\n";
    std::cout << "empty_loop_median: thread_cycles=" << static_cast<uint64_t>(empty_thread_median)
              << " tsc=" << static_cast<uint64_t>(empty_tsc_median)
              << " pmc_core=" << static_cast<uint64_t>(empty_pmc_median)
              << " per_nominal_op_thread=" << std::fixed << std::setprecision(5)
              << empty_thread_median / (static_cast<double>(iters) * OPS_PER_ITER)
              << "\n\n";

    std::cout << std::left << std::setw(20) << "benchmark"
              << std::right
              << std::setw(12) << "med prim"
              << std::setw(12) << "min prim"
              << std::setw(12) << "p90 prim"
              << std::setw(12) << "med thread"
              << std::setw(12) << "med tsc"
              << "\n";
    std::cout << std::string(80, '-') << "\n";

    for (const Benchmark& b : benchmarks) {
        if (b.enabled) {
            run_benchmark(b, iters, samples, empty_tsc_median, empty_thread_median,
                          empty_pmc_median);
        } else {
            std::cout << std::left << std::setw(20) << b.name
                      << " skipped: CPU feature unavailable\n";
        }
    }

    return 0;
}
