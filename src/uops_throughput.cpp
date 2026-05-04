// Reciprocal-throughput companion to uops_latency_linux.cpp.
//
// Same overall structure: pin to a logical CPU, subtract an empty-loop
// baseline, report median/min/p90 over many short samples. The difference
// is that each unrolled block uses N independent destination registers
// so adjacent ops have no data dependency. Median ns/op then reflects
// the issue/throughput limit instead of the dependent-chain latency.
//
// Same 13-instruction set as the latency harness; the L1 pointer chase is
// replaced with an "8-stream independent L1 load" test (pointer chase is
// intrinsically serial).
//
// Per-bench register width (chosen to expose enough parallelism on Zen 3):
//   ALU 64-bit (add r,imm/add r,reg/imul/shl/ror/popcnt/lzcnt/crc32): 4 regs.
//     Zen 3 has 4 ALUs and most of these dispatch on >=2 ports, so 4-way
//     suffices to expose 1/cycle (or close to it) reciprocal throughput.
//   addsd/mulsd: 4 xmm regs (Zen 3 FADD/FMUL latency 3, 2 pipes -> 0.5 c/op).
//   divsd/sqrtsd: 8 xmm regs. These are not pipelined (latency ~13/20),
//     but using 8 independent dests lets the front-end keep them in flight.
//   independent L1 loads: 8 chains stepping through distinct cache lines.

#define _GNU_SOURCE

#include <algorithm>
#include <cerrno>
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

// REPN macros: same idiom as the latency harness. We aim for 64 emitted
// instructions per outer iteration so the empty-loop subtraction normalizes
// in the same way.
#define REP2(x) x x
#define REP4(x) REP2(x) REP2(x)
#define REP8(x) REP4(x) REP4(x)
#define REP16(x) REP8(x) REP8(x)
#define REP32(x) REP16(x) REP16(x)
#define REP64(x) REP32(x) REP32(x)

// 64 instructions per iteration is the same denominator the latency harness
// uses, which makes the empty-loop baseline directly comparable.
static constexpr int OPS_PER_ITER = 64;

static volatile uint64_t sink_u64 = 0x9e3779b97f4a7c15ull;
static volatile double sink_f64 = 1.0000001;
// Independent-load throughput: array of pointers to distinct cache lines.
// Each "stream" reads one element; streams are independent.
struct alignas(64) Cacheline { uintptr_t p; uint8_t pad[56]; };
static std::vector<Cacheline> indep_lines;

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

static uint64_t read_thread_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts);
    return static_cast<uint64_t>(ts.tv_sec) * 1000000000ull +
           static_cast<uint64_t>(ts.tv_nsec);
}

static bool pin_to_logical_cpu(unsigned cpu) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    int rc = pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
    return rc == 0;
}

__attribute__((noinline)) static uint64_t bench_empty(int iters) {
    uint64_t x = sink_u64;
    for (int i = 0; i < iters; ++i) {
        asm volatile("" : [x] "+r"(x) : : "cc");
    }
    sink_u64 = x;
    return x;
}

// --------------------------- 4-way ALU benches ---------------------------
// Each block emits 4 independent ops (one per dest); REP16 -> 64 ops/iter.

__attribute__((noinline)) static uint64_t bench_add_r64_imm(int iters) {
    uint64_t a = sink_u64 | 1ull, b = a ^ 0x11ull, c = a ^ 0x22ull, d = a ^ 0x33ull;
    for (int i = 0; i < iters; ++i) {
        asm volatile(REP16(
            "addq $1, %[a]\n\t"
            "addq $1, %[b]\n\t"
            "addq $1, %[c]\n\t"
            "addq $1, %[d]\n\t")
            : [a]"+r"(a), [b]"+r"(b), [c]"+r"(c), [d]"+r"(d)
            :
            : "cc");
    }
    sink_u64 = a ^ b ^ c ^ d;
    return sink_u64;
}

__attribute__((noinline)) static uint64_t bench_add_r64_reg(int iters) {
    uint64_t a = sink_u64 | 1ull, b = a ^ 0x11ull, c = a ^ 0x22ull, d = a ^ 0x33ull;
    const uint64_t y = 0x9e3779b97f4a7c15ull;
    for (int i = 0; i < iters; ++i) {
        asm volatile(REP16(
            "addq %[y], %[a]\n\t"
            "addq %[y], %[b]\n\t"
            "addq %[y], %[c]\n\t"
            "addq %[y], %[d]\n\t")
            : [a]"+r"(a), [b]"+r"(b), [c]"+r"(c), [d]"+r"(d)
            : [y]"r"(y)
            : "cc");
    }
    sink_u64 = a ^ b ^ c ^ d;
    return sink_u64;
}

__attribute__((noinline)) static uint64_t bench_imul_r64_imm(int iters) {
    uint64_t a = sink_u64 | 1ull, b = a ^ 0x11ull, c = a ^ 0x22ull, d = a ^ 0x33ull;
    for (int i = 0; i < iters; ++i) {
        asm volatile(REP16(
            "imulq $3, %[a], %[a]\n\t"
            "imulq $3, %[b], %[b]\n\t"
            "imulq $3, %[c], %[c]\n\t"
            "imulq $3, %[d], %[d]\n\t")
            : [a]"+r"(a), [b]"+r"(b), [c]"+r"(c), [d]"+r"(d)
            :
            : "cc");
    }
    sink_u64 = a ^ b ^ c ^ d;
    return sink_u64;
}

__attribute__((noinline)) static uint64_t bench_shl_r64_1(int iters) {
    uint64_t a = sink_u64 | 1ull, b = a ^ 0x11ull, c = a ^ 0x22ull, d = a ^ 0x33ull;
    for (int i = 0; i < iters; ++i) {
        asm volatile(REP16(
            "shlq $1, %[a]\n\t"
            "shlq $1, %[b]\n\t"
            "shlq $1, %[c]\n\t"
            "shlq $1, %[d]\n\t")
            : [a]"+r"(a), [b]"+r"(b), [c]"+r"(c), [d]"+r"(d)
            :
            : "cc");
    }
    sink_u64 = a ^ b ^ c ^ d;
    return sink_u64;
}

__attribute__((noinline)) static uint64_t bench_ror_r64_7(int iters) {
    uint64_t a = sink_u64 | 1ull, b = a ^ 0x11ull, c = a ^ 0x22ull, d = a ^ 0x33ull;
    for (int i = 0; i < iters; ++i) {
        asm volatile(REP16(
            "rorq $7, %[a]\n\t"
            "rorq $7, %[b]\n\t"
            "rorq $7, %[c]\n\t"
            "rorq $7, %[d]\n\t")
            : [a]"+r"(a), [b]"+r"(b), [c]"+r"(c), [d]"+r"(d)
            :
            : "cc");
    }
    sink_u64 = a ^ b ^ c ^ d;
    return sink_u64;
}

// popcnt/lzcnt: the dest-only form (popcnt rax, rax) creates a self
// dependency. Use 4 separate dests reading from a single source register
// to break the chain across the 4-way unroll.
__attribute__((noinline)) static uint64_t bench_popcnt_r64(int iters) {
    uint64_t s = sink_u64 | 0xf0f0f0f0f0f0f0f1ull;
    uint64_t a = 0, b = 0, c = 0, d = 0;
    for (int i = 0; i < iters; ++i) {
        asm volatile(REP16(
            "popcntq %[s], %[a]\n\t"
            "popcntq %[s], %[b]\n\t"
            "popcntq %[s], %[c]\n\t"
            "popcntq %[s], %[d]\n\t")
            : [a]"=&r"(a), [b]"=&r"(b), [c]"=&r"(c), [d]"=&r"(d)
            : [s]"r"(s)
            : "cc");
    }
    sink_u64 = a ^ b ^ c ^ d;
    return sink_u64;
}

__attribute__((noinline)) static uint64_t bench_lzcnt_r64(int iters) {
    uint64_t s = sink_u64 | 0x0102030405060708ull;
    uint64_t a = 0, b = 0, c = 0, d = 0;
    for (int i = 0; i < iters; ++i) {
        asm volatile(REP16(
            "lzcntq %[s], %[a]\n\t"
            "lzcntq %[s], %[b]\n\t"
            "lzcntq %[s], %[c]\n\t"
            "lzcntq %[s], %[d]\n\t")
            : [a]"=&r"(a), [b]"=&r"(b), [c]"=&r"(c), [d]"=&r"(d)
            : [s]"r"(s)
            : "cc");
    }
    sink_u64 = a ^ b ^ c ^ d;
    return sink_u64;
}

// crc32 r64,r64 has a true dependency on dest: dst = crc32(dst, src).
// To measure throughput we need 4 independent crc32 streams; each stream
// must keep its own running CRC across the inner block, so we maintain
// 4 dest accumulators and just feed them all from the same src.
__attribute__((noinline)) static uint64_t bench_crc32_r64(int iters) {
    uint64_t s = sink_u64 | 0x123456789abcdef1ull;
    uint64_t a = s, b = s ^ 0x11ull, c = s ^ 0x22ull, d = s ^ 0x33ull;
    for (int i = 0; i < iters; ++i) {
        // NOTE: each accumulator is still self-dependent across iterations
        // (a = crc32(a, s) chains through a). Within one REP16 block all
        // four streams advance once each, so 4 independent crc32 ops are
        // in flight at any time, which is enough to expose Zen 3's two
        // crc32 pipes.
        asm volatile(REP16(
            "crc32q %[s], %[a]\n\t"
            "crc32q %[s], %[b]\n\t"
            "crc32q %[s], %[c]\n\t"
            "crc32q %[s], %[d]\n\t")
            : [a]"+r"(a), [b]"+r"(b), [c]"+r"(c), [d]"+r"(d)
            : [s]"r"(s)
            : "cc");
    }
    sink_u64 = a ^ b ^ c ^ d;
    return sink_u64;
}

// --------------------------- FP benches ---------------------------
// addsd/mulsd: 4 independent xmm dests (latency 3, two FP pipes -> recip 0.5).
__attribute__((noinline)) static uint64_t bench_addsd_xmm(int iters) {
    double a = sink_f64, b = sink_f64 + 1.0, c = sink_f64 + 2.0, d = sink_f64 + 3.0;
    const double k = 1.0;
    for (int i = 0; i < iters; ++i) {
        asm volatile(REP16(
            "addsd %[k], %[a]\n\t"
            "addsd %[k], %[b]\n\t"
            "addsd %[k], %[c]\n\t"
            "addsd %[k], %[d]\n\t")
            : [a]"+x"(a), [b]"+x"(b), [c]"+x"(c), [d]"+x"(d)
            : [k]"x"(k));
    }
    sink_f64 = a + b + c + d;
    return static_cast<uint64_t>(sink_f64);
}

__attribute__((noinline)) static uint64_t bench_mulsd_xmm(int iters) {
    double a = sink_f64, b = sink_f64 + 1.0, c = sink_f64 + 2.0, d = sink_f64 + 3.0;
    const double k = 1.0000000001;
    for (int i = 0; i < iters; ++i) {
        asm volatile(REP16(
            "mulsd %[k], %[a]\n\t"
            "mulsd %[k], %[b]\n\t"
            "mulsd %[k], %[c]\n\t"
            "mulsd %[k], %[d]\n\t")
            : [a]"+x"(a), [b]"+x"(b), [c]"+x"(c), [d]"+x"(d)
            : [k]"x"(k));
    }
    sink_f64 = a + b + c + d;
    return static_cast<uint64_t>(sink_f64);
}

// divsd/sqrtsd: 8-way to give the unpipelined divider/sqrt unit room to
// queue work. 8 ops per inner block * REP8 = 64 ops/iter.
__attribute__((noinline)) static uint64_t bench_divsd_xmm(int iters) {
    double a = sink_f64 + 1000.0, b = a + 1.0, c = a + 2.0, d = a + 3.0;
    double e = a + 4.0,            f = a + 5.0, g = a + 6.0, h = a + 7.0;
    const double k = 1.0000000001;
    for (int i = 0; i < iters; ++i) {
        asm volatile(REP8(
            "divsd %[k], %[a]\n\t"
            "divsd %[k], %[b]\n\t"
            "divsd %[k], %[c]\n\t"
            "divsd %[k], %[d]\n\t"
            "divsd %[k], %[e]\n\t"
            "divsd %[k], %[f]\n\t"
            "divsd %[k], %[g]\n\t"
            "divsd %[k], %[h]\n\t")
            : [a]"+x"(a), [b]"+x"(b), [c]"+x"(c), [d]"+x"(d),
              [e]"+x"(e), [f]"+x"(f), [g]"+x"(g), [h]"+x"(h)
            : [k]"x"(k));
    }
    sink_f64 = a + b + c + d + e + f + g + h;
    return static_cast<uint64_t>(sink_f64);
}

// sqrtsd: each dest reads from a fresh source so the streams are
// fully independent (sqrtsd dst, src). 8 dests, 8-op block, REP8.
__attribute__((noinline)) static uint64_t bench_sqrtsd_xmm(int iters) {
    double s0 = sink_f64 + 100.0, s1 = sink_f64 + 200.0;
    double s2 = sink_f64 + 300.0, s3 = sink_f64 + 400.0;
    double s4 = sink_f64 + 500.0, s5 = sink_f64 + 600.0;
    double s6 = sink_f64 + 700.0, s7 = sink_f64 + 800.0;
    double a, b, c, d, e, f, g, h;
    a = b = c = d = e = f = g = h = 0.0;
    for (int i = 0; i < iters; ++i) {
        asm volatile(REP8(
            "sqrtsd %[s0], %[a]\n\t"
            "sqrtsd %[s1], %[b]\n\t"
            "sqrtsd %[s2], %[c]\n\t"
            "sqrtsd %[s3], %[d]\n\t"
            "sqrtsd %[s4], %[e]\n\t"
            "sqrtsd %[s5], %[f]\n\t"
            "sqrtsd %[s6], %[g]\n\t"
            "sqrtsd %[s7], %[h]\n\t")
            : [a]"=&x"(a), [b]"=&x"(b), [c]"=&x"(c), [d]"=&x"(d),
              [e]"=&x"(e), [f]"=&x"(f), [g]"=&x"(g), [h]"=&x"(h)
            : [s0]"x"(s0), [s1]"x"(s1), [s2]"x"(s2), [s3]"x"(s3),
              [s4]"x"(s4), [s5]"x"(s5), [s6]"x"(s6), [s7]"x"(s7));
    }
    sink_f64 = a + b + c + d + e + f + g + h;
    return static_cast<uint64_t>(sink_f64);
}

// --------------------------- Independent L1 loads ---------------------------
// Pointer chase is dependent by definition. The throughput counterpart is
// 8 parallel L1 loads to distinct cache lines; the loads can issue in
// parallel (Zen 3 has 3 AGUs, 2 load ports; ~2 loads/cycle).
__attribute__((noinline)) static uint64_t bench_l1_indep_loads(int iters) {
    Cacheline* base = indep_lines.data();
    uintptr_t a = 0, b = 0, c = 0, d = 0, e = 0, f = 0, g = 0, h = 0;
    for (int i = 0; i < iters; ++i) {
        // 8 indep loads * REP8 = 64 loads/iter. Each load reads a different
        // cache line from indep_lines[]. We use displacement addressing so
        // the dest reg has no dependency on prior loads.
        asm volatile(REP8(
            "movq   0(%[base]), %[a]\n\t"
            "movq  64(%[base]), %[b]\n\t"
            "movq 128(%[base]), %[c]\n\t"
            "movq 192(%[base]), %[d]\n\t"
            "movq 256(%[base]), %[e]\n\t"
            "movq 320(%[base]), %[f]\n\t"
            "movq 384(%[base]), %[g]\n\t"
            "movq 448(%[base]), %[h]\n\t")
            : [a]"=&r"(a), [b]"=&r"(b), [c]"=&r"(c), [d]"=&r"(d),
              [e]"=&r"(e), [f]"=&r"(f), [g]"=&r"(g), [h]"=&r"(h)
            : [base]"r"(base)
            : "memory");
    }
    sink_u64 = a ^ b ^ c ^ d ^ e ^ f ^ g ^ h;
    return sink_u64;
}

struct Timing {
    uint64_t tsc = 0;
    uint64_t thread_ns = 0;
};

static Timing measure_once(BenchFn fn, int iters) {
    uint64_t tn0 = read_thread_ns();
    uint64_t t0 = read_tsc_begin();
    fn(iters);
    uint64_t t1 = read_tsc_end();
    uint64_t tn1 = read_thread_ns();
    return Timing{t1 - t0, tn1 - tn0};
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

static void init_indep_lines(size_t count) {
    // We only need >=8 cache lines for the throughput probe; allocate a
    // few more so the compiler/loader can't fold accesses, and so the
    // working set comfortably fits in L1.
    indep_lines.assign(count, Cacheline{});
    for (size_t i = 0; i < count; ++i) {
        indep_lines[i].p = static_cast<uintptr_t>(i + 1);
    }
}

struct Benchmark {
    std::string name;
    BenchFn fn;
    bool enabled;
};

static void run_benchmark(const Benchmark& b, int iters, int samples,
                          double empty_tsc_median, double empty_thread_median) {
    const double ops = static_cast<double>(iters) * OPS_PER_ITER;
    for (int i = 0; i < 8; ++i) {
        measure_once(b.fn, iters);
    }

    std::vector<double> tsc_adj;
    std::vector<double> thread_adj;
    tsc_adj.reserve(samples);
    thread_adj.reserve(samples);

    for (int i = 0; i < samples; ++i) {
        Timing t = measure_once(b.fn, iters);
        tsc_adj.push_back((static_cast<double>(t.tsc) - empty_tsc_median) / ops);
        thread_adj.push_back((static_cast<double>(t.thread_ns) - empty_thread_median) / ops);
    }

    double primary = median(thread_adj);
    double primary_min = *std::min_element(thread_adj.begin(), thread_adj.end());
    double primary_p90 = percentile(thread_adj, 0.90);

    std::cout << std::left << std::setw(20) << b.name
              << std::right << std::fixed << std::setprecision(3)
              << std::setw(12) << primary
              << std::setw(12) << primary_min
              << std::setw(12) << primary_p90
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

    bool pinned = pin_to_logical_cpu(logical_cpu);
    usleep(10000);

    CpuFeatures features = get_cpu_features();
    init_indep_lines(64);

    std::vector<Benchmark> benchmarks = {
        {"add r64, imm",   bench_add_r64_imm,   true},
        {"add r64, reg",   bench_add_r64_reg,   true},
        {"imul r64, imm",  bench_imul_r64_imm,  true},
        {"shl r64, 1",     bench_shl_r64_1,     true},
        {"ror r64, 7",     bench_ror_r64_7,     true},
        {"popcnt r64",     bench_popcnt_r64,    features.popcnt},
        {"lzcnt r64",      bench_lzcnt_r64,     features.lzcnt},
        {"crc32 r64,r64",  bench_crc32_r64,     features.sse42},
        {"addsd xmm",      bench_addsd_xmm,     true},
        {"mulsd xmm",      bench_mulsd_xmm,     true},
        {"divsd xmm",      bench_divsd_xmm,     true},
        {"sqrtsd xmm",     bench_sqrtsd_xmm,    true},
        {"L1 indep loads", bench_l1_indep_loads, true},
    };

    for (int i = 0; i < 16; ++i) {
        measure_once(bench_empty, iters);
    }

    std::vector<double> empty_tsc;
    std::vector<double> empty_thread;
    empty_tsc.reserve(samples);
    empty_thread.reserve(samples);
    for (int i = 0; i < samples; ++i) {
        Timing t = measure_once(bench_empty, iters);
        empty_tsc.push_back(static_cast<double>(t.tsc));
        empty_thread.push_back(static_cast<double>(t.thread_ns));
    }
    double empty_tsc_median = median(empty_tsc);
    double empty_thread_median = median(empty_thread);

    std::cout << "uops-style reciprocal-throughput probe (linux)\n";
    std::cout << "logical_cpu_requested=" << logical_cpu
              << " pinned=" << (pinned ? "yes" : "no")
              << " current_cpu=" << sched_getcpu()
              << " ops_per_sample=" << (static_cast<uint64_t>(iters) * OPS_PER_ITER)
              << " samples=" << samples << "\n";
    std::cout << "features: invariant_tsc=" << (features.invariant_tsc ? "yes" : "no")
              << " popcnt=" << (features.popcnt ? "yes" : "no")
              << " lzcnt=" << (features.lzcnt ? "yes" : "no")
              << " sse4.2=" << (features.sse42 ? "yes" : "no") << "\n";
    std::cout << "timing: primary_column=clock_gettime_thread_ns"
              << " (rdpmc unavailable without perf_event_open)\n";
    std::cout << "parallelism: ALU/FPadd/FPmul=4-way, divsd/sqrtsd=8-way, L1 loads=8-stream\n";
    std::cout << "empty_loop_median: thread_ns=" << static_cast<uint64_t>(empty_thread_median)
              << " tsc=" << static_cast<uint64_t>(empty_tsc_median)
              << " per_nominal_op_thread_ns=" << std::fixed << std::setprecision(5)
              << empty_thread_median / (static_cast<double>(iters) * OPS_PER_ITER)
              << "\n\n";

    std::cout << std::left << std::setw(20) << "benchmark"
              << std::right
              << std::setw(12) << "med ns/op"
              << std::setw(12) << "min ns/op"
              << std::setw(12) << "p90 ns/op"
              << std::setw(12) << "med tsc/op"
              << "\n";
    std::cout << std::string(68, '-') << "\n";

    for (const Benchmark& b : benchmarks) {
        if (b.enabled) {
            run_benchmark(b, iters, samples, empty_tsc_median, empty_thread_median);
        } else {
            std::cout << std::left << std::setw(20) << b.name
                      << " skipped: CPU feature unavailable\n";
        }
    }

    return 0;
}
