// Operand-to-operand latency probe.
//
// For instructions whose individual input->output paths can have different
// latencies, build dependent chains that thread through *only* one path so
// the median ns/op reflects that single path. Same harness machinery as
// uops_latency_linux.cpp (CPU pin, empty-loop subtraction, median/min/p90).
//
// Path-isolation tricks used here:
//   * "src->dst" for instructions with separate src/dst (adc/sbb/cmov):
//     alternate `op dst, src` and `op src, dst` so the value flowing through
//     the chain is read from the source operand each step.
//   * "dst->dst" for the same instructions: keep the source operand a
//     loop-invariant register, so only the dst<-dst feedback closes the loop.
//   * "CF->dst" (adc/sbb): use an immediate or constant source so the data
//     ALU isn't stressed; the only fast-changing input across iterations is
//     CF itself. We alternate ops that *read* CF and *write* CF without
//     touching the destination value's high-order bits significantly.
//   * "flags->dst" (cmov): build a chain test r3,r3 -> cmovne dst, r2 ->
//     ... by feeding the cmov result back into the test source via a mov.
//   * "simple vs complex addressing" (mov): pointer chase using either
//     `(%[p])` or `(%[base], %[p], 8)` indexed addressing; the AGU latency
//     diff (if any) shows up as ns/op delta.
//   * "add r,[mem] fused vs split": fused = single insn closing the chain
//     through r1; split = `add r1,r2; add r1,[mem]` adds an arithmetic op
//     in front of the load-add to expose the load-use latency separately.
//
// NOTE: these isolations are approximate. CF and flag-output ports also
// chain implicitly in adc/sbb chains; we mark each row with the path the
// chain was *designed* to expose, not a guarantee of perfect isolation.

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

#define REP2(x) x x
#define REP4(x) REP2(x) REP2(x)
#define REP8(x) REP4(x) REP4(x)
#define REP16(x) REP8(x) REP8(x)
#define REP32(x) REP16(x) REP16(x)
#define REP64(x) REP32(x) REP32(x)
static constexpr int OPS_PER_ITER = 64;

static volatile uint64_t sink_u64 = 0x9e3779b97f4a7c15ull;
static std::vector<uintptr_t> pointer_ring;       // for simple-addressing chase
static std::vector<size_t>    index_ring;         // for complex-addressing chase
static std::vector<uintptr_t> add_mem_buf;        // single hot location for add r,[mem]

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

// =============================== ADC ===============================

// adc dst, src   :   dst = dst + src + CF;  CF' = carry_out
// dst->dst path: src is a loop-invariant register, CF is reset/unchanging
// in spirit. We can't truly remove the CF feedback, but with src=0 the
// data path is dominated by the dst->dst feedback (dst = dst + 0 + CF).
// The chain that closes is dst -> dst (plus an unavoidable CF -> dst).
__attribute__((noinline)) static uint64_t bench_adc_dst_dst(int iters) {
    uint64_t a = sink_u64 | 1ull;
    const uint64_t z = 0;
    for (int i = 0; i < iters; ++i) {
        asm volatile(REP64("adcq %[z], %[a]\n\t")
                     : [a]"+r"(a)
                     : [z]"r"(z)
                     : "cc");
    }
    sink_u64 = a;
    return a;
}

// src->dst path: alternate `adc a, b; adc b, a` so the value carried by
// `a` propagates through the *source* operand of the second insn (b reads
// a as src) and back. With 64 ops/iter we close 32 a->b->a round trips.
// What we measure: the per-op latency along the src->dst data path.
__attribute__((noinline)) static uint64_t bench_adc_src_dst(int iters) {
    uint64_t a = sink_u64 | 1ull;
    uint64_t b = a ^ 0x5a5aull;
    for (int i = 0; i < iters; ++i) {
        asm volatile(REP32(
            "adcq %[b], %[a]\n\t"
            "adcq %[a], %[b]\n\t")
                     : [a]"+r"(a), [b]"+r"(b)
                     :
                     : "cc");
    }
    sink_u64 = a ^ b;
    return sink_u64;
}

// CF->dst path: `adc dst, $0` repeatedly. The data input from imm is 0,
// the data input from dst is the previous dst (small chain), and CF feeds
// in. Across many iterations the only thing that toggles is CF (dst grows
// only when CF=1). The chain CF_out -> CF_in is the dominant feedback.
__attribute__((noinline)) static uint64_t bench_adc_cf_dst(int iters) {
    uint64_t a = sink_u64 | 1ull;
    for (int i = 0; i < iters; ++i) {
        asm volatile(REP64("adcq $0, %[a]\n\t")
                     : [a]"+r"(a)
                     :
                     : "cc");
    }
    sink_u64 = a;
    return a;
}

// =============================== SBB ===============================
// Same shape as adc.
__attribute__((noinline)) static uint64_t bench_sbb_dst_dst(int iters) {
    uint64_t a = sink_u64 | 1ull;
    const uint64_t z = 0;
    for (int i = 0; i < iters; ++i) {
        asm volatile(REP64("sbbq %[z], %[a]\n\t")
                     : [a]"+r"(a)
                     : [z]"r"(z)
                     : "cc");
    }
    sink_u64 = a;
    return a;
}

__attribute__((noinline)) static uint64_t bench_sbb_src_dst(int iters) {
    uint64_t a = sink_u64 | 1ull;
    uint64_t b = a ^ 0x5a5aull;
    for (int i = 0; i < iters; ++i) {
        asm volatile(REP32(
            "sbbq %[b], %[a]\n\t"
            "sbbq %[a], %[b]\n\t")
                     : [a]"+r"(a), [b]"+r"(b)
                     :
                     : "cc");
    }
    sink_u64 = a ^ b;
    return sink_u64;
}

__attribute__((noinline)) static uint64_t bench_sbb_cf_dst(int iters) {
    uint64_t a = sink_u64 | 1ull;
    for (int i = 0; i < iters; ++i) {
        asm volatile(REP64("sbbq $0, %[a]\n\t")
                     : [a]"+r"(a)
                     :
                     : "cc");
    }
    sink_u64 = a;
    return a;
}

// =============================== CMOVNE ===============================
// cmovne dst, src : dst = (ZF==0) ? src : dst.   reads dst, src, ZF.

// dst->dst: src is a constant register, ZF is set once outside the chain
// and not modified inside (no flag-writing insns in the loop body).
// The cmov either keeps dst or replaces it with src. With ZF=1 (taken
// branch in cmovne semantics: ZF=0 means "not equal", so move occurs)
// we want the move to happen so dst is overwritten by src each step --
// but then there's no dst->dst chain. So we use ZF=1 (i.e. compare-eq:
// "not-not-equal") so the move does NOT happen and dst feeds itself.
// We seed ZF with `xor reg, reg` outside the inner unroll? That sets
// ZF=1 which means cmovne is NOT taken => dst keeps its value. To still
// observe a dst chain we instead want cmovne to fire so dst depends on
// src. Conclusion: the "dst->dst" measurement here is dominated by the
// path "previous dst -> next dst-or-src multiplexer -> dst", which on
// Zen 3 is the cmov latency (1c). We pick ZF such that the move *fires*
// every time (so dst <- src each time), and rely on src being set such
// that src is also a function of prior dst -- simplest: src and dst are
// the *same* register? Can't; cmov needs two regs.
//
// The cleanest "dst->dst" probe: have src be a constant, ZF such that the
// move never fires. Then dst feeds itself trivially through the cmov mux.
// This measures the dst-passthrough latency (essentially 1c on Zen 3).
__attribute__((noinline)) static uint64_t bench_cmovne_dst_dst(int iters) {
    uint64_t a = sink_u64 | 1ull;
    const uint64_t s = 0xdeadbeefull;
    // Set ZF=1 (so cmovne does NOT move). xor on a sacrificial reg sets ZF=1.
    uint64_t z = 0;
    asm volatile("xorl %k[z], %k[z]\n\t" : [z]"+r"(z) :: "cc");
    for (int i = 0; i < iters; ++i) {
        asm volatile(REP64(
            "cmovneq %[s], %[a]\n\t")
                     : [a]"+r"(a)
                     : [s]"r"(s)
                     : "cc");
        // ZF is preserved across cmovne (cmov doesn't write flags).
        // But we re-establish it each outer iter via the asm above being
        // outside the inner block; the C-level loop counter increment
        // (a register add) clobbers flags, so we re-xor.
        asm volatile("xorl %k[z], %k[z]\n\t" : [z]"+r"(z) :: "cc");
    }
    sink_u64 = a;
    return a;
}

// src->dst: alternate `cmovne a, b; cmovne b, a` with ZF=0 so the move
// fires every time. Then a <- b, b <- a alternately, and the value
// propagates through the source operand each step.
__attribute__((noinline)) static uint64_t bench_cmovne_src_dst(int iters) {
    uint64_t a = sink_u64 | 1ull;
    uint64_t b = a ^ 0x5a5aull;
    uint64_t z = 1; // nonzero so test sets ZF=0 (cmovne fires)
    asm volatile("testq %[z], %[z]\n\t" : : [z]"r"(z) : "cc");
    for (int i = 0; i < iters; ++i) {
        asm volatile(REP32(
            "cmovneq %[b], %[a]\n\t"
            "cmovneq %[a], %[b]\n\t")
                     : [a]"+r"(a), [b]"+r"(b)
                     :
                     : "cc");
        asm volatile("testq %[z], %[z]\n\t" : : [z]"r"(z) : "cc");
    }
    sink_u64 = a ^ b;
    return sink_u64;
}

// flags->dst: chain ZF through cmov by making the next test depend on the
// cmov output. Sequence: cmovne a, b; test a, a; (this updates ZF based
// on a, which depends on previous flags+dst). Repeat. Now the chain that
// must close is: prev ZF -> cmovne writes a -> test reads a writes ZF.
// That is, flags-out latency includes the test op too -- so we report
// (cmovne_flag_lat + test_lat) per pair. Document this as the path.
__attribute__((noinline)) static uint64_t bench_cmovne_flags_dst(int iters) {
    uint64_t a = sink_u64 | 1ull;
    const uint64_t s = 0xdeadbeefull;
    for (int i = 0; i < iters; ++i) {
        // 32 (cmovne;test) pairs = 64 ops/iter
        asm volatile(REP32(
            "testq %[a], %[a]\n\t"
            "cmovneq %[s], %[a]\n\t")
                     : [a]"+r"(a)
                     : [s]"r"(s)
                     : "cc");
    }
    sink_u64 = a;
    return a;
}

// =============================== mov r64, [mem] ===============================
// Pointer chase using simple addressing `(reg)` vs complex addressing
// `(base, idx, 8)` -- the latter routes through the AGU's index path which
// historically can add a cycle on some cores.

// Simple addressing: each load is `mov dst, (dst)`, dst is the next pointer.
// The classical L1 latency probe.
__attribute__((noinline)) static uint64_t bench_mov_simple_addr(int iters) {
    uintptr_t p = pointer_ring.empty() ? 0 : reinterpret_cast<uintptr_t>(&pointer_ring[0]);
    for (int i = 0; i < iters; ++i) {
        asm volatile(REP64("movq (%[p]), %[p]\n\t")
                     : [p]"+r"(p)
                     :
                     : "memory");
    }
    sink_u64 = p;
    return p;
}

// Complex addressing: chase through an *index* table. Each entry holds the
// next index; the load is `mov idx, (base, idx, 8)`. Same number of loads,
// same cache-resident working set, but the address calculation goes through
// the scaled-index path of the AGU.
__attribute__((noinline)) static uint64_t bench_mov_complex_addr(int iters) {
    if (index_ring.empty()) return 0;
    const size_t* base = index_ring.data();
    size_t idx = 0;
    for (int i = 0; i < iters; ++i) {
        asm volatile(REP64("movq (%[base], %[idx], 8), %[idx]\n\t")
                     : [idx]"+r"(idx)
                     : [base]"r"(base)
                     : "memory");
    }
    sink_u64 = static_cast<uint64_t>(idx);
    return sink_u64;
}

// =============================== add r, [mem] ===============================
// Variant A (fused): `add r1, [mem]` chained through r1. The chain is:
// load(mem) + add r1 -> r1.  Memory operand is a single hot L1 line so
// load-from-L1 dominates; the chain edge is r1 (dst).
__attribute__((noinline)) static uint64_t bench_add_mem_fused(int iters) {
    uintptr_t mem_addr = reinterpret_cast<uintptr_t>(add_mem_buf.data());
    uint64_t a = sink_u64 | 1ull;
    for (int i = 0; i < iters; ++i) {
        asm volatile(REP64("addq (%[m]), %[a]\n\t")
                     : [a]"+r"(a)
                     : [m]"r"(mem_addr)
                     : "cc", "memory");
    }
    sink_u64 = a;
    return a;
}

// Variant B (split): `add r1, r2; add r1, [mem]`  -- inserting a register
// add in front of each load-add. The r1 chain now has 2 ops per "step",
// so per-op latency should differ from the fused variant in a way that
// isolates the load-use vs register-add contributions.
__attribute__((noinline)) static uint64_t bench_add_mem_split(int iters) {
    uintptr_t mem_addr = reinterpret_cast<uintptr_t>(add_mem_buf.data());
    uint64_t a = sink_u64 | 1ull;
    const uint64_t y = 1;
    for (int i = 0; i < iters; ++i) {
        // 32 (add reg; add mem) pairs = 64 ops/iter.
        asm volatile(REP32(
            "addq %[y], %[a]\n\t"
            "addq (%[m]), %[a]\n\t")
                     : [a]"+r"(a)
                     : [y]"r"(y), [m]"r"(mem_addr)
                     : "cc", "memory");
    }
    sink_u64 = a;
    return a;
}

// =============================== Harness machinery ===============================
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
    if (n == 0) return 0.0;
    if ((n & 1u) != 0) return values[n / 2];
    return (values[n / 2 - 1] + values[n / 2]) / 2.0;
}

static double percentile(std::vector<double> values, double p) {
    std::sort(values.begin(), values.end());
    if (values.empty()) return 0.0;
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

// Build an index ring: index_ring[cur] = next, forming one Hamiltonian
// cycle through 0..count-1. Used by the complex-addressing chase.
static void init_index_ring(size_t count) {
    index_ring.assign(count, 0);
    std::vector<size_t> perm(count);
    std::iota(perm.begin(), perm.end(), 0);
    uint64_t state = 0x9e3779b97f4a7c15ull;
    for (size_t i = count - 1; i > 0; --i) {
        state = state * 2862933555777941757ull + 3037000493ull;
        size_t j = static_cast<size_t>(state % (i + 1));
        std::swap(perm[i], perm[j]);
    }
    for (size_t i = 0; i < count; ++i) {
        size_t cur = perm[i];
        size_t next = perm[(i + 1) % count];
        index_ring[cur] = next;
    }
}

static void init_add_mem_buf() {
    // Single hot 64-bit slot the add r,[mem] benches read.
    add_mem_buf.assign(8, 1ull);
}

struct Benchmark {
    std::string name;
    std::string path;
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

    std::cout << std::left << std::setw(22) << b.name
              << std::setw(20) << b.path
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
    init_pointer_ring(4096);
    init_index_ring(4096);
    init_add_mem_buf();

    std::vector<Benchmark> benchmarks = {
        {"adc r1, r2",      "dst->dst",     bench_adc_dst_dst,      true},
        {"adc r1, r2",      "src->dst",     bench_adc_src_dst,      true},
        {"adc r1, $0",      "CF->dst",      bench_adc_cf_dst,       true},
        {"sbb r1, r2",      "dst->dst",     bench_sbb_dst_dst,      true},
        {"sbb r1, r2",      "src->dst",     bench_sbb_src_dst,      true},
        {"sbb r1, $0",      "CF->dst",      bench_sbb_cf_dst,       true},
        {"cmovne r1, r2",   "dst->dst",     bench_cmovne_dst_dst,   true},
        {"cmovne r1, r2",   "src->dst",     bench_cmovne_src_dst,   true},
        {"cmovne+test",     "flags->dst",   bench_cmovne_flags_dst, true},
        {"mov r64,(r)",     "simple addr",  bench_mov_simple_addr,  true},
        {"mov r64,(b,i,8)", "indexed addr", bench_mov_complex_addr, true},
        {"add r,[mem]",     "fused",        bench_add_mem_fused,    true},
        {"add r,r;add r,[m]", "split via r1", bench_add_mem_split,  true},
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

    std::cout << "uops-style operand-to-operand latency probe (linux)\n";
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
    std::cout << "empty_loop_median: thread_ns=" << static_cast<uint64_t>(empty_thread_median)
              << " tsc=" << static_cast<uint64_t>(empty_tsc_median)
              << " per_nominal_op_thread_ns=" << std::fixed << std::setprecision(5)
              << empty_thread_median / (static_cast<double>(iters) * OPS_PER_ITER)
              << "\n\n";

    std::cout << std::left << std::setw(22) << "benchmark"
              << std::setw(20) << "path measured"
              << std::right
              << std::setw(12) << "med ns/op"
              << std::setw(12) << "min ns/op"
              << std::setw(12) << "p90 ns/op"
              << std::setw(12) << "med tsc/op"
              << "\n";
    std::cout << std::string(90, '-') << "\n";

    for (const Benchmark& b : benchmarks) {
        if (b.enabled) {
            run_benchmark(b, iters, samples, empty_tsc_median, empty_thread_median);
        } else {
            std::cout << std::left << std::setw(22) << b.name
                      << " skipped: CPU feature unavailable\n";
        }
    }

    return 0;
}
