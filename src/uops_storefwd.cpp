// Store-to-load forwarding and 4K aliasing microbenchmarks.
//
// Each sub-experiment is a dependent-chain loop where one iteration's load
// result feeds the next iteration's store address-or-data. The dependent
// chain exposes the round-trip latency through the store/load pipeline:
//
//   A. Aligned 8B store, 8B load at same address.        (forwarding ok)
//   B. 8B store at +0, 8B load at +4 (partial overlap).  (forwarding fails)
//   C. 4B store at +0, 8B load at +0 (load wider).       (forwarding fails)
//   D. 8B store at +0, 4B load at +0 (load narrower).    (forwarding ok)
//   E. 8B store at +0, 8B load at +4096 (4K alias).      (Intel-only stall)
//      (control: 8B store at +0, 8B load at +64.         no aliasing)
//
// We need a real chain so the load result genuinely depends on the prior
// store, otherwise the OoO core hides the round-trip. Pattern:
//
//     mov [rdi], rax       ; store rax
//     mov rax, [rdi+disp]  ; load -> rax
//     and rax, mask        ; force rax to be a small offset
//     ...                  ; (use rax in next store)
//
// In our setup we keep the addresses fixed (rdi + constant offset), and we
// store rax itself. The load returns the value rax just stored (modulo
// forwarding rules), so the next store's *data* depends on the previous
// load. That is enough to serialize the chain because a store can't issue
// data until the producing load completes.
//
// We mask rax with a small constant after each iter to keep the value
// well-defined and prevent pathological corner cases. The mask costs ~1
// cycle but is constant across all sub-experiments so it cancels in
// comparisons.
//
// Buffer is mmap'd page-aligned and at least 8 KiB so [rdi+4096] is valid.

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
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#include <cpuid.h>
#include <x86intrin.h>

#define REP2(x) x x
#define REP4(x) REP2(x) REP2(x)
#define REP8(x) REP4(x) REP4(x)
#define REP16(x) REP8(x) REP8(x)
#define REP32(x) REP16(x) REP16(x)
// Each "op" here is one store+load round-trip (plus a tiny mask op). We use
// REP32 (32 round-trips per iter) to keep the inline-asm body a manageable
// size, since each round-trip is 3 instructions not 1.
static constexpr int OPS_PER_ITER = 32;

static volatile uint64_t sink_u64 = 0;

using BenchFn = uint64_t (*)(int, void*);

struct CpuFeatures {
    bool invariant_tsc = false;
};

static bool cpuid_leaf(uint32_t leaf, uint32_t subleaf,
                       uint32_t& eax, uint32_t& ebx,
                       uint32_t& ecx, uint32_t& edx) {
    uint32_t max_leaf = __get_cpuid_max(leaf & 0x80000000u, nullptr);
    if (max_leaf < leaf) return false;
    __cpuid_count(leaf, subleaf, eax, ebx, ecx, edx);
    return true;
}

static CpuFeatures get_cpu_features() {
    CpuFeatures f;
    uint32_t eax = 0, ebx = 0, ecx = 0, edx = 0;
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

// =====================================================================
// Empty baseline. Same loop structure but the body is a no-op chain.
// We do an "and rax, 0xff" to match the masking-cost overhead in the
// real benchmarks (so that the empty subtraction removes mostly the
// loop-overhead and the per-iter mask).
// =====================================================================
__attribute__((noinline)) static uint64_t bench_empty(int iters, void* /*buf*/) {
    uint64_t x = sink_u64 & 0xffull;
    for (int i = 0; i < iters; ++i) {
        asm volatile(REP32("andq $0xff, %[x]\n\t")
                     : [x] "+r"(x)
                     :
                     : "cc");
    }
    sink_u64 = x;
    return x;
}

// =====================================================================
// A. Aligned 8B store + 8B load at same address. Forwarding succeeds.
//    Pattern per round-trip:
//        movq rax, [rdi]        (store)
//        movq [rdi], rax        (load)   -- wait, store-then-load:
//        movq [rdi], rax
//        movq rax, [rdi]
//        andq rax, 0xff
//    The load's result rax becomes the next store's data, so iters chain.
// =====================================================================
__attribute__((noinline)) static uint64_t bench_fwd_aligned(int iters, void* buf) {
    uint64_t x = sink_u64 & 0xffull;
    uint8_t* p = static_cast<uint8_t*>(buf);
    for (int i = 0; i < iters; ++i) {
        asm volatile(REP32(
                         "movq %[x], (%[p])\n\t"
                         "movq (%[p]), %[x]\n\t"
                         "andq $0xff, %[x]\n\t")
                     : [x] "+r"(x)
                     : [p] "r"(p)
                     : "memory", "cc");
    }
    sink_u64 = x;
    return x;
}

// =====================================================================
// B. Misaligned partial overlap. Store 8B at +0, load 8B at +4. The load
//    needs bytes 4..11 but the store buffer entry only covers 0..7, so
//    forwarding cannot satisfy the load and it stalls until the store
//    drains to the L1.
// =====================================================================
__attribute__((noinline)) static uint64_t bench_fwd_partial_overlap(int iters, void* buf) {
    uint64_t x = sink_u64 & 0xffull;
    uint8_t* p = static_cast<uint8_t*>(buf);
    for (int i = 0; i < iters; ++i) {
        asm volatile(REP32(
                         "movq %[x], (%[p])\n\t"
                         "movq 4(%[p]), %[x]\n\t"
                         "andq $0xff, %[x]\n\t")
                     : [x] "+r"(x)
                     : [p] "r"(p)
                     : "memory", "cc");
    }
    sink_u64 = x;
    return x;
}

// =====================================================================
// C. Wider load than store. Store 4B at +0, load 8B at +0. Forwarding
//    fails because the load asks for more bytes than the store provides.
// =====================================================================
__attribute__((noinline)) static uint64_t bench_fwd_wider_load(int iters, void* buf) {
    uint64_t x = sink_u64 & 0xffull;
    uint8_t* p = static_cast<uint8_t*>(buf);
    for (int i = 0; i < iters; ++i) {
        asm volatile(REP32(
                         "movl %k[x], (%[p])\n\t"      // 4B store
                         "movq (%[p]), %[x]\n\t"        // 8B load
                         "andq $0xff, %[x]\n\t")
                     : [x] "+r"(x)
                     : [p] "r"(p)
                     : "memory", "cc");
    }
    sink_u64 = x;
    return x;
}

// =====================================================================
// D. Narrower load than store. Store 8B at +0, load 4B at +0. Forwarding
//    succeeds because the load is a strict subset of the store.
// =====================================================================
__attribute__((noinline)) static uint64_t bench_fwd_narrower_load(int iters, void* buf) {
    uint64_t x = sink_u64 & 0xffull;
    uint8_t* p = static_cast<uint8_t*>(buf);
    for (int i = 0; i < iters; ++i) {
        asm volatile(REP32(
                         "movq %[x], (%[p])\n\t"        // 8B store
                         "movl (%[p]), %k[x]\n\t"       // 4B load (zero-extends)
                         "andq $0xff, %[x]\n\t")
                     : [x] "+r"(x)
                     : [p] "r"(p)
                     : "memory", "cc");
    }
    sink_u64 = x;
    return x;
}

// =====================================================================
// E1. 4K alias. Store at [rdi], load at [rdi + 4096]. Same low 12 bits;
//    on Intel the store-buffer disambiguation uses only the low bits and
//    triggers a false dependency. Zen 3 should not see this stall.
// =====================================================================
__attribute__((noinline)) static uint64_t bench_alias_4k(int iters, void* buf) {
    uint64_t x = sink_u64 & 0xffull;
    uint8_t* p = static_cast<uint8_t*>(buf);
    for (int i = 0; i < iters; ++i) {
        asm volatile(REP32(
                         "movq %[x], (%[p])\n\t"
                         "movq 4096(%[p]), %[x]\n\t"
                         "andq $0xff, %[x]\n\t")
                     : [x] "+r"(x)
                     : [p] "r"(p)
                     : "memory", "cc");
    }
    sink_u64 = x;
    return x;
}

// =====================================================================
// E2. Control: store at [rdi], load at [rdi + 64]. Different cache line,
//    different low-12 bits, no aliasing. This isolates the cost of the
//    "store data is independent of load result" case (the load value is
//    *not* the data we just stored, so the chain is broken on data side).
//    Caveat: this changes the dependency structure; we report it as a
//    reference point, not a strict apples-to-apples with E1.
//    To keep the dependent chain alive we still and-mask the loaded value.
// =====================================================================
__attribute__((noinline)) static uint64_t bench_alias_64(int iters, void* buf) {
    uint64_t x = sink_u64 & 0xffull;
    uint8_t* p = static_cast<uint8_t*>(buf);
    for (int i = 0; i < iters; ++i) {
        asm volatile(REP32(
                         "movq %[x], (%[p])\n\t"
                         "movq 64(%[p]), %[x]\n\t"
                         "andq $0xff, %[x]\n\t")
                     : [x] "+r"(x)
                     : [p] "r"(p)
                     : "memory", "cc");
    }
    sink_u64 = x;
    return x;
}

// =====================================================================
// Driver.
// =====================================================================

struct Timing {
    uint64_t tsc = 0;
    uint64_t thread_ns = 0;
};

static Timing measure_once(BenchFn fn, int iters, void* buf) {
    uint64_t tn0 = read_thread_ns();
    uint64_t t0 = read_tsc_begin();
    fn(iters, buf);
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

struct Benchmark {
    std::string name;
    std::string section;   // "A","B","C","D","E1","E2"
    std::string desc;
    BenchFn fn;
};

static void run_benchmark(const Benchmark& b, int iters, int samples, void* buf,
                          double empty_tsc_median, double empty_thread_median,
                          double cycles_per_ns) {
    const double ops = static_cast<double>(iters) * OPS_PER_ITER;
    for (int i = 0; i < 8; ++i) {
        measure_once(b.fn, iters, buf);
    }

    std::vector<double> tsc_adj;
    std::vector<double> thread_adj;
    tsc_adj.reserve(samples);
    thread_adj.reserve(samples);

    for (int i = 0; i < samples; ++i) {
        Timing t = measure_once(b.fn, iters, buf);
        tsc_adj.push_back((static_cast<double>(t.tsc) - empty_tsc_median) / ops);
        thread_adj.push_back((static_cast<double>(t.thread_ns) - empty_thread_median) / ops);
    }

    double primary = median(thread_adj);
    double primary_min = *std::min_element(thread_adj.begin(), thread_adj.end());
    double primary_p90 = percentile(thread_adj, 0.90);
    double tsc = median(tsc_adj);
    // Convert ns to cycles using cycles-per-ns derived from tsc/thread ratio.
    double est_cycles = primary * cycles_per_ns;

    std::cout << std::left << std::setw(4) << b.section
              << std::setw(28) << b.desc
              << std::right << std::fixed << std::setprecision(3)
              << std::setw(12) << primary
              << std::setw(12) << primary_min
              << std::setw(12) << primary_p90
              << std::setw(12) << tsc
              << std::setw(12) << est_cycles
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

    // Allocate a 16 KiB buffer (page-aligned, well past 8 KiB) so
    // [rdi+4096] is valid and on a different page from [rdi].
    const size_t buf_bytes = 16 * 1024;
    void* buf = mmap(nullptr, buf_bytes, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (buf == MAP_FAILED) {
        std::cerr << "mmap failed: " << std::strerror(errno) << "\n";
        return 1;
    }
    std::memset(buf, 0, buf_bytes);

    // Warm and measure baseline.
    for (int i = 0; i < 16; ++i) {
        measure_once(bench_empty, iters, buf);
    }
    std::vector<double> empty_tsc;
    std::vector<double> empty_thread;
    empty_tsc.reserve(samples);
    empty_thread.reserve(samples);
    for (int i = 0; i < samples; ++i) {
        Timing t = measure_once(bench_empty, iters, buf);
        empty_tsc.push_back(static_cast<double>(t.tsc));
        empty_thread.push_back(static_cast<double>(t.thread_ns));
    }
    double empty_tsc_median = median(empty_tsc);
    double empty_thread_median = median(empty_thread);

    // Estimate cycles per ns from the empty-loop ratio so we can convert
    // the primary thread_ns column to cycles. Using tsc and thread_ns of
    // the same loop is a reasonable proxy when the TSC runs at base freq.
    double cycles_per_ns = 0.0;
    if (empty_thread_median > 0.0) {
        cycles_per_ns = empty_tsc_median / empty_thread_median;
    }

    std::cout << "uops-style store-forwarding + 4K-aliasing probe\n";
    std::cout << "logical_cpu_requested=" << logical_cpu
              << " pinned=" << (pinned ? "yes" : "no")
              << " current_cpu=" << sched_getcpu()
              << " ops_per_sample=" << (static_cast<uint64_t>(iters) * OPS_PER_ITER)
              << " (each op = 1 store + 1 load + 1 mask)"
              << " samples=" << samples << "\n";
    std::cout << "features: invariant_tsc=" << (features.invariant_tsc ? "yes" : "no") << "\n";
    std::cout << "buffer: mmap " << buf_bytes << " bytes at " << buf << "\n";
    std::cout << "empty_loop_median: thread_ns=" << static_cast<uint64_t>(empty_thread_median)
              << " tsc=" << static_cast<uint64_t>(empty_tsc_median)
              << " est_cycles_per_ns=" << std::fixed << std::setprecision(3)
              << cycles_per_ns << "\n\n";

    std::cout << std::left << std::setw(4) << "id"
              << std::setw(28) << "scenario"
              << std::right
              << std::setw(12) << "med ns/op"
              << std::setw(12) << "min ns/op"
              << std::setw(12) << "p90 ns/op"
              << std::setw(12) << "med tsc/op"
              << std::setw(12) << "est cyc/op"
              << "\n";
    std::cout << std::string(88, '-') << "\n";

    std::vector<Benchmark> benchmarks = {
        {"A", "A", "aligned 8B->8B same addr",  bench_fwd_aligned},
        {"B", "B", "8B@0 -> 8B@4 (partial)",    bench_fwd_partial_overlap},
        {"C", "C", "4B@0 -> 8B@0 (wider load)", bench_fwd_wider_load},
        {"D", "D", "8B@0 -> 4B@0 (narrower)",   bench_fwd_narrower_load},
        {"E1","E1","8B@0 -> 8B@4096 (4K alias)",bench_alias_4k},
        {"E2","E2","8B@0 -> 8B@64 (control)",   bench_alias_64},
    };

    for (const Benchmark& b : benchmarks) {
        run_benchmark(b, iters, samples, buf,
                      empty_tsc_median, empty_thread_median, cycles_per_ns);
    }

    munmap(buf, buf_bytes);
    return 0;
}
