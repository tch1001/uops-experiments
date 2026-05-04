// uops_memory.cpp -- memory hierarchy + cross-CCD pointer-chase benchmarks.
// Mirrors the timing methodology of uops_latency_linux.cpp:
//   - CPU pinning via pthread_setaffinity_np
//   - clock_gettime(CLOCK_THREAD_CPUTIME_ID) primary timer + rdtsc/rdtscp
//   - empty-loop subtraction of harness overhead
//   - median / min / p90 reduction across many samples
//
// Three sub-experiments produced as separate sections:
//   A. Working-set sweep across the cache hierarchy (pointer-chase).
//   B. TLB sweep (4KB-page stride; 2MB-page stride if THP cooperates).
//   C. Cross-CCD latency: setup thread populates the ring on one CCD, the
//      measuring thread chases it from the other CCD.
//
// Build:
//   g++ -O3 -std=c++17 -Wall -Wextra
//       -mcrc32 -mlzcnt -mpopcnt -msse4.2 -mavx2 -mfma
//       -o build/uops_memory src/uops_memory.cpp -lpthread

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

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

#include <sys/mman.h>

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

// -------------------- timing primitives (mirrored from latency harness) --------------------

struct CpuFeatures {
    bool invariant_tsc = false;
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

static bool pin_thread_to_cpu(pthread_t th, unsigned cpu) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    int rc = pthread_setaffinity_np(th, sizeof(set), &set);
    return rc == 0;
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

// -------------------- pointer-chase ring (mmap-backed; configurable stride) --------------------

struct Ring {
    void*  base = nullptr;     // mmap-allocated
    size_t bytes = 0;          // total mmap bytes
    size_t stride = 64;        // distance between adjacent slots, in bytes
    size_t count = 0;          // number of slots in the ring
    bool   used_hugepages = false;
};

static void ring_free(Ring& r) {
    if (r.base != nullptr) {
        munmap(r.base, r.bytes);
    }
    r.base = nullptr;
    r.bytes = 0;
    r.count = 0;
}

// Allocate `bytes` of anonymous memory, optionally request transparent huge
// pages. Returns nullptr on failure.
static void* alloc_aligned(size_t bytes, bool want_hugepage, bool* got_hugepage) {
    if (got_hugepage != nullptr) {
        *got_hugepage = false;
    }
    void* p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) {
        return nullptr;
    }
    if (want_hugepage) {
        int rc = madvise(p, bytes, MADV_HUGEPAGE);
        if (rc == 0 && got_hugepage != nullptr) {
            // madvise succeeding does not guarantee actual hugepage backing,
            // but it is the best signal we can give without /proc inspection.
            *got_hugepage = true;
        }
    }
    return p;
}

// Build a randomized pointer ring inside `buf`. There are `count` slots,
// each `stride` bytes apart. Each slot stores a pointer to the next slot
// in a Fisher-Yates-shuffled cycle, so a dependent chase visits all slots
// exactly once before returning to the start.
static void init_ring_in_buffer(void* buf, size_t count, size_t stride) {
    std::vector<size_t> perm(count);
    std::iota(perm.begin(), perm.end(), 0);

    // Same Park-Miller-ish LCG used in init_pointer_ring.
    uint64_t state = 0x4d595df4d0f33173ull;
    for (size_t i = count - 1; i > 0; --i) {
        state = state * 2862933555777941757ull + 3037000493ull;
        size_t j = static_cast<size_t>(state % (i + 1));
        std::swap(perm[i], perm[j]);
    }

    char* base = static_cast<char*>(buf);
    for (size_t i = 0; i < count; ++i) {
        size_t cur = perm[i];
        size_t next = perm[(i + 1) % count];
        uintptr_t* slot = reinterpret_cast<uintptr_t*>(base + cur * stride);
        *slot = reinterpret_cast<uintptr_t>(base + next * stride);
    }
}

// Allocate a ring sized so `count` slots of `stride` bytes fit. We round
// the buffer up to a page boundary; only `count*stride` bytes are touched.
static bool ring_alloc(Ring& r, size_t count, size_t stride, bool want_hugepage) {
    ring_free(r);
    if (count == 0 || stride < sizeof(uintptr_t)) {
        return false;
    }
    size_t bytes = count * stride;
    // Round up to 2 MiB so THP has a chance even for the small cases.
    size_t round = (1ull << 21);
    bytes = (bytes + round - 1) & ~(round - 1);

    bool got_hp = false;
    void* p = alloc_aligned(bytes, want_hugepage, &got_hp);
    if (p == nullptr) {
        return false;
    }
    r.base = p;
    r.bytes = bytes;
    r.stride = stride;
    r.count = count;
    r.used_hugepages = got_hp;
    init_ring_in_buffer(p, count, stride);
    return true;
}

// -------------------- chase kernels --------------------

// Empty loop with the same shape as the chase loop. Used to subtract the
// constant overhead of the surrounding measurement code.
__attribute__((noinline)) static uint64_t bench_empty(int iters) {
    uint64_t x = sink_u64;
    for (int i = 0; i < iters; ++i) {
        asm volatile("" : [x] "+r"(x) : : "cc");
    }
    sink_u64 = x;
    return x;
}

// 64 dependent loads per iteration, identical to bench_l1_pointer_chase
// in the latency harness.
__attribute__((noinline)) static uint64_t bench_chase(int iters, void* start) {
    uintptr_t p = reinterpret_cast<uintptr_t>(start);
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
    uint64_t thread_ns = 0;
};

static Timing measure_empty(int iters) {
    uint64_t tn0 = read_thread_ns();
    uint64_t t0 = read_tsc_begin();
    bench_empty(iters);
    uint64_t t1 = read_tsc_end();
    uint64_t tn1 = read_thread_ns();
    return Timing{t1 - t0, tn1 - tn0};
}

static Timing measure_chase(int iters, void* start) {
    uint64_t tn0 = read_thread_ns();
    uint64_t t0 = read_tsc_begin();
    bench_chase(iters, start);
    uint64_t t1 = read_tsc_end();
    uint64_t tn1 = read_thread_ns();
    return Timing{t1 - t0, tn1 - tn0};
}

// Full-buffer touch: linearly walk every cache line so each page is faulted
// in and resident in the working CPU's caches before the chase starts.
static void touch_buffer(void* base, size_t bytes) {
    volatile uint8_t* p = static_cast<volatile uint8_t*>(base);
    uint64_t acc = 0;
    for (size_t off = 0; off < bytes; off += 64) {
        acc += p[off];
    }
    sink_u64 ^= acc;
}

// Calibrate empty-loop overhead at the current pinning. Returns medians.
struct EmptyMedians {
    double tsc = 0.0;
    double thread_ns = 0.0;
};

static EmptyMedians calibrate_empty(int iters, int samples) {
    for (int i = 0; i < 16; ++i) {
        measure_empty(iters);
    }
    std::vector<double> tsc_v;
    std::vector<double> thread_v;
    tsc_v.reserve(samples);
    thread_v.reserve(samples);
    for (int i = 0; i < samples; ++i) {
        Timing t = measure_empty(iters);
        tsc_v.push_back(static_cast<double>(t.tsc));
        thread_v.push_back(static_cast<double>(t.thread_ns));
    }
    return EmptyMedians{median(tsc_v), median(thread_v)};
}

struct ChaseStats {
    double med_ns = 0.0;
    double min_ns = 0.0;
    double p90_ns = 0.0;
    double med_tsc = 0.0;
};

// Run the dependent chase at the given starting pointer. Returns the
// per-load statistics after subtracting empty-loop overhead. We allow the
// caller to pass `iters` so very large buffers can use fewer iterations to
// keep the wall-clock per-sample cost in check.
static ChaseStats run_chase(void* start, int iters, int samples,
                            const EmptyMedians& empty) {
    const double ops = static_cast<double>(iters) * OPS_PER_ITER;
    // Warmup: walk the ring a few times so the first measured sample is not
    // dominated by cold-fault / page-walk traffic.
    for (int i = 0; i < 8; ++i) {
        measure_chase(iters, start);
    }
    std::vector<double> tsc_adj;
    std::vector<double> thread_adj;
    tsc_adj.reserve(samples);
    thread_adj.reserve(samples);
    for (int i = 0; i < samples; ++i) {
        Timing t = measure_chase(iters, start);
        tsc_adj.push_back((static_cast<double>(t.tsc) - empty.tsc) / ops);
        thread_adj.push_back((static_cast<double>(t.thread_ns) - empty.thread_ns) / ops);
    }
    ChaseStats s;
    s.med_ns = median(thread_adj);
    s.min_ns = *std::min_element(thread_adj.begin(), thread_adj.end());
    s.p90_ns = percentile(thread_adj, 0.90);
    s.med_tsc = median(tsc_adj);
    return s;
}

// -------------------- experiment A: working-set sweep --------------------

struct Sweep {
    size_t bytes;
    int iters;       // total iters to run; OPS_PER_ITER loads per iter
};

// Pick iters so that total dependent loads is a few hundred thousand even
// for the largest buffers, but keeps small sizes from running absurdly long.
static int choose_iters(size_t bytes) {
    if (bytes <= (256ull * 1024)) {
        return 4096;
    }
    if (bytes <= (4ull * 1024 * 1024)) {
        return 2048;
    }
    if (bytes <= (32ull * 1024 * 1024)) {
        return 1024;
    }
    if (bytes <= (128ull * 1024 * 1024)) {
        return 512;
    }
    return 256;
}

static void section_a_working_set_sweep(int samples, const EmptyMedians& empty,
                                        double ns_per_cycle_estimate) {
    static const size_t sizes[] = {
        8ull * 1024,
        16ull * 1024,
        32ull * 1024,
        64ull * 1024,
        128ull * 1024,
        256ull * 1024,
        512ull * 1024,
        1ull * 1024 * 1024,
        2ull * 1024 * 1024,
        4ull * 1024 * 1024,
        8ull * 1024 * 1024,
        16ull * 1024 * 1024,
        32ull * 1024 * 1024,
        64ull * 1024 * 1024,
        128ull * 1024 * 1024,
        256ull * 1024 * 1024,
        512ull * 1024 * 1024,
        1024ull * 1024 * 1024,
    };

    std::cout << "\n=== Section A: working-set sweep (64B stride, randomized ring) ===\n";
    std::cout << std::left << std::setw(14) << "size"
              << std::right
              << std::setw(10) << "slots"
              << std::setw(12) << "med ns/ld"
              << std::setw(12) << "min ns/ld"
              << std::setw(12) << "p90 ns/ld"
              << std::setw(12) << "med tsc/ld"
              << std::setw(14) << "est cyc/ld"
              << "\n";
    std::cout << std::string(86, '-') << "\n";

    for (size_t bytes : sizes) {
        size_t stride = 64;
        size_t count = bytes / stride;
        if (count < 8) {
            continue;
        }
        int iters = choose_iters(bytes);

        Ring r;
        if (!ring_alloc(r, count, stride, /*want_hugepage=*/false)) {
            std::cout << std::left << std::setw(14) << bytes
                      << " mmap failed\n";
            continue;
        }
        // Touch the buffer once to establish residency in this thread's caches.
        touch_buffer(r.base, r.bytes);

        ChaseStats s = run_chase(r.base, iters, samples, empty);
        double est_cyc = (ns_per_cycle_estimate > 0.0)
                            ? s.med_ns / ns_per_cycle_estimate
                            : s.med_tsc;

        // Pretty-print the size as KB / MB.
        char sizebuf[32];
        if (bytes >= (1ull << 20)) {
            std::snprintf(sizebuf, sizeof(sizebuf), "%zuMB", bytes >> 20);
        } else {
            std::snprintf(sizebuf, sizeof(sizebuf), "%zuKB", bytes >> 10);
        }

        std::cout << std::left << std::setw(14) << sizebuf
                  << std::right
                  << std::setw(10) << count
                  << std::fixed << std::setprecision(3)
                  << std::setw(12) << s.med_ns
                  << std::setw(12) << s.min_ns
                  << std::setw(12) << s.p90_ns
                  << std::setw(12) << s.med_tsc
                  << std::setw(14) << est_cyc
                  << "\n";

        ring_free(r);
    }
}

// -------------------- experiment B: TLB sweep --------------------

// 4 KiB-page stride: every chase step lands on a distinct virtual page.
// Page counts: 64, 256, 1024, 4096, 16384.
//   Zen 3 dTLB L1 = 64 entries (4K), L2 = ~2K entries.
//   The 64-entry case should fit L1 dTLB; 256 should miss L1 dTLB but hit L2;
//   16K should overflow L2 dTLB and pay the page-walk penalty.
//
// 2 MiB-page stride: optionally try MADV_HUGEPAGE. Even if THP refuses to
// promote the region we still report the result for transparency.
static void section_b_tlb_sweep(int samples, const EmptyMedians& empty,
                                double ns_per_cycle_estimate) {
    std::cout << "\n=== Section B: TLB sweep ===\n";

    // ---------- 4 KiB-page sub-sweep ----------
    std::cout << "\n[B.1] 4KB-page stride (each step crosses to a new 4KB page)\n";
    std::cout << std::left << std::setw(12) << "pages"
              << std::right
              << std::setw(14) << "size"
              << std::setw(12) << "med ns/ld"
              << std::setw(12) << "min ns/ld"
              << std::setw(12) << "p90 ns/ld"
              << std::setw(12) << "med tsc/ld"
              << std::setw(14) << "est cyc/ld"
              << "\n";
    std::cout << std::string(86, '-') << "\n";

    static const size_t page_counts[] = {64, 256, 1024, 4096, 16384};
    for (size_t pages : page_counts) {
        size_t stride = 4096;
        size_t count = pages;
        size_t bytes = stride * count;
        int iters = choose_iters(bytes);

        Ring r;
        if (!ring_alloc(r, count, stride, /*want_hugepage=*/false)) {
            std::cout << std::left << std::setw(12) << pages << " mmap failed\n";
            continue;
        }
        touch_buffer(r.base, r.bytes);

        ChaseStats s = run_chase(r.base, iters, samples, empty);
        double est_cyc = (ns_per_cycle_estimate > 0.0)
                            ? s.med_ns / ns_per_cycle_estimate
                            : s.med_tsc;

        char sizebuf[32];
        if (bytes >= (1ull << 20)) {
            std::snprintf(sizebuf, sizeof(sizebuf), "%zuMB", bytes >> 20);
        } else {
            std::snprintf(sizebuf, sizeof(sizebuf), "%zuKB", bytes >> 10);
        }
        std::cout << std::left << std::setw(12) << pages
                  << std::right
                  << std::setw(14) << sizebuf
                  << std::fixed << std::setprecision(3)
                  << std::setw(12) << s.med_ns
                  << std::setw(12) << s.min_ns
                  << std::setw(12) << s.p90_ns
                  << std::setw(12) << s.med_tsc
                  << std::setw(14) << est_cyc
                  << "\n";

        ring_free(r);
    }

    // ---------- 2 MiB-page sub-sweep (optional, best-effort) ----------
    std::cout << "\n[B.2] 2MB-page stride (THP via MADV_HUGEPAGE; best-effort)\n";
    std::cout << std::left << std::setw(12) << "pages"
              << std::right
              << std::setw(14) << "size"
              << std::setw(12) << "med ns/ld"
              << std::setw(12) << "min ns/ld"
              << std::setw(12) << "p90 ns/ld"
              << std::setw(12) << "med tsc/ld"
              << std::setw(14) << "est cyc/ld"
              << std::setw(8)  << "thp?"
              << "\n";
    std::cout << std::string(94, '-') << "\n";

    // Use page counts that cross typical 1G-dTLB-L2 capacities.
    static const size_t hp_page_counts[] = {8, 16, 64, 256, 1024};
    for (size_t pages : hp_page_counts) {
        size_t stride = 2ull * 1024 * 1024;
        size_t count = pages;
        size_t bytes = stride * count;
        int iters = (bytes >= (256ull << 20)) ? 256 : 512;

        Ring r;
        if (!ring_alloc(r, count, stride, /*want_hugepage=*/true)) {
            std::cout << std::left << std::setw(12) << pages << " mmap failed\n";
            continue;
        }
        touch_buffer(r.base, r.bytes);

        ChaseStats s = run_chase(r.base, iters, samples, empty);
        double est_cyc = (ns_per_cycle_estimate > 0.0)
                            ? s.med_ns / ns_per_cycle_estimate
                            : s.med_tsc;

        char sizebuf[32];
        if (bytes >= (1ull << 30)) {
            std::snprintf(sizebuf, sizeof(sizebuf), "%zuGB", bytes >> 30);
        } else {
            std::snprintf(sizebuf, sizeof(sizebuf), "%zuMB", bytes >> 20);
        }
        std::cout << std::left << std::setw(12) << pages
                  << std::right
                  << std::setw(14) << sizebuf
                  << std::fixed << std::setprecision(3)
                  << std::setw(12) << s.med_ns
                  << std::setw(12) << s.min_ns
                  << std::setw(12) << s.p90_ns
                  << std::setw(12) << s.med_tsc
                  << std::setw(14) << est_cyc
                  << std::setw(8)  << (r.used_hugepages ? "adv" : "no")
                  << "\n";

        ring_free(r);
    }
    std::cout << "(thp=adv means MADV_HUGEPAGE accepted; actual backing depends "
                 "on /sys/kernel/mm/transparent_hugepage/enabled.)\n";
}

// -------------------- experiment C: cross-CCD --------------------

struct SetupArgs {
    Ring* ring = nullptr;
    unsigned cpu = 0;
    bool ok = false;
};

// Setup thread: pin to its CPU and write/touch the entire ring buffer.
// Writing every cache line forces them to settle in this CPU's caches and,
// when those caches spill, into this CCD's L3.
static void* setup_thread_main(void* arg) {
    SetupArgs* a = static_cast<SetupArgs*>(arg);
    a->ok = pin_to_logical_cpu(a->cpu);
    if (!a->ok) {
        return nullptr;
    }
    init_ring_in_buffer(a->ring->base, a->ring->count, a->ring->stride);
    // Re-touch so the just-written lines stay hot here.
    char* p = static_cast<char*>(a->ring->base);
    uint64_t acc = 0;
    for (size_t i = 0; i < a->ring->count; ++i) {
        // Read each pointer to keep the line in the writer's cache state
        // expected by the protocol (M/E/S depending on platform).
        acc += *reinterpret_cast<uintptr_t*>(p + i * a->ring->stride);
    }
    sink_u64 ^= acc;
    return nullptr;
}

// One scenario of the cross-CCD experiment: setup on `setup_cpu`, measure
// on `measure_cpu`. We must measure on the *current* thread (so the pinning
// affects the chase loop directly), so we move our affinity to measure_cpu
// after the setup thread has joined.
static ChaseStats run_cross_ccd_scenario(unsigned setup_cpu, unsigned measure_cpu,
                                          size_t bytes, int iters, int samples,
                                          const EmptyMedians& empty,
                                          bool* setup_ok_out) {
    Ring r;
    if (!ring_alloc(r, bytes / 64, 64, /*want_hugepage=*/false)) {
        if (setup_ok_out) {
            *setup_ok_out = false;
        }
        return ChaseStats{};
    }

    SetupArgs sa;
    sa.ring = &r;
    sa.cpu = setup_cpu;

    pthread_t th;
    int rc = pthread_create(&th, nullptr, setup_thread_main, &sa);
    if (rc != 0) {
        ring_free(r);
        if (setup_ok_out) {
            *setup_ok_out = false;
        }
        return ChaseStats{};
    }
    // Belt-and-braces: pin the new thread directly too.
    pin_thread_to_cpu(th, setup_cpu);
    pthread_join(th, nullptr);

    if (setup_ok_out) {
        *setup_ok_out = sa.ok;
    }

    // Now move ourselves to measure_cpu and run the chase. We do NOT touch
    // the buffer first -- that would re-pull the lines into measure_cpu's
    // caches and erase the cross-CCD signal we are trying to observe.
    pin_to_logical_cpu(measure_cpu);
    usleep(5000);

    ChaseStats s = run_chase(r.base, iters, samples, empty);
    ring_free(r);
    return s;
}

static void section_c_cross_ccd(int samples, const EmptyMedians& empty,
                                double ns_per_cycle_estimate) {
    std::cout << "\n=== Section C: cross-CCD pointer-chase ===\n";
    std::cout << "1MB working set (fits in L3 of one CCD; spans inter-CCD link "
                 "when mismatched).\n";
    std::cout << std::left << std::setw(34) << "scenario"
              << std::right
              << std::setw(12) << "med ns/ld"
              << std::setw(12) << "min ns/ld"
              << std::setw(12) << "p90 ns/ld"
              << std::setw(12) << "med tsc/ld"
              << std::setw(14) << "est cyc/ld"
              << "\n";
    std::cout << std::string(96, '-') << "\n";

    size_t bytes = 1ull * 1024 * 1024;
    int iters = choose_iters(bytes);

    struct Scenario {
        const char* name;
        unsigned setup_cpu;
        unsigned measure_cpu;
    };
    Scenario scenarios[] = {
        {"same-CCD baseline (CCD0->CCD0)",  1, 0},
        {"cross-CCD CCD1->CCD0",            8, 0},
        {"cross-CCD CCD0->CCD1",            0, 8},
        {"same-CCD baseline (CCD1->CCD1)",  9, 8},
    };

    for (const Scenario& sc : scenarios) {
        bool setup_ok = false;
        ChaseStats s = run_cross_ccd_scenario(
            sc.setup_cpu, sc.measure_cpu, bytes, iters, samples, empty, &setup_ok);
        double est_cyc = (ns_per_cycle_estimate > 0.0)
                            ? s.med_ns / ns_per_cycle_estimate
                            : s.med_tsc;

        std::cout << std::left << std::setw(34) << sc.name
                  << std::right
                  << std::fixed << std::setprecision(3)
                  << std::setw(12) << s.med_ns
                  << std::setw(12) << s.min_ns
                  << std::setw(12) << s.p90_ns
                  << std::setw(12) << s.med_tsc
                  << std::setw(14) << est_cyc;
        if (!setup_ok) {
            std::cout << "  [setup-pin-failed]";
        }
        std::cout << "\n";
    }
}

// -------------------- main --------------------

// Estimate ns-per-cycle from the empty-loop calibration: thread_ns / tsc
// gives a clean ratio because both are measured over identical work.
// Fallback: assume a 3.4 GHz nominal -> ~0.294 ns/cycle if the empty loop
// somehow returned zero.
static double estimate_ns_per_cycle(const EmptyMedians& empty) {
    if (empty.tsc <= 0.0 || empty.thread_ns <= 0.0) {
        return 1.0 / 3.4;
    }
    return empty.thread_ns / empty.tsc;
}

int main(int argc, char** argv) {
    unsigned logical_cpu = 0;
    int samples = 51;
    if (argc > 1) {
        logical_cpu = static_cast<unsigned>(std::strtoul(argv[1], nullptr, 10));
    }
    if (argc > 2) {
        samples = std::max(3, std::atoi(argv[2]));
    }

    bool pinned = pin_to_logical_cpu(logical_cpu);
    usleep(10000);

    CpuFeatures features = get_cpu_features();

    // Calibrate empty loop overhead with a moderate iter count -- this also
    // gives us a stable empty.tsc / empty.thread_ns ratio for the cycle estimate.
    EmptyMedians empty = calibrate_empty(/*iters=*/2048, /*samples=*/101);
    double ns_per_cycle = estimate_ns_per_cycle(empty);

    std::cout << "uops-style memory hierarchy + cross-CCD probe (linux)\n";
    std::cout << "logical_cpu_requested=" << logical_cpu
              << " pinned=" << (pinned ? "yes" : "no")
              << " current_cpu=" << sched_getcpu()
              << " samples_per_point=" << samples << "\n";
    std::cout << "features: invariant_tsc="
              << (features.invariant_tsc ? "yes" : "no") << "\n";
    std::cout << "timing: primary_column=clock_gettime_thread_ns;"
              << " ns_per_cycle_est=" << std::fixed << std::setprecision(5)
              << ns_per_cycle
              << " (= " << std::setprecision(2) << (1.0 / ns_per_cycle)
              << " GHz nominal)\n";
    std::cout << "empty_loop_median: thread_ns=" << static_cast<uint64_t>(empty.thread_ns)
              << " tsc=" << static_cast<uint64_t>(empty.tsc) << "\n";

    // ---- Section A
    section_a_working_set_sweep(samples, empty, ns_per_cycle);

    // ---- Section B
    // Re-pin to the original CPU in case any earlier code changed affinity.
    pin_to_logical_cpu(logical_cpu);
    section_b_tlb_sweep(samples, empty, ns_per_cycle);

    // ---- Section C
    pin_to_logical_cpu(logical_cpu);
    section_c_cross_ccd(samples, empty, ns_per_cycle);

    return 0;
}
