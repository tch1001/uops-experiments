// Vector instruction microbenchmarks. For each instruction we measure two
// regimes:
//   * LATENCY: dependent chain, op1==dst, so each op waits for the previous
//   * THROUGHPUT: 4-8 independent destination registers rotated, so the
//     scheduler can issue them in parallel
// Output: ns/op median + min + p90, plus tsc/op median, plus a "kind" column
// indicating which regime.
//
// Methodology mirrors src/uops_latency_linux.cpp: REP64 unrolled inline asm
// inside a small loop, clock_gettime(CLOCK_THREAD_CPUTIME_ID) primary timer,
// rdtsc/rdtscp secondary, empty-loop subtraction, median over many samples.
//
// Inline-asm constraint notes for ymm: gcc accepts the "x" constraint for
// __m256/__m256d (and __m256i with movdqa-style ops). We use generic
// __m256 carriers and write the explicit instruction names; the assembler
// only sees register operands like %ymm0, %ymm1. For shuffle byte-control
// vectors we just use any vector value -- we are timing throughput/latency,
// not asserting the result.

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

// Throughput tests use 8 independent destinations. With 64 ops/iter and
// 8 carriers the same register is reused every 8 ops, which is far enough
// apart to break any practical latency chain.
#define REP8_ROT(prefix) \
    prefix(0) prefix(1) prefix(2) prefix(3) \
    prefix(4) prefix(5) prefix(6) prefix(7)
#define REP64_ROT(prefix) \
    REP8_ROT(prefix) REP8_ROT(prefix) REP8_ROT(prefix) REP8_ROT(prefix) \
    REP8_ROT(prefix) REP8_ROT(prefix) REP8_ROT(prefix) REP8_ROT(prefix)

static volatile uint64_t sink_u64 = 0x9e3779b97f4a7c15ull;

using BenchFn = uint64_t (*)(int);

struct CpuFeatures {
    bool invariant_tsc = false;
    bool sse42 = false;
    bool avx2 = false;
    bool fma = false;
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
        f.fma = (ecx & (1u << 12)) != 0;
    }
    if (cpuid_leaf(7, 0, eax, ebx, ecx, edx)) {
        f.avx2 = (ebx & (1u << 5)) != 0;
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

// =====================================================================
// Empty loop baseline.
// =====================================================================
__attribute__((noinline)) static uint64_t bench_empty(int iters) {
    uint64_t x = sink_u64;
    for (int i = 0; i < iters; ++i) {
        asm volatile("" : [x] "+r"(x) : : "cc");
    }
    sink_u64 = x;
    return x;
}

// =====================================================================
// 128-bit SSE/SSE2 integer.
// =====================================================================

__attribute__((noinline)) static uint64_t bench_paddd_xmm_lat(int iters) {
    __m128i v = _mm_set1_epi32(1);
    __m128i c = _mm_set1_epi32(7);
    for (int i = 0; i < iters; ++i) {
        asm volatile(REP64("paddd %[c], %[v]\n\t")
                     : [v] "+x"(v)
                     : [c] "x"(c));
    }
    sink_u64 ^= static_cast<uint64_t>(_mm_cvtsi128_si32(v));
    return sink_u64;
}

__attribute__((noinline)) static uint64_t bench_paddd_xmm_tput(int iters) {
    __m128i v0 = _mm_set1_epi32(1), v1 = _mm_set1_epi32(2);
    __m128i v2 = _mm_set1_epi32(3), v3 = _mm_set1_epi32(4);
    __m128i v4 = _mm_set1_epi32(5), v5 = _mm_set1_epi32(6);
    __m128i v6 = _mm_set1_epi32(7), v7 = _mm_set1_epi32(8);
    __m128i c = _mm_set1_epi32(11);
    for (int i = 0; i < iters; ++i) {
#define ROT(k) "paddd %[c], %[v" #k "]\n\t"
        asm volatile(REP64_ROT(ROT)
                     : [v0] "+x"(v0), [v1] "+x"(v1), [v2] "+x"(v2), [v3] "+x"(v3),
                       [v4] "+x"(v4), [v5] "+x"(v5), [v6] "+x"(v6), [v7] "+x"(v7)
                     : [c] "x"(c));
#undef ROT
    }
    __m128i s = _mm_add_epi32(_mm_add_epi32(_mm_add_epi32(v0, v1), _mm_add_epi32(v2, v3)),
                              _mm_add_epi32(_mm_add_epi32(v4, v5), _mm_add_epi32(v6, v7)));
    sink_u64 ^= static_cast<uint64_t>(_mm_cvtsi128_si32(s));
    return sink_u64;
}

__attribute__((noinline)) static uint64_t bench_pmulld_xmm_lat(int iters) {
    __m128i v = _mm_set1_epi32(3);
    __m128i c = _mm_set1_epi32(1);
    for (int i = 0; i < iters; ++i) {
        asm volatile(REP64("pmulld %[c], %[v]\n\t")
                     : [v] "+x"(v)
                     : [c] "x"(c));
    }
    sink_u64 ^= static_cast<uint64_t>(_mm_cvtsi128_si32(v));
    return sink_u64;
}

__attribute__((noinline)) static uint64_t bench_pmulld_xmm_tput(int iters) {
    __m128i v0 = _mm_set1_epi32(3), v1 = _mm_set1_epi32(5);
    __m128i v2 = _mm_set1_epi32(7), v3 = _mm_set1_epi32(9);
    __m128i v4 = _mm_set1_epi32(11), v5 = _mm_set1_epi32(13);
    __m128i v6 = _mm_set1_epi32(15), v7 = _mm_set1_epi32(17);
    __m128i c = _mm_set1_epi32(1);
    for (int i = 0; i < iters; ++i) {
#define ROT(k) "pmulld %[c], %[v" #k "]\n\t"
        asm volatile(REP64_ROT(ROT)
                     : [v0] "+x"(v0), [v1] "+x"(v1), [v2] "+x"(v2), [v3] "+x"(v3),
                       [v4] "+x"(v4), [v5] "+x"(v5), [v6] "+x"(v6), [v7] "+x"(v7)
                     : [c] "x"(c));
#undef ROT
    }
    __m128i s = _mm_add_epi32(_mm_add_epi32(_mm_add_epi32(v0, v1), _mm_add_epi32(v2, v3)),
                              _mm_add_epi32(_mm_add_epi32(v4, v5), _mm_add_epi32(v6, v7)));
    sink_u64 ^= static_cast<uint64_t>(_mm_cvtsi128_si32(s));
    return sink_u64;
}

__attribute__((noinline)) static uint64_t bench_pshufb_xmm_lat(int iters) {
    __m128i v = _mm_set_epi8(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15);
    __m128i ctl = _mm_set_epi8(15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0);
    for (int i = 0; i < iters; ++i) {
        asm volatile(REP64("pshufb %[c], %[v]\n\t")
                     : [v] "+x"(v)
                     : [c] "x"(ctl));
    }
    sink_u64 ^= static_cast<uint64_t>(_mm_cvtsi128_si32(v));
    return sink_u64;
}

__attribute__((noinline)) static uint64_t bench_pshufb_xmm_tput(int iters) {
    __m128i v0 = _mm_set1_epi8(1), v1 = _mm_set1_epi8(2);
    __m128i v2 = _mm_set1_epi8(3), v3 = _mm_set1_epi8(4);
    __m128i v4 = _mm_set1_epi8(5), v5 = _mm_set1_epi8(6);
    __m128i v6 = _mm_set1_epi8(7), v7 = _mm_set1_epi8(8);
    __m128i ctl = _mm_set_epi8(15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0);
    for (int i = 0; i < iters; ++i) {
#define ROT(k) "pshufb %[c], %[v" #k "]\n\t"
        asm volatile(REP64_ROT(ROT)
                     : [v0] "+x"(v0), [v1] "+x"(v1), [v2] "+x"(v2), [v3] "+x"(v3),
                       [v4] "+x"(v4), [v5] "+x"(v5), [v6] "+x"(v6), [v7] "+x"(v7)
                     : [c] "x"(ctl));
#undef ROT
    }
    __m128i s = _mm_xor_si128(_mm_xor_si128(_mm_xor_si128(v0, v1), _mm_xor_si128(v2, v3)),
                              _mm_xor_si128(_mm_xor_si128(v4, v5), _mm_xor_si128(v6, v7)));
    sink_u64 ^= static_cast<uint64_t>(_mm_cvtsi128_si32(s));
    return sink_u64;
}

// pcmpeqb is a "compare against another reg" instruction; result bits are set
// when bytes match. Latency form: dst depends on dst+src each iter.
__attribute__((noinline)) static uint64_t bench_pcmpeqb_xmm_lat(int iters) {
    __m128i v = _mm_set1_epi8(0x55);
    __m128i c = _mm_set1_epi8(0x55);
    for (int i = 0; i < iters; ++i) {
        asm volatile(REP64("pcmpeqb %[c], %[v]\n\t")
                     : [v] "+x"(v)
                     : [c] "x"(c));
    }
    sink_u64 ^= static_cast<uint64_t>(_mm_cvtsi128_si32(v));
    return sink_u64;
}

__attribute__((noinline)) static uint64_t bench_pcmpeqb_xmm_tput(int iters) {
    __m128i v0 = _mm_set1_epi8(0x11), v1 = _mm_set1_epi8(0x22);
    __m128i v2 = _mm_set1_epi8(0x33), v3 = _mm_set1_epi8(0x44);
    __m128i v4 = _mm_set1_epi8(0x55), v5 = _mm_set1_epi8(0x66);
    __m128i v6 = _mm_set1_epi8(0x77), v7 = _mm_set1_epi8(0x77);
    __m128i c = _mm_set1_epi8(0x77);
    for (int i = 0; i < iters; ++i) {
#define ROT(k) "pcmpeqb %[c], %[v" #k "]\n\t"
        asm volatile(REP64_ROT(ROT)
                     : [v0] "+x"(v0), [v1] "+x"(v1), [v2] "+x"(v2), [v3] "+x"(v3),
                       [v4] "+x"(v4), [v5] "+x"(v5), [v6] "+x"(v6), [v7] "+x"(v7)
                     : [c] "x"(c));
#undef ROT
    }
    __m128i s = _mm_xor_si128(_mm_xor_si128(_mm_xor_si128(v0, v1), _mm_xor_si128(v2, v3)),
                              _mm_xor_si128(_mm_xor_si128(v4, v5), _mm_xor_si128(v6, v7)));
    sink_u64 ^= static_cast<uint64_t>(_mm_cvtsi128_si32(s));
    return sink_u64;
}

__attribute__((noinline)) static uint64_t bench_por_xmm_lat(int iters) {
    __m128i v = _mm_set1_epi32(0x1);
    __m128i c = _mm_set1_epi32(0x2);
    for (int i = 0; i < iters; ++i) {
        asm volatile(REP64("por %[c], %[v]\n\t")
                     : [v] "+x"(v)
                     : [c] "x"(c));
    }
    sink_u64 ^= static_cast<uint64_t>(_mm_cvtsi128_si32(v));
    return sink_u64;
}

__attribute__((noinline)) static uint64_t bench_por_xmm_tput(int iters) {
    __m128i v0 = _mm_set1_epi32(1), v1 = _mm_set1_epi32(2);
    __m128i v2 = _mm_set1_epi32(4), v3 = _mm_set1_epi32(8);
    __m128i v4 = _mm_set1_epi32(16), v5 = _mm_set1_epi32(32);
    __m128i v6 = _mm_set1_epi32(64), v7 = _mm_set1_epi32(128);
    __m128i c = _mm_set1_epi32(0x12345);
    for (int i = 0; i < iters; ++i) {
#define ROT(k) "por %[c], %[v" #k "]\n\t"
        asm volatile(REP64_ROT(ROT)
                     : [v0] "+x"(v0), [v1] "+x"(v1), [v2] "+x"(v2), [v3] "+x"(v3),
                       [v4] "+x"(v4), [v5] "+x"(v5), [v6] "+x"(v6), [v7] "+x"(v7)
                     : [c] "x"(c));
#undef ROT
    }
    __m128i s = _mm_xor_si128(_mm_xor_si128(_mm_xor_si128(v0, v1), _mm_xor_si128(v2, v3)),
                              _mm_xor_si128(_mm_xor_si128(v4, v5), _mm_xor_si128(v6, v7)));
    sink_u64 ^= static_cast<uint64_t>(_mm_cvtsi128_si32(s));
    return sink_u64;
}

// =====================================================================
// 256-bit AVX2 integer.
// =====================================================================

__attribute__((noinline)) static uint64_t bench_vpaddd_ymm_lat(int iters) {
    __m256i v = _mm256_set1_epi32(1);
    __m256i c = _mm256_set1_epi32(7);
    for (int i = 0; i < iters; ++i) {
        asm volatile(REP64("vpaddd %[c], %[v], %[v]\n\t")
                     : [v] "+x"(v)
                     : [c] "x"(c));
    }
    sink_u64 ^= static_cast<uint64_t>(_mm256_extract_epi32(v, 0));
    return sink_u64;
}

__attribute__((noinline)) static uint64_t bench_vpaddd_ymm_tput(int iters) {
    __m256i v0 = _mm256_set1_epi32(1), v1 = _mm256_set1_epi32(2);
    __m256i v2 = _mm256_set1_epi32(3), v3 = _mm256_set1_epi32(4);
    __m256i v4 = _mm256_set1_epi32(5), v5 = _mm256_set1_epi32(6);
    __m256i v6 = _mm256_set1_epi32(7), v7 = _mm256_set1_epi32(8);
    __m256i c = _mm256_set1_epi32(11);
    for (int i = 0; i < iters; ++i) {
#define ROT(k) "vpaddd %[c], %[v" #k "], %[v" #k "]\n\t"
        asm volatile(REP64_ROT(ROT)
                     : [v0] "+x"(v0), [v1] "+x"(v1), [v2] "+x"(v2), [v3] "+x"(v3),
                       [v4] "+x"(v4), [v5] "+x"(v5), [v6] "+x"(v6), [v7] "+x"(v7)
                     : [c] "x"(c));
#undef ROT
    }
    __m256i s = _mm256_add_epi32(
        _mm256_add_epi32(_mm256_add_epi32(v0, v1), _mm256_add_epi32(v2, v3)),
        _mm256_add_epi32(_mm256_add_epi32(v4, v5), _mm256_add_epi32(v6, v7)));
    sink_u64 ^= static_cast<uint64_t>(_mm256_extract_epi32(s, 0));
    return sink_u64;
}

__attribute__((noinline)) static uint64_t bench_vpmulld_ymm_lat(int iters) {
    __m256i v = _mm256_set1_epi32(3);
    __m256i c = _mm256_set1_epi32(1);
    for (int i = 0; i < iters; ++i) {
        asm volatile(REP64("vpmulld %[c], %[v], %[v]\n\t")
                     : [v] "+x"(v)
                     : [c] "x"(c));
    }
    sink_u64 ^= static_cast<uint64_t>(_mm256_extract_epi32(v, 0));
    return sink_u64;
}

__attribute__((noinline)) static uint64_t bench_vpmulld_ymm_tput(int iters) {
    __m256i v0 = _mm256_set1_epi32(3), v1 = _mm256_set1_epi32(5);
    __m256i v2 = _mm256_set1_epi32(7), v3 = _mm256_set1_epi32(9);
    __m256i v4 = _mm256_set1_epi32(11), v5 = _mm256_set1_epi32(13);
    __m256i v6 = _mm256_set1_epi32(15), v7 = _mm256_set1_epi32(17);
    __m256i c = _mm256_set1_epi32(1);
    for (int i = 0; i < iters; ++i) {
#define ROT(k) "vpmulld %[c], %[v" #k "], %[v" #k "]\n\t"
        asm volatile(REP64_ROT(ROT)
                     : [v0] "+x"(v0), [v1] "+x"(v1), [v2] "+x"(v2), [v3] "+x"(v3),
                       [v4] "+x"(v4), [v5] "+x"(v5), [v6] "+x"(v6), [v7] "+x"(v7)
                     : [c] "x"(c));
#undef ROT
    }
    __m256i s = _mm256_add_epi32(
        _mm256_add_epi32(_mm256_add_epi32(v0, v1), _mm256_add_epi32(v2, v3)),
        _mm256_add_epi32(_mm256_add_epi32(v4, v5), _mm256_add_epi32(v6, v7)));
    sink_u64 ^= static_cast<uint64_t>(_mm256_extract_epi32(s, 0));
    return sink_u64;
}

__attribute__((noinline)) static uint64_t bench_vpshufb_ymm_lat(int iters) {
    __m256i v = _mm256_set_epi8(
        0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
        15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0);
    __m256i ctl = _mm256_set_epi8(
        15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0,
        0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15);
    for (int i = 0; i < iters; ++i) {
        asm volatile(REP64("vpshufb %[c], %[v], %[v]\n\t")
                     : [v] "+x"(v)
                     : [c] "x"(ctl));
    }
    sink_u64 ^= static_cast<uint64_t>(_mm256_extract_epi32(v, 0));
    return sink_u64;
}

__attribute__((noinline)) static uint64_t bench_vpshufb_ymm_tput(int iters) {
    __m256i v0 = _mm256_set1_epi8(1), v1 = _mm256_set1_epi8(2);
    __m256i v2 = _mm256_set1_epi8(3), v3 = _mm256_set1_epi8(4);
    __m256i v4 = _mm256_set1_epi8(5), v5 = _mm256_set1_epi8(6);
    __m256i v6 = _mm256_set1_epi8(7), v7 = _mm256_set1_epi8(8);
    __m256i ctl = _mm256_set_epi8(
        15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0,
        0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15);
    for (int i = 0; i < iters; ++i) {
#define ROT(k) "vpshufb %[c], %[v" #k "], %[v" #k "]\n\t"
        asm volatile(REP64_ROT(ROT)
                     : [v0] "+x"(v0), [v1] "+x"(v1), [v2] "+x"(v2), [v3] "+x"(v3),
                       [v4] "+x"(v4), [v5] "+x"(v5), [v6] "+x"(v6), [v7] "+x"(v7)
                     : [c] "x"(ctl));
#undef ROT
    }
    __m256i s = _mm256_xor_si256(
        _mm256_xor_si256(_mm256_xor_si256(v0, v1), _mm256_xor_si256(v2, v3)),
        _mm256_xor_si256(_mm256_xor_si256(v4, v5), _mm256_xor_si256(v6, v7)));
    sink_u64 ^= static_cast<uint64_t>(_mm256_extract_epi32(s, 0));
    return sink_u64;
}

__attribute__((noinline)) static uint64_t bench_vpor_ymm_lat(int iters) {
    __m256i v = _mm256_set1_epi32(1);
    __m256i c = _mm256_set1_epi32(2);
    for (int i = 0; i < iters; ++i) {
        asm volatile(REP64("vpor %[c], %[v], %[v]\n\t")
                     : [v] "+x"(v)
                     : [c] "x"(c));
    }
    sink_u64 ^= static_cast<uint64_t>(_mm256_extract_epi32(v, 0));
    return sink_u64;
}

__attribute__((noinline)) static uint64_t bench_vpor_ymm_tput(int iters) {
    __m256i v0 = _mm256_set1_epi32(1), v1 = _mm256_set1_epi32(2);
    __m256i v2 = _mm256_set1_epi32(4), v3 = _mm256_set1_epi32(8);
    __m256i v4 = _mm256_set1_epi32(16), v5 = _mm256_set1_epi32(32);
    __m256i v6 = _mm256_set1_epi32(64), v7 = _mm256_set1_epi32(128);
    __m256i c = _mm256_set1_epi32(0x12345);
    for (int i = 0; i < iters; ++i) {
#define ROT(k) "vpor %[c], %[v" #k "], %[v" #k "]\n\t"
        asm volatile(REP64_ROT(ROT)
                     : [v0] "+x"(v0), [v1] "+x"(v1), [v2] "+x"(v2), [v3] "+x"(v3),
                       [v4] "+x"(v4), [v5] "+x"(v5), [v6] "+x"(v6), [v7] "+x"(v7)
                     : [c] "x"(c));
#undef ROT
    }
    __m256i s = _mm256_xor_si256(
        _mm256_xor_si256(_mm256_xor_si256(v0, v1), _mm256_xor_si256(v2, v3)),
        _mm256_xor_si256(_mm256_xor_si256(v4, v5), _mm256_xor_si256(v6, v7)));
    sink_u64 ^= static_cast<uint64_t>(_mm256_extract_epi32(s, 0));
    return sink_u64;
}

__attribute__((noinline)) static uint64_t bench_vpaddq_ymm_lat(int iters) {
    __m256i v = _mm256_set1_epi64x(1);
    __m256i c = _mm256_set1_epi64x(7);
    for (int i = 0; i < iters; ++i) {
        asm volatile(REP64("vpaddq %[c], %[v], %[v]\n\t")
                     : [v] "+x"(v)
                     : [c] "x"(c));
    }
    sink_u64 ^= static_cast<uint64_t>(_mm256_extract_epi64(v, 0));
    return sink_u64;
}

__attribute__((noinline)) static uint64_t bench_vpaddq_ymm_tput(int iters) {
    __m256i v0 = _mm256_set1_epi64x(1), v1 = _mm256_set1_epi64x(2);
    __m256i v2 = _mm256_set1_epi64x(3), v3 = _mm256_set1_epi64x(4);
    __m256i v4 = _mm256_set1_epi64x(5), v5 = _mm256_set1_epi64x(6);
    __m256i v6 = _mm256_set1_epi64x(7), v7 = _mm256_set1_epi64x(8);
    __m256i c = _mm256_set1_epi64x(11);
    for (int i = 0; i < iters; ++i) {
#define ROT(k) "vpaddq %[c], %[v" #k "], %[v" #k "]\n\t"
        asm volatile(REP64_ROT(ROT)
                     : [v0] "+x"(v0), [v1] "+x"(v1), [v2] "+x"(v2), [v3] "+x"(v3),
                       [v4] "+x"(v4), [v5] "+x"(v5), [v6] "+x"(v6), [v7] "+x"(v7)
                     : [c] "x"(c));
#undef ROT
    }
    __m256i s = _mm256_add_epi64(
        _mm256_add_epi64(_mm256_add_epi64(v0, v1), _mm256_add_epi64(v2, v3)),
        _mm256_add_epi64(_mm256_add_epi64(v4, v5), _mm256_add_epi64(v6, v7)));
    sink_u64 ^= static_cast<uint64_t>(_mm256_extract_epi64(s, 0));
    return sink_u64;
}

// =====================================================================
// 256-bit AVX FP add/mul.
// =====================================================================

__attribute__((noinline)) static uint64_t bench_vaddps_ymm_lat(int iters) {
    __m256 v = _mm256_set1_ps(1.0f);
    __m256 c = _mm256_set1_ps(0.0001f);
    for (int i = 0; i < iters; ++i) {
        asm volatile(REP64("vaddps %[c], %[v], %[v]\n\t")
                     : [v] "+x"(v)
                     : [c] "x"(c));
    }
    sink_u64 ^= static_cast<uint64_t>(_mm256_cvtss_f32(v));
    return sink_u64;
}

__attribute__((noinline)) static uint64_t bench_vaddps_ymm_tput(int iters) {
    __m256 v0 = _mm256_set1_ps(1.0f), v1 = _mm256_set1_ps(2.0f);
    __m256 v2 = _mm256_set1_ps(3.0f), v3 = _mm256_set1_ps(4.0f);
    __m256 v4 = _mm256_set1_ps(5.0f), v5 = _mm256_set1_ps(6.0f);
    __m256 v6 = _mm256_set1_ps(7.0f), v7 = _mm256_set1_ps(8.0f);
    __m256 c = _mm256_set1_ps(0.0001f);
    for (int i = 0; i < iters; ++i) {
#define ROT(k) "vaddps %[c], %[v" #k "], %[v" #k "]\n\t"
        asm volatile(REP64_ROT(ROT)
                     : [v0] "+x"(v0), [v1] "+x"(v1), [v2] "+x"(v2), [v3] "+x"(v3),
                       [v4] "+x"(v4), [v5] "+x"(v5), [v6] "+x"(v6), [v7] "+x"(v7)
                     : [c] "x"(c));
#undef ROT
    }
    __m256 s = _mm256_add_ps(_mm256_add_ps(_mm256_add_ps(v0, v1), _mm256_add_ps(v2, v3)),
                             _mm256_add_ps(_mm256_add_ps(v4, v5), _mm256_add_ps(v6, v7)));
    sink_u64 ^= static_cast<uint64_t>(_mm256_cvtss_f32(s));
    return sink_u64;
}

__attribute__((noinline)) static uint64_t bench_vmulps_ymm_lat(int iters) {
    __m256 v = _mm256_set1_ps(1.0f);
    __m256 c = _mm256_set1_ps(1.0000001f);
    for (int i = 0; i < iters; ++i) {
        asm volatile(REP64("vmulps %[c], %[v], %[v]\n\t")
                     : [v] "+x"(v)
                     : [c] "x"(c));
    }
    sink_u64 ^= static_cast<uint64_t>(_mm256_cvtss_f32(v));
    return sink_u64;
}

__attribute__((noinline)) static uint64_t bench_vmulps_ymm_tput(int iters) {
    __m256 v0 = _mm256_set1_ps(1.0f), v1 = _mm256_set1_ps(1.1f);
    __m256 v2 = _mm256_set1_ps(1.2f), v3 = _mm256_set1_ps(1.3f);
    __m256 v4 = _mm256_set1_ps(1.4f), v5 = _mm256_set1_ps(1.5f);
    __m256 v6 = _mm256_set1_ps(1.6f), v7 = _mm256_set1_ps(1.7f);
    __m256 c = _mm256_set1_ps(1.0000001f);
    for (int i = 0; i < iters; ++i) {
#define ROT(k) "vmulps %[c], %[v" #k "], %[v" #k "]\n\t"
        asm volatile(REP64_ROT(ROT)
                     : [v0] "+x"(v0), [v1] "+x"(v1), [v2] "+x"(v2), [v3] "+x"(v3),
                       [v4] "+x"(v4), [v5] "+x"(v5), [v6] "+x"(v6), [v7] "+x"(v7)
                     : [c] "x"(c));
#undef ROT
    }
    __m256 s = _mm256_add_ps(_mm256_add_ps(_mm256_add_ps(v0, v1), _mm256_add_ps(v2, v3)),
                             _mm256_add_ps(_mm256_add_ps(v4, v5), _mm256_add_ps(v6, v7)));
    sink_u64 ^= static_cast<uint64_t>(_mm256_cvtss_f32(s));
    return sink_u64;
}

__attribute__((noinline)) static uint64_t bench_vaddpd_ymm_lat(int iters) {
    __m256d v = _mm256_set1_pd(1.0);
    __m256d c = _mm256_set1_pd(0.0001);
    for (int i = 0; i < iters; ++i) {
        asm volatile(REP64("vaddpd %[c], %[v], %[v]\n\t")
                     : [v] "+x"(v)
                     : [c] "x"(c));
    }
    sink_u64 ^= static_cast<uint64_t>(_mm256_cvtsd_f64(v));
    return sink_u64;
}

__attribute__((noinline)) static uint64_t bench_vaddpd_ymm_tput(int iters) {
    __m256d v0 = _mm256_set1_pd(1.0), v1 = _mm256_set1_pd(2.0);
    __m256d v2 = _mm256_set1_pd(3.0), v3 = _mm256_set1_pd(4.0);
    __m256d v4 = _mm256_set1_pd(5.0), v5 = _mm256_set1_pd(6.0);
    __m256d v6 = _mm256_set1_pd(7.0), v7 = _mm256_set1_pd(8.0);
    __m256d c = _mm256_set1_pd(0.0001);
    for (int i = 0; i < iters; ++i) {
#define ROT(k) "vaddpd %[c], %[v" #k "], %[v" #k "]\n\t"
        asm volatile(REP64_ROT(ROT)
                     : [v0] "+x"(v0), [v1] "+x"(v1), [v2] "+x"(v2), [v3] "+x"(v3),
                       [v4] "+x"(v4), [v5] "+x"(v5), [v6] "+x"(v6), [v7] "+x"(v7)
                     : [c] "x"(c));
#undef ROT
    }
    __m256d s = _mm256_add_pd(_mm256_add_pd(_mm256_add_pd(v0, v1), _mm256_add_pd(v2, v3)),
                              _mm256_add_pd(_mm256_add_pd(v4, v5), _mm256_add_pd(v6, v7)));
    sink_u64 ^= static_cast<uint64_t>(_mm256_cvtsd_f64(s));
    return sink_u64;
}

__attribute__((noinline)) static uint64_t bench_vmulpd_ymm_lat(int iters) {
    __m256d v = _mm256_set1_pd(1.0);
    __m256d c = _mm256_set1_pd(1.0000000001);
    for (int i = 0; i < iters; ++i) {
        asm volatile(REP64("vmulpd %[c], %[v], %[v]\n\t")
                     : [v] "+x"(v)
                     : [c] "x"(c));
    }
    sink_u64 ^= static_cast<uint64_t>(_mm256_cvtsd_f64(v));
    return sink_u64;
}

__attribute__((noinline)) static uint64_t bench_vmulpd_ymm_tput(int iters) {
    __m256d v0 = _mm256_set1_pd(1.0), v1 = _mm256_set1_pd(1.1);
    __m256d v2 = _mm256_set1_pd(1.2), v3 = _mm256_set1_pd(1.3);
    __m256d v4 = _mm256_set1_pd(1.4), v5 = _mm256_set1_pd(1.5);
    __m256d v6 = _mm256_set1_pd(1.6), v7 = _mm256_set1_pd(1.7);
    __m256d c = _mm256_set1_pd(1.0000000001);
    for (int i = 0; i < iters; ++i) {
#define ROT(k) "vmulpd %[c], %[v" #k "], %[v" #k "]\n\t"
        asm volatile(REP64_ROT(ROT)
                     : [v0] "+x"(v0), [v1] "+x"(v1), [v2] "+x"(v2), [v3] "+x"(v3),
                       [v4] "+x"(v4), [v5] "+x"(v5), [v6] "+x"(v6), [v7] "+x"(v7)
                     : [c] "x"(c));
#undef ROT
    }
    __m256d s = _mm256_add_pd(_mm256_add_pd(_mm256_add_pd(v0, v1), _mm256_add_pd(v2, v3)),
                              _mm256_add_pd(_mm256_add_pd(v4, v5), _mm256_add_pd(v6, v7)));
    sink_u64 ^= static_cast<uint64_t>(_mm256_cvtsd_f64(s));
    return sink_u64;
}

// =====================================================================
// FMA. vfmadd231ps dst, a, b   computes  dst = a*b + dst.
// For latency we pass dst as one of the muls so the chain runs through dst.
// vfmadd231 reads dst, mul-adds a*b into it: dst = dst + a*b. Latency is
// dst -> dst because the next op also reads dst.
// =====================================================================

__attribute__((noinline)) static uint64_t bench_vfmadd231ps_ymm_lat(int iters) {
    __m256 v = _mm256_set1_ps(1.0f);
    __m256 a = _mm256_set1_ps(0.0f);
    __m256 b = _mm256_set1_ps(0.0001f);
    for (int i = 0; i < iters; ++i) {
        asm volatile(REP64("vfmadd231ps %[b], %[a], %[v]\n\t")
                     : [v] "+x"(v)
                     : [a] "x"(a), [b] "x"(b));
    }
    sink_u64 ^= static_cast<uint64_t>(_mm256_cvtss_f32(v));
    return sink_u64;
}

__attribute__((noinline)) static uint64_t bench_vfmadd231ps_ymm_tput(int iters) {
    __m256 v0 = _mm256_set1_ps(1.0f), v1 = _mm256_set1_ps(2.0f);
    __m256 v2 = _mm256_set1_ps(3.0f), v3 = _mm256_set1_ps(4.0f);
    __m256 v4 = _mm256_set1_ps(5.0f), v5 = _mm256_set1_ps(6.0f);
    __m256 v6 = _mm256_set1_ps(7.0f), v7 = _mm256_set1_ps(8.0f);
    __m256 a = _mm256_set1_ps(0.0f);
    __m256 b = _mm256_set1_ps(0.0001f);
    for (int i = 0; i < iters; ++i) {
#define ROT(k) "vfmadd231ps %[b], %[a], %[v" #k "]\n\t"
        asm volatile(REP64_ROT(ROT)
                     : [v0] "+x"(v0), [v1] "+x"(v1), [v2] "+x"(v2), [v3] "+x"(v3),
                       [v4] "+x"(v4), [v5] "+x"(v5), [v6] "+x"(v6), [v7] "+x"(v7)
                     : [a] "x"(a), [b] "x"(b));
#undef ROT
    }
    __m256 s = _mm256_add_ps(_mm256_add_ps(_mm256_add_ps(v0, v1), _mm256_add_ps(v2, v3)),
                             _mm256_add_ps(_mm256_add_ps(v4, v5), _mm256_add_ps(v6, v7)));
    sink_u64 ^= static_cast<uint64_t>(_mm256_cvtss_f32(s));
    return sink_u64;
}

__attribute__((noinline)) static uint64_t bench_vfmadd231pd_ymm_lat(int iters) {
    __m256d v = _mm256_set1_pd(1.0);
    __m256d a = _mm256_set1_pd(0.0);
    __m256d b = _mm256_set1_pd(0.0001);
    for (int i = 0; i < iters; ++i) {
        asm volatile(REP64("vfmadd231pd %[b], %[a], %[v]\n\t")
                     : [v] "+x"(v)
                     : [a] "x"(a), [b] "x"(b));
    }
    sink_u64 ^= static_cast<uint64_t>(_mm256_cvtsd_f64(v));
    return sink_u64;
}

__attribute__((noinline)) static uint64_t bench_vfmadd231pd_ymm_tput(int iters) {
    __m256d v0 = _mm256_set1_pd(1.0), v1 = _mm256_set1_pd(2.0);
    __m256d v2 = _mm256_set1_pd(3.0), v3 = _mm256_set1_pd(4.0);
    __m256d v4 = _mm256_set1_pd(5.0), v5 = _mm256_set1_pd(6.0);
    __m256d v6 = _mm256_set1_pd(7.0), v7 = _mm256_set1_pd(8.0);
    __m256d a = _mm256_set1_pd(0.0);
    __m256d b = _mm256_set1_pd(0.0001);
    for (int i = 0; i < iters; ++i) {
#define ROT(k) "vfmadd231pd %[b], %[a], %[v" #k "]\n\t"
        asm volatile(REP64_ROT(ROT)
                     : [v0] "+x"(v0), [v1] "+x"(v1), [v2] "+x"(v2), [v3] "+x"(v3),
                       [v4] "+x"(v4), [v5] "+x"(v5), [v6] "+x"(v6), [v7] "+x"(v7)
                     : [a] "x"(a), [b] "x"(b));
#undef ROT
    }
    __m256d s = _mm256_add_pd(_mm256_add_pd(_mm256_add_pd(v0, v1), _mm256_add_pd(v2, v3)),
                              _mm256_add_pd(_mm256_add_pd(v4, v5), _mm256_add_pd(v6, v7)));
    sink_u64 ^= static_cast<uint64_t>(_mm256_cvtsd_f64(s));
    return sink_u64;
}

__attribute__((noinline)) static uint64_t bench_vfnmadd231ps_ymm_lat(int iters) {
    __m256 v = _mm256_set1_ps(1.0f);
    __m256 a = _mm256_set1_ps(0.0f);
    __m256 b = _mm256_set1_ps(0.0001f);
    for (int i = 0; i < iters; ++i) {
        asm volatile(REP64("vfnmadd231ps %[b], %[a], %[v]\n\t")
                     : [v] "+x"(v)
                     : [a] "x"(a), [b] "x"(b));
    }
    sink_u64 ^= static_cast<uint64_t>(_mm256_cvtss_f32(v));
    return sink_u64;
}

__attribute__((noinline)) static uint64_t bench_vfnmadd231ps_ymm_tput(int iters) {
    __m256 v0 = _mm256_set1_ps(1.0f), v1 = _mm256_set1_ps(2.0f);
    __m256 v2 = _mm256_set1_ps(3.0f), v3 = _mm256_set1_ps(4.0f);
    __m256 v4 = _mm256_set1_ps(5.0f), v5 = _mm256_set1_ps(6.0f);
    __m256 v6 = _mm256_set1_ps(7.0f), v7 = _mm256_set1_ps(8.0f);
    __m256 a = _mm256_set1_ps(0.0f);
    __m256 b = _mm256_set1_ps(0.0001f);
    for (int i = 0; i < iters; ++i) {
#define ROT(k) "vfnmadd231ps %[b], %[a], %[v" #k "]\n\t"
        asm volatile(REP64_ROT(ROT)
                     : [v0] "+x"(v0), [v1] "+x"(v1), [v2] "+x"(v2), [v3] "+x"(v3),
                       [v4] "+x"(v4), [v5] "+x"(v5), [v6] "+x"(v6), [v7] "+x"(v7)
                     : [a] "x"(a), [b] "x"(b));
#undef ROT
    }
    __m256 s = _mm256_add_ps(_mm256_add_ps(_mm256_add_ps(v0, v1), _mm256_add_ps(v2, v3)),
                             _mm256_add_ps(_mm256_add_ps(v4, v5), _mm256_add_ps(v6, v7)));
    sink_u64 ^= static_cast<uint64_t>(_mm256_cvtss_f32(s));
    return sink_u64;
}

// =====================================================================
// Lane-crossing shuffles. Zen 3 doc says +1 cycle vs in-lane shuffles.
// =====================================================================

__attribute__((noinline)) static uint64_t bench_vpermq_ymm_lat(int iters) {
    __m256i v = _mm256_set_epi64x(1, 2, 3, 4);
    for (int i = 0; i < iters; ++i) {
        asm volatile(REP64("vpermq $0x1b, %[v], %[v]\n\t")
                     : [v] "+x"(v));
    }
    sink_u64 ^= static_cast<uint64_t>(_mm256_extract_epi64(v, 0));
    return sink_u64;
}

__attribute__((noinline)) static uint64_t bench_vpermq_ymm_tput(int iters) {
    __m256i v0 = _mm256_set_epi64x(1,2,3,4), v1 = _mm256_set_epi64x(5,6,7,8);
    __m256i v2 = _mm256_set_epi64x(9,10,11,12), v3 = _mm256_set_epi64x(13,14,15,16);
    __m256i v4 = _mm256_set_epi64x(17,18,19,20), v5 = _mm256_set_epi64x(21,22,23,24);
    __m256i v6 = _mm256_set_epi64x(25,26,27,28), v7 = _mm256_set_epi64x(29,30,31,32);
    for (int i = 0; i < iters; ++i) {
#define ROT(k) "vpermq $0x1b, %[v" #k "], %[v" #k "]\n\t"
        asm volatile(REP64_ROT(ROT)
                     : [v0] "+x"(v0), [v1] "+x"(v1), [v2] "+x"(v2), [v3] "+x"(v3),
                       [v4] "+x"(v4), [v5] "+x"(v5), [v6] "+x"(v6), [v7] "+x"(v7));
#undef ROT
    }
    __m256i s = _mm256_xor_si256(
        _mm256_xor_si256(_mm256_xor_si256(v0, v1), _mm256_xor_si256(v2, v3)),
        _mm256_xor_si256(_mm256_xor_si256(v4, v5), _mm256_xor_si256(v6, v7)));
    sink_u64 ^= static_cast<uint64_t>(_mm256_extract_epi64(s, 0));
    return sink_u64;
}

// vpermd takes a register-form 32-bit lane index in src1, reorders src2.
// Latency form: dst depends on dst (and the constant index).
__attribute__((noinline)) static uint64_t bench_vpermd_ymm_lat(int iters) {
    __m256i v = _mm256_set_epi32(0,1,2,3,4,5,6,7);
    __m256i idx = _mm256_set_epi32(7,6,5,4,3,2,1,0);
    for (int i = 0; i < iters; ++i) {
        asm volatile(REP64("vpermd %[v], %[i], %[v]\n\t")
                     : [v] "+x"(v)
                     : [i] "x"(idx));
    }
    sink_u64 ^= static_cast<uint64_t>(_mm256_extract_epi32(v, 0));
    return sink_u64;
}

__attribute__((noinline)) static uint64_t bench_vpermd_ymm_tput(int iters) {
    __m256i v0 = _mm256_set1_epi32(1), v1 = _mm256_set1_epi32(2);
    __m256i v2 = _mm256_set1_epi32(3), v3 = _mm256_set1_epi32(4);
    __m256i v4 = _mm256_set1_epi32(5), v5 = _mm256_set1_epi32(6);
    __m256i v6 = _mm256_set1_epi32(7), v7 = _mm256_set1_epi32(8);
    __m256i idx = _mm256_set_epi32(7,6,5,4,3,2,1,0);
    for (int i = 0; i < iters; ++i) {
#define ROT(k) "vpermd %[v" #k "], %[i], %[v" #k "]\n\t"
        asm volatile(REP64_ROT(ROT)
                     : [v0] "+x"(v0), [v1] "+x"(v1), [v2] "+x"(v2), [v3] "+x"(v3),
                       [v4] "+x"(v4), [v5] "+x"(v5), [v6] "+x"(v6), [v7] "+x"(v7)
                     : [i] "x"(idx));
#undef ROT
    }
    __m256i s = _mm256_add_epi32(
        _mm256_add_epi32(_mm256_add_epi32(v0, v1), _mm256_add_epi32(v2, v3)),
        _mm256_add_epi32(_mm256_add_epi32(v4, v5), _mm256_add_epi32(v6, v7)));
    sink_u64 ^= static_cast<uint64_t>(_mm256_extract_epi32(s, 0));
    return sink_u64;
}

// =====================================================================
// 256-bit divides / sqrt. High latency, expose via dependent chain.
// Throughput regime uses 8 independent dsts; effective tput is one issue per
// many cycles on Zen 3 (FP divide is non-pipelined). Operands are kept finite
// and far from edge cases (no 0, NaN).
// =====================================================================

__attribute__((noinline)) static uint64_t bench_vdivps_ymm_lat(int iters) {
    __m256 v = _mm256_set1_ps(1.0e6f);
    __m256 c = _mm256_set1_ps(1.0001f);
    for (int i = 0; i < iters; ++i) {
        asm volatile(REP64("vdivps %[c], %[v], %[v]\n\t")
                     : [v] "+x"(v)
                     : [c] "x"(c));
    }
    sink_u64 ^= static_cast<uint64_t>(_mm256_cvtss_f32(v));
    return sink_u64;
}

__attribute__((noinline)) static uint64_t bench_vdivps_ymm_tput(int iters) {
    __m256 v0 = _mm256_set1_ps(1.0e6f), v1 = _mm256_set1_ps(2.0e6f);
    __m256 v2 = _mm256_set1_ps(3.0e6f), v3 = _mm256_set1_ps(4.0e6f);
    __m256 v4 = _mm256_set1_ps(5.0e6f), v5 = _mm256_set1_ps(6.0e6f);
    __m256 v6 = _mm256_set1_ps(7.0e6f), v7 = _mm256_set1_ps(8.0e6f);
    __m256 c = _mm256_set1_ps(1.0001f);
    for (int i = 0; i < iters; ++i) {
#define ROT(k) "vdivps %[c], %[v" #k "], %[v" #k "]\n\t"
        asm volatile(REP64_ROT(ROT)
                     : [v0] "+x"(v0), [v1] "+x"(v1), [v2] "+x"(v2), [v3] "+x"(v3),
                       [v4] "+x"(v4), [v5] "+x"(v5), [v6] "+x"(v6), [v7] "+x"(v7)
                     : [c] "x"(c));
#undef ROT
    }
    __m256 s = _mm256_add_ps(_mm256_add_ps(_mm256_add_ps(v0, v1), _mm256_add_ps(v2, v3)),
                             _mm256_add_ps(_mm256_add_ps(v4, v5), _mm256_add_ps(v6, v7)));
    sink_u64 ^= static_cast<uint64_t>(_mm256_cvtss_f32(s));
    return sink_u64;
}

__attribute__((noinline)) static uint64_t bench_vsqrtpd_ymm_lat(int iters) {
    __m256d v = _mm256_set1_pd(2.0);
    for (int i = 0; i < iters; ++i) {
        asm volatile(REP64("vsqrtpd %[v], %[v]\n\t")
                     : [v] "+x"(v));
    }
    sink_u64 ^= static_cast<uint64_t>(_mm256_cvtsd_f64(v));
    return sink_u64;
}

__attribute__((noinline)) static uint64_t bench_vsqrtpd_ymm_tput(int iters) {
    __m256d v0 = _mm256_set1_pd(2.0), v1 = _mm256_set1_pd(3.0);
    __m256d v2 = _mm256_set1_pd(4.0), v3 = _mm256_set1_pd(5.0);
    __m256d v4 = _mm256_set1_pd(6.0), v5 = _mm256_set1_pd(7.0);
    __m256d v6 = _mm256_set1_pd(8.0), v7 = _mm256_set1_pd(9.0);
    for (int i = 0; i < iters; ++i) {
#define ROT(k) "vsqrtpd %[v" #k "], %[v" #k "]\n\t"
        asm volatile(REP64_ROT(ROT)
                     : [v0] "+x"(v0), [v1] "+x"(v1), [v2] "+x"(v2), [v3] "+x"(v3),
                       [v4] "+x"(v4), [v5] "+x"(v5), [v6] "+x"(v6), [v7] "+x"(v7));
#undef ROT
    }
    __m256d s = _mm256_add_pd(_mm256_add_pd(_mm256_add_pd(v0, v1), _mm256_add_pd(v2, v3)),
                              _mm256_add_pd(_mm256_add_pd(v4, v5), _mm256_add_pd(v6, v7)));
    sink_u64 ^= static_cast<uint64_t>(_mm256_cvtsd_f64(s));
    return sink_u64;
}

// =====================================================================
// Driver.
// =====================================================================

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

enum class Kind { Lat, Tput };

struct Benchmark {
    std::string name;
    BenchFn fn;
    Kind kind;
    bool enabled;
};

static const char* kind_str(Kind k) {
    return k == Kind::Lat ? "lat" : "tput";
}

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

    std::cout << std::left << std::setw(24) << b.name
              << std::setw(6) << kind_str(b.kind)
              << std::right << std::fixed << std::setprecision(3)
              << std::setw(12) << primary
              << std::setw(12) << primary_min
              << std::setw(12) << primary_p90
              << std::setw(12) << median(tsc_adj)
              << "\n";
}

static void print_section_header(const char* title) {
    std::cout << "\n=== " << title << " ===\n";
    std::cout << std::left << std::setw(24) << "benchmark"
              << std::setw(6) << "kind"
              << std::right
              << std::setw(12) << "med ns/op"
              << std::setw(12) << "min ns/op"
              << std::setw(12) << "p90 ns/op"
              << std::setw(12) << "med tsc/op"
              << "\n";
    std::cout << std::string(78, '-') << "\n";
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

    // Warm up the empty loop and measure baseline.
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

    std::cout << "uops-style vector latency/throughput probe\n";
    std::cout << "logical_cpu_requested=" << logical_cpu
              << " pinned=" << (pinned ? "yes" : "no")
              << " current_cpu=" << sched_getcpu()
              << " ops_per_sample=" << (static_cast<uint64_t>(iters) * OPS_PER_ITER)
              << " samples=" << samples << "\n";
    std::cout << "features: invariant_tsc=" << (features.invariant_tsc ? "yes" : "no")
              << " avx2=" << (features.avx2 ? "yes" : "no")
              << " fma=" << (features.fma ? "yes" : "no")
              << " sse4.2=" << (features.sse42 ? "yes" : "no") << "\n";
    std::cout << "timing: primary_column=clock_gettime_thread_ns\n";
    std::cout << "empty_loop_median: thread_ns=" << static_cast<uint64_t>(empty_thread_median)
              << " tsc=" << static_cast<uint64_t>(empty_tsc_median)
              << " per_nominal_op_thread_ns=" << std::fixed << std::setprecision(5)
              << empty_thread_median / (static_cast<double>(iters) * OPS_PER_ITER)
              << "\n";

    // Section: 128-bit SSE
    std::vector<Benchmark> sse128 = {
        {"paddd xmm",   bench_paddd_xmm_lat,   Kind::Lat,  true},
        {"paddd xmm",   bench_paddd_xmm_tput,  Kind::Tput, true},
        {"pmulld xmm",  bench_pmulld_xmm_lat,  Kind::Lat,  true},
        {"pmulld xmm",  bench_pmulld_xmm_tput, Kind::Tput, true},
        {"pshufb xmm",  bench_pshufb_xmm_lat,  Kind::Lat,  true},
        {"pshufb xmm",  bench_pshufb_xmm_tput, Kind::Tput, true},
        {"pcmpeqb xmm", bench_pcmpeqb_xmm_lat, Kind::Lat,  true},
        {"pcmpeqb xmm", bench_pcmpeqb_xmm_tput,Kind::Tput, true},
        {"por xmm",     bench_por_xmm_lat,     Kind::Lat,  true},
        {"por xmm",     bench_por_xmm_tput,    Kind::Tput, true},
    };
    print_section_header("128-bit SSE/SSE4 integer");
    for (const Benchmark& b : sse128) {
        if (b.enabled) {
            run_benchmark(b, iters, samples, empty_tsc_median, empty_thread_median);
        }
    }

    // Section: 256-bit AVX2 integer
    std::vector<Benchmark> avx2int = {
        {"vpaddd ymm",  bench_vpaddd_ymm_lat,  Kind::Lat,  features.avx2},
        {"vpaddd ymm",  bench_vpaddd_ymm_tput, Kind::Tput, features.avx2},
        {"vpmulld ymm", bench_vpmulld_ymm_lat, Kind::Lat,  features.avx2},
        {"vpmulld ymm", bench_vpmulld_ymm_tput,Kind::Tput, features.avx2},
        {"vpshufb ymm", bench_vpshufb_ymm_lat, Kind::Lat,  features.avx2},
        {"vpshufb ymm", bench_vpshufb_ymm_tput,Kind::Tput, features.avx2},
        {"vpor ymm",    bench_vpor_ymm_lat,    Kind::Lat,  features.avx2},
        {"vpor ymm",    bench_vpor_ymm_tput,   Kind::Tput, features.avx2},
        {"vpaddq ymm",  bench_vpaddq_ymm_lat,  Kind::Lat,  features.avx2},
        {"vpaddq ymm",  bench_vpaddq_ymm_tput, Kind::Tput, features.avx2},
    };
    print_section_header("256-bit AVX2 integer");
    for (const Benchmark& b : avx2int) {
        if (b.enabled) {
            run_benchmark(b, iters, samples, empty_tsc_median, empty_thread_median);
        }
    }

    // Section: 256-bit AVX FP add/mul
    std::vector<Benchmark> avxfp = {
        {"vaddps ymm",  bench_vaddps_ymm_lat,  Kind::Lat,  features.avx2},
        {"vaddps ymm",  bench_vaddps_ymm_tput, Kind::Tput, features.avx2},
        {"vmulps ymm",  bench_vmulps_ymm_lat,  Kind::Lat,  features.avx2},
        {"vmulps ymm",  bench_vmulps_ymm_tput, Kind::Tput, features.avx2},
        {"vaddpd ymm",  bench_vaddpd_ymm_lat,  Kind::Lat,  features.avx2},
        {"vaddpd ymm",  bench_vaddpd_ymm_tput, Kind::Tput, features.avx2},
        {"vmulpd ymm",  bench_vmulpd_ymm_lat,  Kind::Lat,  features.avx2},
        {"vmulpd ymm",  bench_vmulpd_ymm_tput, Kind::Tput, features.avx2},
    };
    print_section_header("256-bit AVX FP add/mul");
    for (const Benchmark& b : avxfp) {
        if (b.enabled) {
            run_benchmark(b, iters, samples, empty_tsc_median, empty_thread_median);
        }
    }

    // Section: FMA
    std::vector<Benchmark> fma_b = {
        {"vfmadd231ps ymm",  bench_vfmadd231ps_ymm_lat,  Kind::Lat,  features.fma},
        {"vfmadd231ps ymm",  bench_vfmadd231ps_ymm_tput, Kind::Tput, features.fma},
        {"vfmadd231pd ymm",  bench_vfmadd231pd_ymm_lat,  Kind::Lat,  features.fma},
        {"vfmadd231pd ymm",  bench_vfmadd231pd_ymm_tput, Kind::Tput, features.fma},
        {"vfnmadd231ps ymm", bench_vfnmadd231ps_ymm_lat, Kind::Lat,  features.fma},
        {"vfnmadd231ps ymm", bench_vfnmadd231ps_ymm_tput,Kind::Tput, features.fma},
    };
    print_section_header("FMA (256-bit)");
    for (const Benchmark& b : fma_b) {
        if (b.enabled) {
            run_benchmark(b, iters, samples, empty_tsc_median, empty_thread_median);
        }
    }

    // Section: lane-crossing shuffles
    std::vector<Benchmark> shuf = {
        {"vpermq ymm,imm8", bench_vpermq_ymm_lat,  Kind::Lat,  features.avx2},
        {"vpermq ymm,imm8", bench_vpermq_ymm_tput, Kind::Tput, features.avx2},
        {"vpermd ymm,ymm",  bench_vpermd_ymm_lat,  Kind::Lat,  features.avx2},
        {"vpermd ymm,ymm",  bench_vpermd_ymm_tput, Kind::Tput, features.avx2},
    };
    print_section_header("Lane-crossing shuffles (256-bit)");
    for (const Benchmark& b : shuf) {
        if (b.enabled) {
            run_benchmark(b, iters, samples, empty_tsc_median, empty_thread_median);
        }
    }

    // Section: divides / sqrt
    std::vector<Benchmark> div_b = {
        {"vdivps ymm",  bench_vdivps_ymm_lat,  Kind::Lat,  features.avx2},
        {"vdivps ymm",  bench_vdivps_ymm_tput, Kind::Tput, features.avx2},
        {"vsqrtpd ymm", bench_vsqrtpd_ymm_lat, Kind::Lat,  features.avx2},
        {"vsqrtpd ymm", bench_vsqrtpd_ymm_tput,Kind::Tput, features.avx2},
    };
    print_section_header("256-bit divide / sqrt");
    for (const Benchmark& b : div_b) {
        if (b.enabled) {
            run_benchmark(b, iters, samples, empty_tsc_median, empty_thread_median);
        }
    }

    return 0;
}
