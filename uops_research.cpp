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
#include <sstream>
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

static constexpr int LAT_OPS_PER_ITER = 64;
static constexpr int TP_CHAINS = 8;
static constexpr int TP_OPS_PER_ITER = 64 * TP_CHAINS;

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

struct CpuIdInfo {
    std::string brand;
    uint32_t family = 0;
    uint32_t model = 0;
    uint32_t stepping = 0;
    uint32_t native_model_id = 0;
    uint32_t core_type = 0;
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

static CpuIdInfo get_cpuid_info() {
    CpuIdInfo info;
    uint32_t eax = 0, ebx = 0, ecx = 0, edx = 0;
    if (cpuid_leaf(1, 0, eax, ebx, ecx, edx)) {
        uint32_t base_family = (eax >> 8) & 0xf;
        uint32_t base_model = (eax >> 4) & 0xf;
        uint32_t ext_family = (eax >> 20) & 0xff;
        uint32_t ext_model = (eax >> 16) & 0xf;
        info.stepping = eax & 0xf;
        info.family = base_family == 0xf ? base_family + ext_family : base_family;
        info.model = base_family == 0x6 || base_family == 0xf
                         ? base_model + (ext_model << 4)
                         : base_model;
    }

    uint32_t max_ext = __get_cpuid_max(0x80000000u, nullptr);
    if (max_ext >= 0x80000004u) {
        char brand[49] = {};
        uint32_t* out = reinterpret_cast<uint32_t*>(brand);
        for (uint32_t leaf = 0x80000002u; leaf <= 0x80000004u; ++leaf) {
            __cpuid(leaf, out[0], out[1], out[2], out[3]);
            out += 4;
        }
        info.brand = brand;
        while (!info.brand.empty() && info.brand[0] == ' ') {
            info.brand.erase(info.brand.begin());
        }
    }
    if (cpuid_leaf(0x1au, 0, eax, ebx, ecx, edx)) {
        info.native_model_id = eax & 0xffffffu;
        info.core_type = (eax >> 24) & 0xffu;
    }
    return info;
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
        ep->ContextRecord->Rip += 2;
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

__attribute__((noinline)) static uint64_t lat_add_imm(int iters) {
    uint64_t x = sink_u64 | 1ull;
    for (int i = 0; i < iters; ++i) {
        asm volatile(REP64("addq $1, %[x]\n\t") : [x] "+r"(x) : : "cc");
    }
    sink_u64 = x;
    return x;
}

__attribute__((noinline)) static uint64_t lat_add_zero(int iters) {
    uint64_t x = sink_u64 | 1ull;
    for (int i = 0; i < iters; ++i) {
        asm volatile(REP64("addq $0, %[x]\n\t") : [x] "+r"(x) : : "cc");
    }
    sink_u64 = x;
    return x;
}

__attribute__((noinline)) static uint64_t lat_add_reg(int iters) {
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

__attribute__((noinline)) static uint64_t lat_imul_imm(int iters) {
    uint64_t x = sink_u64 | 1ull;
    for (int i = 0; i < iters; ++i) {
        asm volatile(REP64("imulq $3, %[x], %[x]\n\t") : [x] "+r"(x) : : "cc");
    }
    sink_u64 = x;
    return x;
}

__attribute__((noinline)) static uint64_t lat_shl_1(int iters) {
    uint64_t x = sink_u64 | 1ull;
    for (int i = 0; i < iters; ++i) {
        asm volatile(REP64("shlq $1, %[x]\n\t") : [x] "+r"(x) : : "cc");
    }
    sink_u64 = x;
    return x;
}

__attribute__((noinline)) static uint64_t lat_ror_7(int iters) {
    uint64_t x = sink_u64 | 1ull;
    for (int i = 0; i < iters; ++i) {
        asm volatile(REP64("rorq $7, %[x]\n\t") : [x] "+r"(x) : : "cc");
    }
    sink_u64 = x;
    return x;
}

__attribute__((noinline)) static uint64_t lat_popcnt(int iters) {
    uint64_t x = sink_u64 | 0xf0f0f0f0f0f0f0f1ull;
    for (int i = 0; i < iters; ++i) {
        asm volatile(REP64("popcntq %[x], %[x]\n\t") : [x] "+r"(x) : : "cc");
    }
    sink_u64 = x;
    return x;
}

__attribute__((noinline)) static uint64_t lat_lzcnt(int iters) {
    uint64_t x = sink_u64 | 0x0102030405060708ull;
    for (int i = 0; i < iters; ++i) {
        asm volatile(REP64("lzcntq %[x], %[x]\n\t") : [x] "+r"(x) : : "cc");
    }
    sink_u64 = x;
    return x;
}

__attribute__((noinline)) static uint64_t lat_crc32(int iters) {
    uint64_t x = sink_u64 | 0x123456789abcdef1ull;
    for (int i = 0; i < iters; ++i) {
        asm volatile(REP64("crc32q %[x], %[x]\n\t") : [x] "+r"(x) : : "cc");
    }
    sink_u64 = x;
    return x;
}

__attribute__((noinline)) static uint64_t lat_addsd(int iters) {
    double v = sink_f64;
    const double c = 1.0;
    for (int i = 0; i < iters; ++i) {
        asm volatile(REP64("addsd %[c], %[v]\n\t") : [v] "+&x"(v) : [c] "x"(c));
    }
    sink_f64 = v;
    return static_cast<uint64_t>(v);
}

__attribute__((noinline)) static uint64_t lat_mulsd(int iters) {
    double v = sink_f64;
    const double c = 1.0000000001;
    for (int i = 0; i < iters; ++i) {
        asm volatile(REP64("mulsd %[c], %[v]\n\t") : [v] "+&x"(v) : [c] "x"(c));
    }
    sink_f64 = v;
    return static_cast<uint64_t>(v);
}

__attribute__((noinline)) static uint64_t lat_divsd(int iters) {
    double v = sink_f64 + 1000.0;
    const double c = 1.0000000001;
    for (int i = 0; i < iters; ++i) {
        asm volatile(REP64("divsd %[c], %[v]\n\t") : [v] "+&x"(v) : [c] "x"(c));
    }
    sink_f64 = v;
    return static_cast<uint64_t>(v);
}

__attribute__((noinline)) static uint64_t lat_sqrtsd(int iters) {
    double v = sink_f64 + 123.0;
    for (int i = 0; i < iters; ++i) {
        asm volatile(REP64("sqrtsd %[v], %[v]\n\t") : [v] "+x"(v));
    }
    sink_f64 = v;
    return static_cast<uint64_t>(v);
}

#define TP_INT_BODY(op) \
    op("%[k]", "%[a]") op("%[k]", "%[b]") op("%[k]", "%[c]") op("%[k]", "%[d]") \
    op("%[k]", "%[e]") op("%[k]", "%[f]") op("%[k]", "%[g]") op("%[k]", "%[h]")

#define OP_ADD_IMM(src, dst) "addq $1, " dst "\n\t"
#define OP_ADD_ZERO(src, dst) "addq $0, " dst "\n\t"
#define OP_ADD_REG(src, dst) "addq " src ", " dst "\n\t"
#define OP_IMUL_IMM(src, dst) "imulq $3, " dst ", " dst "\n\t"
#define OP_SHL_1(src, dst) "shlq $1, " dst "\n\t"
#define OP_ROR_7(src, dst) "rorq $7, " dst "\n\t"
#define OP_POPCNT(src, dst) "popcntq " dst ", " dst "\n\t"
#define OP_LZCNT(src, dst) "lzcntq " dst ", " dst "\n\t"
#define OP_CRC32(src, dst) "crc32q " dst ", " dst "\n\t"

#define MAKE_TP_INT_FN(fn_name, op_macro) \
__attribute__((noinline)) static uint64_t fn_name(int iters) { \
    uint64_t a = sink_u64 + 1, b = sink_u64 + 3, c = sink_u64 + 5, d = sink_u64 + 7; \
    uint64_t e = sink_u64 + 11, f = sink_u64 + 13, g = sink_u64 + 17, h = sink_u64 + 19; \
    const uint64_t k = 0x9e3779b97f4a7c15ull; \
    for (int i = 0; i < iters; ++i) { \
        asm volatile(REP64(TP_INT_BODY(op_macro)) \
                     : [a] "+&r"(a), [b] "+&r"(b), [c] "+&r"(c), [d] "+&r"(d), \
                       [e] "+&r"(e), [f] "+&r"(f), [g] "+&r"(g), [h] "+&r"(h) \
                     : [k] "r"(k) \
                     : "cc"); \
    } \
    sink_u64 = a ^ b ^ c ^ d ^ e ^ f ^ g ^ h; \
    return sink_u64; \
}

MAKE_TP_INT_FN(tp_add_imm, OP_ADD_IMM)
MAKE_TP_INT_FN(tp_add_zero, OP_ADD_ZERO)
MAKE_TP_INT_FN(tp_add_reg, OP_ADD_REG)
MAKE_TP_INT_FN(tp_imul_imm, OP_IMUL_IMM)
MAKE_TP_INT_FN(tp_shl_1, OP_SHL_1)
MAKE_TP_INT_FN(tp_ror_7, OP_ROR_7)
MAKE_TP_INT_FN(tp_popcnt, OP_POPCNT)
MAKE_TP_INT_FN(tp_lzcnt, OP_LZCNT)
MAKE_TP_INT_FN(tp_crc32, OP_CRC32)

#define TP_FP_BODY(op) \
    op("%[x0]") op("%[x1]") op("%[x2]") op("%[x3]") \
    op("%[x4]") op("%[x5]") op("%[x6]") op("%[x7]")

#define OP_ADDSD(dst) "addsd %[k], " dst "\n\t"
#define OP_MULSD(dst) "mulsd %[k], " dst "\n\t"
#define OP_DIVSD(dst) "divsd %[k], " dst "\n\t"
#define OP_SQRTSD(dst) "sqrtsd " dst ", " dst "\n\t"

#define MAKE_TP_FP_FN(fn_name, op_macro, k_value) \
__attribute__((noinline)) static uint64_t fn_name(int iters) { \
    double x0 = sink_f64 + 1.0, x1 = sink_f64 + 2.0, x2 = sink_f64 + 3.0, x3 = sink_f64 + 4.0; \
    double x4 = sink_f64 + 5.0, x5 = sink_f64 + 6.0, x6 = sink_f64 + 7.0, x7 = sink_f64 + 8.0; \
    const double k = k_value; \
    for (int i = 0; i < iters; ++i) { \
        asm volatile(REP64(TP_FP_BODY(op_macro)) \
                     : [x0] "+&x"(x0), [x1] "+&x"(x1), [x2] "+&x"(x2), [x3] "+&x"(x3), \
                       [x4] "+&x"(x4), [x5] "+&x"(x5), [x6] "+&x"(x6), [x7] "+&x"(x7) \
                     : [k] "x"(k)); \
    } \
    sink_f64 = x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7; \
    return static_cast<uint64_t>(sink_f64); \
}

MAKE_TP_FP_FN(tp_addsd, OP_ADDSD, 1.0)
MAKE_TP_FP_FN(tp_mulsd, OP_MULSD, 1.0000000001)
MAKE_TP_FP_FN(tp_divsd, OP_DIVSD, 1.0000000001)
MAKE_TP_FP_FN(tp_sqrtsd, OP_SQRTSD, 0.0)

__attribute__((noinline)) static uint64_t lat_l1_pointer_chase(int iters) {
    uintptr_t p = pointer_ring.empty() ? 0 : reinterpret_cast<uintptr_t>(&pointer_ring[0]);
    for (int i = 0; i < iters; ++i) {
        asm volatile(REP64("movq (%[p]), %[p]\n\t") : [p] "+r"(p) : : "memory");
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
    uint64_t pc0 = rdpmc_fixed_core_available ? read_rdpmc_ordered(RDPMC_FIXED_CORE_CYCLES) : 0;
    uint64_t t0 = read_tsc_begin();
    fn(iters);
    uint64_t t1 = read_tsc_end();
    uint64_t pc1 = rdpmc_fixed_core_available ? read_rdpmc_ordered(RDPMC_FIXED_CORE_CYCLES) : 0;
    uint64_t tc1 = read_thread_cycles();
    return Timing{t1 - t0, tc1 - tc0, pc1 - pc0};
}

static double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    size_t n = values.size();
    if (n == 0) return 0.0;
    return (n & 1u) ? values[n / 2] : (values[n / 2 - 1] + values[n / 2]) / 2.0;
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

struct Benchmark {
    std::string kind;
    std::string name;
    BenchFn fn;
    int ops_per_iter;
    bool enabled;
};

static std::string csv_quote(const std::string& value) {
    std::string out = "\"";
    for (char c : value) {
        if (c == '"') out += "\"\"";
        else out += c;
    }
    out += "\"";
    return out;
}

static void run_benchmark(const Benchmark& b, int iters, int samples,
                          double empty_tsc_median, double empty_thread_median,
                          double empty_pmc_median) {
    const double ops = static_cast<double>(iters) * b.ops_per_iter;
    for (int i = 0; i < 8; ++i) measure_once(b.fn, iters);

    std::vector<double> primary;
    std::vector<double> thread_adj;
    std::vector<double> tsc_adj;
    primary.reserve(samples);
    thread_adj.reserve(samples);
    tsc_adj.reserve(samples);

    for (int i = 0; i < samples; ++i) {
        Timing t = measure_once(b.fn, iters);
        double tc = (static_cast<double>(t.thread_cycles) - empty_thread_median) / ops;
        double ts = (static_cast<double>(t.tsc) - empty_tsc_median) / ops;
        double pr = tc;
        if (rdpmc_fixed_core_available) {
            pr = (static_cast<double>(t.pmc_core_cycles) - empty_pmc_median) / ops;
        }
        primary.push_back(pr);
        thread_adj.push_back(tc);
        tsc_adj.push_back(ts);
    }

    std::cout << csv_quote(b.kind) << ","
              << csv_quote(b.name) << ","
              << b.ops_per_iter << ","
              << std::fixed << std::setprecision(6)
              << median(primary) << ","
              << *std::min_element(primary.begin(), primary.end()) << ","
              << percentile(primary, 0.90) << ","
              << median(thread_adj) << ","
              << median(tsc_adj) << "\n";
}

int main(int argc, char** argv) {
    unsigned logical_cpu = 16;
    int iters = 100000;
    int samples = 31;
    if (argc > 1) logical_cpu = static_cast<unsigned>(std::strtoul(argv[1], nullptr, 10));
    if (argc > 2) iters = std::max(1, std::atoi(argv[2]));
    if (argc > 3) samples = std::max(3, std::atoi(argv[3]));

    raise_priority();
    bool pinned = pin_to_logical_cpu(logical_cpu);
    Sleep(10);
    rdpmc_fixed_core_available = detect_rdpmc_fixed_core();

    CpuFeatures features = get_cpu_features();
    CpuIdInfo cpu = get_cpuid_info();
    init_pointer_ring(4096);

    std::vector<Benchmark> benchmarks = {
        {"latency", "ADD_R64_I32 dest->dest", lat_add_imm, LAT_OPS_PER_ITER, true},
        {"latency", "ADD_R64_0 dest->dest", lat_add_zero, LAT_OPS_PER_ITER, true},
        {"latency", "ADD_R64_R64 dest->dest", lat_add_reg, LAT_OPS_PER_ITER, true},
        {"latency", "IMUL_R64_R64_I32 dest->dest", lat_imul_imm, LAT_OPS_PER_ITER, true},
        {"latency", "SHL_R64_1 dest->dest", lat_shl_1, LAT_OPS_PER_ITER, true},
        {"latency", "ROR_R64_7 dest->dest", lat_ror_7, LAT_OPS_PER_ITER, true},
        {"latency", "POPCNT_R64_R64 src->dest", lat_popcnt, LAT_OPS_PER_ITER, features.popcnt},
        {"latency", "LZCNT_R64_R64 src->dest", lat_lzcnt, LAT_OPS_PER_ITER, features.lzcnt},
        {"latency", "CRC32_R64_R64 src/dest chain", lat_crc32, LAT_OPS_PER_ITER, features.sse42},
        {"latency", "ADDSD_XMM_XMM dest->dest", lat_addsd, LAT_OPS_PER_ITER, true},
        {"latency", "MULSD_XMM_XMM dest->dest", lat_mulsd, LAT_OPS_PER_ITER, true},
        {"latency", "DIVSD_XMM_XMM dest->dest", lat_divsd, LAT_OPS_PER_ITER, true},
        {"latency", "SQRTSD_XMM_XMM src->dest", lat_sqrtsd, LAT_OPS_PER_ITER, true},
        {"latency", "MOV_R64_M64 pointer chase", lat_l1_pointer_chase, LAT_OPS_PER_ITER, true},

        {"throughput", "ADD_R64_I32 independent", tp_add_imm, TP_OPS_PER_ITER, true},
        {"throughput", "ADD_R64_0 independent", tp_add_zero, TP_OPS_PER_ITER, true},
        {"throughput", "ADD_R64_R64 independent", tp_add_reg, TP_OPS_PER_ITER, true},
        {"throughput", "IMUL_R64_R64_I32 independent", tp_imul_imm, TP_OPS_PER_ITER, true},
        {"throughput", "SHL_R64_1 independent", tp_shl_1, TP_OPS_PER_ITER, true},
        {"throughput", "ROR_R64_7 independent", tp_ror_7, TP_OPS_PER_ITER, true},
        {"throughput", "POPCNT_R64_R64 independent", tp_popcnt, TP_OPS_PER_ITER, features.popcnt},
        {"throughput", "LZCNT_R64_R64 independent", tp_lzcnt, TP_OPS_PER_ITER, features.lzcnt},
        {"throughput", "CRC32_R64_R64 independent", tp_crc32, TP_OPS_PER_ITER, features.sse42},
        {"throughput", "ADDSD_XMM_XMM independent", tp_addsd, TP_OPS_PER_ITER, true},
        {"throughput", "MULSD_XMM_XMM independent", tp_mulsd, TP_OPS_PER_ITER, true},
        {"throughput", "DIVSD_XMM_XMM independent", tp_divsd, TP_OPS_PER_ITER, true},
        {"throughput", "SQRTSD_XMM_XMM independent", tp_sqrtsd, TP_OPS_PER_ITER, true},
    };

    for (int i = 0; i < 16; ++i) measure_once(bench_empty, iters);
    std::vector<double> empty_tsc;
    std::vector<double> empty_thread;
    std::vector<double> empty_pmc;
    for (int i = 0; i < samples; ++i) {
        Timing t = measure_once(bench_empty, iters);
        empty_tsc.push_back(static_cast<double>(t.tsc));
        empty_thread.push_back(static_cast<double>(t.thread_cycles));
        if (rdpmc_fixed_core_available) empty_pmc.push_back(static_cast<double>(t.pmc_core_cycles));
    }
    double empty_tsc_median = median(empty_tsc);
    double empty_thread_median = median(empty_thread);
    double empty_pmc_median = rdpmc_fixed_core_available ? median(empty_pmc) : 0.0;

    std::cerr << "cpu_brand=" << cpu.brand << "\n";
    std::cerr << "family=" << cpu.family << " model=" << cpu.model
              << " stepping=" << cpu.stepping
              << " native_model_id=" << cpu.native_model_id
              << " core_type=0x" << std::hex << cpu.core_type << std::dec;
    if (cpu.core_type == 0x40) {
        std::cerr << " (Intel Core/P-core)";
    } else if (cpu.core_type == 0x20) {
        std::cerr << " (Intel Atom/E-core)";
    }
    std::cerr << "\n";
    std::cerr << "logical_cpu_requested=" << logical_cpu
              << " pinned=" << (pinned ? "yes" : "no")
              << " current_cpu=" << GetCurrentProcessorNumber() << "\n";
    std::cerr << "features invariant_tsc=" << (features.invariant_tsc ? "yes" : "no")
              << " popcnt=" << (features.popcnt ? "yes" : "no")
              << " lzcnt=" << (features.lzcnt ? "yes" : "no")
              << " sse4.2=" << (features.sse42 ? "yes" : "no") << "\n";
    std::cerr << "rdpmc_fixed_core_cycles=" << (rdpmc_fixed_core_available ? "yes" : "no")
              << " primary=" << (rdpmc_fixed_core_available ? "rdpmc_core_cycles" : "QueryThreadCycleTime")
              << "\n";
    std::cerr << "empty_median_thread=" << static_cast<uint64_t>(empty_thread_median)
              << " empty_median_tsc=" << static_cast<uint64_t>(empty_tsc_median)
              << " empty_median_pmc=" << static_cast<uint64_t>(empty_pmc_median) << "\n";

    std::cout << "kind,benchmark,ops_per_iter,median_primary,min_primary,p90_primary,median_thread,median_tsc\n";
    for (const Benchmark& b : benchmarks) {
        if (b.enabled) {
            run_benchmark(b, iters, samples, empty_tsc_median,
                          empty_thread_median, empty_pmc_median);
        }
    }

    return 0;
}
