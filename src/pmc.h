// Minimal PMC helper: opens a perf_event_open counter and reads it via the
// rdpmc fast path. Designed for thread-pinned, self-process measurements.
//
// Requires:
//   sysctl kernel.perf_event_paranoid <= 1
//   echo 2 > /sys/bus/event_source/devices/cpu/rdpmc

#pragma once

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <linux/perf_event.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <unistd.h>
#include <x86intrin.h>

struct PmcCounter {
    int fd = -1;
    perf_event_mmap_page* page = nullptr;
    bool rdpmc_ok = false;
};

static inline long perf_event_open_syscall(struct perf_event_attr* attr,
                                           pid_t pid, int cpu, int group_fd,
                                           unsigned long flags) {
    return syscall(SYS_perf_event_open, attr, pid, cpu, group_fd, flags);
}

static inline bool pmc_open(PmcCounter& c, uint32_t type, uint64_t config) {
    perf_event_attr attr{};
    attr.type = type;
    attr.size = sizeof(attr);
    attr.config = config;
    attr.disabled = 0;
    attr.exclude_kernel = 1;
    attr.exclude_hv = 1;
    attr.pinned = 1;

    c.fd = static_cast<int>(perf_event_open_syscall(&attr, 0, -1, -1, 0));
    if (c.fd < 0) {
        std::fprintf(stderr, "perf_event_open type=%u config=0x%lx failed: %s\n",
                     type, static_cast<unsigned long>(config), std::strerror(errno));
        return false;
    }
    void* p = mmap(nullptr, sysconf(_SC_PAGESIZE), PROT_READ, MAP_SHARED, c.fd, 0);
    if (p == MAP_FAILED) {
        std::fprintf(stderr, "mmap perf fd failed: %s\n", std::strerror(errno));
        close(c.fd);
        c.fd = -1;
        return false;
    }
    c.page = static_cast<perf_event_mmap_page*>(p);
    c.rdpmc_ok = (c.page->cap_user_rdpmc != 0) && (c.page->index != 0);
    return true;
}

// Open a system-wide counter (e.g., RAPL power/energy-pkg/). Must specify CPU
// because system-wide events can't be opened with cpu=-1 from unprivileged
// callers when paranoid > -1.
static inline bool pmc_open_system(PmcCounter& c, uint32_t type, uint64_t config,
                                   int cpu) {
    perf_event_attr attr{};
    attr.type = type;
    attr.size = sizeof(attr);
    attr.config = config;
    attr.disabled = 0;
    attr.pinned = 1;

    c.fd = static_cast<int>(perf_event_open_syscall(&attr, -1, cpu, -1, 0));
    if (c.fd < 0) {
        std::fprintf(stderr, "perf_event_open SYSTEM type=%u config=0x%lx cpu=%d failed: %s\n",
                     type, static_cast<unsigned long>(config), cpu, std::strerror(errno));
        return false;
    }
    c.page = nullptr;
    c.rdpmc_ok = false;
    return true;
}

static inline void pmc_close(PmcCounter& c) {
    if (c.page) {
        munmap(c.page, sysconf(_SC_PAGESIZE));
        c.page = nullptr;
    }
    if (c.fd >= 0) {
        close(c.fd);
        c.fd = -1;
    }
}

// Fast read via rdpmc (preferred). Falls back to read() if rdpmc isn't enabled.
static inline uint64_t pmc_read(const PmcCounter& c) {
    if (c.rdpmc_ok && c.page) {
        // Standard pattern: read mmap page seqlock, get index/offset, do rdpmc
        // with idx-1, combine with offset.
        const perf_event_mmap_page* page = c.page;
        uint32_t seq, idx;
        uint64_t offset, count;
        do {
            seq = page->lock;
            __atomic_thread_fence(__ATOMIC_ACQUIRE);
            idx = page->index;
            offset = page->offset;
            if (idx == 0) {
                count = 0;
                break;
            }
            uint32_t lo, hi;
            asm volatile("rdpmc" : "=a"(lo), "=d"(hi) : "c"(idx - 1));
            count = offset + ((static_cast<uint64_t>(hi) << 32) | lo);
            // Apply pmc_width sign-extension truncation if needed:
            if (page->pmc_width != 0 && page->pmc_width < 64) {
                uint64_t mask = (1ull << page->pmc_width) - 1;
                count &= mask;
            }
            __atomic_thread_fence(__ATOMIC_ACQUIRE);
        } while (page->lock != seq);
        return count;
    }
    // Fallback: read() syscall.
    uint64_t val = 0;
    if (::read(c.fd, &val, sizeof(val)) != static_cast<ssize_t>(sizeof(val))) {
        return 0;
    }
    return val;
}

// AMD Zen 3 raw event helper. Encodes (event, umask) per AMD's perf raw format:
//   config = (umask << 8) | (event & 0xFF) | ((event >> 8) << 32)
// Most events fit in 8 bits; the >>32 handles the few extended ones.
static inline uint64_t amd_raw(uint32_t event, uint32_t umask) {
    return (static_cast<uint64_t>(umask) << 8) |
           (static_cast<uint64_t>(event & 0xFF)) |
           (static_cast<uint64_t>(event >> 8) << 32);
}

// Read a small uint from a sysfs path (e.g., /sys/bus/event_source/devices/power/type).
// Used to discover PMU types (power/, msr/, etc.) without hardcoding.
static inline bool sysfs_read_u32(const char* path, uint32_t* out) {
    FILE* f = std::fopen(path, "r");
    if (!f) return false;
    bool ok = std::fscanf(f, "%u", out) == 1;
    std::fclose(f);
    return ok;
}

// Read a perf event config string from sysfs (e.g., "event=0x02").
// Parses "event=0xNN[,umask=0xMM]" into a config value. Used for RAPL events.
static inline bool sysfs_read_perf_event(const char* path, uint64_t* config_out) {
    FILE* f = std::fopen(path, "r");
    if (!f) return false;
    char buf[256] = {0};
    if (!std::fgets(buf, sizeof(buf), f)) {
        std::fclose(f);
        return false;
    }
    std::fclose(f);
    uint64_t config = 0;
    // Look for "event=0xNN"
    char* p = std::strstr(buf, "event=");
    if (p) {
        unsigned long v = std::strtoul(p + 6, nullptr, 0);
        config |= v & 0xFFull;
    }
    p = std::strstr(buf, "umask=");
    if (p) {
        unsigned long v = std::strtoul(p + 6, nullptr, 0);
        config |= (v & 0xFFull) << 8;
    }
    *config_out = config;
    return true;
}
