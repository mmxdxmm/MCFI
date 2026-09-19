#pragma once

#include <sys/types.h>
#include <stddef.h>
#include "pac_kit.h"

#include <sys/mman.h>
#include <unistd.h>
#include "linear_allocator.h"

#if defined(__arm64e__) && __has_feature(ptrauth_calls)
#include <ptrauth.h>
#endif

namespace features {

template <typename T> inline T arm_thumb_fix_addr(T &addr) {
#if defined(__arm__) || defined(__aarch64__)
  addr = (T)((uintptr_t)addr & ~1);
#endif
  return addr;
}

namespace apple {
template <typename T> inline T arm64e_pac_strip(T &addr) {
  return pac_strip(addr);
}

template <typename T> inline T arm64e_pac_sign(T &addr) {
  return pac_sign(addr);
}

template <typename T> inline T arm64e_pac_strip_and_sign(T &addr) {
  return pac_strip_and_sign(addr);
}
} // namespace apple

namespace android {
inline void make_memory_readable(void *address, size_t size) {
#if defined(ANDROID)
  long page_size = sysconf(_SC_PAGESIZE);
  if (page_size <= 0) {
    page_size = 4096;
  }
  auto page = (void *)ALIGN_FLOOR(address, (uintptr_t)page_size);
  if (mprotect(page, (size_t)page_size, PROT_READ | PROT_EXEC) != 0) {
    return;
  }
#endif
}
} // namespace android
} // namespace features