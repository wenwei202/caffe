#pragma once

// From Alex Heinecke

#define SSC_MARK(mark_value)                  \
    {__asm  mov ebx, mark_value           \
    __asm  _emit 0x64                     \
    __asm  _emit 0x67                     \
    __asm  _emit 0x90}

#define SSC_MARK_START_PERFORMANCE      SSC_MARK(111) // 6f
#define SSC_MARK_STOP_PERFORMANCE       SSC_MARK(222) // de
#define SSC_MARK_INITIALIZATION         SSC_MARK(333) // 14d
#define SSC_MARK_STOP_SIMULATION        SSC_MARK(444) // 1bc
#define SSC_MARK_CLEAR_STATS            SSC_MARK(555)

#ifdef __cplusplus
extern "C" {
#endif

__attribute__((noinline)) static void ssc_start_performance() {
  SSC_MARK_START_PERFORMANCE
} 

__attribute__((noinline)) static void ssc_stop_performance() {
  SSC_MARK_STOP_PERFORMANCE
} 

__attribute__((noinline)) static void ssc_initialization() { // 14d
  SSC_MARK_INITIALIZATION
} 

__attribute__((noinline)) static void ssc_stop_simulation() { // 1bc
  SSC_MARK_STOP_SIMULATION
} 

__attribute__((noinline)) static void ssc_clear_stats() {
  SSC_MARK_CLEAR_STATS
}

#ifdef __cplusplus
}
#endif
