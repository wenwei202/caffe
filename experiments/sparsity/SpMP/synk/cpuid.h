/**
Copyright (c) 2015, Intel Corporation. All rights reserved.
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the Intel Corporation nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL INTEL CORPORATION BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

/* synk -- MIC-optimized synchronization constructs
 *
 * CPUID
 *
 * 2013.06.12   kiran.pamnany   Imported from OMPTB
 */

/*
    On KNC, when EAX==1 on input, CPUID returns the following:

        EAX[3:0]   = Stepping ID;
        EAX[7:4]   = 0001B;      // Model
        EAX[11:8]  = 1011B;      // Family
        EAX[13:12] = 00B;        // Processor type
        EAX[15:14] = 00B;        // Reserved
        EAX[19:16] = 0000B;      // Extended Model
        EAX[23:20] = 00000000B;  // Extended Family
        EAX[31:24] = 00H;        // Reserved;

        EBX[7:0]   = 00H;        // Brand Index (* Reserved if value is zero *)
        EBX[15:8]  = 8;          // CLEVICT1/CLFLUSH Line Size (x8)
        EBX[23:16] = 248;        // Maximum number of logical processors
        EBX[31:24] = Initial Apic ID;

        ECX = 00000000H;         // Feature flags

        EDX = 110193FFH;         // Feature flags
 */

#ifndef __CPUID_H
#define __CPUID_H

#include <xmmintrin.h>

#define CacheLineSize  64

#if (__MIC__)
# define cpu_pause() _mm_delay_64(100)
# define mfence()    __asm__ __volatile__ ("":::"memory")
# define sfence()    __asm__ __volatile__ ("":::"memory")
# define lfence()    __asm__ __volatile__ ("":::"memory")
#else
# define cpu_pause() _mm_pause()
# define mfence()    _mm_mfence()
# define sfence()    _mm_sfence()
# define lfence()    _mm_lfence()
#endif

#endif  /* __CPUID_H */

