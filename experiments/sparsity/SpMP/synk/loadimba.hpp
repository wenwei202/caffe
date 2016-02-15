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
 * Barrier instrumented to report load imbalance seen (~9500 cycles on KNC B0)
 *
 * 2013.06.12   kiran.pamnany   Release within PCL
 */

#ifndef _LOADIMBA_HPP
#define _LOADIMBA_HPP

#include "atomic.h"
#include "barrier.hpp"
#include "reduce.hpp"


namespace synk
{

/* load imbalance measurement data; one per thread */
struct ThreadLoadImba {
    uint64_t            *min, *max, *tot, *cnt, *bar;
};


/* barrier container */
class LoadImba : public Barrier {
public:
    ~LoadImba();

    // If you want to construct the singleton with options other than
    // the default one, call this function before ANY invocation of
    // getInstance
    static void initializeInstance(int numCores, int numThreadsPerCore);
    static LoadImba *getInstance();
    static void deleteInstance();

    void init(int tid, int numUses = 1);
    void wait(int tid, int barNum = 0);
    void reset();
    void print(bool all = false);
    void printLoadImbalance(unsigned long long refTime);

    ReduceBarrier<uint64_t>     *red;
    ThreadLoadImba              **limba;
    int                         numUses;

protected:
    LoadImba(int numCores, int numThreadsPerCore);
      /* not public to be used as a singleton */
};

}

#endif  /* _LOADIMBA_HPP */

