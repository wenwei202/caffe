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
 * Dissemination barrier (~4800 cycles on KNC B0)
 *
 * 2013.06.12   kiran.pamnany   Release within PCL
 */

#ifndef _BARRIER_HPP
#define _BARRIER_HPP

#include "atomic.h"
#include "synk.hpp"


namespace synk
{

/* dissemination barrier; one per core */
struct CoreBarrier {
    int8_t              coreId;
    volatile uint8_t    coreSense;
    volatile uint8_t    *threadSenses;
    volatile uint8_t    *myFlags[2];
    uint8_t             **partnerFlags[2];
    uint8_t             parity;
    uint8_t             sense;
};


/* barrier container */
class Barrier : public Synk {
public:
    ~Barrier();

    // If you want to construct the singleton with options other than
    // the default one, call this function before ANY invocation of
    // getInstance
    static void initializeInstance(int numCores, int numThreadsPerCore);
    static Barrier *getInstance();
    static void deleteInstance();

    void init(int tid);
    void wait(int tid);

protected:
    Barrier(int numCores, int numThreadsPerCore);
      /* not public to be used as a singleton */

    CoreBarrier **cores;
    CoreBarrier **threadCores;
    int8_t      *coreTids;
};

}

#endif  /* _BARRIER_HPP */

