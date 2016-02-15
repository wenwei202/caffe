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
 * Reducing dissemination barrier (~9500 cycles on KNC B0)
 *
 * 2013.06.12   kiran.pamnany   Release within PCL
 */

#ifndef _REDUCE_HPP
#define _REDUCE_HPP

#include <stdint.h>

#include "cpuid.h"
#include "atomic.h"
#include "synk.hpp"

#include <new>
using namespace std;

#ifndef MALLOC
#define MALLOC(type, len) (type *)_mm_malloc(sizeof(type)*(len), 64)
#endif

namespace synk
{

/* pseudo-core for reducing barrier */
template <typename T>
struct PseudoCore {
    int8_t              coreId;
    volatile uint8_t    *myFlags[2];
    uint8_t             **partnerFlags[2];
    uint8_t             parity;
    uint8_t             sense;

    volatile T          *myRvals;
    T                   **partnerRvals;
};


/* reducing dissemination barrier; one per core */
template <typename T>
struct ReduceCoreBarrier {
    int8_t              coreId;
    volatile uint8_t    coreSense;
    volatile uint8_t    *threadSenses;
    volatile uint8_t    *myFlags[2];
    uint8_t             **partnerFlags[2];
    uint8_t             parity;
    uint8_t             sense;

    volatile T          rval;
    volatile T          *threadRvals;
    volatile T          *myRvals;
    T                   **partnerRvals;

    PseudoCore<T>       *pseudoCore;
};


/* reducing barrier container */
template <typename T>
class ReduceBarrier : public Synk {
public:
    ReduceBarrier(int numCores, int numThreadsPerCore);
    ~ReduceBarrier();

    void init(int tid);
    T wait(int tid, T zeroVal, T redVal, T (*redFunc)(T, T));

protected:
    ReduceCoreBarrier<T> **cores;
    ReduceCoreBarrier<T> **threadCores;
    int8_t               *coreTids;
};



/* constructor */
template <typename T>
ReduceBarrier<T>::ReduceBarrier(int numCores_, int numThreadsPerCore_)
        : Synk(numCores_, numThreadsPerCore_)
{
    cores = MALLOC(ReduceCoreBarrier<T> *, numCores);
    if (cores == NULL) throw bad_alloc();
    threadCores = MALLOC(ReduceCoreBarrier<T> *, numThreads);
    if (threadCores == NULL) throw bad_alloc();
    coreTids = MALLOC(int8_t, numThreads);
    if (coreTids == NULL) throw bad_alloc();
}



/* barrier initialization, called by each thread in the team */
template <typename T>
void ReduceBarrier<T>::init(int tid)
{
    ReduceCoreBarrier<T> *core;
    PseudoCore<T> *pseudoCore = NULL;

    /* this thread's core ID and core thread ID */
    int cid = tid / numThreadsPerCore;
    int coreTid = tid % numThreadsPerCore;

    /* core thread 0 sets up */
    if (coreTid == 0) {
        core = MALLOC(ReduceCoreBarrier<T>, 1);
        core->coreId = cid;
        core->coreSense = 1;
        core->threadSenses = MALLOC(uint8_t, numThreadsPerCore);
        for (int i = 0;  i < numThreadsPerCore;  i++)
            core->threadSenses[i] = 1;
        for (int i = 0;  i < 2;  i++) {
            core->myFlags[i] = MALLOC(uint8_t, lgNumCores*CacheLineSize/sizeof(uint8_t));
            for (int j = 0;  j < lgNumCores;  j++)
                core->myFlags[i][j * CacheLineSize] = 0;
            core->partnerFlags[i] = MALLOC(uint8_t *, lgNumCores);
        }
        core->parity = 0;
        core->sense = 1;

        core->threadRvals = MALLOC(volatile T, numThreadsPerCore);
        core->myRvals = MALLOC(volatile T, lgNumCores);
        core->partnerRvals = MALLOC(T *, lgNumCores);

        core->pseudoCore = NULL;

        /* cores between numCores and nearestPow2Cores are pseudo-cores
           which must be handled by real cores; must this core handle one? */
        int reduceCid = cid + numCores;
        if (reduceCid < nearestPow2Cores) {
            pseudoCore = MALLOC(PseudoCore<T>, 1);
            pseudoCore->coreId = reduceCid;
            for (int i = 0;  i < 2;  i++) {
                pseudoCore->myFlags[i] = MALLOC(uint8_t, lgNumCores*CacheLineSize/sizeof(uint8_t));
                for (int j = 0;  j < lgNumCores;  j++)
                    pseudoCore->myFlags[i][j * CacheLineSize] = 0;
                pseudoCore->partnerFlags[i] = MALLOC(uint8_t *, lgNumCores);
            }
            pseudoCore->parity = 0;
            pseudoCore->sense = 1;

            pseudoCore->myRvals = MALLOC(volatile T, lgNumCores);
            pseudoCore->partnerRvals = MALLOC(T *, lgNumCores);

            core->pseudoCore = pseudoCore;
        }

        cores[cid] = core;
    }

    /* barrier to let all the allocations complete */
    if (atomic_dec_and_test(&threadsWaiting)) {
        atomic_set(&threadsWaiting, numThreads);
        initState = 1;
    } else while (initState == 0);

    /* map thread id to its core and its core thread ID */
    threadCores[tid] = cores[cid];
    coreTids[tid] = coreTid;

    /* core thread 0 finishes setup */
    if (coreTid == 0) {
        for (int i = 0;  i < lgNumCores;  i++) {
            core->myRvals[i] = (volatile T)NULL;

            /* find reduction partner and link */
            int reduceCid = (cid + (1 << i)) % nearestPow2Cores;

            /* is our reduction partner a real core? */
            if (reduceCid < numCores) {
                core->partnerRvals[i] = (T *)&cores[reduceCid]->myRvals[i];
                for (int j = 0;  j < 2;  j++)
                    core->partnerFlags[j][i] = (uint8_t *)
                        &cores[reduceCid]->myFlags[j][i * CacheLineSize];
            }

            /* no, it's a pseudo core */
            else {
                reduceCid -= numCores;
                core->partnerRvals[i] = (T *)
                    &cores[reduceCid]->pseudoCore->myRvals[i];
                for (int j = 0;  j < 2;  j++)
                    core->partnerFlags[j][i] = (uint8_t *)
                        &cores[reduceCid]->
                        pseudoCore->myFlags[j][i * CacheLineSize];
            }

            /* if this is one of the cores that represents a pseudo
               core, we have to initialize and link its fields too */
            if (pseudoCore) {
                pseudoCore->myRvals[i] = (volatile T)NULL;

                reduceCid = (pseudoCore->coreId + (1 << i)) % nearestPow2Cores;

                /* this pseudo core's reduction partner might be a real
                   core or a pseudo core */
                if (reduceCid < numCores) {
                    pseudoCore->partnerRvals[i] = (T *)
                        &cores[reduceCid]->myRvals[i];
                    for (int j = 0;  j < 2;  j++)
                        pseudoCore->partnerFlags[j][i] = (uint8_t *)
                            &cores[reduceCid]->myFlags[j][i * CacheLineSize];
                }
                else {
                    reduceCid -= numCores;
                    pseudoCore->partnerRvals[i] = (T *)
                        &cores[reduceCid]->pseudoCore->myRvals[i];
                    for (int j = 0;  j < 2;  j++)
                        pseudoCore->partnerFlags[j][i] = (uint8_t *)
                            &cores[reduceCid]->
                            pseudoCore->myFlags[j][i * CacheLineSize];
                }
            }
        }
    }

    /* barrier to let initialization complete */
    if (atomic_dec_and_test(&threadsWaiting)) {
        atomic_set(&threadsWaiting, numThreads);
        initState = 2;
    } else while (initState == 1);


}


/* reducing barrier */
template <typename T>
T ReduceBarrier<T>::wait(int tid, T zeroVal, T redVal, T (*redFunc)(T, T))
{
    int i, di;

    /* find thread's core and core thread id */
    ReduceCoreBarrier<T> *bar = threadCores[tid];
    int8_t coreTid = coreTids[tid];

    /* set thread's reduction value and signal thread arrival in core */
    bar->threadRvals[coreTid] = redVal;
    sfence();
    bar->threadSenses[coreTid] = !bar->threadSenses[coreTid];

    /* core thread 0 syncs across cores */
    if (coreTid == 0) {
        for (i = 1;  i < numThreadsPerCore;  i++) {
            while (bar->threadSenses[i] == bar->coreSense)
                cpu_pause();

            /* reduction operation */
            redVal = redFunc(redVal, (T)bar->threadRvals[i]);
        }

        /* sync across cores: if there's only this thread in the core and
           this core also represents a pseudo core, the thread has to
           handle both */
        if (numThreadsPerCore == 1  &&  bar->pseudoCore) {
            for (i = di = 0;  i < lgNumCores;  i++, di += CacheLineSize) {
                *bar->partnerRvals[i] = redVal;
                *bar->pseudoCore->partnerRvals[i] = zeroVal;

                sfence();

                *bar->partnerFlags[bar->parity][i] = bar->sense;
                *bar->pseudoCore->partnerFlags[bar->pseudoCore->parity][i]
                    = bar->pseudoCore->sense;

                while (bar->myFlags[bar->parity][di] != bar->sense)
                    cpu_pause();
                while (bar->pseudoCore->myFlags[bar->pseudoCore->parity][di]
                       != bar->pseudoCore->sense)
                    cpu_pause();

                /* reduction operations */
                redVal = redFunc(redVal, (T)bar->myRvals[i]);
                zeroVal = redFunc(zeroVal, (T)bar->pseudoCore->myRvals[i]);
            }

            if (bar->parity == 1) bar->sense = !bar->sense;
            bar->parity = 1 - bar->parity;
            if (bar->pseudoCore->parity == 1)
                bar->pseudoCore->sense = !bar->pseudoCore->sense;
            bar->pseudoCore->parity = 1 - bar->pseudoCore->parity;
        }

        /* no pseudo core, only handle self */
        else {
            for (i = di = 0;  i < lgNumCores;  i++, di += CacheLineSize) {
                *bar->partnerRvals[i] = redVal;
                sfence();
                *bar->partnerFlags[bar->parity][i] = bar->sense;
                while (bar->myFlags[bar->parity][di] != bar->sense)
                    cpu_pause();

                /* reduction operations */
                redVal = redFunc(redVal, (T)bar->myRvals[i]);
            }

            if (bar->parity == 1) bar->sense = !bar->sense;
            bar->parity = 1 - bar->parity;
        }

        /* wake up the waiting slave threads in this core */
        bar->rval = redVal;
        sfence();
        bar->coreSense = bar->threadSenses[0];
    }

    /* we're another thread in this core; if this core represents a
       pseudo core, handle it */
    else {
        if (coreTid == 1  &&  bar->pseudoCore) {
            for (i = di = 0;  i < lgNumCores;  i++, di += CacheLineSize) {
                *bar->pseudoCore->partnerRvals[i] = zeroVal;
                sfence();
                *bar->pseudoCore->partnerFlags[bar->pseudoCore->parity][i]
                    = bar->pseudoCore->sense;
                while (bar->pseudoCore->myFlags[bar->pseudoCore->parity][di]
                       != bar->pseudoCore->sense)
                    cpu_pause();

                /* reduction operations */
                zeroVal = redFunc(zeroVal, (T)bar->pseudoCore->myRvals[i]);
            }
            if (bar->pseudoCore->parity == 1)
                bar->pseudoCore->sense = !bar->pseudoCore->sense;
            bar->pseudoCore->parity = 1 - bar->pseudoCore->parity;
        }

        /* wait for cross-core sync to be complete */
        while (bar->coreSense != bar->threadSenses[coreTid])
            cpu_pause();
    }

    return (T)bar->rval;
}



/* destructor */
template <typename T>
ReduceBarrier<T>::~ReduceBarrier()
{
    if (initState) {
        for (int i = 0;  i < numCores;  i++) {
            if (cores[i]->pseudoCore) {
                _mm_free((void *)cores[i]->pseudoCore->partnerRvals);
                _mm_free((void *)cores[i]->pseudoCore->myRvals);
                for (int j = 0;  j < 2;  j++) {
                    _mm_free((void *)cores[i]->pseudoCore->partnerFlags[j]);
                    _mm_free((void *)cores[i]->pseudoCore->myFlags[j]);
                }
                _mm_free((void *)cores[i]->pseudoCore);
            }
            _mm_free((void *)cores[i]->partnerRvals);
            _mm_free((void *)cores[i]->myRvals);
            _mm_free((void *)cores[i]->threadRvals);
            _mm_free((void *)cores[i]->threadSenses);
            for (int j = 0;  j < 2;  j++) {
                _mm_free((void *)cores[i]->partnerFlags[j]);
                _mm_free((void *)cores[i]->myFlags[j]);
            }
            _mm_free(cores[i]);
        }
    }

    _mm_free(coreTids);
    _mm_free(threadCores);
    _mm_free(cores);
}

}

#endif  /* _REDUCE_HPP */

