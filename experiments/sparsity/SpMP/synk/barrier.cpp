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
 * Dissemination barrier (~4400 cycles on KNC B0)
 *
 * 2013.06.12   kiran.pamnany   Release within PCL
 */

#include <stdint.h>
#include <omp.h>
#include <immintrin.h>

#include "cpuid.h"
#include "barrier.hpp"

#include <new>
using namespace std;

namespace synk
{

static Barrier *instance = NULL;

void Barrier::initializeInstance(int numCores, int numThreadsPerCore)
{
    if (!instance)
    {
        if (omp_in_parallel())
        {
#pragma omp barrier
#pragma omp master
            instance = new Barrier(numCores, numThreadsPerCore);
#pragma omp barrier
            instance->init(omp_get_thread_num());
        }
        else
        {
            instance = new Barrier(numCores, numThreadsPerCore);
#pragma omp parallel
            {
                instance->init(omp_get_thread_num());
            }
        }
    }
}

Barrier *Barrier::getInstance()
{
    if (!instance)
    {
        int threadsPerCore =
#ifdef __MIC__
            4;
#else
            1;
#endif

        if (omp_in_parallel())
        {
#pragma omp barrier
#pragma omp master
            instance = new Barrier(omp_get_num_threads()/threadsPerCore, threadsPerCore);
#pragma omp barrier
            instance->init(omp_get_thread_num());
        }
        else
        {
            instance = new Barrier(omp_get_max_threads()/threadsPerCore, threadsPerCore);
#pragma omp parallel
            {
                instance->init(omp_get_thread_num());
            }
        }
    }
    return instance;
}

void Barrier::deleteInstance()
{
    delete instance;
}

/* constructor */
Barrier::Barrier(int numCores_, int numThreadsPerCore_)
        : Synk(numCores_, numThreadsPerCore_)
{
    cores = (CoreBarrier **)
        _mm_malloc(numCores * sizeof (CoreBarrier *), 64);
    if (cores == NULL) throw bad_alloc();
    threadCores = (CoreBarrier **)
        _mm_malloc(numThreads * sizeof (CoreBarrier *), 64);
    if (threadCores == NULL) throw bad_alloc();
    coreTids = (int8_t *)
        _mm_malloc(numThreads * sizeof (int8_t), 64);
    if (coreTids == NULL) throw bad_alloc();
}



/* barrier initialization, called by each thread in the team */
void Barrier::init(int tid)
{
    CoreBarrier *core;

    /* this thread's core ID and core thread ID */
    int cid = tid / numThreadsPerCore;
    int coreTid = tid % numThreadsPerCore;

    /* core thread 0 sets up */
    if (coreTid == 0) {
        core = (CoreBarrier *)_mm_malloc(sizeof (CoreBarrier), 64);
        core->coreId = cid;
        core->coreSense = 1;
        core->threadSenses = (uint8_t *)
            _mm_malloc(numThreadsPerCore * sizeof (uint8_t), 64);
        for (int i = 0;  i < numThreadsPerCore;  i++)
            core->threadSenses[i] = 1;
        for (int i = 0;  i < 2;  i++) {
            core->myFlags[i] = (uint8_t *)
                _mm_malloc(lgNumCores * CacheLineSize, 64);
            for (int j = 0;  j < lgNumCores;  j++)
                core->myFlags[i][j * CacheLineSize] = 0;
            core->partnerFlags[i] = (uint8_t **)
                _mm_malloc(lgNumCores * sizeof (uint8_t *), 64);
        }
        core->parity = 0;
        core->sense = 1;

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
            /* find dissemination partner and link */
            int partnerCid = (cid + (1 << i)) % numCores;
            for (int j = 0;  j < 2;  j++)
                core->partnerFlags[j][i] = (uint8_t *)
                    &cores[partnerCid]->myFlags[j][i * CacheLineSize];
        }
    }

    /* barrier to let initialization complete */
    if (atomic_dec_and_test(&threadsWaiting)) {
        atomic_set(&threadsWaiting, numThreads);
        initState = 2;
    } else while (initState == 1);
}



/* barrier */
void Barrier::wait(int tid)
{
    int i, di;

#if (__MIC__)
    uint8_t sendbuf[CacheLineSize] __attribute((aligned(CacheLineSize)));
    __m512d Vt;
#endif

    /* find thread's core and core thread id */
    CoreBarrier *bar = threadCores[tid];
    int8_t coreTid = coreTids[tid];

    /* signal thread arrival in core */
    bar->threadSenses[coreTid] = !bar->threadSenses[coreTid];

    /* core thread 0 syncs across cores */
    if (coreTid == 0) {

        /* wait for the core's remaining threads */
        for (i = 1;  i < numThreadsPerCore;  i++) {
            while (bar->threadSenses[i] == bar->coreSense)
                cpu_pause();
        }

        /* sync with other cores */
        if (numCores > 1) {
#if (__MIC__)
            _mm_prefetch((const char *)bar->partnerFlags[bar->parity][0],
                         _MM_HINT_ET1);

            /* on MIC, set up to use an NGO store */
            sendbuf[0] = bar->sense;
            Vt = _mm512_load_pd((void *)sendbuf);
#endif

            /* lgNumCores rounds of core-core synchronization */
            for (i = di = 0;  i < lgNumCores-1;  i++, di += CacheLineSize) {

#if (__MIC__)
                _mm_prefetch((const char *)bar->partnerFlags[bar->parity][i+1],
                             _MM_HINT_ET1);

                /* tell this round partner we've arrived */
                _mm512_storenrngo_pd((void *)bar->partnerFlags[bar->parity][i],
                                     Vt);
#else
                *bar->partnerFlags[bar->parity][i] = bar->sense;
#endif

                /* wait for this round's predecessor to say they've arrived */
                while (bar->myFlags[bar->parity][di] != bar->sense)
                    cpu_pause();
            }
#if (__MIC__)
            _mm512_storenrngo_pd((void *)bar->partnerFlags[bar->parity][i], Vt);
#else
            *bar->partnerFlags[bar->parity][i] = bar->sense;
#endif
            while (bar->myFlags[bar->parity][di] != bar->sense)
                cpu_pause();

            /* adjust sense and parity for next round */
            if (bar->parity == 1) bar->sense = !bar->sense;
            bar->parity = 1-bar->parity;
        }

        /* wake up the core's remaining threads */
        bar->coreSense = bar->threadSenses[0];
    }

    /* core's other threads just wait for cross-core sync to complete */
    else {
#if 0
        if (coreTid == 1) {
            for (i = di = 0;  i < lgNumCores;  i++, di += CacheLineSize)
                _mm_prefetch((const char *)bar->partnerFlags[bar->parity][i],
                             _MM_HINT_ET1);
        }
#endif

        while (bar->coreSense != bar->threadSenses[coreTid])
            cpu_pause();
    }
}



/* destructor */
Barrier::~Barrier()
{
    if (initState) {
        for (int i = 0;  i < numCores;  i++) {
            _mm_free((void *)cores[i]->threadSenses);
            for (int j = 0;  j < 2;  j++) {
                _mm_free((void *)cores[i]->myFlags[j]);
                _mm_free((void *)cores[i]->partnerFlags[j]);
            }
            _mm_free(cores[i]);
        }
    }

    _mm_free(coreTids);
    _mm_free(threadCores);
    _mm_free(cores);
}

}

