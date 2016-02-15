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

#include <stdint.h>
#include <omp.h>
#include <stdio.h>

#include "cpuid.h"
#include "loadimba.hpp"

#include <new>
#include <algorithm>
using namespace std;


/* reduction helper */
static uint64_t ull_max(uint64_t a, uint64_t b)
{
    return std::max(a, b);
}



namespace synk
{

static LoadImba *instance = NULL;

void LoadImba::initializeInstance(int numCores, int numThreadsPerCore)
{
    if (!instance)
    {
        if (omp_in_parallel())
        {
#pragma omp barrier
#pragma omp master
            instance = new LoadImba(numCores, numThreadsPerCore);
#pragma omp barrier
            instance->init(omp_get_thread_num());
        }
        else
        {
            instance = new LoadImba(numCores, numThreadsPerCore);
#pragma omp parallel
            {
                instance->init(omp_get_thread_num());
            }
        }
    }
}

LoadImba *LoadImba::getInstance()
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
            instance = new LoadImba(omp_get_num_threads()/threadsPerCore, threadsPerCore);
#pragma omp barrier
            instance->init(omp_get_thread_num());
        }
        else
        {
            instance = new LoadImba(omp_get_max_threads()/threadsPerCore, threadsPerCore);
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
LoadImba::LoadImba(int numCores_, int numThreadsPerCore_)
        : Barrier(numCores_, numThreadsPerCore_)
{
    red = new ReduceBarrier<uint64_t>(numCores_, numThreadsPerCore_);
    limba = (ThreadLoadImba **)
        _mm_malloc(numThreads * sizeof (ThreadLoadImba *), 64);
    if (limba == NULL) throw bad_alloc();
}



/* initialization, called by each thread in the team */
void LoadImba::init(int tid, int numUses_)
{
    ThreadLoadImba *lim;

    Barrier::init(tid);
    red->init(tid);

    numUses = numUses_;
    lim = (ThreadLoadImba *)_mm_malloc(sizeof (ThreadLoadImba), 64);
    lim->min = (uint64_t *)_mm_malloc(numUses * sizeof (uint64_t), 64);
    lim->max = (uint64_t *)_mm_malloc(numUses * sizeof (uint64_t), 64);
    lim->tot = (uint64_t *)_mm_malloc(numUses * sizeof (uint64_t), 64);
    lim->cnt = (uint64_t *)_mm_malloc(numUses * sizeof (uint64_t), 64);
    lim->bar = (uint64_t *)_mm_malloc(numUses * sizeof (uint64_t), 64);
    for (int i = 0;  i < numUses;  i++) {
        lim->min[i] = 18446744073709551615ULL;
        lim->max[i] = 0;
        lim->tot[i] = 0;
        lim->cnt[i] = 0;
        lim->bar[i] = 0;
    }
    limba[tid] = lim;

    /* barrier to let all the allocations complete */
    if (atomic_dec_and_test(&threadsWaiting)) {
        atomic_set(&threadsWaiting, numThreads);
        initState = 1;
    } else while (initState == 0);
}



/* instrumented barrier to measure load imbalance */
void LoadImba::wait(int tid, int barNum)
{
    uint64_t czero = 0;
    uint64_t cstart = _rdtsc();
    uint64_t clast = red->wait(tid, czero, cstart, ull_max);
    uint64_t cout = _rdtsc();
    uint64_t cycles = clast - cstart;

    ThreadLoadImba *lim = limba[tid];
    lim->min[barNum] = std::min(lim->min[barNum], cycles);
    lim->max[barNum] = std::max(lim->max[barNum], cycles);
    lim->tot[barNum] += cycles;
    lim->cnt[barNum]++;
    lim->bar[barNum] += (cout - clast);
}



/* reset gathered load imbalance information */
void LoadImba::reset()
{
    for (int j = 0;  j < numUses; j++) {
        for (int i = 0;  i < numThreads;  i++) {
            limba[i]->min[j] = 18446744073709551615ULL;
            limba[i]->max[j] = 0;
            limba[i]->tot[j] = 0;
            limba[i]->cnt[j] = 0;
            limba[i]->bar[j] = 0;
        }
    }
}



/* dump gathered load imbalance information; called from a single thread */
void LoadImba::print(bool all)
{
    uint64_t min, max, tot, cnt, bar;

    if (all) {
        printf("\n-------- Load imbalance details (per thread) --------\n");
        printf("Barrier #, tid, min, max, total, barrier total, #iterations, "
               "avg/iteration, barrier avg/iteration\n");
        for (int j = 0;  j < numUses;  j++) {
            for (int i = 0;  i < numThreads;  i++) {
                printf("%d, %d, %llu, %llu, %llu, %llu, %llu, %llu, %llu\n",
                       j, i, limba[i]->min[j], limba[i]->max[j],
                       limba[i]->tot[j], limba[i]->bar[j], limba[i]->cnt[j],
                       limba[i]->cnt[j] == 0 ? 0
                         : limba[i]->tot[j]/limba[i]->cnt[j],
                       limba[i]->cnt[j] == 0 ? 0
                         : limba[i]->bar[j]/limba[i]->cnt[j]);
            }
        }
        printf("--------------------\n");
    }

    printf("\n-------- Load imbalance summary (across all threads) --------\n");
    printf("Barrier #, #threads, min, max, total, barrier total, #iterations, "
           "avg/iteration, barrier avg/iteration\n");
    for (int j = 0;  j < numUses;  j++) {
        max = tot = bar = 0;
        min = 18446744073709551615ULL;
        cnt = limba[0]->cnt[j];
        for (int i = 0;  i < numThreads;  i++) {
            min = std::min(min, limba[i]->min[j]);
            max = std::max(max, limba[i]->max[j]);
            tot += limba[i]->tot[j];
            bar += limba[i]->bar[j];
        }
        tot /= numThreads;
        bar /= numThreads;
        printf("%d, %d, %llu, %llu, %llu, %llu, %llu, %llu, %llu\n",
               j, numThreads, min, max, tot, bar, cnt,
               cnt == 0 ? 0 : tot/cnt, cnt == 0 ? 0 : bar/cnt);
    }
    printf("--------------------\n");
}


void LoadImba::printLoadImbalance(unsigned long long refTime) {
  uint64_t tot = 0, bar = 0;
  //uint64_t sumOfMinMaxDiff = 0;
  for (int j = 0; j < numUses; ++j) {
    //uint64_t min = 18446744073709551615ULL, max = 0;
    for (int i = 0; i < numThreads; ++i) {
      //min = std::min(min, limba[i]->min[j]);
      //max = std::max(max, limba[i]->max[j]);
      tot += limba[i]->tot[j];
      bar += limba[i]->bar[j];
    }
    //sumOfMinMaxDiff += (max - min);
  }

  printf(
    "Load imbalance = %f\n",
    (double)(tot - bar)/numThreads/refTime);
}


/* destructor */
LoadImba::~LoadImba()
{
    delete red;
    for (int i = 0;  i < numThreads;  i++) {
        _mm_free(limba[i]->min);
        _mm_free(limba[i]->max);
        _mm_free(limba[i]->tot);
        _mm_free(limba[i]->cnt);
        _mm_free(limba[i]->bar);
    }
    _mm_free(limba);
}

}

