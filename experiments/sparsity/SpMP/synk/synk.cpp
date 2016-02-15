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
 * Abstract base class for the various constructs.
 *
 * 2013.06.12   kiran.pamnany   Release within PCL
 */


#include <stdint.h>
#include <math.h>
#include <stdio.h>

#include "synk.hpp"


namespace synk
{

#ifdef USE_MCRT
int ceil_log2(int n)
{
   for (int i = 0; i < 32; ++i) {
     if ((n >> (i + 1)) == 0) {
       printf("n = %d, ret = %d\n", n, i);
       return i;
     }
   }
   return 0;
}
#endif

/* constructor */
Synk::Synk(int numCores_, int numThreadsPerCore_)
{
    int i, incr;

    numCores = numCores_;
    numThreadsPerCore = numThreadsPerCore_;

    numThreads = numCores * numThreadsPerCore;
#ifdef USE_MCRT
    lgNumCores = ceil_log2(numCores);
#else
    lgNumCores = (int)ceil(log2(numCores));
#endif
    if (lgNumCores == 0) lgNumCores = 1;

    nearestPow2Cores = 1;
    while (nearestPow2Cores < numCores)
        nearestPow2Cores <<= 1;

    atomic_set(&threadsWaiting, numThreads);
    initState = 0;
}


/* empty implementation for pure virtual destructor */
Synk::~Synk()
{
}

}

