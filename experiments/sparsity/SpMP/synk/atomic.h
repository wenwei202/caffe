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
 * Wrappers for atomic primitives.
 *
 * 2013.06.13   kiran.pamnany   Release within PCL
 */

#ifndef _ATOMIC_H
#define _ATOMIC_H


#include <stddef.h>

typedef struct {
    volatile size_t value;
} atomic_t;


#define atomic_read(v)            \
    ((v)->value)

#define atomic_set(v,i)           \
    (((v)->value) = (i))

#define atomic_fetch_and_add(v,i) \
    __sync_fetch_and_add(&(v)->value,(i))

#define atomic_fetch_and_xor(v,i) \
    __sync_fetch_and_xor(&(v)->value,(i))

#define atomic_dec_and_test(v)    \
    !(__sync_sub_and_fetch(&(v)->value,1))

#define atomic_swap(v,i)          \
    __sync_lock_test_and_set(&(v)->value,(i))

#define atomic_test_and_set(v,i)  \
    __sync_lock_test_and_set(&(v)->value,(i))

#define atomic_release(v)         \
    __sync_lock_release(&(v)->value)

#define atomic_cas(v,o,i)         \
    __sync_bool_compare_and_swap(&(v)->value,(o),(i))


#endif  /* _ATOMIC_H */

