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

/** A very simple memory pool to minimize allocation time in memory-limited
 * Xeon Phi cards.
 * Basically, it's a circular buffer.
 * Allocation is simply done by returning the current offset and then shifting
 * the offset (and wrapping around it when necessary).
 *
 * Example:
 *
 * foo(); // internally uses pool->Allocate(), which will shift the offset
 * size_t fooEnd = pool->getTail();
 * ...
 * boo(); // internally uses pool->Allocate()
 * ...
 * size_t barBegin = pool->getTail();
 * bar();
 * ...
 * pool->setHeadOffset(fooEnd); // free those allocated before foo
 * ...
 * pool->setTailOffset(barBegin); // free those allocated after boo
 */

#pragma once

#include <cstdlib>

namespace SpMP
{

class MemoryPool
{
public :
  MemoryPool();
  MemoryPool(size_t sz);
  ~MemoryPool();

  void initialize(size_t sz);
  void finalize();

  void setHead(size_t offset);
  void setTail(size_t offset);

  size_t getHead() const;
  size_t getTail() const;

  void *allocate(size_t sz, int align = 64);
  void *allocateFront(size_t sz, int align = 64);
  void deallocateAll();

  template<typename T> T *allocate(size_t cnt) {
    return (T *)allocate(sizeof(T)*cnt);
  }
  template<typename T> T *allocateFront(size_t cnt) {
    return (T *)allocateFront(sizeof(T)*cnt);
  }

  /**
   * @return true if ptr is in this pool
   */
  bool contains(const void *ptr) const;
  
  static MemoryPool *getSingleton();

private :
  size_t size_;
  size_t head_, tail_;
  char *buffer_;
};

} // namespace SpMP
