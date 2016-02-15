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

#include "MemoryPool.hpp"
#include "Utils.hpp"

namespace SpMP
{

static MemoryPool singleton;

MemoryPool::MemoryPool() : size_(0), head_(0), tail_(0), buffer_(NULL)
{
}

MemoryPool::MemoryPool(size_t sz)
{
  initialize(sz);
}

MemoryPool::~MemoryPool()
{
  finalize();
}

void MemoryPool::initialize(size_t sz)
{
  size_ = sz;
  head_ = 0;
  tail_ = 0;
  buffer_ = (char *)malloc_huge_pages(sz);
  assert(buffer_);
}

void MemoryPool::finalize()
{
  if (buffer_) free_huge_pages(buffer_);
  buffer_ = NULL;
  head_ = 0;
  tail_ = 0;
}

void MemoryPool::setHead(size_t offset)
{
  assert(offset < size_);
  if (tail_ >= head_) { // not wrapped around
    assert(offset >= head_ && offset <= tail_); // must shrink
  }
  else { // wrapped around
    assert(offset <= tail_ || offset >= head_); // must shrink
  }

  head_ = offset;
}

void MemoryPool::setTail(size_t offset)
{
  assert(offset < size_);
  if (tail_ >= head_) { // not wrapped around
    assert(offset >= head_ && offset <= tail_); // must shrink
  }
  else { // wrapped around
    assert(offset <= tail_ || offset >= head_); // must shrink
  }

  tail_ = offset;
}

size_t MemoryPool::getHead() const
{
  return head_;
}

size_t MemoryPool::getTail() const
{
  return tail_;
}

void *MemoryPool::allocate(size_t sz, int align /*=64*/)
{
  sz = (sz + align - 1)/align*align;

  if (tail_ >= head_) { // not wrapped around
    if (tail_ + sz >= size_) { // wrap around
      tail_ = sz;
      assert(tail_ < head_); // check overflow
      return buffer_;
    }
    else {
      tail_ += sz;
      return buffer_ + (tail_ - sz);
    }
  }
  else { // wrapped around
    assert(tail_ + sz < head_); // check overflow
    tail_ += sz;
    return buffer_ + (tail_ - sz);
  }
}

void *MemoryPool::allocateFront(size_t sz, int align /*=64*/)
{
  sz = (sz + align - 1)/align*align;

  if (tail_ >= head_) { // not wrapped around
    if (head_ < sz) { // wrap around
      head_ = size_ - sz;
      assert(head_ > tail_); // check overflow
    }
    else {
      head_ -= sz;
    }
  }
  else { // wrapped around
    assert(head_ > tail_ + sz); // check overflow
    head_ -= sz;
  }

  return buffer_ + head_;
}

void MemoryPool::deallocateAll()
{
  head_ = 0;
  tail_ = 0;
}

bool MemoryPool::contains(const void *ptr) const
{
  if (!ptr || !buffer_ || ptr < buffer_ || ptr >= buffer_ + size_) return false;
  if (tail_ >= head_) {
    return ptr >= buffer_ + head_ && ptr < buffer_ + tail_;
  }
  else { // wrapped around
    return !(ptr >= buffer_ + tail_ && ptr < buffer_ + head_);
  }
}

MemoryPool *MemoryPool::getSingleton()
{
  return &singleton;
}

} // namespace SpMP
