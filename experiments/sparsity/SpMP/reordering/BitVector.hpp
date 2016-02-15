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

#pragma once

namespace SpMP
{

typedef char BitVectorType;

class BitVector
{
public :
  BitVector(int n) {
    int n_ = (n + sizeof(BitVectorType) - 1)/sizeof(BitVectorType);
    bv_ = new BitVectorType[n_];
#pragma omp parallel for
    for (int i = 0; i < n_; ++i) {
      bv_[i] = 0;
    }
  }

  ~BitVector() { delete[] bv_; }

  void set(int i) {
    bv_[getIndexOf_(i)] |= getMaskOf_(i);
  }

  bool get(int i) const {
    return bv_[getIndexOf_(i)] & getMaskOf_(i);
  }

  bool testAndSet(int i) {
    if (!get(i)) {
      BitVectorType mask = getMaskOf_(i);
      BitVectorType prev = __sync_fetch_and_or(bv_ + getIndexOf_(i), mask);
      return !(prev & mask);
    }
    else {
      return false;
    }
  }

  void atomicClear(int i) {
    __sync_fetch_and_and(bv_ + getIndexOf_(i), ~getMaskOf_(i));
  }

private :
  typedef char BitVectorType;

  static int getIndexOf_(int i) { return i/sizeof(BitVectorType); }
  static int getBitIndexOf_(int i) { return i%sizeof(BitVectorType); }
  static BitVectorType getMaskOf_(int i) { return 1 << getBitIndexOf_(i); }

  BitVectorType *bv_;
};

} // namespace SpMP
