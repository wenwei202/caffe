#include <algorithm>
#include <vector>

#include "caffe/layers/neuron_layer.hpp"
#include "caffe/layers/symmetric_rectify_layer.hpp"

namespace caffe {

// CUDA kernele for forward
template <typename Dtype>
__global__ void SymmetricRectifyForward(const int n, const int channels, const int dim,
    const Dtype* in, Dtype* out, const Dtype* thre_data,
    const int div_factor) {
  CUDA_KERNEL_LOOP(index, n) {
    int c = (index / dim) % channels / div_factor;
    //out[index] = in[index] > 0 ? in[index] : in[index] * thre_data[c];
    if( in[index]>thre_data[c] ) out[index] = in[index] - thre_data[c];
	else if (in[index]<-thre_data[c] ) out[index] = in[index] + thre_data[c];
	else out[index] = 0;
  }
}

// CUDA kernel for blob
template <typename Dtype>
__global__ void SymmetricRectifyZeroutBlob(const int n, Dtype* thre_data) {
  CUDA_KERNEL_LOOP(index, n) {
    if( thre_data[index]<0 ) thre_data[index] = 0;
  }
}

// CUDA kernel for blob
template <typename Dtype>
__global__ void SymmetricRectifyRegularizeBlob(const int n, const Dtype thre_decay,
		const Dtype* thre_data, Dtype* thre_diff) {
  CUDA_KERNEL_LOOP(index, n) {
	  thre_diff[index] += ((thre_data[index]<0) - (thre_data[index]>=0)) * thre_decay;
  }
}

// CUDA kernel for bottom backward
template <typename Dtype>
__global__ void SymmetricRectifyBackward(const int n, const Dtype* in_diff, const Dtype* out_data, Dtype* out_diff) {
  CUDA_KERNEL_LOOP(index, n) {
    out_diff[index] = in_diff[index] * (out_data[index] != 0);
  }
}

// CUDA kernel for element-wise parameter backward
template <typename Dtype>
__global__ void SymmetricRectifyParamBackward(const int n,
    const int rows, const int rowPitch, const Dtype* in_diff,
    const Dtype* out_data, Dtype* out_diff) {
  CUDA_KERNEL_LOOP(index, n) {
    out_diff[index] = in_diff[index] * ((out_data[index] < 0) - (out_data[index] > 0) );
    for ( int k = 1; k < rows; k++ ) {
        //out_diff[index] += in_diff[index + k*rowPitch]
        //   * out_data[index + k*rowPitch] * (out_data[index + k*rowPitch] <= 0);
        out_diff[index] += in_diff[index + k*rowPitch]
             * ((out_data[index + k*rowPitch] < 0) - (out_data[index + k*rowPitch] > 0) );
    }
  }
}

template <typename Dtype>
void SymmetricRectifyLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  const int dim = bottom[0]->count(2);
  const int channels = bottom[0]->channels();
  const Dtype* thre_data = this->blobs_[0]->gpu_data();
  const int div_factor = channel_shared_ ? channels : 1;

  // For in-place computation
  //if (top[0] == bottom[0]) {
  //  caffe_copy(count, bottom_data, bottom_memory_.mutable_gpu_data());
  //}
  const int blob_count = this->blobs_[0]->count();
  SymmetricRectifyZeroutBlob<Dtype><<<CAFFE_GET_BLOCKS(blob_count), CAFFE_CUDA_NUM_THREADS>>>(
		  blob_count, this->blobs_[0]->mutable_gpu_data());
  CUDA_POST_KERNEL_CHECK;

  // NOLINT_NEXT_LINE(whitespace/operators)
  SymmetricRectifyForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, channels, dim, bottom_data, top_data, thre_data, div_factor);
  CUDA_POST_KERNEL_CHECK;
  //display
  if(this->layer_param_.display()){
	  ostringstream msg_stream;
	  msg_stream << "Threshold(s) of layer " << this->layer_param_.name() <<": ";
	  for(int i=0; i<this->blobs_[0]->count(); i++){
		msg_stream << this->blobs_[0]->cpu_data()[i] << " ";
	  }
	  LOG(INFO) << msg_stream.str();
  }
}

template <typename Dtype>
void SymmetricRectifyLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_data = top[0]->gpu_data();
  const Dtype* top_diff = top[0]->gpu_diff();
  const int count = bottom[0]->count();
  const int dim = bottom[0]->count(2);
  const int channels = bottom[0]->channels();

  // For in-place computation
  //if (top[0] == bottom[0]) {
  //  bottom_data = bottom_memory_.gpu_data();
  //}

  // Propagate to param
  // Since to write bottom diff will affect top diff if top and bottom blobs
  // are identical (in-place computaion), we first compute param backward to
  // keep top_diff unchanged.
  if (this->param_propagate_down_[0]) {
    Dtype* thre_diff = this->blobs_[0]->mutable_gpu_diff();
    const Dtype* thre_data = this->blobs_[0]->gpu_data();
    int cdim = channels * dim;

    // compute element-wise diff
    // NOLINT_NEXT_LINE(whitespace/operators)
    SymmetricRectifyParamBackward<Dtype><<<CAFFE_GET_BLOCKS(cdim),
      CAFFE_CUDA_NUM_THREADS>>>(
      cdim, bottom[0]->num(), top[0]->offset(1), top_diff ,
      top_data ,
      backward_buff_.mutable_gpu_diff());
    CUDA_POST_KERNEL_CHECK;
    if (channel_shared_) {
      Dtype dsum;
      caffe_gpu_dot<Dtype>(channels * dim, backward_buff_.gpu_diff(),
       multiplier_.gpu_data(), &dsum);
      caffe_gpu_add_scalar(this->blobs_[0]->count(), Dtype(dsum), thre_diff);
    } else {
      caffe_gpu_gemv<Dtype>(CblasNoTrans, channels, dim, 1.,
        backward_buff_.gpu_diff(), multiplier_.gpu_data(), 1.,
        thre_diff);
    }

    const int blob_count = this->blobs_[0]->count();
    SymmetricRectifyRegularizeBlob<Dtype><<<CAFFE_GET_BLOCKS(blob_count), CAFFE_CUDA_NUM_THREADS>>>(
		  blob_count,
		  this->layer_param().symmetric_rectify_param().thre_decay(),
		  thre_data, thre_diff);
    CUDA_POST_KERNEL_CHECK;
  }
  // Propagate to bottom
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    // NOLINT_NEXT_LINE(whitespace/operators)
    SymmetricRectifyBackward<Dtype><<<CAFFE_GET_BLOCKS(count),
        CAFFE_CUDA_NUM_THREADS>>>(count, top_diff, top_data, bottom_diff);
    CUDA_POST_KERNEL_CHECK;
  }
}


INSTANTIATE_LAYER_GPU_FUNCS(SymmetricRectifyLayer);


}  // namespace caffe
