#include <algorithm>
#include <vector>

#include "caffe/layers/sparsify_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void SparsifyForward(const int n,
    const Dtype* in_data, Dtype* out_data, Dtype thre) {
  CUDA_KERNEL_LOOP(index, n) {
	  if( (in_data[index]<=thre) && (in_data[index]>=-thre) ){
		  out_data[index] = 0;
	  } else if (in_data[index]>thre) {
		  out_data[index] = in_data[index] - thre;
	  } else{
		  out_data[index] = in_data[index] + thre;
	  }
  }
}

template <typename Dtype>
void SparsifyLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
	CHECK_EQ(bottom[0]->count(),top[0]->count());
	int count = bottom[0]->count();
	if(0 == sparsify_param_.thre()){
		caffe_copy(count,
				bottom[0]->gpu_data(),
				top[0]->mutable_gpu_data());
	} else {
		SparsifyForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
				count, bottom[0]->gpu_data(), top[0]->mutable_gpu_data(),
				sparsify_param_.thre());
		CUDA_POST_KERNEL_CHECK;
	}

	if(this->layer_param_.display()){
		LOG(INFO) << "Sparsity of outputs of layer "
				<< this->layer_param_.name()
				<< " = " << top[0]->GetSparsity(sparsify_param_.thre());
	}
}

template <typename Dtype>
__global__ void SparsifyBackward(const int n, const Dtype* in_diff,
    const Dtype* out_data, Dtype* out_diff, const Dtype coef, const Dtype thre) {
  CUDA_KERNEL_LOOP(index, n) {
	  if(0==thre){
		out_diff[index] = (in_diff[index] + coef * ( (out_data[index] > 0) - (out_data[index] < 0) ));
	  } else {
		// The intrinsic symmetric rectifying and L1 regularization on outputs
		out_diff[index] = (in_diff[index] + coef * ( (out_data[index] > 0) - (out_data[index] < 0) ))
				* (out_data[index]!=0);
	  }
  }
}

template <typename Dtype>
void SparsifyLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  Dtype thre = sparsify_param_.thre();
  if (propagate_down[0]) {
	  const Dtype* bottom_data = bottom[0]->gpu_data();
	  const Dtype* top_data = top[0]->gpu_data();
	  const Dtype* top_diff = top[0]->gpu_diff();
	  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
	  const int count = bottom[0]->count();
	  // NOLINT_NEXT_LINE(whitespace/operators)
	  SparsifyBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
		  count, top_diff, top_data, bottom_diff, coef_, thre);
	  CUDA_POST_KERNEL_CHECK;
  }

  if(this->layer_param_.display()){
  	  Dtype total_gradients;
  	  caffe_gpu_asum(top[0]->count(),top[0]->gpu_diff(),&total_gradients);
  	  LOG(INFO) << "Average abs gradient of layer "
  			<< this->layer_param_.name()
  			<< " = " << total_gradients / top[0]->count();
  }
}


INSTANTIATE_LAYER_GPU_FUNCS(SparsifyLayer);


}  // namespace caffe
