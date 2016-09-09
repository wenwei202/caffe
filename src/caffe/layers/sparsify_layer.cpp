#include <algorithm>
#include <vector>

#include "caffe/layers/sparsify_layer.hpp"
#include <math.h>

namespace caffe {

template <typename Dtype>
void SparsifyLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
	sparsify_param_ = this->layer_param_.sparsify_param();
	coef_ = sparsify_param_.coef();
	CHECK_GE(coef_,0);
	CHECK_GE(sparsify_param_.thre(),0);
}

template <typename Dtype>
void SparsifyLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
	const Dtype* bottom_data = bottom[0]->cpu_data();
	Dtype* top_data = top[0]->mutable_cpu_data();
	CHECK_EQ(bottom[0]->count(),top[0]->count());
	if(0 == sparsify_param_.thre()){
		caffe_copy(bottom[0]->count(), bottom_data, top_data);
	} else {
		for(int i=0; i<bottom[0]->count(); i++){
			if(fabs(bottom_data[i])<=sparsify_param_.thre()) top_data[i] = 0;
			else top_data[i] = bottom_data[i];
		}
	}
	if(sparsify_param_.display()){
		LOG(INFO) << "Sparsity of inputs of layer "
				<< this->layer_param_.name()
				<< " = " << bottom[0]->GetSparsity(sparsify_param_.thre());
	}
}

template <typename Dtype>
void SparsifyLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const int count = bottom[0]->count();
    for (int i = 0; i < count; ++i) {
      bottom_diff[i] = top_diff[i] +
    		  ( (bottom_data[i] > 0) - (bottom_data[i] < 0) ) * coef_;
    }
  }

  if(sparsify_param_.display()){
	  Dtype total_gradients =
			  caffe_cpu_asum(top[0]->count(), top[0]->cpu_diff());
	  LOG(INFO) << "Average abs gradient = " << total_gradients / top[0]->count();
  }
}


#ifdef CPU_ONLY
STUB_GPU(SparsifyLayer);
#endif

INSTANTIATE_CLASS(SparsifyLayer);
REGISTER_LAYER_CLASS(Sparsify);

}  // namespace caffe
