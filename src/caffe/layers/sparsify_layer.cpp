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
	Dtype thre = sparsify_param_.thre();
	if(0 == thre){
		caffe_copy(bottom[0]->count(), bottom_data, top_data);
	} else {
		for(int i=0; i<bottom[0]->count(); i++){
			if(fabs(bottom_data[i])<=thre) top_data[i] = 0;
			else if (bottom_data[i] > thre ) top_data[i] = bottom_data[i] - thre;
			else top_data[i] = bottom_data[i] + thre;
		}
	}
	if(this->layer_param_.display()){
		LOG(INFO) << "Sparsity of output of layer "
				<< this->layer_param_.name()
				<< " = " << top[0]->GetSparsity(thre);
	}
}

template <typename Dtype>
void SparsifyLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  Dtype thre = sparsify_param_.thre();
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    const Dtype* top_data = top[0]->cpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const int count = bottom[0]->count();
    for (int i = 0; i < count; ++i) {
      if(0==thre){
    	// The intrinsic symmetric rectifying and L1 regularization on outputs
		bottom_diff[i] = ( top_diff[i] + ( (top_data[i] > 0) - (top_data[i] < 0) ) * coef_ ); // L1 regularization on outputs
      } else {
        // The intrinsic symmetric rectifying and L1 regularization on outputs
        bottom_diff[i] = ( top_diff[i] + ( (top_data[i] > 0) - (top_data[i] < 0) ) * coef_ ) // L1 regularization on outputs
    		  * (top_data[i]!=0); //intrinsic symmetric rectifying, bottom blob may be changed if the layer is in-place, so we use top bottom
      }

    }
  }

  if(this->layer_param_.display()){
	  Dtype total_gradients =
			  caffe_cpu_asum(top[0]->count(), top[0]->cpu_diff());
	  LOG(INFO) << "Average abs gradient of layer "
			  << this->layer_param_.name()
			  <<" = " << total_gradients / top[0]->count();
  }
}


#ifdef CPU_ONLY
STUB_GPU(SparsifyLayer);
#endif

INSTANTIATE_CLASS(SparsifyLayer);
REGISTER_LAYER_CLASS(Sparsify);

}  // namespace caffe
