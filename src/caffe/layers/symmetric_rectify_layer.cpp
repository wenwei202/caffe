#include <algorithm>
#include <vector>

#include "caffe/filler.hpp"

#include "caffe/layers/neuron_layer.hpp"
#include "caffe/layers/symmetric_rectify_layer.hpp"

namespace caffe {

template <typename Dtype>
void SymmetricRectifyLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  CHECK_GE(bottom[0]->num_axes(), 2)
      << "Number of axes of bottom blob must be >=2.";
  SymmetricRectifyParameter symm_rectify_param = this->layer_param().symmetric_rectify_param();
  int channels = bottom[0]->channels();
  channel_shared_ = symm_rectify_param.channel_shared();
  if (this->blobs_.size() > 0) {
	//LOG(INFO) << "Skipping parameter initialization";
    LOG(FATAL) << "Skipping parameter initialization";
  } else {
    this->blobs_.resize(1);
    if (channel_shared_) {
      this->blobs_[0].reset(new Blob<Dtype>(vector<int>(0)));
    } else {
      this->blobs_[0].reset(new Blob<Dtype>(vector<int>(1, channels)));
    }
    shared_ptr<Filler<Dtype> > filler;
    if (symm_rectify_param.has_filler()) {
      filler.reset(GetFiller<Dtype>(symm_rectify_param.filler()));
    } else {
      LOG(INFO)<<"Using default filler for layer "
    		  << this->layer_param_.name();
      FillerParameter filler_param;
      filler_param.set_type("constant");
      filler_param.set_value(0.0);
      filler.reset(GetFiller<Dtype>(filler_param));
    }
    filler->Fill(this->blobs_[0].get());

  }
  if (channel_shared_) {
    CHECK_EQ(this->blobs_[0]->count(), 1)
        << "Threshold size is inconsistent with prototxt config";
  } else {
    CHECK_EQ(this->blobs_[0]->count(), channels)
        << "Threshold size is inconsistent with prototxt config";
  }

  for(int i=0;i<this->blobs_[0]->count();i++){
    CHECK_GE(this->blobs_[0]->cpu_data()[i],0)
  	  << "Threshold of rectifying must not be negtive.";
  }

  // Propagate gradients to the parameters (as directed by backward pass).
  this->param_propagate_down_.resize(this->blobs_.size(), true);
  multiplier_.Reshape(vector<int>(1, bottom[0]->count(1)));
  backward_buff_.Reshape(vector<int>(1, bottom[0]->count(1)));
  caffe_set(multiplier_.count(), Dtype(1), multiplier_.mutable_cpu_data());
}

template <typename Dtype>
void SymmetricRectifyLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  CHECK_GE(bottom[0]->num_axes(), 2)
      << "Number of axes of bottom blob must be >=2.";
  top[0]->ReshapeLike(*bottom[0]);
  //if (bottom[0] == top[0]) {
    // For in-place computation
    //bottom_memory_.ReshapeLike(*bottom[0]);
  //}
}

template <typename Dtype>
void SymmetricRectifyLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  const int dim = bottom[0]->count(2);
  const int channels = bottom[0]->channels();
  const Dtype* thre_data = this->blobs_[0]->cpu_data();

  // For in-place computation
  //if (bottom[0] == top[0]) {
  //  caffe_copy(count, bottom_data, bottom_memory_.mutable_cpu_data());
  //}

  // After parameter updating, thresholds of rectifying must not be negtive.
  for(int i=0; i<this->blobs_[0]->count(); i++){
  	if(this->blobs_[0]->cpu_data()[i] < 0){
  		this->blobs_[0]->mutable_cpu_data()[i] = 0;
  	}
  }

  // if channel_shared, channel index in the following computation becomes
  // always zero.
  const int div_factor = channel_shared_ ? channels : 1;
  for (int i = 0; i < count; ++i) {
    int c = (i / dim) % channels / div_factor;

    if(fabs(bottom_data[i])<=thre_data[c]) top_data[i] = 0;
	else if (bottom_data[i] > thre_data[c] ) top_data[i] = bottom_data[i] - thre_data[c];
	else top_data[i] = bottom_data[i] + thre_data[c];
  }

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
void SymmetricRectifyLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_data = top[0]->cpu_data();
  //const Dtype* thre_data = this->blobs_[0]->cpu_data();
  const Dtype* top_diff = top[0]->cpu_diff();
  const int count = bottom[0]->count();
  const int dim = bottom[0]->count(2);
  const int channels = bottom[0]->channels();

  // For in-place computation
  //if (top[0] == bottom[0]) {
  //  bottom_data = bottom_memory_.cpu_data();
  //}

  // if channel_shared, channel index in the following computation becomes
  // always zero.
  const int div_factor = channel_shared_ ? channels : 1;

  // Propagte to param
  // Since to write bottom diff will affect top diff if top and bottom blobs
  // are identical (in-place computaion), we first compute param backward to
  // keep top_diff unchanged.
  if (this->param_propagate_down_[0]) {
    Dtype* thre_diff = this->blobs_[0]->mutable_cpu_diff();
    const Dtype* thre_data = this->blobs_[0]->cpu_data();
    for (int i = 0; i < count; ++i) {
      int c = (i / dim) % channels / div_factor;
      thre_diff[c] += top_diff[i] * ( (top_data[i] < 0) - (top_data[i] > 0) );
    }
    for (int c_i = 0; c_i < this->blobs_[0]->count(); c_i++){
      //biasing threshold to larger value for higher sparsity
	  thre_diff[c_i] += ((thre_data[c_i]<0) - (thre_data[c_i]>=0))
			  * this->layer_param().symmetric_rectify_param().thre_decay();
    }
  }
  // Propagate to bottom
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    for (int i = 0; i < count; ++i) {
      bottom_diff[i] = top_diff[i] * (fabs(top_data[i])>0);
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(SymmetricRectifyLayer);
#endif

INSTANTIATE_CLASS(SymmetricRectifyLayer);
REGISTER_LAYER_CLASS(SymmetricRectify);

}  // namespace caffe
