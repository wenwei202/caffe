#include <algorithm>
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/base_conv_layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::WeightAlign(){
	is_sparse_format_weights_ = false;
	const LayerParameter& layerparam = this->layer_param();
	LOG(INFO)<<"layer\t"<<layerparam.name()<<"\t"<<"has sparsity of "<< this->blobs_[0]->GetSparsity();
	this->blobs_[0]->WriteToNistMMIO(layerparam.name()+".weight");
#ifdef USE_MKL
	is_sparse_format_weights_ = this->blobs_[0]->GetSparsity() > 0.8;
	const int M = this->blobs_[0]->shape(0)/group_;
	const int N = this->blobs_[0]->count(1,4);
	const int weight_offset = this->blobs_[0]->count()/group_;
	const int row_offset = this->blobs_[0]->shape(0)/group_ + 1;
    switch (Caffe::mode()) {
    case Caffe::CPU:
    	//convert trained weights in dense format to CSR format
    	for (int g = 0; g < group_; ++g) {
    		caffe_cpu_sparse_dense2csr(M, N,
					this->blobs_[0]->mutable_cpu_data() + weight_offset * g,
					nz_weight_values_.mutable_cpu_data()+ weight_offset * g,
					nz_weight_indices_.mutable_cpu_data()+ weight_offset * g,
					nz_weight_index_pointers_.mutable_cpu_data() + row_offset * g);
    	}
    	break;

    case Caffe::GPU:
#ifndef CPU_ONLY
    	LOG(INFO)<<"Sparse storage format of weights in GPU model  is unimplemented!";
#else
    	NO_GPU;
#endif
    	break;

    default:
    	LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
    }
#else
    LOG(WARNING)<<"Sparse Math BLAS is unsupported without MKL (when it's on CPU mode)!";
#endif

	// is_concatenating_weights_features_ has higher priority than is_sparse_format_weights_
	//get the all-zero column mask for weight matrix
	//caffe_cpu_if_all_zero(const int M, const int N, const Dtype *x, int* y);
	//const int M = this->blobs_[0]->shape(0)/group_;
	//const int N = this->blobs_[0]->count(1,4);
	//const int weight_offset = this->blobs_[0]->count()/group_;
	//const int col_buf_offset = N/group_;
	for (int g = 0; g < group_; ++g) {
		caffe_cpu_if_all_zero(this->blobs_[0]->shape(0)/group_,
				this->blobs_[0]->count(1,4),
				this->blobs_[0]->cpu_data() + this->blobs_[0]->count()/group_ * g,
				col_buf_mask_.mutable_cpu_data() + this->blobs_[0]->count(1,4) * g);
	}
	int masked_col_num = 0;
	for(int idx=0; idx<col_buf_mask_.count();++idx){
		if(col_buf_mask_.cpu_data()[idx]){
			masked_col_num++;
		}
	}
	Dtype group_sparsity = (Dtype)masked_col_num/(Dtype)col_buf_mask_.count();
	LOG(INFO) << Layer<Dtype>::layer_param().name() << " column sparsity: " << group_sparsity;
	is_concatenating_weights_features_ = group_sparsity > 0.05;

	// compress weight matrix
	int left_cols = 0;
	for (int g = 0; g < group_; ++g) {
		caffe_cpu_del_zero_cols(conv_out_channels_ /group_,
			  kernel_dim_ ,
			  this->blobs_[0]->cpu_data() + weight_offset_ * g,
			  squeezed_weight_buffer_.mutable_cpu_data() + weight_offset_ * g,
			  &left_cols,
			  col_buf_mask_.cpu_data() + kernel_dim_ * g );
		left_columns_.push_back(left_cols);
	}

	//get the mask for connections
	if( layerparam.disconnect() ){
		LOG(INFO)<<"zero weights of "<<layerparam.name()<<" are frozen";
		this->blobs_[0]->Disconnect();
	}

}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Configure the kernel size, padding, stride, and inputs.
  ConvolutionParameter conv_param = this->layer_param_.convolution_param();
  force_nd_im2col_ = conv_param.force_nd_im2col();
  channel_axis_ = bottom[0]->CanonicalAxisIndex(conv_param.axis());
  const int first_spatial_axis = channel_axis_ + 1;
  const int num_axes = bottom[0]->num_axes();
  num_spatial_axes_ = num_axes - first_spatial_axis;
  CHECK_GE(num_spatial_axes_, 0);
  vector<int> bottom_dim_blob_shape(1, num_spatial_axes_ + 1);
  vector<int> spatial_dim_blob_shape(1, std::max(num_spatial_axes_, 1));
  // Setup filter kernel dimensions (kernel_shape_).
  kernel_shape_.Reshape(spatial_dim_blob_shape);
  int* kernel_shape_data = kernel_shape_.mutable_cpu_data();
  if (conv_param.has_kernel_h() || conv_param.has_kernel_w()) {
    CHECK_EQ(num_spatial_axes_, 2)
        << "kernel_h & kernel_w can only be used for 2D convolution.";
    CHECK_EQ(0, conv_param.kernel_size_size())
        << "Either kernel_size or kernel_h/w should be specified; not both.";
    kernel_shape_data[0] = conv_param.kernel_h();
    kernel_shape_data[1] = conv_param.kernel_w();
  } else {
    const int num_kernel_dims = conv_param.kernel_size_size();
    CHECK(num_kernel_dims == 1 || num_kernel_dims == num_spatial_axes_)
        << "kernel_size must be specified once, or once per spatial dimension "
        << "(kernel_size specified " << num_kernel_dims << " times; "
        << num_spatial_axes_ << " spatial dims).";
      for (int i = 0; i < num_spatial_axes_; ++i) {
        kernel_shape_data[i] =
            conv_param.kernel_size((num_kernel_dims == 1) ? 0 : i);
      }
  }
  for (int i = 0; i < num_spatial_axes_; ++i) {
    CHECK_GT(kernel_shape_data[i], 0) << "Filter dimensions must be nonzero.";
  }
  // Setup stride dimensions (stride_).
  stride_.Reshape(spatial_dim_blob_shape);
  int* stride_data = stride_.mutable_cpu_data();
  if (conv_param.has_stride_h() || conv_param.has_stride_w()) {
    CHECK_EQ(num_spatial_axes_, 2)
        << "stride_h & stride_w can only be used for 2D convolution.";
    CHECK_EQ(0, conv_param.stride_size())
        << "Either stride or stride_h/w should be specified; not both.";
    stride_data[0] = conv_param.stride_h();
    stride_data[1] = conv_param.stride_w();
  } else {
    const int num_stride_dims = conv_param.stride_size();
    CHECK(num_stride_dims == 0 || num_stride_dims == 1 ||
          num_stride_dims == num_spatial_axes_)
        << "stride must be specified once, or once per spatial dimension "
        << "(stride specified " << num_stride_dims << " times; "
        << num_spatial_axes_ << " spatial dims).";
    const int kDefaultStride = 1;
    for (int i = 0; i < num_spatial_axes_; ++i) {
      stride_data[i] = (num_stride_dims == 0) ? kDefaultStride :
          conv_param.stride((num_stride_dims == 1) ? 0 : i);
      CHECK_GT(stride_data[i], 0) << "Stride dimensions must be nonzero.";
    }
  }
  // Setup pad dimensions (pad_).
  pad_.Reshape(spatial_dim_blob_shape);
  int* pad_data = pad_.mutable_cpu_data();
  if (conv_param.has_pad_h() || conv_param.has_pad_w()) {
    CHECK_EQ(num_spatial_axes_, 2)
        << "pad_h & pad_w can only be used for 2D convolution.";
    CHECK_EQ(0, conv_param.pad_size())
        << "Either pad or pad_h/w should be specified; not both.";
    pad_data[0] = conv_param.pad_h();
    pad_data[1] = conv_param.pad_w();
  } else {
    const int num_pad_dims = conv_param.pad_size();
    CHECK(num_pad_dims == 0 || num_pad_dims == 1 ||
          num_pad_dims == num_spatial_axes_)
        << "pad must be specified once, or once per spatial dimension "
        << "(pad specified " << num_pad_dims << " times; "
        << num_spatial_axes_ << " spatial dims).";
    const int kDefaultPad = 0;
    for (int i = 0; i < num_spatial_axes_; ++i) {
      pad_data[i] = (num_pad_dims == 0) ? kDefaultPad :
          conv_param.pad((num_pad_dims == 1) ? 0 : i);
    }
  }
  // Setup dilation dimensions (dilation_).
  dilation_.Reshape(spatial_dim_blob_shape);
  int* dilation_data = dilation_.mutable_cpu_data();
  const int num_dilation_dims = conv_param.dilation_size();
  CHECK(num_dilation_dims == 0 || num_dilation_dims == 1 ||
        num_dilation_dims == num_spatial_axes_)
      << "dilation must be specified once, or once per spatial dimension "
      << "(dilation specified " << num_dilation_dims << " times; "
      << num_spatial_axes_ << " spatial dims).";
  const int kDefaultDilation = 1;
  for (int i = 0; i < num_spatial_axes_; ++i) {
    dilation_data[i] = (num_dilation_dims == 0) ? kDefaultDilation :
                       conv_param.dilation((num_dilation_dims == 1) ? 0 : i);
  }
  // Special case: im2col is the identity for 1x1 convolution with stride 1
  // and no padding, so flag for skipping the buffer and transformation.
  is_1x1_ = true;
  for (int i = 0; i < num_spatial_axes_; ++i) {
    is_1x1_ &=
        kernel_shape_data[i] == 1 && stride_data[i] == 1 && pad_data[i] == 0;
    if (!is_1x1_) { break; }
  }
  // Configure output channels and groups.
  channels_ = bottom[0]->shape(channel_axis_);
  num_output_ = this->layer_param_.convolution_param().num_output();
  CHECK_GT(num_output_, 0);
  group_ = this->layer_param_.convolution_param().group();
  CHECK_EQ(channels_ % group_, 0);
  CHECK_EQ(num_output_ % group_, 0)
      << "Number of output should be multiples of group.";
  if (reverse_dimensions()) {
    conv_out_channels_ = channels_;
    conv_in_channels_ = num_output_;
  } else {
    conv_out_channels_ = num_output_;
    conv_in_channels_ = channels_;
  }
  // Handle the parameters: weights and biases.
  // - blobs_[0] holds the filter weights
  // - blobs_[1] holds the biases (optional)
  vector<int> weight_shape(2);
  weight_shape[0] = conv_out_channels_;
  weight_shape[1] = conv_in_channels_ / group_;
  for (int i = 0; i < num_spatial_axes_; ++i) {
    weight_shape.push_back(kernel_shape_data[i]);
  }
  bias_term_ = this->layer_param_.convolution_param().bias_term();
  vector<int> bias_shape(bias_term_, num_output_);
  if (this->blobs_.size() > 0) {
    CHECK_EQ(1 + bias_term_, this->blobs_.size())
        << "Incorrect number of weight blobs.";
    if (weight_shape != this->blobs_[0]->shape()) {
      Blob<Dtype> weight_shaped_blob(weight_shape);
      LOG(FATAL) << "Incorrect weight shape: expected shape "
          << weight_shaped_blob.shape_string() << "; instead, shape was "
          << this->blobs_[0]->shape_string();
    }
    if (bias_term_ && bias_shape != this->blobs_[1]->shape()) {
      Blob<Dtype> bias_shaped_blob(bias_shape);
      LOG(FATAL) << "Incorrect bias shape: expected shape "
          << bias_shaped_blob.shape_string() << "; instead, shape was "
          << this->blobs_[1]->shape_string();
    }
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    if (bias_term_) {
      this->blobs_.resize(2);
    } else {
      this->blobs_.resize(1);
    }
    // Initialize and fill the weights:
    // output channels x input channels per-group x kernel height x kernel width
    this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.convolution_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
    // If necessary, initialize and fill the biases.
    if (bias_term_) {
      this->blobs_[1].reset(new Blob<Dtype>(bias_shape));
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
          this->layer_param_.convolution_param().bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());
    }
  }
  kernel_dim_ = this->blobs_[0]->count(1);
  weight_offset_ = conv_out_channels_ * kernel_dim_ / group_;
  // Propagate gradients to the parameters (as directed by backward pass).
  this->param_propagate_down_.resize(this->blobs_.size(), true);

  is_concatenating_weights_features_ = false;
  dense_feature_map_mask_.Reshape(1,1,1,channels_);
  squeezed_weight_buffer_.Reshape(this->blobs_[0]->shape(0),this->blobs_[0]->shape(1),this->blobs_[0]->shape(2),this->blobs_[0]->shape(3));
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int first_spatial_axis = channel_axis_ + 1;
  CHECK_EQ(bottom[0]->num_axes(), first_spatial_axis + num_spatial_axes_)
      << "bottom num_axes may not change.";
  num_ = bottom[0]->count(0, channel_axis_);
  CHECK_EQ(bottom[0]->shape(channel_axis_), channels_)
      << "Input size incompatible with convolution kernel.";
  // TODO: generalize to handle inputs of different shapes.
  for (int bottom_id = 1; bottom_id < bottom.size(); ++bottom_id) {
    CHECK(bottom[0]->shape() == bottom[bottom_id]->shape())
        << "All inputs must have the same shape.";
  }
  // Shape the tops.
  bottom_shape_ = &bottom[0]->shape();
  compute_output_shape();
  vector<int> top_shape(bottom[0]->shape().begin(),
      bottom[0]->shape().begin() + channel_axis_);
  top_shape.push_back(num_output_);
  for (int i = 0; i < num_spatial_axes_; ++i) {
    top_shape.push_back(output_shape_[i]);
  }
  for (int top_id = 0; top_id < top.size(); ++top_id) {
    top[top_id]->Reshape(top_shape);
  }
  if (reverse_dimensions()) {
    conv_out_spatial_dim_ = bottom[0]->count(first_spatial_axis);
  } else {
    conv_out_spatial_dim_ = top[0]->count(first_spatial_axis);
  }
  col_offset_ = kernel_dim_ * conv_out_spatial_dim_;
  output_offset_ = conv_out_channels_ * conv_out_spatial_dim_ / group_;
  // Setup input dimensions (conv_input_shape_).
  vector<int> bottom_dim_blob_shape(1, num_spatial_axes_ + 1);
  conv_input_shape_.Reshape(bottom_dim_blob_shape);
  int* conv_input_shape_data = conv_input_shape_.mutable_cpu_data();
  for (int i = 0; i < num_spatial_axes_ + 1; ++i) {
    if (reverse_dimensions()) {
      conv_input_shape_data[i] = top[0]->shape(channel_axis_ + i);
    } else {
      conv_input_shape_data[i] = bottom[0]->shape(channel_axis_ + i);
    }
  }
  // The im2col result buffer will only hold one image at a time to avoid
  // overly large memory usage. In the special case of 1x1 convolution
  // it goes lazily unused to save memory.
  col_buffer_shape_.clear();
  col_buffer_shape_.push_back(kernel_dim_ * group_);
  for (int i = 0; i < num_spatial_axes_; ++i) {
    if (reverse_dimensions()) {
      col_buffer_shape_.push_back(input_shape(i + 1));
    } else {
      col_buffer_shape_.push_back(output_shape_[i]);
    }
  }
  col_buffer_.Reshape(col_buffer_shape_);

  if(!reverse_dimensions()){
	  col_buf_mask_.Reshape(1,1,1,kernel_dim_*group_);
#ifdef	GPU_USE_CUSPARSE
	  nonzero_elements_buffer_.Reshape(1, 1, 1, col_buffer_.count());//WARNING: real sparse matrix needs many less memory
	  nonzero_indices_buffer_.Reshape(1,1,1,nonzero_elements_buffer_.count());
	  index_pointers_buffer_.Reshape(1,1,1,col_buffer_.shape(1)+1);
	  nonzero_per_rowcol_buffer_.Reshape(1,1,1,col_buffer_.shape(1));
#endif
	  nz_weight_values_.Reshape(1, 1, 1, this->blobs_[0]->count());//nonzero elements
	  nz_weight_indices_.Reshape(1,1,1,nz_weight_values_.count());//index of nonzero
	  nz_weight_index_pointers_.Reshape(1,1,1,this->blobs_[0]->shape(0)+group_);//pointer(index) of indices
  }

  bottom_dim_ = bottom[0]->count(channel_axis_);
  top_dim_ = top[0]->count(channel_axis_);
  num_kernels_im2col_ = conv_in_channels_ * conv_out_spatial_dim_;
  num_kernels_col2im_ = reverse_dimensions() ? top_dim_ : bottom_dim_;
  // Set up the all ones "bias multiplier" for adding biases by BLAS
  out_spatial_dim_ = top[0]->count(first_spatial_axis);
  if (bias_term_) {
    vector<int> bias_multiplier_shape(1, out_spatial_dim_);
    bias_multiplier_.Reshape(bias_multiplier_shape);
    caffe_set(bias_multiplier_.count(), Dtype(1),
        bias_multiplier_.mutable_cpu_data());
  }
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::forward_cpu_gemm(const Dtype* input,
    const Dtype* weights, Dtype* output, bool skip_im2col) {
  const Dtype* col_buff = input;

  Timer timer;
  timer.Start();
  if (!is_1x1_ ||  is_concatenating_weights_features_) {
    if (!skip_im2col || is_concatenating_weights_features_) {
      if(is_concatenating_weights_features_){
    	  conv_im2col_cpu(input, col_buffer_.mutable_cpu_data(),col_buf_mask_.mutable_cpu_data()/*, dense_feature_map_mask_.mutable_cpu_data()*/);
      }else{
    	  conv_im2col_cpu(input, col_buffer_.mutable_cpu_data());
      }
    }
    col_buff = col_buffer_.cpu_data();
  }
  float lowering_passed_time = timer.MicroSeconds();
  //LOG(INFO)<<this->layer_param().name()<<"\t group all"<<": "<<lowering_passed_time<<" us (Lowering Timing)";

  /*
  col_buffer_.Snapshot(Layer<Dtype>::layer_param().name()+".blob");
  LOG(INFO)<<Layer<Dtype>::layer_param().name();
  for(int tmp_i=0;tmp_i<dense_feature_map_mask_.count();tmp_i++){
  	  LOG(INFO) << tmp_i<<"\t"<<dense_feature_map_mask_.cpu_data()[tmp_i];
  }
  for(int tmp_i=0;tmp_i<col_buf_mask_.count();tmp_i++){
    	  LOG(INFO) << tmp_i<<"\t"<<col_buf_mask_.cpu_data()[tmp_i];
  }*/

  int offset_sum = 0;
  Timer total_timer;
  total_timer.Start();
  for (int g = 0; g < group_; ++g) {
		  if(is_concatenating_weights_features_){
//			  int left_cols = 0;
//			  caffe_cpu_del_zero_cols(conv_out_channels_ /group_,
//					  kernel_dim_ / group_,
//					  weights + weight_offset_ * g,
//					  squeezed_weight_buffer_.mutable_cpu_data(),
//					  &left_cols,
//					  col_buf_mask_.cpu_data() + kernel_dim_ / group_ * g );
			  //Timer timer;
			  timer.Start();
			  int left_cols = left_columns_[g];
			  caffe_cpu_cblas_gemm(conv_out_channels_ /
					  group_, conv_out_spatial_dim_, left_cols,
					  (Dtype)1., squeezed_weight_buffer_.cpu_data() + weight_offset_ * g,
					  kernel_dim_ , col_buff + offset_sum,
					conv_out_spatial_dim_, (Dtype)0., output + output_offset_ * g, conv_out_spatial_dim_);
			  offset_sum += left_cols * conv_out_spatial_dim_;
			  //float passed_time = timer.MicroSeconds();
			  //LOG(INFO)<<this->layer_param().name()<<"\t group "<<g<<": "<<passed_time<<" us (Column Concatenation Timing)";

		  }else{
			  int M = conv_out_channels_ /group_;
			  int N = conv_out_spatial_dim_;
			  int K = kernel_dim_;
			  if(is_sparse_format_weights_){
				  int row_offset = conv_out_channels_ /group_ + 1;
				  //int nnz = *(nz_weight_index_pointers_.cpu_data() + row_offset * g + conv_out_channels_ /group_);
				  //Timer timer;
				  timer.Start();
				  caffe_cpu_sparse_mmcsr(M,
						  N,
						  K,
						  (Dtype)1.,
						  nz_weight_values_.cpu_data()+ weight_offset_ * g,
						  nz_weight_indices_.cpu_data()+ weight_offset_ * g,
						  nz_weight_index_pointers_.cpu_data() + row_offset * g,
						  nz_weight_index_pointers_.cpu_data() + row_offset * g + 1,
						  col_buff + col_offset_ * g,
						  (Dtype)0.,output + output_offset_ * g);
				  //float passed_time = timer.MicroSeconds();
				  //LOG(INFO)<<this->layer_param().name()<<"\t group "<<g<<": "<<passed_time<<" us (Compressed Row Storage Timing)";
//				  long mem_bytes = nnz*(sizeof(Dtype)+sizeof(int))+M*sizeof(int)+K*N*sizeof(Dtype)+M*N*sizeof(Dtype);//
//				  LOG(INFO)<<this->layer_param().name()<<"\t group "<<g<<": "
//						  <<"A("<<M<<"x"<<K<<" nnz:"<<nnz<<")*B("<<K<<"x"<<N<<")=C("<<M<<"x"<<N<<") "
//						  <<mem_bytes<<" B/ "<<passed_time<<" us = "
//						  <<""<<mem_bytes/passed_time << " MB/s";

			  }else{
				//Timer timer;
				timer.Start();
				caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M, N, K,
						  (Dtype)1., weights + weight_offset_ * g, col_buff + col_offset_ * g,
						  (Dtype)0., output + output_offset_ * g);
				//float passed_time = timer.MicroSeconds();
				//LOG(INFO)<<this->layer_param().name()<<"\t group "<<g<<": "<<passed_time<<" us (Dense Scheme Timing)";
//				int M = conv_out_channels_ /group_;
//				int N = conv_out_spatial_dim_;
//				int K = kernel_dim_ / group_;

//				long mem_bytes = (M*K+K*N+M*N)*sizeof(Dtype);
//				LOG(INFO)<<this->layer_param().name()<<"\t group "<<g<<": "
//						<<"A("<<M<<"x"<<K<<")*B("<<K<<"x"<<N<<")=C("<<M<<"x"<<N<<") "
//						<<mem_bytes<<" B/ "<<passed_time<<" us = "
//						<<""<<mem_bytes/passed_time << " MB/s";

			  }
			  //LOG(INFO)<<"mkl_get_max_threads="<<mkl_get_max_threads();
		  }
  }
  float mm_passed_time = total_timer.MicroSeconds();
  //LOG(INFO)<<this->layer_param().name()<<"\t group all"<<": "<< lowering_passed_time*100/(mm_passed_time+lowering_passed_time)<<" % (Lowering Over Matrix Multiplying Timing)";

}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::forward_cpu_bias(Dtype* output,
    const Dtype* bias) {
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
      out_spatial_dim_, 1, (Dtype)1., bias, bias_multiplier_.cpu_data(),
      (Dtype)1., output);
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::backward_cpu_gemm(const Dtype* output,
    const Dtype* weights, Dtype* input) {
  Dtype* col_buff = col_buffer_.mutable_cpu_data();
  if (is_1x1_) {
    col_buff = input;
  }
  for (int g = 0; g < group_; ++g) {
    caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, kernel_dim_,
        conv_out_spatial_dim_, conv_out_channels_ / group_,
        (Dtype)1., weights + weight_offset_ * g, output + output_offset_ * g,
        (Dtype)0., col_buff + col_offset_ * g);
  }
  if (!is_1x1_) {
    conv_col2im_cpu(col_buff, input);
  }
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::weight_cpu_gemm(const Dtype* input,
    const Dtype* output, Dtype* weights) {
  const Dtype* col_buff = input;
  if (!is_1x1_) {
    conv_im2col_cpu(input, col_buffer_.mutable_cpu_data());
    col_buff = col_buffer_.cpu_data();
  }
  for (int g = 0; g < group_; ++g) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, conv_out_channels_ / group_,
        kernel_dim_, conv_out_spatial_dim_,
        (Dtype)1., output + output_offset_ * g, col_buff + col_offset_ * g,
        (Dtype)1., weights + weight_offset_ * g);
  }
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::backward_cpu_bias(Dtype* bias,
    const Dtype* input) {
  caffe_cpu_gemv<Dtype>(CblasNoTrans, num_output_, out_spatial_dim_, 1.,
      input, bias_multiplier_.cpu_data(), 1., bias);
}

#ifndef CPU_ONLY

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::forward_gpu_gemm(const Dtype* input,
    const Dtype* weights, Dtype* output, bool skip_im2col) {
  const Dtype* col_buff = input;
  if (!is_1x1_) {
    if (!skip_im2col) {
      conv_im2col_gpu(input, col_buffer_.mutable_gpu_data());
    }
    col_buff = col_buffer_.gpu_data();
  }

  for (int g = 0; g < group_; ++g) {
#ifdef	GPU_USE_CUSPARSE
	  int total_nonzero = 0;
	  caffe_gpu_sparse_dense2csr(kernel_dim_ / group_, conv_out_spatial_dim_,
						  col_buff + col_offset_ * g,
						  nonzero_per_rowcol_buffer_.mutable_gpu_data(),
						  nonzero_elements_buffer_.mutable_gpu_data(),
						  index_pointers_buffer_.mutable_gpu_data(),
						  nonzero_indices_buffer_.mutable_gpu_data(), &total_nonzero);
	  Dtype sparsity = (Dtype)1.0 - (Dtype)total_nonzero/(Dtype)(kernel_dim_*height_out_*width_out_);
	  //LOG(INFO)<<"Sparsity of "<< Layer<Dtype>::layer_param().name() << ": "<< sparsity;
	  if(sparsity<(Dtype)0.9){
#endif
		 caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, conv_out_channels_ /
        group_, conv_out_spatial_dim_, kernel_dim_,
			 (Dtype)1., weights + weight_offset_ * g, col_buff + col_offset_ * g,
			 (Dtype)0., output + output_offset_ * g);
#ifdef	GPU_USE_CUSPARSE
	  }else{
		 //dense weight matrix multi. sparse feature map matrix
		 //WARNING WARNING WARNING: When A*B, B in format CSR is slow
		 caffe_gpu_sparse_mmcsr(conv_out_channels_ /group_, conv_out_spatial_dim_, kernel_dim_ / group_,
				  (Dtype)1., weights + weight_offset_ * g,
				  total_nonzero,
				  nonzero_elements_buffer_.gpu_data(),
				  index_pointers_buffer_.gpu_data(),
				  nonzero_indices_buffer_.gpu_data(),
				  (Dtype)0., output + output_offset_ * g);
	  }
#endif
  }
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::forward_gpu_bias(Dtype* output,
    const Dtype* bias) {
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
      out_spatial_dim_, 1, (Dtype)1., bias, bias_multiplier_.gpu_data(),
      (Dtype)1., output);
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::backward_gpu_gemm(const Dtype* output,
    const Dtype* weights, Dtype* input) {
  Dtype* col_buff = col_buffer_.mutable_gpu_data();
  if (is_1x1_) {
    col_buff = input;
  }
  for (int g = 0; g < group_; ++g) {
    caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, kernel_dim_,
        conv_out_spatial_dim_, conv_out_channels_ / group_,
        (Dtype)1., weights + weight_offset_ * g, output + output_offset_ * g,
        (Dtype)0., col_buff + col_offset_ * g);
  }
  if (!is_1x1_) {
    conv_col2im_gpu(col_buff, input);
  }
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::weight_gpu_gemm(const Dtype* input,
    const Dtype* output, Dtype* weights) {
  const Dtype* col_buff = input;
  if (!is_1x1_) {
    conv_im2col_gpu(input, col_buffer_.mutable_gpu_data());
    col_buff = col_buffer_.gpu_data();
  }
  for (int g = 0; g < group_; ++g) {
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, conv_out_channels_ / group_,
        kernel_dim_, conv_out_spatial_dim_,
        (Dtype)1., output + output_offset_ * g, col_buff + col_offset_ * g,
        (Dtype)1., weights + weight_offset_ * g);
  }
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::backward_gpu_bias(Dtype* bias,
    const Dtype* input) {
  caffe_gpu_gemv<Dtype>(CblasNoTrans, num_output_, out_spatial_dim_, 1.,
      input, bias_multiplier_.gpu_data(), 1., bias);
}

#endif  // !CPU_ONLY

INSTANTIATE_CLASS(BaseConvolutionLayer);

}  // namespace caffe
