#include <string>
#include <vector>

#include "caffe/sgd_solvers.hpp"
#include "caffe/util/hdf5.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/upgrade_proto.hpp"

namespace caffe {

// Return the current learning rate. The currently implemented learning rate
// policies are as follows:
//    - fixed: always return base_lr.
//    - step: return base_lr * gamma ^ (floor(iter / step))
//    - exp: return base_lr * gamma ^ iter
//    - inv: return base_lr * (1 + gamma * iter) ^ (- power)
//    - multistep: similar to step but it allows non uniform steps defined by
//      stepvalue
//    - poly: the effective learning rate follows a polynomial decay, to be
//      zero by the max_iter. return base_lr (1 - iter/max_iter) ^ (power)
//    - sigmoid: the effective learning rate follows a sigmod decay
//      return base_lr ( 1/(1 + exp(-gamma * (iter - stepsize))))
//
// where base_lr, max_iter, gamma, step, stepvalue and power are defined
// in the solver parameter protocol buffer, and iter is the current iteration.
template <typename Dtype>
Dtype SGDSolver<Dtype>::GetLearningRate() {
  Dtype rate;
  const string& lr_policy = this->param_.lr_policy();
  if (lr_policy == "fixed") {
    rate = this->param_.base_lr();
  } else if (lr_policy == "step") {
    this->current_step_ = this->iter_ / this->param_.stepsize();
    rate = this->param_.base_lr() *
        pow(this->param_.gamma(), this->current_step_);
  } else if (lr_policy == "exp") {
    rate = this->param_.base_lr() * pow(this->param_.gamma(), this->iter_);
  } else if (lr_policy == "inv") {
    rate = this->param_.base_lr() *
        pow(Dtype(1) + this->param_.gamma() * this->iter_,
            - this->param_.power());
  } else if (lr_policy == "multistep") {
    if (this->current_step_ < this->param_.stepvalue_size() &&
          this->iter_ >= this->param_.stepvalue(this->current_step_)) {
      this->current_step_++;
      LOG(INFO) << "MultiStep Status: Iteration " <<
      this->iter_ << ", step = " << this->current_step_;
    }
    rate = this->param_.base_lr() *
        pow(this->param_.gamma(), this->current_step_);
  } else if (lr_policy == "poly") {
    rate = this->param_.base_lr() * pow(Dtype(1.) -
        (Dtype(this->iter_) / Dtype(this->param_.max_iter())),
        this->param_.power());
  } else if (lr_policy == "sigmoid") {
    rate = this->param_.base_lr() * (Dtype(1.) /
        (Dtype(1.) + exp(-this->param_.gamma() * (Dtype(this->iter_) -
          Dtype(this->param_.stepsize())))));
  } else {
    LOG(FATAL) << "Unknown learning rate policy: " << lr_policy;
  }
  return rate;
}

template <typename Dtype>
void SGDSolver<Dtype>::PreSolve() {
  // Initialize the history
  const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
  history_.clear();
  update_.clear();
  temp_.clear();
  for (int i = 0; i < net_params.size(); ++i) {
    const vector<int>& shape = net_params[i]->shape();
    history_.push_back(shared_ptr<Blob<Dtype> >(new Blob<Dtype>(shape)));
    update_.push_back(shared_ptr<Blob<Dtype> >(new Blob<Dtype>(shape)));
    temp_.push_back(shared_ptr<Blob<Dtype> >(new Blob<Dtype>(shape)));

    temp_2_.push_back(shared_ptr<Blob<Dtype> >(new Blob<Dtype>()));
    temp_3_.push_back(shared_ptr<Blob<Dtype> >(new Blob<Dtype>()));
    temp_n_.push_back(shared_ptr<Blob<Dtype> >(new Blob<Dtype>()));
    temp_n_2_.push_back(shared_ptr<Blob<Dtype> >(new Blob<Dtype>()));
    temp_c_.push_back(shared_ptr<Blob<Dtype> >(new Blob<Dtype>()));
    temp_nxn_.push_back(shared_ptr<Blob<Dtype> >(new Blob<Dtype>()));
    temp_nxn_2_.push_back(shared_ptr<Blob<Dtype> >(new Blob<Dtype>()));

    vector<int> n_shape(1, net_params[i]->shape(0));
    vector<int> c_shape(1, net_params[i]->count()/net_params[i]->shape(0));
    ones_n_.push_back(shared_ptr<Blob<Dtype> >(new Blob<Dtype>(n_shape)));
    ones_c_.push_back(shared_ptr<Blob<Dtype> >(new Blob<Dtype>(c_shape)));
    caffe_set(ones_n_[i]->count(), Dtype(1),ones_n_[i]->mutable_cpu_data());
    caffe_set(ones_c_[i]->count(), Dtype(1),ones_c_[i]->mutable_cpu_data());
  }
}

template <typename Dtype>
void SGDSolver<Dtype>::ClipGradients() {
  const Dtype clip_gradients = this->param_.clip_gradients();
  if (clip_gradients < 0) { return; }
  const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
  Dtype sumsq_diff = 0;
  for (int i = 0; i < net_params.size(); ++i) {
    sumsq_diff += net_params[i]->sumsq_diff();
  }
  const Dtype l2norm_diff = std::sqrt(sumsq_diff);
  if (l2norm_diff > clip_gradients) {
    Dtype scale_factor = clip_gradients / l2norm_diff;
    LOG(INFO) << "Gradient clipping: scaling down gradients (L2 norm "
        << l2norm_diff << " > " << clip_gradients << ") "
        << "by scale factor " << scale_factor;
    for (int i = 0; i < net_params.size(); ++i) {
      net_params[i]->scale_diff(scale_factor);
    }
  }
}

template <typename Dtype>
void SGDSolver<Dtype>::ApplyUpdate() {
  CHECK(Caffe::root_solver());
  Dtype rate = GetLearningRate();
  if (this->param_.display() && this->iter_ % this->param_.display() == 0) {
    LOG(INFO) << "Iteration " << this->iter_ << ", lr = " << rate;
  }
  ClipGradients();
  for (int param_id = 0; param_id < this->net_->learnable_params().size();
       ++param_id) {
    Normalize(param_id);
    Regularize(param_id);
    ForceRegularize(param_id);
    ComputeUpdateValue(param_id, rate);
  }
  this->net_->Update();
}

template <typename Dtype>
void SGDSolver<Dtype>::Normalize(int param_id) {
  if (this->param_.iter_size() == 1) { return; }
  // Scale gradient to counterbalance accumulation.
  const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
  const Dtype accum_normalization = Dtype(1.) / this->param_.iter_size();
  switch (Caffe::mode()) {
  case Caffe::CPU: {
    caffe_scal(net_params[param_id]->count(), accum_normalization,
        net_params[param_id]->mutable_cpu_diff());
    break;
  }
  case Caffe::GPU: {
#ifndef CPU_ONLY
    caffe_gpu_scal(net_params[param_id]->count(), accum_normalization,
        net_params[param_id]->mutable_gpu_diff());
#else
    NO_GPU;
#endif
    break;
  }
  default:
    LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
  }
}

template <typename Dtype>
void SGDSolver<Dtype>::Regularize(int param_id) {
  const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
  const vector<float>& net_params_weight_decay =
      this->net_->params_weight_decay();
  Dtype weight_decay = this->param_.weight_decay();
  string regularization_type = this->param_.regularization_type();
  Dtype local_decay = weight_decay * net_params_weight_decay[param_id];
  switch (Caffe::mode()) {
  case Caffe::CPU: {
    if (local_decay) {
      if (regularization_type == "L2") {
        // add weight decay
        caffe_axpy(net_params[param_id]->count(),
            local_decay,
            net_params[param_id]->cpu_data(),
            net_params[param_id]->mutable_cpu_diff());
      } else if (regularization_type == "L1") {
        caffe_cpu_sign(net_params[param_id]->count(),
            net_params[param_id]->cpu_data(),
            temp_[param_id]->mutable_cpu_data());
        caffe_axpy(net_params[param_id]->count(),
            local_decay,
            temp_[param_id]->cpu_data(),
            net_params[param_id]->mutable_cpu_diff());
      } else {
        LOG(FATAL) << "Unknown regularization type: " << regularization_type;
      }
    }
    break;
  }
  case Caffe::GPU: {
#ifndef CPU_ONLY
    if (local_decay) {
      if (regularization_type == "L2") {
        // add weight decay
        caffe_gpu_axpy(net_params[param_id]->count(),
            local_decay,
            net_params[param_id]->gpu_data(),
            net_params[param_id]->mutable_gpu_diff());
      } else if (regularization_type == "L1") {
        caffe_gpu_sign(net_params[param_id]->count(),
            net_params[param_id]->gpu_data(),
            temp_[param_id]->mutable_gpu_data());
        caffe_gpu_axpy(net_params[param_id]->count(),
            local_decay,
            temp_[param_id]->gpu_data(),
            net_params[param_id]->mutable_gpu_diff());
      } else {
        LOG(FATAL) << "Unknown regularization type: " << regularization_type;
      }
    }
#else
    NO_GPU;
#endif
    break;
  }
  default:
    LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
  }
}

template <typename Dtype>
void SGDSolver<Dtype>::ForceRegularize(int param_id) {
  const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
  if(net_params[param_id]->num_axes() < 2){ return ;}
  const vector<float>& net_params_force_mult = this->net_->params_force_mult();
  Dtype force_decay = this->param_.force_decay();
  string force_type = this->param_.force_type();
  string force_decay_type = this->param_.force_decay_type();
  string force_direction = this->param_.force_direction();
  Dtype local_force_decay = force_decay * net_params_force_mult[param_id];
  int num_columns = net_params[param_id]->count(1);
  int num_rows = net_params[param_id]->shape(0);
  switch (Caffe::mode()) {
  case Caffe::CPU: {
    if (local_force_decay) {
		// add force decay
    	temp_2_[param_id]->Reshape(net_params[param_id]->shape());
    	// temporary diff in temp_3_
    	temp_3_[param_id]->Reshape(net_params[param_id]->shape());
    	caffe_set(temp_3_[param_id]->count(), (Dtype)(0.0), temp_3_[param_id]->mutable_cpu_diff());
		for (int i=0; i<num_rows-1; i++){
			for (int j=i+1; j<num_rows; j++){
				// force regularization between every pair of kernels
				const Dtype * kernel0_data = net_params[param_id]->cpu_data() + i * num_columns;
				const Dtype * kernel1_data = net_params[param_id]->cpu_data() + j * num_columns;
				Dtype * kernel0_diff = temp_3_[param_id]->mutable_cpu_diff() + i * num_columns;
				Dtype * kernel1_diff = temp_3_[param_id]->mutable_cpu_diff() + j * num_columns;
				Dtype kernel0_length = caffe_cpu_dot(num_columns, kernel0_data, kernel0_data);
				Dtype kernel1_length = caffe_cpu_dot(num_columns, kernel1_data, kernel1_data);
				kernel0_length = sqrt(kernel0_length);
				kernel1_length = sqrt(kernel1_length);
				if( (kernel0_length<=ZERO_THRESHOLD) ||
					(kernel1_length<=ZERO_THRESHOLD)	){
					//LOG(WARNING) << "Near zero kernels exist! Skip!";
					continue; // too small
				}
				caffe_copy(num_columns,kernel1_data,temp_[param_id]->mutable_cpu_data());
				caffe_cpu_axpby(num_columns, (Dtype)(1.0)/kernel0_length, kernel0_data,
						(Dtype)(-1.0)/kernel1_length, temp_[param_id]->mutable_cpu_data());

				Dtype distance_coef = 1.0;
				if("Degradation" != force_type) {
					distance_coef = caffe_cpu_dot(num_columns,
						temp_[param_id]->cpu_data(),temp_[param_id]->cpu_data());
					distance_coef = sqrt(distance_coef);
				}
				if(distance_coef<=ZERO_THRESHOLD){
					//LOG(WARNING) << "Very close kernels exist! Skip!";
					continue;
				}
				if (force_type == "Gravity") {
					distance_coef = (Dtype)pow(distance_coef,3);
				} else if (force_type == "Linear") {
					distance_coef = (Dtype)pow(distance_coef,2);
				} else if(force_type == "Constant") {//group Lasso
					//distance_coef = distance_coef;
				} else if (force_type == "Degradation"){//normalized (wik-wjk)^2
					distance_coef = 1.0;
				} else {
					LOG(FATAL) << "Unknown force type: " << force_type;
				}

				caffe_cpu_axpby(num_columns, (Dtype)(0.0), temp_2_[param_id]->cpu_data(),
						(Dtype)(1.0/distance_coef), temp_[param_id]->mutable_cpu_data());

				caffe_cpu_axpby(num_columns, (Dtype)(-1.0), temp_[param_id]->cpu_data(),
						(Dtype)(0.0), temp_2_[param_id]->mutable_cpu_data());

				// MUSH PROJECT TO THE TANGENT DIRECTION
				Dtype projection_length = caffe_cpu_dot(num_columns, kernel0_data, temp_[param_id]->cpu_data())/kernel0_length;
				caffe_axpy(num_columns, -projection_length/kernel0_length, kernel0_data,
										 temp_[param_id]->mutable_cpu_data());
				// scale and add gradients to drag kernels together (local_force_decay>0)
				caffe_axpy(num_columns, kernel0_length, // SHOULD WE divide kernel0_length?
						temp_[param_id]->cpu_data(), kernel0_diff);

				caffe_axpy(num_columns, -projection_length/kernel1_length, kernel1_data,
										temp_2_[param_id]->mutable_cpu_data());
				caffe_axpy(num_columns, kernel1_length,
						temp_2_[param_id]->cpu_data(), kernel1_diff);
			}
		}

		// control the direction of regularization gradients
		if("same"==force_direction){
			// zero out regularization gradients with the opposite directions with error gradient
			caffe_cpu_keep_same_direction(temp_3_[param_id]->count(),
					net_params[param_id]->cpu_diff(),
					temp_3_[param_id]->mutable_cpu_diff());
		} else if ("all"!=force_direction){
			LOG(FATAL)<<"Unsupported force_direction = " << force_direction;
		}//esle keep the original

		// decay strength
		Dtype final_force_decay = 0;
		if("fixed"==force_decay_type){
			final_force_decay = local_force_decay;
		} else if ("adaptive"==force_decay_type){
			// adapt the strength to the error gradients
			Dtype error_length = sqrt(
					caffe_cpu_dot(net_params[param_id]->count(),
					net_params[param_id]->cpu_diff(),
					net_params[param_id]->cpu_diff()));
			Dtype regularization_length = sqrt(
					caffe_cpu_dot(temp_3_[param_id]->count(),
					temp_3_[param_id]->cpu_diff(),
					temp_3_[param_id]->cpu_diff()));
			final_force_decay = 0;
			if(fabs(regularization_length)>=(Dtype)1.0e-8) {
				final_force_decay = local_force_decay * (error_length / regularization_length);
			} else {
				LOG(WARNING)<<"Small force regularization. Set to 0!";
			}
		} else {
			LOG(FATAL)<<"Unsupported force_decay_type = " << force_decay_type;
		}
		caffe_axpy(net_params[param_id]->count(),
				final_force_decay,
				temp_3_[param_id]->cpu_diff(),
				net_params[param_id]->mutable_cpu_diff());
    }
    break;
  }
  case Caffe::GPU: {
#ifndef CPU_ONLY
    if (local_force_decay) {
    	//reshape the blobs
    	int n_size = net_params[param_id]->shape(0);
    	int c_size = net_params[param_id]->count()/net_params[param_id]->shape(0);
    	temp_2_[param_id]->Reshape(net_params[param_id]->shape());
    	temp_3_[param_id]->Reshape(net_params[param_id]->shape());
		vector<int> n_shape(1, n_size);
		vector<int> nxn_shape(1, n_size*n_size);
		vector<int> c_shape(1, c_size);
		temp_n_[param_id]->Reshape(n_shape);
		temp_n_2_[param_id]->Reshape(n_shape);
		temp_c_[param_id]->Reshape(c_shape);
		// square of weights in temp_
		caffe_gpu_powx(net_params[param_id]->count(), net_params[param_id]->gpu_data(),
				(Dtype)(2.0), temp_[param_id]->mutable_gpu_data());
		// sum of square of rows
//			caffe_gpu_gemv(CblasNoTrans, n_size, c_size,
//					(Dtype)(1.0), temp_[param_id]->gpu_data(), ones_c_[param_id]->gpu_data(), (Dtype)(0.0),
//					temp_n_[param_id]->mutable_gpu_data());
		caffe_gpu_gemm(CblasNoTrans,CblasNoTrans, n_size, 1, c_size,
				(Dtype)(1.0), temp_[param_id]->gpu_data(), ones_c_[param_id]->gpu_data(), (Dtype)(0.0),
				temp_n_[param_id]->mutable_gpu_data());
		// length of row vector in temp_n_ ***
		caffe_gpu_powx(temp_n_[param_id]->count(), temp_n_[param_id]->gpu_data(),
				(Dtype)(0.5), temp_n_[param_id]->mutable_gpu_data());
		// replicated lengths in temp_
		caffe_gpu_gemm(CblasNoTrans,CblasNoTrans, n_size, c_size, 1,
				(Dtype)(1.0), temp_n_[param_id]->gpu_data(), ones_c_[param_id]->gpu_data(), (Dtype)(0.0),
				temp_[param_id]->mutable_gpu_data());
		// normalized weights in temp_ ***
		caffe_gpu_div_check_zero(net_params[param_id]->count(), net_params[param_id]->gpu_data(),
				temp_[param_id]->gpu_data(), temp_[param_id]->mutable_gpu_data());

    	if (force_type == "Constant") {
    		//A in temp_nxn_ and temp_nxn_2_
    		temp_nxn_[param_id]->Reshape(nxn_shape);
    		temp_nxn_2_[param_id]->Reshape(nxn_shape);
    		caffe_gpu_gemm(CblasNoTrans,CblasNoTrans, n_size, n_size, 1,
					(Dtype)(1.0), ones_n_[param_id]->gpu_data(), ones_n_[param_id]->gpu_data(), (Dtype)(0.0),
					temp_nxn_[param_id]->mutable_gpu_data());
    		caffe_copy(temp_nxn_[param_id]->count(),temp_nxn_[param_id]->gpu_data(),temp_nxn_2_[param_id]->mutable_gpu_data());
    		// A -2 * w_norm * w_norm^T in temp_nxn_
    		caffe_gpu_gemm(CblasNoTrans,CblasTrans, n_size, n_size, c_size,
					(Dtype)(-2.0), temp_[param_id]->gpu_data(), temp_[param_id]->gpu_data(), (Dtype)(1.0),
					temp_nxn_[param_id]->mutable_gpu_data());
    		// D in temp_nxn_
    		caffe_gpu_geam(CblasNoTrans,CblasTrans, n_size, n_size,
    				(Dtype)(1.0), temp_nxn_[param_id]->gpu_data(),
    				(Dtype)(1.0), temp_nxn_2_[param_id]->gpu_data(),
    				temp_nxn_[param_id]->mutable_gpu_data());
    		caffe_gpu_powx_check_negative(temp_nxn_[param_id]->count(), temp_nxn_[param_id]->gpu_data(),
					(Dtype)(0.5), temp_nxn_[param_id]->mutable_gpu_data());
			// Dr = 1./D in temp_nxn_
			caffe_gpu_gemm(CblasNoTrans,CblasNoTrans, n_size, n_size, 1,
					(Dtype)(1.0), ones_n_[param_id]->gpu_data(), ones_n_[param_id]->gpu_data(), (Dtype)(0.0),
					temp_nxn_2_[param_id]->mutable_gpu_data());
			caffe_gpu_div_check_zero(temp_nxn_[param_id]->count(), temp_nxn_2_[param_id]->gpu_data(),
					temp_nxn_[param_id]->gpu_data(), temp_nxn_[param_id]->mutable_gpu_data());
			// sum of forces in temp_2_
			caffe_gpu_gemm(CblasNoTrans,CblasNoTrans, n_size, 1, n_size,
					(Dtype)(1.0), temp_nxn_[param_id]->gpu_data(), ones_n_[param_id]->gpu_data(), (Dtype)(0.0),
					temp_n_2_[param_id]->mutable_gpu_data());
			caffe_gpu_gemm(CblasNoTrans,CblasNoTrans, n_size, c_size, 1,
					(Dtype)(1.0), temp_n_2_[param_id]->gpu_data(), ones_c_[param_id]->gpu_data(), (Dtype)(0.0),
					temp_2_[param_id]->mutable_gpu_data());
			caffe_gpu_mul(temp_[param_id]->count(), temp_[param_id]->gpu_data(),
					temp_2_[param_id]->gpu_data(), temp_2_[param_id]->mutable_gpu_data());
			caffe_gpu_gemm(CblasNoTrans,CblasNoTrans, n_size, c_size, n_size,
					(Dtype)(-1.0), temp_nxn_[param_id]->gpu_data(), temp_[param_id]->gpu_data(), (Dtype)(1.0),
					temp_2_[param_id]->mutable_gpu_data());

		} else if (force_type == "Degradation"){//normalized (wik-wjk)^2
			// -sum of normalized weights along columns in temp_c_
			caffe_gpu_gemm(CblasNoTrans,CblasNoTrans, 1, c_size, n_size,
					(Dtype)(-1.0), ones_n_[param_id]->gpu_data(), temp_[param_id]->gpu_data(),  (Dtype)(0.0),
					temp_c_[param_id]->mutable_gpu_data());
			// replicated -sum of normalized weights in temp_2_
			caffe_gpu_gemm(CblasNoTrans,CblasNoTrans, n_size, c_size, 1,
					(Dtype)(1.0), ones_n_[param_id]->gpu_data(), temp_c_[param_id]->gpu_data(),  (Dtype)(0.0),
					temp_2_[param_id]->mutable_gpu_data());
			// sum of forces in temp_2_
			caffe_gpu_axpy(temp_[param_id]->count(), (Dtype)(n_size), temp_[param_id]->gpu_data(),
					temp_2_[param_id]->mutable_gpu_data());
		} else if(force_type == "Gravity" || force_type == "Linear"){
			NOT_IMPLEMENTED;
		} else {
			LOG(FATAL) << "Unknown force type: " << force_type;
		}
		// the lengths of projection in temp_n_2_
		caffe_gpu_mul(temp_[param_id]->count(), temp_[param_id]->gpu_data(),
					temp_2_[param_id]->gpu_data(), temp_3_[param_id]->mutable_gpu_data());
		caffe_gpu_gemv(CblasNoTrans, n_size, c_size,
					(Dtype)(1.0), temp_3_[param_id]->gpu_data(), ones_c_[param_id]->gpu_data(), (Dtype)(0.0),
					temp_n_2_[param_id]->mutable_gpu_data());
		// force regularization in temp_3_
		caffe_gpu_gemm(CblasNoTrans,CblasNoTrans, n_size, c_size, 1,
					(Dtype)(1.0), temp_n_2_[param_id]->gpu_data(), ones_c_[param_id]->gpu_data(),  (Dtype)(0.0),
					temp_3_[param_id]->mutable_gpu_data());
		caffe_gpu_mul(temp_3_[param_id]->count(), temp_3_[param_id]->gpu_data(),
					temp_[param_id]->gpu_data(), temp_3_[param_id]->mutable_gpu_data());
		caffe_gpu_sub(temp_3_[param_id]->count(), temp_2_[param_id]->gpu_data(),
					temp_3_[param_id]->gpu_data(), temp_3_[param_id]->mutable_gpu_data());
		// scale and update diff
		caffe_gpu_gemm(CblasNoTrans,CblasNoTrans, n_size, c_size, 1,
					(Dtype)(1.0), temp_n_[param_id]->gpu_data(), ones_c_[param_id]->gpu_data(), (Dtype)(0.0),
					temp_[param_id]->mutable_gpu_data());
		caffe_gpu_mul(temp_[param_id]->count(), temp_[param_id]->gpu_data(),
					temp_3_[param_id]->gpu_data(), temp_[param_id]->mutable_gpu_data());
		//caffe_gpu_div_check_zero(temp_[param_id]->count(), temp_3_[param_id]->gpu_data(),
		//			temp_[param_id]->gpu_data(), temp_[param_id]->mutable_gpu_data());

		// control the direction of regularization gradients
		if("same"==force_direction){
			// zero out regularization gradients with the opposite directions with error gradient
			caffe_gpu_keep_same_direction(temp_[param_id]->count(),
					net_params[param_id]->gpu_diff(),
					temp_[param_id]->mutable_gpu_data());//diff is stored in temp_.data
		} else if ("all"!=force_direction){
			LOG(FATAL)<<"Unsupported force_direction = " << force_direction;
		}//esle keep the original

		// decay strength
		Dtype final_force_decay = 0;
		if("fixed"==force_decay_type){
			final_force_decay = local_force_decay;
		} else if ("adaptive"==force_decay_type){
			// adapt the strength to the error gradients
			Dtype error_length = 0;
			caffe_gpu_dot(net_params[param_id]->count(),
					net_params[param_id]->gpu_diff(),
					net_params[param_id]->gpu_diff(),
					&error_length);
			error_length = sqrt(error_length);
			Dtype regularization_length = 0;
			caffe_gpu_dot(temp_[param_id]->count(),
					temp_[param_id]->gpu_data(),
					temp_[param_id]->gpu_data(),
					&regularization_length);
			regularization_length = sqrt(regularization_length);
			final_force_decay = 0;
			if(fabs(regularization_length)>=(Dtype)1.0e-8) {
				final_force_decay = local_force_decay * (error_length / regularization_length);
			} else {
				LOG(WARNING)<<"Small force regularization. Set to 0!";
			}
		} else {
			LOG(FATAL)<<"Unsupported force_decay_type = " << force_decay_type;
		}

		caffe_gpu_axpy(net_params[param_id]->count(),
				final_force_decay,
				temp_[param_id]->gpu_data(),
				net_params[param_id]->mutable_gpu_diff());
    }
#else
    NO_GPU;
#endif
    break;
  }
  default:
    LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
  }
}

#ifndef CPU_ONLY
template <typename Dtype>
void sgd_update_gpu(int N, Dtype* g, Dtype* h, Dtype momentum,
    Dtype local_rate);
#endif

template <typename Dtype>
void SGDSolver<Dtype>::ComputeUpdateValue(int param_id, Dtype rate) {
  const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
  const vector<float>& net_params_lr = this->net_->params_lr();
  Dtype momentum = this->param_.momentum();
  Dtype local_rate = rate * net_params_lr[param_id];
  // Compute the update to history, then copy it to the parameter diff.
  switch (Caffe::mode()) {
  case Caffe::CPU: {
    caffe_cpu_axpby(net_params[param_id]->count(), local_rate,
              net_params[param_id]->cpu_diff(), momentum,
              history_[param_id]->mutable_cpu_data());
    caffe_copy(net_params[param_id]->count(),
        history_[param_id]->cpu_data(),
        net_params[param_id]->mutable_cpu_diff());
    break;
  }
  case Caffe::GPU: {
#ifndef CPU_ONLY
    sgd_update_gpu(net_params[param_id]->count(),
        net_params[param_id]->mutable_gpu_diff(),
        history_[param_id]->mutable_gpu_data(),
        momentum, local_rate);
#else
    NO_GPU;
#endif
    break;
  }
  default:
    LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
  }
}

template <typename Dtype>
void SGDSolver<Dtype>::SnapshotSolverState(const string& model_filename) {
  switch (this->param_.snapshot_format()) {
    case caffe::SolverParameter_SnapshotFormat_BINARYPROTO:
      SnapshotSolverStateToBinaryProto(model_filename);
      break;
    case caffe::SolverParameter_SnapshotFormat_HDF5:
      SnapshotSolverStateToHDF5(model_filename);
      break;
    default:
      LOG(FATAL) << "Unsupported snapshot format.";
  }
}

template <typename Dtype>
void SGDSolver<Dtype>::SnapshotSolverStateToBinaryProto(
    const string& model_filename) {
  SolverState state;
  state.set_iter(this->iter_);
  state.set_learned_net(model_filename);
  state.set_current_step(this->current_step_);
  state.clear_history();
  for (int i = 0; i < history_.size(); ++i) {
    // Add history
    BlobProto* history_blob = state.add_history();
    history_[i]->ToProto(history_blob);
  }
  string snapshot_filename = Solver<Dtype>::SnapshotFilename(".solverstate");
  LOG(INFO)
    << "Snapshotting solver state to binary proto file " << snapshot_filename;
  WriteProtoToBinaryFile(state, snapshot_filename.c_str());
}

template <typename Dtype>
void SGDSolver<Dtype>::SnapshotSolverStateToHDF5(
    const string& model_filename) {
  string snapshot_filename =
      Solver<Dtype>::SnapshotFilename(".solverstate.h5");
  LOG(INFO) << "Snapshotting solver state to HDF5 file " << snapshot_filename;
  hid_t file_hid = H5Fcreate(snapshot_filename.c_str(), H5F_ACC_TRUNC,
      H5P_DEFAULT, H5P_DEFAULT);
  CHECK_GE(file_hid, 0)
      << "Couldn't open " << snapshot_filename << " to save solver state.";
  hdf5_save_int(file_hid, "iter", this->iter_);
  hdf5_save_string(file_hid, "learned_net", model_filename);
  hdf5_save_int(file_hid, "current_step", this->current_step_);
  hid_t history_hid = H5Gcreate2(file_hid, "history", H5P_DEFAULT, H5P_DEFAULT,
      H5P_DEFAULT);
  CHECK_GE(history_hid, 0)
      << "Error saving solver state to " << snapshot_filename << ".";
  for (int i = 0; i < history_.size(); ++i) {
    ostringstream oss;
    oss << i;
    hdf5_save_nd_dataset<Dtype>(history_hid, oss.str(), *history_[i]);
  }
  H5Gclose(history_hid);
  H5Fclose(file_hid);
}

template <typename Dtype>
void SGDSolver<Dtype>::RestoreSolverStateFromBinaryProto(
    const string& state_file) {
  SolverState state;
  ReadProtoFromBinaryFile(state_file, &state);
  this->iter_ = state.iter();
  if (state.has_learned_net()) {
    NetParameter net_param;
    ReadNetParamsFromBinaryFileOrDie(state.learned_net().c_str(), &net_param);
    this->net_->CopyTrainedLayersFrom(net_param);
  }
  this->current_step_ = state.current_step();
  CHECK_EQ(state.history_size(), history_.size())
      << "Incorrect length of history blobs.";
  LOG(INFO) << "SGDSolver: restoring history";
  for (int i = 0; i < history_.size(); ++i) {
    history_[i]->FromProto(state.history(i));
  }
}

template <typename Dtype>
void SGDSolver<Dtype>::RestoreSolverStateFromHDF5(const string& state_file) {
  hid_t file_hid = H5Fopen(state_file.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  CHECK_GE(file_hid, 0) << "Couldn't open solver state file " << state_file;
  this->iter_ = hdf5_load_int(file_hid, "iter");
  if (H5LTfind_dataset(file_hid, "learned_net")) {
    string learned_net = hdf5_load_string(file_hid, "learned_net");
    this->net_->CopyTrainedLayersFrom(learned_net);
  }
  this->current_step_ = hdf5_load_int(file_hid, "current_step");
  hid_t history_hid = H5Gopen2(file_hid, "history", H5P_DEFAULT);
  CHECK_GE(history_hid, 0) << "Error reading history from " << state_file;
  int state_history_size = hdf5_get_num_links(history_hid);
  CHECK_EQ(state_history_size, history_.size())
      << "Incorrect length of history blobs.";
  for (int i = 0; i < history_.size(); ++i) {
    ostringstream oss;
    oss << i;
    hdf5_load_nd_dataset<Dtype>(history_hid, oss.str().c_str(), 0,
                                kMaxBlobAxes, history_[i].get());
  }
  H5Gclose(history_hid);
  H5Fclose(file_hid);
}

INSTANTIATE_CLASS(SGDSolver);
REGISTER_SOLVER_CLASS(SGD);

}  // namespace caffe
