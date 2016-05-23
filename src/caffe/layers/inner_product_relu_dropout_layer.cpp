#include <vector>

#include <cfloat>
#include <omp.h>

#include "caffe/filler.hpp"
#include "caffe/layers/inner_product_relu_dropout_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/spgemm.hpp"
#include "CSR.hpp"
#include "reordering/BFSBipartite.hpp"

std::map<std::string, CSR> layer2weight;
std::map<std::string, float *> layer2bottom;
std::map<std::string, float *> layer2bias;
extern std::map<std::string, unsigned long long> total_conv_cycles;
extern std::map<std::string, double> total_conv_flops;
extern int total_files;

double get_cpu_freq();

namespace caffe {

template<typename Dtype>
InnerProductReLUDropoutLayer<Dtype>::InnerProductReLUDropoutLayer(const LayerParameter& param) :
    Layer<Dtype>(param),
    bottom_values_(NULL), bottom_j_(NULL), bottom_i_(NULL),
    top_values_(NULL), top_j_(NULL), top_i_(NULL),
    weight_values_(NULL), weight_j_(NULL), weight_i_(NULL),
    bottom_transposed_(NULL), spgemm_buf_(NULL),
    weight_values_blocked_(NULL), weight_j_blocked_(NULL), weight_i_blocked_(NULL)
{

}

template<typename Dtype>
InnerProductReLUDropoutLayer<Dtype>::~InnerProductReLUDropoutLayer()
{
  free(bottom_values_);
  free(bottom_j_);
  free(bottom_i_);
  free(bottom_transposed_);
  free(spgemm_buf_);

  free(top_values_);
  free(top_j_);
  free(top_i_);

  free(weight_values_);
  free(weight_j_);
  free(weight_i_);

  free(weight_values_blocked_);
  free(weight_j_blocked_);
  free(weight_i_blocked_);
}

enum
{
  SPGEMM_CSR,
  SPGEMM_CSC,
  SPMDM_CSR,
  SPMDM_CSC,
  GEMM,
};

static int method = SPMDM_CSR;

template<>
void InnerProductReLUDropoutLayer<double>::WeightAlign(){
  NOT_IMPLEMENTED;
}

static int col_block_size = 128;

template<>
void InnerProductReLUDropoutLayer<float>::WeightAlign(){
	const LayerParameter& layerparam = this->layer_param();
	LOG(INFO)<<"layer\t"<<layerparam.name()<<"\t"<<"has sparsity of "<< this->blobs_[0]->GetSparsity() << " transpose " << transpose_;
//	this->blobs_[0]->WriteToNistMMIO(layerparam.name()+".weight");

	posix_memalign((void **)&weight_i_, 4096, sizeof(int)*(std::max(K_, N_) + 1));
	posix_memalign((void **)&weight_j_, 4096, sizeof(int)*K_*N_);
	posix_memalign((void **)&weight_values_, 4096, sizeof(float)*K_*N_);

	CSR csr;
	csr.values = weight_values_;
	csr.rowptr = weight_i_;
	csr.colidx = weight_j_;

  MKL_INT job[] = {
      0 /*dense->CSR*/,
      0 /*0-based indexing in dense matrix */,
      0 /*0-based CSR*/,
      2 /* whole matrix*/,
      K_*N_, /* nzmax */
      1 /* generate a, i, and j */
  };
  MKL_INT info;
  if (SPMDM_CSR == method && !transpose_) {
    int m = transpose_ ? K_ : N_;
    int n = transpose_ ? N_ : K_;

    int ncolblocks = K_/col_block_size;
    posix_memalign((void **)&weight_i_blocked_, 4096, sizeof(int)*(N_*ncolblocks + 1));
    posix_memalign((void **)&weight_j_blocked_, 4096, sizeof(int)*K_*N_);
    posix_memalign((void **)&weight_values_blocked_, 4096, sizeof(float)*K_*N_);

    weight_i_blocked_[0] = 0;
    int nnz = 0;
    for (int cb = 0; cb < ncolblocks; ++cb) {
      for (int i = 0; i < N_; ++i) {
        for (int j = cb*col_block_size; j < (cb + 1)*col_block_size; ++j) {
          float v = this->blobs_[0]->mutable_cpu_data()[i*K_ + j];
          if (v != 0) {
            weight_j_blocked_[nnz] = M_*j;
            weight_values_blocked_[nnz] = v;
            ++nnz;
          }
        }
        weight_i_blocked_[cb*N_ + i + 1] = nnz;
      }
    }
  }
  else if (SPGEMM_CSR == method && transpose_ || SPGEMM_CSC == method && !transpose_) {
	  int m = transpose_ ? K_ : N_;
	  int n = transpose_ ? N_ : K_;
	  mkl_sdnscsr(job, &m, &n, this->blobs_[0]->mutable_cpu_data(), &n, weight_values_, weight_j_, weight_i_, &info);
	  if(info) {
	    LOG(FATAL)<<"The routine is interrupted processing the "<<
	        info<<"-th row "
	        <<"because there is no space in the arrays acsr and ja according to the value nzmax.";
	  }
	  csr.m = m;
	  csr.n = n;

    SpMP::CSR A(m, n, weight_i_[m]);
    for (int i = 0; i <= m; ++i) {
      A.rowptr[i] = weight_i_[i];
    }
    for (int i = 0; i < weight_i_[m]; ++i) {
      A.colidx[i] = weight_j_[i];
    }

    SpMP::CSR *AT = A.transpose();
    int *rowPerm = new int[m], *rowInversePerm = new int[m];
    int *colPerm = new int[n], *colInversePerm = new int[n];
    bfsBipartite(A, *AT, rowPerm, rowInversePerm, colPerm, colInversePerm);
    FREE(A.diagptr);
    SpMP::CSR *AReordered = A.permute(colPerm, rowInversePerm);
    SpMP::CSR *ATReordered = AReordered->transpose();

    LOG(INFO) << "Average width of " << m << " x " << n << " matrix = " << A.getAverageWidth() << " " << AT->getAverageWidth();
    LOG(INFO) << "Average width after reordering = " << AReordered->getAverageWidth() << " " << ATReordered->getAverageWidth();

    delete[] rowPerm;
    delete[] rowInversePerm;
    delete[] colPerm;
    delete[] colInversePerm;
    delete AT;
    delete AReordered;
    delete ATReordered;
	}
	else if (SPGEMM_CSR == method && !transpose_ || SPGEMM_CSC == method && transpose_) {
	  int m = transpose_ ? K_ : N_;
	  int n = transpose_ ? N_ : K_;

    float *weight_transposed;
    posix_memalign((void **)&weight_transposed, 4096, sizeof(float)*K_*N_);
    mkl_somatcopy('R', 'T', m, n, 1, this->blobs_[0]->mutable_cpu_data(), n, weight_transposed, m);
    mkl_sdnscsr(job, &n, &m, weight_transposed, &m, weight_values_, weight_j_, weight_i_, &info);
    if(info) {
      LOG(FATAL)<<"The routine is interrupted processing the "<<
          info<<"-th row "
          <<"because there is no space in the arrays acsr and ja according to the value nzmax.";
    }
    free(weight_transposed);
    csr.m = n;
    csr.n = m;

    SpMP::CSR A(n, m, weight_i_[n]);
    for (int i = 0; i <= n; ++i) {
      A.rowptr[i] = weight_i_[i];
    }
    for (int i = 0; i < weight_i_[n]; ++i) {
      A.colidx[i] = weight_j_[i];
    }

    SpMP::CSR *AT = A.transpose();
    int *rowPerm = new int[n], *rowInversePerm = new int[n];
    int *colPerm = new int[m], *colInversePerm = new int[m];
    bfsBipartite(A, *AT, rowPerm, rowInversePerm, colPerm, colInversePerm);
    FREE(A.diagptr);
    SpMP::CSR *AReordered = A.permute(colPerm, rowInversePerm);
    SpMP::CSR *ATReordered = AReordered->transpose();

    LOG(INFO) << "Average width of " << n << " x " << m << " matrix = " << A.getAverageWidth() << " " << AT->getAverageWidth();
    LOG(INFO) << "Average width after reordering = " << AReordered->getAverageWidth() << " " << ATReordered->getAverageWidth();

    delete[] rowPerm;
    delete[] rowInversePerm;
    delete[] colPerm;
    delete[] colInversePerm;
    delete AT;
    delete AReordered;
    delete ATReordered;
	}

  layer2weight[layerparam.name()] = csr;
  layer2bias[layerparam.name()] = this->blobs_[1]->mutable_cpu_data();

  posix_memalign((void **)&spgemm_buf_, 4096, sizeof(float)*omp_get_max_threads()*(SPGEMM_CSR == method ? N_ : M_));

  posix_memalign((void **)&bottom_i_, 4096, sizeof(int)*(std::max(M_, K_) + 1));
  posix_memalign((void **)&bottom_j_, 4096, sizeof(int)*M_*K_);
  posix_memalign((void **)&bottom_values_, 4096, sizeof(float)*M_*K_);

  posix_memalign((void **)&top_i_, 4096, sizeof(int)*(std::max(M_, N_) + 1));
  posix_memalign((void **)&top_j_, 4096, sizeof(int)*M_*N_);
  posix_memalign((void **)&top_values_, 4096, sizeof(float)*M_*N_);

  posix_memalign((void **)&bottom_transposed_, 4096, sizeof(int)*M_*std::max(K_, N_));

	//disconnect connections
	if( layerparam.connectivity_mode() == caffe::LayerParameter_ConnectivityMode_DISCONNECTED_ELTWISE ){
		LOG(INFO)<<"all zero weights of "<<layerparam.name()<<" are frozen";
		this->blobs_[0]->Disconnect(Blob<float>::ELTWISE);
	}else if(layerparam.connectivity_mode() == caffe::LayerParameter_ConnectivityMode_DISCONNECTED_GRPWISE){
		LOG(INFO)<<"weights lying in all-zero groups of "<<layerparam.name()<<" are frozen";
		this->blobs_[0]->Disconnect(Blob<float>::GRPWISE);
	}
}

template <typename Dtype>
void InnerProductReLUDropoutLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int num_output = this->layer_param_.inner_product_param().num_output();
  bias_term_ = this->layer_param_.inner_product_param().bias_term();
  transpose_ = this->layer_param_.inner_product_param().transpose();
    // if true, weight is in row-major, otherwise it's in col-major
  N_ = num_output;
  const int axis = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.inner_product_param().axis());
  // Dimensions starting from "axis" are "flattened" into a single
  // length K_ vector. For example, if bottom[0]'s shape is (N, C, H, W),
  // and axis == 1, N inner products with dimension CHW are performed.
  K_ = bottom[0]->count(axis);
  // Check if we need to set up the weights
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    if (bias_term_) {
      this->blobs_.resize(2);
    } else {
      this->blobs_.resize(1);
    }
    // Initialize the weights
    vector<int> weight_shape(2);
    if (transpose_) {
      weight_shape[0] = K_;
      weight_shape[1] = N_;
    } else {
      weight_shape[0] = N_;
      weight_shape[1] = K_;
    }
    this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
    // fill the weights
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.inner_product_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
    // If necessary, intiialize and fill the bias term
    if (bias_term_) {
      vector<int> bias_shape(1, N_);
      this->blobs_[1].reset(new Blob<Dtype>(bias_shape));
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
          this->layer_param_.inner_product_param().bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());
    }
  }  // parameter initialization
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void InnerProductReLUDropoutLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Figure out the dimensions
  const int axis = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.inner_product_param().axis());
  const int new_K = bottom[0]->count(axis);
  CHECK_EQ(K_, new_K)
      << "Input size incompatible with inner product parameters.";
  // The first "axis" dimensions are independent inner products; the total
  // number of these is M_, the product over these dimensions.
  M_ = bottom[0]->count(0, axis);
  // The top shape will be the bottom shape with the flattened axes dropped,
  // and replaced by a single axis with dimension num_output (N_).
  vector<int> top_shape = bottom[0]->shape();
  top_shape.resize(axis + 1);
  top_shape[axis] = N_;
  top[0]->Reshape(top_shape);
  // Set up the bias multiplier
  if (bias_term_) {
    vector<int> bias_shape(1, M_);
    bias_multiplier_.Reshape(bias_shape);
    caffe_set(M_, Dtype(1), bias_multiplier_.mutable_cpu_data());
  }
}

template<>
void InnerProductReLUDropoutLayer<double>::Forward_cpu(const vector<Blob<double>*>& bottom,
    const vector<Blob<double>*>& top) {
  NOT_IMPLEMENTED;
}

template<>
void InnerProductReLUDropoutLayer<float>::Forward_cpu(const vector<Blob<float>*>& bottom,
    const vector<Blob<float>*>& top) {
  float* bottom_data = bottom[0]->mutable_cpu_data();
  float* top_data = top[0]->mutable_cpu_data();
  float* weight = this->blobs_[0]->mutable_cpu_data();

  layer2bottom[this->layer_param().name()] = bottom_data;

  bool PRINT_FEATURE_SPARSITY = false;
  if (PRINT_FEATURE_SPARSITY) {
    int cnt = 0;
#pragma omp parallel for reduction(+:cnt)
    for (int i = 0; i < M_*K_; ++i) {
      if (bottom_data[i] == 0) ++cnt;
    }
    LOG(INFO) << this->layer_param_.name() << " M " << M_ << " K " << K_ << " N " << N_ << " sparsity " << (double)cnt/(M_*K_);
  }

  MKL_INT job[] = {
      0 /*dense->CSR*/,
      0 /*0-based indexing in dense matrix */,
      0 /*0-based CSR*/,
      2 /* whole matrix*/,
      M_*K_, /* nzmax */
      1 /* generate a, i, and j */
  };
  MKL_INT info;

  if (SPMDM_CSR == method) {
    mkl_somatcopy('R', 'T', M_, K_, 1, bottom_data, K_, bottom_transposed_, M_);

    int ncolblocks = K_/col_block_size;
    double t = omp_get_wtime();
    csrmm_fused(
        weight_values_blocked_, weight_j_blocked_, weight_i_blocked_,
        bottom_transposed_,
        top_data,
        N_, M_, K_,
        this->blobs_[1]->cpu_data(),
        col_block_size);
    t = omp_get_wtime() - t;
    LOG(INFO) << "csrmm takes " << t << " effective GF/s " << 2.*K_*N_*M_/t/1e9 << " real GF/s " << 2.*weight_i_blocked_[ncolblocks*N_]*M_/t/1e9;

    memcpy(bottom_transposed_, top_data, sizeof(float)*M_*N_);
    mkl_somatcopy('R', 'T', N_, M_, 1, bottom_transposed_, M_, top_data, N_);

    std::string name(this->layer_param_.name());
    if (total_conv_cycles.find(name) == total_conv_cycles.end()) {
      total_conv_cycles[name] = 0;
      total_conv_flops[name] = 0;
    }
    total_conv_cycles[name] += t*get_cpu_freq();
    total_conv_flops[name] += 2.*M_*K_*N_;
    total_files += M_;
  }
  else if (SPGEMM_CSR == method) {
    assert(this->layer_param_.relu_param().negative_slope() == 0);

    mkl_sdnscsr(job, &M_, &K_, bottom_data, &K_, bottom_values_, bottom_j_, bottom_i_, &info);
    if(info) {
      LOG(FATAL)<<"The routine is interrupted processing the "<<
          info<<"-th row "
          <<"because there is no space in the arrays acsr and ja according to the value nzmax.";
    }

    int flops = spgemm_flops(
        bottom_values_, bottom_j_, bottom_i_,
        weight_values_, weight_j_, weight_i_,
        M_);
    LOG(INFO) << "flop-sparsity " << 1 - (double)flops/(2.*M_*N_*K_);

//    return; // fused with fc8

    int nnz;
    double t = omp_get_wtime();
    csrmultd(
        bottom_values_, bottom_j_, bottom_i_,
        weight_values_, weight_j_, weight_i_,
        this->blobs_[1]->cpu_data(),
        top_data,
//        top_values_, top_j_, top_i_, &nnz,
        M_, N_/*, spgemm_buf_*/);

    t = omp_get_wtime() - t;
    LOG(INFO) << "spgemm takes " << t << " GF/s= " << (double)flops/t/1e9 << " nnz-sparsity " << 1 - (double)nnz/(M_*N_);

//#pragma omp parallel for
//    for (int i = 0; i < M_; ++i) {
//      for (int j = 0; j < N_; ++j) {
//        top_data[i*N_ + j] = 0;
//      }
//      for (int j = top_i_[i]; j < top_i_[i + 1]; ++j) {
//        top_data[i*N_ + top_j_[j]] = top_values_[j];
//      }
//    }
  }
  else if (SPGEMM_CSC == method) {
    mkl_somatcopy('R', 'T', M_, K_, 1, bottom_data, K_, bottom_transposed_, M_);
    mkl_sdnscsr(job, &K_, &M_, bottom_transposed_, &M_, bottom_values_, bottom_j_, bottom_i_, &info);
    if(info) {
      LOG(FATAL)<<"The routine is interrupted processing the "<<
          info<<"-th row "
          <<"because there is no space in the arrays acsr and ja according to the value nzmax.";
    }

    int flops = spgemm_flops(
        weight_values_, weight_j_, weight_i_,
        bottom_values_, bottom_j_, bottom_i_,
        N_);

    int nnz;
    double t = omp_get_wtime();
    csrmultd_csc(
        weight_values_, weight_j_, weight_i_,
        bottom_values_, bottom_j_, bottom_i_,
        this->blobs_[1]->cpu_data(), // FIXME - should access bias differently
        bottom_transposed_,
        N_, M_);
    LOG(INFO) << "spgemm takes " << omp_get_wtime() - t << " nnz-sparsity " << 1 - (double)nnz/(M_*N_) << " flop-sparsity " << 1 - (double)flops/(2.*M_*N_*K_);

    mkl_somatcopy('R', 'T', N_, M_, 1, bottom_transposed_, M_, top_data, N_);

//#pragma omp parallel for
//    for (int i = 0; i < N_; ++i) {
//      for (int j = 0; j < M_; ++j) {
//        top_data[j*N_ + i] = 0;
//      }
//      for (int j = top_i_[i]; j < top_i_[i + 1]; ++j) {
//        top_data[top_j_[j]*N_ + i] = top_values_[j];
//      }
//    }
  }
  else {
    assert(GEMM == method);
    caffe_cpu_gemm<float>(CblasNoTrans, transpose_ ? CblasNoTrans : CblasTrans,
        M_, N_, K_, (float)1.,
        bottom_data, weight, (float)0., top_data);
    if (bias_term_) {
      // JSP: common path for AlexNet
      caffe_cpu_gemm<float>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (float)1.,
          bias_multiplier_.cpu_data(),
          this->blobs_[1]->cpu_data(), (float)1., top_data);
    }

    const int count = top[0]->count();
    float negative_slope = this->layer_param_.relu_param().negative_slope();
    if (0 == negative_slope) {
#pragma omp parallel for
      for (int i = 0; i < count; ++i) {
        top_data[i] = std::max(top_data[i], float(0));
      }
    }
    else {
#pragma omp parallel for
      for (int i = 0; i < count; ++i) {
        top_data[i] = std::max(top_data[i], float(0))
            + negative_slope * std::min(top_data[i], float(0));
      }
    }
  }
}

template <typename Dtype>
void InnerProductReLUDropoutLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (this->param_propagate_down_[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    const Dtype* bottom_data = bottom[0]->cpu_data();
    // Gradient with respect to weight
    if (transpose_) {
      caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
          K_, N_, M_,
          (Dtype)1., bottom_data, top_diff,
          (Dtype)1., this->blobs_[0]->mutable_cpu_diff());
    } else {
      caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
          N_, K_, M_,
          (Dtype)1., top_diff, bottom_data,
          (Dtype)1., this->blobs_[0]->mutable_cpu_diff());
    }
  }
  if (bias_term_ && this->param_propagate_down_[1]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    // Gradient with respect to bias
    caffe_cpu_gemv<Dtype>(CblasTrans, M_, N_, (Dtype)1., top_diff,
        bias_multiplier_.cpu_data(), (Dtype)1.,
        this->blobs_[1]->mutable_cpu_diff());
  }
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    // Gradient with respect to bottom data
    if (transpose_) {
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
          M_, K_, N_,
          (Dtype)1., top_diff, this->blobs_[0]->cpu_data(),
          (Dtype)0., bottom[0]->mutable_cpu_diff());
    } else {
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
          M_, K_, N_,
          (Dtype)1., top_diff, this->blobs_[0]->cpu_data(),
          (Dtype)0., bottom[0]->mutable_cpu_diff());
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(InnerProductReLUDropoutLayer);
#endif

INSTANTIATE_CLASS(InnerProductReLUDropoutLayer);
REGISTER_LAYER_CLASS(InnerProductReLUDropout);

}  // namespace caffe
