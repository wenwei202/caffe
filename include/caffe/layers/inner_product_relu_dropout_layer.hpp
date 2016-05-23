#ifndef CAFFE_INNER_PRODUCT_RELU_DROPOUT_LAYER_HPP_
#define CAFFE_INNER_PRODUCT_RELU_DROPOUT_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Also known as a "fully-connected" layer, computes an inner product
 *        with a set of learned weights, and (optionally) adds biases.
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
template <typename Dtype>
class InnerProductReLUDropoutLayer : public Layer<Dtype> {
 public:
  explicit InnerProductReLUDropoutLayer(const LayerParameter& param);
  ~InnerProductReLUDropoutLayer();
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "InnerProductReLUDropout"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }
  virtual void WeightAlign();

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int M_;
  int K_;
  int N_;
  bool bias_term_;
  Blob<Dtype> bias_multiplier_;
  bool transpose_;  ///< if true, assume transposed weights

  Dtype *bottom_values_;
  int *bottom_j_;
  int *bottom_i_;

  Dtype *top_values_;
  int *top_j_;
  int *top_i_;

  Dtype *weight_values_;
  int *weight_j_;
  int *weight_i_;

  Dtype *weight_values_blocked_;
  int *weight_j_blocked_;
  int *weight_i_blocked_;

  Dtype *bottom_transposed_;
  Dtype *spgemm_buf_;
};

}  // namespace caffe

#endif  // CAFFE_INNER_PRODUCT_RELU_DROPOUT_LAYER_HPP_
