#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

using std::min;
using std::max;

template <typename Dtype>
void ExpandLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  ExpandParameter expand_param = this->layer_param_.expand_param();
  CHECK(!expand_param.has_expand_size() !=
    !(expand_param.has_expand_w() && expand_param.has_expand_h()))
    << "Expand size is expand_size OR expand_width and expand_height; not both.";
  CHECK (expand_param.has_expand_size() ||
    (expand_param.has_expand_w() && expand_param.has_expand_h()))
    << "For non-square output shape, both expand_width and expand_height are required.";
  CHECK (bottom[0]->width() == 1 && bottom[0]->height() == 1)
    << "The bottom data's width and height must both be 1.";

  if (expand_param.has_expand_size()) {
      expand_h_ = expand_w_ = expand_param.expand_size();
  } else {
    expand_h_ = expand_param.expand_h();
    expand_w_ = expand_param.expand_w();
  }
  
  CHECK_GT(expand_h_, 0) << "Expand dimensions cannot be zero.";
  CHECK_GT(expand_w_, 0) << "Expand dimensions cannot be zero.";
}

template <typename Dtype>
void ExpandLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  channels_ = bottom[0]->channels();
  top[0]->Reshape(bottom[0]->num(), channels_, expand_h_, expand_w_);
}

template <typename Dtype>
void ExpandLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  Dtype* top_data = top[0]->mutable_cpu_data();
  const Dtype* bottom_data = bottom[0]->cpu_data();
  int spatial_dim = expand_w_ * expand_h_;
  for (int n = 0; n < bottom[0]->num(); n++) {
    for (int c = 0; c < channels_; c++) {
      caffe_set(spatial_dim, *bottom_data, 
            top_data);
      bottom_data += bottom[0]->offset(0,1);
      top_data += top[0]->offset(0,1);
    }
  }
}

template <typename Dtype>
void ExpandLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0])
    return;
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  const Dtype* top_diff = top[0]->cpu_diff();
  for (int n = 0; n < top[0]->num(); n++) {
    for (int c = 0; c < channels_; c++) {
      for (int h = 0; h < expand_h_; h++) {
        for (int w = 0; w < expand_w_; w++) {
          *bottom_diff += *top_diff;
          top_diff++;
        }
      }
      bottom_diff++;
    }
  }
}



INSTANTIATE_CLASS(ExpandLayer);
REGISTER_LAYER_CLASS(Expand);
} // namespace caffe
