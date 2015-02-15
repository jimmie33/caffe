#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void ReshapeLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  int dim = bottom[0]->channels() * bottom[0]->height()
      * bottom[0]->width();
  int channel = this->layer_param_.reshape_param().channel();
  int height = this->layer_param_.reshape_param().height();
  int width = this->layer_param_.reshape_param().width();

  // checks the condition that at least two shape parameters are initialized
  int n_param = (channel > 0 ? 1:0) + (height > 0 ? 1:0) + (width > 0 ? 1:0);
  CHECK_GE(n_param,2) <<
    "At least two reshape paramters should be specified";

  if (width == 0)
    width = dim/(channel * height);
  else if (height == 0)
    height = dim/(channel * width);
  else if (channel == 0)
    channel = dim/(width * height); 

  top[0]->Reshape(bottom[0]->num(), channel, height, width);
  count_ = bottom[0]->num() * channel * height * width;
  CHECK_EQ(count_, bottom[0]->count());
  CHECK_EQ(count_, top[0]->count());
}

template <typename Dtype>
void ReshapeLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  top[0]->ShareData(*bottom[0]);
}

template <typename Dtype>
void ReshapeLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  bottom[0]->ShareDiff(*top[0]);
}

#ifdef CPU_ONLY
STUB_GPU(ReshapeLayer);
#endif

INSTANTIATE_CLASS(ReshapeLayer);
REGISTER_LAYER_CLASS(Reshape);

}  // namespace caffe
