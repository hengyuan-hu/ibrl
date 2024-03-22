// common components for implementing R2D2 actors
#pragma once

#include "rela/tensor_dict.h"
#include "rela/transition.h"

namespace rela {

class Episode {
 public:
  Episode(int multiStep, int maxSeqLen, float gamma)
      : multiStep_(multiStep)
      , maxSeqLen_(maxSeqLen)
      , gamma_(gamma)
      , callOrder_(0)
      , seqLen_(0)
      , reward_(maxSeqLen)
      , terminal_(maxSeqLen) {
    assert(maxSeqLen > 0);
    transition_.reward = torch::zeros(maxSeqLen, torch::kFloat32);
    transition_.bootstrap = torch::zeros(maxSeqLen, torch::kFloat32);
    transition_.seqLen = torch::tensor(float(0));
  }

  void init(const TensorDict& h0) {
    // h0_ = h0;
    if (seqLen_ != 0 || callOrder_ != 0) {
      std::cout << "Error Episode.init: seqLen_ = " << seqLen_
                << ", callOrder_ = " << callOrder_ << std::endl;
      assert(false);
    }
    transition_.h0 = h0;
  }

  int len() const {
    return seqLen_;
  }

  void pushObs(const TensorDict& obs);

  void pushAction(const TensorDict& action);

  void pushReward(float r);

  void pushTerminal(float t);

  void reset();

  std::vector<float> getRewards();

  void resetRewards(const std::vector<float>& rewards);

  MultiStepTransition popTransition();

 private:
  const int multiStep_;
  const int maxSeqLen_;
  const float gamma_;

  MultiStepTransition transition_;

  int callOrder_;
  int seqLen_;
  std::vector<float> reward_;
  std::vector<float> terminal_;
};

}  // namespace rela
