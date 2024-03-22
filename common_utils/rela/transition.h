// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

#pragma once

#include <torch/extension.h>
#include "tensor_dict.h"

namespace rela {

class SingleStepTransition {
 public:
  SingleStepTransition() = default;

  TensorDict obs;
  TensorDict nextObs;
  TensorDict action;
  torch::Tensor reward;
  torch::Tensor bootstrap;
};

class MultiStepTransition {
 public:
  MultiStepTransition() = default;

  MultiStepTransition(const MultiStepTransition&, int bsz);

  void paste_(const MultiStepTransition&, int idx);

  MultiStepTransition index(int i) const;

  void copyTo(int from, MultiStepTransition& dst, int to) const;

  void to_(const std::string& device);

  void seqFirst_();

  TensorDict obs;
  TensorDict h0;
  TensorDict action;
  torch::Tensor reward;
  torch::Tensor bootstrap;
  torch::Tensor seqLen;

 private:
  bool batchFirst_ = false;
};

MultiStepTransition makeBatch(
    const std::vector<MultiStepTransition>& transitions, const std::string& device);

SingleStepTransition makeBatch(
    const std::vector<SingleStepTransition>& transitions, const std::string& device);

}  // namespace rela
