#include "rela/episode.h"

using namespace rela;

void Episode::pushObs(const TensorDict& obs) {
  assert(callOrder_ == 0);
  ++callOrder_;
  assert(seqLen_ < maxSeqLen_);

  if (transition_.obs.size() == 0) {
    transition_.obs = tensor_dict::allocateBatchStorage(obs, maxSeqLen_);
  }

  assert(transition_.obs.size() == obs.size());
  for (auto& kv : obs) {
    transition_.obs[kv.first][seqLen_] = kv.second;
  }
}

void Episode::pushAction(const TensorDict& action) {
  assert(callOrder_ == 1);
  ++callOrder_;

  if (transition_.action.size() == 0) {
    transition_.action = tensor_dict::allocateBatchStorage(action, maxSeqLen_);
  }

  assert(transition_.action.size() == action.size());
  for (auto& kv : action) {
    transition_.action[kv.first][seqLen_] = kv.second;
  }
}

void Episode::pushReward(float r) {
  assert(callOrder_ == 2);
  ++callOrder_;
  reward_[seqLen_] = r;
}

void Episode::pushTerminal(float t) {
  assert(callOrder_ == 3);
  callOrder_ = 0;
  terminal_[seqLen_] = t;
  ++seqLen_;
}

void Episode::reset() {
  assert(callOrder_ == 0);
  assert(terminal_[seqLen_ - 1] == 1.0f);
  seqLen_ = 0;
  callOrder_ = 0;
}

void Episode::resetRewards(const std::vector<float>& rewards) {
  assert((int)rewards.size() == seqLen_);
  for (int i = 0; i < seqLen_; ++i) {
    reward_[i] = rewards[i];
  }
}

std::vector<float> Episode::getRewards() {
  assert(callOrder_ == 0);
  assert(terminal_[seqLen_ - 1] == 1.0f);

  return std::vector<float>(reward_.begin(), reward_.begin() + seqLen_);
}

MultiStepTransition Episode::popTransition() {
  assert(callOrder_ == 0);
  // episode has to terminate
  assert(terminal_[seqLen_ - 1] == 1.0f);

  transition_.seqLen.data_ptr<float>()[0] = float(seqLen_);
  auto accReward = transition_.reward.accessor<float, 1>();
  auto bootstrap = transition_.bootstrap.accessor<float, 1>();
  // acc reward
  for (int i = 0; i < seqLen_; ++i) {
    float factor = 1;
    float acc = 0;
    for (int j = 0; j < multiStep_; ++j) {
      if (i + j >= seqLen_) {
        break;
      }
      acc += factor * reward_[i + j];
      factor *= gamma_;
    }
    accReward[i] = acc;
  }

  for (int i = 0; i < seqLen_; ++i) {
    if (i < seqLen_ - multiStep_) {
      bootstrap[i] = float(1.0f * pow(gamma_, multiStep_));
    } else {
      bootstrap[i] = 0.0f;
    }
  }

  seqLen_ = 0;
  callOrder_ = 0;
  return transition_;
}
