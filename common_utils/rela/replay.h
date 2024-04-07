#pragma once

#include <queue>
#include <random>
#include <thread>
#include <vector>

#include "rela/concurrent_queue.h"
#include "rela/tensor_dict.h"
#include "rela/transition.h"

namespace rela {

template <typename T>
class Replay {
 public:
  Replay(int capacity, int seed, int prefetch, float extra)
      : prefetch_(prefetch)
      , capacity_(capacity)
      , storage_(int((1 + extra) * capacity))
      , numAdd_(0) {
    assert(extra > 0);
    rng_.seed(seed);
  }

  void terminate() {
    storage_.terminate();
  }

  void add(const MultiStepTransition& sample) {
    numAdd_ += 1;
    storage_.append(sample, 1);
  }

  T sample(int batchsize, const std::string& device) {
    // simple, single thread version
    if (prefetch_ == 0) {
      return sample_(batchsize, device);
    }

    if (samplerThread_ == nullptr) {
      // create sampler thread
      samplerThread_ = std::make_unique<std::thread>(
        &Replay::sampleLoop_, this, batchsize, device);
    }

    std::unique_lock<std::mutex> lk(mSampler_);
    cvSampler_.wait(lk, [this] { return samples_.size() > 0; });
    auto batch = samples_.front();
    samples_.pop();

    lk.unlock();
    cvSampler_.notify_all();
    return batch;
  }

  MultiStepTransition get(int idx) {
    return storage_.get(idx);
  }

  MultiStepTransition getRange(int start, int end, const std::string& device) {
    std::vector<MultiStepTransition> samples;
    for (int i = start; i < end; ++i) {
      samples.push_back(storage_.get(i));
    };
    return makeBatch(samples, device);
  }

  int size() const {
    return storage_.safeSize(nullptr);
  }

  int numAdd() const {
    return numAdd_;
  }

 protected:
  void sampleLoop_(int batchsize, const std::string& device) {
    while (true) {
      auto batch = sample_(batchsize, device);
      std::unique_lock<std::mutex> lk(mSampler_);
      cvSampler_.wait(lk, [this] { return (int)samples_.size() < prefetch_; });
      samples_.push(batch);
      lk.unlock();
      cvSampler_.notify_all();
    }
  }

  virtual T sample_(int batchsize, const std::string& device) = 0;

  const int prefetch_;
  const int capacity_;

  // make sure that multiple calls of sample does not overlap
  std::unique_ptr<std::thread> samplerThread_;
  // basic concurrent queue for read and write data
  std::queue<T> samples_;
  std::mutex mSampler_;
  std::condition_variable cvSampler_;

  ConcurrentQueue storage_;
  std::atomic<int> numAdd_;

  std::mt19937 rng_;
};

class MultiStepTransitionReplay : public Replay<MultiStepTransition> {
 public:
  MultiStepTransitionReplay(int capacity, int seed, int prefetch, float extra)
      : Replay<MultiStepTransition>(capacity, seed, prefetch, extra) {
  }

 protected:
  MultiStepTransition sample_(int batchsize, const std::string& device) override;
};

class SingleStepTransitionReplay : public Replay<SingleStepTransition> {
 public:
  SingleStepTransitionReplay(
      int frameStack, int nStep, int capacity, int seed, int prefetch, float extra)
      : Replay(capacity, seed, prefetch, extra)
      , frameStack_(frameStack)
      , nStep_(nStep) {
    assert(nStep_ >= 1);
  }

 protected:
  SingleStepTransition sample_(int batchsize, const std::string& device) override;

  int frameStack_;
  int nStep_;
};

}  // namespace rela
