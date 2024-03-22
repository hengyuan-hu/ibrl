#pragma once

#include <vector>

#include "rela/tensor_dict.h"
#include "rela/transition.h"

namespace rela {

// template <class DataType>
class ConcurrentQueue {
 public:
  ConcurrentQueue(int capacity)
      : capacity(capacity)
      , head_(0)
      , tail_(0)
      , size_(0)
      , safeTail_(0)
      , safeSize_(0)
      , sum_(0)
      , evicted_(capacity, false)
      , weights_(capacity, 0) {
  }

  int safeSize(float* sum) const;

  int size() const;

  void clear();

  void terminate();

  void append(const MultiStepTransition& data, float weight);

  // ------------------------------------------------------------- //
  // blockPop, update are thread-safe against blockAppend
  // but they are NOT thread-safe against each other
  void blockPop(int blockSize);

  void update(const std::vector<int>& ids, const torch::Tensor& weights);

  // ------------------------------------------------------------- //
  // accessing elements is never locked, operate safely!
  MultiStepTransition get(int idx);

  void copyTo(int idx, MultiStepTransition& dst, int dstSlot);

  MultiStepTransition getElementAndMark(int idx);

  float getWeight(int idx, int* id);

  const int capacity;

 private:
  void checkSize(int head, int tail, int size);

  mutable std::mutex m_;
  std::condition_variable cvSize_;
  std::condition_variable cvTail_;

  int head_;
  int tail_;
  int size_;

  int safeTail_;
  int safeSize_;
  double sum_;
  std::vector<bool> evicted_;

  std::unique_ptr<MultiStepTransition> elements_;
  std::vector<float> weights_;

  bool terminated_ = false;
};
}  // namespace rela
