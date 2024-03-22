#include "rela/concurrent_queue.h"

using namespace rela;

int ConcurrentQueue::safeSize(float* sum) const {
  std::unique_lock<std::mutex> lk(m_);
  if (sum != nullptr) {
    *sum = sum_;
  }
  return safeSize_;
}

int ConcurrentQueue::size() const {
  std::unique_lock<std::mutex> lk(m_);
  return size_;
}

void ConcurrentQueue::clear() {
  std::unique_lock<std::mutex> lk(m_);
  head_ = 0;
  tail_ = 0;
  size_ = 0;
  safeTail_ = 0;
  safeSize_ = 0;
  sum_ = 0;
  std::fill(evicted_.begin(), evicted_.end(), false);
  std::fill(weights_.begin(), weights_.end(), 0.0);
}

void ConcurrentQueue::terminate() {
  terminated_ = true;
  cvSize_.notify_all();
}

void ConcurrentQueue::append(const MultiStepTransition& data, float weight) {
  int blockSize = 1;
  std::unique_lock<std::mutex> lk(m_);
  cvSize_.wait(lk, [=] { return terminated_ || (size_ + blockSize <= capacity); });
  if (terminated_) {
    return;
  }

  if (elements_ == nullptr) {
    // elements_ is batchFirst
    elements_ = std::make_unique<MultiStepTransition>(data, capacity);
  }

  int start = tail_;
  int end = (tail_ + blockSize) % capacity;

  tail_ = end;
  size_ += blockSize;
  checkSize(head_, tail_, size_);

  lk.unlock();

  elements_->paste_(data, start);
  weights_[start] = weight;

  lk.lock();

  cvTail_.wait(lk, [=] { return safeTail_ == start; });
  safeTail_ = end;
  safeSize_ += blockSize;
  sum_ += weight;
  checkSize(head_, safeTail_, safeSize_);

  lk.unlock();
  cvTail_.notify_all();
}

// ------------------------------------------------------------- //
// blockPop, update are thread-safe against blockAppend
// but they are NOT thread-safe against each other
void ConcurrentQueue::blockPop(int blockSize) {
  double diff = 0;
  int head = head_;
  for (int i = 0; i < blockSize; ++i) {
    diff -= weights_[head];
    evicted_[head] = true;
    head = (head + 1) % capacity;
  }

  {
    std::lock_guard<std::mutex> lk(m_);
    sum_ += diff;
    head_ = head;
    safeSize_ -= blockSize;
    size_ -= blockSize;
    assert(safeSize_ >= 0);
    checkSize(head_, safeTail_, safeSize_);
  }
  cvSize_.notify_all();
}

void ConcurrentQueue::update(const std::vector<int>& ids, const torch::Tensor& weights) {
  double diff = 0;
  auto weightAcc = weights.accessor<float, 1>();
  for (int i = 0; i < (int)ids.size(); ++i) {
    auto id = ids[i];
    if (evicted_[id]) {
      continue;
    }
    diff += (weightAcc[i] - weights_[id]);
    weights_[id] = weightAcc[i];
  }

  std::lock_guard<std::mutex> lk_(m_);
  sum_ += diff;
}

// ------------------------------------------------------------- //
// accessing elements is never locked, operate safely!
MultiStepTransition ConcurrentQueue::get(int idx) {
  int id = (head_ + idx) % capacity;
  return elements_->index(id);
}

void ConcurrentQueue::copyTo(int idx, MultiStepTransition& dst, int dstSlot) {
  int id = (head_ + idx) % capacity;
  elements_->copyTo(id, dst, dstSlot);
}

MultiStepTransition ConcurrentQueue::getElementAndMark(int idx) {
  int id = (head_ + idx) % capacity;
  evicted_[id] = false;
  return elements_->index(id);
}

float ConcurrentQueue::getWeight(int idx, int* id) {
  assert(id != nullptr);
  *id = (head_ + idx) % capacity;
  return weights_[*id];
}

void ConcurrentQueue::checkSize(int head, int tail, int size) {
  if (size == 0) {
    assert(tail == head);
  } else if (tail > head) {
    if (tail - head != size) {
      std::cout << "tail-head: " << tail - head << " vs size: " << size << std::endl;
    }
    assert(tail - head == size);
  } else {
    if (tail + capacity - head != size) {
      std::cout << "tail-head: " << tail + capacity - head << " vs size: " << size
                << std::endl;
    }
    assert(tail + capacity - head == size);
  }
}
