#include "rela/replay.h"

using namespace rela;

MultiStepTransition MultiStepTransitionReplay::sample_(
    int batchsize, const std::string& device) {
  // assert(frameStack_ == 1);
  float sum;
  int size = storage_.safeSize(&sum);
  assert(int(sum) == size);
  assert(size >= batchsize);
  // storage_ [0, size) remains static in the subsequent section
  int segment = size / batchsize;
  std::uniform_int_distribution<int> dist(0, segment - 1);

  assert(batchsize > 0);
  MultiStepTransition batch(storage_.get(0), batchsize);

  // MultiStepTransition batch;
  for (int i = 0; i < batchsize; ++i) {
    int rand = dist(rng_) + i * segment;
    assert(rand < size);
    storage_.copyTo(rand, batch, i);
  }

  // pop storage if full
  size = storage_.size();
  if (size > capacity_) {
    storage_.blockPop(size - capacity_);
  }
  batch.seqFirst_();
  batch.to_(device);
  return batch;
}

SingleStepTransition SingleStepTransitionReplay::sample_(
    int batchsize, const std::string& device) {
  float sum;
  int size = storage_.safeSize(&sum);
  assert(int(sum) == size);  // uniform weights

  std::uniform_int_distribution<int> episodeDist(0, size - 1);
  std::unique_ptr<std::uniform_int_distribution<int>> tDist = nullptr;

  std::vector<SingleStepTransition> samples;
  while (batchsize > 0) {
    int episodeIdx = episodeDist(rng_);
    auto episode = storage_.get(episodeIdx);
    if (tDist == nullptr) {
      int maxSeqLen = episode.reward.size(0);
      tDist = std::make_unique<std::uniform_int_distribution<int>>(0, maxSeqLen - 1);
    }

    int tIdx = (*tDist)(rng_);
    int seqLen = episode.seqLen.item<int>();
    if (tIdx >= seqLen) {
      continue;
    }

    // a valid tIdx, use this sample
    --batchsize;

    SingleStepTransition transition;

    // process observations, maybe frame stack
    {
      // frame stack
      std::vector<TensorDict> frames;
      std::vector<TensorDict> nextFrames;
      for (int d = frameStack_ - 1; d >= 0; --d) {
        int fIdx = std::max(0, tIdx - d);
        frames.push_back(tensor_dict::index(episode.obs, fIdx));

        fIdx = std::min(seqLen - 1, tIdx + nStep_ - d);
        nextFrames.push_back(tensor_dict::index(episode.obs, fIdx));
      }
      transition.obs = tensor_dict::cat(frames, 0);
      transition.nextObs = tensor_dict::cat(nextFrames, 0);
    }

    transition.action = tensor_dict::index(episode.action, tIdx);
    transition.reward = episode.reward[tIdx];
    transition.bootstrap = episode.bootstrap[tIdx];

    samples.push_back(transition);
  }

  size = storage_.size();
  if (size > capacity_) {
    storage_.blockPop(size - capacity_);
  }

  return makeBatch(samples, device);
}
