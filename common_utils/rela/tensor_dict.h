#pragma once

#include <torch/extension.h>
#include <unordered_map>

namespace rela {

using TensorDict = std::unordered_map<std::string, torch::Tensor>;

namespace tensor_dict {

std::vector<int64_t> getBatchedSize(torch::Tensor t, int bsz);

TensorDict allocateBatchStorage(const TensorDict& data, int bsz);

void compareShape(const TensorDict& src, const TensorDict& dest);

void copy(const TensorDict& src, TensorDict& dest);

void copy(const TensorDict& src, TensorDict& dest, const torch::Tensor& index);

bool eq(const TensorDict& d0, const TensorDict& d1);

/*
 * indexes into a TensorDict
 */
TensorDict index(const TensorDict& batch, size_t i);

TensorDict narrow(const TensorDict& batch, size_t dim, size_t i, size_t len);

TensorDict clone(const TensorDict& input);

TensorDict zerosLike(const TensorDict& input);

// TODO: rewrite the above functions with this template
template <typename Func>
inline TensorDict apply(TensorDict& dict, Func f) {
  TensorDict output;
  for (const auto& name2tensor : dict) {
    auto tensor = f(name2tensor.second);
    output.insert({name2tensor.first, tensor});
  }
  return output;
}

TensorDict stack(const std::vector<TensorDict>& vec, int stackdim);

TensorDict cat(const std::vector<TensorDict>& vec, int catdim);

std::vector<std::string> getKeys(const TensorDict d);
}  // namespace tensor_dict
}  // namespace rela
