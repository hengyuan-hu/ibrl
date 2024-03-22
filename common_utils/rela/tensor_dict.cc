#include "rela/tensor_dict.h"
#include "rela/utils.h"

using rela::TensorDict;

namespace rela::tensor_dict {

std::vector<int64_t> getBatchedSize(torch::Tensor t, int bsz) {
  auto tsize = t.sizes();
  std::vector<int64_t> sizes(t.dim() + 1);
  sizes[0] = bsz;
  for (size_t i = 1; i < sizes.size(); ++i) {
    sizes[i] = tsize[i - 1];
  }
  return sizes;
}

TensorDict allocateBatchStorage(const TensorDict& data, int bsz) {
  TensorDict storage;
  for (const auto& kv : data) {
    auto sizes = getBatchedSize(kv.second, bsz);
    storage[kv.first] = torch::zeros(sizes, kv.second.dtype());
  }
  return storage;
}

void compareShape(const TensorDict& src, const TensorDict& dest) {
  if (src.size() != dest.size()) {
    std::cout << "src.size()[" << src.size() << "] != dest.size()[" << dest.size() << "]"
              << std::endl;
    std::cout << "src keys: ";
    for (const auto& p : src)
      std::cout << p.first << " ";
    std::cout << "dest keys: ";
    for (const auto& p : dest)
      std::cout << p.first << " ";
    std::cout << std::endl;
    assert(false);
  }

  for (const auto& name2tensor : src) {
    const auto& name = name2tensor.first;
    const auto& srcTensor = name2tensor.second;
    const auto& destTensor = dest.at(name);
    if (destTensor.sizes() != srcTensor.sizes()) {
      std::cout << name << ", dstSize: " << destTensor.sizes()
                << ", srcSize: " << srcTensor.sizes() << std::endl;
      assert(false);
    }
  }
}

void copy(const TensorDict& src, TensorDict& dest) {
  compareShape(src, dest);
  for (const auto& name2tensor : src) {
    const auto& name = name2tensor.first;
    const auto& srcTensor = name2tensor.second;
    auto& destTensor = dest.at(name);
    destTensor.copy_(srcTensor);
  }
}

void copy(const TensorDict& src, TensorDict& dest, const torch::Tensor& index) {
  assert(src.size() == dest.size());
  assert(index.size(0) > 0);
  for (const auto& name2tensor : src) {
    const auto& name = name2tensor.first;
    const auto& srcTensor = name2tensor.second;
    auto& destTensor = dest.at(name);
    assert(destTensor.dtype() == srcTensor.dtype());
    assert(index.size(0) == srcTensor.size(0));
    destTensor.index_copy_(0, index, srcTensor);
  }
}

bool eq(const TensorDict& d0, const TensorDict& d1) {
  if (d0.size() != d1.size()) {
    return false;
  }

  for (const auto& name2tensor : d0) {
    auto key = name2tensor.first;
    if ((d1.at(key) != name2tensor.second).all().item<bool>()) {
      return false;
    }
  }
  return true;
}

/*
 * indexes into a TensorDict
 */
TensorDict index(const TensorDict& batch, size_t i) {
  TensorDict result;
  for (const auto& name2tensor : batch) {
    result.insert({name2tensor.first, name2tensor.second[i]});
  }
  return result;
}

TensorDict narrow(
    const TensorDict& batch, size_t dim, size_t i, size_t len) {
  TensorDict result;
  for (auto& name2tensor : batch) {
    auto t = name2tensor.second.narrow(dim, i, len);
    result.insert({name2tensor.first, std::move(t)});
  }
  return result;
}

TensorDict clone(const TensorDict& input) {
  TensorDict output;
  for (auto& name2tensor : input) {
    output.insert({name2tensor.first, name2tensor.second.clone()});
  }
  return output;
}

TensorDict zerosLike(const TensorDict& input) {
  TensorDict output;
  for (auto& name2tensor : input) {
    output.insert({name2tensor.first, torch::zeros_like(name2tensor.second)});
  }
  return output;
}

TensorDict stack(const std::vector<TensorDict>& vec, int stackdim) {
  assert(vec.size() >= 1);
  size_t numKey = vec[0].size();
  TensorDict ret;
  for (auto& name2tensor : vec[0]) {
    std::vector<torch::Tensor> buffer(vec.size());
    for (size_t i = 0; i < vec.size(); ++i) {
      if (vec[i].size() != numKey) {
        std::cout << "i: " << i << std::endl;
        std::cout << "ref keys: " << std::endl;
        for (auto& kv : vec[0]) {
          std::cout << kv.first << ", ";
        }
        std::cout << std::endl;

        std::cout << "new keys: " << std::endl;
        for (auto& kv : vec[i]) {
          std::cout << kv.first << ", ";
        }
        std::cout << std::endl;
      }
      assert(vec[i].size() == numKey);
      buffer[i] = vec[i].at(name2tensor.first);
    }
    ret[name2tensor.first] = torch::stack(buffer, stackdim);
  }
  return ret;
}

TensorDict cat(const std::vector<TensorDict>& vec, int catdim) {
  assert(vec.size() >= 1);
  size_t numKey = vec[0].size();
  TensorDict ret;
  for (auto& name2tensor : vec[0]) {
    std::vector<torch::Tensor> buffer(vec.size());
    for (size_t i = 0; i < vec.size(); ++i) {
      if (vec[i].size() != numKey) {
        std::cout << "i: " << i << std::endl;
        std::cout << "ref keys: " << std::endl;
        for (auto& kv : vec[0]) {
          std::cout << kv.first << ", ";
        }
        std::cout << std::endl;

        std::cout << "new keys: " << std::endl;
        for (auto& kv : vec[i]) {
          std::cout << kv.first << ", ";
        }
        std::cout << std::endl;
      }
      assert(vec[i].size() == numKey);
      buffer[i] = vec[i].at(name2tensor.first);
    }
    ret[name2tensor.first] = torch::cat(buffer, catdim);
  }
  return ret;
}

std::vector<std::string> getKeys(const TensorDict d) {
  std::vector<std::string> keys;
  for (auto& kv : d) {
    keys.push_back(kv.first);
  }
  return keys;
}
}  // namespace rela::tensor_dict
