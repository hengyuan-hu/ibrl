// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

#pragma once

#include <iostream>
#include <unordered_map>
#include <vector>

namespace rela {

namespace utils {

inline int getProduct(const std::vector<int64_t>& nums) {
  int prod = 1;
  for (auto v : nums) {
    prod *= v;
  }
  return prod;
}

template <typename T>
inline void printVector(const std::vector<T>& vec) {
  for (const auto& v : vec) {
    std::cout << v << ", ";
  }
  std::cout << std::endl;
}

template <typename T>
inline void printMapKey(const T& map) {
  for (const auto& name2sth : map) {
    std::cout << name2sth.first << ", ";
  }
  std::cout << std::endl;
}

template <typename T>
inline void printMap(const T& map) {
  for (const auto& name2sth : map) {
    std::cout << name2sth.first << ": " << name2sth.second << std::endl;
  }
}
}  // namespace utils
}  // namespace rela
