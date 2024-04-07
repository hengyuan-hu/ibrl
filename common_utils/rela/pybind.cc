// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "rela/episode.h"
#include "rela/replay.h"
#include "rela/transition.h"

namespace py = pybind11;
using namespace rela;

PYBIND11_MODULE(rela, m) {
  py::class_<MultiStepTransition, std::shared_ptr<MultiStepTransition>>(
      m, "MultiStepTransition")
      .def_readwrite("obs", &MultiStepTransition::obs)
      .def_readwrite("h0", &MultiStepTransition::h0)
      .def_readwrite("action", &MultiStepTransition::action)
      .def_readwrite("reward", &MultiStepTransition::reward)
      .def_readwrite("bootstrap", &MultiStepTransition::bootstrap)
      .def_readwrite("seq_len", &MultiStepTransition::seqLen);

  py::class_<SingleStepTransition, std::shared_ptr<SingleStepTransition>>(
      m, "SingleStepTransition")
      .def_readwrite("obs", &SingleStepTransition::obs)
      .def_readwrite("next_obs", &SingleStepTransition::nextObs)
      .def_readwrite("action", &SingleStepTransition::action)
      .def_readwrite("reward", &SingleStepTransition::reward)
      .def_readwrite("bootstrap", &SingleStepTransition::bootstrap);

  py::class_<MultiStepTransitionReplay, std::shared_ptr<MultiStepTransitionReplay>>(
      m, "MultiStepTransitionReplay")
      .def(py::init<int, int, int, float>(),
           py::arg("capacity"),
           py::arg("seed"),
           py::arg("prefetch"),
           py::arg("extra"))
      .def("terminate", &MultiStepTransitionReplay::terminate)
      .def("size", &MultiStepTransitionReplay::size)
      .def("num_add", &MultiStepTransitionReplay::numAdd)
      .def("sample", &MultiStepTransitionReplay::sample)
      .def("get", &MultiStepTransitionReplay::get)
      .def("add", &MultiStepTransitionReplay::add)
      .def("get_range", &MultiStepTransitionReplay::getRange);

  py::class_<SingleStepTransitionReplay, std::shared_ptr<SingleStepTransitionReplay>>(
      m, "SingleStepTransitionReplay")
      .def(py::init<int, int, int, int, int, float>(),
           py::arg("frame_stack"),
           py::arg("n_step"),
           py::arg("capacity"),
           py::arg("seed"),
           py::arg("prefetch"),
           py::arg("extra"))
      .def("terminate", &SingleStepTransitionReplay::terminate)
      .def("size", &SingleStepTransitionReplay::size)
      .def("num_add", &SingleStepTransitionReplay::numAdd)
      .def("sample", &SingleStepTransitionReplay::sample)
      .def("get", &SingleStepTransitionReplay::get)
      .def("add", &SingleStepTransitionReplay::add)
      .def("get_range", &SingleStepTransitionReplay::getRange);

  py::class_<Episode, std::shared_ptr<Episode>>(m, "Episode")
      .def(py::init<int, int, float>())
      .def("init", &Episode::init)
      .def("len", &Episode::len)
      .def("push_obs", &Episode::pushObs)
      .def("push_action", &Episode::pushAction)
      .def("push_reward", &Episode::pushReward)
      .def("push_terminal", &Episode::pushTerminal)
      .def("reset", &Episode::reset)
      .def("reset_rewards", &Episode::resetRewards)
      .def("get_rewards", &Episode::getRewards)
      .def("pop_transition", &Episode::popTransition);
}
