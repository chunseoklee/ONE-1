/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "WhileLayer.h"

#include <backend/ITensor.h>
#include "exec/ExecutorBase.h"
#include "PermuteLayer.h"

namespace onert
{
namespace backend
{
namespace controlflow
{
namespace kernel
{

WhileLayer::WhileLayer(const std::vector<std::shared_ptr<backend::ITensor>> &input_tensors,
                       const std::vector<std::shared_ptr<backend::ITensor>> &output_tensors,
                       const ir::OperandIndexSequence &output_indices, const ir::Graph &graph,
                       const exec::DynAllocInfoMap &outputs_dyn_alloc_info,
                       const ir::SubgraphIndex &cond_subg_index,
                       const ir::SubgraphIndex &body_subg_index, exec::ExecutorMap *executor_map)
    : _cond_subg_index{cond_subg_index}, _body_subg_index{body_subg_index},
      _output_indices{output_indices}, _graph{graph}, _input_tensors{input_tensors},
      _output_tensors{output_tensors}, _outputs_dyn_alloc_info{outputs_dyn_alloc_info},
      _executor_map{executor_map}
{
  // At this point, executor_map may not have executors of cond subg and body subg
}

void WhileLayer::run()
{
  // Copy "_input_tensors" -> "cond subg inputs"
  // Run cond subg
  // Start loop while output of cond subg is ture
  // // Copy "_input_tensors" -> "body subg inputs" in the first iteration, then copy "body subg
  // outputs" -> "body subg inputs" in the second or more iterations
  // // Run body subg
  // // Copy "body subg outputs" -> "cond subg inputs"
  // // Run cond subg
  // If there is no loop copy "_input_tensors" -> "_dst_tensors", else copy "cond subg inputs" ->
  // "_dst_tensors"
  auto cond_exec = dynamic_cast<exec::ExecutorBase *>(_executor_map->at(_cond_subg_index).get());
  auto body_exec = dynamic_cast<exec::ExecutorBase *>(_executor_map->at(_body_subg_index).get());
  if ((cond_exec == nullptr) || (body_exec == nullptr))
  {
    throw std::runtime_error{"While: Invalid condition or body"};
  }

  const auto &cond_graph = cond_exec->graph();
  const auto &cond_inputs_dyn_alloc = cond_exec->getInputsDynamicAllocInfo();
  const auto &body_graph = body_exec->graph();
  const auto &body_inputs_dyn_alloc = body_exec->getInputsDynamicAllocInfo();

  std::vector<std::shared_ptr<backend::ITensor>> input_tensors;
  std::vector<std::shared_ptr<backend::ITensor>> cond_input_tensors;
  std::vector<std::shared_ptr<backend::ITensor>> body_input_tensors;
  std::vector<std::shared_ptr<backend::ITensor>> body_output_tensors;
  std::vector<std::shared_ptr<backend::ITensor>> output_tensors;

  // Add only used tensors in cond subgraph
  assert(_input_tensors.size() == cond_exec->getInputTensors().size());
  for (uint32_t i = 0; i < cond_graph.getInputs().size(); ++i)
  {
    const auto &cond_input = cond_graph.operands().at(cond_graph.getInputs().at(i));
    if (cond_input.getUses().size() > 0)
    {
      input_tensors.emplace_back(_input_tensors.at(i));
      cond_input_tensors.emplace_back(cond_exec->getInputTensors().at(i));
    }
  }
  const auto permute_op_input_to_cond_input =
      std::make_shared<PermuteLayer>(input_tensors, cond_input_tensors, cond_inputs_dyn_alloc);

  // Add only used tensors among outputs of while operation
  assert(_input_tensors.size() == _output_tensors.size());
  assert(_output_indices.size() == _output_tensors.size());
  input_tensors.clear();
  output_tensors.clear();
  for (size_t i = 0; i < _output_tensors.size(); ++i)
  {
    const auto &output_index = _output_indices.at(i);
    const auto &output = _graph.operands().at(output_index);
    if (output.getUses().size() > 0 || _graph.getOutputs().contains(output_index))
    {
      input_tensors.emplace_back(_input_tensors.at(i));
      output_tensors.emplace_back(_output_tensors.at(i));
    }
  }
  const auto permute_op_input_to_op_output =
      std::make_shared<PermuteLayer>(input_tensors, output_tensors, _outputs_dyn_alloc_info);

  // Add all tensors with unused tensors in body subgraph because unused input tensors will be
  // copied output tensors in body subgraph
  assert(_input_tensors.size() == body_exec->getInputTensors().size());
  input_tensors = _input_tensors;
  body_input_tensors = body_exec->getInputTensors();
  const auto permute_op_input_to_body_input =
      std::make_shared<PermuteLayer>(input_tensors, body_input_tensors, body_inputs_dyn_alloc);

  // Add only used tensors in cond subgraph
  assert(body_exec->getOutputTensors().size() == cond_exec->getInputTensors().size());
  body_output_tensors.clear();
  cond_input_tensors.clear();
  for (uint32_t i = 0; i < cond_graph.getInputs().size(); ++i)
  {
    const auto &cond_input = cond_graph.operands().at(cond_graph.getInputs().at(i));
    if (cond_input.getUses().size() > 0)
    {
      body_output_tensors.emplace_back(body_exec->getOutputTensors().at(i));
      cond_input_tensors.emplace_back(cond_exec->getInputTensors().at(i));
    }
  }
  const auto permute_body_output_to_cond_input = std::make_shared<PermuteLayer>(
      body_output_tensors, cond_input_tensors, cond_inputs_dyn_alloc);

  // Add only used tensors in body subgraph
  assert(body_exec->getOutputTensors().size() == body_exec->getInputTensors().size());
  body_output_tensors.clear();
  body_input_tensors.clear();
  for (uint32_t i = 0; i < body_graph.getInputs().size(); ++i)
  {
    const auto &body_input_index = body_graph.getInputs().at(i);
    const auto &body_input = body_graph.operands().at(body_input_index);
    if (body_input.getUses().size() > 0 &&
        !body_exec->graph().getOutputs().contains(body_input_index))
    {
      body_output_tensors.emplace_back(body_exec->getOutputTensors().at(i));
      body_input_tensors.emplace_back(body_exec->getInputTensors().at(i));
    }
  }
  const auto permute_body_output_to_body_input = std::make_shared<PermuteLayer>(
      body_output_tensors, body_input_tensors, body_inputs_dyn_alloc);

  // Add only used tensors among outputs of while operation
  assert(body_exec->getOutputTensors().size() == _output_tensors.size());
  assert(_output_indices.size() == _output_tensors.size());
  body_output_tensors.clear();
  output_tensors.clear();
  for (size_t i = 0; i < _output_tensors.size(); ++i)
  {
    const auto &output_index = _output_indices.at(i);
    const auto &output = _graph.operands().at(output_index);
    if (output.getUses().size() > 0 || _graph.getOutputs().contains(output_index))
    {
      body_output_tensors.emplace_back(body_exec->getOutputTensors().at(i));
      output_tensors.emplace_back(_output_tensors.at(i));
    }
  }
  const auto permute_body_output_to_op_output =
      std::make_shared<PermuteLayer>(body_output_tensors, output_tensors, _outputs_dyn_alloc_info);

  // Remove copying of unused tensor
  permute_op_input_to_cond_input->prepare();
  permute_op_input_to_op_output->prepare();
  permute_op_input_to_body_input->prepare();
  permute_body_output_to_cond_input->prepare();
  permute_body_output_to_body_input->prepare();
  permute_body_output_to_op_output->prepare();

  cond_exec->execute(_input_tensors, permute_op_input_to_cond_input);

  assert(cond_exec->getOutputTensors().size() == 1);
  auto &cond_output_tensor = cond_exec->getOutputTensors().at(0);
  auto getResultCond = [](backend::ITensor *tensor) -> bool {
    bool ret = false;
    tensor->access([&](ITensor &tensor) { ret = *reinterpret_cast<bool *>(tensor.buffer()); });
    return ret;
  };

  auto setOpOutputDynamic = [&](const std::vector<std::shared_ptr<backend::ITensor>> &src_tensors) {
    assert(_output_tensors.size() == src_tensors.size());
    for (size_t i = 0; i < _output_tensors.size(); ++i)
    {
      const auto output_tensor = _output_tensors.at(i);
      const auto orig_output_shape = output_tensor->getShape();
      const auto changed_output_shape = src_tensors.at(i)->getShape();
      const auto &output_dyn_alloc_info = _outputs_dyn_alloc_info.find(output_tensor);
      if (orig_output_shape != changed_output_shape &&
          output_dyn_alloc_info != _outputs_dyn_alloc_info.end() &&
          (_graph.operands().at(output_dyn_alloc_info->second.ind).getUses().size() > 0 ||
           _graph.getOutputs().contains(output_dyn_alloc_info->second.ind)))
      {
        output_tensor->set_dynamic();
      }
    }
  };

  const auto body_execute_with_op_inputs = [&]() {
    body_exec->execute(_input_tensors, permute_op_input_to_body_input);
  };

  const auto body_execute_with_body_outputs = [&]() {
    body_exec->execute(body_exec->getOutputTensors(), permute_body_output_to_body_input);
  };

  std::function<void()> body_execute = body_execute_with_op_inputs;
  const auto cond_execute = [&]() {
    cond_exec->execute(body_exec->getOutputTensors(), permute_body_output_to_cond_input);
  };
  auto permute_to_outputs_fn = permute_op_input_to_op_output;

  // Loop while Cond subgraph's output is true
  while (getResultCond(cond_output_tensor.get()))
  {
    body_execute();
    cond_execute();
    body_execute = body_execute_with_body_outputs;
    permute_to_outputs_fn = permute_body_output_to_op_output;
  }
  setOpOutputDynamic(body_exec->getOutputTensors());
  permute_to_outputs_fn->run();
}

} // namespace kernel
} // namespace controlflow
} // namespace backend
} // namespace onert
