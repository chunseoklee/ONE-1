/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "luci/Import/Nodes/CircleFullyConnected.h"

#include <luci/IR/Nodes/CircleFullyConnected.h>

#include <loco.h>
#include <oops/UserExn.h>

namespace luci
{

bool CircleFullyConnectedGraphBuilder::validate(const ValidateArgs &args) const
{
  if (args.op.inputs.size() != 3)
    return false;

  return true;
}

CircleNode *CircleFullyConnectedGraphBuilder::build_node(const circle::OperatorT &op,
                                                         const std::vector<CircleNode *> &inputs,
                                                         loco::Graph *graph) const
{
  auto *node = graph->nodes()->create<CircleFullyConnected>();
  node->input(inputs[0]);
  node->weights(inputs[1]);
  node->bias(inputs[2]);

  const auto *options = op.builtin_options.AsFullyConnectedOptions();
  node->fusedActivationFunction(luci_actfunc(options->fused_activation_function));
  if (options->weights_format != circle::FullyConnectedOptionsWeightsFormat_DEFAULT)
  {
    throw oops::UserExn(
        "Unsupported weights format",
        circle::EnumNameFullyConnectedOptionsWeightsFormat(options->weights_format));
  }

  return node;
}

} // namespace luci
