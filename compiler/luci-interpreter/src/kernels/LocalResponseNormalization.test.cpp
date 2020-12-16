/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

#include "kernels/LocalResponseNormalization.h"
#include "kernels/TestUtils.h"

namespace luci_interpreter
{
namespace kernels
{
namespace
{

using namespace testing;

TEST(LocalResponseNormalizationTest, SameAsL2Norm)
{
  Tensor input_tensor =
    makeInputTensor<DataType::FLOAT32>({1, 1, 1, 6}, {-1.1, 0.6, 0.7, 1.2, -0.7, 0.1});
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);

  LocalResponseNormalizationParams params{};
  params.radius = 20;
  params.bias = 0.0;
  params.alpha = 1.0;
  params.beta = 0.5;

  LocalResponseNormalization kernel(&input_tensor, &output_tensor, params);
  kernel.configure();
  kernel.execute();

  EXPECT_THAT(extractTensorData<float>(output_tensor),
              FloatArrayNear({-0.55, 0.3, 0.35, 0.6, -0.35, 0.05}));
}

TEST(LocalResponseNormalizationTest, WithAlpha)
{
  Tensor input_tensor =
    makeInputTensor<DataType::FLOAT32>({1, 1, 1, 6}, {-1.1, 0.6, 0.7, 1.2, -0.7, 0.1});
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);

  LocalResponseNormalizationParams params{};
  params.radius = 20;
  params.bias = 0.0;
  params.alpha = 4.0;
  params.beta = 0.5;

  LocalResponseNormalization kernel(&input_tensor, &output_tensor, params);
  kernel.configure();
  kernel.execute();

  EXPECT_THAT(extractTensorData<float>(output_tensor),
              FloatArrayNear({-0.275, 0.15, 0.175, 0.3, -0.175, 0.025}));
}

TEST(LocalResponseNormalizationTest, WithBias)
{
  Tensor input_tensor =
    makeInputTensor<DataType::FLOAT32>({1, 1, 1, 6}, {-1.1, 0.6, 0.7, 1.2, -0.7, 0.1});
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);

  LocalResponseNormalizationParams params{};
  params.radius = 20;
  params.bias = 9.0;
  params.alpha = 4.0;
  params.beta = 0.5;

  LocalResponseNormalization kernel(&input_tensor, &output_tensor, params);
  kernel.configure();
  kernel.execute();

  EXPECT_THAT(extractTensorData<float>(output_tensor),
              FloatArrayNear({-0.22, 0.12, 0.14, 0.24, -0.14, 0.02}));
}

TEST(LocalResponseNormalizationTest, SmallRadius)
{
  Tensor input_tensor =
    makeInputTensor<DataType::FLOAT32>({1, 1, 1, 6}, {-1.1, 0.6, 0.7, 1.2, -0.7, 0.1});
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);

  LocalResponseNormalizationParams params{};
  params.radius = 2;
  params.bias = 9.0;
  params.alpha = 4.0;
  params.beta = 0.5;

  LocalResponseNormalization kernel(&input_tensor, &output_tensor, params);
  kernel.configure();
  kernel.execute();

  EXPECT_THAT(extractTensorData<float>(output_tensor),
              FloatArrayNear({-0.264926, 0.125109, 0.140112, 0.267261, -0.161788, 0.0244266}));
}

TEST(LocalResponseNormalizationTest, InvalidInputDimension_NEG)
{
  Tensor input_tensor =
    makeInputTensor<DataType::FLOAT32>({1, 1, 6}, {-1.1, 0.6, 0.7, 1.2, -0.7, 0.1});
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);

  LocalResponseNormalizationParams params{};
  params.radius = 20;
  params.bias = 0.0;
  params.alpha = 1.0;
  params.beta = 0.5;

  LocalResponseNormalization kernel(&input_tensor, &output_tensor, params);
  EXPECT_ANY_THROW(kernel.configure());
}

TEST(LocalResponseNormalizationTest, InvalidInputOutputType_NEG)
{
  Tensor input_tensor =
    makeInputTensor<DataType::FLOAT32>({1, 1, 1, 6}, {-1.1, 0.6, 0.7, 1.2, -0.7, 0.1});
  Tensor output_tensor = makeOutputTensor(DataType::U8);

  LocalResponseNormalizationParams params{};
  params.radius = 20;
  params.bias = 0.0;
  params.alpha = 1.0;
  params.beta = 0.5;

  LocalResponseNormalization kernel(&input_tensor, &output_tensor, params);
  EXPECT_ANY_THROW(kernel.configure());
}

} // namespace
} // namespace kernels
} // namespace luci_interpreter
