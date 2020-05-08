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

#include "StridedSliceOptions.h"
#include "DataLookup.h"

#include <cassert>

namespace tflite2circle
{

flatbuffers::Offset<circle::StridedSliceOptions>
build_circle_StridedSliceOptions(flatbuffers::FlatBufferBuilder &fb, const tflite::Operator *op)
{
  auto tflite_builtin_options = op->builtin_options_as_StridedSliceOptions();
  assert(tflite_builtin_options);
  circle::StridedSliceOptionsBuilder builtin_options_builder{fb};
  builtin_options_builder.add_begin_mask(tflite_builtin_options->begin_mask());
  builtin_options_builder.add_end_mask(tflite_builtin_options->end_mask());
  builtin_options_builder.add_ellipsis_mask(tflite_builtin_options->ellipsis_mask());
  builtin_options_builder.add_new_axis_mask(tflite_builtin_options->new_axis_mask());
  builtin_options_builder.add_shrink_axis_mask(tflite_builtin_options->shrink_axis_mask());
  return builtin_options_builder.Finish();
}

} // namespace tflite2circle
