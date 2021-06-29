/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

#ifndef __NNFW_CKER_NEON_TENSOR_UTILS_H__
#define __NNFW_CKER_NEON_TENSOR_UTILS_H__

#include <ruy/path.h>
#include <ruy/ruy.h>
#include "cker/Types.h"
#include "cker/neon/neon_check.h"
#include "cker/ruy/RuySupport.h"
#include "util/logging.h"
#if defined __linux__ && defined __aarch64__
#include <sys/auxv.h>
#endif

#include <cassert>
#include <cmath>

#ifdef USE_NEON

#define kFloatWeightsPerNeonLane 4

namespace nnfw
{
namespace cker
{

namespace
{

// These definitions are used for labels within assembly code. Required for
// iOS toolchain compatibility.
#define GEMMLOWP_LABEL_AFTER_LOOP "1"
#define GEMMLOWP_LABEL_LOOP "2"
#define GEMMLOWP_LABEL_ACCUMULATE_EXISTING_DST_VALUES "3"
#define GEMMLOWP_LABEL_STORE "4"

template <class T>
inline T DivideRoundUp(T x, T q) {
  return x / q + T(x % q != 0);
}

template <class T>
inline T RoundUp(T x, T q) {
  return q * DivideRoundUp(x, q);
}

inline static size_t round_down_po2(size_t n, size_t q) {
  assert(q != 0);
  assert((q & (q - 1)) == 0);
  return n & -q;
}

inline static size_t round_up_po2(size_t n, size_t q) {
  return round_down_po2(n + q - 1, q);
}

#if defined(__has_builtin)
  #define XNN_COMPILER_HAS_BUILTIN(builtin) __has_builtin(builtin)
#else
  #define XNN_COMPILER_HAS_BUILTIN(builtin) 0
#endif

#if XNN_COMPILER_HAS_BUILTIN(__builtin_unpredictable)
  #define XNN_UNPREDICTABLE(condition) (__builtin_unpredictable(!!(condition)))
#elif defined(__GNUC__) && (__GNUC__ >= 9) && !defined(__INTEL_COMPILER)
  #define XNN_UNPREDICTABLE(condition) (__builtin_expect_with_probability(!!(condition), 0, 0.5))
#else
  #define XNN_UNPREDICTABLE(condition) (!!(condition))
#endif

#if defined(__GNUC__)
  #define XNN_LIKELY(condition) (__builtin_expect(!!(condition), 1))
  #define XNN_UNLIKELY(condition) (__builtin_expect(!!(condition), 0))
#else
  #define XNN_LIKELY(condition) (!!(condition))
  #define XNN_UNLIKELY(condition) (!!(condition))
#endif

#ifndef __aarch64__
inline int16x8_t vpaddq_s16(int16x8_t a, int16x8_t b) {
  const int16x4_t c = vpadd_s16(vget_low_s16(a), vget_high_s16(a));
  const int16x4_t d = vpadd_s16(vget_low_s16(b), vget_high_s16(b));
  return vcombine_s16(c, d);
}
#endif

#ifndef __aarch64__
inline int32x4_t vpaddq_s32(int32x4_t a, int32x4_t b) {
  int32x4x2_t deinterleaved = vuzpq_s32(a, b);
  return vqaddq_s32(deinterleaved.val[0], deinterleaved.val[1]);
}
#endif  // !__aarch64__

  typedef int32_t AccumulatorType;
 
  static void NEON_GEMM_Int8Operands_AccumTwoWithin16Bits(const int8_t* lhs_ptr, const int8_t* rhs_ptr,
                  int32_t* accum_ptr, int depth) {
    std::size_t start_depth = 123;
    std::size_t run_depth = depth;
    AccumulatorType* dst_ptr = accum_ptr;
    asm volatile(

        // Overview of register layout:
        //
        // A 2x16 block of Rhs is stored in 8 bit in d0--d3.
        // A 4x16 block of Lhs is stored in 8 bit in d4--d7. That is only
        // half of the register space required, so we loop over these registers
        // twice. Only half of it, a 2x16 block, is stored in d4--d7 at
        // any given time.
        //
        // A 4x2 block of accumulators is stored in q8--q15 (as 4x32 bit
        // components which need to be horizontally-added at the end)
        //
        // The Lhs vectors are multiplied by the Rhs vectors with a widening
        // multiply over the 8 first levels of depth, producing int16x8
        // vectors of products for each position in the accumulator matrix.
        // Here comes the special trick: since the operands are signed int8,
        // their range being [ -2^7 , 2^7 ), their products are in range
        // [ -2^14 , 2^14 - 1 ), meaning that we can add two such values
        // without any risk of overflowing int16.
        // We thus proceed with the 8 next levels of depth, multiplying
        // again Lhs by Rhs, accumulating into this existing int16x8 vector.
        //
        // Only then, having processed 16 levels of depth, do we need to
        // horizontally add these int16x8 accumulators into the final
        // int32x4 accumulators.
        //
        // As we do not have enough registers to store all 16 int16x8
        // temporary-16bit-accumulators, we have them cycle through q4--q7.
        //
        //
        // Register layout (ignoring the q4--q7 temporary 16bit accumulators):
        //
        //                               +----+----+
        //                               | d0 | d2 |
        //                               | .  | .  |
        //                               | .  | .  |
        //                               | .  | .  |
        //                       Rhs     +----+----+
        //                               | d1 | d3 |
        //                               | .  | .  |
        //                               | .  | .  |
        //                               | .  | .  |
        //                               +----+----+
        //
        //                               |    |    |
        //
        //    Lhs                        |    |    |
        //
        //  +--------+--------+ - - - -  +----+----+
        //  | d4 ... | d5 ... |          | q8 | q9 |
        //  | d6 ... | d7 ... |          | q10| q11|
        //  | d4 ... | d5 ... |          | q12| q13|
        //  | d6 ... | d7 ... |          | q14| q15|
        //  +--------+--------+ - - - -  +----+----+
        //
        //                               Accumulator
        //

        // Clear accumulators, and, interleaved with it,
        // initial loads of the first loop iteration,
        // taken out of the loop so that in the loop itself we have
        // optimal streaming of data from memory.
        "vldr d0, [%[rhs_ptr], #0]\n"
        "vmov.i32 q8, #0\n"
        "vldr d4, [%[lhs_ptr], #0]\n"
        "vmov.i32 q9, #0\n"
        "vldr d2, [%[rhs_ptr], #16]\n"
        "vmov.i32 q10, q8\n"
        "vldr d6, [%[lhs_ptr], #16]\n"
        "vmov.i32 q11, q8\n"
        "vldr d1, [%[rhs_ptr], #8]\n"
        "vmov.i32 q12, q8\n"
        "vldr d5, [%[lhs_ptr], #8]\n"
        "vmov.i32 q13, q8\n"
        "vldr d3, [%[rhs_ptr], #24]\n"
        "vmov.i32 q14, q8\n"
        "vldr d7, [%[lhs_ptr], #24]\n"
        "vmov.i32 q15, q8\n"

        // General loop.
        GEMMLOWP_LABEL_LOOP
        ":\n"

        // Multiply 8 first levels of depth.
        "vmull.s8    q4,  d0,  d4\n"
        "add %[rhs_ptr], %[rhs_ptr], #32\n"
        "vmull.s8    q5,  d2,  d4\n"
        "vldr d4, [%[lhs_ptr], #32]\n"
        "vmull.s8    q6,  d0,  d6\n"
        "vmull.s8    q7,  d2,  d6\n"
        "vldr d6, [%[lhs_ptr], #48]\n"

        // Multiply-accumulate second-half, again into the same
        // 16bit local accumulator registers. This is where we
        // take advantage of having int8 instead of uint8 and therefore
        // being able to accumulate two products into int16.
        "vmlal.s8    q4,  d1,  d5\n"
        "vmlal.s8    q5,  d3,  d5\n"
        "vldr d5, [%[lhs_ptr], #40]\n"
        "vmlal.s8    q6,  d1,  d7\n"
        "vmlal.s8    q7,  d3,  d7\n"
        "vldr d7, [%[lhs_ptr], #56]\n"

        // Add pairwise, accumulate into 32-bit accumulators.
        "vpadal.s16   q8,  q4\n"
        "add %[lhs_ptr], %[lhs_ptr], #64\n"
        "vpadal.s16   q9,  q5\n"
        "subs %[run_depth], %[run_depth], #16\n"
        "vpadal.s16   q10, q6\n"
        "vpadal.s16   q11, q7\n"

        "beq " GEMMLOWP_LABEL_AFTER_LOOP
        "f\n"

        // Multiply first half.
        "vmull.s8    q4,  d0,  d4\n"
        "vmull.s8    q5,  d2,  d4\n"
        "vldr d4, [%[lhs_ptr], #0]\n"
        "vmull.s8    q6,  d0,  d6\n"
        "vldr d0, [%[rhs_ptr], #0]\n"
        "vmull.s8    q7,  d2,  d6\n"
        "vldr d2, [%[rhs_ptr], #16]\n"

        // Multiply-accumulate second-half, again into the same
        // 16bit local accumulator registers. This is where we
        // take advantage of having int8 instead of uint8 and therefore
        // being able to accumulate two products into int16.
        "vmlal.s8    q4,  d1,  d5\n"
        "vldr d6, [%[lhs_ptr], #16]\n"
        "vmlal.s8    q5,  d3,  d5\n"
        "vldr d5, [%[lhs_ptr], #8]\n"
        "vmlal.s8    q6,  d1,  d7\n"
        "vldr d1, [%[rhs_ptr], #8]\n"
        "vmlal.s8    q7,  d3,  d7\n"
        "vldr d3, [%[rhs_ptr], #24]\n"

        // Add pairwise, accumulate into 32-bit accumulators.
        "vpadal.s16   q12, q4\n"
        "vldr d7, [%[lhs_ptr], #24]\n"
        "vpadal.s16   q13, q5\n"
        "vpadal.s16   q14, q6\n"
        "vpadal.s16   q15, q7\n"

        "b " GEMMLOWP_LABEL_LOOP "b\n"

        GEMMLOWP_LABEL_AFTER_LOOP
        ":\n"

        // Multiply first half.
        "vmull.s8    q4,  d0,  d4\n"
        "vmull.s8    q5,  d2,  d4\n"
        "vmull.s8    q6,  d0,  d6\n"
        "vmull.s8    q7,  d2,  d6\n"

        // Multiply-accumulate second-half, again into the same
        // 16bit local accumulator registers. This is where we
        // take advantage of having int8 instead of uint8 and therefore
        // being able to accumulate two products into int16.
        "vmlal.s8    q4,  d1,  d5\n"
        "vmlal.s8    q5,  d3,  d5\n"
        "vmlal.s8    q6,  d1,  d7\n"
        "vmlal.s8    q7,  d3,  d7\n"

        // Add pairwise, accumulate into 32-bit accumulators.
        "vpadal.s16   q12, q4\n"
        "vpadal.s16   q13, q5\n"
        "vpadal.s16   q14, q6\n"
        "vpadal.s16   q15, q7\n"
        "cmp %[start_depth], #0\n"

        // Reduce 32bit accumulators horizontally.
        "vpadd.s32 d0, d16, d17\n"
        "vpadd.s32 d1, d18, d19\n"
        "vpadd.s32 d2, d20, d21\n"
        "vpadd.s32 d3, d22, d23\n"
        "vpadd.s32 d4, d24, d25\n"
        "vpadd.s32 d5, d26, d27\n"
        "vpadd.s32 d6, d28, d29\n"
        "vpadd.s32 d7, d30, d31\n"

        "bne " GEMMLOWP_LABEL_ACCUMULATE_EXISTING_DST_VALUES
        "f\n"

        // Reduce 32bit accumulators horizontally, second pass
        // (each pass adds pairwise. we need to add 4-wise).
        "vpadd.s32 d8, d0, d2\n"
        "vpadd.s32 d9, d4, d6\n"
        "vpadd.s32 d10, d1, d3\n"
        "vpadd.s32 d11, d5, d7\n"

        "b " GEMMLOWP_LABEL_STORE "f\n"

        GEMMLOWP_LABEL_ACCUMULATE_EXISTING_DST_VALUES
        ":\n"

        // Reduce 32bit accumulators horizontally, second pass
        // (each pass adds pairwise. we need to add 4-wise),
        // and load destination values from memory.
        "mov r0, %[dst_ptr]\n"
        "vld1.32 {d16, d17}, [r0]!\n"
        "vpadd.s32 d8, d0, d2\n"
        "vpadd.s32 d9, d4, d6\n"
        "vld1.32 {d18, d19}, [r0]\n"
        "vpadd.s32 d10, d1, d3\n"
        "vpadd.s32 d11, d5, d7\n"

        // Add horizontally-reduced accumulators into
        // the values loaded from memory
        "vadd.s32 q4, q8, q4\n"
        "vadd.s32 q5, q9, q5\n"

        GEMMLOWP_LABEL_STORE
        ":\n"
        // Store back into memory
        "mov r0, %[dst_ptr]\n"
        "vst1.32 {d8, d9}, [r0]!\n"
        "vst1.32 {d10, d11}, [r0]\n"
        :  // outputs
        [lhs_ptr] "+r"(lhs_ptr), [rhs_ptr] "+r"(rhs_ptr),
        [dst_ptr] "+r"(dst_ptr), [run_depth] "+r"(run_depth)
        :  // inputs
        [start_depth] "r"(start_depth)
        :  // clobbers
        "cc", "memory", "r0", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7",
        "d8", "d9", "d10", "d11", "d12", "d13", "d14", "d15", "d16", "d17",
        "d18", "d19", "d20", "d21", "d22", "d23", "d24", "d25", "d26", "d27",
        "d28", "d29", "d30", "d31");
  }


void NEON_GEMM_Int8Operands_AccumTwoWithin16Bits_intrinsics(const int8_t* lhs_ptr, const int8_t* rhs_ptr,
         int32_t* accum_ptr, int depth) {
  int32x4_t acc[4][2];
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 2; j++) {
      acc[i][j] = vdupq_n_s32(0);
    }
  }
  for (int d = 0; d < depth; d += 16) {
    int8x16_t lhs[4];
    for (int i = 0; i < 4; i++) {
      lhs[i] = vld1q_s8(lhs_ptr + 16 * i);
    }
    int8x16_t rhs[2];
    for (int i = 0; i < 2; i++) {
      rhs[i] = vld1q_s8(rhs_ptr + 16 * i);
    }
    for (int i = 0; i < 4; i++) {
      for (int j = 0; j < 2; j++) {
        int16x8_t local_acc =
          vmull_s8(vget_low_s8(lhs[i]), vget_low_s8(rhs[j]));
        local_acc =
          vmlal_s8(local_acc, vget_high_s8(lhs[i]), vget_high_s8(rhs[j]));
        acc[i][j] = vpadalq_s16(acc[i][j], local_acc);
      }
    }
    lhs_ptr += 64;
    rhs_ptr += 16 * 2;
  }
  for (int i = 0; i < 2; i++) {
    int32x4_t acc_2x_0 = vpaddq_s32(acc[0][i], acc[1][i]);
    int32x4_t acc_2x_1 = vpaddq_s32(acc[2][i], acc[3][i]);
    int32x4_t acc_4x = vpaddq_s32(acc_2x_0, acc_2x_1);
    int32x4_t dst_val = vld1q_s32(accum_ptr + 4 * i);
    dst_val = vaddq_s32(dst_val, acc_4x);
    vst1q_s32(accum_ptr + 4 * i, dst_val);
  }
}

void xnn_qs8_gemm_minmax_ukernel_1x16c2__neon_mlal_padal_dup(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* __restrict__ a,
    size_t a_stride,
    const void* __restrict__ w,
    int8_t* __restrict__ c,
    size_t cm_stride,
    size_t cn_stride)
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 2);
  const int8_t* a0 = a;
  int8_t* c0 = c;

  do {
    int32x4_t vacc0x0123 = vld1q_s32((int32_t*)w); //w = (const void*) ((uintptr_t) w + 4 * sizeof(int32_t));
    int32x4_t vacc0x4567 = vld1q_s32((int32_t*)w); //w = (const void*) ((uintptr_t) w + 4 * sizeof(int32_t));
    int32x4_t vacc0x89AB = vld1q_s32((int32_t*)w); //w = (const void*) ((uintptr_t) w + 4 * sizeof(int32_t));
    int32x4_t vacc0xCDEF = vld1q_s32((int32_t*)w); //w = (const void*) ((uintptr_t) w + 4 * sizeof(int32_t));

    size_t k = kc;

    while (k >= 16 * sizeof(int8_t)) {
      const int8x8_t va0x0 = vld1_s8(a0); a0 += 8;
      const int8x8_t va0x1 = vld1_s8(a0); a0 += 8;

      const int8x8_t vb0123c0x0 = vld1_s8((int8_t*)w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
      const int8x8_t vb4567c0x0 = vld1_s8((int8_t*)w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
      const int8x8_t vb89ABc0x0 = vld1_s8((int8_t*)w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
      const int8x8_t vbCDEFc0x0 = vld1_s8((int8_t*)w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
      const int8x8_t vb0123c1x0 = vld1_s8((int8_t*)w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
      const int8x8_t vb4567c1x0 = vld1_s8((int8_t*)w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
      const int8x8_t vb89ABc1x0 = vld1_s8((int8_t*)w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
      const int8x8_t vbCDEFc1x0 = vld1_s8((int8_t*)w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
      const int8x8_t vb0123c2x0 = vld1_s8((int8_t*)w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
      const int8x8_t vb4567c2x0 = vld1_s8((int8_t*)w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
      const int8x8_t vb89ABc2x0 = vld1_s8((int8_t*)w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
      const int8x8_t vbCDEFc2x0 = vld1_s8((int8_t*)w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
      const int8x8_t vb0123c3x0 = vld1_s8((int8_t*)w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
      const int8x8_t vb4567c3x0 = vld1_s8((int8_t*)w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
      const int8x8_t vb89ABc3x0 = vld1_s8((int8_t*)w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
      const int8x8_t vbCDEFc3x0 = vld1_s8((int8_t*)w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));

      int16x8_t vprod0x0123c0 = vmull_s8(vb0123c0x0, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0x0), 0)));
      const int8x8_t vb0123c0x1 = vld1_s8((int8_t*)w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
      vprod0x0123c0 = vmlal_s8(vprod0x0123c0, vb0123c0x1, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0x1), 0)));
      vacc0x0123 = vpadalq_s16(vacc0x0123, vprod0x0123c0);
      int16x8_t vprod0x4567c0 = vmull_s8(vb4567c0x0, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0x0), 0)));
      const int8x8_t vb4567c0x1 = vld1_s8((int8_t*)w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
      vprod0x4567c0 = vmlal_s8(vprod0x4567c0, vb4567c0x1, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0x1), 0)));
      vacc0x4567 = vpadalq_s16(vacc0x4567, vprod0x4567c0);
      int16x8_t vprod0x89ABc0 = vmull_s8(vb89ABc0x0, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0x0), 0)));
      const int8x8_t vb89ABc0x1 = vld1_s8((int8_t*)w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
      vprod0x89ABc0 = vmlal_s8(vprod0x89ABc0, vb89ABc0x1, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0x1), 0)));
      vacc0x89AB = vpadalq_s16(vacc0x89AB, vprod0x89ABc0);
      int16x8_t vprod0xCDEFc0 = vmull_s8(vbCDEFc0x0, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0x0), 0)));
      const int8x8_t vbCDEFc0x1 = vld1_s8((int8_t*)w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
      vprod0xCDEFc0 = vmlal_s8(vprod0xCDEFc0, vbCDEFc0x1, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0x1), 0)));
      vacc0xCDEF = vpadalq_s16(vacc0xCDEF, vprod0xCDEFc0);
      int16x8_t vprod0x0123c1 = vmull_s8(vb0123c1x0, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0x0), 1)));
      const int8x8_t vb0123c1x1 = vld1_s8((int8_t*)w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
      vprod0x0123c1 = vmlal_s8(vprod0x0123c1, vb0123c1x1, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0x1), 1)));
      vacc0x0123 = vpadalq_s16(vacc0x0123, vprod0x0123c1);
      int16x8_t vprod0x4567c1 = vmull_s8(vb4567c1x0, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0x0), 1)));
      const int8x8_t vb4567c1x1 = vld1_s8((int8_t*)w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
      vprod0x4567c1 = vmlal_s8(vprod0x4567c1, vb4567c1x1, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0x1), 1)));
      vacc0x4567 = vpadalq_s16(vacc0x4567, vprod0x4567c1);
      int16x8_t vprod0x89ABc1 = vmull_s8(vb89ABc1x0, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0x0), 1)));
      const int8x8_t vb89ABc1x1 = vld1_s8((int8_t*)w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
      vprod0x89ABc1 = vmlal_s8(vprod0x89ABc1, vb89ABc1x1, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0x1), 1)));
      vacc0x89AB = vpadalq_s16(vacc0x89AB, vprod0x89ABc1);
      int16x8_t vprod0xCDEFc1 = vmull_s8(vbCDEFc1x0, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0x0), 1)));
      const int8x8_t vbCDEFc1x1 = vld1_s8((int8_t*)w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
      vprod0xCDEFc1 = vmlal_s8(vprod0xCDEFc1, vbCDEFc1x1, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0x1), 1)));
      vacc0xCDEF = vpadalq_s16(vacc0xCDEF, vprod0xCDEFc1);
      int16x8_t vprod0x0123c2 = vmull_s8(vb0123c2x0, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0x0), 2)));
      const int8x8_t vb0123c2x1 = vld1_s8((int8_t*)w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
      vprod0x0123c2 = vmlal_s8(vprod0x0123c2, vb0123c2x1, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0x1), 2)));
      vacc0x0123 = vpadalq_s16(vacc0x0123, vprod0x0123c2);
      int16x8_t vprod0x4567c2 = vmull_s8(vb4567c2x0, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0x0), 2)));
      const int8x8_t vb4567c2x1 = vld1_s8((int8_t*)w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
      vprod0x4567c2 = vmlal_s8(vprod0x4567c2, vb4567c2x1, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0x1), 2)));
      vacc0x4567 = vpadalq_s16(vacc0x4567, vprod0x4567c2);
      int16x8_t vprod0x89ABc2 = vmull_s8(vb89ABc2x0, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0x0), 2)));
      const int8x8_t vb89ABc2x1 = vld1_s8((int8_t*)w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
      vprod0x89ABc2 = vmlal_s8(vprod0x89ABc2, vb89ABc2x1, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0x1), 2)));
      vacc0x89AB = vpadalq_s16(vacc0x89AB, vprod0x89ABc2);
      int16x8_t vprod0xCDEFc2 = vmull_s8(vbCDEFc2x0, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0x0), 2)));
      const int8x8_t vbCDEFc2x1 = vld1_s8((int8_t*)w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
      vprod0xCDEFc2 = vmlal_s8(vprod0xCDEFc2, vbCDEFc2x1, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0x1), 2)));
      vacc0xCDEF = vpadalq_s16(vacc0xCDEF, vprod0xCDEFc2);
      int16x8_t vprod0x0123c3 = vmull_s8(vb0123c3x0, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0x0), 3)));
      const int8x8_t vb0123c3x1 = vld1_s8((int8_t*)w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
      vprod0x0123c3 = vmlal_s8(vprod0x0123c3, vb0123c3x1, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0x1), 3)));
      vacc0x0123 = vpadalq_s16(vacc0x0123, vprod0x0123c3);
      int16x8_t vprod0x4567c3 = vmull_s8(vb4567c3x0, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0x0), 3)));
      const int8x8_t vb4567c3x1 = vld1_s8((int8_t*)w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
      vprod0x4567c3 = vmlal_s8(vprod0x4567c3, vb4567c3x1, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0x1), 3)));
      vacc0x4567 = vpadalq_s16(vacc0x4567, vprod0x4567c3);
      int16x8_t vprod0x89ABc3 = vmull_s8(vb89ABc3x0, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0x0), 3)));
      const int8x8_t vb89ABc3x1 = vld1_s8((int8_t*)w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
      vprod0x89ABc3 = vmlal_s8(vprod0x89ABc3, vb89ABc3x1, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0x1), 3)));
      vacc0x89AB = vpadalq_s16(vacc0x89AB, vprod0x89ABc3);
      int16x8_t vprod0xCDEFc3 = vmull_s8(vbCDEFc3x0, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0x0), 3)));
      const int8x8_t vbCDEFc3x1 = vld1_s8((int8_t*)w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
      vprod0xCDEFc3 = vmlal_s8(vprod0xCDEFc3, vbCDEFc3x1, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0x1), 3)));
      vacc0xCDEF = vpadalq_s16(vacc0xCDEF, vprod0xCDEFc3);

      k -= 16 * sizeof(int8_t);
    }

    if (k >= 8 * sizeof(int8_t)) {
      const int8x8_t va0 = vld1_s8(a0); a0 += 8;

      const int8x8_t vb0123c0 = vld1_s8((int8_t*)w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
      const int8x8_t vb4567c0 = vld1_s8((int8_t*)w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
      const int8x8_t vb89ABc0 = vld1_s8((int8_t*)w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
      const int8x8_t vbCDEFc0 = vld1_s8((int8_t*)w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
      const int8x8_t vb0123c1 = vld1_s8((int8_t*)w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
      const int8x8_t vb4567c1 = vld1_s8((int8_t*)w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
      const int8x8_t vb89ABc1 = vld1_s8((int8_t*)w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
      const int8x8_t vbCDEFc1 = vld1_s8((int8_t*)w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
      const int8x8_t vb0123c2 = vld1_s8((int8_t*)w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
      const int8x8_t vb4567c2 = vld1_s8((int8_t*)w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
      const int8x8_t vb89ABc2 = vld1_s8((int8_t*)w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
      const int8x8_t vbCDEFc2 = vld1_s8((int8_t*)w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
      const int8x8_t vb0123c3 = vld1_s8((int8_t*)w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
      const int8x8_t vb4567c3 = vld1_s8((int8_t*)w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
      const int8x8_t vb89ABc3 = vld1_s8((int8_t*)w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
      const int8x8_t vbCDEFc3 = vld1_s8((int8_t*)w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));

      const int16x8_t vprod0x0123c0 = vmull_s8(vb0123c0, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0), 0)));
      const int16x8_t vprod0x0123c1 = vmull_s8(vb0123c1, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0), 1)));
      const int16x8_t vprod0x0123c2 = vmull_s8(vb0123c2, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0), 2)));
      const int16x8_t vprod0x0123c3 = vmull_s8(vb0123c3, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0), 3)));
      vacc0x0123 = vpadalq_s16(vacc0x0123, vprod0x0123c0);
      vacc0x0123 = vpadalq_s16(vacc0x0123, vprod0x0123c1);
      vacc0x0123 = vpadalq_s16(vacc0x0123, vprod0x0123c2);
      vacc0x0123 = vpadalq_s16(vacc0x0123, vprod0x0123c3);
      const int16x8_t vprod0x4567c0 = vmull_s8(vb4567c0, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0), 0)));
      const int16x8_t vprod0x4567c1 = vmull_s8(vb4567c1, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0), 1)));
      const int16x8_t vprod0x4567c2 = vmull_s8(vb4567c2, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0), 2)));
      const int16x8_t vprod0x4567c3 = vmull_s8(vb4567c3, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0), 3)));
      vacc0x4567 = vpadalq_s16(vacc0x4567, vprod0x4567c0);
      vacc0x4567 = vpadalq_s16(vacc0x4567, vprod0x4567c1);
      vacc0x4567 = vpadalq_s16(vacc0x4567, vprod0x4567c2);
      vacc0x4567 = vpadalq_s16(vacc0x4567, vprod0x4567c3);
      const int16x8_t vprod0x89ABc0 = vmull_s8(vb89ABc0, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0), 0)));
      const int16x8_t vprod0x89ABc1 = vmull_s8(vb89ABc1, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0), 1)));
      const int16x8_t vprod0x89ABc2 = vmull_s8(vb89ABc2, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0), 2)));
      const int16x8_t vprod0x89ABc3 = vmull_s8(vb89ABc3, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0), 3)));
      vacc0x89AB = vpadalq_s16(vacc0x89AB, vprod0x89ABc0);
      vacc0x89AB = vpadalq_s16(vacc0x89AB, vprod0x89ABc1);
      vacc0x89AB = vpadalq_s16(vacc0x89AB, vprod0x89ABc2);
      vacc0x89AB = vpadalq_s16(vacc0x89AB, vprod0x89ABc3);
      const int16x8_t vprod0xCDEFc0 = vmull_s8(vbCDEFc0, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0), 0)));
      const int16x8_t vprod0xCDEFc1 = vmull_s8(vbCDEFc1, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0), 1)));
      const int16x8_t vprod0xCDEFc2 = vmull_s8(vbCDEFc2, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0), 2)));
      const int16x8_t vprod0xCDEFc3 = vmull_s8(vbCDEFc3, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0), 3)));
      vacc0xCDEF = vpadalq_s16(vacc0xCDEF, vprod0xCDEFc0);
      vacc0xCDEF = vpadalq_s16(vacc0xCDEF, vprod0xCDEFc1);
      vacc0xCDEF = vpadalq_s16(vacc0xCDEF, vprod0xCDEFc2);
      vacc0xCDEF = vpadalq_s16(vacc0xCDEF, vprod0xCDEFc3);

      k -= 8 * sizeof(int8_t);
    }

    if XNN_UNLIKELY(k != 0) {
      const int8x8_t va0 = vld1_s8(a0); a0 = (const int8_t*) ((uintptr_t) a0 + k);

      const int8x8_t vb0123c0 = vld1_s8((int8_t*)w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
      const int8x8_t vb4567c0 = vld1_s8((int8_t*)w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
      const int8x8_t vb89ABc0 = vld1_s8((int8_t*)w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
      const int8x8_t vbCDEFc0 = vld1_s8((int8_t*)w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));

      const int16x8_t vprod0x0123c0 = vmull_s8(vb0123c0, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0), 0)));
      vacc0x0123 = vpadalq_s16(vacc0x0123, vprod0x0123c0);
      const int16x8_t vprod0x4567c0 = vmull_s8(vb4567c0, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0), 0)));
      vacc0x4567 = vpadalq_s16(vacc0x4567, vprod0x4567c0);
      const int16x8_t vprod0x89ABc0 = vmull_s8(vb89ABc0, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0), 0)));
      vacc0x89AB = vpadalq_s16(vacc0x89AB, vprod0x89ABc0);
      const int16x8_t vprod0xCDEFc0 = vmull_s8(vbCDEFc0, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0), 0)));
      vacc0xCDEF = vpadalq_s16(vacc0xCDEF, vprod0xCDEFc0);

      if (k > 2 * sizeof(int8_t)) {
        const int8x8_t vb0123c1 = vld1_s8((int8_t*)w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
        const int8x8_t vb4567c1 = vld1_s8((int8_t*)w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
        const int8x8_t vb89ABc1 = vld1_s8((int8_t*)w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
        const int8x8_t vbCDEFc1 = vld1_s8((int8_t*)w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));

        const int16x8_t vprod0x0123c1 = vmull_s8(vb0123c1, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0), 1)));
        vacc0x0123 = vpadalq_s16(vacc0x0123, vprod0x0123c1);
        const int16x8_t vprod0x4567c1 = vmull_s8(vb4567c1, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0), 1)));
        vacc0x4567 = vpadalq_s16(vacc0x4567, vprod0x4567c1);
        const int16x8_t vprod0x89ABc1 = vmull_s8(vb89ABc1, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0), 1)));
        vacc0x89AB = vpadalq_s16(vacc0x89AB, vprod0x89ABc1);
        const int16x8_t vprod0xCDEFc1 = vmull_s8(vbCDEFc1, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0), 1)));
        vacc0xCDEF = vpadalq_s16(vacc0xCDEF, vprod0xCDEFc1);

        if (k > 4 * sizeof(int8_t)) {
          const int8x8_t vb0123c2 = vld1_s8((int8_t*)w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
          const int8x8_t vb4567c2 = vld1_s8((int8_t*)w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
          const int8x8_t vb89ABc2 = vld1_s8((int8_t*)w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
          const int8x8_t vbCDEFc2 = vld1_s8((int8_t*)w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));

          const int16x8_t vprod0x0123c2 = vmull_s8(vb0123c2, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0), 2)));
          vacc0x0123 = vpadalq_s16(vacc0x0123, vprod0x0123c2);
          const int16x8_t vprod0x4567c2 = vmull_s8(vb4567c2, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0), 2)));
          vacc0x4567 = vpadalq_s16(vacc0x4567, vprod0x4567c2);
          const int16x8_t vprod0x89ABc2 = vmull_s8(vb89ABc2, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0), 2)));
          vacc0x89AB = vpadalq_s16(vacc0x89AB, vprod0x89ABc2);
          const int16x8_t vprod0xCDEFc2 = vmull_s8(vbCDEFc2, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0), 2)));
          vacc0xCDEF = vpadalq_s16(vacc0xCDEF, vprod0xCDEFc2);
        }
      }
    }
    const int32x4_t vmultiplier = {0,};//vld1q_dup_s32(&params->neon.multiplier);
    vacc0x0123 = vqrdmulhq_s32(vacc0x0123, vmultiplier);
    vacc0x4567 = vqrdmulhq_s32(vacc0x4567, vmultiplier);
    vacc0x89AB = vqrdmulhq_s32(vacc0x89AB, vmultiplier);
    vacc0xCDEF = vqrdmulhq_s32(vacc0xCDEF, vmultiplier);

    const int32x4_t vright_shift = {0,};//vld1q_dup_s32(&params->neon.right_shift);
    const int32x4_t vzero_shift_mask = vreinterpretq_s32_u32(vceqq_s32(vright_shift, vmovq_n_s32(0)));
    vacc0x0123 = vsraq_n_s32(vacc0x0123, vbicq_s32(vacc0x0123, vzero_shift_mask), 31);
    vacc0x4567 = vsraq_n_s32(vacc0x4567, vbicq_s32(vacc0x4567, vzero_shift_mask), 31);
    vacc0x89AB = vsraq_n_s32(vacc0x89AB, vbicq_s32(vacc0x89AB, vzero_shift_mask), 31);
    vacc0xCDEF = vsraq_n_s32(vacc0xCDEF, vbicq_s32(vacc0xCDEF, vzero_shift_mask), 31);

    vacc0x0123 = vrshlq_s32(vacc0x0123, vright_shift);
    vacc0x4567 = vrshlq_s32(vacc0x4567, vright_shift);
    vacc0x89AB = vrshlq_s32(vacc0x89AB, vright_shift);
    vacc0xCDEF = vrshlq_s32(vacc0xCDEF, vright_shift);

    const int16x8_t voutput_zero_point = {0,};//vld1q_dup_s16(&params->neon.output_zero_point);
#if XNN_ARCH_ARM64
    const int16x8_t vacc0x01234567 = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc0x0123), vacc0x4567), voutput_zero_point);
    const int16x8_t vacc0x89ABCDEF = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc0x89AB), vacc0xCDEF), voutput_zero_point);

    int8x16_t vout0x0123456789ABCDEF = vqmovn_high_s16(vqmovn_s16(vacc0x01234567), vacc0x89ABCDEF);
#else
    const int16x8_t vacc0x01234567 = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc0x0123), vqmovn_s32(vacc0x4567)), voutput_zero_point);
    const int16x8_t vacc0x89ABCDEF = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc0x89AB), vqmovn_s32(vacc0xCDEF)), voutput_zero_point);

    int8x16_t vout0x0123456789ABCDEF = vcombine_s8(vqmovn_s16(vacc0x01234567), vqmovn_s16(vacc0x89ABCDEF));
#endif
    const int8x16_t voutput_min = {0,};//vld1q_dup_s8(&params->neon.output_min);
    const int8x16_t voutput_max = {0,};//vld1q_dup_s8(&params->neon.output_max);

    vout0x0123456789ABCDEF = vmaxq_s8(vout0x0123456789ABCDEF, voutput_min);

    vout0x0123456789ABCDEF = vminq_s8(vout0x0123456789ABCDEF, voutput_max);

    if (nc >= 16) {
      //vst1q_s8(c0 + 0, vout0x0123456789ABCDEF);

      c0 = (int8_t*) ((uintptr_t) c0 + cn_stride);

      a0 = (const int8_t*) ((uintptr_t) a0 - kc);

      nc -= 16;
    } else {
      int8x8_t vout0x01234567;// = vget_low_s8(vout0x0123456789ABCDEF);
      if (nc & 8) {
        vst1_s8(c0, vout0x01234567); c0 += 8;
        vout0x01234567 = vget_high_s8(vout0x0123456789ABCDEF);
      }
      if (nc & 4) {
        vst1_lane_u32((uint32_t*)__builtin_assume_aligned(c0, 1), vreinterpret_u32_s8(vout0x01234567), 0); c0 += 4;
        vout0x01234567 = vext_s8(vout0x01234567, vout0x01234567, 4);
      }
      if (nc & 2) {
        vst1_lane_u16((uint16_t*)__builtin_assume_aligned(c0, 1), vreinterpret_u16_s8(vout0x01234567), 0); c0 += 2;
        vout0x01234567 = vext_s8(vout0x01234567, vout0x01234567, 2);
      }
      if (nc & 1) {
        vst1_lane_s8(c0, vout0x01234567, 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}


  inline void xnn_qs8_gemm_minmax_ukernel_2x8c2__neon_mlal_padal_dup(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* __restrict__ a,
    size_t a_stride,
    const void* __restrict__ w,
    int8_t* __restrict__ c,
    size_t cm_stride,
    size_t cn_stride)
{
  assert(mr != 0);
  assert(mr <= 2);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 2);
  const int8_t* a0 = a;
  int8_t* c0 = c;
  const int8_t* a1 = (const int8_t*) ((uintptr_t) a0 + a_stride);
  int8_t* c1 = (int8_t*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 2) {
    a1 = a0;
    c1 = c0;
  }

  do {
    int32x4_t vacc0x0123 = vld1q_s32((int32_t*)w); //w = (const void*) ((uintptr_t) w + 4 * sizeof(int32_t));
    int32x4_t vacc0x4567 = vld1q_s32((int32_t*)w); //w = (const void*) ((uintptr_t) w + 4 * sizeof(int32_t));
    int32x4_t vacc1x0123 = vacc0x0123;
    int32x4_t vacc1x4567 = vacc0x4567;

    size_t k = kc;

    while (k >= 16 * sizeof(int8_t)) {
      const int8x8_t va0x0 = vld1_s8(a0); a0 += 8;
      const int8x8_t va0x1 = vld1_s8(a0); a0 += 8;
      const int8x8_t va1x0 = vld1_s8(a1); a1 += 8;
      const int8x8_t va1x1 = vld1_s8(a1); a1 += 8;

      const int8x8_t vb0123c0x0 = vld1_s8((int8_t*)w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
      const int8x8_t vb4567c0x0 = vld1_s8((int8_t*)w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
      const int8x8_t vb0123c1x0 = vld1_s8((int8_t*)w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
      const int8x8_t vb4567c1x0 = vld1_s8((int8_t*)w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
      const int8x8_t vb0123c2x0 = vld1_s8((int8_t*)w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
      const int8x8_t vb4567c2x0 = vld1_s8((int8_t*)w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
      const int8x8_t vb0123c3x0 = vld1_s8((int8_t*)w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
      const int8x8_t vb4567c3x0 = vld1_s8((int8_t*)w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));

      int16x8_t vprod0x0123c0 = vmull_s8(vb0123c0x0, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0x0), 0)));
      int16x8_t vprod1x0123c0 = vmull_s8(vb0123c0x0, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va1x0), 0)));
      const int8x8_t vb0123c0x1 = vld1_s8((int8_t*)w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
      vprod0x0123c0 = vmlal_s8(vprod0x0123c0, vb0123c0x1, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0x1), 0)));
      vprod1x0123c0 = vmlal_s8(vprod1x0123c0, vb0123c0x1, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va1x1), 0)));
      vacc0x0123 = vpadalq_s16(vacc0x0123, vprod0x0123c0);
      vacc1x0123 = vpadalq_s16(vacc1x0123, vprod1x0123c0);
      int16x8_t vprod0x4567c0 = vmull_s8(vb4567c0x0, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0x0), 0)));
      int16x8_t vprod1x4567c0 = vmull_s8(vb4567c0x0, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va1x0), 0)));
      const int8x8_t vb4567c0x1 = vld1_s8((int8_t*)w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
      vprod0x4567c0 = vmlal_s8(vprod0x4567c0, vb4567c0x1, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0x1), 0)));
      vprod1x4567c0 = vmlal_s8(vprod1x4567c0, vb4567c0x1, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va1x1), 0)));
      vacc0x4567 = vpadalq_s16(vacc0x4567, vprod0x4567c0);
      vacc1x4567 = vpadalq_s16(vacc1x4567, vprod1x4567c0);
      int16x8_t vprod0x0123c1 = vmull_s8(vb0123c1x0, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0x0), 1)));
      int16x8_t vprod1x0123c1 = vmull_s8(vb0123c1x0, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va1x0), 1)));
      const int8x8_t vb0123c1x1 = vld1_s8((int8_t*)w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
      vprod0x0123c1 = vmlal_s8(vprod0x0123c1, vb0123c1x1, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0x1), 1)));
      vprod1x0123c1 = vmlal_s8(vprod1x0123c1, vb0123c1x1, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va1x1), 1)));
      vacc0x0123 = vpadalq_s16(vacc0x0123, vprod0x0123c1);
      vacc1x0123 = vpadalq_s16(vacc1x0123, vprod1x0123c1);
      int16x8_t vprod0x4567c1 = vmull_s8(vb4567c1x0, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0x0), 1)));
      int16x8_t vprod1x4567c1 = vmull_s8(vb4567c1x0, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va1x0), 1)));
      const int8x8_t vb4567c1x1 = vld1_s8((int8_t*)w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
      vprod0x4567c1 = vmlal_s8(vprod0x4567c1, vb4567c1x1, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0x1), 1)));
      vprod1x4567c1 = vmlal_s8(vprod1x4567c1, vb4567c1x1, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va1x1), 1)));
      vacc0x4567 = vpadalq_s16(vacc0x4567, vprod0x4567c1);
      vacc1x4567 = vpadalq_s16(vacc1x4567, vprod1x4567c1);
      int16x8_t vprod0x0123c2 = vmull_s8(vb0123c2x0, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0x0), 2)));
      int16x8_t vprod1x0123c2 = vmull_s8(vb0123c2x0, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va1x0), 2)));
      const int8x8_t vb0123c2x1 = vld1_s8((int8_t*)w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
      vprod0x0123c2 = vmlal_s8(vprod0x0123c2, vb0123c2x1, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0x1), 2)));
      vprod1x0123c2 = vmlal_s8(vprod1x0123c2, vb0123c2x1, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va1x1), 2)));
      vacc0x0123 = vpadalq_s16(vacc0x0123, vprod0x0123c2);
      vacc1x0123 = vpadalq_s16(vacc1x0123, vprod1x0123c2);
      int16x8_t vprod0x4567c2 = vmull_s8(vb4567c2x0, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0x0), 2)));
      int16x8_t vprod1x4567c2 = vmull_s8(vb4567c2x0, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va1x0), 2)));
      const int8x8_t vb4567c2x1 = vld1_s8((int8_t*)w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
      vprod0x4567c2 = vmlal_s8(vprod0x4567c2, vb4567c2x1, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0x1), 2)));
      vprod1x4567c2 = vmlal_s8(vprod1x4567c2, vb4567c2x1, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va1x1), 2)));
      vacc0x4567 = vpadalq_s16(vacc0x4567, vprod0x4567c2);
      vacc1x4567 = vpadalq_s16(vacc1x4567, vprod1x4567c2);
      int16x8_t vprod0x0123c3 = vmull_s8(vb0123c3x0, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0x0), 3)));
      int16x8_t vprod1x0123c3 = vmull_s8(vb0123c3x0, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va1x0), 3)));
      const int8x8_t vb0123c3x1 = vld1_s8((int8_t*)w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
      vprod0x0123c3 = vmlal_s8(vprod0x0123c3, vb0123c3x1, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0x1), 3)));
      vprod1x0123c3 = vmlal_s8(vprod1x0123c3, vb0123c3x1, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va1x1), 3)));
      vacc0x0123 = vpadalq_s16(vacc0x0123, vprod0x0123c3);
      vacc1x0123 = vpadalq_s16(vacc1x0123, vprod1x0123c3);
      int16x8_t vprod0x4567c3 = vmull_s8(vb4567c3x0, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0x0), 3)));
      int16x8_t vprod1x4567c3 = vmull_s8(vb4567c3x0, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va1x0), 3)));
      const int8x8_t vb4567c3x1 = vld1_s8((int8_t*)w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
      vprod0x4567c3 = vmlal_s8(vprod0x4567c3, vb4567c3x1, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0x1), 3)));
      vprod1x4567c3 = vmlal_s8(vprod1x4567c3, vb4567c3x1, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va1x1), 3)));
      vacc0x4567 = vpadalq_s16(vacc0x4567, vprod0x4567c3);
      vacc1x4567 = vpadalq_s16(vacc1x4567, vprod1x4567c3);

      k -= 16 * sizeof(int8_t);
    }

    if (k >= 8 * sizeof(int8_t)) {
      const int8x8_t va0 = vld1_s8(a0); a0 += 8;
      const int8x8_t va1 = vld1_s8(a1); a1 += 8;

      const int8x8_t vb0123c0 = vld1_s8((int8_t*)w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
      const int8x8_t vb4567c0 = vld1_s8((int8_t*)w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
      const int8x8_t vb0123c1 = vld1_s8((int8_t*)w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
      const int8x8_t vb4567c1 = vld1_s8((int8_t*)w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
      const int8x8_t vb0123c2 = vld1_s8((int8_t*)w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
      const int8x8_t vb4567c2 = vld1_s8((int8_t*)w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
      const int8x8_t vb0123c3 = vld1_s8((int8_t*)w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
      const int8x8_t vb4567c3 = vld1_s8((int8_t*)w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));

      const int16x8_t vprod0x0123c0 = vmull_s8(vb0123c0, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0), 0)));
      const int16x8_t vprod0x0123c1 = vmull_s8(vb0123c1, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0), 1)));
      const int16x8_t vprod0x0123c2 = vmull_s8(vb0123c2, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0), 2)));
      const int16x8_t vprod0x0123c3 = vmull_s8(vb0123c3, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0), 3)));
      vacc0x0123 = vpadalq_s16(vacc0x0123, vprod0x0123c0);
      vacc0x0123 = vpadalq_s16(vacc0x0123, vprod0x0123c1);
      vacc0x0123 = vpadalq_s16(vacc0x0123, vprod0x0123c2);
      vacc0x0123 = vpadalq_s16(vacc0x0123, vprod0x0123c3);
      const int16x8_t vprod0x4567c0 = vmull_s8(vb4567c0, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0), 0)));
      const int16x8_t vprod0x4567c1 = vmull_s8(vb4567c1, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0), 1)));
      const int16x8_t vprod0x4567c2 = vmull_s8(vb4567c2, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0), 2)));
      const int16x8_t vprod0x4567c3 = vmull_s8(vb4567c3, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0), 3)));
      vacc0x4567 = vpadalq_s16(vacc0x4567, vprod0x4567c0);
      vacc0x4567 = vpadalq_s16(vacc0x4567, vprod0x4567c1);
      vacc0x4567 = vpadalq_s16(vacc0x4567, vprod0x4567c2);
      vacc0x4567 = vpadalq_s16(vacc0x4567, vprod0x4567c3);
      const int16x8_t vprod1x0123c0 = vmull_s8(vb0123c0, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va1), 0)));
      const int16x8_t vprod1x0123c1 = vmull_s8(vb0123c1, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va1), 1)));
      const int16x8_t vprod1x0123c2 = vmull_s8(vb0123c2, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va1), 2)));
      const int16x8_t vprod1x0123c3 = vmull_s8(vb0123c3, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va1), 3)));
      vacc1x0123 = vpadalq_s16(vacc1x0123, vprod1x0123c0);
      vacc1x0123 = vpadalq_s16(vacc1x0123, vprod1x0123c1);
      vacc1x0123 = vpadalq_s16(vacc1x0123, vprod1x0123c2);
      vacc1x0123 = vpadalq_s16(vacc1x0123, vprod1x0123c3);
      const int16x8_t vprod1x4567c0 = vmull_s8(vb4567c0, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va1), 0)));
      const int16x8_t vprod1x4567c1 = vmull_s8(vb4567c1, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va1), 1)));
      const int16x8_t vprod1x4567c2 = vmull_s8(vb4567c2, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va1), 2)));
      const int16x8_t vprod1x4567c3 = vmull_s8(vb4567c3, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va1), 3)));
      vacc1x4567 = vpadalq_s16(vacc1x4567, vprod1x4567c0);
      vacc1x4567 = vpadalq_s16(vacc1x4567, vprod1x4567c1);
      vacc1x4567 = vpadalq_s16(vacc1x4567, vprod1x4567c2);
      vacc1x4567 = vpadalq_s16(vacc1x4567, vprod1x4567c3);

      k -= 8 * sizeof(int8_t);
    }

    if XNN_UNLIKELY(k != 0) {
      const int8x8_t va0 = vld1_s8(a0); a0 = (const int8_t*) ((uintptr_t) a0 + k);
      const int8x8_t va1 = vld1_s8(a1); a1 = (const int8_t*) ((uintptr_t) a1 + k);

      const int8x8_t vb0123c0 = vld1_s8((int8_t*)w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
      const int8x8_t vb4567c0 = vld1_s8((int8_t*)w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));

      const int16x8_t vprod0x0123c0 = vmull_s8(vb0123c0, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0), 0)));
      vacc0x0123 = vpadalq_s16(vacc0x0123, vprod0x0123c0);
      const int16x8_t vprod0x4567c0 = vmull_s8(vb4567c0, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0), 0)));
      vacc0x4567 = vpadalq_s16(vacc0x4567, vprod0x4567c0);
      const int16x8_t vprod1x0123c0 = vmull_s8(vb0123c0, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va1), 0)));
      vacc1x0123 = vpadalq_s16(vacc1x0123, vprod1x0123c0);
      const int16x8_t vprod1x4567c0 = vmull_s8(vb4567c0, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va1), 0)));
      vacc1x4567 = vpadalq_s16(vacc1x4567, vprod1x4567c0);

      if (k > 2 * sizeof(int8_t)) {
        const int8x8_t vb0123c1 = vld1_s8((int8_t*)w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
        const int8x8_t vb4567c1 = vld1_s8((int8_t*)w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));

        const int16x8_t vprod0x0123c1 = vmull_s8(vb0123c1, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0), 1)));
        vacc0x0123 = vpadalq_s16(vacc0x0123, vprod0x0123c1);
        const int16x8_t vprod0x4567c1 = vmull_s8(vb4567c1, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0), 1)));
        vacc0x4567 = vpadalq_s16(vacc0x4567, vprod0x4567c1);
        const int16x8_t vprod1x0123c1 = vmull_s8(vb0123c1, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va1), 1)));
        vacc1x0123 = vpadalq_s16(vacc1x0123, vprod1x0123c1);
        const int16x8_t vprod1x4567c1 = vmull_s8(vb4567c1, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va1), 1)));
        vacc1x4567 = vpadalq_s16(vacc1x4567, vprod1x4567c1);

        if (k > 4 * sizeof(int8_t)) {
          const int8x8_t vb0123c2 = vld1_s8((int8_t*)w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
          const int8x8_t vb4567c2 = vld1_s8((int8_t*)w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));

          const int16x8_t vprod0x0123c2 = vmull_s8(vb0123c2, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0), 2)));
          vacc0x0123 = vpadalq_s16(vacc0x0123, vprod0x0123c2);
          const int16x8_t vprod0x4567c2 = vmull_s8(vb4567c2, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0), 2)));
          vacc0x4567 = vpadalq_s16(vacc0x4567, vprod0x4567c2);
          const int16x8_t vprod1x0123c2 = vmull_s8(vb0123c2, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va1), 2)));
          vacc1x0123 = vpadalq_s16(vacc1x0123, vprod1x0123c2);
          const int16x8_t vprod1x4567c2 = vmull_s8(vb4567c2, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va1), 2)));
          vacc1x4567 = vpadalq_s16(vacc1x4567, vprod1x4567c2);
        }
      }
    }
    const int32x4_t vmultiplier = {0,};//vld1q_dup_s32(&params->neon.multiplier);
    vacc0x0123 = vqrdmulhq_s32(vacc0x0123, vmultiplier);
    vacc0x4567 = vqrdmulhq_s32(vacc0x4567, vmultiplier);
    vacc1x0123 = vqrdmulhq_s32(vacc1x0123, vmultiplier);
    vacc1x4567 = vqrdmulhq_s32(vacc1x4567, vmultiplier);

    const int32x4_t vright_shift = {0,};//vld1q_dup_s32(&params->neon.right_shift);
    const int32x4_t vzero_shift_mask = vreinterpretq_s32_u32(vceqq_s32(vright_shift, vmovq_n_s32(0)));
    vacc0x0123 = vsraq_n_s32(vacc0x0123, vbicq_s32(vacc0x0123, vzero_shift_mask), 31);
    vacc0x4567 = vsraq_n_s32(vacc0x4567, vbicq_s32(vacc0x4567, vzero_shift_mask), 31);
    vacc1x0123 = vsraq_n_s32(vacc1x0123, vbicq_s32(vacc1x0123, vzero_shift_mask), 31);
    vacc1x4567 = vsraq_n_s32(vacc1x4567, vbicq_s32(vacc1x4567, vzero_shift_mask), 31);

    vacc0x0123 = vrshlq_s32(vacc0x0123, vright_shift);
    vacc0x4567 = vrshlq_s32(vacc0x4567, vright_shift);
    vacc1x0123 = vrshlq_s32(vacc1x0123, vright_shift);
    vacc1x4567 = vrshlq_s32(vacc1x4567, vright_shift);

    const int16x8_t voutput_zero_point = {0,};//vld1q_dup_s16(&params->neon.output_zero_point);
#if XNN_ARCH_ARM64
    const int16x8_t vacc0x01234567 = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc0x0123), vacc0x4567), voutput_zero_point);
    const int16x8_t vacc1x01234567 = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc1x0123), vacc1x4567), voutput_zero_point);

    int8x16_t vout0x01234567_1x01234567 = vqmovn_high_s16(vqmovn_s16(vacc0x01234567), vacc1x01234567);
#else
    const int16x8_t vacc0x01234567 = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc0x0123), vqmovn_s32(vacc0x4567)), voutput_zero_point);
    const int16x8_t vacc1x01234567 = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc1x0123), vqmovn_s32(vacc1x4567)), voutput_zero_point);

    int8x16_t vout0x01234567_1x01234567 = vcombine_s8(vqmovn_s16(vacc0x01234567), vqmovn_s16(vacc1x01234567));
#endif
    const int8x16_t voutput_min = {0,};//vld1q_dup_s8(&params->neon.output_min);
    const int8x16_t voutput_max = {0,};//vld1q_dup_s8(&params->neon.output_max);

    vout0x01234567_1x01234567 = vmaxq_s8(vout0x01234567_1x01234567, voutput_min);

    vout0x01234567_1x01234567 = vminq_s8(vout0x01234567_1x01234567, voutput_max);

    if (nc >= 8) {
      vst1_s8(c0 + 0, vget_low_s8(vout0x01234567_1x01234567));
      vst1_s8(c1 + 0, vget_high_s8(vout0x01234567_1x01234567));

      c0 = (int8_t*) ((uintptr_t) c0 + cn_stride);
      c1 = (int8_t*) ((uintptr_t) c1 + cn_stride);

      a0 = (const int8_t*) ((uintptr_t) a0 - kc);
      a1 = (const int8_t*) ((uintptr_t) a1 - kc);

      nc -= 8;
    } else {
      if (nc & 4) {
        vst1q_lane_u32((uint32_t*)__builtin_assume_aligned(c0, 1), vreinterpretq_u32_s8(vout0x01234567_1x01234567), 0); c0 += 4;
        vst1q_lane_u32((uint32_t*)__builtin_assume_aligned(c1, 1), vreinterpretq_u32_s8(vout0x01234567_1x01234567), 2); c1 += 4;
        vout0x01234567_1x01234567 = vextq_s8(vout0x01234567_1x01234567, vout0x01234567_1x01234567, 4);
      }
      if (nc & 2) {
        vst1q_lane_u16((uint16_t*)__builtin_assume_aligned(c0, 1), vreinterpretq_u16_s8(vout0x01234567_1x01234567), 0); c0 += 2;
        vst1q_lane_u16((uint16_t*)__builtin_assume_aligned(c1, 1), vreinterpretq_u16_s8(vout0x01234567_1x01234567), 4); c1 += 2;
        vout0x01234567_1x01234567 = vextq_s8(vout0x01234567_1x01234567, vout0x01234567_1x01234567, 2);
      }
      if (nc & 1) {
        vst1q_lane_s8(c0, vout0x01234567_1x01234567, 0);
        vst1q_lane_s8(c1, vout0x01234567_1x01234567, 8);
      }

      nc = 0;
    }
  } while (nc != 0);
}

inline void xnn_qs8_gemm_minmax_ukernel_4x8c2__neon_mlal_padal_dup(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* __restrict__ a,
    size_t a_stride,
    const void* __restrict__ w,
    int8_t* __restrict__ c,
    size_t cm_stride,
    size_t cn_stride)
{
  assert(mr != 0);
  assert(mr <= 4);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 2);
  const int8_t* a0 = a;
  int8_t* c0 = c;
  const int8_t* a1 = (const int8_t*) ((uintptr_t) a0 + a_stride);
  int8_t* c1 = (int8_t*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    a1 = a0;
    c1 = c0;
  }
  const int8_t* a2 = (const int8_t*) ((uintptr_t) a1 + a_stride);
  int8_t* c2 = (int8_t*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    a2 = a1;
    c2 = c1;
  }
  const int8_t* a3 = (const int8_t*) ((uintptr_t) a2 + a_stride);
  int8_t* c3 = (int8_t*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 4) {
    a3 = a2;
    c3 = c2;
  }

  do {
    int32x4_t vacc0x0123 = vld1q_s32((int32_t*)w);// w = (const void*) ((uintptr_t) w + 4 * sizeof(int32_t));
    int32x4_t vacc0x4567 = vld1q_s32((int32_t*)w);// w = (const void*) ((uintptr_t) w + 4 * sizeof(int32_t));
    int32x4_t vacc1x0123 = vacc0x0123;
    int32x4_t vacc1x4567 = vacc0x4567;
    int32x4_t vacc2x0123 = vacc0x0123;
    int32x4_t vacc2x4567 = vacc0x4567;
    int32x4_t vacc3x0123 = vacc0x0123;
    int32x4_t vacc3x4567 = vacc0x4567;

    size_t k = kc;

    while (k >= 16 * sizeof(int8_t)) {
      const int8x8_t va0x0 = vld1_s8(a0); a0 += 8;
      const int8x8_t va0x1 = vld1_s8(a0); a0 += 8;
      const int8x8_t va1x0 = vld1_s8(a1); a1 += 8;
      const int8x8_t va1x1 = vld1_s8(a1); a1 += 8;
      const int8x8_t va2x0 = vld1_s8(a2); a2 += 8;
      const int8x8_t va2x1 = vld1_s8(a2); a2 += 8;
      const int8x8_t va3x0 = vld1_s8(a3); a3 += 8;
      const int8x8_t va3x1 = vld1_s8(a3); a3 += 8;

      const int8x8_t vb0123c0x0 = vld1_s8((int8_t*)w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
      const int8x8_t vb4567c0x0 = vld1_s8((int8_t*)w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
      const int8x8_t vb0123c1x0 = vld1_s8((int8_t*)w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
      const int8x8_t vb4567c1x0 = vld1_s8((int8_t*)w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
      const int8x8_t vb0123c2x0 = vld1_s8((int8_t*)w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
      const int8x8_t vb4567c2x0 = vld1_s8((int8_t*)w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
      const int8x8_t vb0123c3x0 = vld1_s8((int8_t*)w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
      const int8x8_t vb4567c3x0 = vld1_s8((int8_t*)w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));

      int16x8_t vprod0x0123c0 = vmull_s8(vb0123c0x0, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0x0), 0)));
      int16x8_t vprod1x0123c0 = vmull_s8(vb0123c0x0, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va1x0), 0)));
      int16x8_t vprod2x0123c0 = vmull_s8(vb0123c0x0, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va2x0), 0)));
      int16x8_t vprod3x0123c0 = vmull_s8(vb0123c0x0, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va3x0), 0)));
      const int8x8_t vb0123c0x1 = vld1_s8((int8_t*)w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
      vprod0x0123c0 = vmlal_s8(vprod0x0123c0, vb0123c0x1, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0x1), 0)));
      vprod1x0123c0 = vmlal_s8(vprod1x0123c0, vb0123c0x1, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va1x1), 0)));
      vprod2x0123c0 = vmlal_s8(vprod2x0123c0, vb0123c0x1, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va2x1), 0)));
      vprod3x0123c0 = vmlal_s8(vprod3x0123c0, vb0123c0x1, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va3x1), 0)));
      vacc0x0123 = vpadalq_s16(vacc0x0123, vprod0x0123c0);
      vacc1x0123 = vpadalq_s16(vacc1x0123, vprod1x0123c0);
      vacc2x0123 = vpadalq_s16(vacc2x0123, vprod2x0123c0);
      vacc3x0123 = vpadalq_s16(vacc3x0123, vprod3x0123c0);
      int16x8_t vprod0x4567c0 = vmull_s8(vb4567c0x0, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0x0), 0)));
      int16x8_t vprod1x4567c0 = vmull_s8(vb4567c0x0, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va1x0), 0)));
      int16x8_t vprod2x4567c0 = vmull_s8(vb4567c0x0, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va2x0), 0)));
      int16x8_t vprod3x4567c0 = vmull_s8(vb4567c0x0, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va3x0), 0)));
      const int8x8_t vb4567c0x1 = vld1_s8((int8_t*)w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
      vprod0x4567c0 = vmlal_s8(vprod0x4567c0, vb4567c0x1, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0x1), 0)));
      vprod1x4567c0 = vmlal_s8(vprod1x4567c0, vb4567c0x1, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va1x1), 0)));
      vprod2x4567c0 = vmlal_s8(vprod2x4567c0, vb4567c0x1, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va2x1), 0)));
      vprod3x4567c0 = vmlal_s8(vprod3x4567c0, vb4567c0x1, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va3x1), 0)));
      vacc0x4567 = vpadalq_s16(vacc0x4567, vprod0x4567c0);
      vacc1x4567 = vpadalq_s16(vacc1x4567, vprod1x4567c0);
      vacc2x4567 = vpadalq_s16(vacc2x4567, vprod2x4567c0);
      vacc3x4567 = vpadalq_s16(vacc3x4567, vprod3x4567c0);
      int16x8_t vprod0x0123c1 = vmull_s8(vb0123c1x0, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0x0), 1)));
      int16x8_t vprod1x0123c1 = vmull_s8(vb0123c1x0, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va1x0), 1)));
      int16x8_t vprod2x0123c1 = vmull_s8(vb0123c1x0, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va2x0), 1)));
      int16x8_t vprod3x0123c1 = vmull_s8(vb0123c1x0, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va3x0), 1)));
      const int8x8_t vb0123c1x1 = vld1_s8((int8_t*)w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
      vprod0x0123c1 = vmlal_s8(vprod0x0123c1, vb0123c1x1, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0x1), 1)));
      vprod1x0123c1 = vmlal_s8(vprod1x0123c1, vb0123c1x1, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va1x1), 1)));
      vprod2x0123c1 = vmlal_s8(vprod2x0123c1, vb0123c1x1, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va2x1), 1)));
      vprod3x0123c1 = vmlal_s8(vprod3x0123c1, vb0123c1x1, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va3x1), 1)));
      vacc0x0123 = vpadalq_s16(vacc0x0123, vprod0x0123c1);
      vacc1x0123 = vpadalq_s16(vacc1x0123, vprod1x0123c1);
      vacc2x0123 = vpadalq_s16(vacc2x0123, vprod2x0123c1);
      vacc3x0123 = vpadalq_s16(vacc3x0123, vprod3x0123c1);
      int16x8_t vprod0x4567c1 = vmull_s8(vb4567c1x0, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0x0), 1)));
      int16x8_t vprod1x4567c1 = vmull_s8(vb4567c1x0, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va1x0), 1)));
      int16x8_t vprod2x4567c1 = vmull_s8(vb4567c1x0, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va2x0), 1)));
      int16x8_t vprod3x4567c1 = vmull_s8(vb4567c1x0, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va3x0), 1)));
      const int8x8_t vb4567c1x1 = vld1_s8((int8_t*)w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
      vprod0x4567c1 = vmlal_s8(vprod0x4567c1, vb4567c1x1, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0x1), 1)));
      vprod1x4567c1 = vmlal_s8(vprod1x4567c1, vb4567c1x1, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va1x1), 1)));
      vprod2x4567c1 = vmlal_s8(vprod2x4567c1, vb4567c1x1, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va2x1), 1)));
      vprod3x4567c1 = vmlal_s8(vprod3x4567c1, vb4567c1x1, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va3x1), 1)));
      vacc0x4567 = vpadalq_s16(vacc0x4567, vprod0x4567c1);
      vacc1x4567 = vpadalq_s16(vacc1x4567, vprod1x4567c1);
      vacc2x4567 = vpadalq_s16(vacc2x4567, vprod2x4567c1);
      vacc3x4567 = vpadalq_s16(vacc3x4567, vprod3x4567c1);
      int16x8_t vprod0x0123c2 = vmull_s8(vb0123c2x0, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0x0), 2)));
      int16x8_t vprod1x0123c2 = vmull_s8(vb0123c2x0, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va1x0), 2)));
      int16x8_t vprod2x0123c2 = vmull_s8(vb0123c2x0, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va2x0), 2)));
      int16x8_t vprod3x0123c2 = vmull_s8(vb0123c2x0, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va3x0), 2)));
      const int8x8_t vb0123c2x1 = vld1_s8((int8_t*)w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
      vprod0x0123c2 = vmlal_s8(vprod0x0123c2, vb0123c2x1, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0x1), 2)));
      vprod1x0123c2 = vmlal_s8(vprod1x0123c2, vb0123c2x1, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va1x1), 2)));
      vprod2x0123c2 = vmlal_s8(vprod2x0123c2, vb0123c2x1, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va2x1), 2)));
      vprod3x0123c2 = vmlal_s8(vprod3x0123c2, vb0123c2x1, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va3x1), 2)));
      vacc0x0123 = vpadalq_s16(vacc0x0123, vprod0x0123c2);
      vacc1x0123 = vpadalq_s16(vacc1x0123, vprod1x0123c2);
      vacc2x0123 = vpadalq_s16(vacc2x0123, vprod2x0123c2);
      vacc3x0123 = vpadalq_s16(vacc3x0123, vprod3x0123c2);
      int16x8_t vprod0x4567c2 = vmull_s8(vb4567c2x0, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0x0), 2)));
      int16x8_t vprod1x4567c2 = vmull_s8(vb4567c2x0, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va1x0), 2)));
      int16x8_t vprod2x4567c2 = vmull_s8(vb4567c2x0, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va2x0), 2)));
      int16x8_t vprod3x4567c2 = vmull_s8(vb4567c2x0, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va3x0), 2)));
      const int8x8_t vb4567c2x1 = vld1_s8((int8_t*)w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
      vprod0x4567c2 = vmlal_s8(vprod0x4567c2, vb4567c2x1, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0x1), 2)));
      vprod1x4567c2 = vmlal_s8(vprod1x4567c2, vb4567c2x1, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va1x1), 2)));
      vprod2x4567c2 = vmlal_s8(vprod2x4567c2, vb4567c2x1, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va2x1), 2)));
      vprod3x4567c2 = vmlal_s8(vprod3x4567c2, vb4567c2x1, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va3x1), 2)));
      vacc0x4567 = vpadalq_s16(vacc0x4567, vprod0x4567c2);
      vacc1x4567 = vpadalq_s16(vacc1x4567, vprod1x4567c2);
      vacc2x4567 = vpadalq_s16(vacc2x4567, vprod2x4567c2);
      vacc3x4567 = vpadalq_s16(vacc3x4567, vprod3x4567c2);
      int16x8_t vprod0x0123c3 = vmull_s8(vb0123c3x0, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0x0), 3)));
      int16x8_t vprod1x0123c3 = vmull_s8(vb0123c3x0, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va1x0), 3)));
      int16x8_t vprod2x0123c3 = vmull_s8(vb0123c3x0, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va2x0), 3)));
      int16x8_t vprod3x0123c3 = vmull_s8(vb0123c3x0, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va3x0), 3)));
      const int8x8_t vb0123c3x1 = vld1_s8((int8_t*)w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
      vprod0x0123c3 = vmlal_s8(vprod0x0123c3, vb0123c3x1, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0x1), 3)));
      vprod1x0123c3 = vmlal_s8(vprod1x0123c3, vb0123c3x1, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va1x1), 3)));
      vprod2x0123c3 = vmlal_s8(vprod2x0123c3, vb0123c3x1, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va2x1), 3)));
      vprod3x0123c3 = vmlal_s8(vprod3x0123c3, vb0123c3x1, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va3x1), 3)));
      vacc0x0123 = vpadalq_s16(vacc0x0123, vprod0x0123c3);
      vacc1x0123 = vpadalq_s16(vacc1x0123, vprod1x0123c3);
      vacc2x0123 = vpadalq_s16(vacc2x0123, vprod2x0123c3);
      vacc3x0123 = vpadalq_s16(vacc3x0123, vprod3x0123c3);
      int16x8_t vprod0x4567c3 = vmull_s8(vb4567c3x0, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0x0), 3)));
      int16x8_t vprod1x4567c3 = vmull_s8(vb4567c3x0, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va1x0), 3)));
      int16x8_t vprod2x4567c3 = vmull_s8(vb4567c3x0, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va2x0), 3)));
      int16x8_t vprod3x4567c3 = vmull_s8(vb4567c3x0, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va3x0), 3)));
      const int8x8_t vb4567c3x1 = vld1_s8((int8_t*)w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
      vprod0x4567c3 = vmlal_s8(vprod0x4567c3, vb4567c3x1, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0x1), 3)));
      vprod1x4567c3 = vmlal_s8(vprod1x4567c3, vb4567c3x1, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va1x1), 3)));
      vprod2x4567c3 = vmlal_s8(vprod2x4567c3, vb4567c3x1, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va2x1), 3)));
      vprod3x4567c3 = vmlal_s8(vprod3x4567c3, vb4567c3x1, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va3x1), 3)));
      vacc0x4567 = vpadalq_s16(vacc0x4567, vprod0x4567c3);
      vacc1x4567 = vpadalq_s16(vacc1x4567, vprod1x4567c3);
      vacc2x4567 = vpadalq_s16(vacc2x4567, vprod2x4567c3);
      vacc3x4567 = vpadalq_s16(vacc3x4567, vprod3x4567c3);

      k -= 16 * sizeof(int8_t);
    }

    if (k >= 8 * sizeof(int8_t)) {
      const int8x8_t va0 = vld1_s8(a0); a0 += 8;
      const int8x8_t va1 = vld1_s8(a1); a1 += 8;
      const int8x8_t va2 = vld1_s8(a2); a2 += 8;
      const int8x8_t va3 = vld1_s8(a3); a3 += 8;

      const int8x8_t vb0123c0 = vld1_s8((int8_t*)w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
      const int8x8_t vb4567c0 = vld1_s8((int8_t*)w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
      const int8x8_t vb0123c1 = vld1_s8((int8_t*)w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
      const int8x8_t vb4567c1 = vld1_s8((int8_t*)w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
      const int8x8_t vb0123c2 = vld1_s8((int8_t*)w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
      const int8x8_t vb4567c2 = vld1_s8((int8_t*)w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
      const int8x8_t vb0123c3 = vld1_s8((int8_t*)w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
      const int8x8_t vb4567c3 = vld1_s8((int8_t*)w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));

      const int16x8_t vprod0x0123c0 = vmull_s8(vb0123c0, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0), 0)));
      const int16x8_t vprod0x0123c1 = vmull_s8(vb0123c1, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0), 1)));
      const int16x8_t vprod0x0123c2 = vmull_s8(vb0123c2, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0), 2)));
      const int16x8_t vprod0x0123c3 = vmull_s8(vb0123c3, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0), 3)));
      vacc0x0123 = vpadalq_s16(vacc0x0123, vprod0x0123c0);
      vacc0x0123 = vpadalq_s16(vacc0x0123, vprod0x0123c1);
      vacc0x0123 = vpadalq_s16(vacc0x0123, vprod0x0123c2);
      vacc0x0123 = vpadalq_s16(vacc0x0123, vprod0x0123c3);
      const int16x8_t vprod0x4567c0 = vmull_s8(vb4567c0, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0), 0)));
      const int16x8_t vprod0x4567c1 = vmull_s8(vb4567c1, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0), 1)));
      const int16x8_t vprod0x4567c2 = vmull_s8(vb4567c2, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0), 2)));
      const int16x8_t vprod0x4567c3 = vmull_s8(vb4567c3, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0), 3)));
      vacc0x4567 = vpadalq_s16(vacc0x4567, vprod0x4567c0);
      vacc0x4567 = vpadalq_s16(vacc0x4567, vprod0x4567c1);
      vacc0x4567 = vpadalq_s16(vacc0x4567, vprod0x4567c2);
      vacc0x4567 = vpadalq_s16(vacc0x4567, vprod0x4567c3);
      const int16x8_t vprod1x0123c0 = vmull_s8(vb0123c0, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va1), 0)));
      const int16x8_t vprod1x0123c1 = vmull_s8(vb0123c1, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va1), 1)));
      const int16x8_t vprod1x0123c2 = vmull_s8(vb0123c2, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va1), 2)));
      const int16x8_t vprod1x0123c3 = vmull_s8(vb0123c3, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va1), 3)));
      vacc1x0123 = vpadalq_s16(vacc1x0123, vprod1x0123c0);
      vacc1x0123 = vpadalq_s16(vacc1x0123, vprod1x0123c1);
      vacc1x0123 = vpadalq_s16(vacc1x0123, vprod1x0123c2);
      vacc1x0123 = vpadalq_s16(vacc1x0123, vprod1x0123c3);
      const int16x8_t vprod1x4567c0 = vmull_s8(vb4567c0, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va1), 0)));
      const int16x8_t vprod1x4567c1 = vmull_s8(vb4567c1, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va1), 1)));
      const int16x8_t vprod1x4567c2 = vmull_s8(vb4567c2, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va1), 2)));
      const int16x8_t vprod1x4567c3 = vmull_s8(vb4567c3, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va1), 3)));
      vacc1x4567 = vpadalq_s16(vacc1x4567, vprod1x4567c0);
      vacc1x4567 = vpadalq_s16(vacc1x4567, vprod1x4567c1);
      vacc1x4567 = vpadalq_s16(vacc1x4567, vprod1x4567c2);
      vacc1x4567 = vpadalq_s16(vacc1x4567, vprod1x4567c3);
      const int16x8_t vprod2x0123c0 = vmull_s8(vb0123c0, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va2), 0)));
      const int16x8_t vprod2x0123c1 = vmull_s8(vb0123c1, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va2), 1)));
      const int16x8_t vprod2x0123c2 = vmull_s8(vb0123c2, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va2), 2)));
      const int16x8_t vprod2x0123c3 = vmull_s8(vb0123c3, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va2), 3)));
      vacc2x0123 = vpadalq_s16(vacc2x0123, vprod2x0123c0);
      vacc2x0123 = vpadalq_s16(vacc2x0123, vprod2x0123c1);
      vacc2x0123 = vpadalq_s16(vacc2x0123, vprod2x0123c2);
      vacc2x0123 = vpadalq_s16(vacc2x0123, vprod2x0123c3);
      const int16x8_t vprod2x4567c0 = vmull_s8(vb4567c0, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va2), 0)));
      const int16x8_t vprod2x4567c1 = vmull_s8(vb4567c1, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va2), 1)));
      const int16x8_t vprod2x4567c2 = vmull_s8(vb4567c2, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va2), 2)));
      const int16x8_t vprod2x4567c3 = vmull_s8(vb4567c3, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va2), 3)));
      vacc2x4567 = vpadalq_s16(vacc2x4567, vprod2x4567c0);
      vacc2x4567 = vpadalq_s16(vacc2x4567, vprod2x4567c1);
      vacc2x4567 = vpadalq_s16(vacc2x4567, vprod2x4567c2);
      vacc2x4567 = vpadalq_s16(vacc2x4567, vprod2x4567c3);
      const int16x8_t vprod3x0123c0 = vmull_s8(vb0123c0, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va3), 0)));
      const int16x8_t vprod3x0123c1 = vmull_s8(vb0123c1, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va3), 1)));
      const int16x8_t vprod3x0123c2 = vmull_s8(vb0123c2, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va3), 2)));
      const int16x8_t vprod3x0123c3 = vmull_s8(vb0123c3, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va3), 3)));
      vacc3x0123 = vpadalq_s16(vacc3x0123, vprod3x0123c0);
      vacc3x0123 = vpadalq_s16(vacc3x0123, vprod3x0123c1);
      vacc3x0123 = vpadalq_s16(vacc3x0123, vprod3x0123c2);
      vacc3x0123 = vpadalq_s16(vacc3x0123, vprod3x0123c3);
      const int16x8_t vprod3x4567c0 = vmull_s8(vb4567c0, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va3), 0)));
      const int16x8_t vprod3x4567c1 = vmull_s8(vb4567c1, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va3), 1)));
      const int16x8_t vprod3x4567c2 = vmull_s8(vb4567c2, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va3), 2)));
      const int16x8_t vprod3x4567c3 = vmull_s8(vb4567c3, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va3), 3)));
      vacc3x4567 = vpadalq_s16(vacc3x4567, vprod3x4567c0);
      vacc3x4567 = vpadalq_s16(vacc3x4567, vprod3x4567c1);
      vacc3x4567 = vpadalq_s16(vacc3x4567, vprod3x4567c2);
      vacc3x4567 = vpadalq_s16(vacc3x4567, vprod3x4567c3);

      k -= 8 * sizeof(int8_t);
    }

    if XNN_UNLIKELY(k != 0) {
      const int8x8_t va0 = vld1_s8(a0); a0 = (const int8_t*) ((uintptr_t) a0 + k);
      const int8x8_t va1 = vld1_s8(a1); a1 = (const int8_t*) ((uintptr_t) a1 + k);
      const int8x8_t va2 = vld1_s8(a2); a2 = (const int8_t*) ((uintptr_t) a2 + k);
      const int8x8_t va3 = vld1_s8(a3); a3 = (const int8_t*) ((uintptr_t) a3 + k);

      const int8x8_t vb0123c0 = vld1_s8((int8_t*)w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
      const int8x8_t vb4567c0 = vld1_s8((int8_t*)w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));

      const int16x8_t vprod0x0123c0 = vmull_s8(vb0123c0, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0), 0)));
      vacc0x0123 = vpadalq_s16(vacc0x0123, vprod0x0123c0);
      const int16x8_t vprod0x4567c0 = vmull_s8(vb4567c0, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0), 0)));
      vacc0x4567 = vpadalq_s16(vacc0x4567, vprod0x4567c0);
      const int16x8_t vprod1x0123c0 = vmull_s8(vb0123c0, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va1), 0)));
      vacc1x0123 = vpadalq_s16(vacc1x0123, vprod1x0123c0);
      const int16x8_t vprod1x4567c0 = vmull_s8(vb4567c0, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va1), 0)));
      vacc1x4567 = vpadalq_s16(vacc1x4567, vprod1x4567c0);
      const int16x8_t vprod2x0123c0 = vmull_s8(vb0123c0, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va2), 0)));
      vacc2x0123 = vpadalq_s16(vacc2x0123, vprod2x0123c0);
      const int16x8_t vprod2x4567c0 = vmull_s8(vb4567c0, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va2), 0)));
      vacc2x4567 = vpadalq_s16(vacc2x4567, vprod2x4567c0);
      const int16x8_t vprod3x0123c0 = vmull_s8(vb0123c0, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va3), 0)));
      vacc3x0123 = vpadalq_s16(vacc3x0123, vprod3x0123c0);
      const int16x8_t vprod3x4567c0 = vmull_s8(vb4567c0, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va3), 0)));
      vacc3x4567 = vpadalq_s16(vacc3x4567, vprod3x4567c0);

      if (k > 2 * sizeof(int8_t)) {
        const int8x8_t vb0123c1 = vld1_s8((int8_t*)w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
        const int8x8_t vb4567c1 = vld1_s8((int8_t*)w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));

        const int16x8_t vprod0x0123c1 = vmull_s8(vb0123c1, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0), 1)));
        vacc0x0123 = vpadalq_s16(vacc0x0123, vprod0x0123c1);
        const int16x8_t vprod0x4567c1 = vmull_s8(vb4567c1, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0), 1)));
        vacc0x4567 = vpadalq_s16(vacc0x4567, vprod0x4567c1);
        const int16x8_t vprod1x0123c1 = vmull_s8(vb0123c1, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va1), 1)));
        vacc1x0123 = vpadalq_s16(vacc1x0123, vprod1x0123c1);
        const int16x8_t vprod1x4567c1 = vmull_s8(vb4567c1, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va1), 1)));
        vacc1x4567 = vpadalq_s16(vacc1x4567, vprod1x4567c1);
        const int16x8_t vprod2x0123c1 = vmull_s8(vb0123c1, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va2), 1)));
        vacc2x0123 = vpadalq_s16(vacc2x0123, vprod2x0123c1);
        const int16x8_t vprod2x4567c1 = vmull_s8(vb4567c1, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va2), 1)));
        vacc2x4567 = vpadalq_s16(vacc2x4567, vprod2x4567c1);
        const int16x8_t vprod3x0123c1 = vmull_s8(vb0123c1, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va3), 1)));
        vacc3x0123 = vpadalq_s16(vacc3x0123, vprod3x0123c1);
        const int16x8_t vprod3x4567c1 = vmull_s8(vb4567c1, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va3), 1)));
        vacc3x4567 = vpadalq_s16(vacc3x4567, vprod3x4567c1);

        if (k > 4 * sizeof(int8_t)) {
          const int8x8_t vb0123c2 = vld1_s8((int8_t*)w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
          const int8x8_t vb4567c2 = vld1_s8((int8_t*)w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));

          const int16x8_t vprod0x0123c2 = vmull_s8(vb0123c2, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0), 2)));
          vacc0x0123 = vpadalq_s16(vacc0x0123, vprod0x0123c2);
          const int16x8_t vprod0x4567c2 = vmull_s8(vb4567c2, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0), 2)));
          vacc0x4567 = vpadalq_s16(vacc0x4567, vprod0x4567c2);
          const int16x8_t vprod1x0123c2 = vmull_s8(vb0123c2, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va1), 2)));
          vacc1x0123 = vpadalq_s16(vacc1x0123, vprod1x0123c2);
          const int16x8_t vprod1x4567c2 = vmull_s8(vb4567c2, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va1), 2)));
          vacc1x4567 = vpadalq_s16(vacc1x4567, vprod1x4567c2);
          const int16x8_t vprod2x0123c2 = vmull_s8(vb0123c2, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va2), 2)));
          vacc2x0123 = vpadalq_s16(vacc2x0123, vprod2x0123c2);
          const int16x8_t vprod2x4567c2 = vmull_s8(vb4567c2, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va2), 2)));
          vacc2x4567 = vpadalq_s16(vacc2x4567, vprod2x4567c2);
          const int16x8_t vprod3x0123c2 = vmull_s8(vb0123c2, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va3), 2)));
          vacc3x0123 = vpadalq_s16(vacc3x0123, vprod3x0123c2);
          const int16x8_t vprod3x4567c2 = vmull_s8(vb4567c2, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va3), 2)));
          vacc3x4567 = vpadalq_s16(vacc3x4567, vprod3x4567c2);
        }
      }
    }
    const int32x4_t vmultiplier = {0,}; //vld1q_dup_s32(&params->neon.multiplier);
    vacc0x0123 = vqrdmulhq_s32(vacc0x0123, vmultiplier);
    vacc0x4567 = vqrdmulhq_s32(vacc0x4567, vmultiplier);
    vacc1x0123 = vqrdmulhq_s32(vacc1x0123, vmultiplier);
    vacc1x4567 = vqrdmulhq_s32(vacc1x4567, vmultiplier);
    vacc2x0123 = vqrdmulhq_s32(vacc2x0123, vmultiplier);
    vacc2x4567 = vqrdmulhq_s32(vacc2x4567, vmultiplier);
    vacc3x0123 = vqrdmulhq_s32(vacc3x0123, vmultiplier);
    vacc3x4567 = vqrdmulhq_s32(vacc3x4567, vmultiplier);

    const int32x4_t vright_shift = {0,}; //vld1q_dup_s32(&params->neon.right_shift);
    const int32x4_t vzero_shift_mask = vreinterpretq_s32_u32(vceqq_s32(vright_shift, vmovq_n_s32(0)));
    vacc0x0123 = vsraq_n_s32(vacc0x0123, vbicq_s32(vacc0x0123, vzero_shift_mask), 31);
    vacc0x4567 = vsraq_n_s32(vacc0x4567, vbicq_s32(vacc0x4567, vzero_shift_mask), 31);
    vacc1x0123 = vsraq_n_s32(vacc1x0123, vbicq_s32(vacc1x0123, vzero_shift_mask), 31);
    vacc1x4567 = vsraq_n_s32(vacc1x4567, vbicq_s32(vacc1x4567, vzero_shift_mask), 31);
    vacc2x0123 = vsraq_n_s32(vacc2x0123, vbicq_s32(vacc2x0123, vzero_shift_mask), 31);
    vacc2x4567 = vsraq_n_s32(vacc2x4567, vbicq_s32(vacc2x4567, vzero_shift_mask), 31);
    vacc3x0123 = vsraq_n_s32(vacc3x0123, vbicq_s32(vacc3x0123, vzero_shift_mask), 31);
    vacc3x4567 = vsraq_n_s32(vacc3x4567, vbicq_s32(vacc3x4567, vzero_shift_mask), 31);

    vacc0x0123 = vrshlq_s32(vacc0x0123, vright_shift);
    vacc0x4567 = vrshlq_s32(vacc0x4567, vright_shift);
    vacc1x0123 = vrshlq_s32(vacc1x0123, vright_shift);
    vacc1x4567 = vrshlq_s32(vacc1x4567, vright_shift);
    vacc2x0123 = vrshlq_s32(vacc2x0123, vright_shift);
    vacc2x4567 = vrshlq_s32(vacc2x4567, vright_shift);
    vacc3x0123 = vrshlq_s32(vacc3x0123, vright_shift);
    vacc3x4567 = vrshlq_s32(vacc3x4567, vright_shift);

    const int16x8_t voutput_zero_point = {0,}; //vld1q_dup_s16(&params->neon.output_zero_point);
#if XNN_ARCH_ARM64
    const int16x8_t vacc0x01234567 = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc0x0123), vacc0x4567), voutput_zero_point);
    const int16x8_t vacc1x01234567 = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc1x0123), vacc1x4567), voutput_zero_point);
    const int16x8_t vacc2x01234567 = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc2x0123), vacc2x4567), voutput_zero_point);
    const int16x8_t vacc3x01234567 = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc3x0123), vacc3x4567), voutput_zero_point);

    int8x16_t vout0x01234567_1x01234567 = vqmovn_high_s16(vqmovn_s16(vacc0x01234567), vacc1x01234567);
    int8x16_t vout2x01234567_3x01234567 = vqmovn_high_s16(vqmovn_s16(vacc2x01234567), vacc3x01234567);
#else
    const int16x8_t vacc0x01234567 = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc0x0123), vqmovn_s32(vacc0x4567)), voutput_zero_point);
    const int16x8_t vacc1x01234567 = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc1x0123), vqmovn_s32(vacc1x4567)), voutput_zero_point);
    const int16x8_t vacc2x01234567 = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc2x0123), vqmovn_s32(vacc2x4567)), voutput_zero_point);
    const int16x8_t vacc3x01234567 = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc3x0123), vqmovn_s32(vacc3x4567)), voutput_zero_point);

    int8x16_t vout0x01234567_1x01234567 = vcombine_s8(vqmovn_s16(vacc0x01234567), vqmovn_s16(vacc1x01234567));
    int8x16_t vout2x01234567_3x01234567 = vcombine_s8(vqmovn_s16(vacc2x01234567), vqmovn_s16(vacc3x01234567));
#endif
    const int8x16_t voutput_min = {0,}; //vld1q_dup_s8(&params->neon.output_min);
    const int8x16_t voutput_max = {1,}; //vld1q_dup_s8(&params->neon.output_max);

    vout0x01234567_1x01234567 = vmaxq_s8(vout0x01234567_1x01234567, voutput_min);
    vout2x01234567_3x01234567 = vmaxq_s8(vout2x01234567_3x01234567, voutput_min);

    vout0x01234567_1x01234567 = vminq_s8(vout0x01234567_1x01234567, voutput_max);
    vout2x01234567_3x01234567 = vminq_s8(vout2x01234567_3x01234567, voutput_max);

    if (nc >= 8) {
      vst1_s8(c0 + 0, vget_low_s8(vout0x01234567_1x01234567));
      vst1_s8(c1 + 0, vget_high_s8(vout0x01234567_1x01234567));
      vst1_s8(c2 + 0, vget_low_s8(vout2x01234567_3x01234567));
      vst1_s8(c3 + 0, vget_high_s8(vout2x01234567_3x01234567));

      c0 = (int8_t*) ((uintptr_t) c0 + cn_stride);
      c1 = (int8_t*) ((uintptr_t) c1 + cn_stride);
      c2 = (int8_t*) ((uintptr_t) c2 + cn_stride);
      c3 = (int8_t*) ((uintptr_t) c3 + cn_stride);

      a0 = (const int8_t*) ((uintptr_t) a0 - kc);
      a1 = (const int8_t*) ((uintptr_t) a1 - kc);
      a2 = (const int8_t*) ((uintptr_t) a2 - kc);
      a3 = (const int8_t*) ((uintptr_t) a3 - kc);

      nc -= 8;
    } else {
      if (nc & 4) {
        vst1q_lane_u32((uint32_t*)__builtin_assume_aligned(c0, 1), vreinterpretq_u32_s8(vout0x01234567_1x01234567), 0); c0 += 4;
        vst1q_lane_u32((uint32_t*)__builtin_assume_aligned(c1, 1), vreinterpretq_u32_s8(vout0x01234567_1x01234567), 2); c1 += 4;
        vst1q_lane_u32((uint32_t*)__builtin_assume_aligned(c2, 1), vreinterpretq_u32_s8(vout2x01234567_3x01234567), 0); c2 += 4;
        vst1q_lane_u32((uint32_t*)__builtin_assume_aligned(c3, 1), vreinterpretq_u32_s8(vout2x01234567_3x01234567), 2); c3 += 4;
        vout0x01234567_1x01234567 = vextq_s8(vout0x01234567_1x01234567, vout0x01234567_1x01234567, 4);
        vout2x01234567_3x01234567 = vextq_s8(vout2x01234567_3x01234567, vout2x01234567_3x01234567, 4);
      }
      if (nc & 2) {
        vst1q_lane_u16((uint16_t*)__builtin_assume_aligned(c0, 1), vreinterpretq_u16_s8(vout0x01234567_1x01234567), 0); c0 += 2;
        vst1q_lane_u16((uint16_t*)__builtin_assume_aligned(c1, 1), vreinterpretq_u16_s8(vout0x01234567_1x01234567), 4); c1 += 2;
        vst1q_lane_u16((uint16_t*)__builtin_assume_aligned(c2, 1), vreinterpretq_u16_s8(vout2x01234567_3x01234567), 0); c2 += 2;
        vst1q_lane_u16((uint16_t*)__builtin_assume_aligned(c3, 1), vreinterpretq_u16_s8(vout2x01234567_3x01234567), 4); c3 += 2;
        vout0x01234567_1x01234567 = vextq_s8(vout0x01234567_1x01234567, vout0x01234567_1x01234567, 2);
        vout2x01234567_3x01234567 = vextq_s8(vout2x01234567_3x01234567, vout2x01234567_3x01234567, 2);
      }
      if (nc & 1) {
        vst1q_lane_s8(c0, vout0x01234567_1x01234567, 0);
        vst1q_lane_s8(c1, vout0x01234567_1x01234567, 8);
        vst1q_lane_s8(c2, vout2x01234567_3x01234567, 0);
        vst1q_lane_s8(c3, vout2x01234567_3x01234567, 8);
      }

      nc = 0;
    }
  } while (nc != 0);
}

inline void xnn_qs8_gemm_minmax_ukernel_4x8c2__neon_mull_padal_dup(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* __restrict__ a,
    size_t a_stride,
    const void* __restrict__ w,
    int8_t* __restrict__ c,
    size_t cm_stride,
    size_t cn_stride)
{
  assert(mr != 0);
  assert(mr <= 4);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 2);
  const int8_t* a0 = a;
  int8_t* c0 = c;
  const int8_t* a1 = (const int8_t*) ((uintptr_t) a0 + a_stride);
  int8_t* c1 = (int8_t*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    a1 = a0;
    c1 = c0;
  }
  const int8_t* a2 = (const int8_t*) ((uintptr_t) a1 + a_stride);
  int8_t* c2 = (int8_t*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    a2 = a1;
    c2 = c1;
  }
  const int8_t* a3 = (const int8_t*) ((uintptr_t) a2 + a_stride);
  int8_t* c3 = (int8_t*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 4) {
    a3 = a2;
    c3 = c2;
  }

  do {
    int32x4_t vacc0x0123 = vld1q_s32((int*)w); //w = (const void*) ((uintptr_t) w + 4 * sizeof(int32_t));
    int32x4_t vacc0x4567 = vld1q_s32((int*)w); //w = (const void*) ((uintptr_t) w + 4 * sizeof(int32_t));
    int32x4_t vacc1x0123 = vacc0x0123;
    int32x4_t vacc1x4567 = vacc0x4567;
    int32x4_t vacc2x0123 = vacc0x0123;
    int32x4_t vacc2x4567 = vacc0x4567;
    int32x4_t vacc3x0123 = vacc0x0123;
    int32x4_t vacc3x4567 = vacc0x4567;

    size_t k = kc;


    while (k >= 8 * sizeof(int8_t)) {
      const int8x8_t va0 = vld1_s8(a0); a0 += 8;
      const int8x8_t va1 = vld1_s8(a1); a1 += 8;
      const int8x8_t va2 = vld1_s8(a2); a2 += 8;
      const int8x8_t va3 = vld1_s8(a3); a3 += 8;

      const int8x8_t vb0123c0 = vld1_s8((int8_t*)w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
      const int8x8_t vb4567c0 = vld1_s8((int8_t*)w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
      const int8x8_t vb0123c1 = vld1_s8((int8_t*)w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
      const int8x8_t vb4567c1 = vld1_s8((int8_t*)w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
      const int8x8_t vb0123c2 = vld1_s8((int8_t*)w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
      const int8x8_t vb4567c2 = vld1_s8((int8_t*)w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
      const int8x8_t vb0123c3 = vld1_s8((int8_t*)w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
      const int8x8_t vb4567c3 = vld1_s8((int8_t*)w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));

      const int16x8_t vprod0x0123c0 = vmull_s8(vb0123c0, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0), 0)));
      const int16x8_t vprod0x0123c1 = vmull_s8(vb0123c1, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0), 1)));
      const int16x8_t vprod0x0123c2 = vmull_s8(vb0123c2, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0), 2)));
      const int16x8_t vprod0x0123c3 = vmull_s8(vb0123c3, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0), 3)));
      vacc0x0123 = vpadalq_s16(vacc0x0123, vprod0x0123c0);
      vacc0x0123 = vpadalq_s16(vacc0x0123, vprod0x0123c1);
      vacc0x0123 = vpadalq_s16(vacc0x0123, vprod0x0123c2);
      vacc0x0123 = vpadalq_s16(vacc0x0123, vprod0x0123c3);
      const int16x8_t vprod0x4567c0 = vmull_s8(vb4567c0, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0), 0)));
      const int16x8_t vprod0x4567c1 = vmull_s8(vb4567c1, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0), 1)));
      const int16x8_t vprod0x4567c2 = vmull_s8(vb4567c2, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0), 2)));
      const int16x8_t vprod0x4567c3 = vmull_s8(vb4567c3, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0), 3)));
      vacc0x4567 = vpadalq_s16(vacc0x4567, vprod0x4567c0);
      vacc0x4567 = vpadalq_s16(vacc0x4567, vprod0x4567c1);
      vacc0x4567 = vpadalq_s16(vacc0x4567, vprod0x4567c2);
      vacc0x4567 = vpadalq_s16(vacc0x4567, vprod0x4567c3);
      const int16x8_t vprod1x0123c0 = vmull_s8(vb0123c0, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va1), 0)));
      const int16x8_t vprod1x0123c1 = vmull_s8(vb0123c1, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va1), 1)));
      const int16x8_t vprod1x0123c2 = vmull_s8(vb0123c2, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va1), 2)));
      const int16x8_t vprod1x0123c3 = vmull_s8(vb0123c3, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va1), 3)));
      vacc1x0123 = vpadalq_s16(vacc1x0123, vprod1x0123c0);
      vacc1x0123 = vpadalq_s16(vacc1x0123, vprod1x0123c1);
      vacc1x0123 = vpadalq_s16(vacc1x0123, vprod1x0123c2);
      vacc1x0123 = vpadalq_s16(vacc1x0123, vprod1x0123c3);
      const int16x8_t vprod1x4567c0 = vmull_s8(vb4567c0, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va1), 0)));
      const int16x8_t vprod1x4567c1 = vmull_s8(vb4567c1, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va1), 1)));
      const int16x8_t vprod1x4567c2 = vmull_s8(vb4567c2, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va1), 2)));
      const int16x8_t vprod1x4567c3 = vmull_s8(vb4567c3, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va1), 3)));
      vacc1x4567 = vpadalq_s16(vacc1x4567, vprod1x4567c0);
      vacc1x4567 = vpadalq_s16(vacc1x4567, vprod1x4567c1);
      vacc1x4567 = vpadalq_s16(vacc1x4567, vprod1x4567c2);
      vacc1x4567 = vpadalq_s16(vacc1x4567, vprod1x4567c3);
      const int16x8_t vprod2x0123c0 = vmull_s8(vb0123c0, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va2), 0)));
      const int16x8_t vprod2x0123c1 = vmull_s8(vb0123c1, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va2), 1)));
      const int16x8_t vprod2x0123c2 = vmull_s8(vb0123c2, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va2), 2)));
      const int16x8_t vprod2x0123c3 = vmull_s8(vb0123c3, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va2), 3)));
      vacc2x0123 = vpadalq_s16(vacc2x0123, vprod2x0123c0);
      vacc2x0123 = vpadalq_s16(vacc2x0123, vprod2x0123c1);
      vacc2x0123 = vpadalq_s16(vacc2x0123, vprod2x0123c2);
      vacc2x0123 = vpadalq_s16(vacc2x0123, vprod2x0123c3);
      const int16x8_t vprod2x4567c0 = vmull_s8(vb4567c0, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va2), 0)));
      const int16x8_t vprod2x4567c1 = vmull_s8(vb4567c1, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va2), 1)));
      const int16x8_t vprod2x4567c2 = vmull_s8(vb4567c2, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va2), 2)));
      const int16x8_t vprod2x4567c3 = vmull_s8(vb4567c3, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va2), 3)));
      vacc2x4567 = vpadalq_s16(vacc2x4567, vprod2x4567c0);
      vacc2x4567 = vpadalq_s16(vacc2x4567, vprod2x4567c1);
      vacc2x4567 = vpadalq_s16(vacc2x4567, vprod2x4567c2);
      vacc2x4567 = vpadalq_s16(vacc2x4567, vprod2x4567c3);
      const int16x8_t vprod3x0123c0 = vmull_s8(vb0123c0, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va3), 0)));
      const int16x8_t vprod3x0123c1 = vmull_s8(vb0123c1, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va3), 1)));
      const int16x8_t vprod3x0123c2 = vmull_s8(vb0123c2, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va3), 2)));
      const int16x8_t vprod3x0123c3 = vmull_s8(vb0123c3, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va3), 3)));
      vacc3x0123 = vpadalq_s16(vacc3x0123, vprod3x0123c0);
      vacc3x0123 = vpadalq_s16(vacc3x0123, vprod3x0123c1);
      vacc3x0123 = vpadalq_s16(vacc3x0123, vprod3x0123c2);
      vacc3x0123 = vpadalq_s16(vacc3x0123, vprod3x0123c3);
      const int16x8_t vprod3x4567c0 = vmull_s8(vb4567c0, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va3), 0)));
      const int16x8_t vprod3x4567c1 = vmull_s8(vb4567c1, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va3), 1)));
      const int16x8_t vprod3x4567c2 = vmull_s8(vb4567c2, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va3), 2)));
      const int16x8_t vprod3x4567c3 = vmull_s8(vb4567c3, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va3), 3)));
      vacc3x4567 = vpadalq_s16(vacc3x4567, vprod3x4567c0);
      vacc3x4567 = vpadalq_s16(vacc3x4567, vprod3x4567c1);
      vacc3x4567 = vpadalq_s16(vacc3x4567, vprod3x4567c2);
      vacc3x4567 = vpadalq_s16(vacc3x4567, vprod3x4567c3);

      k -= 8 * sizeof(int8_t);
    }

    if XNN_UNLIKELY(k != 0) {
      const int8x8_t va0 = vld1_s8(a0); a0 = (const int8_t*) ((uintptr_t) a0 + k);
      const int8x8_t va1 = vld1_s8(a1); a1 = (const int8_t*) ((uintptr_t) a1 + k);
      const int8x8_t va2 = vld1_s8(a2); a2 = (const int8_t*) ((uintptr_t) a2 + k);
      const int8x8_t va3 = vld1_s8(a3); a3 = (const int8_t*) ((uintptr_t) a3 + k);

      const int8x8_t vb0123c0 = vld1_s8((int8_t*)w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
      const int8x8_t vb4567c0 = vld1_s8((int8_t*)w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));

      const int16x8_t vprod0x0123c0 = vmull_s8(vb0123c0, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0), 0)));
      vacc0x0123 = vpadalq_s16(vacc0x0123, vprod0x0123c0);
      const int16x8_t vprod0x4567c0 = vmull_s8(vb4567c0, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0), 0)));
      vacc0x4567 = vpadalq_s16(vacc0x4567, vprod0x4567c0);
      const int16x8_t vprod1x0123c0 = vmull_s8(vb0123c0, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va1), 0)));
      vacc1x0123 = vpadalq_s16(vacc1x0123, vprod1x0123c0);
      const int16x8_t vprod1x4567c0 = vmull_s8(vb4567c0, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va1), 0)));
      vacc1x4567 = vpadalq_s16(vacc1x4567, vprod1x4567c0);
      const int16x8_t vprod2x0123c0 = vmull_s8(vb0123c0, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va2), 0)));
      vacc2x0123 = vpadalq_s16(vacc2x0123, vprod2x0123c0);
      const int16x8_t vprod2x4567c0 = vmull_s8(vb4567c0, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va2), 0)));
      vacc2x4567 = vpadalq_s16(vacc2x4567, vprod2x4567c0);
      const int16x8_t vprod3x0123c0 = vmull_s8(vb0123c0, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va3), 0)));
      vacc3x0123 = vpadalq_s16(vacc3x0123, vprod3x0123c0);
      const int16x8_t vprod3x4567c0 = vmull_s8(vb4567c0, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va3), 0)));
      vacc3x4567 = vpadalq_s16(vacc3x4567, vprod3x4567c0);

      if (k > 2 * sizeof(int8_t)) {
        const int8x8_t vb0123c1 = vld1_s8((int8_t*)w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
        const int8x8_t vb4567c1 = vld1_s8((int8_t*)w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));

        const int16x8_t vprod0x0123c1 = vmull_s8(vb0123c1, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0), 1)));
        vacc0x0123 = vpadalq_s16(vacc0x0123, vprod0x0123c1);
        const int16x8_t vprod0x4567c1 = vmull_s8(vb4567c1, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0), 1)));
        vacc0x4567 = vpadalq_s16(vacc0x4567, vprod0x4567c1);
        const int16x8_t vprod1x0123c1 = vmull_s8(vb0123c1, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va1), 1)));
        vacc1x0123 = vpadalq_s16(vacc1x0123, vprod1x0123c1);
        const int16x8_t vprod1x4567c1 = vmull_s8(vb4567c1, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va1), 1)));
        vacc1x4567 = vpadalq_s16(vacc1x4567, vprod1x4567c1);
        const int16x8_t vprod2x0123c1 = vmull_s8(vb0123c1, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va2), 1)));
        vacc2x0123 = vpadalq_s16(vacc2x0123, vprod2x0123c1);
        const int16x8_t vprod2x4567c1 = vmull_s8(vb4567c1, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va2), 1)));
        vacc2x4567 = vpadalq_s16(vacc2x4567, vprod2x4567c1);
        const int16x8_t vprod3x0123c1 = vmull_s8(vb0123c1, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va3), 1)));
        vacc3x0123 = vpadalq_s16(vacc3x0123, vprod3x0123c1);
        const int16x8_t vprod3x4567c1 = vmull_s8(vb4567c1, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va3), 1)));
        vacc3x4567 = vpadalq_s16(vacc3x4567, vprod3x4567c1);

        if (k > 4 * sizeof(int8_t)) {
          const int8x8_t vb0123c2 = vld1_s8((int8_t*)w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
          const int8x8_t vb4567c2 = vld1_s8((int8_t*)w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));

          const int16x8_t vprod0x0123c2 = vmull_s8(vb0123c2, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0), 2)));
          vacc0x0123 = vpadalq_s16(vacc0x0123, vprod0x0123c2);
          const int16x8_t vprod0x4567c2 = vmull_s8(vb4567c2, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0), 2)));
          vacc0x4567 = vpadalq_s16(vacc0x4567, vprod0x4567c2);
          const int16x8_t vprod1x0123c2 = vmull_s8(vb0123c2, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va1), 2)));
          vacc1x0123 = vpadalq_s16(vacc1x0123, vprod1x0123c2);
          const int16x8_t vprod1x4567c2 = vmull_s8(vb4567c2, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va1), 2)));
          vacc1x4567 = vpadalq_s16(vacc1x4567, vprod1x4567c2);
          const int16x8_t vprod2x0123c2 = vmull_s8(vb0123c2, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va2), 2)));
          vacc2x0123 = vpadalq_s16(vacc2x0123, vprod2x0123c2);
          const int16x8_t vprod2x4567c2 = vmull_s8(vb4567c2, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va2), 2)));
          vacc2x4567 = vpadalq_s16(vacc2x4567, vprod2x4567c2);
          const int16x8_t vprod3x0123c2 = vmull_s8(vb0123c2, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va3), 2)));
          vacc3x0123 = vpadalq_s16(vacc3x0123, vprod3x0123c2);
          const int16x8_t vprod3x4567c2 = vmull_s8(vb4567c2, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va3), 2)));
          vacc3x4567 = vpadalq_s16(vacc3x4567, vprod3x4567c2);
        }
      }
    }
    const int32x4_t vmultiplier = {0,}; // = vld1q_dup_s32(0);
    vacc0x0123 = vqrdmulhq_s32(vacc0x0123, vmultiplier);
    vacc0x4567 = vqrdmulhq_s32(vacc0x4567, vmultiplier);
    vacc1x0123 = vqrdmulhq_s32(vacc1x0123, vmultiplier);
    vacc1x4567 = vqrdmulhq_s32(vacc1x4567, vmultiplier);
    vacc2x0123 = vqrdmulhq_s32(vacc2x0123, vmultiplier);
    vacc2x4567 = vqrdmulhq_s32(vacc2x4567, vmultiplier);
    vacc3x0123 = vqrdmulhq_s32(vacc3x0123, vmultiplier);
    vacc3x4567 = vqrdmulhq_s32(vacc3x4567, vmultiplier);

    const int32x4_t vright_shift = {0,};// = vld1q_dup_s32(0);
    const int32x4_t vzero_shift_mask = {0,};// = vreinterpretq_s32_u32(vceqq_s32(vright_shift, vmovq_n_s32(0)));
    vacc0x0123 = vsraq_n_s32(vacc0x0123, vbicq_s32(vacc0x0123, vzero_shift_mask), 31);
    vacc0x4567 = vsraq_n_s32(vacc0x4567, vbicq_s32(vacc0x4567, vzero_shift_mask), 31);
    vacc1x0123 = vsraq_n_s32(vacc1x0123, vbicq_s32(vacc1x0123, vzero_shift_mask), 31);
    vacc1x4567 = vsraq_n_s32(vacc1x4567, vbicq_s32(vacc1x4567, vzero_shift_mask), 31);
    vacc2x0123 = vsraq_n_s32(vacc2x0123, vbicq_s32(vacc2x0123, vzero_shift_mask), 31);
    vacc2x4567 = vsraq_n_s32(vacc2x4567, vbicq_s32(vacc2x4567, vzero_shift_mask), 31);
    vacc3x0123 = vsraq_n_s32(vacc3x0123, vbicq_s32(vacc3x0123, vzero_shift_mask), 31);
    vacc3x4567 = vsraq_n_s32(vacc3x4567, vbicq_s32(vacc3x4567, vzero_shift_mask), 31);

    vacc0x0123 = vrshlq_s32(vacc0x0123, vright_shift);
    vacc0x4567 = vrshlq_s32(vacc0x4567, vright_shift);
    vacc1x0123 = vrshlq_s32(vacc1x0123, vright_shift);
    vacc1x4567 = vrshlq_s32(vacc1x4567, vright_shift);
    vacc2x0123 = vrshlq_s32(vacc2x0123, vright_shift);
    vacc2x4567 = vrshlq_s32(vacc2x4567, vright_shift);
    vacc3x0123 = vrshlq_s32(vacc3x0123, vright_shift);
    vacc3x4567 = vrshlq_s32(vacc3x4567, vright_shift);

    const int16x8_t voutput_zero_point = {0,};// = vld1q_dup_s16(0);
#if XNN_ARCH_ARM64
    const int16x8_t vacc0x01234567 = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc0x0123), vacc0x4567), voutput_zero_point);
    const int16x8_t vacc1x01234567 = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc1x0123), vacc1x4567), voutput_zero_point);
    const int16x8_t vacc2x01234567 = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc2x0123), vacc2x4567), voutput_zero_point);
    const int16x8_t vacc3x01234567 = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc3x0123), vacc3x4567), voutput_zero_point);

    int8x16_t vout0x01234567_1x01234567 = vqmovn_high_s16(vqmovn_s16(vacc0x01234567), vacc1x01234567);
    int8x16_t vout2x01234567_3x01234567 = vqmovn_high_s16(vqmovn_s16(vacc2x01234567), vacc3x01234567);
#else
    const int16x8_t vacc0x01234567 = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc0x0123), vqmovn_s32(vacc0x4567)), voutput_zero_point);
    const int16x8_t vacc1x01234567 = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc1x0123), vqmovn_s32(vacc1x4567)), voutput_zero_point);
    const int16x8_t vacc2x01234567 = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc2x0123), vqmovn_s32(vacc2x4567)), voutput_zero_point);
    const int16x8_t vacc3x01234567 = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc3x0123), vqmovn_s32(vacc3x4567)), voutput_zero_point);

    int8x16_t vout0x01234567_1x01234567 = vcombine_s8(vqmovn_s16(vacc0x01234567), vqmovn_s16(vacc1x01234567));
    int8x16_t vout2x01234567_3x01234567 = vcombine_s8(vqmovn_s16(vacc2x01234567), vqmovn_s16(vacc3x01234567));
#endif
    const int8x16_t voutput_min = {0,};// = vld1q_dup_s8(0);
    const int8x16_t voutput_max = {0,};// = vld1q_dup_s8(0);

    vout0x01234567_1x01234567 = vmaxq_s8(vout0x01234567_1x01234567, voutput_min);
    vout2x01234567_3x01234567 = vmaxq_s8(vout2x01234567_3x01234567, voutput_min);

    vout0x01234567_1x01234567 = vminq_s8(vout0x01234567_1x01234567, voutput_max);
    vout2x01234567_3x01234567 = vminq_s8(vout2x01234567_3x01234567, voutput_max);

    if (nc >= 8) {
      vst1_s8(c0 + 0, vget_low_s8(vout0x01234567_1x01234567));
      vst1_s8(c1 + 0, vget_high_s8(vout0x01234567_1x01234567));
      vst1_s8(c2 + 0, vget_low_s8(vout2x01234567_3x01234567));
      vst1_s8(c3 + 0, vget_high_s8(vout2x01234567_3x01234567));

      c0 = (int8_t*) ((uintptr_t) c0 + cn_stride);
      c1 = (int8_t*) ((uintptr_t) c1 + cn_stride);
      c2 = (int8_t*) ((uintptr_t) c2 + cn_stride);
      c3 = (int8_t*) ((uintptr_t) c3 + cn_stride);

      a0 = (const int8_t*) ((uintptr_t) a0 - kc);
      a1 = (const int8_t*) ((uintptr_t) a1 - kc);
      a2 = (const int8_t*) ((uintptr_t) a2 - kc);
      a3 = (const int8_t*) ((uintptr_t) a3 - kc);

      nc -= 8;
    } else {
      if (nc & 4) {
        vst1q_lane_u32((uint32_t*)__builtin_assume_aligned((int8_t*)c0, 1), vreinterpretq_u32_s8(vout0x01234567_1x01234567), 0); c0 += 4;
        vst1q_lane_u32((uint32_t*)__builtin_assume_aligned((int8_t*)c1, 1), vreinterpretq_u32_s8(vout0x01234567_1x01234567), 2); c1 += 4;
        vst1q_lane_u32((uint32_t*)__builtin_assume_aligned((int8_t*)c2, 1), vreinterpretq_u32_s8(vout2x01234567_3x01234567), 0); c2 += 4;
        vst1q_lane_u32((uint32_t*)__builtin_assume_aligned((int8_t*)c3, 1), vreinterpretq_u32_s8(vout2x01234567_3x01234567), 2); c3 += 4;
        vout0x01234567_1x01234567 = vextq_s8(vout0x01234567_1x01234567, vout0x01234567_1x01234567, 4);
        vout2x01234567_3x01234567 = vextq_s8(vout2x01234567_3x01234567, vout2x01234567_3x01234567, 4);
      }
      if (nc & 2) {
        vst1q_lane_u16((uint16_t*)__builtin_assume_aligned((int8_t*)c0, 1), vreinterpretq_u16_s8(vout0x01234567_1x01234567), 0); c0 += 2;
        vst1q_lane_u16((uint16_t*)__builtin_assume_aligned((int8_t*)c1, 1), vreinterpretq_u16_s8(vout0x01234567_1x01234567), 4); c1 += 2;
        vst1q_lane_u16((uint16_t*)__builtin_assume_aligned((int8_t*)c2, 1), vreinterpretq_u16_s8(vout2x01234567_3x01234567), 0); c2 += 2;
        vst1q_lane_u16((uint16_t*)__builtin_assume_aligned((int8_t*)c3, 1), vreinterpretq_u16_s8(vout2x01234567_3x01234567), 4); c3 += 2;
        vout0x01234567_1x01234567 = vextq_s8(vout0x01234567_1x01234567, vout0x01234567_1x01234567, 2);
        vout2x01234567_3x01234567 = vextq_s8(vout2x01234567_3x01234567, vout2x01234567_3x01234567, 2);
      }
      if (nc & 1) {
        vst1q_lane_s8(c0, vout0x01234567_1x01234567, 0);
        vst1q_lane_s8(c1, vout0x01234567_1x01234567, 8);
        vst1q_lane_s8(c2, vout2x01234567_3x01234567, 0);
        vst1q_lane_s8(c3, vout2x01234567_3x01234567, 8);
      }

      nc = 0;
    }
  } while (nc != 0);
}




constexpr int kFloatValuesPerNeonVector = 4;

// TODO(ahentz): Clean up.
using int8 = std::int8_t;
using uint8 = std::uint8_t;
using int16 = std::int16_t;
using uint16 = std::uint16_t;
using int32 = std::int32_t;
using uint32 = std::uint32_t;

template <int PerNeonSize> inline int RoundDownVectors(int size)
{
  return size & ~(PerNeonSize - 1);
}

// Allocates, at least, size bytes of uninitialized storage whose alignment is
// specified by alignment. The size parameter must be an integral multiple of
// alignment.
// Caller is responsible by freeing the allocated memory by calling free on
// the passed freeing_buffer pointer.
void *aligned_alloc(size_t alignment, size_t size, void **freeing_buffer)
{
  *freeing_buffer = malloc(size + alignment);
  const size_t offset = ((uintptr_t)*freeing_buffer) % alignment;                          // NOLINT
  return offset == 0 ? *freeing_buffer : ((char *)*freeing_buffer + (alignment - offset)); // NOLINT
}

inline int32_t AccumulateNeonLane(const int32x4_t lane)
{
#ifdef __aarch64__
  return vaddvq_s32(lane);
#else
  int64x2_t pairwiseAdded = vpaddlq_s32(lane);
  return vgetq_lane_s64(pairwiseAdded, 0) + vgetq_lane_s64(pairwiseAdded, 1);
#endif
}

} // namespace

// The implementation of dotprod detection is copied from ruy's internal
// function DetectDotprod().
// At the moment it's only implemented on Linux ARM64. Consider syncing again
// with ruy in the future to share improvements.
#if defined __linux__ && defined __aarch64__
inline bool DetectDotprodByLinuxAuxvMethod()
{
  // This is the value of HWCAP_ASIMDDP in sufficiently recent Linux headers,
  // however we need to support building against older headers for the time
  // being.
  const int kLocalHwcapAsimddp = 1 << 20;
  return getauxval(AT_HWCAP) & kLocalHwcapAsimddp;
}
#endif

inline bool DetectArmNeonDotprod()
{
#if defined __linux__ && defined __aarch64__
  return DetectDotprodByLinuxAuxvMethod();
#endif

  return false;
}

inline bool HasSdotInstruction()
{
  static const bool has_dotprod = DetectArmNeonDotprod();
  return has_dotprod;
}

#ifdef __aarch64__
// We interleave vector data to make the dot product logic more efficient.
// Suppose that vectors is:
//     a0 a1 a2 a3 a4 a5 ...
//     b0 b1 b2 b3 b4 b5 ...
//     c0 c1 c2 c3 c4 c5 ...
//     d0 d1 d2 d3 d4 d5 ...
//     e0 e1 e2 e3 e4 e5 ...
// This code interleaves them like this:
//     a0 a1 a2 a3 b0 b1 b2 b3 c0 c1 c2 c3 d0 d1 d2 d3 a4 a5 a6 a7 b4 ...
//     e0 e1 e2 e3 f0 f1 f2 f3 ...
// Once the data is interleaved, each 16-byte read from the vectors pointer
// contains 4 bytes from each of 4 vectors.
inline const int8_t *ShuffleVectors(const int8_t *vectors, const int n_batch, const int m_cols,
                                    void **shuffled_vectors_free)
{
  const int kWeightsPerUint32 = 4;

  int8 *shuffled_vectors = reinterpret_cast<int8 *>(
    aligned_alloc(kWeightsPerUint32, n_batch * m_cols, shuffled_vectors_free));

  for (int i = 0; i < n_batch; i += 4)
  {
    int8 *shuffled_vectors_ptr = shuffled_vectors + (i * m_cols);
    const int8 *unshuffled_vec0_ptr = reinterpret_cast<const int8 *>(vectors) + (i * m_cols);
    const int8 *unshuffled_vec1_ptr = reinterpret_cast<const int8 *>(vectors) + ((i + 1) * m_cols);
    const int8 *unshuffled_vec2_ptr = reinterpret_cast<const int8 *>(vectors) + ((i + 2) * m_cols);
    const int8 *unshuffled_vec3_ptr = reinterpret_cast<const int8 *>(vectors) + ((i + 3) * m_cols);
    const int8 *const end_vec0_ptr = unshuffled_vec1_ptr;

    while (unshuffled_vec0_ptr != end_vec0_ptr)
    {
      asm volatile(
        // This code path requires that (n_cols % 16) == 0 so we can safely
        // read in 16-byte chunks from each row.
        "ld1 {v0.16b}, [%[unshuffled_vec0_ptr]], #16\n"
        "ld1 {v1.16b}, [%[unshuffled_vec1_ptr]], #16\n"
        "ld1 {v2.16b}, [%[unshuffled_vec2_ptr]], #16\n"
        "ld1 {v3.16b}, [%[unshuffled_vec3_ptr]], #16\n"

        "st4 {v0.s, v1.s, v2.s, v3.s}[0], [%[shuffled_vectors_ptr]], #16\n"
        "st4 {v0.s, v1.s, v2.s, v3.s}[1], [%[shuffled_vectors_ptr]], #16\n"
        "st4 {v0.s, v1.s, v2.s, v3.s}[2], [%[shuffled_vectors_ptr]], #16\n"
        "st4 {v0.s, v1.s, v2.s, v3.s}[3], [%[shuffled_vectors_ptr]], #16\n"

        : [ unshuffled_vec0_ptr ] "+r"(unshuffled_vec0_ptr),
          [ unshuffled_vec1_ptr ] "+r"(unshuffled_vec1_ptr),
          [ unshuffled_vec2_ptr ] "+r"(unshuffled_vec2_ptr),
          [ unshuffled_vec3_ptr ] "+r"(unshuffled_vec3_ptr),
          [ shuffled_vectors_ptr ] "+r"(shuffled_vectors_ptr)
        :
        : "v0", "v1", "v2", "v3", "cc", "memory");
    }
  }

  return reinterpret_cast<const int8_t *>(shuffled_vectors);
}

// Notes about the speed of this version vs. the baseline (from memory):
// - With 256K of L1, we can keep a lot of vectors in cache.
//   I recall a reasonable speedup just by rearranging the loop to have
//   row on the outside and batch on the inside.
// - I also recall getting a nice speedup from sdot.
// - I tried many times to do better than the current implementation, using
//   loop unrolling and instruction reordering to avoid stalls, etc.
//   but I was not able to do significantly better. This code is, however,
//   much worse than what the processor spec sheet suggests is possible.
static void DotprodMatrixBatchFourVectorMultiplyAccumulate(const int8_t *__restrict__ matrix,
                                                           const int m_rows, const int m_cols,
                                                           const int8_t *vectors,
                                                           const float *scaling_factors,
                                                           int n_batch, float *__restrict__ result)
{
  void *shuffled_vectors_free;

  const int8_t *shuffled_vectors = ShuffleVectors(vectors, n_batch, m_cols, &shuffled_vectors_free);

  for (int row = 0; row < m_rows; row += 2)
  {
    for (int batch = 0; batch < n_batch; batch += 4)
    {
      float *result_ptr = result + (batch * m_rows) + row;
      const int8 *mat_ptr0 = matrix + (row * m_cols);
      const int8 *mat_ptr1 = matrix + ((row + 1) * m_cols);
      const int8 *mat_ptr0_end = mat_ptr1;
      const int8 *vec_ptr = shuffled_vectors + (batch * m_cols);
      const float *scaling_factors_ptr = scaling_factors + batch;
      const uint64_t wide_rows = m_rows * sizeof(float);
      const int8 *mat_ptr2 = matrix + ((row + 2) * m_cols);
      const int8 *mat_ptr3 = matrix + ((row + 3) * m_cols);

      asm volatile(
        // Zero out the accumulator registers.
        "dup v0.4s, wzr\n"
        "dup v1.4s, wzr\n"
        "dup v2.4s, wzr\n"
        "dup v3.4s, wzr\n"

        "1:\n" // batch_cols_loop

        // Read 16 more bytes from a pair of matrix rows.
        "ld1 {v12.16b}, [%[mat_ptr0]], #16\n"

        // Prefetch two rows ahead.
        "prfm pldl1strm, [%[mat_ptr2]]\n"
        "prfm pldl1strm, [%[mat_ptr3]]\n"

        // Read from input vectors 4 times; 64 bytes total.
        // Each 16-byte register contains parts of 4 vectors; see the
        // shuffle logic above.

        // From Benoit, places to look in the future:
        // - Move load instructions further from sdot
        // - Switch loop use-then-reload
        // - Do partial unrolling to use register space better
        "ld1 {v8.16b}, [%[vec_ptr]], #16\n"
        ".word 0x4f8ce100  // sdot v0.4s, v8.16b, v12.4b[0]\n"
        "ld1 {v9.16b}, [%[vec_ptr]], #16\n"
        ".word 0x4face121  // sdot v1.4s, v9.16b, v12.4b[1]\n"
        "ld1 {v10.16b}, [%[vec_ptr]], #16\n"
        ".word 0x4f8ce940  // sdot v0.4s, v10.16b, v12.4b[2]\n"
        "ld1 {v11.16b}, [%[vec_ptr]], #16\n"
        ".word 0x4face961  // sdot v1.4s, v11.16b, v12.4b[3]\n"

        // Update prefetch pointers.
        "add %[mat_ptr2], %[mat_ptr2], #16\n"
        "add %[mat_ptr3], %[mat_ptr3], #16\n"

        // Re-use those vectors for the next row as well.
        "ld1 {v13.16b}, [%[mat_ptr1]], #16\n"
        ".word 0x4f8de102  // sdot v2.4s, v8.16b, v13.4b[0]\n"
        ".word 0x4fade123  // sdot v3.4s, v9.16b, v13.4b[1]\n"
        ".word 0x4f8de942  // sdot v2.4s, v10.16b, v13.4b[2]\n"
        ".word 0x4fade963  // sdot v3.4s, v11.16b, v13.4b[3]\n"

        // If we're not done with these rows, continue.
        "cmp %[mat_ptr0], %[mat_ptr0_end]\n"
        "bne 1b\n" // batch_cols_loop

        // Done with the rows, sum the results.
        "add v0.4s, v0.4s, v1.4s\n"
        "add v2.4s, v2.4s, v3.4s\n"

        // Convert the per-vector sums to floating point.
        "scvtf v0.4s, v0.4s\n"
        "scvtf v1.4s, v2.4s\n"

        // Fetch scale factors.
        "ld1 {v4.4s}, [%[scaling_factors_ptr]]\n"

        // Multiply scale factors times sums.
        "fmul v0.4s, v4.4s, v0.4s\n"
        "fmul v1.4s, v4.4s, v1.4s\n"

        // Load previous result values.
        // The result position is:
        //   result[batch * m_rows + row]
        // Here that is factored into:
        //   result_ptr = result + row
        //   *result_ptr = res[0]
        //   (uint8*)result_ptr += (m_rows * sizeof(float))
        //   *result_ptr = res[1]
        //   ...
        // Since we're reading two rows at a time, though, we read both
        //   result[batch * m_rows + row]
        // and
        //   result[batch * m_rows + row + 1]
        "ld2 {v9.s, v10.s}[0], [%[result_ptr]], %[wide_rows]\n"
        "ld2 {v9.s, v10.s}[1], [%[result_ptr]], %[wide_rows]\n"
        "ld2 {v9.s, v10.s}[2], [%[result_ptr]], %[wide_rows]\n"
        "ld2 {v9.s, v10.s}[3], [%[result_ptr]], %[wide_rows]\n"

        // Go back to the starting position (subtract wide_rows * 4).
        "sub %[result_ptr], %[result_ptr], %[wide_rows], lsl #2\n"

        // Add previous result values.
        "fadd v9.4s, v9.4s, v0.4s\n"
        "fadd v10.4s, v10.4s, v1.4s\n"

        // Store results.
        "st2 {v9.s, v10.s}[0], [%[result_ptr]], %[wide_rows]\n"
        "st2 {v9.s, v10.s}[1], [%[result_ptr]], %[wide_rows]\n"
        "st2 {v9.s, v10.s}[2], [%[result_ptr]], %[wide_rows]\n"
        "st2 {v9.s, v10.s}[3], [%[result_ptr]], %[wide_rows]\n"
        : [ mat_ptr0 ] "+r"(mat_ptr0), [ mat_ptr1 ] "+r"(mat_ptr1), [ vec_ptr ] "+r"(vec_ptr),
          [ result_ptr ] "+r"(result_ptr), [ mat_ptr2 ] "+r"(mat_ptr2), [ mat_ptr3 ] "+r"(mat_ptr3)
        : [ mat_ptr0_end ] "r"(mat_ptr0_end), [ scaling_factors_ptr ] "r"(scaling_factors_ptr),
          [ wide_rows ] "r"(wide_rows)
        : "x0", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12",
          "v13", "cc", "memory");
    }
  }

  free(shuffled_vectors_free);
}

static void DotprodMatrixBatchFourVectorMultiplyAccumulate(
  const int8_t *__restrict__ matrix, const int m_rows, const int m_cols, const int8_t *vectors,
  const float *scaling_factors, int n_batch, float *__restrict__ result,
  const float *per_channel_scale, const int32_t *input_offset, int32_t *row_sums)
{
  void *shuffled_vectors_free;
  const int8_t *shuffled_vectors = ShuffleVectors(vectors, n_batch, m_cols, &shuffled_vectors_free);

  for (int row = 0; row < m_rows; row += 2)
  {
    const float *channel_scales_ptr = per_channel_scale + row;
    int32_t *row_sums_ptr = row_sums ? row_sums + row : nullptr;
    for (int batch = 0; batch < n_batch; batch += 4)
    {
      float *result_ptr = result + (batch * m_rows) + row;
      const int8 *mat_ptr0 = matrix + (row * m_cols);
      const int8 *mat_ptr1 = matrix + ((row + 1) * m_cols);
      const int8 *mat_ptr0_end = mat_ptr1;
      const int8 *vec_ptr = shuffled_vectors + (batch * m_cols);
      const float *scaling_factors_ptr = scaling_factors + batch;
      const uint64_t wide_rows = m_rows * sizeof(float);
      const int32_t *batch_offsets_ptr = input_offset + batch;
      const int32_t is_channel_scale_nullptr = per_channel_scale == nullptr;
      const int32_t is_row_sums_nullptr = row_sums_ptr == nullptr;
      asm volatile(
        "dup v0.4s, wzr\n"
        "dup v1.4s, wzr\n"
        "dup v2.4s, wzr\n"
        "dup v3.4s, wzr\n"
        // Load zero points.
        "ld1 {v7.4s}, [%[batch_offsets_ptr]]\n"
        "ld1 {v4.4s}, [%[scaling_factors_ptr]]\n"
        // Zero out zero point accumulators.
        "dup v14.4s, wzr\n"
        "dup v15.4s, wzr\n"

        // Load per channel scales if not null.
        "cmp %w[is_channel_scale_nullptr], #0\n"
        "bne 1f\n"
        "ld1r {v16.4s}, [%[channel_scales_ptr]], #4\n"
        "ld1r {v17.4s}, [%[channel_scales_ptr]]\n"
        "fmul v16.4s, v16.4s, v4.4s\n"
        "fmul v17.4s, v17.4s, v4.4s\n"
        "b 2f\n"
        "1:\n"
        "mov v16.16b, v4.16b\n"
        "mov v17.16b, v4.16b\n"
        "2:\n"
        "ld1 {v12.16b}, [%[mat_ptr0]], #16\n"
        "ld1 {v8.16b}, [%[vec_ptr]], #16\n"
        ".word 0x4f8ce100  // sdot v0.4s, v8.16b, v12.4b[0]\n"
        "ld1 {v9.16b}, [%[vec_ptr]], #16\n"
        ".word 0x4face121  // sdot v1.4s, v9.16b, v12.4b[1]\n"
        "ld1 {v10.16b}, [%[vec_ptr]], #16\n"
        ".word 0x4f8ce940  // sdot v0.4s, v10.16b, v12.4b[2]\n"
        "ld1 {v11.16b}, [%[vec_ptr]], #16\n"
        ".word 0x4face961  // sdot v1.4s, v11.16b, v12.4b[3]\n"
        "ld1 {v13.16b}, [%[mat_ptr1]], #16\n"
        ".word 0x4f8de102  // sdot v2.4s, v8.16b, v13.4b[0]\n"
        ".word 0x4fade123  // sdot v3.4s, v9.16b, v13.4b[1]\n"
        ".word 0x4f8de942  // sdot v2.4s, v10.16b, v13.4b[2]\n"
        ".word 0x4fade963  // sdot v3.4s, v11.16b, v13.4b[3]\n"
        "cmp %w[is_row_sums_nullptr], #1\n"
        "bne 3f\n"
        // Accumulate row_sums for zero point calculations.
        "saddlp v12.8h, v12.16b\n"
        "saddlp v13.8h, v13.16b\n"
        "sadalp v14.4s, v12.8h\n"
        "sadalp v15.4s, v13.8h\n"
        "3:\n"
        "cmp %[mat_ptr0], %[mat_ptr0_end]\n"
        "bne 2b\n"
        "add v0.4s, v0.4s, v1.4s\n"
        "add v2.4s, v2.4s, v3.4s\n"

        "cmp %w[is_row_sums_nullptr], #1\n"
        "bne 4f\n"
        // Calculate zero point offsets.
        "addv s14, v14.4s\n"
        "addv s15, v15.4s\n"
        "dup v14.4s, v14.s[0]\n"
        "dup v15.4s, v15.s[0]\n"
        "b 5f\n"
        "4:\n"
        "ld1r {v14.4s}, [%[row_sums_ptr]], #4\n"
        "ld1r {v15.4s}, [%[row_sums_ptr]]\n"
        "5:\n"

        "mul v14.4s, v14.4s, v7.4s\n"
        "mul v15.4s, v15.4s, v7.4s\n"
        "sub v0.4s, v0.4s, v14.4s\n"
        "sub v2.4s, v2.4s, v15.4s\n"

        "scvtf v0.4s, v0.4s\n"
        "scvtf v1.4s, v2.4s\n"

        // Multiply scale.
        "fmul v0.4s, v16.4s, v0.4s\n"
        "fmul v1.4s, v17.4s, v1.4s\n"

        "ld2 {v9.s, v10.s}[0], [%[result_ptr]], %[wide_rows]\n"
        "ld2 {v9.s, v10.s}[1], [%[result_ptr]], %[wide_rows]\n"
        "ld2 {v9.s, v10.s}[2], [%[result_ptr]], %[wide_rows]\n"
        "ld2 {v9.s, v10.s}[3], [%[result_ptr]], %[wide_rows]\n"
        "sub %[result_ptr], %[result_ptr], %[wide_rows], lsl #2\n"
        "fadd v9.4s, v9.4s, v0.4s\n"
        "fadd v10.4s, v10.4s, v1.4s\n"
        "st2 {v9.s, v10.s}[0], [%[result_ptr]], %[wide_rows]\n"
        "st2 {v9.s, v10.s}[1], [%[result_ptr]], %[wide_rows]\n"
        "st2 {v9.s, v10.s}[2], [%[result_ptr]], %[wide_rows]\n"
        "st2 {v9.s, v10.s}[3], [%[result_ptr]], %[wide_rows]\n"
        : [ mat_ptr0 ] "+r"(mat_ptr0), [ mat_ptr1 ] "+r"(mat_ptr1), [ vec_ptr ] "+r"(vec_ptr),
          [ result_ptr ] "+r"(result_ptr), [ row_sums_ptr ] "+r"(row_sums_ptr)
        : [ mat_ptr0_end ] "r"(mat_ptr0_end), [ scaling_factors_ptr ] "r"(scaling_factors_ptr),
          [ wide_rows ] "r"(wide_rows), [ channel_scales_ptr ] "r"(channel_scales_ptr),
          [ batch_offsets_ptr ] "r"(batch_offsets_ptr),
          [ is_channel_scale_nullptr ] "r"(is_channel_scale_nullptr),
          [ is_row_sums_nullptr ] "r"(is_row_sums_nullptr)
        : "x0", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12",
          "v13", "v14", "v15", "v16", "v17", "w0", "w1", "cc", "memory");
    }
  }

  free(shuffled_vectors_free);
}

// The DotprodMatrixBatchFourVectorMultiplyAccumulate kernel processes 4
// vectors in the same time as the baseline processes 1 vector. However, it
// requires 4 vectors of input.
//
// To take advantage of this speed difference, we add some zero-valued
// vectors to the batch so that n_batch is a multiple of 4. Then we execute
// DotprodMatrixBatchPaddedFourVectorMultiplyAccumulate on that padded batch,
// then extract just the results we want at the end (ignoring the extra padding
// outputs).
//
// The relative cost of the padding is large when the matrix is smaller than
// 128x128, so we don't use this code path on small matrices. On larger
// matrices, the computation cost dwarfs the padding cost, making this code
// viable.
//
// If we ignore the cost of padding, this kernel is:
//    1x the speed of NeonMatrixBatchVectorMultiplyImpl for n_batch = 1
//    2x the speed of NeonMatrixBatchVectorMultiplyImpl for n_batch = 2
//    3x the speed of NeonMatrixBatchVectorMultiplyImpl for n_batch = 3
//    ...
//
// We don't use this kernel when n_batch = 1 because the baseline kernel
// is fine for that case.
inline void DotprodMatrixBatchPaddedFourVectorMultiplyAccumulate(
  const int8_t *__restrict__ matrix, const int m_rows, const int m_cols, const int8_t *vectors,
  const float *scaling_factors, int n_batch, float *__restrict__ result,
  const float *per_channel_scale, const int32_t *input_offset, int32_t *row_sums)
{
  const int kWeightsPerUint32 = 4;

  // Round to the nearest multiple of 4.
  int batch_round_up = n_batch;
  if (n_batch % 4 != 0)
  {
    batch_round_up += (4 - n_batch % 4);
  }
  assert(n_batch <= batch_round_up);

  void *padded_vectors_free;
  const int padded_vectors_size = batch_round_up * m_cols;
  int8_t *padded_vectors = reinterpret_cast<int8_t *>(
    aligned_alloc(kWeightsPerUint32, padded_vectors_size, &padded_vectors_free));
  memset(padded_vectors, 0, padded_vectors_size);

  void *padded_result_free;
  const int result_size = n_batch * m_rows * sizeof(float);
  const int padded_result_size = batch_round_up * m_rows * sizeof(float);
  float *padded_result = reinterpret_cast<float *>(
    aligned_alloc(kWeightsPerUint32, padded_result_size, &padded_result_free));
  memcpy(padded_result, result, result_size);
  memset(reinterpret_cast<char *>(padded_result) + result_size, 0,
         padded_result_size - result_size);

  // Copy the input into the padded data structure.
  assert(n_batch * m_cols <= padded_vectors_size);
  memcpy(padded_vectors, vectors, n_batch * m_cols);

  void *padded_scaling_factors_free;
  const int padded_scaling_factors_size = batch_round_up * sizeof(float);
  float *padded_scaling_factors = reinterpret_cast<float *>(
    aligned_alloc(kWeightsPerUint32, padded_scaling_factors_size, &padded_scaling_factors_free));
  assert(static_cast<int>(n_batch * sizeof(float)) <= padded_scaling_factors_size);
  assert(static_cast<int>(batch_round_up * sizeof(float)) <= padded_scaling_factors_size);
  memset(padded_scaling_factors, 0, batch_round_up * sizeof(float));
  memcpy(padded_scaling_factors, scaling_factors, n_batch * sizeof(float));

  if (input_offset != nullptr)
  {
    void *padded_input_offset_free;
    const int padded_input_offset_size = batch_round_up * sizeof(int32_t);
    int32_t *padded_input_offset = reinterpret_cast<int32_t *>(
      aligned_alloc(kWeightsPerUint32, padded_input_offset_size, &padded_input_offset_free));
    assert(static_cast<int>(n_batch * sizeof(int32_t)) <= padded_input_offset_size);
    assert(static_cast<int>(batch_round_up * sizeof(int32_t)) <= padded_input_offset_size);
    memset(padded_input_offset, 0, batch_round_up * sizeof(int32_t));
    memcpy(padded_input_offset, input_offset, n_batch * sizeof(int32_t));

    // Call the main kernel.
    DotprodMatrixBatchFourVectorMultiplyAccumulate(
      matrix, m_rows, m_cols, padded_vectors, padded_scaling_factors, batch_round_up, padded_result,
      per_channel_scale, padded_input_offset, row_sums);

    free(padded_input_offset_free);
  }
  else
  {
    // Call the main kernel.
    DotprodMatrixBatchFourVectorMultiplyAccumulate(matrix, m_rows, m_cols, padded_vectors,
                                                   padded_scaling_factors, batch_round_up,
                                                   padded_result);
  }
  memcpy(result, padded_result, result_size);

  free(padded_result_free);
  free(padded_vectors_free);
  free(padded_scaling_factors_free);
}

inline void DotprodMatrixBatchPaddedFourVectorMultiplyAccumulate(
  const int8_t *__restrict__ matrix, const int m_rows, const int m_cols, const int8_t *vectors,
  const float *scaling_factors, int n_batch, float *__restrict__ result)
{
  DotprodMatrixBatchPaddedFourVectorMultiplyAccumulate(
    matrix, m_rows, m_cols, vectors, scaling_factors, n_batch, result,
    /*per_channel_scale=*/nullptr, /*input_offset=*/nullptr,
    /*row_sums=*/nullptr);
}
#endif // __aarch64__

inline void NeonCwiseClipping(float *vector, const int v_size, const float clipping_value)
{
  const float32x4_t clipping_value_f32x4 = vmovq_n_f32(clipping_value);
  const float32x4_t neg_clipping_value_f32x4 = vmovq_n_f32(-clipping_value);

  int i = 0;
  for (; i <= v_size - kFloatValuesPerNeonVector; i += kFloatValuesPerNeonVector)
  {
    // Load from memory to vector.
    float32x4_t v_f32x4 = vld1q_f32(vector + i);
    // Clip between clipping_value and -clipping_value.
    v_f32x4 = vminq_f32(clipping_value_f32x4, v_f32x4);
    v_f32x4 = vmaxq_f32(neg_clipping_value_f32x4, v_f32x4);
    // Save to output.
    vst1q_f32(vector + i, v_f32x4);
  }
  for (; i < v_size; i++)
  {
    vector[i] = std::max(std::min(clipping_value, vector[i]), -clipping_value);
  }
}

inline bool NeonIsZeroVector(const float *vector, int v_size)
{
  // If v_size is not divisible by kFloatWeightsPerNeonLane, we cannot
  // use the main vectorized loop, and we need to process sequentially.
  // postamble_start shows the start index where this should happen.
  const int postamble_start = v_size - (v_size & (kFloatWeightsPerNeonLane - 1));

  const float32x4_t zero_x4_float = vmovq_n_f32(0.0f);
  for (int v = 0; v < postamble_start; v += kFloatWeightsPerNeonLane)
  {
    const float32x4_t i_x4_float = vld1q_f32(vector + v);
    uint32x4_t cmp_result = vceqq_f32(i_x4_float, zero_x4_float);
    if (vgetq_lane_u32(cmp_result, 0) == 0)
      return false;
    if (vgetq_lane_u32(cmp_result, 1) == 0)
      return false;
    if (vgetq_lane_u32(cmp_result, 2) == 0)
      return false;
    if (vgetq_lane_u32(cmp_result, 3) == 0)
      return false;
  }

  // Postamble loop
  for (int v = postamble_start; v < v_size; ++v)
  {
    if (vector[v] != 0.0)
      return false;
  }
  return true;
}

inline void NeonCpuBackendGemm(const int8_t *input, const int32_t *bias,
                               const int8_t *input_to_gate_weights, int32_t n_batch,
                               int32_t n_input, int32_t n_output, int32_t, int32_t *scratch,
                               ruy::Context *ruy_context)
{
  MatrixParams<int8_t> lhs_params;
  lhs_params.order = Order::kRowMajor;
  lhs_params.rows = n_output;
  lhs_params.cols = n_input;
  lhs_params.cache_policy = CachePolicy::kAlwaysCache;

  MatrixParams<int8_t> rhs_params;
  rhs_params.order = Order::kColMajor;
  rhs_params.rows = n_input;
  rhs_params.cols = n_batch;

  MatrixParams<int32_t> dst_params;
  dst_params.order = Order::kColMajor;
  dst_params.rows = n_output;
  dst_params.cols = n_batch;

  GemmParams<int32_t, int32_t> gemm_params;
  if (bias)
  {
    gemm_params.bias = bias;
  }

  // Below code is from tflite::cpu_backend_gemm::detail::GemmImplUsingRuy
  ruy::Matrix<int8_t> ruy_lhs;
  ruy::Matrix<int8_t> ruy_rhs;
  ruy::Matrix<int32_t> ruy_dst;
  // Note that cache is always enabled for input and weight tensors
  ruy_support::MakeRuyMatrix(lhs_params, input_to_gate_weights, &ruy_lhs, true);
  ruy_support::MakeRuyMatrix(rhs_params, input, &ruy_rhs, true);
  ruy_support::MakeRuyMatrix(dst_params, scratch, &ruy_dst);

  ruy::BasicSpec<int32_t, int32_t> ruy_mul_params;
  ruy_support::MakeRuyMulParams(gemm_params, &ruy_mul_params);

  ruy::Mul(ruy_lhs, ruy_rhs, ruy_mul_params, ruy_context, &ruy_dst);
}

inline void NeonSub1Vector(const float *vector, int v_size, float *result)
{
  // If v_size is not divisible by the vector size, then we need to process the
  // final few elements sequentially. postamble_start shows the start index
  // where this should happen.
  const int postamble_start = RoundDownVectors<kFloatValuesPerNeonVector>(v_size);

  float32x4_t one_f32x4 = vmovq_n_f32(1.0);
  int v = 0;
  for (; v < postamble_start; v += kFloatValuesPerNeonVector)
  {
    // Load 4 float values from the current pointers of the input column and
    // subtract from 1.
    float32x4_t v_f32x4 = vld1q_f32(vector + v);
    float32x4_t result_f32x4 = vsubq_f32(one_f32x4, v_f32x4);
    // Save to output.
    vst1q_f32(result + v, result_f32x4);
  }
  for (; v < v_size; v++)
  {
    result[v] = 1.0f - vector[v];
  }
}

inline void NeonSymmetricQuantizeFloats(const float *values, const int size,
                                        int8_t *quantized_values, float *min, float *max,
                                        float *scaling_factor)
{
  // TODO(raziel): vectorize min/max calculation.
  auto minmax = std::minmax_element(values, values + size);
  *min = *minmax.first;
  *max = *minmax.second;
  const int kScale = 127;
  const float range = std::max(std::abs(*min), std::abs(*max));
  if (range == 0)
  {
    memset(quantized_values, 0, size * sizeof(int8_t));
    *scaling_factor = 1;
    return;
  }
  *scaling_factor = range / kScale;
  const float scaling_factor_inv = kScale / range;

  const int postamble_start = size - (size & (2 * kFloatWeightsPerNeonLane - 1));

  // Vectorized constants.
  const float32x4_t q_factor_f32x4 = vmovq_n_f32(scaling_factor_inv);
  const float32x4_t point5_f32x4 = vmovq_n_f32(0.5);
  const float32x4_t zero_f32x4 = vmovq_n_f32(0.0);
  const int32x4_t scale_i32x4 = vmovq_n_s32(kScale);
  const int32x4_t neg_scale_i32x4 = vmovq_n_s32(-kScale);

  for (int i = 0; i < postamble_start; i += 2 * kFloatWeightsPerNeonLane)
  {
    // Implements the vectorized version of the following:
    // const int32_t quantized_value = static_cast<int32>(
    //    std::round(*scaling_factor * values[i]));
    // Since the vectorized round intrinsics (vrndqa_f32) is not supported
    // on all Neon flavors, we use the following method for rounding: if (x
    // < 0) (int)(x - 0.5) if (x >= 0) (int)(x + 0.5)
    float32x4_t value0_f32x4 = vld1q_f32(&values[i]);
    float32x4_t value1_f32x4 = vld1q_f32(&values[i + kFloatWeightsPerNeonLane]);
    float32x4_t mul0_f32x4 = vmulq_f32(value0_f32x4, q_factor_f32x4);
    float32x4_t mul1_f32x4 = vmulq_f32(value1_f32x4, q_factor_f32x4);

    int32x4_t cmp_with_zero0_ui32x4 = (int32x4_t)vcltq_f32(mul0_f32x4, zero_f32x4); // NOLINT
    int32x4_t cmp_with_zero1_ui32x4 = (int32x4_t)vcltq_f32(mul1_f32x4, zero_f32x4); // NOLINT

    float32x4_t cmp_with_zero0_f32x4 = vcvtq_f32_s32(cmp_with_zero0_ui32x4);
    float32x4_t cmp_with_zero1_f32x4 = vcvtq_f32_s32(cmp_with_zero1_ui32x4);
    cmp_with_zero0_f32x4 = vaddq_f32(cmp_with_zero0_f32x4, point5_f32x4);
    cmp_with_zero1_f32x4 = vaddq_f32(cmp_with_zero1_f32x4, point5_f32x4);

    mul0_f32x4 = vaddq_f32(mul0_f32x4, cmp_with_zero0_f32x4);
    mul1_f32x4 = vaddq_f32(mul1_f32x4, cmp_with_zero1_f32x4);

    int32x4_t f2i0_i32x4 = vcvtq_s32_f32(mul0_f32x4);
    int32x4_t f2i1_i32x4 = vcvtq_s32_f32(mul1_f32x4);

    // Implements the vectorized version of the folowing block:
    //  quantized_values[i] = std::min(kScale, std::max(-kScale,
    //  quantized_value));
    int32x4_t max0_i32x4 = vmaxq_s32(f2i0_i32x4, neg_scale_i32x4);
    int32x4_t max1_i32x4 = vmaxq_s32(f2i1_i32x4, neg_scale_i32x4);
    int32x4_t min0_i32x4 = vminq_s32(max0_i32x4, scale_i32x4);
    int32x4_t min1_i32x4 = vminq_s32(max1_i32x4, scale_i32x4);

    int16x4_t min0_16x4 = vmovn_s32(min0_i32x4);
    int16x4_t min1_16x4 = vmovn_s32(min1_i32x4);

    int16x8_t min_16x8 = vcombine_s16(min0_16x4, min1_16x4);
    int8x8_t min_s8x8 = vqmovn_s16(min_16x8);
    vst1_s8(&quantized_values[i], min_s8x8);
  }

  for (int i = postamble_start; i < size; ++i)
  {
    const int32_t quantized_value =
      static_cast<int32_t>(std::round(scaling_factor_inv * values[i]));
    quantized_values[i] = std::min(kScale, std::max(-kScale, quantized_value));
  }
}

inline void NeonMatrixBatchVectorMultiplyAccumulate(const int8_t *__restrict__ matrix,
                                                    const int m_rows, const int m_cols,
                                                    const int8_t *__restrict__ vectors,
                                                    const float *scaling_factors, int n_batch,
                                                    float *__restrict__ result, int result_stride)
{
#ifdef __aarch64__
  if (HasSdotInstruction() && m_cols % 16 == 0 && m_rows % 2 == 0 && m_rows >= n_batch)
  {
    if (n_batch % 4 == 0 && result_stride == 1)
    {
      // Benchmarks suggest that it's always better to use the batch code
      // when we can, even on small matrices.
      DotprodMatrixBatchFourVectorMultiplyAccumulate(matrix, m_rows, m_cols, vectors,
                                                     scaling_factors, n_batch, result);
      return;
    }
    else if (result_stride == 1 && n_batch >= 2 && m_rows * m_cols >= 128 * 128)
    {
      DotprodMatrixBatchPaddedFourVectorMultiplyAccumulate(matrix, m_rows, m_cols, vectors,
                                                           scaling_factors, n_batch, result);
      return;
    }
  }
#endif // __aarch64__

  static const int kWeightsPerUint32 = 4;
  static const int kWeightsPerNeonLane = 16;
  // Assuming *matrix is kWeightsPerUint32-byte aligned,
  // every row of the matrix is also
  // kWeightsPerUint32-byte aligned as long as cols is
  // a multiple of kWeightsPerUint32. The assumption
  // is currently satisfied by TFLite's 16-byte memory
  // alignment scheme.
  //
  // Otherwise, we allocate an aligned memory block and set
  // a flag to later copy rows from matrix to the block
  // for aligned multiplication.
  bool unaligned = false;
  int8_t *aligned_row = nullptr;
  void *aligned_row_free = nullptr;
  if ((m_cols & (kWeightsPerUint32 - 1)) != 0)
  {
    unaligned = true;
    aligned_row = (int8_t *)aligned_alloc(kWeightsPerUint32, m_cols, // NOLINT
                                          &aligned_row_free);
  }
  void *aligned_vec_free = nullptr;
  int8_t *aligned_vec = (int8_t *)aligned_alloc(kWeightsPerUint32, m_cols, // NOLINT
                                                &aligned_vec_free);

  // If m_cols is not at least kWeightsPerNeonLane, we cannot use the main
  // vectorized loop, and we need to process sequentially. postamble_half_start
  // shows the start index where this should happen. Between postamble_start and
  // postamble_half_start we can still process kWeightsPerNeonLane >> 1 in a
  // vectorized form.
  const int postamble_half_start = m_cols & ~(kWeightsPerNeonLane - 1);
  const int postamble_start = m_cols & ~((kWeightsPerNeonLane >> 1) - 1);

  uint32_t mr = 2;
  uint32_t mc = n_batch;
  uint32_t nr = 4;
  uint32_t nc = m_rows;
  uint32_t kc = m_cols;
  uint32_t kr = 2;
  const size_t kc_stride = RoundUp(kc, kr);
  const size_t nc_stride = RoundUp(nc, nr);
  //std::vector<int8_t> w(nc_stride*kc_stride);
  //std::vector<int8_t> c(mc*nc);
  int32_t* w = (int32_t*)result;

  for (uint32_t m = 0; m < mc; m += mr) {
    const uint32_t mb = std::min(mc - m, mr);
    for (uint32_t n = 0; n < nc; n += nr) {
      const uint32_t nb = std::min(nc - n, nr);
        // xnn_qs8_gemm_minmax_ukernel_2x8c2__neon_mlal_padal_dup(
        //   mb, nb, kc * sizeof(int8_t),
        //   vectors + m * kc, kc * sizeof(int8_t),
        //   matrix + n * (kc_stride * sizeof(int8_t)),
        //   w + (m) * nc + n, nc * sizeof(int8_t), nr * sizeof(int8_t));
      NEON_GEMM_Int8Operands_AccumTwoWithin16Bits(matrix+n*kc_stride, vectors+m*kc, w+m*nc, kc_stride);
    }
  }
  /* for (int batch = 0; batch < n_batch; ++batch) */
  /* { */
  /*   const float batch_scaling_factor = scaling_factors[batch]; */
  /*   // Copy the vector data to an aligned vector. */
  /*   memcpy(aligned_vec, vectors + batch * m_cols, sizeof(int8_t) * m_cols); */
  /*   // Compute dot-product for every column. */
  /*   for (int row = 0; row < m_rows; ++row, result += result_stride) */
  /*   { */
  /*     // Get the address of the first element of the row. */
  /*     int8_t *row_ptr = (int8_t *)matrix + row * m_cols; // NOLINT */
  /*     if (unaligned) */
  /*     { */
  /*       memcpy(aligned_row, row_ptr, sizeof(int8_t) * m_cols); */
  /*       row_ptr = aligned_row; */
  /*     } */

  /*     // Initialize the dot product sum for the row to 0. */
  /*     int32x4_t dotprod_32x4 = vmovq_n_s32(0); */

  /*     // Prefetch the row to cache. */
  /*     __builtin_prefetch(row_ptr, 0 /\* prefetch for read *\/, 3 /\* temporal locality *\/); */

  /*     // For every block of 16 8-bit elements. */
  /*     int col = 0; */
  /*     for (; col < postamble_half_start; col += kWeightsPerNeonLane) */
  /*     { */
  /*       // Load 16 8-bit values from the row and vector, each, to operate on. */
  /*       // Here the assumption is that each buffer is 4-byte aligned. Otherwise, */
  /*       // performance may suffer significantly. */
  /*       assert( // NOLINT */
  /*         ((uintptr_t)(&row_ptr[col]) & (kWeightsPerUint32 - 1)) == 0); */
  /*       const int8x16_t s1_8x16 = vld1q_s8((const int8_t *)(aligned_vec + col)); */
  /*       const int8x16_t s2_8x16 = vld1q_s8((const int8_t *)(row_ptr + col)); */
  /*       // Multiply the low bits (i.e. the lower 8 8bit numbers in the */
  /*       // registers). */
  /*       int16x8_t prod_16x8 = vmull_s8(vget_low_s8(s1_8x16), vget_low_s8(s2_8x16)); */
  /*       // Multiply the high bits (i.e. the higher 8 8bit numbers in the */
  /*       // registers), and accumulate with the result of the low bits product. */
  /*       // The assumption here is that overflow will not happen as we quantize */
  /*       // our values to be in the range [-127, 127]. As such the sum of the 2 */
  /*       // products is always strictly smaller than 15-bits (32767 in absolute */
  /*       // value). */
  /*       prod_16x8 = vmlal_s8(prod_16x8, vget_high_s8(s1_8x16), vget_high_s8(s2_8x16)); */

  /*       dotprod_32x4 = vpadalq_s16(dotprod_32x4, prod_16x8); */
  /*     } // for col */

  /*     // Half iteration dealing only 8 elements */
  /*     // TODO(raziel): if (ABSL_PREDICT_FALSE(col < postamble_start)) */
  /*     if (col < postamble_start) */
  /*     { */
  /*       // Load 8 8-bit values from the row and column each to operate on. */
  /*       // Here the assumption is that each buffer is 4-bytes aligned. */
  /*       // Otherwise, performance may suffer significantly. */
  /*       assert( // NOLINT */
  /*         ((uintptr_t)(&row_ptr[col]) & (kWeightsPerUint32 - 1)) == 0); */
  /*       const int8x8_t s1_8x8 = vld1_s8((const int8_t *)(aligned_vec + col)); */
  /*       const int8x8_t s2_8x8 = vld1_s8((const int8_t *)(row_ptr + col)); */
  /*       const int16x8_t prod_16x8 = vmull_s8(s1_8x8, s2_8x8); */
  /*       dotprod_32x4 = vpadalq_s16(dotprod_32x4, prod_16x8); */
  /*       col += (kWeightsPerNeonLane >> 1); */
  /*     } */
  /*     // Add the 4 intermediate sum values to get the final dot-prod value for */
  /*     // this row. */
  /*     int32_t dotprod = AccumulateNeonLane(dotprod_32x4); */
  /*     // Postamble loop. */
  /*     // TODO(raziel): if (ABSL_PREDICT_FALSE(col < m_cols)) */
  /*     for (; col < m_cols; ++col) */
  /*     { */
  /*       dotprod += row_ptr[col] * aligned_vec[col]; */
  /*     } // for col */

  /*     *result += dotprod * batch_scaling_factor; */
  /*   } // for row */
  /* }   // for batch */

  if (unaligned)
  {
    free(aligned_row_free);
  }
  free(aligned_vec_free);
}

inline void NeonMatrixBatchVectorMultiplyAccumulate(const float *matrix, int m_rows, int m_cols,
                                                    const float *vector, int n_batch, float *result,
                                                    int result_stride)
{
  // If v_size is not divisible by kWeightsPerNeonLane, we cannot use the main
  // vectorized loop, and we need to process sequentially. postamble_start shows
  // the start index where this should happen.
  const int postamble_start = m_cols - (m_cols & (kFloatWeightsPerNeonLane - 1));

  for (int b = 0; b < n_batch; b++)
  {
    float *result_in_batch = result + b * m_rows * result_stride;
    const float *vector_in_batch = vector + b * m_cols;
    const float *matrix_row = matrix;

    // Main matrix by vector multiplication loop
    for (int r = 0; r < m_rows; r++)
    {
      float32x4_t acc_32x4 = vmovq_n_f32(0.0);
      for (int c = 0; c < postamble_start; c += kFloatWeightsPerNeonLane)
      {
        // Load 4 float values from vector and matrix row.
        float32x4_t vector_f32x4 = vld1q_f32(vector_in_batch + c);
        float32x4_t matrix_f32x4 = vld1q_f32(matrix_row + c);
        // Multiply the vector and matrix row and add to accumulator.
        acc_32x4 = vmlaq_f32(acc_32x4, matrix_f32x4, vector_f32x4);
      }
      // Add the 4 intermediate sum values to get the final dot-prod value for
      // this column.
      *result_in_batch += (vgetq_lane_f32(acc_32x4, 0) + vgetq_lane_f32(acc_32x4, 1) +
                           vgetq_lane_f32(acc_32x4, 2) + vgetq_lane_f32(acc_32x4, 3));
      for (int c = postamble_start; c < m_cols; c++)
      {
        *result_in_batch += matrix_row[c] * vector_in_batch[c];
      }
      matrix_row += m_cols;
      result_in_batch += result_stride;
    }
  }
}

inline void NeonMatrixBatchVectorMultiplyAccumulate(const int8_t *__restrict__ matrix,
                                                    const int m_rows, const int m_cols,
                                                    const int8_t *__restrict__ vectors,
                                                    const float *scaling_factors, int n_batch,
                                                    int32_t *scratch, float *__restrict__ result,
                                                    int result_stride, ruy::Context *ruy_context)
{
  if (m_rows % 4 == 0 && result_stride == 1)
  {
    const int32_t *bias = static_cast<const int32_t *>(nullptr);
    NeonCpuBackendGemm(vectors, bias, matrix, n_batch, m_cols, m_rows,
                       /*output_zp =*/0, scratch, ruy_context);

    // Multiply by float scaling factors and write to result
    const int total_size = n_batch * m_rows;
    int i = 0;
    for (; i <= total_size - 8; i += 8, result += 8 * result_stride)
    {
      const float batch_scaling_factor0 = scaling_factors[i / m_rows];
      const float batch_scaling_factor1 = scaling_factors[(i + 4) / m_rows];
      const float32x4_t scaling_factor0 = vdupq_n_f32(batch_scaling_factor0);
      const float32x4_t scaling_factor1 = vdupq_n_f32(batch_scaling_factor1);
      const int32x4_t scratch_val0 = vld1q_s32(scratch + i);
      const int32x4_t scratch_val1 = vld1q_s32(scratch + i + 4);
      const float32x4_t float_val0 = vcvtq_f32_s32(scratch_val0);
      const float32x4_t float_val1 = vcvtq_f32_s32(scratch_val1);
      const float32x4_t result0 = vmlaq_f32(vld1q_f32(result), float_val0, scaling_factor0);
      const float32x4_t result1 =
        vmlaq_f32(vld1q_f32(result + 4 * result_stride), float_val1, scaling_factor1);
      vst1q_f32(result, result0);
      vst1q_f32(result + 4 * result_stride, result1);
    }
    scratch += i;
    for (; i < total_size; i++, result += result_stride)
    {
      const float batch_scaling_factor = scaling_factors[i / m_rows];
      int32_t x = *(scratch++);
      *result += x * batch_scaling_factor;
    }
    return;
  }
  NeonMatrixBatchVectorMultiplyAccumulate(matrix, m_rows, m_cols, vectors, scaling_factors, n_batch,
                                          result, result_stride);
}

} // namespace cker
} // namespace nnfw

#endif // USE_NEON

#endif // __NNFW_CKER_NEON_TENSOR_UTILS_H__
