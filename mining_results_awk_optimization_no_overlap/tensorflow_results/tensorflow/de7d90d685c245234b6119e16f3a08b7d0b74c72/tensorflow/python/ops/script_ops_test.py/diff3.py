# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for script operations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.eager import def_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import script_ops
from tensorflow.python.ops.script_ops import numpy_function
from tensorflow.python.platform import test


class NumpyFunctionTest(test.TestCase):

  @test_util.run_in_graph_and_eager_modes
  def test_numpy_arguments(self):

    def plus(a, b):
      return a + b

    actual_result = script_ops.numpy_function(plus, [1, 2], dtypes.int32)
    expect_result = constant_op.constant(3, dtypes.int32)
    self.assertAllEqual(actual_result, expect_result)

<<<<<<< /home/ze/miningframework/mining_results_awk_optimization_no_overlap/tensorflow_results/tensorflow/de7d90d685c245234b6119e16f3a08b7d0b74c72/tensorflow/python/ops/script_ops_test.py/left.py
  def test_stateless_flag(self):
    call_count = 0

    def plus(a, b):
      nonlocal call_count
      call_count += 1
      return a + b

    @def_function.function
    def numpy_func_stateless(a, b):
      return numpy_function(plus, [a, b], dtypes.int32, stateful=False)

    @def_function.function
    def func_stateless(a, b):
      sum1 = numpy_func_stateless(a, b)
      sum2 = numpy_func_stateless(a, b)
      return sum1 + sum2

    _ = func_stateless(
      constant_op.constant(1),
      constant_op.constant(2),
    )

    self.assertEqual(call_count, 1)  # the second call should be eliminated
    call_count = 0  # reset

    @def_function.function
    def numpy_func_stateful(a, b):
      return numpy_function(plus, [a, b], dtypes.int32, stateful=True)

    @def_function.function
    def func_stateful(a, b):
      sum1 = numpy_func_stateful(a, b)
      sum2 = numpy_func_stateful(a, b)
      return sum1 + sum2

    _ = func_stateful(
      constant_op.constant(1),
      constant_op.constant(2),
    )
    self.assertEqual(call_count, 2)  # 2 as it is stateful, func was both times executed

=======
  def test_stateless_flag(self):
    call_count = 0

    def plus(a, b):
      nonlocal call_count
      call_count += 1
      return a + b

    @def_function.function
    def tensor_double_plus_stateless(a, b):
      sum1 = numpy_function(plus, [a, b], dtypes.int32, stateful=False)
      sum2 = numpy_function(plus, [a, b], dtypes.int32, stateful=False)
      return sum1 + sum2

    # different argument
    _ = tensor_double_plus_stateless(  # executing empty
      constant_op.constant(1),
      constant_op.constant(2),
    )
    self.assertEqual(call_count, 1)  # +1 as only the first encounter was executed

    @def_function.function
    def tensor_double_plus_stateful(a, b):
      sum1 = numpy_function(plus, [a, b], dtypes.int32, stateful=True)
      sum2 = numpy_function(plus, [a, b], dtypes.int32, stateful=True)
      return sum1 + sum2

    _ = tensor_double_plus_stateful( # executing empty
      constant_op.constant(3),
      constant_op.constant(4),
                          )
    self.assertEqual(call_count, 3)  # +2 as it is stateful, func was both times executed

>>>>>>> /home/ze/miningframework/mining_results_awk_optimization_no_overlap/tensorflow_results/tensorflow/de7d90d685c245234b6119e16f3a08b7d0b74c72/tensorflow/python/ops/script_ops_test.py/right.py

if __name__ == "__main__":
  test.main()