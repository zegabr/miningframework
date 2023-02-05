# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Functional tests for 3d pooling operations."""

import numpy as np

from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import gradient_checker_v2
from tensorflow.python.ops import nn_ops
import tensorflow.python.ops.nn_grad  # pylint: disable=unused-import
from tensorflow.python.platform import test


def GetTestConfigs():
  """Get all the valid tests configs to run.

  Returns:
    all the valid test configs as tuples of data_format and use_gpu.
  """
  test_configs = [("NDHWC", False), ("NDHWC", True)]
  if test.is_gpu_available(cuda_only=True):
    # "NCHW" format is currently supported exclusively on CUDA GPUs.
    test_configs += [("NCDHW", True)]
  return test_configs


# TODO(mjanusz): Add microbenchmarks for 3d pooling.
@test_util.with_eager_op_as_function
class PoolingTest(test.TestCase):

  def _DtypesToTest(self, use_gpu):
    if use_gpu:
      return [dtypes.float32]
    else:
      # It is important that float32 comes before float16 here, as we will be
      # using its gradients as reference for bf16 gradients.
      return [dtypes.float32, dtypes.bfloat16]

  def _VerifyOneTest(self, pool_func, input_sizes, window, strides, padding,
                     data_format, data_type, expected, use_gpu):
    """Verifies the output values of the pooling function.

    Args:
      pool_func: Function to be called: co.MaxPool, co.AvgPool.
      input_sizes: Input tensor dimensions.
      window: Tuple of kernel dims: planes, rows, cols.
      strides: Tuple of strides for dims: planes, rows, cols.
      padding: Padding type.
      data_format: The data format we use to run the pooling operation.
      data_type: The data type to use to run the pooling operation.
      expected: An array containing the expected operation outputs.
      use_gpu: Whether to run ops on GPU.
    """
    total_size = 1
    for s in input_sizes:
      total_size *= s
    # Initializes the input tensor with array containing incrementing
    # numbers from 1.
    x = [f * 1.0 for f in range(1, total_size + 1)]
<<<<<<< ./tensorflow/8fb105dbc43dcaf5eda2707cf4be3f96a4daf183/tensorflow/python/kernel_tests/nn_ops/pooling_ops_3d_test.py/left.py
    for dtype in self._DtypesToTest(use_gpu):
      with self.cached_session
=======
    if data_type == dtypes.bfloat16:
      x = [f * 0.1 for f in x]
      expected = [f * 0.1 for f in expected]
    with self.cached_session
>>>>>>> ./tensorflow/8fb105dbc43dcaf5eda2707cf4be3f96a4daf183/tensorflow/python/kernel_tests/nn_ops/pooling_ops_3d_test.py/right.py
(use_gpu=use_gpu):
        t = constant_op.constant(x, shape=input_sizes
<<<<<<< ./tensorflow/8fb105dbc43dcaf5eda2707cf4be3f96a4daf183/tensorflow/python/kernel_tests/nn_ops/pooling_ops_3d_test.py/left.py
, dtype=dtype
=======
, dtype=data_type
>>>>>>> ./tensorflow/8fb105dbc43dcaf5eda2707cf4be3f96a4daf183/tensorflow/python/kernel_tests/nn_ops/pooling_ops_3d_test.py/right.py
)
        ksize = [1] + list(window) + [1]
        strides_a = [1] + list(strides) + [1]
        if data_format == "NCDHW":
          t = test_util.NHWCToNCHW(t)
          ksize = test_util.NHWCToNCHW(ksize)
          strides_a = test_util.NHWCToNCHW(strides_a)
        t = pool_func(
            t,
            ksize=ksize,
            strides=strides_a,
            padding=padding,
            data_format=data_format)
        if data_format == "NCDHW":
          t = test_util.NCHWToNHWC(t)
        vals = self.evaluate(t)
        # Verifies values.
        actual = vals.flatten()
<<<<<<< ./tensorflow/8fb105dbc43dcaf5eda2707cf4be3f96a4daf183/tensorflow/python/kernel_tests/nn_ops/pooling_ops_3d_test.py/left.py
        self.assertAllCloseAccordingToType
=======
    rtol = atol = 1e-6
    if data_type == dtypes.bfloat16:
      rtol = atol = 2e-2
    self.assertAllClose
>>>>>>> ./tensorflow/8fb105dbc43dcaf5eda2707cf4be3f96a4daf183/tensorflow/python/kernel_tests/nn_ops/pooling_ops_3d_test.py/right.py
(expected, actual, rtol=rtol, atol=atol)

  def _VerifyValues(self, pool_func, input_sizes, window, strides,
                    padding, expected):
    for data_format, use_gpu in GetTestConfigs():
      self._VerifyOneTest(pool_func, input_sizes, window, strides, padding,
                          data_format, dtypes.float32, expected, use_gpu)
      if use_gpu and test_util.is_gpu_available(cuda_only=True):
        self._VerifyOneTest(pool_func, input_sizes, window, strides, padding,
                            data_format, dtypes.bfloat16, expected, use_gpu)

  def testAvgPool3dValidPadding(self):
    expected_output = [20.5, 21.5, 22.5]
    self._VerifyValues(
        nn_ops.avg_pool3d,
        input_sizes=[1, 3, 3, 3, 3],
        window=(2, 2, 2),
        strides=(2, 2, 2),
        padding="VALID",
        expected=expected_output)

  def testAvgPool3dSamePadding(self):
    expected_output = [20.5, 21.5, 22.5, 26.5, 27.5, 28.5]
    self._VerifyValues(
        nn_ops.avg_pool3d,
        input_sizes=[1, 2, 2, 4, 3],
        window=(2, 2, 2),
        strides=(2, 2, 2),
        padding="SAME",
        expected=expected_output)

  def testAvgPool3dSamePaddingDifferentStrides(self):
    expected_output = [1.5, 4.5, 7.5, 17.5, 20.5, 23.5, 33.5, 36.5, 39.5]
    self._VerifyValues(
        nn_ops.avg_pool3d,
        input_sizes=[1, 5, 8, 1, 1],
        window=(1, 2, 3),
        strides=(2, 3, 1),
        padding="SAME",
        expected=expected_output)

  def testAvgPool3dGrad(self):
    with self.assertRaises(
        (errors.ResourceExhaustedError, errors.InvalidArgumentError)):
      use_gpu = test.is_gpu_available(cuda_only=True)
      for dtype in self._DtypesToTest(use_gpu):
        with self.cached_session():
          orig_input_shape = constant_op.constant(
              1879048192, shape=[5], dtype=dtypes.int32)
          grad = constant_op.constant(
              1, shape=[1, 3, 2, 4, 2], dtype=dtype)
          t = gen_nn_ops.AvgPool3DGrad(
              orig_input_shape=orig_input_shape,
              grad=grad,
              ksize=[1, 1, 1, 1, 1],
              strides=[1, 1, 1, 1, 1],
              padding="SAME",
              data_format="NDHWC")
          self.evaluate(t)

  def testAvgPool3dGradEmptyInput(self):
    for data_format, use_gpu in GetTestConfigs():
      with self.cached_session(use_gpu=use_gpu):
        orig_input_shape = constant_op.constant([5, 6, 7, 0, 8],
                                                dtype=dtypes.int32)
        grad = constant_op.constant(
            1, shape=[5, 6, 7, 0, 8], dtype=dtypes.float32)
        t = gen_nn_ops.AvgPool3DGrad(
            orig_input_shape=orig_input_shape,
            grad=grad,
            ksize=[1, 1, 1, 1, 1],
            strides=[1, 1, 1, 1, 1],
            padding="SAME",
            data_format=data_format)
        self.evaluate(t)

  def testMaxPool3dValidPadding(self):
    expected_output = [40.0, 41.0, 42.0]
    self._VerifyValues(
        nn_ops.max_pool3d,
        input_sizes=[1, 3, 3, 3, 3],
        window=(2, 2, 2),
        strides=(2, 2, 2),
        padding="VALID",
        expected=expected_output)

  def testMaxPool3dSamePadding(self):
    expected_output = [31., 32., 33., 34., 35., 36.]
    self._VerifyValues(
        nn_ops.max_pool3d,
        input_sizes=[1, 2, 2, 3, 3],
        window=(2, 2, 2),
        strides=(2, 2, 2),
        padding="SAME",
        expected=expected_output)

  def testMaxPool3dSamePaddingDifferentStrides(self):
    expected_output = [2., 5., 8., 18., 21., 24., 34., 37., 40.]
    self._VerifyValues(
        nn_ops.max_pool3d,
        input_sizes=[1, 5, 8, 1, 1],
        window=(1, 2, 3),
        strides=(2, 3, 1),
        padding="SAME",
        expected=expected_output)

    # Test pooling on a larger input, with different stride and kernel
    # size for the 'z' dimension.

    # Simulate max pooling in numpy to get the expected output.
    input_data = np.arange(1, 5 * 27 * 27 * 64 + 1).reshape((5, 27, 27, 64))
    input_data = np.pad(input_data, [[0, 0], [0, 1], [0, 1], [0, 0]],
                        mode="constant")
    expected_output = input_data[:, 1::2, 1::2, :]
    expected_output[:, -1, :, :] = input_data[:, -2, 1::2, :]
    expected_output[:, :, -1, :] = input_data[:, 1::2, -2, :]
    expected_output[:, -1, -1, :] = input_data[:, -2, -2, :]

    self._VerifyValues(
        nn_ops.max_pool3d,
        input_sizes=[1, 5, 27, 27, 64],
        window=(1, 2, 2),
        strides=(1, 2, 2),
        padding="SAME",
        expected=expected_output.flatten())

  def testKernelSmallerThanStride(self):
    self._VerifyValues(
        nn_ops.max_pool3d,
        input_sizes=[1, 3, 3, 3, 1],
        window=[1, 1, 1],
        strides=[2, 2, 2],
        padding="SAME",
        expected=[1, 3, 7, 9, 19, 21, 25, 27])

    self._VerifyValues(
        nn_ops.max_pool3d,
        input_sizes=[1, 7, 7, 7, 1],
        window=[2, 2, 2],
        strides=[3, 3, 3],
        padding="VALID",
        expected=[58, 61, 79, 82, 205, 208, 226, 229])

    self._VerifyValues(
        nn_ops.avg_pool3d,
        input_sizes=[1, 3, 3, 3, 1],
        window=[1, 1, 1],
        strides=[2, 2, 2],
        padding="SAME",
        expected=[1, 3, 7, 9, 19, 21, 25, 27])

    self._VerifyValues(
        nn_ops.avg_pool3d,
        input_sizes=[1, 7, 7, 7, 1],
        window=[2, 2, 2],
        strides=[3, 3, 3],
        padding="VALID",
        expected=[29.5, 32.5, 50.5, 53.5, 176.5, 179.5, 197.5, 200.5])

  def testMaxPool3DEmptyTensorOutputShape(self):
    """Verifies the output shape of the max pooling function when tensor is empty.

    Args: none
    """
    input_sizes = [0, 112, 112, 112, 64]

    input_data = 1.
    input_tensor = constant_op.constant(
        input_data, shape=input_sizes, name="input")
    max_pool_3d = nn_ops.max_pool3d(
        input_tensor,
        ksize=[2, 2, 2],
        strides=[2, 2, 2],
        padding="VALID",
        data_format="NDHWC",
        name="max_pool_3d")
    values = self.evaluate(max_pool_3d)
    self.assertEqual(values.shape, (0, 56, 56, 56, 64))

  # TODO(penporn): Determine if we will allow input_sizes[3] < ksize[3].
  def DISABLED_testAvgPool3dEmptyOutTensor(self):
    input_sizes = [30, 19, 4, 19, 17]
    input_data = 1.0
    input_tensor = constant_op.constant(
        input_data, shape=input_sizes, name="input")
    avg_pool_3d = nn_ops.avg_pool3d(
        input_tensor,
        ksize=(1, 13, 3, 20, 1),
        strides=(1, 14, 4, 1, 1),
        padding="VALID",
        data_format="NDHWC",
        name="avg_pool_3d")
    values = self.evaluate(avg_pool_3d)
    self.assertEqual(values.shape, (30, 1, 1, 0, 17))

  def _getJacobians(self,
                    pool_func,
                    input_sizes,
                    output_sizes,
                    window,
                    strides,
                    padding,
                    data_format,
                    use_gpu,
                    dtype=np.float32
<<<<<<< ./tensorflow/8fb105dbc43dcaf5eda2707cf4be3f96a4daf183/tensorflow/python/kernel_tests/nn_ops/pooling_ops_3d_test.py/left.py
      total_size *= s
    # float32 should always run before bfloat16 so as to compare results
    for dtype in self._DtypesToTest(use_gpu):
      # Initializes the input tensor with array containing incrementing
      # numbers from 1.
      x = np.arange(1, total_size + 1
=======
>>>>>>> ./tensorflow/8fb105dbc43dcaf5eda2707cf4be3f96a4daf183/tensorflow/python/kernel_tests/nn_ops/pooling_ops_3d_test.py/right.py
):
      with self.cached_session(use_gpu=use_gpu):
<<<<<<< ./tensorflow/8fb105dbc43dcaf5eda2707cf4be3f96a4daf183/tensorflow/python/kernel_tests/nn_ops/pooling_ops_3d_test.py/left.py
        input_tensor = constant_op.constant
=======
      x = np.arange
>>>>>>> ./tensorflow/8fb105dbc43dcaf5eda2707cf4be3f96a4daf183/tensorflow/python/kernel_tests/nn_ops/pooling_ops_3d_test.py/right.py
(
<<<<<<< ./tensorflow/8fb105dbc43dcaf5eda2707cf4be3f96a4daf183/tensorflow/python/kernel_tests/nn_ops/pooling_ops_3d_test.py/left.py
x, shape=input_sizes, name="input",
                                            dtype=dtype
=======
np.prod(input_sizes
>>>>>>> ./tensorflow/8fb105dbc43dcaf5eda2707cf4be3f96a4daf183/tensorflow/python/kernel_tests/nn_ops/pooling_ops_3d_test.py/right.py
)
<<<<<<< ./tensorflow/8fb105dbc43dcaf5eda2707cf4be3f96a4daf183/tensorflow/python/kernel_tests/nn_ops/pooling_ops_3d_test.py/left.py
        if dtype != dtypes.bfloat16:
          err_g_margin = 1e-3
          err_gg_margin = 1.5e-2
        else:
          err_g_margin = 1.5e-2
          err_gg_margin = 1.5e-2

        if pool_func == nn_ops.avg_pool3d:
          func_name = "avg_pool3d"
          x_init_value = None
        else:
          x_init_value = np.asfarray
=======
).reshape
>>>>>>> ./tensorflow/8fb105dbc43dcaf5eda2707cf4be3f96a4daf183/tensorflow/python/kernel_tests/nn_ops/pooling_ops_3d_test.py/right.py
(input_sizes).astype(dtype)
      input_tensor = constant_op.constant
<<<<<<< ./tensorflow/8fb105dbc43dcaf5eda2707cf4be3f96a4daf183/tensorflow/python/kernel_tests/nn_ops/pooling_ops_3d_test.py/left.py
                                    dtype=np.float32).reshape
=======
>>>>>>> ./tensorflow/8fb105dbc43dcaf5eda2707cf4be3f96a4daf183/tensorflow/python/kernel_tests/nn_ops/pooling_ops_3d_test.py/right.py
(x, shape=input_sizes)
<<<<<<< ./tensorflow/8fb105dbc43dcaf5eda2707cf4be3f96a4daf183/tensorflow/python/kernel_tests/nn_ops/pooling_ops_3d_test.py/left.py
          func_name = "max_pool3d"

        ksize = [1
=======
      ksize = [1
>>>>>>> ./tensorflow/8fb105dbc43dcaf5eda2707cf4be3f96a4daf183/tensorflow/python/kernel_tests/nn_ops/pooling_ops_3d_test.py/right.py
, window[0], window[1], window[2], 1]
        strides_a = [1, strides[0], strides[1], strides[2], 1]
<<<<<<< ./tensorflow/8fb105dbc43dcaf5eda2707cf4be3f96a4daf183/tensorflow/python/kernel_tests/nn_ops/pooling_ops_3d_test.py/left.py
        t = input_tensor

        if data_format == "NCDHW":
=======
      if data_format == "NCDHW":
>>>>>>> ./tensorflow/8fb105dbc43dcaf5eda2707cf4be3f96a4daf183/tensorflow/python/kernel_tests/nn_ops/pooling_ops_3d_test.py/right.py
          ksize = test_util.NHWCToNCHW(ksize)
          strides_a = test_util.NHWCToNCHW(strides_a)
<<<<<<< ./tensorflow/8fb105dbc43dcaf5eda2707cf4be3f96a4daf183/tensorflow/python/kernel_tests/nn_ops/pooling_ops_3d_test.py/left.py
          t = test_util.NHWCToNCHW
=======
        input_tensor = test_util.NHWCToNCHW
>>>>>>> ./tensorflow/8fb105dbc43dcaf5eda2707cf4be3f96a4daf183/tensorflow/python/kernel_tests/nn_ops/pooling_ops_3d_test.py/right.py
(input_tensor)
          output_sizes = test_util.NHWCToNCHW(output_sizes)

<<<<<<< ./tensorflow/8fb105dbc43dcaf5eda2707cf4be3f96a4daf183/tensorflow/python/kernel_tests/nn_ops/pooling_ops_3d_test.py/left.py
        t_p = pool_func
=======
      def func(in_tensor):
        return pool_func(
            in_tensor,
            ksize=ksize,
            strides=strides,
            padding=padding,
            data_format=data_format)

      input_jacob_a
>>>>>>> ./tensorflow/8fb105dbc43dcaf5eda2707cf4be3f96a4daf183/tensorflow/python/kernel_tests/nn_ops/pooling_ops_3d_test.py/right.py
, input_jacob_n = gradient_checker_v2.compute_gradient(
<<<<<<< ./tensorflow/8fb105dbc43dcaf5eda2707cf4be3f96a4daf183/tensorflow/python/kernel_tests/nn_ops/pooling_ops_3d_test.py/left.py
            t
=======
          func
>>>>>>> ./tensorflow/8fb105dbc43dcaf5eda2707cf4be3f96a4daf183/tensorflow/python/kernel_tests/nn_ops/pooling_ops_3d_test.py/right.py
, [input_tensor])

      def pool_grad_function(upstream_gradients):
        with backprop.GradientTape() as tape:
<<<<<<< ./tensorflow/8fb105dbc43dcaf5eda2707cf4be3f96a4daf183/tensorflow/python/kernel_tests/nn_ops/pooling_ops_3d_test.py/left.py
            ksize=ksize
=======
          tape.watch(input_tensor)
          pool_output = pool_func(
              input_tensor
>>>>>>> ./tensorflow/8fb105dbc43dcaf5eda2707cf4be3f96a4daf183/tensorflow/python/kernel_tests/nn_ops/pooling_ops_3d_test.py/right.py
,
<<<<<<< ./tensorflow/8fb105dbc43dcaf5eda2707cf4be3f96a4daf183/tensorflow/python/kernel_tests/nn_ops/pooling_ops_3d_test.py/left.py
            strides=strides_a
=======
              ksize=ksize
>>>>>>> ./tensorflow/8fb105dbc43dcaf5eda2707cf4be3f96a4daf183/tensorflow/python/kernel_tests/nn_ops/pooling_ops_3d_test.py/right.py
,
<<<<<<< ./tensorflow/8fb105dbc43dcaf5eda2707cf4be3f96a4daf183/tensorflow/python/kernel_tests/nn_ops/pooling_ops_3d_test.py/left.py
            padding=padding
=======
              strides=strides
>>>>>>> ./tensorflow/8fb105dbc43dcaf5eda2707cf4be3f96a4daf183/tensorflow/python/kernel_tests/nn_ops/pooling_ops_3d_test.py/right.py
,
<<<<<<< ./tensorflow/8fb105dbc43dcaf5eda2707cf4be3f96a4daf183/tensorflow/python/kernel_tests/nn_ops/pooling_ops_3d_test.py/left.py
            data_format=data_format
=======
              padding=padding
>>>>>>> ./tensorflow/8fb105dbc43dcaf5eda2707cf4be3f96a4daf183/tensorflow/python/kernel_tests/nn_ops/pooling_ops_3d_test.py/right.py
,
<<<<<<< ./tensorflow/8fb105dbc43dcaf5eda2707cf4be3f96a4daf183/tensorflow/python/kernel_tests/nn_ops/pooling_ops_3d_test.py/left.py
            name=func_name
=======
              data_format=data_format
>>>>>>> ./tensorflow/8fb105dbc43dcaf5eda2707cf4be3f96a4daf183/tensorflow/python/kernel_tests/nn_ops/pooling_ops_3d_test.py/right.py
)
<<<<<<< ./tensorflow/8fb105dbc43dcaf5eda2707cf4be3f96a4daf183/tensorflow/python/kernel_tests/nn_ops/pooling_ops_3d_test.py/left.py
=======
          gradient_injector_output = pool_output * upstream_gradients
          return tape.gradient(gradient_injector_output, input_tensor)
>>>>>>> ./tensorflow/8fb105dbc43dcaf5eda2707cf4be3f96a4daf183/tensorflow/python/kernel_tests/nn_ops/pooling_ops_3d_test.py/right.py

<<<<<<< ./tensorflow/8fb105dbc43dcaf5eda2707cf4be3f96a4daf183/tensorflow/python/kernel_tests/nn_ops/pooling_ops_3d_test.py/left.py
        err_g = gradient_checker.compute_gradient_error
=======
      upstream_tensor = constant_op.constant(
          2 * np.random.rand
>>>>>>> ./tensorflow/8fb105dbc43dcaf5eda2707cf4be3f96a4daf183/tensorflow/python/kernel_tests/nn_ops/pooling_ops_3d_test.py/right.py
(*output_sizes) - 1, dtype=dtype)
<<<<<<< ./tensorflow/8fb105dbc43dcaf5eda2707cf4be3f96a4daf183/tensorflow/python/kernel_tests/nn_ops/pooling_ops_3d_test.py/left.py
            input_tensor
=======
      grad_jacob_a
>>>>>>> ./tensorflow/8fb105dbc43dcaf5eda2707cf4be3f96a4daf183/tensorflow/python/kernel_tests/nn_ops/pooling_ops_3d_test.py/right.py
, grad_jacob_n = gradient_checker_v2.compute_gradient(
<<<<<<< ./tensorflow/8fb105dbc43dcaf5eda2707cf4be3f96a4daf183/tensorflow/python/kernel_tests/nn_ops/pooling_ops_3d_test.py/left.py
            input_sizes
=======
          pool_grad_function
>>>>>>> ./tensorflow/8fb105dbc43dcaf5eda2707cf4be3f96a4daf183/tensorflow/python/kernel_tests/nn_ops/pooling_ops_3d_test.py/right.py
, [upstream_tensor])

      return 
<<<<<<< ./tensorflow/8fb105dbc43dcaf5eda2707cf4be3f96a4daf183/tensorflow/python/kernel_tests/nn_ops/pooling_ops_3d_test.py/left.py
            t_p
=======
(
>>>>>>> ./tensorflow/8fb105dbc43dcaf5eda2707cf4be3f96a4daf183/tensorflow/python/kernel_tests/nn_ops/pooling_ops_3d_test.py/right.py
(input_jacob_a, grad_jacob_a), (input_jacob_n, grad_jacob_n))
<<<<<<< ./tensorflow/8fb105dbc43dcaf5eda2707cf4be3f96a4daf183/tensorflow/python/kernel_tests/nn_ops/pooling_ops_3d_test.py/left.py
            output_sizes
=======

  def _ConstructAndTestGradientForConfig
>>>>>>> ./tensorflow/8fb105dbc43dcaf5eda2707cf4be3f96a4daf183/tensorflow/python/kernel_tests/nn_ops/pooling_ops_3d_test.py/right.py
(self, pool_func, input_sizes,
<<<<<<< ./tensorflow/8fb105dbc43dcaf5eda2707cf4be3f96a4daf183/tensorflow/python/kernel_tests/nn_ops/pooling_ops_3d_test.py/left.py
            x_init_value=x_init_value
=======
                                         output_sizes
>>>>>>> ./tensorflow/8fb105dbc43dcaf5eda2707cf4be3f96a4daf183/tensorflow/python/kernel_tests/nn_ops/pooling_ops_3d_test.py/right.py
, window, strides, padding,
<<<<<<< ./tensorflow/8fb105dbc43dcaf5eda2707cf4be3f96a4daf183/tensorflow/python/kernel_tests/nn_ops/pooling_ops_3d_test.py/left.py
            delta=1e-2
=======
                                         data_format
>>>>>>> ./tensorflow/8fb105dbc43dcaf5eda2707cf4be3f96a4daf183/tensorflow/python/kernel_tests/nn_ops/pooling_ops_3d_test.py/right.py
, data_type, use_gpu):
    """Verifies the gradients of a pooling function.

    Args:
      pool_func: Function to be called, co.MaxPool, co.AvgPool,
        or the Lua version.
      input_sizes: Input tensor dimensions.
      output_sizes: Output tensor dimensions.
      window: Tuple of kernel dims: planes, rows, cols.
      strides: Tuple of strides for dims: planes, rows, cols.
<<<<<<< ./tensorflow/8fb105dbc43dcaf5eda2707cf4be3f96a4daf183/tensorflow/python/kernel_tests/nn_ops/pooling_ops_3d_test.py/left.py
        t_g = gradients_impl.gradients
=======
      padding: Padding type.
      data_format: Data format string.
      data_type: The data type to use to run the pooling operation.
      use_gpu: Whether to run on GPU.
>>>>>>> ./tensorflow/8fb105dbc43dcaf5eda2707cf4be3f96a4daf183/tensorflow/python/kernel_tests/nn_ops/pooling_ops_3d_test.py/right.py
    """
    jacob_a, jacob_n = self._getJacobians(t_p**2, input_tensor)[0]
        jacob_t, jacob_n = gradient_checker.compute_gradient(
<<<<<<< ./tensorflow/8fb105dbc43dcaf5eda2707cf4be3f96a4daf183/tensorflow/python/kernel_tests/nn_ops/pooling_ops_3d_test.py/left.py
            input_tensor
=======
        pool_func
>>>>>>> ./tensorflow/8fb105dbc43dcaf5eda2707cf4be3f96a4daf183/tensorflow/python/kernel_tests/nn_ops/pooling_ops_3d_test.py/right.py
,
<<<<<<< ./tensorflow/8fb105dbc43dcaf5eda2707cf4be3f96a4daf183/tensorflow/python/kernel_tests/nn_ops/pooling_ops_3d_test.py/left.py
            input_sizes
=======
        input_sizes,
        output_sizes,
        window,
        strides
>>>>>>> ./tensorflow/8fb105dbc43dcaf5eda2707cf4be3f96a4daf183/tensorflow/python/kernel_tests/nn_ops/pooling_ops_3d_test.py/right.py
,
<<<<<<< ./tensorflow/8fb105dbc43dcaf5eda2707cf4be3f96a4daf183/tensorflow/python/kernel_tests/nn_ops/pooling_ops_3d_test.py/left.py
            t_g
=======
        padding
>>>>>>> ./tensorflow/8fb105dbc43dcaf5eda2707cf4be3f96a4daf183/tensorflow/python/kernel_tests/nn_ops/pooling_ops_3d_test.py/right.py
,
<<<<<<< ./tensorflow/8fb105dbc43dcaf5eda2707cf4be3f96a4daf183/tensorflow/python/kernel_tests/nn_ops/pooling_ops_3d_test.py/left.py
            input_sizes
=======
        data_format
>>>>>>> ./tensorflow/8fb105dbc43dcaf5eda2707cf4be3f96a4daf183/tensorflow/python/kernel_tests/nn_ops/pooling_ops_3d_test.py/right.py
,
<<<<<<< ./tensorflow/8fb105dbc43dcaf5eda2707cf4be3f96a4daf183/tensorflow/python/kernel_tests/nn_ops/pooling_ops_3d_test.py/left.py
            x_init_value=x_init_value
=======
        use_gpu
>>>>>>> ./tensorflow/8fb105dbc43dcaf5eda2707cf4be3f96a4daf183/tensorflow/python/kernel_tests/nn_ops/pooling_ops_3d_test.py/right.py
,
<<<<<<< ./tensorflow/8fb105dbc43dcaf5eda2707cf4be3f96a4daf183/tensorflow/python/kernel_tests/nn_ops/pooling_ops_3d_test.py/left.py
            delta=1e-2
=======
        dtype=data_type.as_numpy_dtype
>>>>>>> ./tensorflow/8fb105dbc43dcaf5eda2707cf4be3f96a4daf183/tensorflow/python/kernel_tests/nn_ops/pooling_ops_3d_test.py/right.py
)
<<<<<<< ./tensorflow/8fb105dbc43dcaf5eda2707cf4be3f96a4daf183/tensorflow/python/kernel_tests/nn_ops/pooling_ops_3d_test.py/left.py
        if dtype == dtypes.float32:
          fp32_t_g_val = self.evaluate(t_g)
          fp32_jacob_t = jacob_t
          err_gg = np.fabs(jacob_t - jacob_n).max()
          print
=======

    if data_type == dtypes.bfloat16:
>>>>>>> ./tensorflow/8fb105dbc43dcaf5eda2707cf4be3f96a4daf183/tensorflow/python/kernel_tests/nn_ops/pooling_ops_3d_test.py/right.py
      # Compare bf16 analytical gradients to fp32 numerical gradients.
      _, jacob_n = self._getJacobians(
          pool_func,
          input_sizes,
          output_sizes,
          window,
          strides,
          padding,
          data_format,
<<<<<<< ./tensorflow/8fb105dbc43dcaf5eda2707cf4be3f96a4daf183/tensorflow/python/kernel_tests/nn_ops/pooling_ops_3d_test.py/left.py
          self.assertLess
=======
          use_gpu
>>>>>>> ./tensorflow/8fb105dbc43dcaf5eda2707cf4be3f96a4daf183/tensorflow/python/kernel_tests/nn_ops/pooling_ops_3d_test.py/right.py
,
          dtype=np.float32)

    input_jacob_a
<<<<<<< ./tensorflow/8fb105dbc43dcaf5eda2707cf4be3f96a4daf183/tensorflow/python/kernel_tests/nn_ops/pooling_ops_3d_test.py/left.py
          print
=======
, grad_jacob_a = jacob_a
    input_jacob_n, grad_jacob_n = jacob_n
    self.assertAllClose
>>>>>>> ./tensorflow/8fb105dbc43dcaf5eda2707cf4be3f96a4daf183/tensorflow/python/kernel_tests/nn_ops/pooling_ops_3d_test.py/right.py
(input_jacob_a, input_jacob_n, rtol=1e-3, atol=1e-3)
<<<<<<< ./tensorflow/8fb105dbc43dcaf5eda2707cf4be3f96a4daf183/tensorflow/python/kernel_tests/nn_ops/pooling_ops_3d_test.py/left.py
          self.assertLess(err_gg, err_gg_margin)
        else:
          # Compare bf16 gradients to fp32 gradients, since bf16 numerical
          # gradients are too imprecise.
          t_g_val = self.evaluate(t_g)
          self.assertAllClose(t_g_val, fp32_t_g_val, err_g_margin, err_g_margin)
          err_gg = np.fabs(jacob_t - fp32_jacob_t).max()
          self.assertLess
=======
    self.assertAllClose
>>>>>>> ./tensorflow/8fb105dbc43dcaf5eda2707cf4be3f96a4daf183/tensorflow/python/kernel_tests/nn_ops/pooling_ops_3d_test.py/right.py
(grad_jacob_a, grad_jacob_n, rtol=1e-3, atol=1e-3)

  def _ConstructAndTestGradient(self,
                                pool_func,
                                **kwargs):
    """Runs _ConstructAndTestGradientForConfig for all tests configurations."""

    for data_format, use_gpu in GetTestConfigs():
      self._ConstructAndTestGradientForConfig(
          pool_func,
          data_format=data_format,
          data_type=dtypes.float32,
          use_gpu=use_gpu,
          **kwargs)
      if use_gpu and test_util.is_gpu_available(cuda_only=True):
        self._ConstructAndTestGradientForConfig(
            pool_func,
            data_format=data_format,
            data_type=dtypes.bfloat16,
            use_gpu=use_gpu,
            **kwargs)

  @test_util.run_deprecated_v1
  def testMaxPoolGradValidPadding1_1_3d(self):
    self._ConstructAndTestGradient(
        nn_ops.max_pool3d,
        input_sizes=[1, 3, 3, 3, 1],
        output_sizes=[1, 3, 3, 3, 1],
        window=(1, 1, 1),
        strides=(1, 1, 1),
        padding="VALID")

  @test_util.run_deprecated_v1
  def testMaxPoolGradValidPadding2_1_6_3d(self):
    self._ConstructAndTestGradient(
        nn_ops.max_pool3d,
        input_sizes=[1, 2, 3, 4, 2],
        output_sizes=[1, 1, 2, 3, 2],
        window=(2, 2, 2),
        strides=(1, 1, 1),
        padding="VALID")

  @test_util.run_deprecated_v1
  def testMaxPoolGradValidPadding2_1_7_3d(self):
    self._ConstructAndTestGradient(
        nn_ops.max_pool3d,
        input_sizes=[1, 3, 2, 7, 1],
        output_sizes=[1, 2, 1, 6, 1],
        window=(2, 2, 2),
        strides=(1, 1, 1),
        padding="VALID")

  @test_util.run_deprecated_v1
  def testMaxPoolGradValidPadding1_2_3d(self):
    self._ConstructAndTestGradient(
        nn_ops.max_pool3d,
        input_sizes=[1, 3, 3, 3, 1],
        output_sizes=[1, 2, 2, 2, 1],
        window=(1, 1, 1),
        strides=(2, 2, 2),
        padding="VALID")

  @test_util.run_deprecated_v1
  def testMaxPoolGradValidPadding2_2_3d(self):
    self._ConstructAndTestGradient(
        nn_ops.max_pool3d,
        input_sizes=[2, 2, 2, 2, 1],
        output_sizes=[2, 1, 1, 1, 1],
        window=(2, 2, 2),
        strides=(2, 2, 2),
        padding="VALID")

  @test_util.run_deprecated_v1
  def testMaxPoolGradSamePadding1_1_3d(self):
    self._ConstructAndTestGradient(
        nn_ops.max_pool3d,
        input_sizes=[1, 3, 2, 4, 1],
        output_sizes=[1, 3, 2, 4, 1],
        window=(1, 1, 1),
        strides=(1, 1, 1),
        padding="SAME")

  @test_util.run_deprecated_v1
  def testMaxPoolGradSamePadding1_2_3d(self):
    self._ConstructAndTestGradient(
        nn_ops.max_pool3d,
        input_sizes=[1, 3, 2, 4, 1],
        output_sizes=[1, 2, 1, 2, 1],
        window=(1, 1, 1),
        strides=(2, 2, 2),
        padding="SAME")

  @test_util.run_deprecated_v1
  def testMaxPoolGradSamePadding2_1_3d(self):
    self._ConstructAndTestGradient(
        nn_ops.max_pool3d,
        input_sizes=[1, 3, 2, 4, 1],
        output_sizes=[1, 3, 2, 4, 1],
        window=(2, 2, 2),
        strides=(1, 1, 1),
        padding="SAME")

  @test_util.run_deprecated_v1
  def testMaxPoolGradSamePadding2_2_3d(self):
    self._ConstructAndTestGradient(
        nn_ops.max_pool3d,
        input_sizes=[1, 5, 2, 4, 2],
        output_sizes=[1, 3, 1, 2, 2],
        window=(2, 2, 2),
        strides=(2, 2, 2),
        padding="SAME")

  @test_util.run_deprecated_v1
  def testMaxPoolGradSamePadding3_1_3d(self):
    self._ConstructAndTestGradient(
        nn_ops.max_pool3d,
        input_sizes=[1, 3, 4, 2, 1],
        output_sizes=[1, 3, 4, 2, 1],
        window=(3, 3, 3),
        strides=(1, 1, 1),
        padding="SAME")

  @test_util.run_deprecated_v1
  def testAvgPoolGradValidPadding1_1_3d(self):
    self._ConstructAndTestGradient(
        nn_ops.avg_pool3d,
        input_sizes=[1, 3, 3, 3, 1],
        output_sizes=[1, 3, 3, 3, 1],
        window=(1, 1, 1),
        strides=(1, 1, 1),
        padding="VALID")

  @test_util.run_deprecated_v1
  def testAvgPoolGradValidPadding1_2_3d(self):
    self._ConstructAndTestGradient(
        nn_ops.avg_pool3d,
        input_sizes=[1, 3, 3, 3, 1],
        output_sizes=[1, 2, 2, 2, 1],
        window=(1, 1, 1),
        strides=(2, 2, 2),
        padding="VALID")

  @test_util.run_deprecated_v1
  def testAvgPoolGradValidPadding2_1_3d(self):
    self._ConstructAndTestGradient(
        nn_ops.avg_pool3d,
        input_sizes=[1, 3, 3, 3, 2],
        output_sizes=[1, 2, 2, 2, 2],
        window=(2, 2, 2),
        strides=(1, 1, 1),
        padding="VALID")

  @test_util.run_deprecated_v1
  def testAvgPoolGradValidPadding2_2_3d(self):
    self._ConstructAndTestGradient(
        nn_ops.avg_pool3d,
        input_sizes=[2, 2, 2, 2, 2],
        output_sizes=[2, 1, 1, 1, 2],
        window=(2, 2, 2),
        strides=(2, 2, 2),
        padding="VALID")

  @test_util.run_deprecated_v1
  def testAvgPoolGradSamePadding1_1_3d(self):
    self._ConstructAndTestGradient(
        nn_ops.avg_pool3d,
        input_sizes=[1, 3, 2, 4, 2],
        output_sizes=[1, 3, 2, 4, 2],
        window=(1, 1, 1),
        strides=(1, 1, 1),
        padding="SAME")

  @test_util.run_deprecated_v1
  def testAvgPoolGradSamePadding1_2_3d(self):
    self._ConstructAndTestGradient(
        nn_ops.avg_pool3d,
        input_sizes=[1, 3, 2, 4, 2],
        output_sizes=[1, 2, 1, 2, 2],
        window=(1, 1, 1),
        strides=(2, 2, 2),
        padding="SAME")

  @test_util.run_deprecated_v1
  def testAvgPoolGradSamePadding2_1_3d(self):
    self._ConstructAndTestGradient(
        nn_ops.avg_pool3d,
        input_sizes=[1, 2, 2, 2, 1],
        output_sizes=[1, 2, 2, 2, 1],
        window=(2, 2, 2),
        strides=(1, 1, 1),
        padding="SAME")

  @test_util.run_deprecated_v1
  def testAvgPoolGradSamePadding2_2_3d(self):
    self._ConstructAndTestGradient(
        nn_ops.avg_pool3d,
        input_sizes=[1, 5, 2, 4, 1],
        output_sizes=[1, 3, 1, 2, 1],
        window=(2, 2, 2),
        strides=(2, 2, 2),
        padding="SAME")

  @test_util.run_deprecated_v1
  def testAvgPoolGradSamePadding3_1_3d(self):
    self._ConstructAndTestGradient(
        nn_ops.avg_pool3d,
        input_sizes=[1, 3, 6, 2, 1],
        output_sizes=[1, 3, 6, 2, 1],
        window=(3, 3, 3),
        strides=(1, 1, 1),
        padding="SAME")

  def testMaxPool3DZeroPoolSize(self):
    # Test case for GitHub issue 51936.
    for f in [nn_ops.max_pool3d, nn_ops.avg_pool3d]:
      with self.session():
        with self.assertRaises((errors.InvalidArgumentError, ValueError)):
          input_sizes = [3, 4, 10, 11, 12]

          input_data = 1.
          input_tensor = constant_op.constant(
              input_data, shape=input_sizes, name="input")
          pool_3d = f(input_tensor, ksize=[2, 2, 0], strides=1, padding="VALID")
          self.evaluate(pool_3d)

  @test_util.disable_xla("b/205634417")  # XLA does not raise these errors.
  def testMaxPoolGradEagerShapeErrors(self):
    with context.eager_mode():
      orig_in = array_ops.ones((1, 1, 1, 1, 1))

      # Test invalid orig_out shape
      orig_out = array_ops.ones((1, 1, 1, 1, 2))
      grad = array_ops.ones((1, 1, 1, 1, 1))
      with self.assertRaisesRegex(
          errors_impl.InvalidArgumentError,
          r"Expected orig_output shape to be \[1,1,1,1,1\], but got "
          r"\[1,1,1,1,2\]"):
        gen_nn_ops.max_pool3d_grad(
            orig_in, orig_out, grad, ksize=[1, 1, 1, 1, 1],
            strides=[1, 1, 1, 1, 1], padding="VALID")
      with self.assertRaisesRegex(
          errors_impl.InvalidArgumentError,
          r"Expected orig_output shape to be \[1,1,1,1,1\], but got "
          r"\[1,1,1,1,2\]"):
        gen_nn_ops.max_pool3d_grad_grad(
            orig_in, orig_out, grad, ksize=[1, 1, 1, 1, 1],
            strides=[1, 1, 1, 1, 1], padding="VALID")

      # Test invalid grad shape
      orig_out = array_ops.ones((1, 1, 1, 1, 1))
      grad = array_ops.ones((1, 1, 1, 1, 2))
      with self.assertRaisesRegex(
          errors_impl.InvalidArgumentError,
          r"Expected grad shape to be \[1,1,1,1,1\], but got \[1,1,1,1,2\]"):
        gen_nn_ops.max_pool3d_grad(
            orig_in, orig_out, grad, ksize=[1, 1, 1, 1, 1],
            strides=[1, 1, 1, 1, 1], padding="VALID")
      with self.assertRaisesRegex(
          errors_impl.InvalidArgumentError,
          r"Expected grad shape to be \[1,1,1,1,1\], but got \[1,1,1,1,2\]"):
        gen_nn_ops.max_pool3d_grad_grad(
            orig_in, orig_out, grad, ksize=[1, 1, 1, 1, 1],
            strides=[1, 1, 1, 1, 1], padding="VALID")


if __name__ == "__main__":
  test.main()
