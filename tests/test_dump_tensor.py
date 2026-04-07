"""Tests for dump_tensor."""

import unittest

import paddle
from fxy_debug import dump_tensor


class TestDumpTensor(unittest.TestCase):

    def test_basic(self):
        x = paddle.randn([2, 3])
        y = paddle.zeros([4, 5], dtype="float16")
        z = paddle.ones([1], dtype="int64")
        dump_tensor(x, y, z, title="basic test")

    def test_mixed_types(self):
        x = paddle.randn([2, 3])
        batch_size = 32
        dump_tensor(x, batch_size, title="mixed types")


if __name__ == "__main__":
    unittest.main()
