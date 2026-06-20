from __future__ import annotations

from spectraxgk.operators import hermite_streaming
from spectraxgk.operators.linear import hermite_streaming as package_streaming
from spectraxgk.operators.linear.streaming import hermite_streaming as streaming_impl


def test_linear_operator_package_reexports_streaming_kernel() -> None:
    assert hermite_streaming is package_streaming
    assert package_streaming is streaming_impl
