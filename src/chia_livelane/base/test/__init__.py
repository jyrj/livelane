"""Tests for :mod:`chia_livelane.base`.

This package marker is required, not decorative: without it pytest imports test
modules by bare basename, and `base/test/test_delay.py` collides with
`tests/unit/test_delay.py`, which aborts collection for the WHOLE suite.
"""
