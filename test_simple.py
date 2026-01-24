"""Simple test to verify pytest works."""

import pytest
from hypothesis import given, strategies as st

class TestSimple:
    def test_basic(self):
        assert True
    
    @given(st.integers())
    def test_hypothesis_basic(self, x):
        assert isinstance(x, int)