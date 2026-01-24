"""
Property-based tests for KeywordFilter component.

These tests use Hypothesis to verify universal properties across all inputs.
Each property test validates specific correctness properties from the design document.
"""

import pytest
from hypothesis import given, strategies as st
from relevance_filter.keyword_filter import KeywordFilter


class TestKeywordFilterProperties:
    """Property-based tests for KeywordFilter."""
    
    @given(
        keyword=st.sampled_from([
            "autonomous surgery",
            "robotic surgeon",
            "surgical robot",
            "da vinci",
            "laparoscopic robot",
            "surgical automation",
            "ai surgery",
            "robot-assisted surgery",
            "robotic surgery",
            "cholecystectomy",
            "prostatectomy",
            "hysterectomy",
            "cardiac surgery",
            "minimally invasive",
            "operating room ai",
            "surgical ai",
            "autonomous operation",
            "robotic precision",
            "surgical robotics"
        ]),
        case_transform=st.sampled_from([
            lambda s: s.lower(),
            lambda s: s.upper(),
            lambda s: s.title(),
            lambda s: ''.join(c.upper() if i % 2 == 0 else c.lower() 
                            for i, c in enumerate(s))
        ])
    )
    def test_property_4_keyword_filter_case_insensitivity(self, keyword, case_transform):
        """
        Property 4: Keyword Filter Case Insensitivity
        
        **Validates: Requirements 9.4**
        
        For any text containing autonomous surgery keywords, the keyword filter 
        should match regardless of capitalization (lowercase, uppercase, mixed case).
        """
        kf = KeywordFilter()
        
        # Apply case transformation to the keyword
        transformed_keyword = case_transform(keyword)
        
        # The filter should match regardless of case
        assert kf.matches(transformed_keyword), \
            f"Keyword filter failed to match '{transformed_keyword}' " \
            f"(original: '{keyword}')"
    
    @given(
        keyword=st.sampled_from([
            "autonomous surgery",
            "robotic surgeon",
            "surgical robot",
            "da vinci",
            "laparoscopic robot",
            "surgical automation",
            "ai surgery",
            "robot-assisted surgery",
            "robotic surgery",
            "cholecystectomy",
            "prostatectomy",
            "hysterectomy",
            "cardiac surgery",
            "minimally invasive",
            "operating room ai",
            "surgical ai",
            "autonomous operation",
            "robotic precision",
            "surgical robotics"
        ]),
        prefix=st.text(alphabet=st.characters(blacklist_categories=('Cs',)), min_size=0, max_size=50),
        suffix=st.text(alphabet=st.characters(blacklist_categories=('Cs',)), min_size=0, max_size=50),
        case_transform=st.sampled_from([
            lambda s: s.lower(),
            lambda s: s.upper(),
            lambda s: s.title(),
            lambda s: ''.join(c.upper() if i % 2 == 0 else c.lower() 
                            for i, c in enumerate(s))
        ])
    )
    def test_property_4_case_insensitivity_in_context(
        self, keyword, prefix, suffix, case_transform
    ):
        """
        Property 4: Keyword Filter Case Insensitivity (in context)
        
        **Validates: Requirements 9.4**
        
        For any text containing autonomous surgery keywords embedded in other text,
        the keyword filter should match regardless of capitalization.
        """
        kf = KeywordFilter()
        
        # Apply case transformation to the keyword
        transformed_keyword = case_transform(keyword)
        
        # Create text with keyword embedded in context
        text = f"{prefix} {transformed_keyword} {suffix}"
        
        # The filter should match regardless of case
        assert kf.matches(text), \
            f"Keyword filter failed to match '{transformed_keyword}' in context: '{text}'"
        
        # The matched keywords should include the original keyword (case-insensitive)
        matched = kf.get_matched_keywords(text)
        assert len(matched) > 0, \
            f"get_matched_keywords returned empty list for text: '{text}'"
        
        # At least one matched keyword should be the one we're testing
        # (comparing in lowercase for case-insensitive check)
        matched_lower = [m.lower() for m in matched]
        assert keyword.lower() in matched_lower, \
            f"Expected keyword '{keyword}' not found in matched keywords: {matched}"
