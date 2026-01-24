"""
Unit tests for error handling and edge cases.

This module validates that the relevance filter components gracefully 
handle malformed data, missing fields, and unexpected inputs.
"""

import pytest
from relevance_filter.relevance_filter import RelevanceFilter
from relevance_filter.keyword_filter import KeywordFilter
from relevance_filter.semantic_classifier import SemanticClassifier
from relevance_filter.models import FilterConfig

@pytest.fixture
def filter_system():
    config = FilterConfig()
    kf = KeywordFilter()
    sc = SemanticClassifier()
    return RelevanceFilter(kf, sc, config)

def test_missing_post_fields(filter_system):
    """Test handling of posts with missing id or text."""
    # Post missing everything
    bad_post = {}
    results = filter_system.filter_single_post(bad_post)
    
    # Should not crash, and should return empty list (since no title/text matched)
    assert isinstance(results, list)
    assert len(results) == 0

def test_none_values_in_text(filter_system):
    """Test handling of None values in text fields."""
    post = {
        'id': 'p1',
        'title': None,
        'selftext': None,
        'comments': [
            {'id': 'c1', 'body': None, 'replies': []}
        ]
    }
    
    # Process
    results = filter_system.filter_single_post(post)
    
    # Should handle Nones as empty strings without crashing
    assert isinstance(results, list)
    assert len(results) == 0

def test_malformed_comment_structure(filter_system):
    """Test handling of malformed comment structure (e.g. non-list replies)."""
    post = {
        'id': 'p1',
        'title': 'autonomous surgery',
        'selftext': 'technical',
        'comments': [
            {
                'id': 'c1', 
                'body': 'technical comment about autonomous surgery',
                'replies': "not a list" # Should handle this gracefully
            }
        ]
    }
    
    results = filter_system.filter_single_post(post)
    
    # Should process c1 but skip its replies safely
    ids = [item['id'] for item in results]
    assert 'p1' in ids
    assert 'c1' in ids

def test_non_string_body(filter_system):
    """Test handling of non-string body content."""
    post = {
        'id': 'p1',
        'title': 'robotic surgery',
        'comments': [
            {'id': 'c1', 'body': 12345, 'replies': []} # Number instead of string
        ]
    }
    
    results = filter_system.filter_single_post(post)
    
    # Should at least not crash and likely mark as irrelevant or treat as string
    assert isinstance(results, list)

def test_empty_input_batch(filter_system):
    """Test handling of empty batch."""
    assert filter_system.filter_posts([]) == []

def test_very_long_text_processing(filter_system):
    """Test handling of extremely long text content."""
    long_text = "autonomous surgery " * 10000 
    post = {'id': 'p1', 'title': 'Long post', 'selftext': long_text}
    
    # This might take a while but should not crash
    results = filter_system.filter_single_post(post)
    assert len(results) > 0
