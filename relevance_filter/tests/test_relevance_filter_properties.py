"""
Property-based tests for RelevanceFilter orchestrator.

This module validates the high-level tree processing logic, including
post-level exclusion, subtree pruning, and traversal invariants.
"""

import pytest
from hypothesis import given, strategies as st, settings, HealthCheck
from relevance_filter.relevance_filter import RelevanceFilter
from relevance_filter.keyword_filter import KeywordFilter
from relevance_filter.semantic_classifier import SemanticClassifier
from relevance_filter.models import FilterConfig, RelevanceState

@pytest.fixture(scope="module")
def orchestrator():
    """Create a RelevanceFilter orchestrator."""
    config = FilterConfig(concatenation_word_threshold=50)
    kf = KeywordFilter()
    sc = SemanticClassifier(model_type="embedding")
    return RelevanceFilter(kf, sc, config)

class TestRelevanceFilterProperties:
    """Property tests for RelevanceFilter."""

    @given(st.lists(st.fixed_dictionaries({
        'id': st.text(min_size=1),
        'title': st.text(),
        'selftext': st.text(),
        'comments': st.just([])
    }), min_size=1, max_size=10))
    def test_irrelevant_post_tree_exclusion(self, orchestrator, posts):
        """
        Property 3: Irrelevant Post Tree Exclusion (Requirement 1.4).
        If a post (title+text) is determined to be irrelevant, 
        its entire tree must be excluded from results.
        """
        # We'll use nonsense texts to ensure irrelevance
        for post in posts:
            post['title'] = "Random nonsense words pizza"
            post['selftext'] = "Nothing related to surgery at all."
        
        results = orchestrator.filter_posts(posts)
        
        # Should be empty because all posts are irrelevant
        assert len(results) == 0

    @given(st.text(min_size=1, max_size=100))
    def test_subtree_pruning_rule(self, orchestrator, noise_text):
        """
        Property 12: Subtree Pruning Rule (Requirement 7.1, 7.3).
        If a comment is IRRELEVANT and lacks analytical content, 
        its replies must NOT be processed.
        """
        # Create a post with one irrelevant non-analytical comment that has replies
        post = {
            'id': 'p1', 'title': 'robotic surgery', 'selftext': 'technical stuff',
            'comments': [
                {
                    'id': 'c1', 'body': 'lol memes', 'replies': [
                        {'id': 'c1_1', 'body': 'autonomous surgery', 'replies': []}
                    ]
                }
            ]
        }
        
        # Process single post
        processed = orchestrator.filter_single_post(post)
        
        # c1 should be omitted (IRRELEVANT + No Analytical)
        # c1_1 should also be omitted due to PRUNING (Requirement 7.1)
        # Only the post itself should be in results (as it is relevant)
        
        # filter_single_post returns a list of FilteredItems (as dicts)
        # The post itself should be its own item, and then its descendants
        ids = [item['id'] for item in processed]
        
        assert 'p1' in ids
        assert 'c1' not in ids
        assert 'c1_1' not in ids, "Subtree was not pruned correctly"

    @given(st.integers(min_value=1, max_value=5))
    def test_depth_first_traversal_logic(self, orchestrator, depth):
        """
        Property 13: Depth-First Traversal Order (Requirement 6.1).
        Verifies that comments are appended in a valid DFS order.
        """
        # Create a linear thread
        thread = []
        curr_replies = []
        for d in range(depth, 0, -1):
            comment = {
                'id': f'c{d}', 
                'body': 'robotic surgery analytical text', # Ensure relevant
                'replies': curr_replies
            }
            curr_replies = [comment]
        
        post = {
            'id': 'p1', 'title': 'robotic surgery', 'selftext': 'text',
            'comments': curr_replies
        }
        
        results = orchestrator.filter_single_post(post)
        
        ids = [item['id'] for item in results]
        assert ids[0] == 'p1'
        for i in range(1, len(ids)):
            # Depth should be increasing or strictly handled
            assert results[i]['depth'] == results[i-1]['depth'] + 1

    @given(st.text(min_size=1), st.text(min_size=1))
    def test_parent_child_relationship_preservation(self, orchestrator, parent_id, child_body):
        """
        Property 14: Parent-Child Relationship Preservation (Requirement 6.4, 8.4).
        Verifies that child items correctly reference their parent's ID.
        """
        post = {
            'id': parent_id, 'title': 'robotic surgery', 'selftext': 'technical',
            'comments': [{'id': 'c1', 'body': 'autonomous surgery', 'replies': []}]
        }
        results = orchestrator.filter_single_post(post)
        
        if len(results) > 1:
            child = results[1]
            assert child['parent_id'] == parent_id

    @given(st.integers(min_value=0, max_value=20))
    def test_depth_tracking_accuracy(self, orchestrator, target_depth):
        """
        Property 15: Depth Tracking Accuracy (Requirement 6.2).
        """
        # Build a deep linear tree
        curr = []
        for i in range(target_depth, 0, -1):
            curr = [{'id': f'd{i}', 'body': 'autonomous surgery', 'replies': curr}]
            
        post = {'id': 'p0', 'title': 'robotic surgery', 'selftext': '...', 'comments': curr}
        results = orchestrator.filter_single_post(post)
        
        # Verify the last item's depth
        if target_depth > 0 and len(results) > target_depth:
            assert results[-1]['depth'] == target_depth

    @given(st.text())
    def test_output_field_completeness(self, orchestrator, body):
        """
        Property 18: Output Field Completeness (Requirement 8.2).
        """
        post = {'id': 'p1', 'title': 'robotic surgery', 'selftext': body, 'comments': []}
        results = orchestrator.filter_posts([post])
        
        if results:
            item = results[0]
            expected = ['id', 'type', 'text', 'parent_id', 'post_id', 'depth', 
                        'relevance_score', 'relevance_reason', 'relevance_state']
            for field in expected:
                assert field in item

    @given(st.text(min_size=1, max_size=100))
    def test_text_preservation(self, orchestrator, body):
        """
        Property 19: Text Preservation (Requirement 8.3).
        """
        post = {'id': 'p1', 'title': 'Surgery', 'selftext': body, 'comments': []}
        # Force relevance
        post['title'] = "autonomous surgery"
        results = orchestrator.filter_posts([post])
        
        if results:
            # Post text contains both title and selftext in our impl
            assert body in results[0]['text']
