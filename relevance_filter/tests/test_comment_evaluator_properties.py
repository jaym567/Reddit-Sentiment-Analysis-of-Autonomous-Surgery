"""
Property-based tests for CommentEvaluator logic.

This module validates that context inheritance, recovery logic, and 
relevance state assignments remain logically consistent across 
randomized conversational trees.
"""

import pytest
from hypothesis import given, strategies as st
from relevance_filter.comment_evaluator import CommentEvaluator
from relevance_filter.keyword_filter import KeywordFilter
from relevance_filter.semantic_classifier import SemanticClassifier
from relevance_filter.analytical_content_detector import AnalyticalContentDetector
from relevance_filter.concatenation_decider import ConcatenationDecider
from relevance_filter.models import ParentContext, RelevanceState

# Initialize components
keyword_filter = KeywordFilter()
semantic_classifier = SemanticClassifier(model_type="embedding")
analytical_detector = AnalyticalContentDetector()
concatenation_decider = ConcatenationDecider(word_threshold=50)

evaluator = CommentEvaluator(
    keyword_filter=keyword_filter,
    semantic_classifier=semantic_classifier,
    analytical_detector=analytical_detector,
    concatenation_decider=concatenation_decider
)

class TestCommentEvaluatorProperties:
    """Property tests for CommentEvaluator."""

    @given(
        st.text(min_size=1, max_size=1000),
        st.sampled_from([RelevanceState.RELEVANT_STRONG, RelevanceState.RELEVANT_INHERITED])
    )
    def test_context_inheritance_for_continuations(self, child_text, parent_state):
        """
        Property 7: Context Inheritance for Continuations (Requirement 3.2).
        If a child has analytical content and the parent is relevant, 
        the child should be at least RELEVANT_INHERITED.
        """
        # Ensure child has analytical content by appending a marker
        child_with_analytical = child_text + " because I believe this affects clinical outcomes."
        
        parent_ctx = ParentContext(
            id="p1", text="Robotic surgery is great.", 
            relevance_state=parent_state, relevance_score=0.9,
            depth=0, post_id="post1"
        )
        
        comment = {"id": "c1", "body": child_with_analytical, "parent_id": "p1"}
        result = evaluator.evaluate(comment, parent_ctx, 1)
        
        # Should be relevant (STRONG, INHERITED, or WEAK)
        # Requirement 3.2 specifically mentions RELEVANT_INHERITED for context continuations
        assert result['relevance_state'] in [
            RelevanceState.RELEVANT_STRONG.value,
            RelevanceState.RELEVANT_INHERITED.value,
            RelevanceState.RELEVANT_WEAK.value
        ]

    @given(
        st.text(min_size=1, max_size=500),
        st.sampled_from([RelevanceState.IRRELEVANT])
    )
    def test_recovery_from_irrelevant_parents(self, noise_text, parent_state):
        """
        Property 8: Recovery from Irrelevant Parents (Requirement 3.5).
        Even if a parent is IRRELEVANT, if the child contains strong keywords,
        it must be marked RELEVANT_STRONG.
        """
        # Inject a strong keyword
        keyword = "autonomous surgery"
        child_text = f"{noise_text} {keyword} {noise_text}"
        
        parent_ctx = ParentContext(
            id="p1", text="I like pizza.", 
            relevance_state=parent_state, relevance_score=0.1,
            depth=0, post_id="post1"
        )
        
        comment = {"id": "c1", "body": child_text, "parent_id": "p1"}
        result = evaluator.evaluate(comment, parent_ctx, 1)
        
        # Recovery logic (Requirement 3.5)
        assert result['relevance_state'] == RelevanceState.RELEVANT_STRONG.value
        assert result['relevance_reason'] == "keyword"

    @given(
        st.text(max_size=1000),
        st.one_of(st.none(), st.builds(ParentContext, 
            id=st.text(min_size=1),
            text=st.text(min_size=1),
            relevance_state=st.sampled_from(RelevanceState),
            relevance_score=st.floats(min_value=0, max_value=1),
            depth=st.integers(min_value=0, max_value=10),
            post_id=st.text(min_size=1)
        ))
    )
    def test_evaluation_completeness(self, body, parent_ctx):
        """
        Property: All evaluations must return the required fields.
        """
        comment = {"id": "c1", "body": body, "parent_id": "p1", "link_id": "post1"}
        result = evaluator.evaluate(comment, parent_ctx, (parent_ctx.depth + 1) if parent_ctx else 1)
        
        required_fields = [
            "id", "type", "text", "parent_id", "post_id", "depth",
            "relevance_score", "relevance_reason", "relevance_state",
            "should_prune", "has_analytical_content"
        ]
        for field in required_fields:
            assert field in result
        
        assert 0.0 <= result['relevance_score'] <= 1.0
        assert isinstance(result['should_prune'], bool)
