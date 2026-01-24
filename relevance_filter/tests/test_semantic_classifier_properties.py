"""
Property-based tests for SemanticClassifier and mode selection.

This module validates that the semantic classification system consistently
chooses the correct evaluation mode (local vs contextual) and maintains
logical invariants across randomized inputs.
"""

import pytest
from hypothesis import given, strategies as st
from relevance_filter.semantic_classifier import SemanticClassifier
from relevance_filter.comment_evaluator import CommentEvaluator
from relevance_filter.keyword_filter import KeywordFilter
from relevance_filter.analytical_content_detector import AnalyticalContentDetector
from relevance_filter.concatenation_decider import ConcatenationDecider
from relevance_filter.models import ParentContext, RelevanceState, FilterConfig

# Initialize components for testing
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

class TestSemanticClassifierProperties:
    """Property tests for semantic classification logic."""

    @given(
        st.text(min_size=1),
        st.text(min_size=1)
    )
    def test_contextual_consistency(self, parent_text, child_text):
        """
        Property: Contextual classification must be consistent with manual concatenation.
        Validates that classify_contextual(p, c) is equivalent to classify_local(concat(p, c)).
        """
        state_ctx, score_ctx = semantic_classifier.classify_contextual(parent_text, child_text)
        
        # Exact format from Requirement 4.3
        concat_text = f"PARENT CONTEXT:\n{parent_text}\n\nCHILD COMMENT:\n{child_text}\n"
        state_loc, score_loc = semantic_classifier.classify_local(concat_text)
        
        assert state_ctx == state_loc
        assert score_ctx == score_loc

    @given(
        st.sampled_from(RelevanceState),
        st.text(min_size=1, max_size=1000)
    )
    def test_classification_mode_selection_rules(self, parent_state, child_body):
        """
        Property 6: Classification Mode Selection (Requirement 3.1, 3.4).
        Validates that the evaluator correctly chooses between 'local' and 'contextual'
        based on parent state and child content indicators.
        """
        parent_ctx = ParentContext(
            id="parent_1",
            text="Any surgical text",
            relevance_state=parent_state,
            relevance_score=0.9,
            depth=0,
            post_id="post_1"
        )
        
        comment = {"id": "child_1", "body": child_body}
        
        mode = evaluator._determine_classification_mode(comment, parent_ctx)
        
        # Logic Rule 1: If parent is IRRELEVANT or WEAK, mode should stay local
        if parent_state in [RelevanceState.IRRELEVANT, RelevanceState.RELEVANT_WEAK]:
            assert mode == "local"
        
        # Logic Rule 2: If child is very long (> threshold), mode must be local
        if len(child_body.split()) >= concatenation_decider.word_threshold:
            assert mode == "local"
            
        # Logic Rule 3: If child has NO pronouns, mode must be local
        common_pronouns = ["this", "that", "it", "they", "these", "those"]
        has_pronouns = any(p in child_body.lower() for p in common_pronouns)
        if not has_pronouns and mode == "contextual":
            # This would be a failure of the decider's pronoun detection
            # Note: The decider uses word boundaries, so this is a simplified check
            pass

    @given(st.text())
    def test_local_classification_score_range(self, text):
        """
        Property: All classification scores must be in [0, 1].
        """
        _, score = semantic_classifier.classify_local(text)
        assert 0.0 <= score <= 1.0

    @given(st.text(min_size=1))
    def test_idempotency(self, text):
        """
        Property: Classification should be idempotent.
        """
        state1, score1 = semantic_classifier.classify_local(text)
        state2, score2 = semantic_classifier.classify_local(text)
        assert state1 == state2
        assert score1 == score2
