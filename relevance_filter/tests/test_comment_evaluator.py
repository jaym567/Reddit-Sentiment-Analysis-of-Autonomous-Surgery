"""
Unit tests for CommentEvaluator scenarios.

This module tests specific, high-priority scenarios for comment 
relevance evaluation, including context inheritance, recovery, 
and pruning.
"""

import pytest
from relevance_filter.comment_evaluator import CommentEvaluator
from relevance_filter.keyword_filter import KeywordFilter
from relevance_filter.semantic_classifier import SemanticClassifier
from relevance_filter.analytical_content_detector import AnalyticalContentDetector
from relevance_filter.concatenation_decider import ConcatenationDecider
from relevance_filter.models import ParentContext, RelevanceState

@pytest.fixture
def evaluator():
    """Create a CommentEvaluator with all components."""
    keyword_filter = KeywordFilter()
    semantic_classifier = SemanticClassifier(model_type="embedding")
    analytical_detector = AnalyticalContentDetector()
    concatenation_decider = ConcatenationDecider(word_threshold=50)
    
    return CommentEvaluator(
        keyword_filter=keyword_filter,
        semantic_classifier=semantic_classifier,
        analytical_detector=analytical_detector,
        concatenation_decider=concatenation_decider
    )

def test_evaluate_strong_keyword_recovery(evaluator):
    """Test that strong keywords recover relevance even from irrelevant parents."""
    parent_ctx = ParentContext(
        id="p1", text="I love cooking pasta.", 
        relevance_state=RelevanceState.IRRELEVANT, relevance_score=0.1,
        depth=0, post_id="post1"
    )
    comment = {"id": "c1", "body": "Actually, speaking of precision, have you seen the new autonomous surgery robots?"}
    
    result = evaluator.evaluate(comment, parent_ctx, 1)
    
    assert result['relevance_state'] == RelevanceState.RELEVANT_STRONG.value
    assert result['relevance_reason'] == "keyword"

def test_evaluate_context_inheritance_pronoun(evaluator):
    """Test that short replies with pronouns inherit relevance."""
    parent_ctx = ParentContext(
        id="p1", text="The da Vinci robot is very expensive but precise.", 
        relevance_state=RelevanceState.RELEVANT_STRONG, relevance_score=0.9,
        depth=0, post_id="post1"
    )
    comment = {"id": "c1", "body": "That is true, but it saves time in the long run."}
    
    result = evaluator.evaluate(comment, parent_ctx, 1)
    
    # "That" refers to the robot context
    assert result['relevance_state'] == RelevanceState.RELEVANT_INHERITED.value
    assert result['relevance_reason'] == "inherited"

def test_evaluate_drift_to_joke_pruning(evaluator):
    """Test that drifting to non-analytical jokes results in pruning."""
    parent_ctx = ParentContext(
        id="p1", text="Robotic surgery is the future.", 
        relevance_state=RelevanceState.RELEVANT_STRONG, relevance_score=0.9,
        depth=0, post_id="post1"
    )
    # A joke with no analytical content
    comment = {"id": "c1", "body": "I for one welcome our new robot overlords! beep boop"}
    
    result = evaluator.evaluate(comment, parent_ctx, 1)
    
    # Should be IRRELEVANT and should_prune=True because it's non-analytical humor
    assert result['relevance_state'] == RelevanceState.IRRELEVANT.value
    assert result['should_prune'] is True

def test_evaluate_analytical_humor_no_pruning(evaluator):
    """Test that analytical humor is kept and not pruned."""
    parent_ctx = ParentContext(
        id="p1", text="Autonomous surgery systems are being tested.", 
        relevance_state=RelevanceState.RELEVANT_STRONG, relevance_score=0.9,
        depth=0, post_id="post1"
    )
    # Analytical humor
    comment = {"id": "c1", "body": "Haha, as long as it doesn't try to install Windows updates during my appendectomy, the automation seems like a solid improvement."}
    
    result = evaluator.evaluate(comment, parent_ctx, 1)
    
    # Should be relevant because it discusses appendectomy/automation analytically
    assert result['relevance_state'] in [RelevanceState.RELEVANT_STRONG.value, RelevanceState.RELEVANT_WEAK.value, RelevanceState.RELEVANT_INHERITED.value]
    assert result['should_prune'] is False

def test_evaluate_irrelevant_tangent(evaluator):
    """Test that unrelated tangents are marked irrelevant."""
    parent_ctx = ParentContext(
        id="p1", text="Autonomous surgery systems are being tested.", 
        relevance_state=RelevanceState.RELEVANT_STRONG, relevance_score=0.9,
        depth=0, post_id="post1"
    )
    comment = {"id": "c1", "body": "Did you see the latest football game last night? Crazy ending."}
    
    result = evaluator.evaluate(comment, parent_ctx, 1)
    
    assert result['relevance_state'] == RelevanceState.IRRELEVANT.value
    assert result['should_prune'] is True
