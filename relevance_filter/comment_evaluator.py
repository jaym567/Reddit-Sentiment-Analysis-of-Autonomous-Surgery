"""
Comment evaluator for individual comment assessment.

This module evaluates individual comments with full context, coordinating
keyword filtering, semantic classification, and pruning decisions.
"""

import logging
from typing import Dict, Optional

from .models import ParentContext, RelevanceState

logger = logging.getLogger(__name__)


class CommentEvaluator:
    """
    Evaluator for individual comments with full context.
    
    This class coordinates all filtering components to evaluate a single
    comment and determine its relevance state and pruning decision.
    """
    
    def __init__(
        self,
        keyword_filter,
        semantic_classifier,
        analytical_detector,
        concatenation_decider
    ):
        """
        Initialize the CommentEvaluator.
        
        Args:
            keyword_filter: KeywordFilter instance
            semantic_classifier: SemanticClassifier instance
            analytical_detector: AnalyticalContentDetector instance
            concatenation_decider: ConcatenationDecider instance
        """
        self.keyword_filter = keyword_filter
        self.semantic_classifier = semantic_classifier
        self.analytical_detector = analytical_detector
        self.concatenation_decider = concatenation_decider
    
    def evaluate(
        self,
        comment: Dict,
        parent_context: Optional[ParentContext],
        depth: int,
        subreddit: Optional[str] = None
    ) -> Dict:
        """
        Evaluate a single comment and return enriched result.
        """
        text = comment.get('body', '')
        if not text:
            # Handle possible alternate field names
            text = comment.get('text', '')
            
        # Ensure text is a string (Requirement: Robustness to malformed data)
        text = str(text) if text is not None else ""

        # Step 0: Strong Irrelevance Check (Exclusion Keywords + Noise Subs)
        if self.analytical_detector.is_strongly_irrelevant(text, subreddit):
            return {
                "id": comment.get('id'),
                "type": "comment",
                "text": text,
                "parent_id": comment.get('parent_id', parent_context.id if parent_context else None),
                "post_id": comment.get('link_id', parent_context.post_id if parent_context else None),
                "depth": depth,
                "relevance_score": 1.0,
                "relevance_reason": "exclusion_keyword",
                "relevance_state": RelevanceState.IRRELEVANT.value if hasattr(RelevanceState.IRRELEVANT, 'value') else RelevanceState.IRRELEVANT,
                "should_prune": True,
                "has_analytical_content": False,
                "has_topic_intersection": False
            }
            
        # Step 1: Decision on classification mode
        mode = self._determine_classification_mode(comment, parent_context)
        
        # Step 2: Semantic classification
        if mode == "contextual" and parent_context:
            relevance_state, score = self.semantic_classifier.classify_contextual(
                parent_context.text,
                text
            )
            reason = "concatenated"
        else:
            relevance_state, score = self.semantic_classifier.classify_local(text)
            reason = "semantic"
            
        # Step 3: Keyword Recovery / Override (Requirement 3.5 & 9.1)
        if self.keyword_filter.matches(text):
            # Explicit keywords always signal strong relevance in this domain
            relevance_state = RelevanceState.RELEVANT_STRONG
            reason = "keyword"
            score = max(score, 0.9) # High confidence for explicit target keywords
            
        # Step 3.5: Topic Intersection Recovery (User Request)
        # If it discusses BOTH medical and AI/robot, it's at least WEAK
        has_intersection = self.analytical_detector.has_topic_intersection(text, subreddit=subreddit)
        if relevance_state == RelevanceState.IRRELEVANT and has_intersection:
            relevance_state = RelevanceState.RELEVANT_WEAK
            reason = "topic_intersection"
            score = max(score, 0.5)

        # Step 4: Inheritance Logic (Requirement 3)
        has_analytical = self.analytical_detector.has_analytical_content(text)
        
        # Only apply inheritance if not already marked RELEVANT_STRONG or RELEVANT_WEAK (from intersection)
        if relevance_state == RelevanceState.IRRELEVANT \
           and parent_context \
           and parent_context.relevance_state in [RelevanceState.RELEVANT_STRONG, RelevanceState.RELEVANT_INHERITED]:
            
            # If it's a short contextual reply or has analytical content, inherit
            # User Request: If it has analytical content and isn't a new topic, default to INHERITED
            introduces_new_topic = any(
                indicator in text.lower()
                for indicator in self.concatenation_decider.new_topic_indicators
            )
            
            if (has_analytical and not introduces_new_topic) or mode == "contextual":
                relevance_state = RelevanceState.RELEVANT_INHERITED
                reason = "inherited"
                score = max(score, 0.7)
        
        # Pruning decision is now handled by drift counter in RelevanceFilter
        should_prune = False
                
        return {
            "id": comment.get('id'),
            "type": "comment",
            "text": text,
            "parent_id": comment.get('parent_id', parent_context.id if parent_context else None),
            "post_id": comment.get('link_id', parent_context.post_id if parent_context else None),
            "depth": depth,
            "relevance_score": score,
            "relevance_reason": reason,
            "relevance_state": relevance_state.value if hasattr(relevance_state, 'value') else relevance_state,
            "should_prune": should_prune,
            "has_analytical_content": has_analytical,
            "has_topic_intersection": has_intersection
        }
    
    def _determine_classification_mode(
        self,
        comment: Dict,
        parent_context: Optional[ParentContext]
    ) -> str:
        """
        Determine if we should use local or contextual classification.
        
        Args:
            comment: Comment dictionary
            parent_context: Context from parent comment/post
            
        Returns:
            Classification mode ("local" or "contextual")
        """
        if not parent_context:
            return "local"
            
        text = comment.get('body', comment.get('text', ''))
        if self.concatenation_decider.should_concatenate(parent_context, text):
            return "contextual"
            
        return "local"
