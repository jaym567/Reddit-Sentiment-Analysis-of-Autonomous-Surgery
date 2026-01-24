"""
Concatenation decider for parent-child text combination.

This module determines when to concatenate parent and child text for
contextual classification of short replies with implicit references.
"""

import logging
import re
from typing import Dict, List

from .models import ParentContext, RelevanceState

logger = logging.getLogger(__name__)


class ConcatenationDecider:
    """
    Decider for parent-child text concatenation.
    
    This class determines when a child comment should be evaluated with its
    parent's text for proper context interpretation.
    """
    
    def __init__(self, word_threshold: int = 50):
        """
        Initialize the ConcatenationDecider.
        
        Args:
            word_threshold: Maximum word count for concatenation eligibility
        """
        self.word_threshold = word_threshold
        
        # Pronouns that indicate reference to parent context
        self.pronouns = ["this", "that", "it", "they", "these", "those"]
        
        # Patterns for evaluative statements
        self.evaluative_patterns = [
            r"\b(agree|disagree|correct|wrong|right|exactly|precisely|true|false)\b",
            r"\b(makes sense|good point|fair point|valid|awesome|cool|great|interesting)\b",
            r"\b(good|bad|better|worse|best|worst|solid)\s+(point|idea|argument|development|improvement)\b"
        ]
        
        # Indicators that a new topic is being introduced
        self.new_topic_indicators = [
            "speaking of", "by the way", "off topic",
            "unrelated", "different topic", "changing subject"
        ]
    
    def should_concatenate(
        self,
        parent_context: ParentContext,
        child_text: str
    ) -> bool:
        """
        Determine if child should be evaluated with parent context.
        
        All conditions must be met:
        1. Parent is RELEVANT_STRONG or RELEVANT_INHERITED
        2. Child uses pronouns
        3. Child is under word_threshold words
        4. Child makes evaluative statements
        5. Child does not introduce new unrelated topic
        
        Args:
            parent_context: Context from parent comment/post
            child_text: Text of the child comment
            
        Returns:
            True if concatenation should be used, False otherwise
        """
        # Condition 1: Parent relevance
        if parent_context.relevance_state not in [
            RelevanceState.RELEVANT_STRONG,
            RelevanceState.RELEVANT_INHERITED
        ]:
            logger.debug(
                f"Concatenation rejected: parent state is {parent_context.relevance_state}"
            )
            return False
        
        # Condition 2: Child uses pronouns
        child_lower = str(child_text).lower()
        has_pronouns = any(
            re.search(r'\b' + pronoun + r'\b', child_lower)
            for pronoun in self.pronouns
        )
        if not has_pronouns:
            logger.debug("Concatenation rejected: no pronouns found in child")
            return False
        
        # Condition 3: Child is short
        word_count = len(child_text.split())
        if word_count >= self.word_threshold:
            logger.debug(
                f"Concatenation rejected: child has {word_count} words "
                f"(threshold: {self.word_threshold})"
            )
            return False
        
        # Condition 4: Child makes evaluative statements
        has_evaluative = any(
            re.search(pattern, child_lower, re.IGNORECASE)
            for pattern in self.evaluative_patterns
        )
        if not has_evaluative:
            logger.debug("Concatenation rejected: no evaluative statements found")
            return False
            
        # Condition 5: Child does not introduce new topic (Requirement 4.4)
        introduces_new_topic = any(
            indicator in child_lower
            for indicator in self.new_topic_indicators
        )
        if introduces_new_topic:
            logger.debug("Concatenation rejected: new topic introduced")
            return False
            
        # Condition 6: Questions are generally not evaluative replies for inheritance
        if child_text.strip().endswith('?'):
            logger.debug("Concatenation rejected: child is a question")
            return False
        
        logger.debug(
            f"Concatenation approved: all conditions met "
            f"(words={word_count}, parent_state={parent_context.relevance_state})"
        )
        return True
    
    def concatenate(self, parent_text: str, child_text: str) -> str:
        """
        Format concatenated text for classification.
        
        Args:
            parent_text: Text of the parent comment/post
            child_text: Text of the child comment
            
        Returns:
            Formatted concatenated text
        """
        return f"PARENT CONTEXT:\n{parent_text}\n\nCHILD COMMENT:\n{child_text}\n"
