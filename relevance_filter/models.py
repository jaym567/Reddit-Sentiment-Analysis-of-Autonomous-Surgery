"""
Core data models for the Reddit Relevance Filter.

This module defines the data structures used throughout the relevance filtering
pipeline, including relevance states, context objects, and configuration.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class RelevanceState(Enum):
    """
    Enumeration of relevance states for Reddit content.
    
    States:
        RELEVANT_STRONG: Explicit autonomous surgery content with strong keywords
                        and semantic match
        RELEVANT_INHERITED: Relevant due to parent context inheritance
        RELEVANT_WEAK: Tangentially related but still analytical
        IRRELEVANT: Off-topic, joke-only, meme, or conversational drift
    """
    RELEVANT_STRONG = "RELEVANT_STRONG"
    RELEVANT_INHERITED = "RELEVANT_INHERITED"
    RELEVANT_WEAK = "RELEVANT_WEAK"
    IRRELEVANT = "IRRELEVANT"


@dataclass
class ParentContext:
    """
    Context passed from parent to child during tree traversal.
    
    This object carries information about a parent comment/post that is needed
    when evaluating child comments for context inheritance.
    
    Attributes:
        id: Unique identifier of the parent item
        text: Text content of the parent item
        relevance_state: Relevance state assigned to the parent
        relevance_score: Confidence score for the parent's relevance
        depth: Depth of the parent in the comment tree (0 for posts)
        post_id: ID of the root post
    """
    id: str
    text: str
    relevance_state: RelevanceState
    relevance_score: float
    depth: int
    post_id: str


@dataclass
class FilteredItem:
    """
    Output format for filtered content.
    
    This represents a single item (post or comment) that has been evaluated
    and deemed relevant by the filter.
    
    Attributes:
        id: Unique identifier of the item
        type: Type of item ("post" or "comment")
        text: Text content of the item
        parent_id: ID of the parent item (None for posts)
        post_id: ID of the root post
        depth: Depth in the comment tree (0 for posts)
        relevance_score: Confidence score for relevance (0.0 to 1.0)
        relevance_reason: Explanation of why item was deemed relevant
                         ("keyword", "semantic", "inherited", "concatenated")
        relevance_state: Relevance state assigned to the item
        created_utc: Original timestamp from Reddit
    """
    id: str
    type: str
    text: str
    parent_id: Optional[str]
    post_id: str
    depth: int
    relevance_score: float
    relevance_reason: str
    relevance_state: str
    created_utc: Optional[float] = None

    def to_dict(self):
        """Convert FilteredItem to dictionary format."""
        return {
            'id': self.id,
            'type': self.type,
            'text': self.text,
            'parent_id': self.parent_id,
            'post_id': self.post_id,
            'depth': self.depth,
            'relevance_score': self.relevance_score,
            'relevance_reason': self.relevance_reason,
            'relevance_state': self.relevance_state,
            'created_utc': self.created_utc,
        }


@dataclass
class FilterConfig:
    """
    Configuration for the relevance filter.
    
    This object contains all configurable parameters for the filtering pipeline.
    
    Attributes:
        semantic_model_type: Type of semantic model ("embedding" or "llm")
        embedding_model_name: Name of the sentence-transformers model to use
        llm_model_name: Name of the LLM model (if using LLM-based classification)
        similarity_threshold: Minimum similarity score for relevance (0.0 to 1.0)
        concatenation_word_threshold: Maximum words in child for concatenation
        batch_size: Number of items to process in a batch
        enable_pruning: Whether to enable subtree pruning
        log_level: Logging level ("DEBUG", "INFO", "WARNING", "ERROR")
    """
    semantic_model_type: str = "embedding"
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    llm_model_name: Optional[str] = None
    similarity_threshold: float = 0.4
    concatenation_word_threshold: int = 50
    batch_size: int = 32
    enable_pruning: bool = True
    drift_threshold: int = 3
    log_level: str = "INFO"
