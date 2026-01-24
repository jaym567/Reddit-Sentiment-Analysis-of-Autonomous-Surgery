"""
Unit tests for core data models.

This module tests the data structures defined in models.py.
"""

import pytest
from relevance_filter.models import (
    RelevanceState,
    ParentContext,
    FilteredItem,
    FilterConfig
)


class TestRelevanceState:
    """Tests for RelevanceState enum."""
    
    def test_relevance_state_values(self):
        """Test that all relevance states have correct values."""
        assert RelevanceState.RELEVANT_STRONG.value == "RELEVANT_STRONG"
        assert RelevanceState.RELEVANT_INHERITED.value == "RELEVANT_INHERITED"
        assert RelevanceState.RELEVANT_WEAK.value == "RELEVANT_WEAK"
        assert RelevanceState.IRRELEVANT.value == "IRRELEVANT"
    
    def test_relevance_state_membership(self):
        """Test that we can check membership in RelevanceState."""
        assert RelevanceState.RELEVANT_STRONG in RelevanceState
        assert RelevanceState.RELEVANT_INHERITED in RelevanceState
        assert RelevanceState.RELEVANT_WEAK in RelevanceState
        assert RelevanceState.IRRELEVANT in RelevanceState


class TestParentContext:
    """Tests for ParentContext dataclass."""
    
    def test_parent_context_creation(self):
        """Test creating a ParentContext instance."""
        context = ParentContext(
            id="comment_123",
            text="This is a test comment",
            relevance_state=RelevanceState.RELEVANT_STRONG,
            relevance_score=0.95,
            depth=1,
            post_id="post_456"
        )
        
        assert context.id == "comment_123"
        assert context.text == "This is a test comment"
        assert context.relevance_state == RelevanceState.RELEVANT_STRONG
        assert context.relevance_score == 0.95
        assert context.depth == 1
        assert context.post_id == "post_456"
    
    def test_parent_context_with_irrelevant_state(self):
        """Test ParentContext with IRRELEVANT state."""
        context = ParentContext(
            id="comment_789",
            text="Off-topic comment",
            relevance_state=RelevanceState.IRRELEVANT,
            relevance_score=0.1,
            depth=2,
            post_id="post_456"
        )
        
        assert context.relevance_state == RelevanceState.IRRELEVANT
        assert context.relevance_score == 0.1


class TestFilteredItem:
    """Tests for FilteredItem dataclass."""
    
    def test_filtered_item_creation_post(self):
        """Test creating a FilteredItem for a post."""
        item = FilteredItem(
            id="post_123",
            type="post",
            text="Discussion about robotic surgery",
            parent_id=None,
            post_id="post_123",
            depth=0,
            relevance_score=0.92,
            relevance_reason="keyword+semantic",
            relevance_state="RELEVANT_STRONG"
        )
        
        assert item.id == "post_123"
        assert item.type == "post"
        assert item.parent_id is None
        assert item.depth == 0
        assert item.relevance_state == "RELEVANT_STRONG"
    
    def test_filtered_item_creation_comment(self):
        """Test creating a FilteredItem for a comment."""
        item = FilteredItem(
            id="comment_456",
            type="comment",
            text="I agree with this point",
            parent_id="post_123",
            post_id="post_123",
            depth=1,
            relevance_score=0.78,
            relevance_reason="inherited",
            relevance_state="RELEVANT_INHERITED"
        )
        
        assert item.id == "comment_456"
        assert item.type == "comment"
        assert item.parent_id == "post_123"
        assert item.depth == 1
        assert item.relevance_state == "RELEVANT_INHERITED"
    
    def test_filtered_item_to_dict(self):
        """Test converting FilteredItem to dictionary."""
        item = FilteredItem(
            id="comment_789",
            type="comment",
            text="Test comment",
            parent_id="comment_456",
            post_id="post_123",
            depth=2,
            relevance_score=0.85,
            relevance_reason="concatenated",
            relevance_state="RELEVANT_INHERITED"
        )
        
        item_dict = item.to_dict()
        
        assert isinstance(item_dict, dict)
        assert item_dict['id'] == "comment_789"
        assert item_dict['type'] == "comment"
        assert item_dict['text'] == "Test comment"
        assert item_dict['parent_id'] == "comment_456"
        assert item_dict['post_id'] == "post_123"
        assert item_dict['depth'] == 2
        assert item_dict['relevance_score'] == 0.85
        assert item_dict['relevance_reason'] == "concatenated"
        assert item_dict['relevance_state'] == "RELEVANT_INHERITED"


class TestFilterConfig:
    """Tests for FilterConfig dataclass."""
    
    def test_filter_config_defaults(self):
        """Test FilterConfig with default values."""
        config = FilterConfig()
        
        assert config.semantic_model_type == "embedding"
        assert config.embedding_model_name == "sentence-transformers/all-MiniLM-L6-v2"
        assert config.llm_model_name is None
        assert config.similarity_threshold == 0.7
        assert config.concatenation_word_threshold == 50
        assert config.batch_size == 32
        assert config.enable_pruning is True
        assert config.log_level == "INFO"
    
    def test_filter_config_custom_values(self):
        """Test FilterConfig with custom values."""
        config = FilterConfig(
            semantic_model_type="llm",
            llm_model_name="gpt-4",
            similarity_threshold=0.8,
            concatenation_word_threshold=40,
            batch_size=16,
            enable_pruning=False,
            log_level="DEBUG"
        )
        
        assert config.semantic_model_type == "llm"
        assert config.llm_model_name == "gpt-4"
        assert config.similarity_threshold == 0.8
        assert config.concatenation_word_threshold == 40
        assert config.batch_size == 16
        assert config.enable_pruning is False
        assert config.log_level == "DEBUG"
    
    def test_filter_config_partial_override(self):
        """Test FilterConfig with partial value override."""
        config = FilterConfig(
            similarity_threshold=0.75,
            log_level="WARNING"
        )
        
        # Overridden values
        assert config.similarity_threshold == 0.75
        assert config.log_level == "WARNING"
        
        # Default values
        assert config.semantic_model_type == "embedding"
        assert config.batch_size == 32
        assert config.enable_pruning is True
