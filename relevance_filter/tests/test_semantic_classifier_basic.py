"""
Basic unit tests for SemanticClassifier.

These tests verify the core functionality of the semantic classifier
including model loading, embedding computation, and classification.
"""

import pytest
from relevance_filter.semantic_classifier import SemanticClassifier
from relevance_filter.models import RelevanceState


class TestSemanticClassifierBasic:
    """Basic tests for SemanticClassifier initialization and classification."""
    
    @pytest.fixture
    def classifier(self):
        """Create a SemanticClassifier instance for testing."""
        return SemanticClassifier(model_type="embedding")
    
    def test_initialization(self, classifier):
        """Test that classifier initializes successfully."""
        assert classifier is not None
        assert classifier.model_type == "embedding"
        assert classifier.model is not None
        assert hasattr(classifier, 'pos_embeddings')
        assert hasattr(classifier, 'neg_embeddings')
        assert len(classifier.pos_embeddings) > 0
        assert len(classifier.neg_embeddings) > 0
    
    def test_classify_local_relevant_strong(self, classifier):
        """Test classification of strongly relevant content."""
        text = "The da Vinci surgical robot performs autonomous laparoscopic procedures with high precision."
        state, score = classifier.classify_local(text)
        
        # Should be classified as relevant (STRONG or WEAK)
        assert state in [RelevanceState.RELEVANT_STRONG, RelevanceState.RELEVANT_WEAK]
        assert score > 0.5
    
    def test_classify_local_irrelevant(self, classifier):
        """Test classification of irrelevant content."""
        text = "I love pizza and ice cream on sunny days at the beach."
        state, score = classifier.classify_local(text)
        
        # Should be classified as irrelevant
        assert state == RelevanceState.IRRELEVANT
        assert score < 0.6
    
    def test_classify_local_empty_text(self, classifier):
        """Test classification of empty text."""
        state, score = classifier.classify_local("")
        
        assert state == RelevanceState.IRRELEVANT
        assert score == 0.0
    
    def test_classify_local_whitespace_only(self, classifier):
        """Test classification of whitespace-only text."""
        state, score = classifier.classify_local("   \n\t  ")
        
        assert state == RelevanceState.IRRELEVANT
        assert score == 0.0
    
    def test_classify_contextual(self, classifier):
        """Test contextual classification with parent-child pair."""
        parent_text = "The da Vinci robot is revolutionizing minimally invasive surgery."
        child_text = "I agree, it provides much better precision than traditional methods."
        
        state, score = classifier.classify_contextual(parent_text, child_text)
        
        # Should be classified as relevant when combined
        assert state in [RelevanceState.RELEVANT_STRONG, RelevanceState.RELEVANT_WEAK]
        assert score > 0.5
    
    def test_classify_joke_as_irrelevant(self, classifier):
        """Test that jokes about surgery are classified as irrelevant."""
        text = "I can finally automate my organ harvesting operation! Just kidding lol"
        state, score = classifier.classify_local(text)
        
        # Should be classified as irrelevant (joke without analytical content)
        assert state == RelevanceState.IRRELEVANT
    
    def test_classify_metaphor_as_irrelevant(self, classifier):
        """Test that metaphorical uses of surgery terms are classified as irrelevant."""
        text = "This code needs surgery, it's a complete mess and needs to be fixed."
        state, score = classifier.classify_local(text)
        
        # Should be classified as irrelevant (metaphorical use)
        assert state == RelevanceState.IRRELEVANT

    def test_classify_analytical_humor_as_relevant(self, classifier):
        """
        Test that humor grounded in analytical discussion is kept.
        Requirement 10.5: THE Semantic_Classifier SHALL accept content with light humor 
        if grounded in analytical surgical discussion.
        """
        text = "While people joke about robot overlords, the actual clinical data for robotic cholecystectomy shows significantly improved precision over manual methods."
        state, score = classifier.classify_local(text)
        
        # The presence of domain terms and analytical comparison should override the humor
        assert state in [RelevanceState.RELEVANT_STRONG, RelevanceState.RELEVANT_WEAK]
        assert score >= 0.6

    def test_classify_scifi_as_irrelevant(self, classifier):
        """
        Test that sci-fi robots are rejected.
        Requirement 10.4: THE Semantic_Classifier SHALL reject content about Robot Wars, or unrelated robotics.
        """
        text = "The Terminator and R2-D2 would have a great time performing surgery in Star Wars."
        state, score = classifier.classify_local(text)
        
        # Should be rejected due to negative reference embeddings matching
        assert state == RelevanceState.IRRELEVANT
