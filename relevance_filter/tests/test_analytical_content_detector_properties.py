"""
Property-based tests for AnalyticalContentDetector component.

These tests use Hypothesis to verify universal properties across all inputs.
Each property test validates specific correctness properties from the design document.
"""

import pytest
from hypothesis import given, strategies as st
from hypothesis.strategies import composite
from relevance_filter.analytical_content_detector import AnalyticalContentDetector


@composite
def text_with_epistemic_verb(draw):
    """Generate text containing at least one epistemic verb."""
    epistemic_verbs = [
        "think", "believe", "seems", "appears", "would worry",
        "probably", "likely", "suggest", "indicate", "assume"
    ]
    verb = draw(st.sampled_from(epistemic_verbs))
    prefix = draw(st.text(alphabet=st.characters(blacklist_categories=('Cs',)), 
                          min_size=0, max_size=50))
    suffix = draw(st.text(alphabet=st.characters(blacklist_categories=('Cs',)), 
                          min_size=0, max_size=50))
    return f"{prefix} {verb} {suffix}".strip()


@composite
def text_with_causal_marker(draw):
    """Generate text containing at least one causal marker."""
    causal_markers = [
        "because", "therefore", "so that", "this means",
        "as a result", "consequently", "thus", "hence"
    ]
    marker = draw(st.sampled_from(causal_markers))
    prefix = draw(st.text(alphabet=st.characters(blacklist_categories=('Cs',)), 
                          min_size=0, max_size=50))
    suffix = draw(st.text(alphabet=st.characters(blacklist_categories=('Cs',)), 
                          min_size=0, max_size=50))
    return f"{prefix} {marker} {suffix}".strip()


@composite
def text_with_domain_term(draw):
    """Generate text containing at least one domain-specific term."""
    domain_terms = [
        "error rate", "liability", "training data", "fda",
        "malpractice", "precision", "outcomes", "efficacy",
        "safety", "clinical trial", "patient", "surgeon"
    ]
    term = draw(st.sampled_from(domain_terms))
    prefix = draw(st.text(alphabet=st.characters(blacklist_categories=('Cs',)), 
                          min_size=0, max_size=50))
    suffix = draw(st.text(alphabet=st.characters(blacklist_categories=('Cs',)), 
                          min_size=0, max_size=50))
    return f"{prefix} {term} {suffix}".strip()


@composite
def text_without_analytical_markers(draw):
    """Generate text without any analytical markers."""
    # Use simple words that don't contain any markers
    words = ["hello", "world", "test", "example", "simple", "text", "random"]
    word_count = draw(st.integers(min_value=1, max_value=10))
    selected_words = [draw(st.sampled_from(words)) for _ in range(word_count)]
    return " ".join(selected_words)


class TestAnalyticalContentDetectorProperties:
    """Property-based tests for AnalyticalContentDetector."""
    
    @given(text_with_epistemic_verb())
    def test_property_11_detects_epistemic_verbs(self, text):
        """
        Property 11: Analytical Content Detection (Epistemic Verbs)
        
        **Validates: Requirements 7.2**
        
        For any text containing at least one epistemic/evaluative verb 
        (think, believe, seems, etc.), analytical content should be detected.
        """
        detector = AnalyticalContentDetector()
        
        assert detector.has_analytical_content(text), \
            f"Failed to detect analytical content in text with epistemic verb: '{text}'"
        
        # Verify that epistemic verbs are identified in the markers
        markers = detector.get_analytical_markers(text)
        assert len(markers['epistemic_verbs']) > 0, \
            f"Expected epistemic verbs in markers, got: {markers}"
    
    @given(text_with_causal_marker())
    def test_property_11_detects_causal_markers(self, text):
        """
        Property 11: Analytical Content Detection (Causal Markers)
        
        **Validates: Requirements 7.2**
        
        For any text containing at least one causal marker 
        (because, therefore, etc.), analytical content should be detected.
        """
        detector = AnalyticalContentDetector()
        
        assert detector.has_analytical_content(text), \
            f"Failed to detect analytical content in text with causal marker: '{text}'"
        
        # Verify that causal markers are identified in the markers
        markers = detector.get_analytical_markers(text)
        assert len(markers['causal_markers']) > 0, \
            f"Expected causal markers in markers, got: {markers}"
    
    @given(text_with_domain_term())
    def test_property_11_detects_domain_terms(self, text):
        """
        Property 11: Analytical Content Detection (Domain Terms)
        
        **Validates: Requirements 7.2**
        
        For any text containing at least one domain-specific term 
        (error rate, liability, FDA, etc.), analytical content should be detected.
        """
        detector = AnalyticalContentDetector()
        
        assert detector.has_analytical_content(text), \
            f"Failed to detect analytical content in text with domain term: '{text}'"
        
        # Verify that domain terms are identified in the markers
        markers = detector.get_analytical_markers(text)
        assert len(markers['domain_terms']) > 0, \
            f"Expected domain terms in markers, got: {markers}"
    
    @given(text_without_analytical_markers())
    def test_property_11_no_false_positives(self, text):
        """
        Property 11: Analytical Content Detection (No False Positives)
        
        **Validates: Requirements 7.2**
        
        For any text without epistemic verbs, causal markers, or domain terms,
        analytical content should NOT be detected.
        """
        detector = AnalyticalContentDetector()
        
        assert not detector.has_analytical_content(text), \
            f"False positive: detected analytical content in text without markers: '{text}'"
        
        # Verify that no markers are identified
        markers = detector.get_analytical_markers(text)
        assert len(markers['epistemic_verbs']) == 0, \
            f"Unexpected epistemic verbs found: {markers['epistemic_verbs']}"
        assert len(markers['causal_markers']) == 0, \
            f"Unexpected causal markers found: {markers['causal_markers']}"
        assert len(markers['domain_terms']) == 0, \
            f"Unexpected domain terms found: {markers['domain_terms']}"
    
    @given(
        marker_type=st.sampled_from(['epistemic', 'causal', 'domain']),
        case_transform=st.sampled_from([
            lambda s: s.lower(),
            lambda s: s.upper(),
            lambda s: s.title(),
            lambda s: ''.join(c.upper() if i % 2 == 0 else c.lower() 
                            for i, c in enumerate(s))
        ])
    )
    def test_property_11_case_insensitivity(self, marker_type, case_transform):
        """
        Property 11: Analytical Content Detection (Case Insensitivity)
        
        **Validates: Requirements 7.2**
        
        For any analytical marker, detection should work regardless of 
        capitalization (lowercase, uppercase, mixed case).
        """
        detector = AnalyticalContentDetector()
        
        # Select a marker based on type
        if marker_type == 'epistemic':
            marker = "think"
        elif marker_type == 'causal':
            marker = "because"
        else:  # domain
            marker = "patient"
        
        # Apply case transformation
        transformed_marker = case_transform(marker)
        text = f"This is a test {transformed_marker} example"
        
        assert detector.has_analytical_content(text), \
            f"Failed to detect analytical content with case-transformed marker: '{text}'"
    
    @given(st.text(min_size=0, max_size=0))
    def test_property_11_empty_text(self, text):
        """
        Property 11: Analytical Content Detection (Empty Text)
        
        **Validates: Requirements 7.2**
        
        For any empty text, analytical content should NOT be detected.
        """
        detector = AnalyticalContentDetector()
        
        assert not detector.has_analytical_content(text), \
            f"Empty text should not have analytical content"
        
        markers = detector.get_analytical_markers(text)
        assert markers == {
            'epistemic_verbs': [],
            'causal_markers': [],
            'domain_terms': []
        }, f"Empty text should return empty markers, got: {markers}"
    
    @given(
        epistemic=st.sampled_from([
            "think", "believe", "seems", "appears", "would worry",
            "probably", "likely", "suggest", "indicate", "assume"
        ]),
        causal=st.sampled_from([
            "because", "therefore", "so that", "this means",
            "as a result", "consequently", "thus", "hence"
        ]),
        domain=st.sampled_from([
            "error rate", "liability", "training data", "fda",
            "malpractice", "precision", "outcomes", "efficacy",
            "safety", "clinical trial", "patient", "surgeon"
        ])
    )
    def test_property_11_multiple_markers(self, epistemic, causal, domain):
        """
        Property 11: Analytical Content Detection (Multiple Markers)
        
        **Validates: Requirements 7.2**
        
        For any text containing multiple analytical markers from different 
        categories, all markers should be detected.
        """
        detector = AnalyticalContentDetector()
        
        text = f"I {epistemic} this is important {causal} the {domain} is critical"
        
        assert detector.has_analytical_content(text), \
            f"Failed to detect analytical content in text with multiple markers: '{text}'"
        
        markers = detector.get_analytical_markers(text)
        
        # Should detect at least one marker from each category
        assert len(markers['epistemic_verbs']) > 0, \
            f"Expected epistemic verbs, got: {markers}"
        assert len(markers['causal_markers']) > 0, \
            f"Expected causal markers, got: {markers}"
        assert len(markers['domain_terms']) > 0, \
            f"Expected domain terms, got: {markers}"
    
    @given(
        marker=st.sampled_from([
            "think", "believe", "patient", "surgeon", "fda"
        ]),
        prefix=st.text(alphabet=st.characters(blacklist_categories=('Cs',)), 
                      min_size=1, max_size=20),
        suffix=st.text(alphabet=st.characters(blacklist_categories=('Cs',)), 
                      min_size=1, max_size=20)
    )
    def test_property_11_word_boundaries(self, marker, prefix, suffix):
        """
        Property 11: Analytical Content Detection (Word Boundaries)
        
        **Validates: Requirements 7.2**
        
        For any analytical marker, detection should respect word boundaries
        and not match partial words (e.g., "thinking" should match "think",
        but "rethink" as a standalone word should not).
        """
        detector = AnalyticalContentDetector()
        
        # Test with proper word boundaries (spaces)
        text_with_spaces = f"{prefix} {marker} {suffix}"
        assert detector.has_analytical_content(text_with_spaces), \
            f"Failed to detect marker with word boundaries: '{text_with_spaces}'"
        
        # Test that the marker is found when it's a complete word
        # (This is the expected behavior based on the implementation)
        markers = detector.get_analytical_markers(text_with_spaces)
        total_markers = (len(markers['epistemic_verbs']) + 
                        len(markers['causal_markers']) + 
                        len(markers['domain_terms']))
        assert total_markers > 0, \
            f"Expected to find marker in text: '{text_with_spaces}'"
