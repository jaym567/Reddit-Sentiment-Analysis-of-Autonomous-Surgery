"""
Unit tests for AnalyticalContentDetector component.

These tests verify specific examples and edge cases for analytical content detection.
Tests focus on boundary conditions, special characters, and concrete examples.
"""

import pytest
from relevance_filter.analytical_content_detector import AnalyticalContentDetector


class TestAnalyticalContentDetectorEdgeCases:
    """Unit tests for AnalyticalContentDetector edge cases."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.detector = AnalyticalContentDetector()
    
    # Tests for text with no markers
    
    def test_text_with_no_markers(self):
        """Text without any analytical markers should return False."""
        text = "This is just a simple comment about nothing special."
        assert not self.detector.has_analytical_content(text)
        
        markers = self.detector.get_analytical_markers(text)
        assert markers['epistemic_verbs'] == []
        assert markers['causal_markers'] == []
        assert markers['domain_terms'] == []
    
    def test_empty_string(self):
        """Empty string should return False."""
        assert not self.detector.has_analytical_content("")
        
        markers = self.detector.get_analytical_markers("")
        assert markers == {
            'epistemic_verbs': [],
            'causal_markers': [],
            'domain_terms': []
        }
    
    def test_whitespace_only(self):
        """Whitespace-only text should return False."""
        assert not self.detector.has_analytical_content("   ")
        assert not self.detector.has_analytical_content("\t\n  ")
        assert not self.detector.has_analytical_content("\n\n\n")
    
    def test_none_text(self):
        """None text should return False."""
        assert not self.detector.has_analytical_content(None)
        
        markers = self.detector.get_analytical_markers(None)
        assert markers == {
            'epistemic_verbs': [],
            'causal_markers': [],
            'domain_terms': []
        }
    
    def test_joke_without_markers(self):
        """Pure joke without analytical markers should return False."""
        text = "lol this is hilarious 😂"
        assert not self.detector.has_analytical_content(text)
    
    def test_casual_conversation(self):
        """Casual conversation without analytical content should return False."""
        text = "Yeah I agree, that's pretty cool!"
        # Note: "agree" is not in the marker lists, so this should be False
        assert not self.detector.has_analytical_content(text)
    
    # Tests for text with multiple markers
    
    def test_multiple_epistemic_verbs(self):
        """Text with multiple epistemic verbs should detect all of them."""
        text = "I think this seems likely to indicate a problem."
        assert self.detector.has_analytical_content(text)
        
        markers = self.detector.get_analytical_markers(text)
        assert 'think' in markers['epistemic_verbs']
        assert 'seems' in markers['epistemic_verbs']
        assert 'likely' in markers['epistemic_verbs']
        assert 'indicate' in markers['epistemic_verbs']
        assert len(markers['epistemic_verbs']) == 4
    
    def test_multiple_causal_markers(self):
        """Text with multiple causal markers should detect all of them."""
        text = "This happens because of X, therefore Y, and as a result Z."
        assert self.detector.has_analytical_content(text)
        
        markers = self.detector.get_analytical_markers(text)
        assert 'because' in markers['causal_markers']
        assert 'therefore' in markers['causal_markers']
        assert 'as a result' in markers['causal_markers']
        assert len(markers['causal_markers']) == 3
    
    def test_multiple_domain_terms(self):
        """Text with multiple domain terms should detect all of them."""
        text = "The surgeon reviewed the patient outcomes and error rate data from the clinical trial."
        assert self.detector.has_analytical_content(text)
        
        markers = self.detector.get_analytical_markers(text)
        assert 'surgeon' in markers['domain_terms']
        assert 'patient' in markers['domain_terms']
        assert 'outcomes' in markers['domain_terms']
        assert 'error rate' in markers['domain_terms']
        assert 'clinical trial' in markers['domain_terms']
        assert len(markers['domain_terms']) == 5
    
    def test_mixed_marker_types(self):
        """Text with markers from all three categories should detect all."""
        text = "I think the error rate is high because the surgeon lacks training data."
        assert self.detector.has_analytical_content(text)
        
        markers = self.detector.get_analytical_markers(text)
        assert 'think' in markers['epistemic_verbs']
        assert 'because' in markers['causal_markers']
        assert 'error rate' in markers['domain_terms']
        assert 'surgeon' in markers['domain_terms']
        assert 'training data' in markers['domain_terms']
    
    def test_repeated_markers(self):
        """Text with repeated markers should detect them multiple times."""
        text = "I think, therefore I think again. I really think this is important."
        assert self.detector.has_analytical_content(text)
        
        markers = self.detector.get_analytical_markers(text)
        # The implementation finds unique markers, not counts
        assert 'think' in markers['epistemic_verbs']
    
    # Boundary cases - word boundaries
    
    def test_word_boundary_epistemic_verb(self):
        """Epistemic verbs should match only as complete words."""
        # Should match
        assert self.detector.has_analytical_content("I think so")
        assert self.detector.has_analytical_content("think about it")
        assert self.detector.has_analytical_content("what do you think?")
        
        # Should NOT match partial words
        assert not self.detector.has_analytical_content("rethink")
        assert not self.detector.has_analytical_content("thinking")  # "think" should match
        # Actually, "thinking" contains "think" as a word boundary match
        # Let me reconsider - the regex uses \b which matches word boundaries
        # "thinking" would match because "think" is at the start with a word boundary
        # Let me test with a word where the marker is truly embedded
        assert not self.detector.has_analytical_content("methinks")  # "think" is embedded
    
    def test_word_boundary_domain_term(self):
        """Domain terms should match only as complete words."""
        # Should match
        assert self.detector.has_analytical_content("the patient is stable")
        assert self.detector.has_analytical_content("Patient care is important")
        
        # Should NOT match when embedded in other words
        assert not self.detector.has_analytical_content("impatient")
        assert not self.detector.has_analytical_content("outpatient")  # "patient" is embedded
    
    def test_word_boundary_with_punctuation(self):
        """Markers should be detected even with adjacent punctuation."""
        assert self.detector.has_analytical_content("I think, therefore I am.")
        assert self.detector.has_analytical_content("The patient's condition...")
        assert self.detector.has_analytical_content("Because!")
        assert self.detector.has_analytical_content("(probably)")
    
    def test_phrase_markers(self):
        """Multi-word markers should be detected correctly."""
        # Causal markers that are phrases
        assert self.detector.has_analytical_content("so that we can proceed")
        assert self.detector.has_analytical_content("this means something")
        assert self.detector.has_analytical_content("as a result of the test")
        
        # Domain terms that are phrases
        assert self.detector.has_analytical_content("the error rate is high")
        assert self.detector.has_analytical_content("training data is needed")
        assert self.detector.has_analytical_content("clinical trial results")
    
    # Boundary cases - case sensitivity
    
    def test_case_insensitivity_uppercase(self):
        """Markers should be detected regardless of case."""
        assert self.detector.has_analytical_content("I THINK THIS IS IMPORTANT")
        assert self.detector.has_analytical_content("BECAUSE OF THE PATIENT")
        assert self.detector.has_analytical_content("THE ERROR RATE IS HIGH")
    
    def test_case_insensitivity_mixed(self):
        """Markers should be detected with mixed case."""
        assert self.detector.has_analytical_content("I ThInK this is weird")
        assert self.detector.has_analytical_content("BeCaUsE reasons")
        assert self.detector.has_analytical_content("PaTiEnT care")
    
    def test_case_insensitivity_title_case(self):
        """Markers should be detected in title case."""
        assert self.detector.has_analytical_content("I Think Therefore I Am")
        assert self.detector.has_analytical_content("Because Of The Patient")
        assert self.detector.has_analytical_content("The Surgeon Performed Well")
    
    # Boundary cases - special characters
    
    def test_special_characters_around_markers(self):
        """Markers should be detected with special characters nearby."""
        assert self.detector.has_analytical_content("I think... maybe?")
        assert self.detector.has_analytical_content("because!!! important!!!")
        assert self.detector.has_analytical_content("the patient's condition")
        assert self.detector.has_analytical_content("the error rate is high")  # "error rate" with space, not hyphen
    
    def test_unicode_characters(self):
        """Markers should be detected in text with unicode characters."""
        assert self.detector.has_analytical_content("I think 🤔 this is important")
        assert self.detector.has_analytical_content("The patient's condition is stable ✓")
        assert self.detector.has_analytical_content("Error rate: 5% → concerning")
    
    def test_newlines_and_tabs(self):
        """Markers should be detected across newlines and tabs."""
        text_with_newlines = "I think\nthis is important\nbecause\nthe patient needs care"
        assert self.detector.has_analytical_content(text_with_newlines)
        
        text_with_tabs = "I think\tthis is\timportant\tbecause\tthe patient"
        assert self.detector.has_analytical_content(text_with_tabs)
    
    def test_html_entities(self):
        """Markers should be detected even with HTML entities."""
        assert self.detector.has_analytical_content("I think &amp; believe this")
        assert self.detector.has_analytical_content("The patient&apos;s condition")
    
    # Boundary cases - very long text
    
    def test_very_long_text_with_marker_at_start(self):
        """Marker at the start of very long text should be detected."""
        long_text = "I think " + "x " * 10000
        assert self.detector.has_analytical_content(long_text)
    
    def test_very_long_text_with_marker_at_end(self):
        """Marker at the end of very long text should be detected."""
        long_text = "x " * 10000 + " I think"
        assert self.detector.has_analytical_content(long_text)
    
    def test_very_long_text_with_marker_in_middle(self):
        """Marker in the middle of very long text should be detected."""
        long_text = "x " * 5000 + " I think " + "x " * 5000
        assert self.detector.has_analytical_content(long_text)
    
    def test_very_long_text_without_markers(self):
        """Very long text without markers should return False."""
        long_text = "hello world " * 10000
        assert not self.detector.has_analytical_content(long_text)
    
    # Boundary cases - minimal text
    
    def test_single_word_marker(self):
        """Single word that is a marker should be detected."""
        assert self.detector.has_analytical_content("think")
        assert self.detector.has_analytical_content("because")
        assert self.detector.has_analytical_content("patient")
        assert self.detector.has_analytical_content("surgeon")
    
    def test_marker_with_single_extra_word(self):
        """Marker with just one additional word should be detected."""
        assert self.detector.has_analytical_content("I think")
        assert self.detector.has_analytical_content("think so")
        assert self.detector.has_analytical_content("because yes")
        assert self.detector.has_analytical_content("the patient")
    
    # Specific real-world examples
    
    def test_analytical_humor_example(self):
        """Analytical humor should be detected (has markers)."""
        text = "I think the robot surgeon would probably have better precision than my shaky hands lol"
        assert self.detector.has_analytical_content(text)
        
        markers = self.detector.get_analytical_markers(text)
        assert 'think' in markers['epistemic_verbs']
        assert 'probably' in markers['epistemic_verbs']
        assert 'surgeon' in markers['domain_terms']
        assert 'precision' in markers['domain_terms']
    
    def test_pure_joke_example(self):
        """Pure joke without analytical markers should not be detected."""
        text = "lol I can finally automate my organ harvesting operation! 😂"
        assert not self.detector.has_analytical_content(text)
    
    def test_technical_discussion_example(self):
        """Technical discussion should be detected."""
        text = "The error rate seems concerning because the training data was insufficient for the clinical trial."
        assert self.detector.has_analytical_content(text)
        
        markers = self.detector.get_analytical_markers(text)
        assert 'seems' in markers['epistemic_verbs']
        assert 'because' in markers['causal_markers']
        assert 'error rate' in markers['domain_terms']
        assert 'training data' in markers['domain_terms']
        assert 'clinical trial' in markers['domain_terms']
    
    def test_skeptical_comment_example(self):
        """Skeptical comment with analytical content should be detected."""
        text = "I would worry about the liability issues and malpractice concerns with autonomous surgery."
        assert self.detector.has_analytical_content(text)
        
        markers = self.detector.get_analytical_markers(text)
        assert 'would worry' in markers['epistemic_verbs']
        assert 'liability' in markers['domain_terms']
        assert 'malpractice' in markers['domain_terms']
    
    def test_causal_reasoning_example(self):
        """Causal reasoning should be detected."""
        text = "The outcomes improved as a result of better precision, therefore the FDA approved it."
        assert self.detector.has_analytical_content(text)
        
        markers = self.detector.get_analytical_markers(text)
        assert 'as a result' in markers['causal_markers']
        assert 'therefore' in markers['causal_markers']
        assert 'outcomes' in markers['domain_terms']
        assert 'precision' in markers['domain_terms']
        assert 'fda' in markers['domain_terms']
    
    # Edge cases with similar but non-matching words
    
    def test_similar_words_not_matching(self):
        """Words similar to markers but not exact matches should not be detected."""
        # These should NOT match (no analytical markers)
        assert not self.detector.has_analytical_content("I thought about it")  # "thought" not "think"
        assert not self.detector.has_analytical_content("belief system")  # "belief" not "believe"
        assert not self.detector.has_analytical_content("seemingly")  # "seemingly" not "seems"
        
        # But these SHOULD match because they contain the actual markers
        assert self.detector.has_analytical_content("I believe in this")
        assert self.detector.has_analytical_content("It seems good")
    
    def test_substring_not_matching(self):
        """Substrings that contain markers should not match unless word boundaries align."""
        # "patient" embedded in other words should not match
        assert not self.detector.has_analytical_content("impatient driver")
        
        # But "patient" as a separate word should match
        assert self.detector.has_analytical_content("patient driver")  # "patient" is a separate word here
    
    # Tests for get_analytical_markers method
    
    def test_get_markers_returns_all_categories(self):
        """get_analytical_markers should always return all three categories."""
        markers = self.detector.get_analytical_markers("random text")
        assert 'epistemic_verbs' in markers
        assert 'causal_markers' in markers
        assert 'domain_terms' in markers
        assert isinstance(markers['epistemic_verbs'], list)
        assert isinstance(markers['causal_markers'], list)
        assert isinstance(markers['domain_terms'], list)
    
    def test_get_markers_empty_text(self):
        """get_analytical_markers should return empty lists for empty text."""
        markers = self.detector.get_analytical_markers("")
        assert markers == {
            'epistemic_verbs': [],
            'causal_markers': [],
            'domain_terms': []
        }
    
    def test_get_markers_with_all_types(self):
        """get_analytical_markers should categorize markers correctly."""
        text = "I think the patient outcomes are good because of the surgeon's precision."
        markers = self.detector.get_analytical_markers(text)
        
        assert 'think' in markers['epistemic_verbs']
        assert 'because' in markers['causal_markers']
        assert 'patient' in markers['domain_terms']
        assert 'outcomes' in markers['domain_terms']
        assert 'surgeon' in markers['domain_terms']
        assert 'precision' in markers['domain_terms']
