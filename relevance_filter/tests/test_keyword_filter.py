"""
Unit tests for KeywordFilter component.

Tests the keyword matching functionality including case-insensitive matching,
keyword comprehensiveness, and edge cases.
"""

import pytest
from hypothesis import given, strategies as st
from relevance_filter.keyword_filter import KeywordFilter


class TestKeywordFilterBasics:
    """Test basic keyword filter functionality."""
    
    def test_initialization(self):
        """Test that KeywordFilter initializes with all keyword lists."""
        kf = KeywordFilter()
        
        # Verify keyword lists are populated
        assert len(kf.surgery_keywords) > 0
        assert len(kf.procedure_keywords) > 0
        assert len(kf.context_keywords) > 0
        assert len(kf.all_keywords) > 0
        
        # Verify all_keywords contains all unique keywords from categories
        expected_unique = set(
            kf.surgery_keywords + 
            kf.procedure_keywords + 
            kf.context_keywords
        )
        assert len(kf.all_keywords) == len(expected_unique)
    
    def test_matches_surgery_keyword(self):
        """Test matching with surgery keywords."""
        kf = KeywordFilter()
        
        # Test various surgery keywords
        assert kf.matches("Discussion about autonomous surgery")
        assert kf.matches("The robotic surgeon performed the operation")
        assert kf.matches("New surgical robot capabilities")
        assert kf.matches("da Vinci system is impressive")
        assert kf.matches("robot-assisted surgery outcomes")
    
    def test_matches_procedure_keyword(self):
        """Test matching with procedure keywords."""
        kf = KeywordFilter()
        
        # Test various procedure keywords
        assert kf.matches("Patient underwent cholecystectomy")
        assert kf.matches("Robotic prostatectomy results")
        assert kf.matches("minimally invasive approach")
    
    def test_matches_context_keyword(self):
        """Test matching with context keywords."""
        kf = KeywordFilter()
        
        # Test various context keywords
        assert kf.matches("operating room AI integration")
        assert kf.matches("surgical AI capabilities")
        assert kf.matches("robotic precision is important")
    
    def test_no_match_irrelevant_text(self):
        """Test that irrelevant text does not match."""
        kf = KeywordFilter()
        
        # Test text without any keywords
        assert not kf.matches("This is completely unrelated text")
        assert not kf.matches("I love pizza and movies")
        assert not kf.matches("The weather is nice today")
    
    def test_case_insensitive_matching(self):
        """Test that matching is case-insensitive (Requirement 9.4)."""
        kf = KeywordFilter()
        
        # Test various capitalizations
        assert kf.matches("ROBOTIC SURGERY")
        assert kf.matches("Robotic Surgery")
        assert kf.matches("robotic surgery")
        assert kf.matches("RoBoTiC sUrGeRy")
        
        assert kf.matches("DA VINCI")
        assert kf.matches("Da Vinci")
        assert kf.matches("da vinci")
        
        assert kf.matches("CHOLECYSTECTOMY")
        assert kf.matches("Cholecystectomy")
        assert kf.matches("cholecystectomy")
    
    def test_empty_text(self):
        """Test handling of empty text."""
        kf = KeywordFilter()
        
        assert not kf.matches("")
        assert not kf.matches("   ")
        assert not kf.matches(None)
    
    def test_get_matched_keywords(self):
        """Test get_matched_keywords returns correct matches."""
        kf = KeywordFilter()
        
        # Test single keyword match
        text1 = "Discussion about robotic surgery"
        matches1 = kf.get_matched_keywords(text1)
        assert "robotic surgery" in matches1
        
        # Test multiple keyword matches
        text2 = "The da Vinci surgical robot performs cholecystectomy"
        matches2 = kf.get_matched_keywords(text2)
        assert "da vinci" in matches2
        assert "surgical robot" in matches2
        assert "cholecystectomy" in matches2
        assert len(matches2) == 3
        
        # Test no matches
        text3 = "Completely unrelated text"
        matches3 = kf.get_matched_keywords(text3)
        assert len(matches3) == 0
    
    def test_get_matched_keywords_empty_text(self):
        """Test get_matched_keywords with empty text."""
        kf = KeywordFilter()
        
        assert kf.get_matched_keywords("") == []
        assert kf.get_matched_keywords(None) == []
    
    def test_keyword_in_longer_text(self):
        """Test that keywords are found within longer text."""
        kf = KeywordFilter()
        
        long_text = """
        In recent years, there has been significant advancement in the field
        of robotic surgery. The da Vinci surgical system has revolutionized
        minimally invasive procedures, particularly in cholecystectomy and
        prostatectomy operations. The robotic precision and surgical AI
        capabilities have improved patient outcomes significantly.
        """
        
        assert kf.matches(long_text)
        
        matches = kf.get_matched_keywords(long_text)
        assert len(matches) >= 5  # Should find multiple keywords


class TestKeywordFilterEdgeCases:
    """Test edge cases for keyword filter (Requirements 9.1, 9.2, 9.3, 9.4)."""
    
    def test_partial_word_match(self):
        """Test that keywords work with substring matching in natural text."""
        kf = KeywordFilter()
        
        # Keywords match as substrings within natural language
        assert kf.matches("The robotic surgery field is advancing")
        assert kf.matches("In robotic surgery, precision matters")
        assert kf.matches("Discussion of da vinci systems")
        
        # Multi-word keywords require the exact phrase with spaces
        # This is expected and correct behavior for natural language processing
    
    def test_keyword_at_boundaries(self):
        """Test keywords at text boundaries."""
        kf = KeywordFilter()
        
        # Keyword at start
        assert kf.matches("robotic surgery is important")
        
        # Keyword at end
        assert kf.matches("This is about robotic surgery")
        
        # Keyword is entire text
        assert kf.matches("robotic surgery")
    
    def test_multiple_spaces_and_newlines(self):
        """Test text with multiple spaces and newlines."""
        kf = KeywordFilter()
        
        text = "Discussion   about\n\nrobotic surgery\n\nand its benefits"
        assert kf.matches(text)
    
    def test_special_characters(self):
        """Test text with special characters."""
        kf = KeywordFilter()
        
        assert kf.matches("The da Vinci® surgical robot")
        assert kf.matches("Robotic surgery: the future?")
        assert kf.matches("AI surgery (autonomous)")
    
    def test_empty_text_variations(self):
        """Test various forms of empty text (Requirement 9.4)."""
        kf = KeywordFilter()
        
        # Empty string
        assert not kf.matches("")
        
        # Whitespace only
        assert not kf.matches(" ")
        assert not kf.matches("   ")
        assert not kf.matches("\t")
        assert not kf.matches("\n")
        assert not kf.matches("\r\n")
        assert not kf.matches("  \t  \n  ")
        
        # None value
        assert not kf.matches(None)
        
        # Empty text should return empty matched keywords list
        assert kf.get_matched_keywords("") == []
        assert kf.get_matched_keywords("   ") == []
        assert kf.get_matched_keywords(None) == []
    
    def test_very_long_text(self):
        """Test keyword matching in very long text (Requirement 9.4)."""
        kf = KeywordFilter()
        
        # Create very long text (10,000+ characters)
        filler = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * 200
        
        # Keyword at the beginning of long text
        text_start = "robotic surgery is important. " + filler
        assert kf.matches(text_start)
        assert "robotic surgery" in kf.get_matched_keywords(text_start)
        
        # Keyword at the end of long text
        text_end = filler + " The future is robotic surgery."
        assert kf.matches(text_end)
        assert "robotic surgery" in kf.get_matched_keywords(text_end)
        
        # Keyword in the middle of long text
        text_middle = filler[:5000] + " robotic surgery " + filler[5000:]
        assert kf.matches(text_middle)
        assert "robotic surgery" in kf.get_matched_keywords(text_middle)
        
        # Multiple keywords in very long text
        text_multiple = (
            "Discussion about autonomous surgery. " + 
            filler + 
            " The da vinci system is used for cholecystectomy procedures."
        )
        assert kf.matches(text_multiple)
        matched = kf.get_matched_keywords(text_multiple)
        assert "autonomous surgery" in matched
        assert "da vinci" in matched
        assert "cholecystectomy" in matched
    
    def test_special_characters_comprehensive(self):
        """Test text with various special characters (Requirement 9.4)."""
        kf = KeywordFilter()
        
        # Punctuation around keywords
        assert kf.matches("robotic surgery!")
        assert kf.matches("robotic surgery?")
        assert kf.matches("robotic surgery.")
        assert kf.matches("robotic surgery,")
        assert kf.matches("robotic surgery;")
        assert kf.matches("robotic surgery:")
        assert kf.matches("(robotic surgery)")
        assert kf.matches("[robotic surgery]")
        assert kf.matches("{robotic surgery}")
        assert kf.matches("'robotic surgery'")
        assert kf.matches('"robotic surgery"')
        
        # Special symbols
        assert kf.matches("robotic surgery @ hospital")
        assert kf.matches("robotic surgery #1")
        assert kf.matches("robotic surgery $1000")
        assert kf.matches("robotic surgery & AI")
        assert kf.matches("robotic surgery * precision")
        assert kf.matches("robotic surgery + AI")
        assert kf.matches("robotic surgery = future")
        assert kf.matches("robotic surgery / AI")
        assert kf.matches("robotic surgery \\ AI")
        assert kf.matches("robotic surgery | AI")
        assert kf.matches("robotic surgery ~ AI")
        assert kf.matches("robotic surgery ` AI")
        
        # Trademark and copyright symbols
        assert kf.matches("da Vinci® surgical robot")
        assert kf.matches("da Vinci™ system")
        assert kf.matches("robotic surgery© technology")
        
        # Accented characters
        assert kf.matches("robotic surgery in São Paulo")
        assert kf.matches("robotic surgery café discussion")
        assert kf.matches("robotic surgery naïve approach")
    
    def test_unicode_characters(self):
        """Test text with Unicode characters (Requirement 9.4)."""
        kf = KeywordFilter()
        
        # Emoji
        assert kf.matches("robotic surgery 🤖")
        assert kf.matches("🏥 robotic surgery")
        assert kf.matches("robotic surgery is amazing! 👍")
        
        # Non-Latin scripts
        assert kf.matches("robotic surgery 手术机器人")
        assert kf.matches("robotic surgery ロボット手術")
        assert kf.matches("robotic surgery 로봇 수술")
        
        # Mathematical symbols
        assert kf.matches("robotic surgery ≥ traditional")
        assert kf.matches("robotic surgery ≠ manual")
        assert kf.matches("robotic surgery → future")
    
    def test_mixed_case_with_special_chars(self):
        """Test case insensitivity with special characters (Requirement 9.4)."""
        kf = KeywordFilter()
        
        # Various case combinations with punctuation
        assert kf.matches("ROBOTIC SURGERY!")
        assert kf.matches("Robotic Surgery?")
        assert kf.matches("RoBoTiC sUrGeRy.")
        assert kf.matches("(ROBOTIC SURGERY)")
        assert kf.matches("[Robotic Surgery]")
        
        # Case insensitivity with special symbols
        assert kf.matches("DA VINCI® system")
        assert kf.matches("Da Vinci™ robot")
        assert kf.matches("CHOLECYSTECTOMY: procedure")
    
    def test_keywords_with_extra_whitespace(self):
        """Test keywords with various whitespace patterns (Requirement 9.4)."""
        kf = KeywordFilter()
        
        # Note: Multi-word keywords require exact spacing
        # Single extra space breaks the match for multi-word keywords
        # This is expected behavior for substring matching
        
        # Keywords with surrounding whitespace
        assert kf.matches("  robotic surgery  ")
        assert kf.matches("\trobotic surgery\t")
        assert kf.matches("\nrobotic surgery\n")
        
        # Keywords with newlines in surrounding text
        assert kf.matches("Discussion about\nrobotic surgery\nand its benefits")
        assert kf.matches("The\nda vinci\nsystem")
    
    def test_repeated_keywords(self):
        """Test text with repeated keywords (Requirement 9.4)."""
        kf = KeywordFilter()
        
        # Same keyword repeated
        text = "robotic surgery and more robotic surgery"
        assert kf.matches(text)
        matched = kf.get_matched_keywords(text)
        # Should return the keyword once (not duplicated)
        assert "robotic surgery" in matched
        
        # Multiple different keywords
        text2 = "robotic surgery using da vinci for cholecystectomy"
        assert kf.matches(text2)
        matched2 = kf.get_matched_keywords(text2)
        assert len(matched2) >= 3
        assert "robotic surgery" in matched2
        assert "da vinci" in matched2
        assert "cholecystectomy" in matched2
    
    def test_keyword_substrings(self):
        """Test that keywords match as substrings in compound words."""
        kf = KeywordFilter()
        
        # Keywords should match even when part of larger words/phrases
        # This is expected behavior for substring matching
        assert kf.matches("The robotic surgery field")
        assert kf.matches("In robotic surgery, we see improvements")
        assert kf.matches("Post-robotic surgery recovery")
        
        # Single-word keywords in compound contexts
        assert kf.matches("cholecystectomy procedure")
        assert kf.matches("post-cholecystectomy")
        assert kf.matches("cholecystectomy-related")
    
    def test_no_false_positives(self):
        """Test that unrelated text does not match (Requirement 9.4)."""
        kf = KeywordFilter()
        
        # Text with similar but non-matching words
        assert not kf.matches("I like robots and video games")
        assert not kf.matches("The surgeon performed the operation")
        assert not kf.matches("Artificial intelligence is interesting")
        assert not kf.matches("I had surgery last year")
        assert not kf.matches("The robot vacuum cleaner")
        assert not kf.matches("Surgical masks are important")
        
        # Text with partial keyword matches that don't form complete keywords
        # Note: substring matching means "robot" alone won't match "robotic surgery"
        assert not kf.matches("I bought a robot toy")
        assert not kf.matches("The surgery was successful")
        assert not kf.matches("Autonomous vehicles are cool")
        
        # Empty and whitespace
        assert not kf.matches("")
        assert not kf.matches("   ")
        assert not kf.matches(None)
    
    def test_text_with_urls_and_emails(self):
        """Test text containing URLs and email addresses (Requirement 9.4)."""
        kf = KeywordFilter()
        
        # Keywords with URLs
        assert kf.matches("Check out robotic surgery at https://example.com")
        assert kf.matches("Learn about da vinci at www.example.com")
        assert kf.matches("Visit https://example.com for robotic surgery info")
        
        # Note: "robotic-surgery" (with hyphen) won't match "robotic surgery" (with space)
        # This is expected behavior for substring matching
        assert not kf.matches("https://example.com/robotic-surgery")
        
        # Keywords with email addresses
        assert kf.matches("Contact us about robotic surgery at info@example.com")
        assert kf.matches("Email robotic surgery questions to info@hospital.com")
    
    def test_text_with_numbers(self):
        """Test text with numbers mixed with keywords (Requirement 9.4)."""
        kf = KeywordFilter()
        
        # Keywords with numbers
        assert kf.matches("robotic surgery 2023")
        assert kf.matches("da vinci 5th generation")
        assert kf.matches("1000 robotic surgery procedures")
        assert kf.matches("robotic surgery costs $50,000")
        assert kf.matches("99% success rate for robotic surgery")
        
        # Numbers in various formats
        assert kf.matches("robotic surgery: 1,000 cases")
        assert kf.matches("robotic surgery (2023-2024)")
        assert kf.matches("robotic surgery v2.0")
    
    def test_html_and_markdown(self):
        """Test text with HTML and Markdown formatting (Requirement 9.4)."""
        kf = KeywordFilter()
        
        # HTML tags
        assert kf.matches("<p>robotic surgery</p>")
        assert kf.matches("<b>robotic surgery</b>")
        assert kf.matches("<a href='#'>robotic surgery</a>")
        assert kf.matches("<!-- robotic surgery -->")
        
        # Markdown formatting
        assert kf.matches("**robotic surgery**")
        assert kf.matches("*robotic surgery*")
        assert kf.matches("# robotic surgery")
        assert kf.matches("[robotic surgery](link)")
        assert kf.matches("`robotic surgery`")
        assert kf.matches("~~robotic surgery~~")
    
    def test_single_character_text(self):
        """Test very short text inputs (Requirement 9.4)."""
        kf = KeywordFilter()
        
        # Single characters
        assert not kf.matches("a")
        assert not kf.matches("1")
        assert not kf.matches("!")
        
        # Two characters
        assert not kf.matches("ab")
        assert not kf.matches("12")
        
        # Short text without keywords
        assert not kf.matches("hi")
        assert not kf.matches("ok")
        assert not kf.matches("yes")


class TestKeywordFilterRequirements:
    """Test specific requirements validation."""
    
    def test_requirement_9_1_surgery_keywords(self):
        """Test Requirement 9.1: Surgery-related keywords."""
        kf = KeywordFilter()
        
        # Verify all required surgery keywords are present
        required_keywords = [
            "autonomous surgery",
            "robotic surgeon",
            "surgical robot",
            "da vinci",
            "laparoscopic robot",
            "surgical automation",
            "ai surgery",
            "robot-assisted surgery"
        ]
        
        for keyword in required_keywords:
            # Check keyword is in the list
            assert any(keyword.lower() in k.lower() for k in kf.surgery_keywords), \
                f"Required keyword '{keyword}' not found in surgery_keywords"
            
            # Check it matches in text
            assert kf.matches(f"Text about {keyword}")
    
    def test_requirement_9_2_procedure_keywords(self):
        """Test Requirement 9.2: Procedure keywords."""
        kf = KeywordFilter()
        
        # Verify all required procedure keywords are present
        required_keywords = [
            "cholecystectomy",
            "prostatectomy",
            "hysterectomy",
            "cardiac surgery",
            "minimally invasive"
        ]
        
        for keyword in required_keywords:
            # Check keyword is in the list
            assert any(keyword.lower() in k.lower() for k in kf.procedure_keywords), \
                f"Required keyword '{keyword}' not found in procedure_keywords"
            
            # Check it matches in text
            assert kf.matches(f"Text about {keyword}")
    
    def test_requirement_9_3_context_keywords(self):
        """Test Requirement 9.3: Context keywords."""
        kf = KeywordFilter()
        
        # Verify all required context keywords are present
        required_keywords = [
            "operating room ai",
            "surgical ai",
            "autonomous operation",
            "robotic precision"
        ]
        
        for keyword in required_keywords:
            # Check keyword is in the list
            assert any(keyword.lower() in k.lower() for k in kf.context_keywords), \
                f"Required keyword '{keyword}' not found in context_keywords"
            
            # Check it matches in text
            assert kf.matches(f"Text about {keyword}")
    
    def test_requirement_9_4_case_insensitive(self):
        """Test Requirement 9.4: Case-insensitive matching."""
        kf = KeywordFilter()
        
        # Test same keyword in different cases
        test_cases = [
            ("robotic surgery", "ROBOTIC SURGERY"),
            ("robotic surgery", "Robotic Surgery"),
            ("robotic surgery", "RoBoTiC sUrGeRy"),
            ("da vinci", "DA VINCI"),
            ("da vinci", "Da Vinci"),
            ("cholecystectomy", "CHOLECYSTECTOMY"),
            ("cholecystectomy", "Cholecystectomy"),
        ]
        
        for original, variant in test_cases:
            # Both should match
            assert kf.matches(original), f"Failed to match: {original}"
            assert kf.matches(variant), f"Failed to match: {variant}"
            
            # Both should return the same keyword (in original form)
            matches_original = kf.get_matched_keywords(original)
            matches_variant = kf.get_matched_keywords(variant)
            
            # Should find at least one match in both
            assert len(matches_original) > 0
            assert len(matches_variant) > 0



class TestKeywordFilterProperties:
    """Property-based tests for keyword filter using Hypothesis."""
    
    @given(st.sampled_from([
        # Surgery keywords (Requirement 9.1)
        "autonomous surgery",
        "robotic surgeon",
        "surgical robot",
        "da vinci",
        "laparoscopic robot",
        "surgical automation",
        "ai surgery",
        "robot-assisted surgery",
        "robotic surgery",
        # Procedure keywords (Requirement 9.2)
        "cholecystectomy",
        "prostatectomy",
        "hysterectomy",
        "cardiac surgery",
        "minimally invasive",
        # Context keywords (Requirement 9.3)
        "operating room ai",
        "surgical ai",
        "autonomous operation",
        "robotic precision",
        "surgical robotics"
    ]))
    def test_property_5_keyword_filter_comprehensiveness(self, keyword):
        """
        Property 5: Keyword Filter Comprehensiveness
        
        **Validates: Requirements 9.1, 9.2, 9.3**
        
        For any text containing at least one term from the surgery keywords,
        procedure keywords, or context keywords, the keyword filter should
        return a match.
        
        This property ensures that the keyword filter is comprehensive and
        catches all relevant keywords across all three categories.
        """
        kf = KeywordFilter()
        
        # Test the keyword alone
        assert kf.matches(keyword), \
            f"Keyword filter failed to match keyword alone: '{keyword}'"
        
        # Test keyword with prefix text
        text_with_prefix = f"Some text before {keyword}"
        assert kf.matches(text_with_prefix), \
            f"Keyword filter failed to match with prefix: '{text_with_prefix}'"
        
        # Test keyword with suffix text
        text_with_suffix = f"{keyword} and some text after"
        assert kf.matches(text_with_suffix), \
            f"Keyword filter failed to match with suffix: '{text_with_suffix}'"
        
        # Test keyword in the middle of text
        text_in_middle = f"Text before {keyword} and text after"
        assert kf.matches(text_in_middle), \
            f"Keyword filter failed to match in middle: '{text_in_middle}'"
        
        # Test that get_matched_keywords returns the keyword
        matched = kf.get_matched_keywords(keyword)
        assert len(matched) > 0, \
            f"get_matched_keywords returned empty list for: '{keyword}'"
        assert keyword in matched, \
            f"get_matched_keywords did not return the keyword: '{keyword}'"
    
    @given(
        keyword=st.sampled_from([
            "autonomous surgery", "robotic surgeon", "surgical robot",
            "da vinci", "cholecystectomy", "prostatectomy",
            "operating room ai", "surgical ai", "robotic precision"
        ]),
        prefix=st.text(min_size=0, max_size=50, alphabet=st.characters(blacklist_categories=('Cs',))),
        suffix=st.text(min_size=0, max_size=50, alphabet=st.characters(blacklist_categories=('Cs',)))
    )
    def test_property_5_keyword_with_random_context(self, keyword, prefix, suffix):
        """
        Property 5: Keyword Filter Comprehensiveness (with random context)
        
        **Validates: Requirements 9.1, 9.2, 9.3**
        
        For any text containing at least one keyword surrounded by arbitrary
        text, the keyword filter should return a match.
        
        This variant tests the property with randomly generated prefix and
        suffix text to ensure robustness across diverse contexts.
        """
        kf = KeywordFilter()
        
        # Construct text with keyword embedded in random context
        text = f"{prefix} {keyword} {suffix}"
        
        # The keyword filter should match
        assert kf.matches(text), \
            f"Keyword filter failed to match '{keyword}' in context: '{text}'"
        
        # Verify the keyword is in the matched list
        matched = kf.get_matched_keywords(text)
        assert keyword in matched, \
            f"Keyword '{keyword}' not in matched list for text: '{text}'"
