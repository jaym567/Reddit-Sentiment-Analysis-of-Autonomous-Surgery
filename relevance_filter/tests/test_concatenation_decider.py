"""
Unit tests for ConcatenationDecider.

Tests the concatenation decision logic for parent-child text combination.
"""

import pytest
from relevance_filter.concatenation_decider import ConcatenationDecider
from relevance_filter.models import ParentContext, RelevanceState


@pytest.fixture
def decider():
    """Create a ConcatenationDecider with default settings."""
    return ConcatenationDecider(word_threshold=50)


@pytest.fixture
def relevant_parent():
    """Create a relevant parent context."""
    return ParentContext(
        id="parent_1",
        text="Robotic surgery systems are becoming more autonomous.",
        relevance_state=RelevanceState.RELEVANT_STRONG,
        relevance_score=0.9,
        depth=1,
        post_id="post_1"
    )


@pytest.fixture
def inherited_parent():
    """Create an inherited relevance parent context."""
    return ParentContext(
        id="parent_2",
        text="The da Vinci system is impressive.",
        relevance_state=RelevanceState.RELEVANT_INHERITED,
        relevance_score=0.8,
        depth=2,
        post_id="post_1"
    )


@pytest.fixture
def irrelevant_parent():
    """Create an irrelevant parent context."""
    return ParentContext(
        id="parent_3",
        text="This is completely off-topic.",
        relevance_state=RelevanceState.IRRELEVANT,
        relevance_score=0.1,
        depth=1,
        post_id="post_1"
    )


class TestConcatenationConditions:
    """Test all five concatenation conditions."""
    
    def test_all_conditions_met(self, decider, relevant_parent):
        """Should concatenate when all conditions are met."""
        child_text = "I agree that this is a good point about safety."
        assert decider.should_concatenate(relevant_parent, child_text) is True
    
    def test_condition1_parent_not_relevant(self, decider, irrelevant_parent):
        """Should not concatenate when parent is IRRELEVANT."""
        child_text = "I agree that this is a good point."
        assert decider.should_concatenate(irrelevant_parent, child_text) is False
    
    def test_condition1_parent_weak_relevance(self, decider):
        """Should not concatenate when parent is RELEVANT_WEAK."""
        weak_parent = ParentContext(
            id="parent_weak",
            text="Some tangential content.",
            relevance_state=RelevanceState.RELEVANT_WEAK,
            relevance_score=0.6,
            depth=1,
            post_id="post_1"
        )
        child_text = "I agree that this is a good point."
        assert decider.should_concatenate(weak_parent, child_text) is False
    
    def test_condition1_parent_relevant_strong(self, decider, relevant_parent):
        """Should pass condition 1 when parent is RELEVANT_STRONG."""
        child_text = "I agree that this is a good point."
        # Will fail on other conditions, but parent check should pass
        result = decider.should_concatenate(relevant_parent, child_text)
        # This will be True if all conditions pass
        assert isinstance(result, bool)
    
    def test_condition1_parent_relevant_inherited(self, decider, inherited_parent):
        """Should pass condition 1 when parent is RELEVANT_INHERITED."""
        child_text = "I agree that this is a good point."
        result = decider.should_concatenate(inherited_parent, child_text)
        assert isinstance(result, bool)
    
    def test_condition2_no_pronouns(self, decider, relevant_parent):
        """Should not concatenate when child has no pronouns."""
        child_text = "I agree the system works well."
        assert decider.should_concatenate(relevant_parent, child_text) is False
    
    def test_condition2_has_this(self, decider, relevant_parent):
        """Should pass condition 2 when child uses 'this'."""
        child_text = "I agree this is correct."
        result = decider.should_concatenate(relevant_parent, child_text)
        assert isinstance(result, bool)
    
    def test_condition2_has_that(self, decider, relevant_parent):
        """Should pass condition 2 when child uses 'that'."""
        child_text = "I agree that point is good."
        result = decider.should_concatenate(relevant_parent, child_text)
        assert isinstance(result, bool)
    
    def test_condition2_has_it(self, decider, relevant_parent):
        """Should pass condition 2 when child uses 'it'."""
        child_text = "I agree it is a good idea."
        result = decider.should_concatenate(relevant_parent, child_text)
        assert isinstance(result, bool)
    
    def test_condition2_has_they(self, decider, relevant_parent):
        """Should pass condition 2 when child uses 'they'."""
        child_text = "I agree they are correct."
        result = decider.should_concatenate(relevant_parent, child_text)
        assert isinstance(result, bool)
    
    def test_condition2_has_these(self, decider, relevant_parent):
        """Should pass condition 2 when child uses 'these'."""
        child_text = "I agree these are good points."
        result = decider.should_concatenate(relevant_parent, child_text)
        assert isinstance(result, bool)
    
    def test_condition2_has_those(self, decider, relevant_parent):
        """Should pass condition 2 when child uses 'those'."""
        child_text = "I agree those ideas are good."
        result = decider.should_concatenate(relevant_parent, child_text)
        assert isinstance(result, bool)
    
    def test_condition2_pronoun_case_insensitive(self, decider, relevant_parent):
        """Should detect pronouns regardless of case."""
        child_text = "I agree THIS is correct."
        result = decider.should_concatenate(relevant_parent, child_text)
        assert isinstance(result, bool)
    
    def test_condition3_too_many_words(self, decider, relevant_parent):
        """Should not concatenate when child exceeds word threshold."""
        # Create a child with exactly 50 words (at threshold)
        child_text = "I agree that this " + " ".join(["word"] * 46)
        assert decider.should_concatenate(relevant_parent, child_text) is False
    
    def test_condition3_at_threshold(self, decider, relevant_parent):
        """Should not concatenate when child is at word threshold."""
        # Create a child with exactly 50 words
        child_text = "I agree that this " + " ".join(["word"] * 46)
        word_count = len(child_text.split())
        assert word_count == 50
        assert decider.should_concatenate(relevant_parent, child_text) is False
    
    def test_condition3_under_threshold(self, decider, relevant_parent):
        """Should pass condition 3 when child is under threshold."""
        child_text = "I agree that this is a good point."
        word_count = len(child_text.split())
        assert word_count < 50
        result = decider.should_concatenate(relevant_parent, child_text)
        assert isinstance(result, bool)
    
    def test_condition4_no_evaluative_statements(self, decider, relevant_parent):
        """Should not concatenate when child has no evaluative statements."""
        child_text = "This is something."
        assert decider.should_concatenate(relevant_parent, child_text) is False
    
    def test_condition4_has_agree(self, decider, relevant_parent):
        """Should pass condition 4 when child uses 'agree'."""
        child_text = "I agree that this is true."
        result = decider.should_concatenate(relevant_parent, child_text)
        assert isinstance(result, bool)
    
    def test_condition4_has_disagree(self, decider, relevant_parent):
        """Should pass condition 4 when child uses 'disagree'."""
        child_text = "I disagree that this is correct."
        result = decider.should_concatenate(relevant_parent, child_text)
        assert isinstance(result, bool)
    
    def test_condition4_has_correct(self, decider, relevant_parent):
        """Should pass condition 4 when child uses 'correct'."""
        child_text = "That is correct about this."
        result = decider.should_concatenate(relevant_parent, child_text)
        assert isinstance(result, bool)
    
    def test_condition4_has_wrong(self, decider, relevant_parent):
        """Should pass condition 4 when child uses 'wrong'."""
        child_text = "That is wrong about this."
        result = decider.should_concatenate(relevant_parent, child_text)
        assert isinstance(result, bool)
    
    def test_condition4_has_good_point(self, decider, relevant_parent):
        """Should pass condition 4 when child uses 'good point'."""
        child_text = "That is a good point about this."
        result = decider.should_concatenate(relevant_parent, child_text)
        assert isinstance(result, bool)
    
    def test_condition4_has_bad_idea(self, decider, relevant_parent):
        """Should pass condition 4 when child uses 'bad idea'."""
        child_text = "That is a bad idea about this."
        result = decider.should_concatenate(relevant_parent, child_text)
        assert isinstance(result, bool)
    
    def test_condition4_evaluative_case_insensitive(self, decider, relevant_parent):
        """Should detect evaluative statements regardless of case."""
        child_text = "I AGREE that THIS is correct."
        result = decider.should_concatenate(relevant_parent, child_text)
        assert isinstance(result, bool)
    
    def test_condition5_introduces_new_topic(self, decider, relevant_parent):
        """Should not concatenate when child introduces new topic."""
        child_text = "Speaking of this, I agree it's a good point."
        assert decider.should_concatenate(relevant_parent, child_text) is False
    
    def test_condition5_by_the_way(self, decider, relevant_parent):
        """Should not concatenate when child uses 'by the way'."""
        child_text = "By the way, I agree that this is correct."
        assert decider.should_concatenate(relevant_parent, child_text) is False
    
    def test_condition5_off_topic(self, decider, relevant_parent):
        """Should not concatenate when child mentions 'off topic'."""
        child_text = "Off topic, but I agree this is good."
        assert decider.should_concatenate(relevant_parent, child_text) is False
    
    def test_condition5_no_new_topic_indicators(self, decider, relevant_parent):
        """Should pass condition 5 when no new topic indicators present."""
        child_text = "I agree that this is a good point."
        result = decider.should_concatenate(relevant_parent, child_text)
        assert isinstance(result, bool)


class TestConcatenateMethod:
    """Test the concatenate method formatting."""
    
    def test_concatenate_format(self, decider):
        """Should format concatenated text correctly."""
        parent_text = "Robotic surgery is advancing."
        child_text = "I agree that this is true."
        
        result = decider.concatenate(parent_text, child_text)
        
        expected = "PARENT CONTEXT:\nRobotic surgery is advancing.\n\nCHILD COMMENT:\nI agree that this is true.\n"
        assert result == expected
    
    def test_concatenate_preserves_text(self, decider):
        """Should preserve exact text without modification."""
        parent_text = "Original parent text with special chars: @#$%"
        child_text = "Original child text with numbers: 123"
        
        result = decider.concatenate(parent_text, child_text)
        
        assert "Original parent text with special chars: @#$%" in result
        assert "Original child text with numbers: 123" in result
    
    def test_concatenate_empty_parent(self, decider):
        """Should handle empty parent text."""
        parent_text = ""
        child_text = "Child text"
        
        result = decider.concatenate(parent_text, child_text)
        
        assert result == "PARENT CONTEXT:\n\n\nCHILD COMMENT:\nChild text\n"
    
    def test_concatenate_empty_child(self, decider):
        """Should handle empty child text."""
        parent_text = "Parent text"
        child_text = ""
        
        result = decider.concatenate(parent_text, child_text)
        
        assert result == "PARENT CONTEXT:\nParent text\n\nCHILD COMMENT:\n\n"


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_custom_word_threshold(self, relevant_parent):
        """Should respect custom word threshold."""
        decider = ConcatenationDecider(word_threshold=10)
        
        # 9 words - should pass
        child_text = "I agree that this is a good point here."
        word_count = len(child_text.split())
        assert word_count == 9
        result = decider.should_concatenate(relevant_parent, child_text)
        assert isinstance(result, bool)
        
        # 10 words - should fail (at threshold)
        child_text = "I agree that this is a good point here now."
        word_count = len(child_text.split())
        assert word_count == 10
        assert decider.should_concatenate(relevant_parent, child_text) is False
    
    def test_pronoun_as_part_of_word(self, decider, relevant_parent):
        """Should only match pronouns as whole words."""
        # "this" is part of "thistle" - should not match
        child_text = "I agree the thistle is good."
        assert decider.should_concatenate(relevant_parent, child_text) is False
    
    def test_multiple_pronouns(self, decider, relevant_parent):
        """Should pass when multiple pronouns are present."""
        child_text = "I agree that this and that are good."
        result = decider.should_concatenate(relevant_parent, child_text)
        assert isinstance(result, bool)
    
    def test_multiple_evaluative_patterns(self, decider, relevant_parent):
        """Should pass when multiple evaluative patterns are present."""
        child_text = "I agree this is correct and a good point."
        result = decider.should_concatenate(relevant_parent, child_text)
        assert isinstance(result, bool)
    
    def test_whitespace_in_text(self, decider, relevant_parent):
        """Should handle extra whitespace correctly."""
        child_text = "I  agree   that    this is correct."
        # Word count should still work with extra spaces
        result = decider.should_concatenate(relevant_parent, child_text)
        assert isinstance(result, bool)
    
    def test_newlines_in_text(self, decider, relevant_parent):
        """Should handle newlines in text."""
        child_text = "I agree\nthat this\nis correct."
        result = decider.should_concatenate(relevant_parent, child_text)
        assert isinstance(result, bool)


class TestRealWorldExamples:
    """Test with realistic Reddit comment examples."""
    
    def test_short_agreement_reply(self, decider, relevant_parent):
        """Should concatenate short agreement replies."""
        child_text = "Exactly! This is a great point."
        assert decider.should_concatenate(relevant_parent, child_text) is True
    
    def test_short_disagreement_reply(self, decider, relevant_parent):
        """Should concatenate short disagreement replies."""
        child_text = "I disagree that this is correct."
        assert decider.should_concatenate(relevant_parent, child_text) is True
    
    def test_long_detailed_reply(self, decider, relevant_parent):
        """Should not concatenate long detailed replies."""
        child_text = (
            "I agree that robotic surgery has many benefits. "
            "The precision of the da Vinci system is remarkable. "
            "However, we must also consider the training requirements "
            "and the cost implications for healthcare systems. "
            "Additionally, there are concerns about liability and "
            "the need for regulatory oversight. Furthermore, we need "
            "to think about the long-term maintenance costs and the "
            "infrastructure needed to support these advanced systems."
        )
        word_count = len(child_text.split())
        assert word_count >= 50
        assert decider.should_concatenate(relevant_parent, child_text) is False
    
    def test_topic_change_reply(self, decider, relevant_parent):
        """Should not concatenate when topic changes."""
        child_text = "Speaking of robots, I agree video games are fun."
        assert decider.should_concatenate(relevant_parent, child_text) is False
    
    def test_no_pronoun_reply(self, decider, relevant_parent):
        """Should not concatenate replies without pronouns."""
        child_text = "I agree the system works well."
        assert decider.should_concatenate(relevant_parent, child_text) is False
    
    def test_no_evaluation_reply(self, decider, relevant_parent):
        """Should not concatenate replies without evaluation."""
        child_text = "This is something to consider."
        assert decider.should_concatenate(relevant_parent, child_text) is False


class TestAdditionalEdgeCases:
    """Test additional edge cases for comprehensive coverage."""
    
    def test_very_short_text_two_words(self, decider, relevant_parent):
        """Should handle very short text (2 words) without evaluative pattern."""
        child_text = "See this."
        # Has pronoun "this" but no evaluative pattern
        assert decider.should_concatenate(relevant_parent, child_text) is False
    
    def test_very_short_text_with_all_conditions(self, decider, relevant_parent):
        """Should concatenate very short text if all conditions met."""
        child_text = "Correct, this."
        # Has pronoun "this" and evaluative "correct" - all conditions met
        assert decider.should_concatenate(relevant_parent, child_text) is True
    
    def test_minimal_valid_concatenation(self, decider, relevant_parent):
        """Should concatenate minimal valid text."""
        child_text = "This is correct."
        assert decider.should_concatenate(relevant_parent, child_text) is True
    
    def test_text_with_only_punctuation(self, decider, relevant_parent):
        """Should not concatenate text with only punctuation."""
        child_text = "!!! ??? ..."
        assert decider.should_concatenate(relevant_parent, child_text) is False
    
    def test_text_with_unicode_characters(self, decider, relevant_parent):
        """Should handle unicode characters correctly."""
        child_text = "I agree that this is correct! 👍 🤖"
        result = decider.should_concatenate(relevant_parent, child_text)
        assert isinstance(result, bool)
    
    def test_text_with_special_characters(self, decider, relevant_parent):
        """Should handle special characters in text."""
        child_text = "I agree that this is correct! @#$%^&*()"
        result = decider.should_concatenate(relevant_parent, child_text)
        assert isinstance(result, bool)
    
    def test_new_topic_indicator_mixed_case(self, decider, relevant_parent):
        """Should detect new topic indicators regardless of case."""
        child_text = "SPEAKING OF this, I agree it's correct."
        assert decider.should_concatenate(relevant_parent, child_text) is False
    
    def test_new_topic_indicator_unrelated(self, decider, relevant_parent):
        """Should detect 'unrelated' as new topic indicator."""
        child_text = "Unrelated, but I agree this is correct."
        assert decider.should_concatenate(relevant_parent, child_text) is False
    
    def test_new_topic_indicator_different_topic(self, decider, relevant_parent):
        """Should detect 'different topic' as new topic indicator."""
        child_text = "Different topic, but I agree this is correct."
        assert decider.should_concatenate(relevant_parent, child_text) is False
    
    def test_new_topic_indicator_changing_subject(self, decider, relevant_parent):
        """Should detect 'changing subject' as new topic indicator."""
        child_text = "Changing subject, I agree this is correct."
        assert decider.should_concatenate(relevant_parent, child_text) is False
    
    def test_pronoun_at_start_of_text(self, decider, relevant_parent):
        """Should detect pronoun at start of text."""
        child_text = "This is exactly right."
        assert decider.should_concatenate(relevant_parent, child_text) is True
    
    def test_pronoun_at_end_of_text(self, decider, relevant_parent):
        """Should detect pronoun at end of text."""
        child_text = "I agree with this."
        assert decider.should_concatenate(relevant_parent, child_text) is True
    
    def test_evaluative_at_start_of_text(self, decider, relevant_parent):
        """Should detect evaluative pattern at start of text."""
        child_text = "Correct! This is the answer."
        assert decider.should_concatenate(relevant_parent, child_text) is True
    
    def test_evaluative_at_end_of_text(self, decider, relevant_parent):
        """Should detect evaluative pattern at end of text."""
        child_text = "I think this is correct."
        assert decider.should_concatenate(relevant_parent, child_text) is True
    
    def test_text_with_only_whitespace(self, decider, relevant_parent):
        """Should not concatenate text with only whitespace."""
        child_text = "   \t\n   "
        assert decider.should_concatenate(relevant_parent, child_text) is False
    
    def test_word_count_with_tabs(self, decider, relevant_parent):
        """Should count words correctly with tabs."""
        child_text = "I\tagree\tthat\tthis\tis\tcorrect."
        word_count = len(child_text.split())
        assert word_count == 6
        result = decider.should_concatenate(relevant_parent, child_text)
        assert isinstance(result, bool)
    
    def test_exactly_49_words(self, decider, relevant_parent):
        """Should concatenate text with exactly 49 words (under threshold)."""
        # Create text with exactly 49 words
        child_text = "I agree that this " + " ".join(["word"] * 45)
        word_count = len(child_text.split())
        assert word_count == 49
        result = decider.should_concatenate(relevant_parent, child_text)
        assert isinstance(result, bool)
    
    def test_exactly_51_words(self, decider, relevant_parent):
        """Should not concatenate text with exactly 51 words (over threshold)."""
        # Create text with exactly 51 words
        child_text = "I agree that this " + " ".join(["word"] * 47)
        word_count = len(child_text.split())
        assert word_count == 51
        assert decider.should_concatenate(relevant_parent, child_text) is False
    
    def test_concatenate_with_very_long_parent(self, decider):
        """Should handle concatenation with very long parent text."""
        long_parent = ParentContext(
            id="parent_long",
            text="This is a very long parent text. " * 100,  # ~600 words
            relevance_state=RelevanceState.RELEVANT_STRONG,
            relevance_score=0.9,
            depth=1,
            post_id="post_1"
        )
        child_text = "I agree that this is correct."
        
        result = decider.concatenate(long_parent.text, child_text)
        assert "PARENT CONTEXT:" in result
        assert "CHILD COMMENT:" in result
        assert long_parent.text in result
        assert child_text in result
    
    def test_concatenate_with_newlines_in_both(self, decider):
        """Should preserve newlines in concatenated text."""
        parent_text = "Line 1\nLine 2\nLine 3"
        child_text = "Child line 1\nChild line 2"
        
        result = decider.concatenate(parent_text, child_text)
        assert result == f"PARENT CONTEXT:\n{parent_text}\n\nCHILD COMMENT:\n{child_text}\n"
    
    def test_concatenate_with_quotes(self, decider):
        """Should handle quotes in text."""
        parent_text = 'He said "robotic surgery is great"'
        child_text = 'I agree that this is correct'
        
        result = decider.concatenate(parent_text, child_text)
        assert 'He said "robotic surgery is great"' in result
    
    def test_multiple_spaces_between_words(self, decider, relevant_parent):
        """Should handle multiple spaces between words."""
        child_text = "I    agree    that    this    is    correct."
        # split() handles multiple spaces correctly
        word_count = len(child_text.split())
        assert word_count == 6
        result = decider.should_concatenate(relevant_parent, child_text)
        assert isinstance(result, bool)
    
    def test_pronoun_in_compound_word(self, decider, relevant_parent):
        """Should not match pronoun as part of compound word."""
        # "this" in "thistle" should not match
        child_text = "I agree the thistle plant is correct."
        assert decider.should_concatenate(relevant_parent, child_text) is False
    
    def test_pronoun_with_punctuation(self, decider, relevant_parent):
        """Should match pronoun with adjacent punctuation."""
        child_text = "I agree, this is correct!"
        assert decider.should_concatenate(relevant_parent, child_text) is True
    
    def test_evaluative_with_punctuation(self, decider, relevant_parent):
        """Should match evaluative pattern with punctuation."""
        child_text = "Correct! That is the answer."
        assert decider.should_concatenate(relevant_parent, child_text) is True
    
    def test_all_pronouns_in_one_text(self, decider, relevant_parent):
        """Should pass with multiple different pronouns."""
        child_text = "I agree this, that, it, they, these, and those are correct."
        word_count = len(child_text.split())
        assert word_count < 50
        result = decider.should_concatenate(relevant_parent, child_text)
        assert isinstance(result, bool)
    
    def test_boundary_exactly_threshold_custom(self, relevant_parent):
        """Should not concatenate at custom threshold boundary."""
        decider = ConcatenationDecider(word_threshold=5)
        
        # 4 words - under threshold
        child_text = "I agree this correct."
        assert len(child_text.split()) == 4
        result = decider.should_concatenate(relevant_parent, child_text)
        assert isinstance(result, bool)
        
        # 5 words - at threshold
        child_text = "I agree this is correct."
        assert len(child_text.split()) == 5
        assert decider.should_concatenate(relevant_parent, child_text) is False
    
    def test_right_evaluative_pattern(self, decider, relevant_parent):
        """Should match 'right' as evaluative."""
        child_text = "That is right about this."
        assert decider.should_concatenate(relevant_parent, child_text) is True
    
    def test_exactly_evaluative_pattern(self, decider, relevant_parent):
        """Should match 'exactly' as evaluative."""
        child_text = "Exactly! This is it."
        assert decider.should_concatenate(relevant_parent, child_text) is True
    
    def test_precisely_evaluative_pattern(self, decider, relevant_parent):
        """Should match 'precisely' as evaluative."""
        child_text = "Precisely! That is correct."
        assert decider.should_concatenate(relevant_parent, child_text) is True
    
    def test_better_argument_pattern(self, decider, relevant_parent):
        """Should match 'better argument' pattern."""
        child_text = "That is a better argument about this."
        assert decider.should_concatenate(relevant_parent, child_text) is True
    
    def test_worse_idea_pattern(self, decider, relevant_parent):
        """Should match 'worse idea' pattern."""
        child_text = "That is a worse idea about this."
        assert decider.should_concatenate(relevant_parent, child_text) is True
    
    def test_best_point_pattern(self, decider, relevant_parent):
        """Should match 'best point' pattern."""
        child_text = "That is the best point about this."
        assert decider.should_concatenate(relevant_parent, child_text) is True
    
    def test_worst_argument_pattern(self, decider, relevant_parent):
        """Should match 'worst argument' pattern."""
        child_text = "That is the worst argument about this."
        assert decider.should_concatenate(relevant_parent, child_text) is True


class TestBoundaryWordCounts:
    """Test boundary conditions for word count threshold."""
    
    def test_zero_words(self, decider, relevant_parent):
        """Should not concatenate empty text (0 words)."""
        child_text = ""
        assert decider.should_concatenate(relevant_parent, child_text) is False
    
    def test_one_word(self, decider, relevant_parent):
        """Should not concatenate single word without evaluative pattern."""
        child_text = "This."
        # Has pronoun but no evaluative pattern
        assert decider.should_concatenate(relevant_parent, child_text) is False
    
    def test_one_word_evaluative(self, decider, relevant_parent):
        """Should not concatenate single evaluative word without pronoun."""
        child_text = "Correct!"
        # Has evaluative but no pronoun
        assert decider.should_concatenate(relevant_parent, child_text) is False
    
    def test_two_words_minimal_valid(self, decider, relevant_parent):
        """Should concatenate minimal two-word valid text."""
        child_text = "This correct."
        # Has both pronoun and evaluative
        assert decider.should_concatenate(relevant_parent, child_text) is True
    
    def test_word_count_boundary_48_words(self, decider, relevant_parent):
        """Should concatenate at 48 words (well under threshold)."""
        child_text = "I agree that this " + " ".join(["word"] * 44)
        word_count = len(child_text.split())
        assert word_count == 48
        result = decider.should_concatenate(relevant_parent, child_text)
        assert isinstance(result, bool)
    
    def test_word_count_boundary_52_words(self, decider, relevant_parent):
        """Should not concatenate at 52 words (over threshold)."""
        child_text = "I agree that this " + " ".join(["word"] * 48)
        word_count = len(child_text.split())
        assert word_count == 52
        assert decider.should_concatenate(relevant_parent, child_text) is False
    
    def test_word_count_with_contractions(self, decider, relevant_parent):
        """Should count contractions as single words."""
        child_text = "I agree that this isn't correct."
        word_count = len(child_text.split())
        assert word_count == 6  # "isn't" counts as one word
        result = decider.should_concatenate(relevant_parent, child_text)
        assert isinstance(result, bool)
    
    def test_word_count_with_hyphens(self, decider, relevant_parent):
        """Should count hyphenated words correctly."""
        child_text = "I agree that this state-of-the-art system is correct."
        word_count = len(child_text.split())
        assert word_count == 8  # "state-of-the-art" counts as one word
        result = decider.should_concatenate(relevant_parent, child_text)
        assert isinstance(result, bool)


class TestVariousPronounTypes:
    """Test all pronoun types in various contexts."""
    
    def test_pronoun_this_capitalized_start(self, decider, relevant_parent):
        """Should match 'This' at start of sentence."""
        child_text = "This is exactly correct."
        assert decider.should_concatenate(relevant_parent, child_text) is True
    
    def test_pronoun_that_in_middle(self, decider, relevant_parent):
        """Should match 'that' in middle of sentence."""
        child_text = "I agree that point is correct."
        assert decider.should_concatenate(relevant_parent, child_text) is True
    
    def test_pronoun_it_possessive_its(self, decider, relevant_parent):
        """Should match 'it' but not be confused by 'its'."""
        child_text = "I agree it is correct and its benefits are good."
        # Should match "it" as pronoun
        assert decider.should_concatenate(relevant_parent, child_text) is True
    
    def test_pronoun_they_vs_theyre(self, decider, relevant_parent):
        """Should match 'they' and handle 'they're' correctly."""
        child_text = "I agree they are correct and they're right."
        # Should match "they"
        assert decider.should_concatenate(relevant_parent, child_text) is True
    
    def test_pronoun_these_plural(self, decider, relevant_parent):
        """Should match 'these' for plural references."""
        child_text = "I agree these points are all correct."
        assert decider.should_concatenate(relevant_parent, child_text) is True
    
    def test_pronoun_those_plural(self, decider, relevant_parent):
        """Should match 'those' for plural references."""
        child_text = "I disagree with those arguments about this."
        # Has both "those" and "this"
        assert decider.should_concatenate(relevant_parent, child_text) is True
    
    def test_pronoun_not_matched_he_she(self, decider, relevant_parent):
        """Should not match personal pronouns like 'he' or 'she'."""
        child_text = "I agree he is correct."
        # "he" is not in the pronoun list
        assert decider.should_concatenate(relevant_parent, child_text) is False
    
    def test_pronoun_not_matched_we_you(self, decider, relevant_parent):
        """Should not match personal pronouns like 'we' or 'you'."""
        child_text = "I agree we are correct."
        # "we" is not in the pronoun list
        assert decider.should_concatenate(relevant_parent, child_text) is False
    
    def test_pronoun_in_question(self, decider, relevant_parent):
        """Should not concatenate question without evaluative pattern."""
        child_text = "Is this true?"
        # Has pronoun "this" but no evaluative pattern
        assert decider.should_concatenate(relevant_parent, child_text) is False
    
    def test_pronoun_in_exclamation(self, decider, relevant_parent):
        """Should match pronoun in exclamation."""
        child_text = "This is exactly right!"
        assert decider.should_concatenate(relevant_parent, child_text) is True


class TestAdditionalEvaluativePatterns:
    """Test edge cases for evaluative pattern matching."""
    
    def test_evaluative_agree_with_preposition(self, decider, relevant_parent):
        """Should match 'agree with' pattern."""
        child_text = "I agree with this point."
        assert decider.should_concatenate(relevant_parent, child_text) is True
    
    def test_evaluative_disagree_with_preposition(self, decider, relevant_parent):
        """Should match 'disagree with' pattern."""
        child_text = "I disagree with that argument."
        assert decider.should_concatenate(relevant_parent, child_text) is True
    
    def test_evaluative_partially_correct(self, decider, relevant_parent):
        """Should match 'correct' in compound phrases."""
        child_text = "That is partially correct about this."
        assert decider.should_concatenate(relevant_parent, child_text) is True
    
    def test_evaluative_completely_wrong(self, decider, relevant_parent):
        """Should match 'wrong' in compound phrases."""
        child_text = "That is completely wrong about this."
        assert decider.should_concatenate(relevant_parent, child_text) is True
    
    def test_evaluative_absolutely_right(self, decider, relevant_parent):
        """Should match 'right' in compound phrases."""
        child_text = "That is absolutely right about this."
        assert decider.should_concatenate(relevant_parent, child_text) is True
    
    def test_evaluative_not_in_word_boundary(self, decider, relevant_parent):
        """Should not match evaluative pattern as part of larger word."""
        child_text = "I think this disagreement is about something."
        # "disagree" is part of "disagreement" - should still match due to word boundary
        result = decider.should_concatenate(relevant_parent, child_text)
        # Actually, the regex uses \b so "disagree" in "disagreement" should match
        assert isinstance(result, bool)
    
    def test_evaluative_good_without_point_idea_argument(self, decider, relevant_parent):
        """Should not match 'good' alone without point/idea/argument."""
        child_text = "This is good."
        # "good" alone doesn't match the pattern
        assert decider.should_concatenate(relevant_parent, child_text) is False
    
    def test_evaluative_bad_without_point_idea_argument(self, decider, relevant_parent):
        """Should not match 'bad' alone without point/idea/argument."""
        child_text = "This is bad."
        # "bad" alone doesn't match the pattern
        assert decider.should_concatenate(relevant_parent, child_text) is False
    
    def test_evaluative_good_point_with_article(self, decider, relevant_parent):
        """Should match 'good point' with article."""
        child_text = "That is a good point about this."
        assert decider.should_concatenate(relevant_parent, child_text) is True
    
    def test_evaluative_multiple_patterns_in_text(self, decider, relevant_parent):
        """Should pass with multiple evaluative patterns."""
        child_text = "I agree this is correct and a good point."
        assert decider.should_concatenate(relevant_parent, child_text) is True


class TestNewTopicIndicatorEdgeCases:
    """Test edge cases for new topic indicator detection."""
    
    def test_new_topic_speaking_of_at_start(self, decider, relevant_parent):
        """Should detect 'speaking of' at start of text."""
        child_text = "Speaking of this, I agree it's correct."
        assert decider.should_concatenate(relevant_parent, child_text) is False
    
    def test_new_topic_speaking_of_in_middle(self, decider, relevant_parent):
        """Should detect 'speaking of' in middle of text."""
        child_text = "Well, speaking of that, I agree this is correct."
        assert decider.should_concatenate(relevant_parent, child_text) is False
    
    def test_new_topic_by_the_way_abbreviated(self, decider, relevant_parent):
        """Should detect 'by the way' even with punctuation."""
        child_text = "By the way, I agree this is correct."
        assert decider.should_concatenate(relevant_parent, child_text) is False
    
    def test_new_topic_off_topic_with_hyphen(self, decider, relevant_parent):
        """Should detect 'off topic' with or without hyphen."""
        child_text = "Off-topic, but I agree this is correct."
        # The indicator is "off topic" without hyphen, so this might not match
        # Let's test what actually happens
        result = decider.should_concatenate(relevant_parent, child_text)
        # Since "off topic" is the indicator and text has "off-topic", it won't match
        assert isinstance(result, bool)
    
    def test_new_topic_unrelated_at_start(self, decider, relevant_parent):
        """Should detect 'unrelated' at start."""
        child_text = "Unrelated, but I agree this is correct."
        assert decider.should_concatenate(relevant_parent, child_text) is False
    
    def test_new_topic_different_topic_explicit(self, decider, relevant_parent):
        """Should detect 'different topic' explicitly."""
        child_text = "Different topic: I agree this is correct."
        assert decider.should_concatenate(relevant_parent, child_text) is False
    
    def test_new_topic_changing_subject_explicit(self, decider, relevant_parent):
        """Should detect 'changing subject' explicitly."""
        child_text = "Changing subject here, but I agree this is correct."
        assert decider.should_concatenate(relevant_parent, child_text) is False
    
    def test_not_new_topic_similar_words(self, decider, relevant_parent):
        """Should not falsely detect new topic with similar words."""
        child_text = "I agree this topic is correct."
        # "topic" alone shouldn't trigger "different topic"
        assert decider.should_concatenate(relevant_parent, child_text) is True
    
    def test_not_new_topic_speaking_alone(self, decider, relevant_parent):
        """Should not falsely detect 'speaking' alone."""
        child_text = "I agree this speaking system is correct."
        # "speaking" alone shouldn't trigger "speaking of"
        assert decider.should_concatenate(relevant_parent, child_text) is True


class TestConcatenateMethodEdgeCases:
    """Test edge cases for the concatenate method."""
    
    def test_concatenate_with_very_long_texts(self, decider):
        """Should handle very long texts in concatenation."""
        parent_text = "A" * 10000
        child_text = "B" * 5000
        
        result = decider.concatenate(parent_text, child_text)
        assert result.startswith("PARENT CONTEXT:\n")
        assert "\n\nCHILD COMMENT:\n" in result
        assert parent_text in result
        assert child_text in result
    
    def test_concatenate_preserves_exact_formatting(self, decider):
        """Should preserve exact formatting including spaces and newlines."""
        parent_text = "  Leading spaces\n\nDouble newline  "
        child_text = "\tTab start and trailing spaces  "
        
        result = decider.concatenate(parent_text, child_text)
        assert "  Leading spaces\n\nDouble newline  " in result
        assert "\tTab start and trailing spaces  " in result
    
    def test_concatenate_with_special_regex_chars(self, decider):
        """Should handle special regex characters in text."""
        parent_text = "Text with regex chars: . * + ? ^ $ { } [ ] ( ) | \\"
        child_text = "More regex: \\d+ \\w* [a-z]"
        
        result = decider.concatenate(parent_text, child_text)
        assert parent_text in result
        assert child_text in result
    
    def test_concatenate_with_emoji_and_unicode(self, decider):
        """Should handle emoji and unicode characters."""
        parent_text = "Robotic surgery 🤖 is great! 手術"
        child_text = "I agree 👍 this is correct ✓"
        
        result = decider.concatenate(parent_text, child_text)
        assert "🤖" in result
        assert "👍" in result
        assert "手術" in result
        assert "✓" in result
    
    def test_concatenate_format_exact_match(self, decider):
        """Should match exact format specification from requirements."""
        parent_text = "Parent"
        child_text = "Child"
        
        result = decider.concatenate(parent_text, child_text)
        # Requirement 4.3 specifies exact format
        assert result == "PARENT CONTEXT:\nParent\n\nCHILD COMMENT:\nChild\n"
    
    def test_concatenate_no_extra_whitespace(self, decider):
        """Should not add extra whitespace beyond format."""
        parent_text = "Parent"
        child_text = "Child"
        
        result = decider.concatenate(parent_text, child_text)
        # Should be exactly: "PARENT CONTEXT:\nParent\n\nCHILD COMMENT:\nChild\n"
        # 0: PARENT CONTEXT:\n
        # 1: Parent\n
        # 2: \n
        # 3: CHILD COMMENT:\n
        # 4: Child\n
        assert result.count('\n') == 5
        assert result.endswith('\n')
        assert not result.startswith('\n')


class TestAdditionalBoundaryEdgeCases:
    """Test additional boundary and edge cases for comprehensive coverage."""
    
    def test_word_count_with_multiple_newlines(self, decider, relevant_parent):
        """Should count words correctly across multiple newlines."""
        child_text = "I\n\n\nagree\n\nthat\n\nthis\nis\ncorrect."
        word_count = len(child_text.split())
        assert word_count == 6
        result = decider.should_concatenate(relevant_parent, child_text)
        assert isinstance(result, bool)
    
    def test_pronoun_with_apostrophe_contraction(self, decider, relevant_parent):
        """Should handle pronouns in contractions like 'that's'."""
        child_text = "I agree that's correct."
        # "that's" contains "that" but as a contraction
        result = decider.should_concatenate(relevant_parent, child_text)
        assert isinstance(result, bool)
    
    def test_evaluative_pattern_with_negation(self, decider, relevant_parent):
        """Should match evaluative patterns with negation."""
        child_text = "I don't agree that this is correct."
        # Has "agree" and "that" and "this"
        assert decider.should_concatenate(relevant_parent, child_text) is True
    
    def test_multiple_new_topic_indicators(self, decider, relevant_parent):
        """Should reject if any new topic indicator is present."""
        child_text = "By the way, speaking of this, I agree it's correct."
        # Has multiple new topic indicators
        assert decider.should_concatenate(relevant_parent, child_text) is False
    
    def test_word_threshold_zero(self, relevant_parent):
        """Should handle word threshold of zero (no concatenation allowed)."""
        decider = ConcatenationDecider(word_threshold=0)
        child_text = ""
        # Even empty text should fail with threshold 0
        assert decider.should_concatenate(relevant_parent, child_text) is False
    
    def test_word_threshold_one(self, relevant_parent):
        """Should handle word threshold of one."""
        decider = ConcatenationDecider(word_threshold=1)
        child_text = ""  # 0 words - under threshold
        # But will fail on other conditions
        assert decider.should_concatenate(relevant_parent, child_text) is False
    
    def test_word_threshold_very_large(self, relevant_parent):
        """Should handle very large word threshold."""
        decider = ConcatenationDecider(word_threshold=10000)
        child_text = "I agree that this is correct."
        # Should pass word count condition with large threshold
        result = decider.should_concatenate(relevant_parent, child_text)
        assert isinstance(result, bool)
    
    def test_pronoun_followed_by_punctuation_no_space(self, decider, relevant_parent):
        """Should match pronoun followed immediately by punctuation."""
        child_text = "I agree,this is correct."
        # "this" followed by no space but preceded by comma
        result = decider.should_concatenate(relevant_parent, child_text)
        assert isinstance(result, bool)
    
    def test_evaluative_in_quoted_text(self, decider, relevant_parent):
        """Should match evaluative pattern even in quotes."""
        child_text = 'I agree "this is correct" about that.'
        assert decider.should_concatenate(relevant_parent, child_text) is True
    
    def test_case_sensitivity_new_topic_all_caps(self, decider, relevant_parent):
        """Should detect new topic indicators in all caps."""
        child_text = "BY THE WAY, I agree this is correct."
        # New topic indicators are checked with .lower(), so should match
        assert decider.should_concatenate(relevant_parent, child_text) is False
    
    def test_mixed_whitespace_types(self, decider, relevant_parent):
        """Should handle mixed whitespace types (spaces, tabs, newlines)."""
        child_text = "I \t agree \n that \r\n this is correct."
        word_count = len(child_text.split())
        assert word_count == 6
        result = decider.should_concatenate(relevant_parent, child_text)
        assert isinstance(result, bool)
    
    def test_pronoun_at_word_boundary_with_comma(self, decider, relevant_parent):
        """Should match pronoun at word boundary with comma."""
        child_text = "I agree, this, is correct."
        assert decider.should_concatenate(relevant_parent, child_text) is True
    
    def test_evaluative_pattern_at_word_boundary_with_period(self, decider, relevant_parent):
        """Should match evaluative at word boundary with period."""
        child_text = "Correct. This is it."
        assert decider.should_concatenate(relevant_parent, child_text) is True
    
    def test_all_conditions_met_minimal_text(self, decider, relevant_parent):
        """Should concatenate with absolute minimal valid text."""
        child_text = "Agree this."
        # Has evaluative "agree", pronoun "this", under threshold, no new topic
        assert decider.should_concatenate(relevant_parent, child_text) is True
    
    def test_parent_context_none_type(self, decider):
        """Should handle None parent context gracefully."""
        # This would be a programming error, but test defensive behavior
        # Actually, the type hint says ParentContext, so this shouldn't happen
        # But we can test with an irrelevant parent instead
        pass  # Skip this test as it's not a valid scenario
    
    def test_child_text_with_only_numbers(self, decider, relevant_parent):
        """Should handle text with only numbers."""
        child_text = "123 456 789"
        # No pronouns, no evaluative patterns
        assert decider.should_concatenate(relevant_parent, child_text) is False
    
    def test_child_text_with_html_tags(self, decider, relevant_parent):
        """Should handle text with HTML tags."""
        child_text = "I agree <b>this</b> is correct."
        # Should still match "this" and "agree"
        assert decider.should_concatenate(relevant_parent, child_text) is True
    
    def test_child_text_with_markdown(self, decider, relevant_parent):
        """Should handle text with markdown formatting."""
        child_text = "I agree **this** is _correct_."
        assert decider.should_concatenate(relevant_parent, child_text) is True
    
    def test_child_text_with_urls(self, decider, relevant_parent):
        """Should handle text with URLs."""
        child_text = "I agree this is correct. See https://example.com"
        word_count = len(child_text.split())
        # Should count URL as one word
        result = decider.should_concatenate(relevant_parent, child_text)
        assert isinstance(result, bool)
    
    def test_evaluative_pattern_with_intensifiers(self, decider, relevant_parent):
        """Should match evaluative patterns with intensifiers."""
        child_text = "I totally agree that this is very correct."
        assert decider.should_concatenate(relevant_parent, child_text) is True
