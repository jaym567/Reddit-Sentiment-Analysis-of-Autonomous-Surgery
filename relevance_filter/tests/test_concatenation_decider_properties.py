"""
Property-based tests for ConcatenationDecider component.

These tests use Hypothesis to verify universal properties across all inputs.
Each property test validates specific correctness properties from the design document.
"""

import pytest
from hypothesis import given, strategies as st, assume
from hypothesis.strategies import composite
from relevance_filter.concatenation_decider import ConcatenationDecider
from relevance_filter.models import ParentContext, RelevanceState


@composite
def parent_context_strategy(draw, relevance_state=None):
    """Generate a random ParentContext."""
    if relevance_state is None:
        relevance_state = draw(st.sampled_from([
            RelevanceState.RELEVANT_STRONG,
            RelevanceState.RELEVANT_INHERITED,
            RelevanceState.RELEVANT_WEAK,
            RelevanceState.IRRELEVANT
        ]))
    
    return ParentContext(
        id=draw(st.text(min_size=1, max_size=20)),
        text=draw(st.text(min_size=1, max_size=200)),
        relevance_state=relevance_state,
        relevance_score=draw(st.floats(min_value=0.0, max_value=1.0)),
        depth=draw(st.integers(min_value=0, max_value=10)),
        post_id=draw(st.text(min_size=1, max_size=20))
    )


@composite
def child_text_with_pronoun(draw):
    """Generate child text containing at least one pronoun."""
    pronouns = ["this", "that", "it", "they", "these", "those"]
    pronoun = draw(st.sampled_from(pronouns))
    
    # Generate text parts
    prefix = draw(st.text(alphabet=st.characters(blacklist_categories=('Cs',)), 
                          min_size=0, max_size=20))
    suffix = draw(st.text(alphabet=st.characters(blacklist_categories=('Cs',)), 
                          min_size=0, max_size=20))
    
    return f"{prefix} {pronoun} {suffix}".strip()


@composite
def child_text_without_pronoun(draw):
    """Generate child text without pronouns."""
    # Use words that don't contain pronouns
    safe_words = ["hello", "world", "example", "simple", "random", "word", "sentence"]
    word_count = draw(st.integers(min_value=1, max_value=10))
    words = [draw(st.sampled_from(safe_words)) for _ in range(word_count)]
    return " ".join(words)


@composite
def child_text_with_evaluative(draw):
    """Generate child text with evaluative statements."""
    evaluative_words = ["agree", "disagree", "correct", "wrong", "right", "exactly", "precisely"]
    evaluative_phrases = ["good point", "bad idea", "better argument", "worst case"]
    
    use_phrase = draw(st.booleans())
    if use_phrase:
        phrase = draw(st.sampled_from(evaluative_phrases))
        prefix = draw(st.text(alphabet=st.characters(blacklist_categories=('Cs',)), 
                              min_size=0, max_size=20))
        return f"{prefix} {phrase}".strip()
    else:
        word = draw(st.sampled_from(evaluative_words))
        prefix = draw(st.text(alphabet=st.characters(blacklist_categories=('Cs',)), 
                              min_size=0, max_size=20))
        suffix = draw(st.text(alphabet=st.characters(blacklist_categories=('Cs',)), 
                              min_size=0, max_size=20))
        return f"{prefix} {word} {suffix}".strip()


@composite
def child_text_without_evaluative(draw):
    """Generate child text without evaluative statements."""
    safe_words = ["hello", "world", "example", "simple", "random", "word", "sentence"]
    word_count = draw(st.integers(min_value=1, max_value=10))
    words = [draw(st.sampled_from(safe_words)) for _ in range(word_count)]
    return " ".join(words)


@composite
def short_child_text(draw, max_words=49):
    """Generate child text under the word threshold."""
    word_count = draw(st.integers(min_value=1, max_value=max_words))
    words = [draw(st.text(min_size=1, max_size=10)) for _ in range(word_count)]
    return " ".join(words)


@composite
def long_child_text(draw, min_words=50):
    """Generate child text at or over the word threshold."""
    word_count = draw(st.integers(min_value=min_words, max_value=min_words + 20))
    words = [draw(st.text(min_size=1, max_size=10)) for _ in range(word_count)]
    return " ".join(words)


@composite
def child_text_with_new_topic_indicator(draw):
    """Generate child text with new topic indicators."""
    indicators = [
        "speaking of", "by the way", "off topic",
        "unrelated", "different topic", "changing subject"
    ]
    indicator = draw(st.sampled_from(indicators))
    prefix = draw(st.text(alphabet=st.characters(blacklist_categories=('Cs',)), 
                          min_size=0, max_size=20))
    suffix = draw(st.text(alphabet=st.characters(blacklist_categories=('Cs',)), 
                          min_size=0, max_size=20))
    return f"{prefix} {indicator} {suffix}".strip()


@composite
def child_text_meeting_all_conditions(draw):
    """Generate child text that meets all concatenation conditions."""
    # Must have: pronoun, evaluative, short, no new topic
    pronouns = ["this", "that", "it", "they", "these", "those"]
    evaluative = ["agree", "disagree", "correct", "wrong", "right"]
    
    pronoun = draw(st.sampled_from(pronouns))
    eval_word = draw(st.sampled_from(evaluative))
    
    # Keep it short (under 50 words)
    filler_count = draw(st.integers(min_value=0, max_value=10))
    filler_words = [draw(st.text(min_size=1, max_size=5)) for _ in range(filler_count)]
    
    # Construct text with pronoun and evaluative word
    text = f"I {eval_word} {pronoun} " + " ".join(filler_words)
    
    # Ensure it's under 50 words
    words = text.split()
    if len(words) >= 50:
        text = " ".join(words[:49])
    
    return text.strip()


class TestConcatenationConditionsProperty:
    """Property-based tests for concatenation conditions."""
    
    @given(
        parent_context_strategy(relevance_state=RelevanceState.IRRELEVANT),
        st.text(min_size=1, max_size=100)
    )
    def test_property_9_condition_1_irrelevant_parent(self, parent_context, child_text):
        """
        Property 9: Concatenation Conditions (Condition 1 - Irrelevant Parent)
        
        **Validates: Requirements 4.2, 4.4**
        
        For any comment where the parent is IRRELEVANT, concatenation should NOT occur
        regardless of other conditions.
        """
        decider = ConcatenationDecider(word_threshold=50)
        
        result = decider.should_concatenate(parent_context, child_text)
        
        assert result is False, \
            f"Should not concatenate when parent is IRRELEVANT, but got True for child: '{child_text}'"
    
    @given(
        parent_context_strategy(relevance_state=RelevanceState.RELEVANT_WEAK),
        st.text(min_size=1, max_size=100)
    )
    def test_property_9_condition_1_weak_parent(self, parent_context, child_text):
        """
        Property 9: Concatenation Conditions (Condition 1 - Weak Parent)
        
        **Validates: Requirements 4.2, 4.4**
        
        For any comment where the parent is RELEVANT_WEAK, concatenation should NOT occur
        regardless of other conditions.
        """
        decider = ConcatenationDecider(word_threshold=50)
        
        result = decider.should_concatenate(parent_context, child_text)
        
        assert result is False, \
            f"Should not concatenate when parent is RELEVANT_WEAK, but got True for child: '{child_text}'"
    
    @given(
        st.sampled_from([RelevanceState.RELEVANT_STRONG, RelevanceState.RELEVANT_INHERITED]),
        child_text_without_pronoun()
    )
    def test_property_9_condition_2_no_pronouns(self, parent_state, child_text):
        """
        Property 9: Concatenation Conditions (Condition 2 - No Pronouns)
        
        **Validates: Requirements 4.2, 4.4**
        
        For any comment without pronouns (this, that, it, they, these, those),
        concatenation should NOT occur regardless of other conditions.
        """
        parent_context = ParentContext(
            id="parent_1",
            text="Some parent text about robotic surgery.",
            relevance_state=parent_state,
            relevance_score=0.9,
            depth=1,
            post_id="post_1"
        )
        
        decider = ConcatenationDecider(word_threshold=50)
        
        result = decider.should_concatenate(parent_context, child_text)
        
        assert result is False, \
            f"Should not concatenate when child has no pronouns, but got True for: '{child_text}'"
    
    @given(
        st.sampled_from([RelevanceState.RELEVANT_STRONG, RelevanceState.RELEVANT_INHERITED]),
        long_child_text(min_words=50)
    )
    def test_property_9_condition_3_too_long(self, parent_state, child_text):
        """
        Property 9: Concatenation Conditions (Condition 3 - Too Long)
        
        **Validates: Requirements 4.2, 4.4**
        
        For any comment with 50 or more words, concatenation should NOT occur
        regardless of other conditions.
        """
        parent_context = ParentContext(
            id="parent_1",
            text="Some parent text about robotic surgery.",
            relevance_state=parent_state,
            relevance_score=0.9,
            depth=1,
            post_id="post_1"
        )
        
        decider = ConcatenationDecider(word_threshold=50)
        
        word_count = len(child_text.split())
        assume(word_count >= 50)  # Ensure we're testing the right condition
        
        result = decider.should_concatenate(parent_context, child_text)
        
        assert result is False, \
            f"Should not concatenate when child has {word_count} words (>= 50), but got True"
    
    @given(
        st.sampled_from([RelevanceState.RELEVANT_STRONG, RelevanceState.RELEVANT_INHERITED]),
        child_text_with_pronoun()
    )
    def test_property_9_condition_4_no_evaluative(self, parent_state, child_text):
        """
        Property 9: Concatenation Conditions (Condition 4 - No Evaluative)
        
        **Validates: Requirements 4.2, 4.4**
        
        For any comment without evaluative statements (agree, disagree, correct, etc.),
        concatenation should NOT occur regardless of other conditions.
        """
        parent_context = ParentContext(
            id="parent_1",
            text="Some parent text about robotic surgery.",
            relevance_state=parent_state,
            relevance_score=0.9,
            depth=1,
            post_id="post_1"
        )
        
        decider = ConcatenationDecider(word_threshold=50)
        
        # Ensure the text doesn't accidentally contain evaluative words
        child_lower = child_text.lower()
        evaluative_words = ["agree", "disagree", "correct", "wrong", "right", "exactly", 
                           "precisely", "good point", "bad idea", "better", "worse", 
                           "best", "worst", "true", "false"]
        has_evaluative = any(word in child_lower for word in evaluative_words)
        assume(not has_evaluative)
        
        # Ensure it's short enough
        word_count = len(child_text.split())
        assume(word_count < 50)
        
        result = decider.should_concatenate(parent_context, child_text)
        
        assert result is False, \
            f"Should not concatenate when child has no evaluative statements, but got True for: '{child_text}'"
    
    @given(
        st.sampled_from([RelevanceState.RELEVANT_STRONG, RelevanceState.RELEVANT_INHERITED]),
        child_text_with_new_topic_indicator()
    )
    def test_property_9_condition_5_new_topic(self, parent_state, child_text):
        """
        Property 9: Concatenation Conditions (Condition 5 - New Topic)
        
        **Validates: Requirements 4.2, 4.4**
        
        For any comment that introduces a new topic (speaking of, by the way, etc.),
        concatenation should NOT occur regardless of other conditions.
        """
        parent_context = ParentContext(
            id="parent_1",
            text="Some parent text about robotic surgery.",
            relevance_state=parent_state,
            relevance_score=0.9,
            depth=1,
            post_id="post_1"
        )
        
        decider = ConcatenationDecider(word_threshold=50)
        
        result = decider.should_concatenate(parent_context, child_text)
        
        assert result is False, \
            f"Should not concatenate when child introduces new topic, but got True for: '{child_text}'"
    
    @given(
        st.sampled_from([RelevanceState.RELEVANT_STRONG, RelevanceState.RELEVANT_INHERITED]),
        child_text_meeting_all_conditions()
    )
    def test_property_9_all_conditions_met(self, parent_state, child_text):
        """
        Property 9: Concatenation Conditions (All Conditions Met)
        
        **Validates: Requirements 4.2, 4.4**
        
        For any comment where ALL five conditions are met:
        1. Parent is RELEVANT_STRONG or RELEVANT_INHERITED
        2. Child uses pronouns
        3. Child is under 50 words
        4. Child makes evaluative statements
        5. Child does not introduce new topic
        
        Then concatenation SHOULD occur.
        """
        parent_context = ParentContext(
            id="parent_1",
            text="Some parent text about robotic surgery.",
            relevance_state=parent_state,
            relevance_score=0.9,
            depth=1,
            post_id="post_1"
        )
        
        decider = ConcatenationDecider(word_threshold=50)
        
        # Verify the text meets all conditions
        child_lower = child_text.lower()
        
        # Check pronouns
        pronouns = ["this", "that", "it", "they", "these", "those"]
        has_pronouns = any(pronoun in child_lower for pronoun in pronouns)
        assume(has_pronouns)
        
        # Check evaluative
        evaluative_words = ["agree", "disagree", "correct", "wrong", "right", "exactly", "precisely"]
        has_evaluative = any(word in child_lower for word in evaluative_words)
        assume(has_evaluative)
        
        # Check word count
        word_count = len(child_text.split())
        assume(word_count < 50)
        
        # Check no new topic indicators
        new_topic_indicators = ["speaking of", "by the way", "off topic", 
                               "unrelated", "different topic", "changing subject"]
        has_new_topic = any(indicator in child_lower for indicator in new_topic_indicators)
        assume(not has_new_topic)
        
        result = decider.should_concatenate(parent_context, child_text)
        
        assert result is True, \
            f"Should concatenate when all conditions are met, but got False for: '{child_text}'"
    
    @given(
        st.sampled_from([RelevanceState.RELEVANT_STRONG, RelevanceState.RELEVANT_INHERITED]),
        st.sampled_from(["this", "that", "it", "they", "these", "those"]),
        st.sampled_from([str.lower, str.upper, str.title])
    )
    def test_property_9_pronoun_case_insensitivity(self, parent_state, pronoun, case_transform):
        """
        Property 9: Concatenation Conditions (Pronoun Case Insensitivity)
        
        **Validates: Requirements 4.2, 4.4**
        
        For any pronoun, detection should work regardless of capitalization.
        """
        parent_context = ParentContext(
            id="parent_1",
            text="Some parent text about robotic surgery.",
            relevance_state=parent_state,
            relevance_score=0.9,
            depth=1,
            post_id="post_1"
        )
        
        decider = ConcatenationDecider(word_threshold=50)
        
        # Create text with case-transformed pronoun and evaluative word
        transformed_pronoun = case_transform(pronoun)
        child_text = f"I agree {transformed_pronoun} is correct"
        
        result = decider.should_concatenate(parent_context, child_text)
        
        assert result is True, \
            f"Should detect pronoun regardless of case: '{child_text}'"
    
    @given(
        st.sampled_from([RelevanceState.RELEVANT_STRONG, RelevanceState.RELEVANT_INHERITED]),
        st.sampled_from(["agree", "disagree", "correct", "wrong", "right"]),
        st.sampled_from([str.lower, str.upper, str.title])
    )
    def test_property_9_evaluative_case_insensitivity(self, parent_state, evaluative, case_transform):
        """
        Property 9: Concatenation Conditions (Evaluative Case Insensitivity)
        
        **Validates: Requirements 4.2, 4.4**
        
        For any evaluative word, detection should work regardless of capitalization.
        """
        parent_context = ParentContext(
            id="parent_1",
            text="Some parent text about robotic surgery.",
            relevance_state=parent_state,
            relevance_score=0.9,
            depth=1,
            post_id="post_1"
        )
        
        decider = ConcatenationDecider(word_threshold=50)
        
        # Create text with case-transformed evaluative word and pronoun
        transformed_evaluative = case_transform(evaluative)
        child_text = f"I {transformed_evaluative} that this is true"
        
        result = decider.should_concatenate(parent_context, child_text)
        
        assert result is True, \
            f"Should detect evaluative word regardless of case: '{child_text}'"
    
    @given(
        st.sampled_from([RelevanceState.RELEVANT_STRONG, RelevanceState.RELEVANT_INHERITED]),
        st.integers(min_value=5, max_value=49)
    )
    def test_property_9_word_threshold_boundary_under(self, parent_state, word_count):
        """
        Property 9: Concatenation Conditions (Word Threshold Boundary - Under)
        
        **Validates: Requirements 4.2, 4.4**
        
        For any comment with fewer than 50 words (and meeting all other conditions),
        concatenation should occur.
        """
        parent_context = ParentContext(
            id="parent_1",
            text="Some parent text about robotic surgery.",
            relevance_state=parent_state,
            relevance_score=0.9,
            depth=1,
            post_id="post_1"
        )
        
        decider = ConcatenationDecider(word_threshold=50)
        
        # Create text with exact word count, including pronoun and evaluative
        # Start with required words: "I agree that this"
        words = ["I", "agree", "that", "this"] + ["word"] * (word_count - 4)
        child_text = " ".join(words)
        
        actual_word_count = len(child_text.split())
        assert actual_word_count == word_count
        assert word_count < 50
        
        result = decider.should_concatenate(parent_context, child_text)
        
        assert result is True, \
            f"Should concatenate when word count ({word_count}) is under threshold (50)"
    
    @given(
        st.sampled_from([RelevanceState.RELEVANT_STRONG, RelevanceState.RELEVANT_INHERITED]),
        st.integers(min_value=50, max_value=70)
    )
    def test_property_9_word_threshold_boundary_at_or_over(self, parent_state, word_count):
        """
        Property 9: Concatenation Conditions (Word Threshold Boundary - At or Over)
        
        **Validates: Requirements 4.2, 4.4**
        
        For any comment with 50 or more words, concatenation should NOT occur
        regardless of other conditions.
        """
        parent_context = ParentContext(
            id="parent_1",
            text="Some parent text about robotic surgery.",
            relevance_state=parent_state,
            relevance_score=0.9,
            depth=1,
            post_id="post_1"
        )
        
        decider = ConcatenationDecider(word_threshold=50)
        
        # Create text with exact word count, including pronoun and evaluative
        words = ["I", "agree", "that", "this"] + ["word"] * (word_count - 4)
        child_text = " ".join(words)
        
        actual_word_count = len(child_text.split())
        assert actual_word_count == word_count
        assert word_count >= 50
        
        result = decider.should_concatenate(parent_context, child_text)
        
        assert result is False, \
            f"Should not concatenate when word count ({word_count}) is at or over threshold (50)"
    
    @given(
        st.sampled_from([RelevanceState.RELEVANT_STRONG, RelevanceState.RELEVANT_INHERITED]),
        st.text(min_size=1, max_size=100),
        st.text(min_size=1, max_size=100)
    )
    def test_property_9_deterministic_behavior(self, parent_state, parent_text, child_text):
        """
        Property 9: Concatenation Conditions (Deterministic Behavior)
        
        **Validates: Requirements 4.2, 4.4**
        
        For any parent-child pair, the concatenation decision should be deterministic
        (same inputs always produce same output).
        """
        parent_context = ParentContext(
            id="parent_1",
            text=parent_text,
            relevance_state=parent_state,
            relevance_score=0.9,
            depth=1,
            post_id="post_1"
        )
        
        decider = ConcatenationDecider(word_threshold=50)
        
        # Call twice with same inputs
        result1 = decider.should_concatenate(parent_context, child_text)
        result2 = decider.should_concatenate(parent_context, child_text)
        
        assert result1 == result2, \
            f"Concatenation decision should be deterministic, but got {result1} and {result2}"


class TestConcatenationFormatProperty:
    """Property-based tests for concatenation format consistency."""
    
    @given(
        st.text(min_size=1, max_size=500),
        st.text(min_size=1, max_size=500)
    )
    def test_property_10_concatenation_format_consistency(self, parent_text, child_text):
        """
        Property 10: Concatenation Format Consistency
        
        **Validates: Requirements 4.3**
        
        For any parent-child pair that meets concatenation conditions, the concatenated
        text should follow the exact format "PARENT CONTEXT:\n<parent text>\n\nCHILD COMMENT:\n<child text>\n".
        """
        decider = ConcatenationDecider(word_threshold=50)
        
        # Concatenate the texts
        result = decider.concatenate(parent_text, child_text)
        
        # Verify the format is exactly as specified
        expected_format = f"PARENT CONTEXT:\n{parent_text}\n\nCHILD COMMENT:\n{child_text}\n"
        
        assert result == expected_format, \
            f"Concatenation format mismatch.\nExpected: {repr(expected_format)}\nGot: {repr(result)}"
    
    @given(
        st.text(min_size=1, max_size=500),
        st.text(min_size=1, max_size=500)
    )
    def test_property_10_format_has_parent_prefix(self, parent_text, child_text):
        """
        Property 10: Concatenation Format Consistency (Parent Prefix)
        
        **Validates: Requirements 4.3**
        
        For any parent-child pair, the concatenated text should start with "PARENT CONTEXT:\n".
        """
        decider = ConcatenationDecider(word_threshold=50)
        
        result = decider.concatenate(parent_text, child_text)
        
        assert result.startswith("PARENT CONTEXT:\n"), \
            f"Concatenated text should start with 'PARENT CONTEXT:\\n', but got: {repr(result[:50])}"
    
    @given(
        st.text(min_size=1, max_size=500),
        st.text(min_size=1, max_size=500)
    )
    def test_property_10_format_has_child_prefix(self, parent_text, child_text):
        """
        Property 10: Concatenation Format Consistency (Child Prefix)
        
        **Validates: Requirements 4.3**
        
        For any parent-child pair, the concatenated text should contain "\n\nCHILD COMMENT:\n".
        """
        decider = ConcatenationDecider(word_threshold=50)
        
        result = decider.concatenate(parent_text, child_text)
        
        assert "\n\nCHILD COMMENT:\n" in result, \
            f"Concatenated text should contain '\\n\\nCHILD COMMENT:\\n', but got: {repr(result)}"
    
    @given(
        st.text(min_size=1, max_size=500),
        st.text(min_size=1, max_size=500)
    )
    def test_property_10_format_preserves_parent_text(self, parent_text, child_text):
        """
        Property 10: Concatenation Format Consistency (Parent Text Preservation)
        
        **Validates: Requirements 4.3**
        
        For any parent-child pair, the parent text should appear exactly after "PARENT CONTEXT:\n"
        and before "\n\nCHILD COMMENT:\n".
        """
        decider = ConcatenationDecider(word_threshold=50)
        
        result = decider.concatenate(parent_text, child_text)
        
        # Extract the parent text portion
        prefix = "PARENT CONTEXT:\n"
        separator = "\n\nCHILD COMMENT:\n"
        
        assert result.startswith(prefix), "Result should start with 'PARENT CONTEXT:\\n'"
        assert separator in result, "Result should contain '\\n\\nCHILD COMMENT:\\n'"
        
        # Extract parent text from result
        start_idx = len(prefix)
        end_idx = result.index(separator)
        extracted_parent = result[start_idx:end_idx]
        
        assert extracted_parent == parent_text, \
            f"Parent text not preserved.\nExpected: {repr(parent_text)}\nGot: {repr(extracted_parent)}"
    
    @given(
        st.text(min_size=1, max_size=500),
        st.text(min_size=1, max_size=500)
    )
    def test_property_10_format_preserves_child_text(self, parent_text, child_text):
        """
        Property 10: Concatenation Format Consistency (Child Text Preservation)
        
        **Validates: Requirements 4.3**
        
        For any parent-child pair, the child text should appear exactly after "\n\nCHILD COMMENT:\n"
        and before the final newline.
        """
        decider = ConcatenationDecider(word_threshold=50)
        
        result = decider.concatenate(parent_text, child_text)
        
        # Extract the child text portion
        separator = "\n\nCHILD COMMENT:\n"
        
        assert separator in result, "Result should contain '\\n\\nCHILD COMMENT:\\n'"
        
        # Extract child text from result
        separator_idx = result.index(separator)
        # It's at the end, followed by exactly one newline
        extracted_child = result[separator_idx + len(separator):-1]
        
        assert extracted_child == child_text, \
            f"Child text not preserved.\nExpected: {repr(child_text)}\nGot: {repr(extracted_child)}"
        assert result.endswith("\n"), "Result should end with a newline"
    
    @given(
        st.text(min_size=0, max_size=500),
        st.text(min_size=0, max_size=500)
    )
    def test_property_10_format_handles_empty_strings(self, parent_text, child_text):
        """
        Property 10: Concatenation Format Consistency (Empty String Handling)
        
        **Validates: Requirements 4.3**
        
        For any parent-child pair including empty strings, the format should remain consistent.
        """
        decider = ConcatenationDecider(word_threshold=50)
        
        result = decider.concatenate(parent_text, child_text)
        
        expected_format = f"PARENT CONTEXT:\n{parent_text}\n\nCHILD COMMENT:\n{child_text}\n"
        
        assert result == expected_format, \
            f"Format should be consistent even with empty strings.\nExpected: {repr(expected_format)}\nGot: {repr(result)}"
    
    @given(
        st.text(min_size=1, max_size=200),
        st.text(min_size=1, max_size=200)
    )
    def test_property_10_format_handles_newlines_in_text(self, parent_text, child_text):
        """
        Property 10: Concatenation Format Consistency (Newline Handling)
        
        **Validates: Requirements 4.3**
        
        For any parent-child pair containing newlines, the format should remain consistent
        and preserve the newlines within the text.
        """
        # Add newlines to the text
        parent_with_newline = parent_text + "\nExtra line"
        child_with_newline = child_text + "\nAnother line"
        
        decider = ConcatenationDecider(word_threshold=50)
        
        result = decider.concatenate(parent_with_newline, child_with_newline)
        
        expected_format = f"PARENT CONTEXT:\n{parent_with_newline}\n\nCHILD COMMENT:\n{child_with_newline}\n"
        
        assert result == expected_format, \
            f"Format should preserve newlines in text.\nExpected: {repr(expected_format)}\nGot: {repr(result)}"
    
    @given(
        st.text(min_size=1, max_size=100),
        st.text(min_size=1, max_size=100)
    )
    def test_property_10_format_is_deterministic(self, parent_text, child_text):
        """
        Property 10: Concatenation Format Consistency (Deterministic)
        
        **Validates: Requirements 4.3**
        
        For any parent-child pair, concatenating multiple times should produce identical results.
        """
        decider = ConcatenationDecider(word_threshold=50)
        
        result1 = decider.concatenate(parent_text, child_text)
        result2 = decider.concatenate(parent_text, child_text)
        result3 = decider.concatenate(parent_text, child_text)
        
        assert result1 == result2 == result3, \
            f"Concatenation should be deterministic.\nResult1: {repr(result1)}\nResult2: {repr(result2)}\nResult3: {repr(result3)}"
    
    @given(
        st.text(min_size=1, max_size=500),
        st.text(min_size=1, max_size=500)
    )
    def test_property_10_format_single_separator(self, parent_text, child_text):
        """
        Property 10: Concatenation Format Consistency (Single Separator)
        
        **Validates: Requirements 4.3**
        
        For any parent-child pair, there should be exactly one occurrence of the separator
        "\n\nCHILD COMMENT:\n" between parent and child text (unless it appears in the original text).
        """
        decider = ConcatenationDecider(word_threshold=50)
        
        result = decider.concatenate(parent_text, child_text)
        
        # The format should have the separator exactly once in the structure
        expected_format = f"PARENT CONTEXT:\n{parent_text}\n\nCHILD COMMENT:\n{child_text}\n"
        
        assert result == expected_format, \
            f"Format should have proper structure.\nExpected: {repr(expected_format)}\nGot: {repr(result)}"
    
    @given(
        st.text(min_size=1, max_size=200),
        st.text(min_size=1, max_size=200)
    )
    def test_property_10_format_handles_format_strings_in_text(self, parent_text, child_text):
        """
        Property 10: Concatenation Format Consistency (Format Strings in Text)
        
        **Validates: Requirements 4.3**
        
        For any parent-child pair where the text itself contains "PARENT CONTEXT:" or
        "CHILD COMMENT:", the format should still be correct and preserve the original text.
        """
        # Add format strings to the text
        parent_with_format = parent_text + " PARENT CONTEXT: something"
        child_with_format = child_text + " CHILD COMMENT: something"
        
        decider = ConcatenationDecider(word_threshold=50)
        
        result = decider.concatenate(parent_with_format, child_with_format)
        
        expected_format = f"PARENT CONTEXT:\n{parent_with_format}\n\nCHILD COMMENT:\n{child_with_format}\n"
        
        assert result == expected_format, \
            f"Format should preserve text even when it contains format strings.\nExpected: {repr(expected_format)}\nGot: {repr(result)}"
