# Implementation Plan: Reddit Relevance Filter

## Overview

This implementation plan breaks down the Reddit relevance filter into discrete coding tasks. The approach follows a bottom-up strategy: build core components first (keyword filter, analytical content detector), then semantic classification, then the orchestration layer, and finally integration and testing.

## Tasks

- [x] 1. Set up project structure and core data models
  - Create directory structure for the relevance filter module
  - Define RelevanceState enum and data classes (ParentContext, FilteredItem, FilterConfig)
  - Set up logging configuration
  - Install required dependencies (hypothesis for property testing, sentence-transformers or OpenAI for semantic classification)
  - _Requirements: 5.1, 8.2_

- [ ] 2. Implement KeywordFilter component
  - [x] 2.1 Create KeywordFilter class with keyword lists
    - Implement surgery keywords, procedure keywords, and context keywords
    - Implement case-insensitive matching logic
    - Implement get_matched_keywords for debugging
    - _Requirements: 9.1, 9.2, 9.3, 9.4_
  
  - [x] 2.2 Write property test for keyword filter case insensitivity
    - **Property 4: Keyword Filter Case Insensitivity**
    - **Validates: Requirements 9.4**
  
  - [x] 2.3 Write property test for keyword filter comprehensiveness
    - **Property 5: Keyword Filter Comprehensiveness**
    - **Validates: Requirements 9.1, 9.2, 9.3**
  
  - [x] 2.4 Write unit tests for keyword filter edge cases
    - Test empty text, very long text, special characters
    - _Requirements: 9.1, 9.2, 9.3, 9.4_

- [ ] 3. Implement AnalyticalContentDetector component
  - [x] 3.1 Create AnalyticalContentDetector class
    - Implement epistemic verbs, causal markers, and domain terms lists
    - Implement has_analytical_content method with regex matching
    - Implement get_analytical_markers for debugging
    - _Requirements: 7.2_
  
  - [x] 3.2 Write property test for analytical content detection
    - **Property 11: Analytical Content Detection**
    - **Validates: Requirements 7.2**
  
  - [x] 3.3 Write unit tests for analytical content edge cases
    - Test text with no markers, text with multiple markers, boundary cases
    - _Requirements: 7.2_

- [ ] 4. Implement ConcatenationDecider component
  - [x] 4.1 Create ConcatenationDecider class
    - Implement should_concatenate method with all five conditions
    - Implement pronoun detection, word counting, evaluative pattern matching
    - Implement concatenate method with proper formatting
    - _Requirements: 4.1, 4.2, 4.3, 4.4_
  
  - [x] 4.2 Write property test for concatenation conditions
    - **Property 9: Concatenation Conditions**
    - **Validates: Requirements 4.2, 4.4**
  
  - [x] 4.3 Write property test for concatenation format
    - **Property 10: Concatenation Format Consistency**
    - **Validates: Requirements 4.3**
  
  - [x] 4.4 Write unit tests for concatenation edge cases
    - Test boundary word counts, various pronoun types, edge cases
    - _Requirements: 4.1, 4.2, 4.3, 4.4_

- [x] 5. Checkpoint - Ensure all component tests pass
  - Run all tests for KeywordFilter, AnalyticalContentDetector, and ConcatenationDecider
  - Verify test coverage for these components
  - Ask the user if questions arise

- [ ] 6. Implement SemanticClassifier component
  - [x] 6.1 Create SemanticClassifier base class and embedding-based implementation
    - Implement __init__ with model loading (sentence-transformers)
    - Compute reference embeddings for autonomous surgery concepts
    - Implement classify_local using embedding similarity
    - Implement classify_contextual for parent-child pairs
    - _Requirements: 10.1, 10.2, 10.3_
  
  - [-] 6.2 Write property test for classification mode selection
    - **Property 6: Classification Mode Selection**
    - **Validates: Requirements 3.1, 3.4**
  
  - [ ] 6.3 Write unit tests for semantic classifier
    - Test with known relevant/irrelevant examples
    - Test jokes, metaphors, sci-fi references (should be rejected)
    - Test analytical humor (should be accepted)
    - _Requirements: 10.2, 10.3, 10.4, 10.5_
  
  - [ ] 6.4 (Optional) Implement LLM-based semantic classifier
    - Implement _llm_classify method using OpenAI or similar API
    - Add prompt engineering for relevance classification
    - _Requirements: 10.1, 10.2, 10.3_

- [ ] 7. Implement CommentEvaluator component
  - [ ] 7.1 Create CommentEvaluator class
    - Implement evaluate method with full decision tree logic
    - Implement _determine_classification_mode helper
    - Integrate KeywordFilter, SemanticClassifier, AnalyticalContentDetector, ConcatenationDecider
    - Handle all relevance state assignments
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 5.1, 5.2, 5.3, 5.4, 5.5_
  
  - [ ] 7.2 Write property test for context inheritance
    - **Property 7: Context Inheritance for Continuations**
    - **Validates: Requirements 3.2**
  
  - [ ] 7.3 Write property test for recovery from irrelevant parents
    - **Property 8: Recovery from Irrelevant Parents**
    - **Validates: Requirements 3.5**
  
  - [ ] 7.4 Write unit tests for comment evaluator
    - Test various parent-child scenarios
    - Test all relevance state assignments
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5_

- [ ] 8. Implement RelevanceFilter orchestrator
  - [ ] 8.1 Create RelevanceFilter class with post-level filtering
    - Implement __init__ with component initialization
    - Implement evaluate_post method
    - Implement filter_single_post method
    - Implement filter_posts for batch processing
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 11.1, 11.3_
  
  - [ ] 8.2 Implement recursive comment tree traversal
    - Implement process_comment_tree with depth-first traversal
    - Track parent context, depth, and relevance state
    - Implement subtree pruning logic
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 7.1, 7.3, 7.4, 7.5_
  
  - [ ] 8.3 Write property test for irrelevant post tree exclusion
    - **Property 3: Irrelevant Post Tree Exclusion**
    - **Validates: Requirements 1.4**
  
  - [ ] 8.4 Write property test for subtree pruning
    - **Property 12: Subtree Pruning Rule**
    - **Validates: Requirements 7.1, 7.3, 7.4**
  
  - [ ] 8.5 Write property test for depth-first traversal
    - **Property 13: Depth-First Traversal Order**
    - **Validates: Requirements 6.1**

- [ ] 9. Checkpoint - Ensure core filtering logic works
  - Run all tests for RelevanceFilter and CommentEvaluator
  - Test with sample Reddit data
  - Verify output format is correct
  - Ask the user if questions arise

- [ ] 10. Implement output formatting and validation
  - [ ] 10.1 Add output validation and formatting
    - Ensure all output items have required fields
    - Implement flattening of tree structure to list
    - Validate parent_id and post_id references
    - _Requirements: 8.1, 8.2, 8.3, 8.4_
  
  - [ ] 10.2 Write property test for parent-child relationship preservation
    - **Property 14: Parent-Child Relationship Preservation**
    - **Validates: Requirements 6.4, 8.4**
  
  - [ ] 10.3 Write property test for depth tracking accuracy
    - **Property 15: Depth Tracking Accuracy**
    - **Validates: Requirements 6.2**
  
  - [ ] 10.4 Write property test for output field completeness
    - **Property 18: Output Field Completeness**
    - **Validates: Requirements 8.2**
  
  - [ ] 10.5 Write property test for text preservation
    - **Property 19: Text Preservation**
    - **Validates: Requirements 8.3**

- [ ] 11. Implement error handling and edge cases
  - [ ] 11.1 Add error handling for missing fields and malformed data
    - Handle missing selftext, body, replies fields
    - Handle empty or whitespace-only text
    - Handle very long text (truncation)
    - Add logging for errors and warnings
    - _Requirements: Error Handling section_
  
  - [ ] 11.2 Add retry logic for semantic classifier API failures
    - Implement exponential backoff
    - Fall back to keyword-only classification on failure
    - _Requirements: Error Handling section_
  
  - [ ] 11.3 Write unit tests for error handling
    - Test missing fields, empty text, API failures
    - Test malformed JSON, network timeouts
    - _Requirements: Error Handling section_

- [ ] 12. Implement sentiment and stance independence
  - [ ] 12.1 Write property test for sentiment independence
    - **Property 21: Topic-Based Classification Independence from Sentiment**
    - **Validates: Requirements 13.2, 13.4**
  
  - [ ] 12.2 Write property test for stance independence
    - **Property 22: Topic-Based Classification Independence from Stance**
    - **Validates: Requirements 13.3, 13.4**

- [ ] 13. Implement batch processing and optimization
  - [ ] 13.1 Optimize semantic classifier for batch processing
    - Implement batched embedding computation
    - Add caching for repeated texts
    - _Requirements: 11.2_
  
  - [ ] 13.2 Write property test for batch processing independence
    - **Property 20: Batch Processing Independence**
    - **Validates: Requirements 11.3**
  
  - [ ] 13.3 Write performance tests
    - Test with 1000 posts and 10,000 comments
    - Verify processing time and memory usage
    - _Requirements: 11.4, 11.5_

- [ ] 14. Integration and end-to-end testing
  - [ ] 14.1 Write integration test with real Reddit data
    - Load data from reddit_robotic_surgery_sentiment.json
    - Run complete filtering pipeline
    - Verify output format and content
    - _Requirements: 8.5_
  
  - [ ] 14.2 Write integration test for sentiment pipeline compatibility
    - Verify filtered output can be consumed by sentiment analysis
    - Test that output format matches expected schema
    - _Requirements: 8.5_
  
  - [ ] 14.3 Write unit tests for conversation drift examples
    - Test threads that drift from technical to jokes
    - Test analytical humor vs pure comedy
    - _Requirements: 12.1, 12.2, 12.3, 12.5_

- [ ] 15. Final checkpoint and documentation
  - Run complete test suite (unit tests + property tests)
  - Verify all 22 correctness properties are tested
  - Verify test coverage goals (>90% line coverage, >85% branch coverage)
  - Add docstrings and inline comments
  - Create usage examples and integration guide
  - Ask the user if questions arise

## Notes

- All tasks are required for comprehensive implementation with full testing coverage
- Each task references specific requirements for traceability
- Property tests validate universal correctness properties (22 total)
- Unit tests validate specific examples and edge cases
- The implementation follows a bottom-up approach: components → orchestration → integration
- Python is used throughout based on the project context
- Checkpoints at tasks 5, 9, and 15 ensure incremental validation
