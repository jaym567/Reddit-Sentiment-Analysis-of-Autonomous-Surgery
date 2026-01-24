# Requirements Document

## Introduction

This document specifies requirements for a context-aware relevance filtering module for a Reddit sentiment analysis research project focused on autonomous, robotic, and AI-assisted surgery discussions. The module filters Reddit posts and comments to retain only content relevant to autonomous surgery research, using context inheritance and semantic analysis to handle conversational discourse patterns.

## Glossary

- **Relevance_Filter**: The system that evaluates and filters Reddit content for relevance to autonomous surgery
- **Post**: A top-level Reddit submission containing title, selftext, and a tree of comments
- **Comment**: A Reddit comment that may contain nested replies forming a tree structure
- **Comment_Tree**: The recursive structure of comments and replies under a post
- **Relevance_State**: An enumeration indicating the type and strength of relevance (RELEVANT_STRONG, RELEVANT_INHERITED, RELEVANT_WEAK, IRRELEVANT)
- **Context_Inheritance**: The mechanism by which child comments inherit relevance from parent comments
- **Keyword_Filter**: Stage 1 filter using high-recall keyword matching
- **Semantic_Classifier**: Stage 2 filter using embedding similarity or LLM-based classification
- **Local_Classification**: Semantic classification using only the comment's own text
- **Contextual_Classification**: Semantic classification using concatenated parent and child text
- **Analytical_Content**: Content containing epistemic/evaluative language, causal markers, or domain-specific medical/AI/ethics terminology
- **Subtree_Pruning**: The process of discarding entire branches of replies when a comment is irrelevant
- **Parent_Child_Concatenation**: Combining parent and child text for classification when context is needed
- **Sentiment_Pipeline**: The existing downstream sentiment analysis system that processes filtered content

## Requirements

### Requirement 1: Post-Level Relevance Filtering

**User Story:** As a researcher, I want to filter entire posts based on relevance, so that I only analyze discussions about autonomous/robotic surgery.

#### Acceptance Criteria

1. WHEN a post is evaluated, THE Relevance_Filter SHALL analyze both title and selftext fields
2. WHEN a post contains autonomous surgery keywords (autonomous surgery, robotic surgeon, surgical robot, cholecystectomy, operating room AI, minimally invasive robotics), THE Relevance_Filter SHALL pass it to semantic classification
3. WHEN a post passes semantic classification as relevant to autonomous/robotic surgery, THE Relevance_Filter SHALL mark it as RELEVANT_STRONG
4. IF a post is marked IRRELEVANT, THEN THE Relevance_Filter SHALL discard the entire Comment_Tree without further processing
5. WHEN a post is marked RELEVANT_STRONG, THE Relevance_Filter SHALL process all comments in its Comment_Tree

### Requirement 2: Two-Stage Filtering Architecture

**User Story:** As a researcher, I want a two-stage filtering approach, so that I achieve high recall while maintaining precision through semantic validation.

#### Acceptance Criteria

1. THE Keyword_Filter SHALL use broad keyword matching to identify potentially relevant content
2. WHEN content passes the Keyword_Filter, THE Semantic_Classifier SHALL evaluate whether it is meaningfully about autonomous surgery
3. THE Semantic_Classifier SHALL reject content that is purely jokes, metaphors, memes, or sci-fi references without analytical substance
4. WHEN content contains surgery-related words but lacks autonomous/robotic/AI surgical context, THE Semantic_Classifier SHALL mark it as IRRELEVANT
5. THE Relevance_Filter SHALL apply both stages sequentially to all posts and comments

### Requirement 3: Comment-Level Relevance with Context Inheritance

**User Story:** As a researcher, I want comments to inherit relevance from their parents, so that conversational replies without explicit keywords are not incorrectly filtered out.

#### Acceptance Criteria

1. WHEN a comment's parent is RELEVANT_STRONG or RELEVANT_INHERITED, THE Relevance_Filter SHALL evaluate the child comment using contextual classification
2. WHEN contextual classification determines a child comment continues the parent's discussion, THE Relevance_Filter SHALL mark it as RELEVANT_INHERITED
3. WHEN contextual classification determines a child comment clearly diverges to off-topic content, THE Relevance_Filter SHALL mark it as IRRELEVANT or RELEVANT_WEAK based on analytical content
4. WHEN a comment's parent is IRRELEVANT, THE Relevance_Filter SHALL use local classification on the child's own text
5. WHEN local classification finds strong autonomous surgery content in a child of an irrelevant parent, THE Relevance_Filter SHALL mark it as RELEVANT_STRONG to allow recovery

### Requirement 4: Parent-Child Text Concatenation

**User Story:** As a researcher, I want short contextual replies to be evaluated with their parent text, so that pronouns and implicit references are correctly interpreted.

#### Acceptance Criteria

1. WHEN evaluating a comment, THE Relevance_Filter SHALL determine whether to classify the comment alone or concatenated with parent text
2. WHEN a comment meets all concatenation conditions (parent is RELEVANT_STRONG or RELEVANT_INHERITED, child uses pronouns, child is under 50 words, child makes evaluative statements, child does not introduce new topic), THE Relevance_Filter SHALL concatenate parent and child text
3. WHEN concatenating text, THE Relevance_Filter SHALL use the format "PARENT CONTEXT: <parent text>\nCHILD COMMENT: <child text>"
4. WHEN a comment does not meet concatenation conditions, THE Relevance_Filter SHALL classify it using only its own text
5. WHEN a concatenated text is classified as relevant, THE Relevance_Filter SHALL mark the child comment as RELEVANT_INHERITED

### Requirement 5: Relevance State Classification

**User Story:** As a researcher, I want each piece of content tagged with a relevance state, so that I can understand why content was retained and trace filtering decisions.

#### Acceptance Criteria

1. THE Relevance_Filter SHALL assign exactly one Relevance_State to each evaluated comment
2. WHEN content explicitly discusses autonomous/robotic surgery with strong keywords and semantic match, THE Relevance_Filter SHALL assign RELEVANT_STRONG
3. WHEN content is relevant due to parent context inheritance, THE Relevance_Filter SHALL assign RELEVANT_INHERITED
4. WHEN content is tangentially related but still analytical, THE Relevance_Filter SHALL assign RELEVANT_WEAK
5. WHEN content is off-topic, joke-only, meme, or conversational drift, THE Relevance_Filter SHALL assign IRRELEVANT

### Requirement 6: Recursive Tree Traversal

**User Story:** As a researcher, I want the filter to traverse the entire comment tree recursively, so that all nested replies are evaluated with proper context.

#### Acceptance Criteria

1. THE Relevance_Filter SHALL traverse the Comment_Tree using depth-first recursive traversal
2. WHEN processing a comment, THE Relevance_Filter SHALL track parent relevance state, current depth, and inherited relevance
3. WHEN a comment has replies, THE Relevance_Filter SHALL recursively process each reply with updated context
4. THE Relevance_Filter SHALL maintain the parent-child relationship throughout traversal
5. WHEN traversal is complete, THE Relevance_Filter SHALL have evaluated every comment in the tree

### Requirement 7: Subtree Pruning

**User Story:** As a researcher, I want irrelevant comment branches pruned, so that processing time is reduced and joke threads are excluded.

#### Acceptance Criteria

1. WHEN a comment is marked IRRELEVANT and lacks Analytical_Content, THE Relevance_Filter SHALL not traverse its replies
2. WHEN determining if content has Analytical_Content, THE Relevance_Filter SHALL check for epistemic/evaluative verbs (think, believe, seems, would worry, probably), causal markers (because, therefore, so that, this means), or domain-specific terms (error rate, liability, training data, FDA, malpractice, precision, outcomes)
3. WHEN a comment is marked IRRELEVANT but contains Analytical_Content, THE Relevance_Filter SHALL continue traversing its replies to allow recovery
4. WHEN subtree pruning occurs, THE Relevance_Filter SHALL exclude all descendant comments from output
5. THE Relevance_Filter SHALL evaluate all immediate children of a node before applying pruning decisions

### Requirement 8: Output Format and Integration

**User Story:** As a researcher, I want filtered content in a flattened format, so that it integrates seamlessly with the existing sentiment analysis pipeline.

#### Acceptance Criteria

1. THE Relevance_Filter SHALL output a flattened list containing only relevant items
2. WHEN outputting an item, THE Relevance_Filter SHALL include id, type (post or comment), text, parent_id, post_id, depth, relevance_score, relevance_reason, and relevance_state
3. THE Relevance_Filter SHALL preserve the original text content without modification
4. THE Relevance_Filter SHALL maintain parent-child relationships through parent_id and post_id fields
5. THE Relevance_Filter SHALL output data in a format directly consumable by the Sentiment_Pipeline

### Requirement 9: Keyword Filter Implementation

**User Story:** As a researcher, I want a high-recall keyword filter, so that potentially relevant content is not prematurely discarded.

#### Acceptance Criteria

1. THE Keyword_Filter SHALL match content containing terms related to autonomous surgery (autonomous surgery, robotic surgeon, surgical robot, da Vinci, laparoscopic robot, surgical automation, AI surgery, robot-assisted surgery)
2. THE Keyword_Filter SHALL match content containing surgical procedure terms (cholecystectomy, prostatectomy, hysterectomy, cardiac surgery, minimally invasive)
3. THE Keyword_Filter SHALL match content containing robotics and AI terms in surgical context (operating room AI, surgical AI, autonomous operation, robotic precision)
4. THE Keyword_Filter SHALL use case-insensitive matching
5. WHEN content does not match any keywords, THE Keyword_Filter SHALL mark it for potential context inheritance evaluation if parent is relevant

### Requirement 10: Semantic Classification Implementation

**User Story:** As a researcher, I want semantic classification to distinguish meaningful content from jokes and metaphors, so that precision is maintained after high-recall keyword filtering.

#### Acceptance Criteria

1. THE Semantic_Classifier SHALL provide two classification modes: Local_Classification using only comment text, and Contextual_Classification using parent and child text
2. WHEN evaluating content with Local_Classification, THE Semantic_Classifier SHALL determine if the text alone meaningfully discusses autonomous/robotic surgery
3. WHEN evaluating content with Contextual_Classification, THE Semantic_Classifier SHALL determine if the child comment continues the parent's autonomous surgery discussion
4. THE Semantic_Classifier SHALL reject content that uses surgery terms purely as jokes, metaphors, or sci-fi references without analytical substance
5. THE Semantic_Classifier SHALL accept content with light humor if it contains Analytical_Content grounded in surgical discussion

### Requirement 11: Batch Processing and Scalability

**User Story:** As a researcher, I want the filter to process large datasets efficiently, so that I can analyze thousands of Reddit posts without performance bottlenecks.

#### Acceptance Criteria

1. THE Relevance_Filter SHALL support batch processing of multiple posts
2. WHEN processing batches, THE Relevance_Filter SHALL optimize API calls to semantic classification services
3. THE Relevance_Filter SHALL process posts independently to enable parallel execution
4. THE Relevance_Filter SHALL handle large comment trees (>1000 comments) without memory issues
5. THE Relevance_Filter SHALL log processing time and throughput metrics

### Requirement 12: Conversation Drift Handling

**User Story:** As a researcher, I want the filter to detect when conversations drift from technical discussion to pure comedy, so that meme threads are excluded while maintaining analytical humor.

#### Acceptance Criteria

1. WHEN a comment thread transitions from analytical content to pure jokes or memes, THE Relevance_Filter SHALL mark the transition point as IRRELEVANT
2. THE Relevance_Filter SHALL allow light humor if comments maintain Analytical_Content about autonomous surgery
3. WHEN multiple consecutive comments in a thread contain no Analytical_Content, THE Relevance_Filter SHALL mark them as conversational drift
4. THE Relevance_Filter SHALL distinguish between humor that illustrates a point and humor that derails discussion
5. WHEN drift is detected, THE Relevance_Filter SHALL prune the drifted subtree

### Requirement 13: Topic Relevance vs Sentiment Separation

**User Story:** As a researcher, I want relevance filtering to be independent of sentiment, so that both supportive and critical discussions about autonomous surgery are retained.

#### Acceptance Criteria

1. THE Relevance_Filter SHALL classify content based solely on topic relevance to autonomous/robotic surgery
2. THE Relevance_Filter SHALL NOT use sentiment polarity (positive, negative, neutral) in relevance decisions
3. THE Relevance_Filter SHALL NOT use stance (support, skepticism, opposition) in relevance decisions
4. WHEN content is critical or skeptical about autonomous surgery, THE Relevance_Filter SHALL retain it if topically relevant
5. THE Relevance_Filter SHALL pass all retained content to the Sentiment_Pipeline without sentiment pre-filtering

### Requirement 14: Evaluation and Validation

**User Story:** As a researcher, I want to validate the relevance filter against manually annotated data, so that I can report precision, recall, and inter-annotator agreement in my research methods.

#### Acceptance Criteria

1. THE Relevance_Filter SHALL support evaluation against manually annotated relevance datasets
2. WHEN evaluated, THE Relevance_Filter SHALL report precision and recall for RELEVANT vs IRRELEVANT classifications
3. THE Relevance_Filter SHALL support computing metrics for fine-grained Relevance_State categories (RELEVANT_STRONG, RELEVANT_INHERITED, RELEVANT_WEAK, IRRELEVANT)
4. THE Relevance_Filter SHALL provide output suitable for computing inter-annotator agreement (Cohen's kappa) on relevance judgments
5. THE Relevance_Filter SHALL log classification decisions with sufficient detail to support error analysis and debugging
