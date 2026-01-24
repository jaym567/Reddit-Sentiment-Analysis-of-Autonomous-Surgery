# Reddit Relevance Filter

A context-aware filtering module for Reddit posts and comments that identifies content relevant to autonomous, robotic, and AI-assisted surgery research.

## Overview

The Reddit Relevance Filter is a preprocessing module that filters Reddit discussions to retain only content relevant to autonomous surgery research. It uses a two-stage filtering approach (keyword + semantic) combined with context inheritance to handle conversational discourse patterns.

## Features

- **Two-Stage Filtering**: High-recall keyword filtering followed by precision-focused semantic classification
- **Context Inheritance**: Child comments inherit relevance from parents unless they explicitly diverge
- **Tree-Aware Processing**: Treats Reddit data as hierarchical comment trees where context flows from parent to child
- **Analytical Content Detection**: Distinguishes substantive discussion from pure jokes/memes using linguistic markers
- **Subtree Pruning**: Efficiently discards irrelevant branches while allowing recovery of relevant content

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

```python
from relevance_filter import RelevanceFilter, FilterConfig
from relevance_filter.keyword_filter import KeywordFilter
from relevance_filter.semantic_classifier import SemanticClassifier

# Initialize components
config = FilterConfig(
    semantic_model_type="embedding",
    similarity_threshold=0.7,
    log_level="INFO"
)

keyword_filter = KeywordFilter()
semantic_classifier = SemanticClassifier(model_type="embedding")

# Create filter
filter = RelevanceFilter(keyword_filter, semantic_classifier, config)

# Filter posts
import json
with open('reddit_data.json', 'r') as f:
    posts = json.load(f)

filtered_results = filter.filter_posts(posts)

# Process results
for item in filtered_results:
    print(f"{item['type']}: {item['text'][:50]}... ({item['relevance_state']})")
```

## Architecture

The module consists of the following components:

- **RelevanceFilter**: Main orchestrator that coordinates the filtering pipeline
- **KeywordFilter**: Stage 1 filter using high-recall keyword matching
- **SemanticClassifier**: Stage 2 filter using semantic analysis (embedding or LLM-based)
- **AnalyticalContentDetector**: Detects analytical content markers for pruning decisions
- **ConcatenationDecider**: Determines when to concatenate parent and child text
- **CommentEvaluator**: Evaluates individual comments with full context

## Data Models

- **RelevanceState**: Enum for relevance states (RELEVANT_STRONG, RELEVANT_INHERITED, RELEVANT_WEAK, IRRELEVANT)
- **ParentContext**: Context passed from parent to child during tree traversal
- **FilteredItem**: Output format for filtered content
- **FilterConfig**: Configuration parameters for the filter

## Testing

Run the test suite:

```bash
# Run all tests
pytest relevance_filter/tests/

# Run with coverage
pytest --cov=relevance_filter relevance_filter/tests/

# Run property-based tests only
pytest -k "property" relevance_filter/tests/
```

## Development Status

This module is currently under development. See `.kiro/specs/reddit-relevance-filter/tasks.md` for the implementation plan.

## License

[Add license information here]
