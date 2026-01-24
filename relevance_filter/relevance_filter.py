"""
Main orchestrator for the Reddit Relevance Filter.

This module contains the RelevanceFilter class that coordinates the entire
filtering pipeline.
"""

import logging
from typing import Dict, List, Optional

from .models import FilterConfig, FilteredItem, ParentContext, RelevanceState

logger = logging.getLogger(__name__)


class RelevanceFilter:
    """
    Main orchestrator for relevance filtering of Reddit posts and comments.
    
    This class coordinates the entire filtering pipeline, including post-level
    filtering, recursive comment tree traversal, and output collection.
    """
    
    def __init__(self, keyword_filter, semantic_classifier, config: FilterConfig):
        """
        Initialize the RelevanceFilter.
        
        Args:
            keyword_filter: KeywordFilter instance for stage 1 filtering
            semantic_classifier: SemanticClassifier instance for stage 2 filtering
            config: FilterConfig with configuration parameters
        """
        self.keyword_filter = keyword_filter
        self.semantic_classifier = semantic_classifier
        self.config = config
        
        # Import components here to avoid circular dependencies if any
        from .analytical_content_detector import AnalyticalContentDetector
        from .concatenation_decider import ConcatenationDecider
        from .comment_evaluator import CommentEvaluator
        
        self.analytical_detector = AnalyticalContentDetector()
        self.concatenation_decider = ConcatenationDecider(
            word_threshold=config.concatenation_word_threshold
        )
        self.comment_evaluator = CommentEvaluator(
            keyword_filter=self.keyword_filter,
            semantic_classifier=self.semantic_classifier,
            analytical_detector=self.analytical_detector,
            concatenation_decider=self.concatenation_decider
        )
        
        # Configure logging
        logging.basicConfig(
            level=getattr(logging, config.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        logger.info("RelevanceFilter initialized with all sub-components")
    
    def filter_posts(self, posts: List[Dict]) -> List[Dict]:
        """
        Filter a batch of Reddit posts and their comment trees.
        
        Args:
            posts: List of Reddit post dictionaries
            
        Returns:
            List of filtered items (posts and comments) as dictionaries
        """
        all_results = []
        for post in posts:
            post_results = self.filter_single_post(post)
            if post_results:
                all_results.extend(post_results)
        
        logger.info(f"Filtered {len(posts)} posts into {len(all_results)} relevant items")
        return all_results
    
    def filter_single_post(self, post: Dict) -> List[Dict]:
        """
        Process a single post and its comment tree.
        """
        results = []
        subreddit = post.get('subreddit')
        
        # Evaluate post
        is_relevant, score, reason = self.evaluate_post(post)
        post_id = post.get('id', 'unknown')
        title = post.get('title', '')
        selftext = post.get('selftext', '')
        selftext = selftext if selftext else ""
        combined_text = f"TITLE: {title}\n\nBODY: {selftext}"

        if is_relevant:
            # Create FilteredItem for post
            post_item = FilteredItem(
                id=post_id,
                type="post",
                text=combined_text,
                parent_id=None,
                post_id=post_id,
                depth=0,
                relevance_score=score,
                relevance_reason=reason,
                relevance_state=RelevanceState.RELEVANT_STRONG.value,
                created_utc=post.get('created_utc')
            )
            results.append(post_item.to_dict())
            
            # Create parent context for comments
            parent_context = ParentContext(
                id=post_id,
                text=combined_text,
                relevance_state=RelevanceState.RELEVANT_STRONG,
                relevance_score=score,
                depth=0,
                post_id=post_id
            )
        else:
            # Post itself is not relevant, but we still want to check comments
            # for "RELEVANT_STRONG" content (Topic Shift Recovery)
            parent_context = ParentContext(
                id=post_id,
                text=combined_text,
                relevance_state=RelevanceState.IRRELEVANT,
                relevance_score=score,
                depth=0,
                post_id=post_id
            )
            
        # Process comments recursively
        comments = post.get('comments', [])
        if comments:
            tree_results = self.process_comment_tree(
                comments,
                post_id,
                parent_context,
                subreddit=subreddit
            )
            results.extend(tree_results)
            
        return results
    
    def evaluate_post(self, post: Dict) -> tuple:
        """
        Evaluate post-level relevance.
        """
        title = post.get('title', '')
        selftext = post.get('selftext', '')
        selftext = selftext if selftext else ""
        combined_text = f"{title} {selftext}"
        subreddit = post.get('subreddit')
        
        # Stage 0: Strong Irrelevance Check (Exclusion Keywords + Noise Subs)
        if self.analytical_detector.is_strongly_irrelevant(combined_text, subreddit):
            return False, 1.0, "exclusion_keyword"

        # Stage 1: Mandatory Topic Intersection for Posts
        if not self.analytical_detector.has_topic_intersection(combined_text, subreddit):
            return False, 0.5, "no_intersection"

        # Stage 2: Keyword matching (Requirement 1.2)
        if self.keyword_filter.matches(combined_text):
            return True, 1.0, "keyword"
            
        # Stage 3: Semantic matching (Requirement 1.3)
        relevance_state, score = self.semantic_classifier.classify_local(combined_text)
        if relevance_state in [RelevanceState.RELEVANT_STRONG, RelevanceState.RELEVANT_WEAK]:
            return True, score, "semantic"
            
        return False, score, "irrelevant"
    
    def process_comment_tree(
        self,
        comments: List[Dict],
        post_id: str,
        parent_context: Optional[ParentContext] = None,
        consecutive_drift_count: int = 0,
        subreddit: Optional[str] = None
    ) -> List[Dict]:
        """
        Recursively process comment tree with context inheritance and drift pruning.
        """
        results = []
        if not isinstance(comments, list):
            logger.warning(f"Expected list for comments, got {type(comments)}")
            return results
            
        for comment in comments:
            # Skip if comment is not a dictionary
            if not isinstance(comment, dict):
                logger.warning(f"Expected dict for comment, got {type(comment)}")
                continue
                
            # Evaluate individual comment
            depth = (parent_context.depth + 1) if parent_context else 1
            evaluation = self.comment_evaluator.evaluate(comment, parent_context, depth, subreddit=subreddit)
            
            relevance_state_str = evaluation['relevance_state']
            relevance_state = RelevanceState(relevance_state_str)
            
            # Update drift count (Requirement: Draft Pruning)
            # A comment is "drift" if it has no domain tokens AND no analytical content
            # Reusing detector logic via intersection or broad domain list
            has_domain_terms = self.analytical_detector.has_topic_intersection(evaluation['text'], subreddit=subreddit) or \
                              any(term in evaluation['text'].lower() for term in self.analytical_detector.domain_terms)
            
            if not evaluation['has_analytical_content'] and not has_domain_terms:
                new_drift_count = consecutive_drift_count + 1
            else:
                new_drift_count = 0
                
            # If relevant (at any level), include in results (Requirement 8.4)
            if relevance_state != RelevanceState.IRRELEVANT:
                # Use FilteredItem to ensure schema consistency
                item = FilteredItem(
                    id=evaluation['id'],
                    type='comment',
                    text=evaluation['text'],
                    parent_id=evaluation['parent_id'],
                    post_id=post_id,
                    depth=evaluation['depth'],
                    relevance_score=evaluation['relevance_score'],
                    relevance_reason=evaluation['relevance_reason'],
                    relevance_state=relevance_state_str,
                    created_utc=comment.get('created_utc')
                )
                results.append(item.to_dict())
                
                # Recursively process replies if NOT at drift limit
                replies = comment.get('replies', [])
                if replies and new_drift_count < self.config.drift_threshold:
                    child_context = ParentContext(
                        id=evaluation['id'],
                        text=evaluation['text'],
                        relevance_state=relevance_state,
                        relevance_score=evaluation['relevance_score'],
                        depth=evaluation['depth'],
                        post_id=post_id
                    )
                    results.extend(self.process_comment_tree(replies, post_id, child_context, new_drift_count, subreddit=subreddit))
            else:
                # Recovery logic: Even if IRRELEVANT, if NOT at drift limit, process replies
                if new_drift_count < self.config.drift_threshold:
                    replies = comment.get('replies', [])
                    if replies:
                        child_context = ParentContext(
                            id=evaluation['id'],
                            text=evaluation['text'],
                            relevance_state=relevance_state,
                            relevance_score=evaluation['relevance_score'],
                            depth=evaluation['depth'],
                            post_id=post_id
                        )
                        results.extend(self.process_comment_tree(replies, post_id, child_context, new_drift_count, subreddit=subreddit))
        
        return results
