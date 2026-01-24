"""
Semantic classifier for stage 2 relevance filtering.

This module implements semantic analysis using embeddings or LLM-based
classification to distinguish meaningful content from jokes and metaphors.
"""

import logging
from typing import Tuple, List
import numpy as np

from .models import RelevanceState

logger = logging.getLogger(__name__)


class SemanticClassifier:
    """
    Stage 2 filter using semantic analysis.
    
    This classifier uses embedding similarity or LLM-based classification to
    determine if content meaningfully discusses autonomous/robotic surgery.
    """
    
    def __init__(self, model_type: str = "embedding", model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the SemanticClassifier.
        
        Args:
            model_type: Type of model to use ("embedding" or "llm")
            model_name: Name of the model to load
        """
        self.model_type = model_type
        self.model_name = model_name
        
        if model_type == "embedding":
            self.model = self._load_embedding_model()
            self.pos_embeddings, self.neg_embeddings = self._compute_reference_embeddings()
        else:
            raise NotImplementedError("LLM-based classification not yet implemented")
    
    def _load_embedding_model(self):
        """Load the sentence-transformers embedding model."""
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading embedding model: {self.model_name}")
            model = SentenceTransformer(self.model_name)
            logger.info("Embedding model loaded successfully")
            return model
        except ImportError:
            logger.error("sentence-transformers library not installed. Install with: pip install sentence-transformers")
            raise
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
    
    def _compute_reference_embeddings(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute positive and negative reference embeddings.
        
        Positive texts represent core surgical concepts.
        Negative texts represent common noise (jokes, memes, sci-fi).
        
        Returns:
            Tuple of (positive_embeddings, negative_embeddings)
        """
        positive_texts = [
            "autonomous surgical robot performing operations independently",
            "robotic surgeon with artificial intelligence capabilities",
            "AI-assisted surgical procedures and automation",
            "da Vinci surgical robot system for minimally invasive procedures",
            "robot-assisted laparoscopic surgery with precision instruments",
            "surgical robotics platform with autonomous features",
            "autonomous operation of surgical instruments and tools",
            "machine learning for surgical decision making and planning",
            "computer vision and AI for surgical guidance",
            "robotic cholecystectomy and gallbladder removal",
            "robot-assisted prostatectomy and urological surgery",
            "minimally invasive robotic cardiac surgery",
            "surgical robot safety protocols and error prevention",
            "clinical outcomes of autonomous surgical systems",
            "FDA approval and regulation of surgical robotics",
            "autonomous surgery research and clinical trials",
            "surgical AI training data and machine learning models",
            "future of robotic surgery and automation",
            "surgery",
            "robot",
            "surgeon",
            "medical",
            "hospital",
            "clinical",
            "procedure",
            "operation",
            "treatment"
        ]
        
        negative_texts = [
            # Sci-fi and robots (Requirement 10.4)
            "R2-D2 and C-3PO adventures in space",
            "Terminator style robot uprising and apocalypse",
            "Isaac Asimov three laws of robotics in sci-fi books",
            "Optimus Prime and Transformers battle for Earth",
            "Giant robots fighting monsters in futuristic movies",
            
            # Jokes and Memes (Requirement 10.3 & 12)
            "I for one welcome our new robot overlords",
            "automate my organ harvesting operation for profit",
            "selling kidneys on the black market joke",
            "robot surgeons steal your data and your organs",
            "funny meme about robots failing at simple tasks",
            
            # Unrelated surgical humor/metaphors
            "surgical precision used as a metaphor for gaming",
            "cutting into the code like a surgeon",
            "doctor who style medical humor",
            "Grey's Anatomy drama and office romance"
        ]
        
        logger.info(f"Computing reference embeddings: {len(positive_texts)} pos, {len(negative_texts)} neg")
        pos_embeddings = self.model.encode(positive_texts, convert_to_numpy=True)
        neg_embeddings = self.model.encode(negative_texts, convert_to_numpy=True)
        
        return pos_embeddings, neg_embeddings
    
    def classify_local(self, text: str) -> Tuple[RelevanceState, float]:
        """
        Classify text using only its own content.
        
        Uses embedding similarity to determine if the text meaningfully
        discusses autonomous/robotic surgery.
        
        Args:
            text: Text to classify
            
        Returns:
            Tuple of (relevance_state, confidence_score)
        """
        if not text or not text.strip():
            return (RelevanceState.IRRELEVANT, 0.0)
        
        if self.model_type == "embedding":
            return self._embedding_classify(text)
        else:
            raise NotImplementedError("LLM-based classification not yet implemented")
    
    def classify_contextual(
        self,
        parent_text: str,
        child_text: str
    ) -> Tuple[RelevanceState, float]:
        """
        Classify child text in context of parent.
        
        Concatenates parent and child text to evaluate if the child
        continues the parent's discussion about autonomous surgery.
        
        Args:
            parent_text: Text of the parent comment/post
            child_text: Text of the child comment
            
        Returns:
            Tuple of (relevance_state, confidence_score)
        """
        # Format concatenated text as specified in design (Requirement 4.3 exact)
        concatenated = f"PARENT CONTEXT:\n{parent_text}\n\nCHILD COMMENT:\n{child_text}\n"
        
        # Use the same classification logic as local, but on concatenated text
        return self.classify_local(concatenated)
    
    def _embedding_classify(self, text: str) -> Tuple[RelevanceState, float]:
        """
        Classify using embedding similarity.
        
        Computes the embedding for the input text and compares it with
        reference embeddings using cosine similarity.
        
        Args:
            text: Text to classify
            
        Returns:
            Tuple of (relevance_state, confidence_score)
        """
        try:
            # Compute text embedding
            text_embedding = self.model.encode(text, convert_to_numpy=True)
            
            # Compute cosine similarity with positive reference embeddings
            pos_similarities = self._cosine_similarity(
                text_embedding.reshape(1, -1),
                self.pos_embeddings
            )[0]
            max_pos_sim = float(np.max(pos_similarities))
            
            # Compute cosine similarity with negative reference embeddings
            neg_similarities = self._cosine_similarity(
                text_embedding.reshape(1, -1),
                self.neg_embeddings
            )[0]
            max_neg_sim = float(np.max(neg_similarities))
            
            # Ensure scores are in [0, 1] range (cosine similarity can be slightly negative)
            max_pos_sim = max(0.0, min(1.0, max_pos_sim))
            max_neg_sim = max(0.0, min(1.0, max_neg_sim))
            
            logger.debug(f"Similarity - Pos: {max_pos_sim:.3f}, Neg: {max_neg_sim:.3f}")
            
            # Decision Logic:
            # Only reject if negative similarity is extremely high (>0.85)
            if max_neg_sim > 0.85 and max_neg_sim > (max_pos_sim + 0.05):
                logger.debug(f"Rejected due to extremely high negative similarity ({max_neg_sim})")
                return (RelevanceState.IRRELEVANT, max_pos_sim)
            
            # Classify based on similarity thresholds (Balanced for Recovery logic)
            if max_pos_sim >= 0.6:
                return (RelevanceState.RELEVANT_STRONG, max_pos_sim)
            elif max_pos_sim >= 0.4:
                return (RelevanceState.RELEVANT_WEAK, max_pos_sim)
            else:
                return (RelevanceState.IRRELEVANT, max_pos_sim)
                
        except Exception as e:
            logger.error(f"Error during embedding classification: {e}")
            # Return irrelevant with low confidence on error
            return (RelevanceState.IRRELEVANT, 0.0)
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarity between two sets of vectors.
        
        Args:
            a: Array of shape (n, d) where n is number of vectors and d is dimension
            b: Array of shape (m, d) where m is number of vectors and d is dimension
            
        Returns:
            Array of shape (n, m) containing cosine similarities
        """
        # Normalize vectors (add small epsilon to avoid division by zero)
        a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
        b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
        
        # Compute dot product (cosine similarity for normalized vectors)
        return np.dot(a_norm, b_norm.T)
