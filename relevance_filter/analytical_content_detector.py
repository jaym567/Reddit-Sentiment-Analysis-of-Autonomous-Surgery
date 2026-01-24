"""
Analytical content detector for pruning decisions.

This module detects analytical content markers to inform subtree pruning
decisions, distinguishing substantive discussion from pure jokes/memes.
"""

import logging
import re
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class AnalyticalContentDetector:
    """
    Detector for analytical content markers.
    
    This class identifies epistemic verbs, causal markers, and domain-specific
    terms that indicate analytical content worth preserving.
    """
    
    def __init__(self):
        """Initialize the AnalyticalContentDetector with marker lists."""
        # Epistemic/evaluative verbs indicating analytical thinking
        self.epistemic_verbs = [
            "think", "believe", "seems", "appears", "would worry",
            "probably", "likely", "suggest", "indicate", "assume",
            "say", "mention", "feel", "wonder", "see", "saw", "know",
            "awesome", "cool", "great", "interesting", "nice"
        ]
        
        # Causal markers indicating reasoning
        self.causal_markers = [
            "because", "therefore", "so that", "this means",
            "as a result", "consequently", "thus", "hence",
            "due to", "lead to", "result in", "why", "how"
        ]
        
        # Sub-domain terms for intersection check (Expanded from Audit)
        self.medical_terms = [
            "surgeon", "patient", "surgery", "surger", "operating room", 
            "hospital", "anesthesia", "medical", "doctor", "organ", 
            "tissue", "clinical", "medic", "treatment", "procedure",
            "operation", "ectomy", "otomy", "oscop", "plasty", "incision",
            "appendix", "liver", "spleen", "innards", "scapel", "scalpel",
            "healthcare", "physician", "doctor", "clinical", "diagnostic"
        ]
        
        self.ai_robot_terms = [
            "ai", "robot", "autonomous", "automation", "algorithm", 
            "system", "da vinci", "robotic", "intelligence", "training",
            "machine learning", "computer vision", "neural network",
            "machine", "automation", "artificial intelligence"
        ]

        # Subreddits that inherently provide domain context
        self.medical_subreddits = [
            "surgery", "medicine", "askdocs", "hysterectomy", "fibroids", 
            "hernia", "breastcancer", "ostomy", "healthcare", "medical"
        ]

        self.noise_subreddits = [
            "hfy", "worldbuilding", "stellaris", "fanfiction", "nosleep"
        ]

        # Exclusion keywords (Refined from Audit)
        self.exclusion_keywords = [
            "israel", "palestine", "gaza", "quran", "islam", "religion",
            "politics", "election", "democrat", "republican",
            "valet", "parking", "car", "truck", "driving", "fsd",
            "wakamo", "ritsuka", "kiyohime", "osakabehime", "servant", 
            "manga", "anime", "fanfic", "story", "chapter", "fiction",
            "udemy", "coupon", "discount", "redeem", "offer"
        ]
        
        # Combined list for general check
        self.domain_terms = list(set(self.medical_terms + self.ai_robot_terms))
        
        logger.debug("AnalyticalContentDetector initialized with expanded exclusion lists")

    def is_strongly_irrelevant(self, text: str, subreddit: Optional[str] = None) -> bool:
        """Check if text or subreddit contains exclusion markers."""
        if subreddit and subreddit.lower() in self.noise_subreddits:
            logger.info(f"Subreddit {subreddit} is in noise list.")
            return True
            
        if not text:
            return False
            
        text_lower = text.lower()
        for kw in self.exclusion_keywords:
            if re.search(r'\b' + re.escape(kw) + r'\b', text_lower):
                logger.info(f"Strongly irrelevant keyword found: {kw}")
                return True
        return False
    
    def has_topic_intersection(self, text: str, subreddit: Optional[str] = None, max_distance_words: int = 500) -> bool:
        """
        Check if text discusses both medical and AI/robot domains.
        """
        if not text:
            return False
            
        text_lower = text.lower()
        
        # Identify core domains
        has_medical = any(term in text_lower for term in self.medical_terms)
        if subreddit and subreddit.lower() in self.medical_subreddits:
            has_medical = True
            
        has_ai = any(term in text_lower for term in self.ai_robot_terms)
        
        if not has_medical or not has_ai:
            return False

        # --- RECALL OVERRIDE: Title Match ---
        # If the start of the text (likely a TITLE if coming from RelevanceFilter)
        # matches both, we accept it immediately.
        if "TITLE:" in text and any(m in text[:300].lower() for m in self.medical_terms) \
           and any(a in text[:300].lower() for a in self.ai_robot_terms):
            return True

        # --- ACCURACY: Proximity Check for long texts ---
        words = text_lower.split()
        if len(words) < 75:  # Loosened from 50
            return True

        medical_indices = [i for i, w in enumerate(words) if any(term in w for term in self.medical_terms)]
        ai_indices = [i for i, w in enumerate(words) if any(term in w for term in self.ai_robot_terms)]
        
        for m_idx in medical_indices:
            for a_idx in ai_indices:
                if abs(m_idx - a_idx) <= max_distance_words:
                    return True
        
        return False

    def has_analytical_content(self, text: str) -> bool:
        """
        Check if text contains analytical markers.
        
        Returns True if text contains at least one of:
        - Epistemic/evaluative verbs
        - Causal markers
        - Domain-specific medical/AI/ethics terms
        
        Args:
            text: Text to check for analytical content
            
        Returns:
            True if text contains analytical markers, False otherwise
        """
        if not text:
            return False
        
        text_lower = text.lower()
        
        # Check epistemic verbs with word boundaries
        for verb in self.epistemic_verbs:
            if re.search(r'\b' + re.escape(verb) + r'\b', text_lower):
                logger.debug(f"Found epistemic verb: {verb}")
                return True
        
        # Check causal markers (no word boundaries needed for phrases)
        for marker in self.causal_markers:
            if marker in text_lower:
                logger.debug(f"Found causal marker: {marker}")
                return True
        
        # Check domain terms (substring matching for plurals/variants)
        for term in self.domain_terms:
            if term in text_lower:
                logger.debug(f"Found domain term: {term}")
                return True
        
        return False
    
    def get_analytical_markers(self, text: str) -> Dict[str, List[str]]:
        """
        Return matched markers by category for debugging.
        
        Args:
            text: Text to check for analytical markers
            
        Returns:
            Dictionary mapping marker categories to lists of matched markers:
            {
                'epistemic_verbs': [...],
                'causal_markers': [...],
                'domain_terms': [...]
            }
        """
        if not text:
            return {
                'epistemic_verbs': [],
                'causal_markers': [],
                'domain_terms': []
            }
        
        text_lower = text.lower()
        matched = {
            'epistemic_verbs': [],
            'causal_markers': [],
            'domain_terms': []
        }
        
        # Find all matched epistemic verbs
        for verb in self.epistemic_verbs:
            if re.search(r'\b' + re.escape(verb) + r'\b', text_lower):
                matched['epistemic_verbs'].append(verb)
        
        # Find all matched causal markers
        for marker in self.causal_markers:
            if marker in text_lower:
                matched['causal_markers'].append(marker)
        
        # Find all matched domain terms
        for term in self.domain_terms:
            if re.search(r'\b' + re.escape(term) + r'\b', text_lower):
                matched['domain_terms'].append(term)
        
        logger.debug(f"Analytical markers found: {matched}")
        return matched
