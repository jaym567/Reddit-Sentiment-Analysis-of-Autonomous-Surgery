"""
Keyword filter for stage 1 relevance filtering.

This module implements high-recall keyword matching to identify potentially
relevant content about autonomous/robotic surgery.
"""

import logging
from typing import List

logger = logging.getLogger(__name__)


class KeywordFilter:
    """
    Stage 1 filter using high-recall keyword matching.
    
    This filter uses broad keyword matching to identify potentially relevant
    content, which is then validated by semantic classification.
    """
    
    def __init__(self):
        """Initialize the KeywordFilter with keyword lists."""
        # Surgery-related keywords (Requirement 9.1)
        self.surgery_keywords = [
            "autonomous surgery",
            "robotic surgeon",
            "surgical robot",
            "da vinci",
            "laparoscopic robot",
            "surgical automation",
            "ai surgery",
            "robot-assisted surgery",
            "robotic surgery",
            "cholecystectomy",
            "operating room ai",
            "minimally invasive robotics"
        ]
        
        # Surgical procedure keywords (Requirement 9.2)
        self.procedure_keywords = [
            "cholecystectomy",
            "prostatectomy",
            "hysterectomy",
            "cardiac surgery",
            "minimally invasive"
        ]
        
        # Robotics and AI context keywords (Requirement 9.3)
        self.context_keywords = [
            "operating room ai",
            "surgical ai",
            "autonomous operation",
            "robotic precision",
            "surgical robotics",
            "surger",
            "robot",
            "medic",
            "doctor",
            "procedure",
            "automation",
            "precision",
            "safety",
            "ai"
        ]
        
        # Combine all keywords for efficient matching and de-duplicate
        self.all_keywords = sorted(list(set(
            self.surgery_keywords + 
            self.procedure_keywords + 
            self.context_keywords
        )))
        
        logger.debug(f"KeywordFilter initialized with {len(self.all_keywords)} keywords")
    
    def matches(self, text: str) -> bool:
        """
        Check if text contains any relevant keywords.
        
        Uses case-insensitive matching (Requirement 9.4).
        
        Args:
            text: Text to check for keywords
            
        Returns:
            True if text contains any relevant keywords, False otherwise
        """
        if not text:
            return False
        
        # Convert to lowercase for case-insensitive matching (Requirement 9.4)
        text_lower = str(text).lower()
        
        # Check if any keyword is present in the text
        for keyword in self.all_keywords:
            if keyword.lower() in text_lower:
                logger.debug(f"Keyword match found: '{keyword}'")
                return True
        
        return False
    
    def get_matched_keywords(self, text: str) -> List[str]:
        """
        Return list of matched keywords for debugging.
        
        Args:
            text: Text to check for keywords
            
        Returns:
            List of matched keywords
        """
        if not text:
            return []
        
        # Convert to lowercase for case-insensitive matching
        text_lower = text.lower()
        
        # Collect all matched keywords
        matched = []
        for keyword in self.all_keywords:
            if keyword.lower() in text_lower:
                matched.append(keyword)
        
        return matched
