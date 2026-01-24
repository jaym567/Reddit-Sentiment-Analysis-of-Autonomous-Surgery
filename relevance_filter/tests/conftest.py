"""
Pytest configuration and shared fixtures for relevance filter tests.
"""

import pytest
from hypothesis import settings

# Configure hypothesis for property-based testing
settings.register_profile("default", max_examples=100, deadline=None)
settings.load_profile("default")


@pytest.fixture
def sample_post():
    """Sample Reddit post for testing."""
    return {
        'id': 'post_123',
        'subreddit': 'medicine',
        'title': 'Discussion about robotic surgery',
        'selftext': 'What are your thoughts on autonomous surgical systems?',
        'created_utc': 1234567890,
        'score': 42,
        'url': 'https://reddit.com/r/medicine/post_123',
        'permalink': '/r/medicine/comments/post_123',
        'comments': []
    }


@pytest.fixture
def sample_comment():
    """Sample Reddit comment for testing."""
    return {
        'id': 'comment_456',
        'author': 'user1',
        'body': 'I think robotic surgery has great potential.',
        'created_utc': 1234567900,
        'score': 10,
        'permalink': '/r/medicine/comments/post_123/comment_456',
        'replies': []
    }


@pytest.fixture
def sample_post_with_comments():
    """Sample Reddit post with nested comments for testing."""
    return {
        'id': 'post_789',
        'subreddit': 'medicine',
        'title': 'New da Vinci robot capabilities',
        'selftext': 'The latest da Vinci Xi system can perform complex procedures.',
        'created_utc': 1234567890,
        'score': 100,
        'url': 'https://reddit.com/r/medicine/post_789',
        'permalink': '/r/medicine/comments/post_789',
        'comments': [
            {
                'id': 'comment_1',
                'author': 'user1',
                'body': 'This is amazing technology!',
                'created_utc': 1234567900,
                'score': 20,
                'permalink': '/r/medicine/comments/post_789/comment_1',
                'replies': [
                    {
                        'id': 'comment_1_1',
                        'author': 'user2',
                        'body': 'I agree, the precision is incredible.',
                        'created_utc': 1234567910,
                        'score': 5,
                        'permalink': '/r/medicine/comments/post_789/comment_1_1',
                        'replies': []
                    }
                ]
            },
            {
                'id': 'comment_2',
                'author': 'user3',
                'body': 'What about the cost?',
                'created_utc': 1234567920,
                'score': 15,
                'permalink': '/r/medicine/comments/post_789/comment_2',
                'replies': []
            }
        ]
    }
