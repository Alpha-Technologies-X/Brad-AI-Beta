import re
import numpy as np
from collections import Counter
import logging

logger = logging.getLogger(__name__)

class MLProcessor:
    def __init__(self):
        """Initialize ML Processor"""
        self.sentiment_lexicon = {
            'good': 1, 'great': 2, 'excellent': 3, 'bad': -1, 'terrible': -2,
            'happy': 2, 'sad': -2, 'love': 3, 'hate': -3, 'like': 1, 'dislike': -1,
            'awesome': 3, 'awful': -3, 'fantastic': 3, 'horrible': -3,
            'wonderful': 2, 'terrible': -2, 'perfect': 3, 'worst': -3
        }
        
        self.topic_keywords = {
            'technology': ['computer', 'software', 'hardware', 'code', 'program', 'tech'],
            'science': ['science', 'research', 'experiment', 'theory', 'scientific'],
            'education': ['learn', 'study', 'teach', 'school', 'university', 'education'],
            'business': ['business', 'company', 'market', 'finance', 'investment'],
            'health': ['health', 'medical', 'doctor', 'hospital', 'medicine'],
            'entertainment': ['movie', 'music', 'game', 'entertainment', 'fun'],
            'sports': ['sport', 'game', 'team', 'player', 'score'],
            'food': ['food', 'cook', 'recipe', 'meal', 'restaurant']
        }
    
    def extract_features(self, text):
        """Extract ML features from text"""
        features = {
            'length': len(text),
            'word_count': len(text.split()),
            'sentiment': self.analyze_sentiment(text),
            'sentiment_score': self.calculate_sentiment_score(text),
            'topics': self.extract_topics(text),
            'complexity': self.calculate_complexity(text),
            'has_question': '?' in text,
            'has_exclamation': '!' in text,
            'unique_words': len(set(text.lower().split())),
            'avg_word_length': np.mean([len(word) for word in text.split()]) if text.split() else 0
        }
        
        return features
    
    def analyze_sentiment(self, text):
        """Analyze text sentiment"""
        score = self.calculate_sentiment_score(text)
        
        if score > 0.1:
            return 'positive'
        elif score < -0.1:
            return 'negative'
        else:
            return 'neutral'
    
    def calculate_sentiment_score(self, text):
        """Calculate sentiment score"""
        words = re.findall(r'\b\w+\b', text.lower())
        total_score = 0
        matched_words = 0
        
        for word in words:
            if word in self.sentiment_lexicon:
                total_score += self.sentiment_lexicon[word]
                matched_words += 1
        
        if matched_words > 0:
            return total_score / (matched_words * 3)  # Normalize to [-1, 1]
        return 0
    
    def extract_topics(self, text):
        """Extract topics from text"""
        text_lower = text.lower()
        topics_found = []
        
        for topic, keywords in self.topic_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    topics_found.append(topic)
                    break
        
        # Return unique topics
        return list(set(topics_found))[:3]  # Limit to 3 topics
    
    def calculate_complexity(self, text):
        """Calculate text complexity score"""
        words = text.split()
        if not words:
            return 0
        
        # Simple complexity measure
        long_words = sum(1 for word in words if len(word) > 6)
        sentence_count = text.count('.') + text.count('!') + text.count('?')
        
        if sentence_count == 0:
            sentence_count = 1
        
        complexity = (len(words) / sentence_count) * 0.3 + (long_words / len(words)) * 0.7
        return min(complexity, 1.0)
    
    def generate_embeddings(self, text):
        """Generate simple text embeddings"""
        words = text.lower().split()
        word_count = len(words)
        
        if word_count == 0:
            return np.zeros(10)
        
        # Create a simple embedding based on character distributions
        embeddings = []
        
        # Character distribution
        for char in 'abcdefghijklmnopqrstuvwxyz':
            embeddings.append(text.lower().count(char) / len(text) if len(text) > 0 else 0)
        
        # Text statistics
        embeddings.append(len(text) / 1000)  # Normalized length
        embeddings.append(word_count / 100)   # Normalized word count
        embeddings.append(self.calculate_complexity(text))
        
        return np.array(embeddings[:10])  # Return first 10 features
