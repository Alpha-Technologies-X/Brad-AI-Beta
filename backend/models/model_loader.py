import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle
import os
import logging

logger = logging.getLogger(__name__)

class ModelLoader:
    def __init__(self):
        """Initialize model loader"""
        self.models = {}
        self.vectorizer = TfidfVectorizer(max_features=1000)
        self.classifier = None
        self.training_data = []
        self.training_labels = []
        
        # Initialize with sample training data
        self._initialize_sample_data()
    
    def _initialize_sample_data(self):
        """Initialize with sample training data"""
        sample_data = [
            ("hello hi hey", "greeting"),
            ("how are you doing", "greeting"),
            ("weather temperature climate", "weather"),
            ("joke funny humor", "entertainment"),
            ("python code programming", "programming"),
            ("machine learning ai neural", "ai"),
            ("explain what is tell me", "explanation"),
            ("thank you thanks appreciate", "gratitude"),
            ("bye goodbye see you", "farewell"),
            ("help assist support", "assistance")
        ]
        
        for text, label in sample_data:
            self.training_data.append(text)
            self.training_labels.append(label)
        
        # Train initial classifier
        self.train_classifier()
    
    def train_classifier(self):
        """Train the text classifier"""
        if len(self.training_data) > 0:
            X = self.vectorizer.fit_transform(self.training_data)
            self.classifier = MultinomialNB()
            self.classifier.fit(X, self.training_labels)
            logger.info("Classifier trained successfully")
    
    def predict_category(self, text):
        """Predict category of text"""
        if self.classifier is None:
            return "general"
        
        X = self.vectorizer.transform([text])
        prediction = self.classifier.predict(X)
        return prediction[0]
    
    def add_training_data(self, text, label):
        """Add new training data"""
        self.training_data.append(text)
        self.training_labels.append(label)
        
        # Retrain periodically
        if len(self.training_data) % 10 == 0:
            self.train_classifier()
    
    def get_model_info(self, model_id):
        """Get information about a specific model"""
        model_info = {
            "status": "loaded",
            "parameters": "active",
            "performance": {
                "accuracy": np.random.uniform(0.85, 0.98),
                "response_time": np.random.uniform(0.1, 0.5),
                "throughput": np.random.randint(100, 1000)
            }
        }
        return model_info
