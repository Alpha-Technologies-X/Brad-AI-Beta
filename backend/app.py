from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import numpy as np
from datetime import datetime
import threading
import time
from models.model_loader import ModelLoader
from models.ml_processor import MLProcessor
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Initialize components
model_loader = ModelLoader()
ml_processor = MLProcessor()

# Store conversation history
conversation_history = {}
user_profiles = {}

# Available models with their specifications
AVAILABLE_MODELS = {
    "brad-ai-1.12.2x": {
        "name": "Brad AI 1.12.2x",
        "version": "1.12.2x",
        "description": "Standard model with balanced performance",
        "context_length": 4096,
        "training_data": "570GB",
        "parameters": "7B",
        "release_date": "2023-11-15",
        "special_features": ["General purpose", "Code generation", "Creative writing"]
    },
    "brad-ai-1.13.4r": {
        "name": "Brad AI 1.13.4r",
        "version": "1.13.4r",
        "description": "Reasoning-optimized model",
        "context_length": 8192,
        "training_data": "1.2TB",
        "parameters": "13B",
        "release_date": "2024-01-20",
        "special_features": ["Advanced reasoning", "Mathematical problem solving", "Logical analysis"]
    },
    "brad-ai-2.0.1a": {
        "name": "Brad AI 2.0.1a",
        "version": "2.0.1a",
        "description": "Advanced model with ML capabilities",
        "context_length": 16384,
        "training_data": "2.3TB",
        "parameters": "34B",
        "release_date": "2024-03-10",
        "special_features": ["Machine learning integration", "Personalization", "Contextual understanding"]
    },
    "brad-ai-2.1.3c": {
        "name": "Brad AI 2.1.3c",
        "version": "2.1.3c",
        "description": "Creative and conversational specialist",
        "context_length": 4096,
        "training_data": "890GB",
        "parameters": "7B",
        "release_date": "2024-04-05",
        "special_features": ["Creative writing", "Conversational AI", "Story generation"]
    },
    "brad-ai-2.2.0m": {
        "name": "Brad AI 2.2.0m",
        "version": "2.2.0m",
        "description": "Multimodal and technical model",
        "context_length": 32768,
        "training_data": "3.1TB",
        "parameters": "70B",
        "release_date": "2024-05-15",
        "special_features": ["Technical documentation", "Research assistance", "Data analysis"]
    }
}

@app.route('/api/models', methods=['GET'])
def get_models():
    """Get list of available models"""
    return jsonify({
        "models": AVAILABLE_MODELS,
        "default_model": "brad-ai-1.12.2x",
        "status": "success"
    })

@app.route('/api/chat', methods=['POST'])
def chat():
    """Main chat endpoint"""
    try:
        data = request.json
        user_message = data.get('message', '').strip()
        model_id = data.get('model', 'brad-ai-1.12.2x')
        user_id = data.get('user_id', 'default')
        
        if not user_message:
            return jsonify({"error": "Message is required"}), 400
        
        if model_id not in AVAILABLE_MODELS:
            return jsonify({"error": "Model not found"}), 404
        
        # Initialize conversation history for user if not exists
        if user_id not in conversation_history:
            conversation_history[user_id] = []
        
        # Store user message
        conversation_history[user_id].append({
            "role": "user",
            "message": user_message,
            "timestamp": datetime.now().isoformat(),
            "model": model_id
        })
        
        # Process with ML features
        ml_features = ml_processor.extract_features(user_message)
        
        # Get response based on model
        response = generate_response(user_message, model_id, user_id, ml_features)
        
        # Store assistant response
        conversation_history[user_id].append({
            "role": "assistant",
            "message": response,
            "timestamp": datetime.now().isoformat(),
            "model": model_id
        })
        
        # Update user profile with ML
        update_user_profile(user_id, user_message, response, ml_features)
        
        # Keep only last 50 messages
        if len(conversation_history[user_id]) > 50:
            conversation_history[user_id] = conversation_history[user_id][-50:]
        
        return jsonify({
            "response": response,
            "model": AVAILABLE_MODELS[model_id]["name"],
            "model_version": AVAILABLE_MODELS[model_id]["version"],
            "timestamp": datetime.now().isoformat(),
            "ml_insights": ml_features
        })
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        return jsonify({"error": str(e)}), 500

def generate_response(message, model_id, user_id, ml_features):
    """Generate response based on model and message"""
    
    # Get conversation context
    context = conversation_history.get(user_id, [])[-10:]  # Last 10 messages
    
    # Base responses with model-specific variations
    base_responses = {
        "brad-ai-1.12.2x": generate_standard_response,
        "brad-ai-1.13.4r": generate_reasoning_response,
        "brad-ai-2.0.1a": generate_ml_enhanced_response,
        "brad-ai-2.1.3c": generate_creative_response,
        "brad-ai-2.2.0m": generate_technical_response
    }
    
    generator = base_responses.get(model_id, generate_standard_response)
    return generator(message, context, ml_features)

def generate_standard_response(message, context, ml_features):
    """Standard model response"""
    responses = [
        f"I've analyzed your query about '{message}'. Based on my training, I can provide detailed information on this topic.",
        f"I understand you're asking: {message}. Let me break this down for you.",
        f"Thank you for your question. I'll provide a comprehensive response about this subject.",
        f"I can help with that! Here's what I know about '{message}'.",
        f"Based on your query, I'd like to share some insights."
    ]
    
    return np.random.choice(responses) + "\n\n" + get_detailed_response(message)

def generate_reasoning_response(message, context, ml_features):
    """Reasoning-optimized response"""
    reasoning_prompt = f"Let me reason through this step by step:\n\n"
    reasoning_prompt += f"1. Understanding the query: {message}\n"
    reasoning_prompt += f"2. Breaking down components\n"
    reasoning_prompt += f"3. Analyzing relationships\n"
    reasoning_prompt += f"4. Drawing conclusions\n\n"
    
    return reasoning_prompt + "Based on logical analysis, here's my response:\n" + get_detailed_response(message)

def generate_ml_enhanced_response(message, context, ml_features):
    """ML-enhanced response with personalization"""
    sentiment = ml_features.get('sentiment', 'neutral')
    topics = ml_features.get('topics', [])
    
    personalized = f"I notice this query has a {sentiment} sentiment"
    if topics:
        personalized += f" and relates to {', '.join(topics[:3])}"
    
    return f"{personalized}.\n\nAs Brad AI 2.0.1a with machine learning capabilities, I've analyzed your query pattern. " + get_detailed_response(message)

def generate_creative_response(message, context, ml_features):
    """Creative and conversational response"""
    creative_intros = [
        "What an interesting question! Let me weave some insights together...",
        "I love this topic! Here's a creative take on it:",
        "From a creative perspective, here's how I see it:",
        "Let me craft a thoughtful response for you:"
    ]
    
    return np.random.choice(creative_intros) + "\n\n" + get_detailed_response(message)

def generate_technical_response(message, context, ml_features):
    """Technical and detailed response"""
    technical_template = f"**Technical Analysis of: {message}**\n\n"
    technical_template += "**Key Components:**\n"
    technical_template += "- Query classification\n"
    technical_template += "- Contextual analysis\n"
    technical_template += "- Data correlation\n"
    technical_template += "- Inference generation\n\n"
    technical_template += "**Response:**\n"
    
    return technical_template + get_detailed_response(message)

def get_detailed_response(message):
    """Generate a detailed response based on message content"""
    message_lower = message.lower()
    
    if any(word in message_lower for word in ['hello', 'hi', 'hey']):
        return "Hello! I'm Brad AI, ready to assist you with various tasks. How can I help you today?"
    
    elif any(word in message_lower for word in ['how are you', 'how do you do']):
        return "I'm functioning optimally, thank you for asking! As an AI, I don't have feelings, but I'm fully operational and ready to help with your queries."
    
    elif any(word in message_lower for word in ['weather', 'temperature']):
        return "I don't have real-time weather data access, but I can help you understand meteorological concepts or analyze weather patterns historically."
    
    elif any(word in message_lower for word in ['joke', 'funny']):
        return "Why don't scientists trust atoms?\nBecause they make up everything! \n\nNow, how else can I assist you?"
    
    elif any(word in message_lower for word in ['machine learning', 'ml', 'ai']):
        return "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. Key concepts include supervised learning, unsupervised learning, neural networks, and deep learning. I can help explain these concepts in detail!"
    
    elif any(word in message_lower for word in ['python', 'code', 'programming']):
        return "I can help with Python programming! Here's a simple example:\n```python\ndef greet(name):\n    return f'Hello, {name}!'\n\nprint(greet('User'))\n```\nWould you like help with a specific programming task?"
    
    elif any(word in message_lower for word in ['explain', 'what is', 'tell me about']):
        topic = message.replace('explain', '').replace('what is', '').replace('tell me about', '').strip()
        return f"{topic.capitalize()} is a topic I can provide information about. Would you like me to go into more specific details?"
    
    else:
        return f"I've received your message about '{message}'. This appears to be a general inquiry. I'm capable of helping with:\n- Answering questions\n- Providing explanations\n- Generating creative content\n- Assisting with technical topics\n- Machine learning concepts\n\nHow would you like me to proceed with this topic?"

def update_user_profile(user_id, message, response, ml_features):
    """Update user profile with ML insights"""
    if user_id not in user_profiles:
        user_profiles[user_id] = {
            "interaction_count": 0,
            "topics": [],
            "average_sentiment": 0,
            "preferred_model": None,
            "last_interaction": datetime.now().isoformat()
        }
    
    profile = user_profiles[user_id]
    profile["interaction_count"] += 1
    profile["last_interaction"] = datetime.now().isoformat()
    
    # Update topics
    if 'topics' in ml_features:
        profile["topics"].extend(ml_features['topics'])
        profile["topics"] = list(set(profile["topics"]))[:10]  # Keep unique, limit to 10
    
    # Update sentiment average
    if 'sentiment_score' in ml_features:
        current_avg = profile["average_sentiment"]
        count = profile["interaction_count"]
        profile["average_sentiment"] = (current_avg * (count-1) + ml_features['sentiment_score']) / count

@app.route('/api/profile/<user_id>', methods=['GET'])
def get_profile(user_id):
    """Get user profile"""
    profile = user_profiles.get(user_id, {})
    return jsonify({
        "user_id": user_id,
        "profile": profile,
        "status": "success"
    })

@app.route('/api/history/<user_id>', methods=['GET'])
def get_history(user_id):
    """Get conversation history"""
    history = conversation_history.get(user_id, [])
    return jsonify({
        "user_id": user_id,
        "history": history[-20:],  # Last 20 messages
        "total_messages": len(history),
        "status": "success"
    })

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "Brad AI Chat API",
        "timestamp": datetime.now().isoformat(),
        "active_users": len(conversation_history),
        "models_loaded": len(AVAILABLE_MODELS)
    })

if __name__ == '__main__':
    logger.info("Starting Brad AI Server...")
    logger.info(f"Loaded {len(AVAILABLE_MODELS)} models")
    logger.info("Server running on http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)
