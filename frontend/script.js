class BradAIChat {
    constructor() {
        this.API_URL = 'http://localhost:5000/api';
        this.currentModel = 'brad-ai-1.12.2x';
        this.userId = this.generateUserId();
        this.isLoading = false;
        
        this.initializeElements();
        this.initializeEventListeners();
        this.loadModels();
        this.updateUserProfile();
        this.checkAPIStatus();
    }
    
    generateUserId() {
        return 'user_' + Math.random().toString(36).substr(2, 9);
    }
    
    initializeElements() {
        this.elements = {
            modelList: document.getElementById('modelList'),
            messageInput: document.getElementById('messageInput'),
            sendButton: document.getElementById('sendButton'),
            clearButton: document.getElementById('clearButton'),
            chatMessages: document.getElementById('chatMessages'),
            loadingOverlay: document.getElementById('loadingOverlay'),
            currentModelName: document.getElementById('currentModelName'),
            modelDescription: document.getElementById('modelDescription'),
            interactionCount: document.getElementById('interactionCount'),
            currentModel: document.getElementById('currentModel'),
            sentimentInsight: document.getElementById('sentimentInsight'),
            topicInsight: document.getElementById('topicInsight'),
            complexityInsight: document.getElementById('complexityInsight'),
            apiStatus: document.getElementById('apiStatus')
        };
        
        // Auto-resize textarea
        this.elements.messageInput.addEventListener('input', () => {
            this.autoResizeTextarea();
        });
    }
    
    autoResizeTextarea() {
        const textarea = this.elements.messageInput;
        textarea.style.height = 'auto';
        textarea.style.height = Math.min(textarea.scrollHeight, 120) + 'px';
    }
    
    initializeEventListeners() {
        // Send message on button click
        this.elements.sendButton.addEventListener('click', () => this.sendMessage());
        
        // Send message on Enter (without Shift)
        this.elements.messageInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });
        
        // Clear chat
        this.elements.clearButton.addEventListener('click', () => this.clearChat());
    }
    
    async loadModels() {
        try {
            const response = await fetch(`${this.API_URL}/models`);
            const data = await response.json();
            
            if (data.status === 'success') {
                this.models = data.models;
                this.renderModels();
                this.updateModelInfo(this.currentModel);
            }
        } catch (error) {
            console.error('Error loading models:', error);
            this.showError('Failed to load models. Using default configuration.');
        }
    }
    
    renderModels() {
        this.elements.modelList.innerHTML = '';
        
        Object.entries(this.models).forEach(([id, model]) => {
            const modelElement = document.createElement('div');
            modelElement.className = `model-item ${id === this.currentModel ? 'active' : ''}`;
            modelElement.innerHTML = `
                <div class="model-name">
                    ${model.name}
                    <span class="model-version">${model.version}</span>
                </div>
                <div class="model-desc">${model.description}</div>
                <div class="model-features">
                    ${model.special_features.map(feature => 
                        `<span class="model-feature">${feature}</span>`
                    ).join('')}
                </div>
            `;
            
            modelElement.addEventListener('click', () => this.switchModel(id));
            this.elements.modelList.appendChild(modelElement);
        });
    }
    
    switchModel(modelId) {
        this.currentModel = modelId;
        this.renderModels();
        this.updateModelInfo(modelId);
        
        // Add system message about model switch
        this.addMessage({
            role: 'system',
            content: `Switched to ${this.models[modelId].name}`,
            timestamp: new Date().toISOString()
        });
    }
    
    updateModelInfo(modelId) {
        const model = this.models[modelId];
        if (!model) return;
        
        this.elements.currentModelName.textContent = model.name;
        this.elements.modelDescription.textContent = model.description;
        this.elements.currentModel.textContent = model.version;
        
        // Update specs in header
        const specs = document.querySelector('.model-specs');
        specs.innerHTML = `
            <span class="spec"><i class="fas fa-database"></i> ${model.training_data} Training</span>
            <span class="spec"><i class="fas fa-memory"></i> ${model.parameters} Parameters</span>
            <span class="spec"><i class="fas fa-exchange-alt"></i> ${model.context_length.toLocaleString()} Context</span>
        `;
    }
    
    async sendMessage() {
        const message = this.elements.messageInput.value.trim();
        if (!message || this.isLoading) return;
        
        // Add user message to chat
        this.addMessage({
            role: 'user',
            content: message,
            timestamp: new Date().toISOString()
        });
        
        // Clear input
        this.elements.messageInput.value = '';
        this.autoResizeTextarea();
        
        // Show loading
        this.setLoading(true);
        
        try {
            const response = await fetch(`${this.API_URL}/chat`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    message: message,
                    model: this.currentModel,
                    user_id: this.userId
                })
            });
            
            const data = await response.json();
            
            if (data.error) {
                throw new Error(data.error);
            }
            
            // Add assistant response
            this.addMessage({
                role: 'assistant',
                content: data.response,
                timestamp: data.timestamp,
                model: data.model,
                model_version: data.model_version,
                ml_insights: data.ml_insights
            });
            
            // Update ML insights
            this.updateMLInsights(data.ml_insights);
            
            // Update user profile
            this.updateUserProfile();
            
        } catch (error) {
            console.error('Error sending message:', error);
            this.addMessage({
                role: 'system',
                content: `Error: ${error.message}`,
                timestamp: new Date().toISOString(),
                isError: true
            });
        } finally {
            this.setLoading(false);
        }
    }
    
    addMessage(messageData) {
        const messageElement = document.createElement('div');
        messageElement.className = `message ${messageData.role}`;
        
        const time = new Date(messageData.timestamp).toLocaleTimeString([], { 
            hour: '2-digit', 
            minute: '2-digit' 
        });
        
        let content = '';
        
        if (messageData.role === 'user') {
            content = `
                <div class="message-header">
                    <div class="message-sender">
                        <i class="fas fa-user"></i> You
                    </div>
                    <div class="message-time">${time}</div>
                </div>
                <div class="message-content">${this.escapeHtml(messageData.content)}</div>
            `;
        } else if (messageData.role === 'assistant') {
            content = `
                <div class="message-header">
                    <div class="message-sender">
                        <i class="fas fa-robot"></i> ${messageData.model || 'Brad AI'}
                    </div>
                    <div class="message-time">${time}</div>
                </div>
                <div class="message-content">${this.formatResponse(messageData.content)}</div>
                ${messageData.model_version ? `
                    <div class="message-model">
                        <i class="fas fa-microchip"></i> Model: ${messageData.model_version}
                    </div>
                ` : ''}
                ${messageData.ml_insights ? `
                    <div class="message-ml">
                        <i class="fas fa-brain"></i> ML Analysis: 
                        Sentiment: ${messageData.ml_insights.sentiment}, 
                        Topics: ${messageData.ml_insights.topics.join(', ') || 'General'}
                    </div>
                ` : ''}
            `;
        } else {
            content = `
                <div class="message-header">
                    <div class="message-sender">
                        <i class="fas fa-info-circle"></i> System
                    </div>
                    <div class="message-time">${time}</div>
                </div>
                <div class="message-content" style="color: ${messageData.isError ? '#ef4444' : '#f59e0b'}">
                    ${this.escapeHtml(messageData.content)}
                </div>
            `;
        }
        
        messageElement.innerHTML = content;
        this.elements.chatMessages.appendChild(messageElement);
        
        // Scroll to bottom
        this.elements.chatMessages.scrollTop = this.elements.chatMessages.scrollHeight;
        
        // Remove welcome message if it exists
        const welcomeMessage = document.querySelector('.welcome-message');
        if (welcomeMessage) {
            welcomeMessage.remove();
        }
    }
    
    formatResponse(text) {
        // Convert markdown-style formatting
        let formatted = this.escapeHtml(text);
        
        // Code blocks
        formatted = formatted.replace(/```(\w+)?\n([\s\S]*?)```/g, (match, lang, code) => {
            return `<pre class="code-block"><code>${this.escapeHtml(code)}</code></pre>`;
        });
        
        // Inline code
        formatted = formatted.replace(/`([^`]+)`/g, '<code class="inline-code">$1</code>');
        
        // Bold
        formatted = formatted.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
        
        // Lists
        formatted = formatted.replace(/^\s*[-*]\s+(.+)$/gm, '<li>$1</li>');
        formatted = formatted.replace(/(<li>.*<\/li>)/s, '<ul>$1</ul>');
        
        // Line breaks
        formatted = formatted.replace(/\n/g, '<br>');
        
        return formatted;
    }
    
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
    
    updateMLInsights(insights) {
        if (!insights) return;
        
        // Update sentiment
        const sentimentValue = this.elements.sentimentInsight.querySelector('.insight-value');
        sentimentValue.textContent = insights.sentiment.charAt(0).toUpperCase() + insights.sentiment.slice(1);
        
        // Update topics
        const topicValue = this.elements.topicInsight.querySelector('.insight-value');
        topicValue.textContent = insights.topics.length > 0 
            ? insights.topics.slice(0, 2).join(', ')
            : 'General';
        
        // Update complexity
        const complexityValue = this.elements.complexityInsight.querySelector('.insight-value');
        const complexity = insights.complexity;
        let complexityText = 'Low';
        if (complexity > 0.6) complexityText = 'High';
        else if (complexity > 0.3) complexityText = 'Medium';
        complexityValue.textContent = complexityText;
    }
    
    async updateUserProfile() {
        try {
            const response = await fetch(`${this.API_URL}/profile/${this.userId}`);
            const data = await response.json();
            
            if (data.status === 'success') {
                const profile = data.profile;
                this.elements.interactionCount.textContent = profile.interaction_count || 0;
            }
        } catch (error) {
            console.error('Error updating profile:', error);
        }
    }
    
    async checkAPIStatus() {
        try {
            const response = await fetch(`${this.API_URL}/health`);
            const data = await response.json();
            
            if (data.status === 'healthy') {
                this.elements.apiStatus.textContent = `API: Connected (${data.active_users} users)`;
                this.elements.apiStatus.style.color = '#10b981';
            }
        } catch (error) {
            this.elements.apiStatus.textContent = 'API: Disconnected';
            this.elements.apiStatus.style.color = '#ef4444';
        }
        
        // Check every 30 seconds
        setTimeout(() => this.checkAPIStatus(), 30000);
    }
    
    clearChat() {
        if (confirm('Are you sure you want to clear the chat history?')) {
            this.elements.chatMessages.innerHTML = `
                <div class="welcome-message">
                    <div class="welcome-icon">
                        <i class="fas fa-robot"></i>
                    </div>
                    <h3>Chat Cleared</h3>
                    <p>Select a model from the sidebar and start chatting. Each model has unique capabilities!</p>
                </div>
            `;
        }
    }
    
    setLoading(isLoading) {
        this.isLoading = isLoading;
        if (isLoading) {
            this.elements.loadingOverlay.classList.add('active');
            this.elements.sendButton.disabled = true;
        } else {
            this.elements.loadingOverlay.classList.remove('active');
            this.elements.sendButton.disabled = false;
        }
    }
}

// Initialize the chat when page loads
document.addEventListener('DOMContentLoaded', () => {
    window.bradAI = new BradAIChat();
});
