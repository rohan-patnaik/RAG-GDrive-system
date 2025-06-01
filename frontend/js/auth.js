class AuthManager {
    constructor() {
        this.apiKey = localStorage.getItem('rag_api_key') || '';
        this.setupAuthUI();
        this.validateStoredKey();
    }
    
    setupAuthUI() {
        // Set saved API key in input
        const apiKeyInput = document.getElementById('api-key');
        if (apiKeyInput && this.apiKey) {
            apiKeyInput.value = this.apiKey;
        }
    }
    
    async validateStoredKey() {
        if (this.apiKey) {
            await this.testApiKey(this.apiKey);
        }
    }
    
    async saveApiKey() {
        const apiKeyInput = document.getElementById('api-key');
        const newApiKey = apiKeyInput.value.trim();
        
        if (!newApiKey) {
            this.showAuthStatus('Please enter an API key', 'invalid');
            return;
        }
        
        // Test the API key
        const isValid = await this.testApiKey(newApiKey);
        
        if (isValid) {
            this.apiKey = newApiKey;
            localStorage.setItem('rag_api_key', this.apiKey);
            this.showAuthStatus('✅ API key saved and validated successfully', 'valid');
        } else {
            this.showAuthStatus('❌ Invalid API key. Please check with your administrator.', 'invalid');
        }
    }
    
    async testApiKey(apiKey) {
        try {
            const response = await fetch('/.netlify/functions/health', {
                method: 'GET',
                headers: {
                    'Authorization': `Bearer ${apiKey}`,
                    'X-API-Key': apiKey
                }
            });
            
            return response.ok;
        } catch (error) {
            console.error('API key validation failed:', error);
            return false;
        }
    }
    
    showAuthStatus(message, type) {
        const statusDiv = document.getElementById('auth-status');
        statusDiv.textContent = message;
        statusDiv.className = `auth-status ${type}`;
    }
    
    getAuthHeaders() {
        if (!this.apiKey) {
            throw new Error('❌ API key not configured. Please enter your RAG system API key above.');
        }
        
        return {
            'Authorization': `Bearer ${this.apiKey}`,
            'X-API-Key': this.apiKey
        };
    }
    
    showMessage(message, type = 'info') {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${type}`;
        messageDiv.textContent = message;
        
        document.body.appendChild(messageDiv);
        
        setTimeout(() => {
            messageDiv.remove();
        }, 3000);
    }
}

// Initialize auth manager
const authManager = new AuthManager();