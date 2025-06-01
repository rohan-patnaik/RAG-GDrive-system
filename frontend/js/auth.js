class AuthManager {
    constructor() {
        this.apiKey = localStorage.getItem('rag_api_key') || '';
        this.setupAuthUI();
    }
    
    setupAuthUI() {
        // Add API key input to forms
        const authHTML = `
            <div class="auth-section">
                <label for="api-key">API Key:</label>
                <input type="password" id="api-key" placeholder="Enter your API key" 
                       value="${this.apiKey}" />
                <button type="button" onclick="authManager.saveApiKey()">Save</button>
            </div>
        `;
        
        // Add to existing forms
        const forms = document.querySelectorAll('.api-form');
        forms.forEach(form => {
            form.insertAdjacentHTML('afterbegin', authHTML);
        });
    }
    
    saveApiKey() {
        const apiKeyInput = document.getElementById('api-key');
        this.apiKey = apiKeyInput.value;
        localStorage.setItem('rag_api_key', this.apiKey);
        
        this.showMessage('API key saved successfully', 'success');
    }
    
    getAuthHeaders() {
        if (!this.apiKey) {
            throw new Error('API key not configured');
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