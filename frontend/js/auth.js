class AuthManager {
    constructor() {
        this.apiKey = this.loadApiKey();
        this.setupEventListeners();
        this.updateUI();
    }
    
    loadApiKey() {
        return localStorage.getItem(CONFIG.STORAGE_KEYS.SYSTEM_API_KEY) || '';
    }
    
    saveApiKey(apiKey) {
        this.apiKey = apiKey;
        if (apiKey) {
            localStorage.setItem(CONFIG.STORAGE_KEYS.SYSTEM_API_KEY, apiKey);
        } else {
            localStorage.removeItem(CONFIG.STORAGE_KEYS.SYSTEM_API_KEY);
        }
        this.updateUI();
        this.checkConnection();
    }
    
    setupEventListeners() {
        const saveKeyBtn = document.getElementById('save-key-btn');
        const generateKeyBtn = document.getElementById('generate-key-btn');
        const apiKeyInput = document.getElementById('system-api-key');
        
        if (saveKeyBtn) {
            saveKeyBtn.addEventListener('click', () => {
                const newKey = apiKeyInput.value.trim();
                if (newKey) {
                    this.saveApiKey(newKey);
                    showToast('API key saved successfully', 'success');
                } else {
                    showToast('Please enter a valid API key', 'error');
                }
            });
        }
        
        if (generateKeyBtn) {
            generateKeyBtn.addEventListener('click', () => {
                this.generateSystemKey();
            });
        }
        
        if (apiKeyInput) {
            apiKeyInput.value = this.apiKey;
            
            // Auto-save on Enter key
            apiKeyInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter') {
                    saveKeyBtn.click();
                }
            });
        }
    }
    
    async generateSystemKey() {
        const generateBtn = document.getElementById('generate-key-btn');
        const originalText = generateBtn.textContent;
        
        try {
            generateBtn.disabled = true;
            generateBtn.textContent = 'Generating...';
            
            const response = await fetch(`${CONFIG.API_BASE_URL}/generate_system_key`);
            
            if (!response.ok) {
                throw new Error(`Failed to generate key: ${response.status}`);
            }
            
            const data = await response.json();
            const generatedKey = data.system_api_key;
            
            // Update UI with generated key
            const apiKeyInput = document.getElementById('system-api-key');
            apiKeyInput.value = generatedKey;
            
            // Save automatically
            this.saveApiKey(generatedKey);
            
            showToast('System API key generated and saved!', 'success');
            
        } catch (error) {
            console.error('Key generation failed:', error);
            showToast(`Failed to generate key: ${error.message}`, 'error');
        } finally {
            generateBtn.disabled = false;
            generateBtn.textContent = originalText;
        }
    }
    
    getAuthHeaders() {
        if (!this.apiKey) {
            throw new Error('API key not configured. Please set your API key first.');
        }
        
        return {
            'Authorization': `Bearer ${this.apiKey}`,
            'X-API-Key': this.apiKey
        };
    }
    
    updateUI() {
        const statusElement = document.querySelector('.api-key-status .key-status');
        const statusContainer = document.querySelector('.api-key-status');
        
        if (statusElement && statusContainer) {
            if (this.apiKey) {
                statusElement.textContent = 'Configured';
                statusContainer.classList.add('configured');
            } else {
                statusElement.textContent = 'Not configured';
                statusContainer.classList.remove('configured');
            }
        }
    }
    
    async checkConnection() {
        if (!this.apiKey) return;
        
        try {
            const response = await fetch(`${CONFIG.API_BASE_URL}/health`, {
                headers: this.getAuthHeaders()
            });
            
            const statusDot = document.querySelector('.status-dot');
            const statusText = document.querySelector('.status-text');
            
            if (response.ok) {
                statusDot.className = 'status-dot connected';
                statusText.textContent = 'Connected';
            } else {
                statusDot.className = 'status-dot error';
                statusText.textContent = 'Auth Error';
            }
        } catch (error) {
            const statusDot = document.querySelector('.status-dot');
            const statusText = document.querySelector('.status-text');
            statusDot.className = 'status-dot error';
            statusText.textContent = 'Connection Error';
        }
    }
    
    isAuthenticated() {
        return !!this.apiKey;
    }
    
    clearAuth() {
        this.saveApiKey('');
        showToast('API key cleared', 'info');
    }
}

// Initialize auth manager
const authManager = new AuthManager();

// Export for global use
window.authManager = authManager;