class APIClient {
    constructor() {
        this.baseURL = CONFIG.API_BASE_URL;
        this.timeout = CONFIG.DEFAULT_TIMEOUT;
        this.retryAttempts = CONFIG.RETRY_ATTEMPTS;
    }
    
    async makeRequest(endpoint, options = {}) {
        const url = `${this.baseURL}${endpoint}`;
        
        // Prepare headers
        const headers = {
            'Content-Type': 'application/json',
            ...options.headers
        };
        
        // Add auth headers if available
        try {
            const authHeaders = authManager.getAuthHeaders();
            Object.assign(headers, authHeaders);
        } catch (error) {
            if (endpoint !== '/generate_system_key' && endpoint !== '/health') {
                throw new Error('Authentication required. Please configure your API key.');
            }
        }
        
        // Prepare request options
        const requestOptions = {
            ...options,
            headers,
            signal: AbortSignal.timeout(this.timeout)
        };
        
        let lastError;
        
        // Retry logic
        for (let attempt = 0; attempt <= this.retryAttempts; attempt++) {
            try {
                const response = await fetch(url, requestOptions);
                
                if (!response.ok) {
                    const errorData = await response.json().catch(() => ({}));
                    const errorMessage = errorData.error || `HTTP ${response.status}`;
                    throw new Error(errorMessage);
                }
                
                return await response.json();
                
            } catch (error) {
                lastError = error;
                
                // Don't retry on auth errors or client errors
                if (error.name === 'AbortError' || 
                    error.message.includes('401') || 
                    error.message.includes('400')) {
                    break;
                }
                
                // Wait before retry (exponential backoff)
                if (attempt < this.retryAttempts) {
                    await new Promise(resolve => 
                        setTimeout(resolve, Math.pow(2, attempt) * 1000)
                    );
                }
            }
        }
        
        throw lastError;
    }
    
    async query(queryData) {
        return this.makeRequest('/query', {
            method: 'POST',
            body: JSON.stringify(queryData)
        });
    }
    
    async uploadDocuments(formData) {
        // For file uploads, don't set Content-Type (let browser handle it)
        const headers = authManager.getAuthHeaders();
        
        return fetch(`${this.baseURL}/ingest`, {
            method: 'POST',
            headers,
            body: formData,
            signal: AbortSignal.timeout(this.timeout * 2) // Longer timeout for uploads
        }).then(async response => {
            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                throw new Error(errorData.error || `Upload failed: ${response.status}`);
            }
            return response.json();
        });
    }
    
    async getHealth() {
        return this.makeRequest('/health');
    }
    
    async generateSystemKey() {
        return this.makeRequest('/generate_system_key');
    }
}

// Initialize API client
const apiClient = new APIClient();

// Export for global use
window.apiClient = apiClient;