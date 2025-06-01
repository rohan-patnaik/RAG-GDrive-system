const API_BASE_URL = '/.netlify/functions';

class APIClient {
    constructor() {
        this.rateLimitInfo = {};
    }
    
    async makeRequest(endpoint, options = {}) {
        try {
            // Add authentication headers
            const headers = {
                'Content-Type': 'application/json',
                ...authManager.getAuthHeaders(),
                ...options.headers
            };
            
            const response = await fetch(`${API_BASE_URL}${endpoint}`, {
                ...options,
                headers
            });
            
            // Update rate limit info from headers
            this.updateRateLimitInfo(response.headers);
            
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || `HTTP ${response.status}`);
            }
            
            return await response.json();
            
        } catch (error) {
            console.error('API request failed:', error);
            throw error;
        }
    }
    
    updateRateLimitInfo(headers) {
        const limit = headers.get('X-RateLimit-Limit');
        const remaining = headers.get('X-RateLimit-Remaining');
        const reset = headers.get('X-RateLimit-Reset');
        
        if (limit && remaining && reset) {
            this.rateLimitInfo = {
                limit: parseInt(limit),
                remaining: parseInt(remaining),
                reset: parseInt(reset)
            };
            
            this.updateRateLimitUI();
        }
    }
    
    updateRateLimitUI() {
        const rateLimitElement = document.getElementById('rate-limit-info');
        if (rateLimitElement) {
            const { limit, remaining, reset } = this.rateLimitInfo;
            const resetDate = new Date(reset * 1000);
            
            rateLimitElement.innerHTML = `
                Rate Limit: ${remaining}/${limit} remaining
                (resets at ${resetDate.toLocaleTimeString()})
            `;
        }
    }
    
    async queryBackend(queryData) {
        return this.makeRequest('/query', {
            method: 'POST',
            body: JSON.stringify(queryData)
        });
    }
    
    async getSystemStatus() {
        return this.makeRequest('/health');
    }
    
    async ingestDocuments(formData) {
        // For file uploads, don't set Content-Type (let browser set it)
        const headers = authManager.getAuthHeaders();
        
        return fetch(`${API_BASE_URL}/ingest`, {
            method: 'POST',
            headers,
            body: formData
        }).then(response => {
            this.updateRateLimitInfo(response.headers);
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }
            return response.json();
        });
    }
}

// Initialize API client
const apiClient = new APIClient();

// Legacy functions for backward compatibility
async function queryBackend(queryData) {
    return apiClient.queryBackend(queryData);
}

async function getSystemStatus() {
    return apiClient.getSystemStatus();
}

async function ingestDocuments(formData) {
    return apiClient.ingestDocuments(formData);
}