// Frontend configuration
const CONFIG = {
    API_BASE_URL: '/.netlify/functions',
    STORAGE_KEYS: {
        SYSTEM_API_KEY: 'rag_system_api_key',
        USER_PREFERENCES: 'rag_user_preferences'
    },
    DEFAULT_LLM_PROVIDER: 'gemini',
    SUPPORTED_FILE_TYPES: ['txt', 'pdf', 'docx', 'md'],
    MAX_FILE_SIZE: 10 * 1024 * 1024, // 10MB
    
    // UI Configuration
    TOAST_DURATION: 5000,
    AUTO_HIDE_RESULTS: false,
    
    // API Configuration
    DEFAULT_TIMEOUT: 30000,
    RETRY_ATTEMPTS: 2
};

// Environment-specific overrides
if (window.location.hostname === 'localhost') {
    CONFIG.API_BASE_URL = 'http://localhost:8888/.netlify/functions';
}

// Export for use in other modules
window.CONFIG = CONFIG;