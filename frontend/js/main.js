// Main application logic
class RAGApp {
    constructor() {
        this.initialize();
    }
    
    initialize() {
        this.setupEventListeners();
        this.checkInitialStatus();
        this.loadUserPreferences();
    }
    
    setupEventListeners() {
        // Query form
        const queryForm = document.getElementById('query-form');
        if (queryForm) {
            queryForm.addEventListener('submit', (e) => this.handleQuery(e));
        }
        
        // Upload form
        const uploadForm = document.getElementById('upload-form');
        if (uploadForm) {
            uploadForm.addEventListener('submit', (e) => this.handleUpload(e));
        }
        
        // Status refresh
        const refreshStatusBtn = document.getElementById('refresh-status-btn');
        if (refreshStatusBtn) {
            refreshStatusBtn.addEventListener('click', () => this.checkSystemStatus());
        }
        
        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => this.handleKeyboardShortcuts(e));
    }
    
    async handleQuery(e) {
        e.preventDefault();
        
        const queryText = document.getElementById('query-text').value.trim();
        const llmProvider = document.getElementById('llm-provider').value;
        
        if (!queryText) {
            showToast('Please enter a question', 'error');
            return;
        }
        
        if (!authManager.isAuthenticated()) {
            showToast('Please configure your API key first', 'error');
            return;
        }
        
        setLoadingState('query-submit-btn', true);
        hideResults('query-results');
        
        try {
            const result = await apiClient.query({
                query_text: queryText,
                llm_provider: llmProvider
            });
            
            displayQueryResults(result);
            
            // Save to preferences
            this.saveQueryToHistory(queryText, result);
            
            showToast('Query completed successfully', 'success');
            
        } catch (error) {
            console.error('Query failed:', error);
            showResults('query-results', `❌ Query failed: ${error.message}`, 'error');
            showToast(`Query failed: ${error.message}`, 'error');
        } finally {
            setLoadingState('query-submit-btn', false);
        }
    }
    
    async handleUpload(e) {
        e.preventDefault();
        
        const fileInput = document.getElementById('file-input');
        const sourceInput = document.getElementById('source-input');
        
        if (!fileInput.files.length) {
            showToast('Please select files to upload', 'error');
            return;
        }
        
        if (!authManager.isAuthenticated()) {
            showToast('Please configure your API key first', 'error');
            return;
        }
        
        // Validate files
        const validationErrors = [];
        Array.from(fileInput.files).forEach(file => {
            const errors = validateFile(file);
            if (errors.length > 0) {
                validationErrors.push(`${file.name}: ${errors.join(', ')}`);
            }
        });
        
        if (validationErrors.length > 0) {
            showToast(`File validation failed: ${validationErrors.join('; ')}`, 'error');
            return;
        }
        
        setLoadingState('upload-submit-btn', true);
        hideResults('upload-results');
        
        try {
            const formData = new FormData();
            
            // Add files
            Array.from(fileInput.files).forEach(file => {
                formData.append('files', file);
            });
            
            // Add source if provided
            const source = sourceInput.value.trim();
            if (source) {
                formData.append('source', source);
            }
            
            const result = await apiClient.uploadDocuments(formData);
            
            displayUploadResults(result);
            
            // Clear form on success
            fileInput.value = '';
            sourceInput.value = '';
            document.querySelector('.file-input-display .file-text').textContent = 'Choose files or drag & drop';
            
            showToast('Documents uploaded successfully', 'success');
            
        } catch (error) {
            console.error('Upload failed:', error);
            showResults('upload-results', `❌ Upload failed: ${error.message}`, 'error');
            showToast(`Upload failed: ${error.message}`, 'error');
        } finally {
            setLoadingState('upload-submit-btn', false);
        }
    }
    
    async checkSystemStatus() {
        if (!authManager.isAuthenticated()) {
            showResults('status-results', '❌ Please configure your API key first', 'error');
            return;
        }
        
        const refreshBtn = document.getElementById('refresh-status-btn');
        const originalText = refreshBtn.textContent;
        
        try {
            refreshBtn.disabled = true;
            refreshBtn.textContent = '🔄 Checking...';
            
            const status = await apiClient.getHealth();
            displaySystemStatus(status);
            
        } catch (error) {
            console.error('Status check failed:', error);
            showResults('status-results', `❌ Status check failed: ${error.message}`, 'error');
            showToast(`Status check failed: ${error.message}`, 'error');
        } finally {
            refreshBtn.disabled = false;
            refreshBtn.textContent = originalText;
        }
    }
    
    async checkInitialStatus() {
        // Check connection status on load
        if (authManager.isAuthenticated()) {
            await authManager.checkConnection();
            await this.checkSystemStatus();
        }
    }
    
    handleKeyboardShortcuts(e) {
        // Ctrl/Cmd + Enter to submit query
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
            const queryForm = document.getElementById('query-form');
            if (document.activeElement?.closest('#query-form')) {
                e.preventDefault();
                queryForm.dispatchEvent(new Event('submit'));
            }
        }
        
        // Escape to clear results
        if (e.key === 'Escape') {
            hideResults('query-results');
            hideResults('upload-results');
            hideResults('status-results');
        }
    }
    
    saveQueryToHistory(query, result) {
        try {
            const history = this.getQueryHistory();
            history.unshift({
                query,
                result: {
                    answer: result.llm_answer,
                    provider: result.llm_provider_used,
                    timestamp: new Date().toISOString()
                }
            });
            
            // Keep only last 50 queries
            if (history.length > 50) {
                history.splice(50);
            }
            
            localStorage.setItem('rag_query_history', JSON.stringify(history));
        } catch (error) {
            console.warn('Failed to save query history:', error);
        }
    }
    
    getQueryHistory() {
        try {
            const history = localStorage.getItem('rag_query_history');
            return history ? JSON.parse(history) : [];
        } catch (error) {
            console.warn('Failed to load query history:', error);
            return [];
        }
    }
    
    loadUserPreferences() {
        try {
            const preferences = localStorage.getItem(CONFIG.STORAGE_KEYS.USER_PREFERENCES);
            if (preferences) {
                const prefs = JSON.parse(preferences);
                
                // Restore LLM provider preference
                const llmSelect = document.getElementById('llm-provider');
                if (llmSelect && prefs.defaultLlmProvider) {
                    llmSelect.value = prefs.defaultLlmProvider;
                }
            }
        } catch (error) {
            console.warn('Failed to load user preferences:', error);
        }
    }
    
    saveUserPreferences() {
        try {
            const llmSelect = document.getElementById('llm-provider');
            const preferences = {
                defaultLlmProvider: llmSelect?.value || CONFIG.DEFAULT_LLM_PROVIDER,
                lastSaved: new Date().toISOString()
            };
            
            localStorage.setItem(CONFIG.STORAGE_KEYS.USER_PREFERENCES, JSON.stringify(preferences));
        } catch (error) {
            console.warn('Failed to save user preferences:', error);
        }
    }
}

// Initialize the application
document.addEventListener('DOMContentLoaded', () => {
    window.ragApp = new RAGApp();
    
    // Save preferences on form changes
    const llmSelect = document.getElementById('llm-provider');
    if (llmSelect) {
        llmSelect.addEventListener('change', () => {
            window.ragApp.saveUserPreferences();
        });
    }
});

// Handle page visibility changes (auto-refresh status when page becomes visible)
document.addEventListener('visibilitychange', () => {
    if (!document.hidden && window.ragApp && authManager.isAuthenticated()) {
        setTimeout(() => {
            authManager.checkConnection();
        }, 1000);
    }
});