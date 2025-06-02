// Toast notification system
function showToast(message, type = 'info', duration = CONFIG.TOAST_DURATION) {
    const container = document.getElementById('toast-container');
    if (!container) return;
    
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    
    container.appendChild(toast);
    
    // Trigger animation
    setTimeout(() => toast.classList.add('show'), 10);
    
    // Auto-remove
    setTimeout(() => {
        toast.classList.remove('show');
        setTimeout(() => {
            if (toast.parentNode) {
                toast.parentNode.removeChild(toast);
            }
        }, 300);
    }, duration);
}

// Results display utilities
function showResults(containerId, content, type = 'info') {
    const container = document.getElementById(containerId);
    if (!container) return;
    
    container.innerHTML = content;
    container.className = `results-container show`;
    
    // Add result card styling
    const resultCard = document.createElement('div');
    resultCard.className = `result-card ${type}`;
    resultCard.innerHTML = content;
    
    container.innerHTML = '';
    container.appendChild(resultCard);
}

function hideResults(containerId) {
    const container = document.getElementById(containerId);
    if (container) {
        container.classList.remove('show');
    }
}

// Query results display
function displayQueryResults(result) {
    const container = document.getElementById('query-results');
    if (!container) return;
    
    let html = `
        <div class="answer-container">
            <div class="answer-text">${escapeHtml(result.llm_answer)}</div>
            <div class="answer-meta">
                <span><strong>Provider:</strong> ${result.llm_provider_used}</span>
                <span><strong>Model:</strong> ${result.llm_model_used}</span>
                <span><strong>Sources:</strong> ${result.retrieved_chunks.length} chunks</span>
                ${result.from_cache ? '<span><strong>Source:</strong> Cache</span>' : ''}
            </div>
        </div>
    `;
    
    if (result.retrieved_chunks && result.retrieved_chunks.length > 0) {
        html += '<div class="chunks-container">';
        html += '<h4>📚 Source Documents:</h4>';
        
        result.retrieved_chunks.forEach((chunk, index) => {
            html += `
                <div class="chunk-card">
                    <div class="chunk-meta">
                        <span><strong>Source:</strong> ${chunk.metadata.filename || 'Unknown'}</span>
                        <span><strong>Relevance:</strong> ${(chunk.score * 100).toFixed(1)}%</span>
                    </div>
                    <div class="chunk-content">
                        ${escapeHtml(chunk.content.substring(0, 300))}${chunk.content.length > 300 ? '...' : ''}
                    </div>
                </div>
            `;
        });
        
        html += '</div>';
    }
    
    container.innerHTML = html;
    container.classList.add('show');
}

// Upload results display
function displayUploadResults(result) {
    const container = document.getElementById('upload-results');
    if (!container) return;
    
    let html = `<h4>📤 Upload Results: ${result.message}</h4>`;
    
    if (result.results && result.results.length > 0) {
        result.results.forEach(fileResult => {
            const isSuccess = fileResult.status === 'success';
            const statusIcon = isSuccess ? '✅' : '❌';
            const statusText = isSuccess 
                ? `Processed ${fileResult.total_chunks} chunks (${fileResult.total_characters} characters)`
                : `Error: ${fileResult.error}`;
            
            html += `
                <div class="result-card ${isSuccess ? 'success' : 'error'}">
                    ${statusIcon} <strong>${fileResult.filename}:</strong> ${statusText}
                </div>
            `;
        });
    }
    
    container.innerHTML = html;
    container.classList.add('show');
}

// System status display
function displaySystemStatus(status) {
    const container = document.getElementById('status-results');
    if (!container) return;
    
    const statusIcon = status.status === 'OK' ? '✅' : '⚠️';
    
    let html = `
        <div class="result-card ${status.status === 'OK' ? 'success' : 'error'}">
            <h4>${statusIcon} System Status: ${status.status}</h4>
            <p><strong>Message:</strong> ${status.message}</p>
            ${status.default_provider ? `<p><strong>Default Provider:</strong> ${status.default_provider}</p>` : ''}
        </div>
    `;
    
    if (status.components && status.components.length > 0) {
        html += '<div class="component-status">';
        status.components.forEach(component => {
            const componentIcon = component.status === 'OK' ? '✅' : '❌';
            html += `
                <div class="component-card ${component.status.toLowerCase()}">
                    ${componentIcon} ${component.name}
                </div>
            `;
        });
        html += '</div>';
    }
    
    container.innerHTML = html;
    container.classList.add('show');
}

// Loading state management
function setLoadingState(buttonId, isLoading) {
    const button = document.getElementById(buttonId);
    if (!button) return;
    
    if (isLoading) {
        button.classList.add('loading');
        button.disabled = true;
    } else {
        button.classList.remove('loading');
        button.disabled = false;
    }
}

// File input handling
function setupFileInput() {
    const fileInput = document.getElementById('file-input');
    const fileDisplay = document.querySelector('.file-input-display .file-text');
    
    if (fileInput && fileDisplay) {
        fileInput.addEventListener('change', (e) => {
            const files = e.target.files;
            if (files.length > 0) {
                if (files.length === 1) {
                    fileDisplay.textContent = files[0].name;
                } else {
                    fileDisplay.textContent = `${files.length} files selected`;
                }
            } else {
                fileDisplay.textContent = 'Choose files or drag & drop';
            }
        });
    }
}

// Drag and drop for file input
function setupDragAndDrop() {
    const wrapper = document.querySelector('.file-input-wrapper');
    if (!wrapper) return;
    
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        wrapper.addEventListener(eventName, preventDefaults, false);
    });
    
    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }
    
    ['dragenter', 'dragover'].forEach(eventName => {
        wrapper.addEventListener(eventName, highlight, false);
    });
    
    ['dragleave', 'drop'].forEach(eventName => {
        wrapper.addEventListener(eventName, unhighlight, false);
    });
    
    function highlight() {
        wrapper.classList.add('drag-over');
    }
    
    function unhighlight() {
        wrapper.classList.remove('drag-over');
    }
    
    wrapper.addEventListener('drop', handleDrop, false);
    
    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        const fileInput = document.getElementById('file-input');
        
        if (fileInput) {
            fileInput.files = files;
            fileInput.dispatchEvent(new Event('change'));
        }
    }
}

// Utility functions
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function validateFile(file) {
    const errors = [];
    
    // Check file size
    if (file.size > CONFIG.MAX_FILE_SIZE) {
        errors.push(`File too large. Maximum size: ${formatFileSize(CONFIG.MAX_FILE_SIZE)}`);
    }
    
    // Check file type
    const extension = file.name.split('.').pop().toLowerCase();
    if (!CONFIG.SUPPORTED_FILE_TYPES.includes(extension)) {
        errors.push(`Unsupported file type. Supported: ${CONFIG.SUPPORTED_FILE_TYPES.join(', ')}`);
    }
    
    return errors;
}

// Modal handling
function setupModal() {
    const modal = document.getElementById('help-modal');
    const closeBtn = document.querySelector('.close');
    
    if (closeBtn) {
        closeBtn.addEventListener('click', () => {
            modal.style.display = 'none';
        });
    }
    
    window.addEventListener('click', (e) => {
        if (e.target === modal) {
            modal.style.display = 'none';
        }
    });
}

// Initialize UI components
document.addEventListener('DOMContentLoaded', () => {
    setupFileInput();
    setupDragAndDrop();
    setupModal();
});

// Export utility functions
window.showToast = showToast;
window.showResults = showResults;
window.hideResults = hideResults;
window.displayQueryResults = displayQueryResults;
window.displayUploadResults = displayUploadResults;
window.displaySystemStatus = displaySystemStatus;
window.setLoadingState = setLoadingState;