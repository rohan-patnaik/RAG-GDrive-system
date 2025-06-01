// Query form handler
document.getElementById('query-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const queryText = document.getElementById('query-text').value;
    const llmProvider = document.getElementById('llm-provider').value;
    const useCache = document.getElementById('use-cache').checked;
    
    if (!queryText.trim()) {
        showResults('query-results', 'Please enter a question', 'error');
        return;
    }
    
    const submitButton = e.target.querySelector('button');
    const originalText = submitButton.textContent;
    submitButton.disabled = true;
    submitButton.innerHTML = '<span class="loading"></span> Processing...';
    
    try {
        const result = await apiClient.queryBackend({
            query_text: queryText,
            llm_provider: llmProvider,
            use_cache: useCache
        });
        
        displayQueryResults(result);
        
    } catch (error) {
        showResults('query-results', `Error: ${error.message}`, 'error');
    } finally {
        submitButton.disabled = false;
        submitButton.textContent = originalText;
    }
});

// Upload form handler
document.getElementById('upload-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const filesInput = document.getElementById('files');
    const source = document.getElementById('source').value;
    
    if (!filesInput.files.length) {
        showResults('upload-results', 'Please select files to upload', 'error');
        return;
    }
    
    const formData = new FormData();
    for (let file of filesInput.files) {
        formData.append('files', file);
    }
    if (source) {
        formData.append('source', source);
    }
    
    const submitButton = e.target.querySelector('button');
    const originalText = submitButton.textContent;
    submitButton.disabled = true;
    submitButton.innerHTML = '<span class="loading"></span> Uploading...';
    
    try {
        const result = await apiClient.ingestDocuments(formData);
        displayUploadResults(result);
        
        // Clear form
        filesInput.value = '';
        document.getElementById('source').value = '';
        
    } catch (error) {
        showResults('upload-results', `Error: ${error.message}`, 'error');
    } finally {
        submitButton.disabled = false;
        submitButton.textContent = originalText;
    }
});

function displayQueryResults(result) {
    const container = document.getElementById('query-results');
    
    let html = `
        <h3>Answer ${result.from_cache ? '(from cache)' : ''}</h3>
        <div class="answer">${result.llm_answer}</div>
        
        <h4>Details</h4>
        <p><strong>Provider:</strong> ${result.llm_provider_used}</p>
        <p><strong>Model:</strong> ${result.llm_model_used}</p>
        <p><strong>Retrieved Chunks:</strong> ${result.retrieved_chunks.length}</p>
    `;
    
    if (result.retrieved_chunks.length > 0) {
        html += '<h4>Source Context</h4>';
        result.retrieved_chunks.forEach((chunk, index) => {
            html += `
                <div class="chunk">
                    <div class="chunk-metadata">
                        Source: ${chunk.metadata.filename || 'Unknown'} 
                        (Similarity: ${chunk.score.toFixed(3)})
                    </div>
                    <div>${chunk.content.substring(0, 200)}...</div>
                </div>
            `;
        });
    }
    
    container.innerHTML = html;
    container.className = 'results success';
}

function displayUploadResults(result) {
    const container = document.getElementById('upload-results');
    
    let html = `<h3>${result.message}</h3>`;
    
    result.results.forEach(fileResult => {
        const status = fileResult.status === 'success' ? 'success' : 'error';
        html += `
            <div class="file-result ${status}">
                <strong>${fileResult.filename}:</strong> 
                ${fileResult.status === 'success' 
                    ? `Successfully processed ${fileResult.total_chunks} chunks` 
                    : `Error: ${fileResult.error}`
                }
            </div>
        `;
    });
    
    container.innerHTML = html;
    container.className = 'results success';
}

async function checkSystemStatus() {
    try {
        const status = await apiClient.getSystemStatus();
        displaySystemStatus(status);
    } catch (error) {
        showResults('status-results', `Error: ${error.message}`, 'error');
    }
}

function displaySystemStatus(status) {
    const container = document.getElementById('status-results');
    
    let html = `
        <h3>System Status: ${status.status}</h3>
        <p><strong>Message:</strong> ${status.message}</p>
        <p><strong>Timestamp:</strong> ${new Date(status.timestamp).toLocaleString()}</p>
    `;
    
    if (status.components) {
        html += '<h4>Components</h4>';
        status.components.forEach(component => {
            html += `
                <div class="component ${component.status.toLowerCase()}">
                    ${component.name}: ${component.status}
                </div>
            `;
        });
    }
    
    container.innerHTML = html;
    container.className = `results ${status.status === 'OK' ? 'success' : 'error'}`;
}

function showResults(containerId, message, type) {
    const container = document.getElementById(containerId);
    container.innerHTML = message;
    container.className = `results ${type}`;
}

// Initialize page
document.addEventListener('DOMContentLoaded', () => {
    // Load saved API key
    const savedKey = localStorage.getItem('rag_api_key');
    if (savedKey) {
        document.getElementById('api-key').value = savedKey;
    }
    
    // Check system status on load
    checkSystemStatus();
});