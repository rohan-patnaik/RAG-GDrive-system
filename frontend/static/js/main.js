document.addEventListener('DOMContentLoaded', () => {
    const currentPage = window.location.pathname.split('/').pop() || 'index.html';

    if (currentPage === 'index.html' || currentPage === '') {
        const queryButton = document.getElementById('submit-query');
        const queryTextarea = document.getElementById('query-text');
        const resultsDiv = document.getElementById('query-results');

        if (queryButton) {
            queryButton.addEventListener('click', async () => {
                const query = queryTextarea.value;
                if (!query.trim()) {
                    resultsDiv.innerHTML = '<p style="color: red;">Please enter a query.</p>';
                    return;
                }
                resultsDiv.innerHTML = '<div class="loader">Querying...</div>'; // Use loader class
                try {
                    // Using default values for llmProvider, topK, similarityThreshold from api.js
                    const data = await queryBackend(query);

                    let html = `<h3>Answer:</h3><p class="llm-answer">${data.llm_answer || "No answer provided."}</p>`;
                    html += `<p class="debug-info">LLM Used: ${data.llm_provider_used} - ${data.llm_model_used}</p>`;

                    if (data.retrieved_chunks && data.retrieved_chunks.length > 0) {
                        html += `<h4>Retrieved Chunks:</h4>`;
                        data.retrieved_chunks.forEach(chunk => {
                            html += `<div class="chunk">
                                <p><strong>Content:</strong> ${chunk.content}</p>
                                <p><em>Score: ${chunk.score.toFixed(4)}</em></p>`;
                            if (chunk.metadata) {
                                html += `<p class="metadata"><em>Source: ${chunk.metadata.filename || chunk.metadata.source_id || 'N/A'}</em></p>`;
                                // Optionally display more metadata:
                                // html += `<pre class="metadata-full">${JSON.stringify(chunk.metadata, null, 2)}</pre>`;
                            }
                            html += `</div>`;
                        });
                    } else {
                        html += `<p>No chunks were retrieved for this query.</p>`;
                    }
                    resultsDiv.innerHTML = html;
                } catch (error) {
                    // Log detailed error to console
                    console.error('Query processing error:', error);
                    if (error.body) {
                        console.error('Error body (parsed):', error.body);
                    } else if (error.rawBody) {
                        console.error('Error body (raw):', error.rawBody);
                    }

                    // Display user-friendly message
                    let userMessage = "An error occurred while processing your query. Please try again later.";
                    if (error.body && error.body.error === "InvalidRequest") {
                        userMessage = `There was an issue with your request: ${error.body.message}. Please check your input.`;
                    } else if (response.status === 429) { // Too Many Requests
                        userMessage = "The service is currently experiencing high traffic. Please try again in a few moments.";
                    }
                    resultsDiv.innerHTML = `<p style="color: red;">${userMessage}</p>`;
                }
            });
        }
    } else if (currentPage === 'status.html') {
        const statusDiv = document.getElementById('status-details');
        if (statusDiv) {
            statusDiv.innerHTML = '<div class="loader">Loading system status...</div>'; // Use loader class
            getSystemStatus().then(data => {
                let html = `<p class="status-overall"><strong>Overall System Status:</strong> <span class="status-${data.system_status?.toLowerCase()}">${data.system_status || 'Unknown'}</span></p>`;
                if (data.app_name) html += `<p><strong>Application:</strong> ${data.app_name}</p>`;
                if (data.version) html += `<p><strong>Version:</strong> ${data.version}</p>`;
                if (data.environment) html += `<p><strong>Environment:</strong> ${data.environment}</p>`;
                if (data.timestamp) html += `<p><strong>Timestamp:</strong> ${new Date(data.timestamp).toLocaleString()}</p>`;

                if (data.components && data.components.length > 0) {
                    html += `<h4>System Components:</h4>`;
                    data.components.forEach(comp => {
                        html += `<div class="component-status">
                                    <p><strong>${comp.name}:</strong> <span class="status-${comp.status?.toLowerCase()}">${comp.status || 'Unknown'}</span></p>`;
                        if (comp.message) html += `<p class="component-message"><em>${comp.message}</em></p>`;
                        if (comp.details && Object.keys(comp.details).length > 0) {
                             html += `<pre class="component-details">${JSON.stringify(comp.details, null, 2)}</pre>`;
                        }
                        html += `</div>`;
                    });
                } else {
                    html += `<p>No component statuses available.</p>`;
                }
                statusDiv.innerHTML = html;
            }).catch(error => {
                // Log detailed error to console
                console.error('System status fetch error:', error);
                if (error.body) {
                    console.error('Error body (parsed):', error.body);
                } else if (error.rawBody) {
                    console.error('Error body (raw):', error.rawBody);
                }
                // Display user-friendly message
                statusDiv.innerHTML = `<p style="color: red;">Could not retrieve system status at this time. Please check back later.</p>`;
            });
        }
    } else if (currentPage === 'ingest.html') {
        const ingestButton = document.getElementById('start-ingestion');
        const fileInput = document.getElementById('file-input'); // Changed from sourceInput
        const statusDiv = document.getElementById('ingestion-status');

        if (ingestButton && fileInput) { // Ensure fileInput exists
            ingestButton.addEventListener('click', async () => {
                const files = fileInput.files; // Get FileList object

                if (!files || files.length === 0) {
                    statusDiv.innerHTML = '<p style="color: red;">Please select one or more files to ingest.</p>';
                    return;
                }

                statusDiv.innerHTML = '<div class="loader">Ingesting documents...</div>'; // Use loader class

                try {
                    const data = await startIngestion(files); // Pass FileList to api.js function
                    let html = `<p style="color: green;">${data.message || 'Ingestion process initiated.'}</p>`;
                    if (data.documents_processed !== undefined) {
                        html += `<p>Documents processed: ${data.documents_processed}</p>`;
                    }
                    if (data.chunks_added !== undefined) {
                        html += `<p>Chunks added: ${data.chunks_added}</p>`;
                    }
                    if (data.errors && data.errors.length > 0) {
                        html += `<p style="color: orange;">Encountered issues with some documents:</p><ul>`;
                        data.errors.forEach(err => {
                            html += `<li>${err}</li>`;
                        });
                        html += `</ul>`;
                    }
                    statusDiv.innerHTML = html;
                    fileInput.value = ''; // Clear the file input on success
                } catch (error) {
                    // Log detailed error to console
                    console.error('Ingestion processing error:', error);
                    if (error.body) {
                        console.error('Error body (parsed):', error.body);
                    } else if (error.rawBody) {
                        console.error('Error body (raw):', error.rawBody);
                    }
                    // Display user-friendly message
                    let userMessage = "An error occurred during ingestion. Please try again later.";
                    if (error.body && error.body.message) {
                        userMessage = error.body.message; // Show backend message if available
                    } else if (error.message.includes("NetworkError") || error.message.includes("Failed to fetch")) {
                        userMessage = "Network error. Please check your connection or if the server is reachable.";
                    }
                    statusDiv.innerHTML = `<p style="color: red;"><strong>Ingestion Failed:</strong> ${userMessage}</p>`;
                }
            });
        }
    }
});
