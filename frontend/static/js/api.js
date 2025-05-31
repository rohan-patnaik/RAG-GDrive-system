// API base URL for Netlify functions, routed by netlify.toml
const API_BASE_URL = '/api';

async function queryBackend(queryText, llmProvider = 'gemini', topK = 3, similarityThreshold = 0.7) {
    console.log('Querying backend with:', queryText, "Provider:", llmProvider, "TopK:", topK, "Threshold:", similarityThreshold);

    const payload = {
        query_text: queryText,
        // Only include optional parameters if they are not at their default or explicitly passed
        // For now, sending them always. Can be optimized later if needed.
        llm_provider: llmProvider,
        top_k: topK
    };
    // Add similarity_threshold only if it's not null/undefined, as backend might have specific default
    if (similarityThreshold !== null && similarityThreshold !== undefined) {
        payload.similarity_threshold = similarityThreshold;
    }

    const response = await fetch(`${API_BASE_URL}/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
    });

    if (!response.ok) {
        const errorText = await response.text();
        const error = new Error(`API Error: ${response.status} ${response.statusText}`);
        try {
            error.body = JSON.parse(errorText); // Attempt to parse as JSON
        } catch (e) {
            error.rawBody = errorText; // Fallback to raw text if not JSON
        }
        console.error('API Error Details:', error.body || error.rawBody);
        throw error;
    }
    return response.json();
}

async function getSystemStatus() {
    console.log('Attempting to get system status from /api/health.');
    const response = await fetch(`${API_BASE_URL}/health`, {
        method: 'GET'
    });

    if (!response.ok) {
        const errorText = await response.text();
        const error = new Error(`API Error: ${response.status} ${response.statusText}`);
        try {
            error.body = JSON.parse(errorText); // Attempt to parse as JSON
        } catch (e) {
            error.rawBody = errorText; // Fallback to raw text if not JSON
        }
        console.error('API Error Details:', error.body || error.rawBody);
        throw error;
    }
    return response.json();
}

async function startIngestion(files) { // Expects a FileList object
    console.log('Attempting to start ingestion for', files.length, 'file(s).');

    const formData = new FormData();
    for (let i = 0; i < files.length; i++) {
        formData.append('files', files[i], files[i].name); // 'files' is the field name expected by ingest.py
    }

    const response = await fetch(`${API_BASE_URL}/ingest`, {
        method: 'POST',
        body: formData
        // Note: Content-Type header is NOT set here; browser sets it for FormData
    });

    if (!response.ok) {
        const errorText = await response.text();
        const error = new Error(`API Error: ${response.status} ${response.statusText}`);
        try {
            error.body = JSON.parse(errorText);
        } catch (e) {
            error.rawBody = errorText;
        }
        console.error('API Error Details (Ingestion):', error.body || error.rawBody);
        throw error;
    }
    return response.json();
}
