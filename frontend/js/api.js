const API_BASE_URL = "/.netlify/functions"; // Or '/api' if using redirects like /api/*

class APIClient {
    constructor() {
        this.rateLimitInfo = {};
    }

    async makeRequest(endpoint, options = {}) {
        let headers = {
            // Default Content-Type for JSON, can be overridden
            "Content-Type": "application/json",
            ...options.headers, // Spread existing headers from options
        };

        try {
            // Get auth headers and merge them
            const authHeaders = authManager.getAuthHeaders();
            headers = { ...headers, ...authHeaders };
        } catch (error) {
            // If getAuthHeaders throws (e.g., key not set), propagate the error
            console.error("Authentication error:", error.message);
            authManager.showMessage(error.message, "error"); // Show message in UI
            throw error; // Re-throw to stop the request
        }

        // If body is FormData, remove Content-Type so browser can set it with boundary
        if (options.body instanceof FormData) {
            delete headers["Content-Type"];
        }

        try {
            const response = await fetch(`${API_BASE_URL}${endpoint}`, {
                ...options, // Spread other options like method, body
                headers, // Use the combined headers
            });

            this.updateRateLimitInfo(response.headers);

            if (!response.ok) {
                let errorData;
                try {
                    errorData = await response.json();
                } catch (e) {
                    // If response is not JSON
                    errorData = {
                        error: `HTTP ${response.status} ${response.statusText}`,
                    };
                }
                const errorMessage =
                    errorData.error ||
                    `API Request Failed: ${response.status}`;
                authManager.showMessage(errorMessage, "error");
                throw new Error(errorMessage);
            }

            // Handle cases where response might be empty (e.g., 204 No Content)
            const contentType = response.headers.get("content-type");
            if (contentType && contentType.includes("application/json")) {
                return await response.json();
            }
            return await response.text(); // Or handle as appropriate
        } catch (error) {
            console.error("API request failed:", error.message);
            // authManager.showMessage(error.message, "error"); // Already shown or will be shown
            throw error;
        }
    }

    updateRateLimitInfo(headers) {
        const limit = headers.get("X-RateLimit-Limit");
        const remaining = headers.get("X-RateLimit-Remaining");
        const reset = headers.get("X-RateLimit-Reset");

        if (limit && remaining && reset) {
            this.rateLimitInfo = {
                limit: parseInt(limit),
                remaining: parseInt(remaining),
                reset: parseInt(reset),
            };
            this.updateRateLimitUI();
        }
    }

    updateRateLimitUI() {
        const rateLimitElement = document.getElementById("rate-limit-info");
        if (rateLimitElement && Object.keys(this.rateLimitInfo).length > 0) {
            const { limit, remaining, reset } = this.rateLimitInfo;
            const resetDate = new Date(reset * 1000);
            rateLimitElement.innerHTML = `
                API Rate: ${remaining}/${limit} left
                (Resets: ${resetDate.toLocaleTimeString()})
            `;
        } else if (rateLimitElement) {
            rateLimitElement.innerHTML = ""; // Clear if no info
        }
    }

    async queryBackend(queryData) {
        return this.makeRequest("/query", {
            // Netlify function name is 'query'
            method: "POST",
            body: JSON.stringify(queryData),
        });
    }

    async getSystemStatus() {
        return this.makeRequest("/health"); // Netlify function name is 'health'
    }

    async ingestDocuments(formData) {
        return this.makeRequest("/ingest", {
            // Netlify function name is 'ingest'
            method: "POST",
            body: formData, // FormData will be handled by makeRequest
        });
    }
}

// Initialize API client when the DOM is ready
document.addEventListener("DOMContentLoaded", () => {
    window.apiClient = new APIClient();
});