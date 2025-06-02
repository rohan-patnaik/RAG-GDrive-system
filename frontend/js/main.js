document.addEventListener("DOMContentLoaded", () => {
    // Ensure authManager and apiClient are available
    if (!window.authManager || !window.apiClient) {
        console.error(
            "AuthManager or APIClient not initialized. Ensure auth.js and api.js are loaded correctly.",
        );
        return;
    }

    // Query form handler
    const queryForm = document.getElementById("query-form");
    if (queryForm) {
        queryForm.addEventListener("submit", async (e) => {
            e.preventDefault();
            const queryText = document.getElementById("query-text").value;
            const llmProvider = document.getElementById("llm-provider").value;
            const useCache = document.getElementById("use-cache").checked;

            if (!queryText.trim()) {
                showResults("query-results", "Please enter a question", "error");
                return;
            }

            const submitButton = e.target.querySelector('button[type="submit"]');
            const originalText = submitButton.textContent;
            submitButton.disabled = true;
            submitButton.innerHTML =
                '<span class="loading"></span> Processing...';

            try {
                const result = await window.apiClient.queryBackend({
                    query_text: queryText,
                    llm_provider: llmProvider,
                    use_cache: useCache,
                });
                displayQueryResults(result);
            } catch (error) {
                // Error message is likely already shown by APIClient or AuthManager
                console.error("Query submission error:", error.message);
                if (!document.querySelector(".message.error")) { // Show only if not already shown
                    showResults("query-results", `Error: ${error.message}`, "error");
                }
            } finally {
                submitButton.disabled = false;
                submitButton.textContent = originalText;
            }
        });
    }

    // Upload form handler
    const uploadForm = document.getElementById("upload-form");
    if (uploadForm) {
        uploadForm.addEventListener("submit", async (e) => {
            e.preventDefault();
            const filesInput = document.getElementById("files");
            const source = document.getElementById("source").value;

            if (!filesInput.files.length) {
                showResults(
                    "upload-results",
                    "Please select files to upload",
                    "error",
                );
                return;
            }

            const formData = new FormData();
            for (let file of filesInput.files) {
                formData.append("files", file); // 'files' should match backend expected key
            }
            if (source) {
                formData.append("source", source);
            }

            const submitButton = e.target.querySelector('button[type="submit"]');
            const originalText = submitButton.textContent;
            submitButton.disabled = true;
            submitButton.innerHTML =
                '<span class="loading"></span> Uploading...';

            try {
                const result = await window.apiClient.ingestDocuments(formData);
                displayUploadResults(result);
                filesInput.value = ""; // Clear file input
                document.getElementById("source").value = ""; // Clear source input
            } catch (error) {
                console.error("Upload submission error:", error.message);
                 if (!document.querySelector(".message.error")) {
                    showResults("upload-results", `Error: ${error.message}`, "error");
                }
            } finally {
                submitButton.disabled = false;
                submitButton.textContent = originalText;
            }
        });
    }

    // Initial system status check
    checkSystemStatus();
}); // End of DOMContentLoaded

function displayQueryResults(result) {
    const container = document.getElementById("query-results");
    if (!container) return;

    let html = `
        <h3>Answer ${result.from_cache ? "(from cache)" : ""}</h3>
        <div class="answer">${
            result.llm_answer ? escapeHtml(result.llm_answer) : "No answer provided."
        }</div>
        <h4>Details</h4>
        <p><strong>Provider:</strong> ${escapeHtml(result.llm_provider_used)}</p>
        <p><strong>Model:</strong> ${escapeHtml(result.llm_model_used)}</p>
        <p><strong>Retrieved Chunks:</strong> ${
            result.retrieved_chunks ? result.retrieved_chunks.length : 0
        }</p>
    `;

    if (result.retrieved_chunks && result.retrieved_chunks.length > 0) {
        html += "<h4>Source Context</h4>";
        result.retrieved_chunks.forEach((chunk) => {
            html += `
                <div class="chunk">
                    <div class="chunk-metadata">
                        Source: ${escapeHtml(chunk.metadata?.filename || "Unknown")}
                        (Similarity: ${chunk.score?.toFixed(3) || "N/A"})
                    </div>
                    <div>${escapeHtml(chunk.content?.substring(0, 200) || "")}...</div>
                </div>
            `;
        });
    }
    container.innerHTML = html;
    container.className = "results success";
}

function displayUploadResults(result) {
    const container = document.getElementById("upload-results");
    if (!container) return;

    let html = `<h3>${escapeHtml(result.message)}</h3>`;
    if (result.results && Array.isArray(result.results)) {
        result.results.forEach((fileResult) => {
            const statusClass =
                fileResult.status === "success" ? "success" : "error";
            html += `
                <div class="file-result ${statusClass}">
                    <strong>${escapeHtml(fileResult.filename)}:</strong>
                    ${
                        fileResult.status === "success"
                            ? `Successfully processed ${fileResult.total_chunks} chunks`
                            : `Error: ${escapeHtml(fileResult.error)}`
                    }
                </div>
            `;
        });
    }
    container.innerHTML = html;
    container.className = "results success"; // Or dynamically set based on overall success
}

async function checkSystemStatus() {
    // Ensure apiClient is available
    if (!window.apiClient) {
        console.warn("APIClient not ready for system status check.");
        showResults(
            "status-results",
            "System components not ready.",
            "info",
        );
        return;
    }
    try {
        const status = await window.apiClient.getSystemStatus();
        displaySystemStatus(status);
    } catch (error) {
        console.error("System status check error:", error.message);
        if (!document.querySelector(".message.error")) {
            showResults("status-results", `Error: ${error.message}`, "error");
        }
    }
}

function displaySystemStatus(status) {
    const container = document.getElementById("status-results");
    if (!container) return;

    let html = `
        <h3>System Status: ${escapeHtml(status.status)}</h3>
        <p><strong>Message:</strong> ${escapeHtml(status.message)}</p>
        <p><strong>Timestamp:</strong> ${new Date(
            status.timestamp,
        ).toLocaleString()}</p>
    `;

    if (status.components && Array.isArray(status.components)) {
        html += "<h4>Components</h4>";
        status.components.forEach((component) => {
            html += `
                <div class="component ${component.status?.toLowerCase()}">
                    ${escapeHtml(component.name)}: ${escapeHtml(component.status)}
                </div>
            `;
        });
    }
    container.innerHTML = html;
    container.className = `results ${
        status.status === "OK" ? "success" : "error"
    }`;
}

function showResults(containerId, message, type) {
    const container = document.getElementById(containerId);
    if (container) {
        container.innerHTML = escapeHtml(message);
        container.className = `results ${type}`;
    }
}

function escapeHtml(unsafe) {
    if (typeof unsafe !== "string") {
        if (unsafe === null || unsafe === undefined) return "";
        try {
            unsafe = String(unsafe);
        } catch (e) {
            return "";
        }
    }
    return unsafe
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}