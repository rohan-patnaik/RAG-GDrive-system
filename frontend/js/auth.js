class AuthManager {
    constructor() {
        this.backendAccessKey =
            localStorage.getItem("rag_backend_access_key") || "";
        this.authMessageElement = document.getElementById("auth-message");
        this.loadAccessKeyInput();
    }

    loadAccessKeyInput() {
        const keyInput = document.getElementById("backend-access-key");
        if (keyInput) {
            keyInput.value = this.backendAccessKey;
        }
    }

    saveAccessKey() {
        const keyInput = document.getElementById("backend-access-key");
        if (keyInput) {
            this.backendAccessKey = keyInput.value.trim();
            if (this.backendAccessKey) {
                localStorage.setItem(
                    "rag_backend_access_key",
                    this.backendAccessKey,
                );
                this.showMessage("Backend Access Key saved successfully!", "success");
            } else {
                localStorage.removeItem("rag_backend_access_key");
                this.showMessage("Backend Access Key cleared.", "info");
            }
        } else {
            console.error("Backend access key input field not found.");
            this.showMessage("Error: Input field not found.", "error");
        }
    }

    getAuthHeaders() {
        if (!this.backendAccessKey) {
            this.showMessage(
                "Backend Access Key is not configured. Please save it first.",
                "error",
            );
            throw new Error("Backend Access Key not configured");
        }
        return {
            // Standard Authorization header
            Authorization: `Bearer ${this.backendAccessKey}`,
            // Common alternative for API keys
            "X-API-Key": this.backendAccessKey,
        };
    }

    showMessage(message, type = "info") {
        if (this.authMessageElement) {
            this.authMessageElement.textContent = message;
            this.authMessageElement.className = `message ${type}`;
            this.authMessageElement.style.display = "block";

            setTimeout(() => {
                this.authMessageElement.style.display = "none";
            }, 5000);
        } else {
            // Fallback if the dedicated auth message div isn't there
            const globalMessageDiv = document.createElement("div");
            globalMessageDiv.className = `message ${type}`;
            globalMessageDiv.textContent = message;
            globalMessageDiv.style.position = "fixed";
            globalMessageDiv.style.top = "20px";
            globalMessageDiv.style.right = "20px";
            globalMessageDiv.style.zIndex = "1000";
            document.body.appendChild(globalMessageDiv);
            setTimeout(() => globalMessageDiv.remove(), 5000);
        }
    }
}

// Initialize auth manager when the DOM is ready
document.addEventListener("DOMContentLoaded", () => {
    window.authManager = new AuthManager();
});