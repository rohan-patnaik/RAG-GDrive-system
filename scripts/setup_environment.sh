#!/bin/bash

# Script to set up the development environment for the RAG GDrive System

# Exit immediately if a command exits with a non-zero status.
set -e

PYTHON_VERSION_MIN="3.9"
VENV_DIR=".venv"

echo "Checking Python version..."
# Check if python3 is available
if ! command -v python3 &> /dev/null
then
    echo "python3 could not be found. Please install Python $PYTHON_VERSION_MIN or higher."
    exit 1
fi

# Get python version (handles variations in output format)
PYTHON_VERSION_FULL=$(python3 -V 2>&1 | awk '{print $2}')

# Compare versions (simple lexicographical comparison, works for X.Y.Z formats)
if [[ "$(printf '%s\n' "$PYTHON_VERSION_MIN" "$PYTHON_VERSION_FULL" | sort -V | head -n1)" != "$PYTHON_VERSION_MIN" ]]; then
    echo "Python version $PYTHON_VERSION_FULL is installed, but version $PYTHON_VERSION_MIN or higher is required."
    echo "Please upgrade your Python installation."
    exit 1
else
    echo "Python version $PYTHON_VERSION_FULL found. OK."
fi


# Check if virtual environment directory exists
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating Python virtual environment in '$VENV_DIR'..."
    python3 -m venv $VENV_DIR
    if [ $? -ne 0 ]; then
        echo "Failed to create virtual environment. Please check your Python installation."
        exit 1
    fi
else
    echo "Virtual environment '$VENV_DIR' already exists."
fi

# Activate virtual environment
echo "Activating virtual environment..."
source $VENV_DIR/bin/activate
if [ $? -ne 0 ]; then
    echo "Failed to activate virtual environment."
    echo "Try activating it manually: source $VENV_DIR/bin/activate"
    exit 1
fi
echo "Virtual environment activated. Python executable: $(which python)"

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip
if [ $? -ne 0 ]; then
    echo "Failed to upgrade pip."
    # Continue, as this is not always critical
fi

# Install dependencies
if [ -f "requirements.txt" ]; then
    echo "Installing dependencies from requirements.txt..."
    pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "Failed to install dependencies from requirements.txt."
        echo "Please check the file and your network connection."
        exit 1
    fi
else
    echo "Warning: requirements.txt not found. Skipping dependency installation."
fi

# Install development dependencies (if pyproject.toml and [project.optional-dependencies] dev is used)
# This assumes you might want to install editable mode for the local package too.
echo "Installing project in editable mode with development dependencies..."
# The command `pip install -e .[dev]` installs the current directory (`.`) as an editable package
# and includes the optional dependencies specified under the `dev` group in `pyproject.toml`.
pip install -e ".[dev,streamlit]" # Install main, dev, and streamlit extras
if [ $? -ne 0 ]; then
    echo "Failed to install project in editable mode with dev/streamlit dependencies."
    echo "You might need to run 'pip install -r requirements.txt' again if it includes all deps."
    # exit 1 # Decide if this is critical
fi


# Create .env file from .env.example if it doesn't exist
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        echo "Creating .env file from .env.example..."
        cp .env.example .env
        echo ".env file created. Please fill in your API keys and other configurations in .env"
    else
        echo "Warning: .env.example not found. Cannot create .env file automatically."
        echo "Please create a .env file manually with the required environment variables."
    fi
else
    echo ".env file already exists. Skipping creation."
fi

# Create data directories if they don't exist
echo "Creating data and logs directories if they don't exist..."
mkdir -p data/sample_documents
mkdir -p data/vector_store
mkdir -p logs

echo ""
echo "---------------------------------------------------------------------"
echo "Development environment setup complete!"
echo "---------------------------------------------------------------------"
echo ""
echo "Next steps:"
echo "1. If you haven't already, fill in your API keys in the '.env' file."
echo "2. The virtual environment is active. If you open a new terminal, activate it using:"
echo "   source $VENV_DIR/bin/activate"
echo "3. To run the API (example):"
echo "   uvicorn rag_system.api.app:app --host 0.0.0.0 --port 8000 --reload --app-dir backend/"
echo "4. To use the CLI (after 'pip install -e .'):"
echo "   rag-system --help"
echo "5. To run tests:"
echo "   pytest"
echo ""

exit 0
