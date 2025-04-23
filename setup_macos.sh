#!/bin/bash
# Wake Word Detector - macOS Setup Script
# This script sets up the environment for the wake word detector on macOS

# Set text colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to check command success
check_success() {
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Success${NC}"
    else
        echo -e "${RED}✗ Error: $1${NC}"
        exit 1
    fi
}

# Function to print progress steps
progress_step() {
    echo -e "\n${BLUE}==>${NC} ${YELLOW}$1${NC}"
}

# Function to create a virtual environment
create_venv() {
    progress_step "Creating Python virtual environment..."
    
    # Check if venv directory exists
    if [ -d "venv" ]; then
        echo "Virtual environment already exists. Continuing with existing environment."
    else
        python3 -m venv venv
        check_success "Failed to create virtual environment"
    fi
    
    # Activate virtual environment
    progress_step "Activating virtual environment..."
    source venv/bin/activate
    check_success "Failed to activate virtual environment"
    
    # Upgrade pip
    progress_step "Upgrading pip..."
    pip install --upgrade pip
    check_success "Failed to upgrade pip"
}

# Function to install system dependencies
install_system_deps() {
    progress_step "Checking for Homebrew..."
    
    # Check if brew is installed
    if ! command -v brew &> /dev/null; then
        echo "Homebrew not found. Please install Homebrew first:"
        echo "  /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
        exit 1
    fi
    
    progress_step "Installing system dependencies with Homebrew..."
    brew install portaudio
    check_success "Failed to install system dependencies"
}

# Function to install Python dependencies
install_python_deps() {
    progress_step "Installing Python dependencies..."
    
    # Set environment variables for PyAudio
    export LDFLAGS="-L/opt/homebrew/lib"
    export CPPFLAGS="-I/opt/homebrew/include"
    
    # Install dependencies for Vosk implementation (main)
    echo "Installing packages for Vosk implementation..."
    pip install vosk webrtcvad pyaudio numpy setuptools
    check_success "Failed to install Python dependencies for Vosk"
    
    # Install dependencies for Porcupine implementation (optional)
    echo "Installing packages for Porcupine implementation..."
    pip install pvporcupine
    check_success "Failed to install Python dependencies for Porcupine"
}

# Function to download and set up Vosk model
setup_vosk_model() {
    progress_step "Setting up Vosk model..."
    
    # Check if model directory already exists
    if [ -d "model/am" ] && [ -d "model/conf" ]; then
        echo "Vosk model directory already exists. Skipping download."
        return 0
    fi
    
    # Create model directory
    mkdir -p model
    
    # Download small English model
    progress_step "Downloading Vosk model (small English)..."
    curl -L -o model.zip https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
    check_success "Failed to download Vosk model"
    
    # Unzip model
    progress_step "Extracting model files..."
    unzip -q model.zip
    check_success "Failed to extract model files"
    
    # Move model files
    mv vosk-model-small-en-us-0.15/* model/
    check_success "Failed to move model files"
    
    # Clean up
    rm -rf vosk-model-small-en-us-0.15 model.zip
    check_success "Failed to clean up temporary files"
}

# Function to create recordings directory
setup_recordings_dir() {
    progress_step "Creating recordings directory..."
    mkdir -p recordings
    check_success "Failed to create recordings directory"
}

# Main function
main() {
    echo -e "${GREEN}=====================================${NC}"
    echo -e "${GREEN}  Wake Word Detector Setup - macOS   ${NC}"
    echo -e "${GREEN}=====================================${NC}"
    
    create_venv
    install_system_deps
    install_python_deps
    setup_vosk_model
    setup_recordings_dir
    
    echo -e "\n${GREEN}Setup complete!${NC}"
    echo -e "\nTo use the wake word detector:"
    echo -e "1. Activate the virtual environment:${YELLOW} source venv/bin/activate${NC}"
    echo -e "2. Run the Vosk implementation:${YELLOW} python local_wake_word.py${NC}"
    echo -e "   Or with a custom wake word:${YELLOW} python local_wake_word.py --wake_word \"hey buddy\"${NC}"
    echo -e "\nFor Porcupine implementation (requires access key):"
    echo -e "${YELLOW}python wake_word_recorder.py --access_key YOUR_ACCESS_KEY${NC}"
    echo -e "\nSee README.md for more details."
}

# Run main function
main

