#!/bin/bash
# Wake Word Detector - Linux/Raspberry Pi Setup Script
# This script sets up the environment for the wake word detector on Linux/Raspberry Pi

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
    
    # Check for python3-venv package
    if ! dpkg -l | grep -q python3-venv; then
        progress_step "Installing python3-venv package..."
        sudo apt-get update
        sudo apt-get install -y python3-venv
        check_success "Failed to install python3-venv package"
    fi
    
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
    progress_step "Installing system dependencies..."
    
    sudo apt-get update
    check_success "Failed to update package lists"
    
    sudo apt-get install -y python3-dev python3-pip libportaudio2 libportaudiocpp0 portaudio19-dev libatlas-base-dev
    check_success "Failed to install system dependencies"
}

# Function to install Python dependencies
install_python_deps() {
    progress_step "Installing Python dependencies..."
    
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
    
    # Install wget if not available (some minimal Raspberry Pi OS may not have it)
    if ! command -v wget &> /dev/null; then
        sudo apt-get install -y wget
        check_success "Failed to install wget"
    fi
    
    # Create model directory
    mkdir -p model
    
    # Download small English model
    progress_step "Downloading Vosk model (small English)..."
    wget -q --show-progress -O model.zip https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
    check_success "Failed to download Vosk model"
    
    # Install unzip if not available
    if ! command -v unzip &> /dev/null; then
        sudo apt-get install -y unzip
        check_success "Failed to install unzip"
    fi
    
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

# Function to add user to audio group (for Raspberry Pi)
setup_audio_permissions() {
    progress_step "Setting up audio permissions..."
    
    # Check if we're on a Pi or similar system
    if groups | grep -q "audio"; then
        echo "User already in audio group. Skipping."
    else
        if getent group audio > /dev/null; then
            sudo usermod -a -G audio $USER
            echo "Added user to audio group. You may need to log out and back in for changes to take effect."
        else
            echo "Audio group not found. Skipping audio permissions setup."
        fi
    fi
}

# Main function
main() {
    echo -e "${GREEN}=============================================${NC}"
    echo -e "${GREEN}  Wake Word Detector Setup - Linux/Raspberry Pi   ${NC}"
    echo -e "${GREEN}=============================================${NC}"
    
    create_venv
    install_system_deps
    install_python_deps
    setup_vosk_model
    setup_recordings_dir
    setup_audio_permissions
    
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

