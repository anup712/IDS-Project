#!/bin/bash
# Resume IDS Project - activates venv and opens GUI

echo "🔹 Starting IDS Project..."

# Navigate to project folder
cd ~/IDS_Project

# Activate Python virtual environment
source venv/bin/activate

# Move into scripts so relative paths inside scripts work correctly
cd scripts

# Launch IDS Dashboard (GUI)
python3 ids_dashboard.py

# Or, to run test_samples instead, uncomment:
# python3 test_samples.py
