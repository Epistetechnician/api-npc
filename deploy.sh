#!/bin/bash
echo "Deploying Funding Rate Analysis App..."

# Check Python version
python --version

# Install dependencies
pip install -r requirements.txt

# Run tests if you have them
# python -m pytest tests/

# Start the app
streamlit run scripts/funding_streamlit_app.py 