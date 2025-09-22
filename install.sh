#!/bin/bash
# Installation script for Enhanced SWE-Bench Submission

echo "🚀 Installing Enhanced SWE-Bench Submission Dependencies"
echo "=================================================="

# Check Python version
python_version=$(python3 --version 2>&1)
echo "📋 Python version: $python_version"

# Install core dependencies
echo "📦 Installing core dependencies..."
pip install --upgrade pip

# Install psutil specifically
echo "🔧 Installing psutil..."
pip install psutil>=7.0.0

# Install all requirements
echo "📋 Installing all requirements..."
pip install -r requirements.txt

# Verify critical imports
echo "🧪 Verifying critical imports..."
python3 -c "
try:
    import psutil
    print(f'✅ psutil {psutil.__version__} - OK')
except ImportError as e:
    print(f'❌ psutil - FAILED: {e}')

try:
    import requests
    print(f'✅ requests {requests.__version__} - OK')
except ImportError as e:
    print(f'❌ requests - FAILED: {e}')

try:
    import git
    print(f'✅ GitPython - OK')
except ImportError as e:
    print(f'❌ GitPython - FAILED: {e}')

try:
    import concurrent.futures
    print(f'✅ concurrent.futures - OK')
except ImportError as e:
    print(f'❌ concurrent.futures - FAILED: {e}')
"

echo ""
echo "🎉 Installation complete!"
echo ""
echo "📋 To test the submission:"
echo "   python3 test_compatibility.py"
echo ""
echo "📋 To run a sample task:"
echo "   python3 test_agent.py"
