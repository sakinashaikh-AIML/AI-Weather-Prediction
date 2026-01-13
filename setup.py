#!/usr/bin/env python3
"""
Fixed setup script for Real-Time Weather Prediction AI
"""

import os
import sys
import subprocess
import webbrowser

def main():
    print("=" * 60)
    print("🤖 REAL-TIME WEATHER AI - FIXED SETUP")
    print("=" * 60)
    
    # Check Python version
    print("\n🔍 Checking Python version...")
    if sys.version_info < (3, 8):
        print(f"❌ Python 3.8+ required. You have {sys.version_info.major}.{sys.version_info.minor}")
        return
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro} detected")
    
    # Create virtual environment
    print("\n🐍 Creating virtual environment...")
    if not os.path.exists('venv'):
        subprocess.run([sys.executable, '-m', 'venv', 'venv'], check=True)
        print("✅ Virtual environment created")
    else:
        print("✅ Virtual environment already exists")
    
    # Install dependencies
    print("\n📦 Installing dependencies...")
    requirements = [
        'Flask==3.0.0',
        'requests==2.31.0',
        'numpy==1.26.2',
        'pandas==2.1.4',
        'scikit-learn==1.3.2',
        'joblib==1.3.2',
        'python-dotenv==1.0.0'
    ]
    
    # Use pip from virtual environment
    if os.name == 'nt':  # Windows
        pip_path = 'venv\\Scripts\\pip'
        python_path = 'venv\\Scripts\\python'
    else:  # Linux/Mac
        pip_path = 'venv/bin/pip'
        python_path = 'venv/bin/python'
    
    # Upgrade pip first
    subprocess.run([python_path, '-m', 'pip', 'install', '--upgrade', 'pip'], check=True)
    
    # Install packages
    for package in requirements:
        try:
            subprocess.run([python_path, '-m', 'pip', 'install', package], check=True)
            print(f"   ✅ {package}")
        except:
            print(f"   ⚠ Could not install {package}")
    
    # Create necessary files
    print("\n📁 Creating project structure...")
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Create .env file if it doesn't exist
    if not os.path.exists('.env'):
        with open('.env', 'w') as f:
            f.write('''# OpenWeatherMap API Key
# Get yours from: https://openweathermap.org/api
OPENWEATHER_API_KEY=your_api_key_here

# Flask settings
FLASK_ENV=development
FLASK_DEBUG=True
''')
        print("✅ Created .env file")
    
    # Create requirements.txt
    with open('requirements.txt', 'w') as f:
        f.write('\n'.join(requirements))
    print("✅ Created requirements.txt")
    
    # Open browser for API key
    print("\n🌐 Opening browser for API key...")
    webbrowser.open('https://openweathermap.org/api')
    
    print("\n" + "="*60)
    print("🎉 SETUP COMPLETE!")
    print("="*60)
    print("\n📝 NEXT STEPS:")
    print("1. Activate virtual environment:")
    print("   Windows: venv\\Scripts\\activate")
    print("   Mac/Linux: source venv/bin/activate")
    print("\n2. Get API key from: https://openweathermap.org/api")
    print("\n3. Edit .env file and add your API key")
    print("\n4. Run: python app.py")
    print("\n5. Open: http://localhost:5000")
    print("\n" + "="*60)

if __name__ == '__main__':
    main()