# fix_env.py - Run this to create a proper .env file
import os

print("🔧 Creating .env file for Weather Prediction App")
print("=" * 50)

# Your OpenWeather API key
api_key = "fc37682183690a8f08872139556bc720"

# Create the content
env_content = f"""OPENWEATHER_API_KEY={api_key}
"""

# Write the file
try:
    with open('.env', 'w', encoding='utf-8') as f:
        f.write(env_content)
    
    print("✅ .env file created successfully!")
    print(f"📂 Location: {os.path.abspath('.env')}")
    print("\n📝 Content:")
    print("-" * 30)
    print(env_content)
    print("-" * 30)
    
    # Verify the file
    if os.path.exists('.env'):
        with open('.env', 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if api_key in content:
                print("\n✅ Verification: API key found in .env file")
                print(f"   Key: {api_key[:8]}...{api_key[-4:]}")
            else:
                print("\n❌ Verification failed: API key not found")
    else:
        print("\n❌ Error: .env file was not created")
        
except Exception as e:
    print(f"\n❌ Error creating .env file: {e}")
    print("\nTry creating it manually:")
    print("1. Open Notepad")
    print('2. Type: OPENWEATHER_API_KEY=fc37682183690a8f08872139556bc720')
    print('3. Save as ".env" (with quotes)')

input("\nPress Enter to try running the app...")

# Try to run the app
print("\n🚀 Attempting to run app.py...")
try:
    os.system("python app.py")
except KeyboardInterrupt:
    print("\nApp stopped by user")
except Exception as e:
    print(f"Error running app: {e}")