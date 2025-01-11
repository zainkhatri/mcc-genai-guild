import subprocess
import sys

def install_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

required_packages = [
    'lm-eval',
    'openai',
    'anthropic',
    'google-cloud-aiplatform',
    'python-dotenv',
    'jsonlines',
    'transformers',
    'torch',
    'numpy',
    'pandas',
    'tqdm',
    'datasets',
    'evaluate',
    'scikit-learn'
]

def setup():
    print("Checking and installing required packages...")
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✓ {package} already installed")
        except ImportError:
            print(f"Installing {package}...")
            install_package(package)
            print(f"✓ {package} installed successfully")

if __name__ == "__main__":
    setup()