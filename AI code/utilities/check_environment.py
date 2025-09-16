"""
FireAIDSS Environment Checker
=============================

Quick script to validate the conda "AI" environment meets training requirements.
Run this before launching training.

Usage:
    conda activate AI
    python check_environment.py
"""

import sys
import importlib
from pathlib import Path

def check_python_version():

    version = sys.version_info
    print(f"🐍 Python Version: {version.major}.{version.minor}.{version.micro}")
    
    if version >= (3, 8):
        print("   ✅ Python version OK")
        return True
    else:
        print("   ❌ Need Python 3.8+")
        return False

def check_package(package_name, min_version=None):

    try:
        module = importlib.import_module(package_name)
        version = getattr(module, '__version__', 'unknown')
        print(f"📦 {package_name}: {version}")
        
        if min_version and hasattr(module, '__version__'):
            # Simple version comparison (works for most cases)
            if version >= min_version:
                print(f"   ✅ Version OK")
            else:
                print(f"   ⚠️  Version {version} < {min_version}")
        else:
            print(f"   ✅ Installed")
        return True
        
    except ImportError:
        print(f"❌ {package_name}: Not installed")
        return False

def check_pytorch():

    try:
        import torch
        print(f"🔥 PyTorch: {torch.__version__}")
        
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            
            print(f"   ✅ CUDA Available")
            print(f"   🎮 GPU: {gpu_name}")
            print(f"   💾 Memory: {gpu_memory:.1f} GB")
            
            if gpu_memory >= 8.0:
                print(f"   ✅ GPU memory sufficient")
            elif gpu_memory >= 6.0:
                print(f"   ⚠️  GPU memory adequate (may be slow)")
            else:
                print(f"   ❌ GPU memory insufficient (<6GB)")
            
            return True
        else:
            print(f"   ❌ CUDA not available - training will be very slow")
            return False
            
    except ImportError:
        print(f"❌ PyTorch: Not installed")
        return False

def check_data_directory():

    data_path = Path("data")
    print(f"📁 Data Directory: {data_path.absolute()}")
    
    if not data_path.exists():
        print(f"   ❌ Data directory not found")
        return False
    
    print(f"   ✅ Data directory exists")
    
    # Check for simulation scenarios
    scenarios = []
    for i in range(3):
        for j in range(4):
            scenario = f"gxb{i}-{j}"
            scenario_path = data_path / scenario
            if scenario_path.exists():
                scenarios.append(scenario)
    
    print(f"   📊 Found {len(scenarios)}/12 simulation scenarios")
    
    if len(scenarios) >= 8:
        print(f"   ✅ Sufficient scenarios for training")
        return True
    elif len(scenarios) >= 4:
        print(f"   ⚠️  Limited scenarios - training possible but not optimal")
        return True
    else:
        print(f"   ❌ Insufficient scenarios for training")
        return False

def main():

    print("🔍 FIREAIDSS ENVIRONMENT CHECK")
    print("=" * 50)
    
    checks = []
    
    # Python version
    checks.append(check_python_version())
    print()
    
    # Essential packages
    essential_packages = [
        'numpy',
        'pandas', 
        'matplotlib',
        'yaml',
        'tqdm'
    ]
    
    for package in essential_packages:
        checks.append(check_package(package))
    print()
    
    # PyTorch with CUDA
    checks.append(check_pytorch())
    print()
    
    # Optional packages
    optional_packages = ['wandb']
    print("📦 Optional Packages:")
    for package in optional_packages:
        check_package(package)
    print()
    
    # Data directory
    checks.append(check_data_directory())
    print()
    
    # Summary
    passed = sum(checks)
    total = len(checks)
    
    print("=" * 50)
    print(f"📊 ENVIRONMENT CHECK SUMMARY")
    print("=" * 50)
    print(f"✅ Passed: {passed}/{total} checks")
    
    if passed == total:
        print("🎉 Environment ready for FireAIDSS training!")
        print("\nTo start training:")
        print("   python launch_training.py")
        return True
    elif passed >= total - 1:
        print("⚠️  Environment mostly ready - training possible with limitations")
        print("\nTo start training:")
        print("   python launch_training.py")
        return True
    else:
        print("❌ Environment not ready - please install missing packages")
        print("\nTo install missing packages:")
        print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        print("   pip install numpy pandas matplotlib pyyaml tqdm wandb")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
