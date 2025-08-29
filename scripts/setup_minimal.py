#!/usr/bin/env python3
"""
Minimal setup script - only install missing essential packages.
"""
import subprocess
import sys
import importlib

def check_package(package_name, import_name=None):
    """Check if a package is available."""
    if import_name is None:
        import_name = package_name
    
    try:
        importlib.import_module(import_name)
        return True
    except ImportError:
        return False

def install_package(package_name):
    """Install a package using pip."""
    print(f"Installing {package_name}...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        return True
    except subprocess.CalledProcessError:
        print(f"Failed to install {package_name}")
        return False

def main():
    print("🔍 Checking essential packages for SAQ Phase 1...")
    
    # Essential packages
    essential_packages = [
        ("datasets", "datasets"),
        ("tqdm", "tqdm"),
    ]
    
    # Optional packages
    optional_packages = [
        ("bitsandbytes", "bitsandbytes"),
    ]
    
    missing_essential = []
    missing_optional = []
    
    # Check essential packages
    for pkg_name, import_name in essential_packages:
        if not check_package(pkg_name, import_name):
            missing_essential.append(pkg_name)
        else:
            print(f"{pkg_name} is available")
    
    # Check optional packages
    for pkg_name, import_name in optional_packages:
        if not check_package(pkg_name, import_name):
            missing_optional.append(pkg_name)
        else:
            print(f"{pkg_name} is available")
    
    # Check core packages
    core_packages = [
        ("torch", "torch"),
        ("transformers", "transformers"),
        ("numpy", "numpy"),
    ]
    
    print("\n Core packages status:")
    for pkg_name, import_name in core_packages:
        if check_package(pkg_name, import_name):
            print(f"{pkg_name} is available")
            if pkg_name == "torch":
                import torch
                print(f"   - Version: {torch.__version__}")
                print(f"   - CUDA available: {torch.cuda.is_available()}")
        else:
            print(f"{pkg_name} is missing")
    
    # Install missing essential packages
    if missing_essential:
        print(f"\nInstalling missing essential packages: {missing_essential}")
        for pkg in missing_essential:
            install_package(pkg)
    else:
        print("\n All essential packages are available!")
    
    # Install missing optional packages (if space allows)
    if missing_optional:
        print(f"\nOptional packages missing: {missing_optional}")
        print("Attempting to install (may fail due to disk space)...")
        for pkg in missing_optional:
            install_package(pkg)
    
    print("\n Setup complete! You can now run Phase 1 experiments.")

if __name__ == "__main__":
    main() 