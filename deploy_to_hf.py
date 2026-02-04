"""
Automated deployment script for Hugging Face Spaces
"""

import os
import shutil
import subprocess
import argparse

def deploy_to_huggingface(username: str, space_name: str = "deepfake-detector"):
    """Deploy the app to Hugging Face Spaces"""
    
    print("🚀 Deploying to Hugging Face Spaces...")
    
    # Create temp directory for HF space
    hf_dir = os.path.join(os.path.dirname(__file__), "hf_space_deploy")
    if os.path.exists(hf_dir):
        shutil.rmtree(hf_dir)
    os.makedirs(hf_dir)
    
    # Files to copy
    files_to_copy = [
        ("app_gradio.py", "app.py"),
        ("requirements_hf.txt", "requirements.txt"),
    ]
    
    # Directories to copy
    dirs_to_copy = [
        "models",
        "test_data",
    ]
    
    # Copy files
    for src, dst in files_to_copy:
        src_path = os.path.join(os.path.dirname(__file__), src)
        dst_path = os.path.join(hf_dir, dst)
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
            print(f"  ✓ Copied {src} -> {dst}")
    
    # Copy directories
    for dir_name in dirs_to_copy:
        src_path = os.path.join(os.path.dirname(__file__), dir_name)
        dst_path = os.path.join(hf_dir, dir_name)
        if os.path.exists(src_path):
            shutil.copytree(src_path, dst_path)
            print(f"  ✓ Copied {dir_name}/")
    
    # Create README.md with HF Space config
    readme_content = f"""---
title: DeepFake Detector
emoji: 🔍
colorFrom: purple
colorTo: blue
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: true
license: mit
---

# 🔍 DeepFake & AI-Generated Image Detector

Upload an image to detect deepfakes and AI-generated content.

## Features
- 🧠 EfficientNet deep learning models
- 📊 Frequency domain analysis
- 🔬 Noise pattern detection
- ⚡ GPU acceleration

## How to Use
1. Upload an image
2. Click "Analyze Image"
3. View the detection results

Made with ❤️ by {username}
"""
    
    with open(os.path.join(hf_dir, "README.md"), "w") as f:
        f.write(readme_content)
    print("  ✓ Created README.md")
    
    # Initialize git and push
    print("\n📦 Pushing to Hugging Face...")
    
    space_url = f"https://huggingface.co/spaces/{username}/{space_name}"
    
    commands = [
        f"cd {hf_dir}",
        "git init",
        "git lfs install",
        "git lfs track '*.pth'",
        "git add .",
        f'git commit -m "Deploy DeepFake Detector"',
        f"git remote add origin https://huggingface.co/spaces/{username}/{space_name}",
        "git push -u origin main --force"
    ]
    
    print(f"\n📋 Run these commands to complete deployment:\n")
    print("=" * 60)
    for cmd in commands:
        print(cmd)
    print("=" * 60)
    
    print(f"\n✅ Files prepared in: {hf_dir}")
    print(f"🌐 Your Space will be at: {space_url}")
    print(f"\n⚠️  First, create the Space at: https://huggingface.co/new-space")
    print(f"    - Name: {space_name}")
    print(f"    - SDK: Gradio")
    print(f"    - Hardware: GPU basic (for faster inference)")
    
    return hf_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deploy to Hugging Face Spaces")
    parser.add_argument("--username", required=True, help="Your Hugging Face username")
    parser.add_argument("--space-name", default="deepfake-detector", help="Space name")
    
    args = parser.parse_args()
    deploy_to_huggingface(args.username, args.space_name)
