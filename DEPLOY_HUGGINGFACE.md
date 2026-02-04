# 🚀 Deploy to Hugging Face Spaces with GPU

## Quick Setup (5 minutes)

### Step 1: Create Hugging Face Account
1. Go to https://huggingface.co/join
2. Sign up with email or GitHub

### Step 2: Login to Hugging Face CLI
```bash
huggingface-cli login
```
Enter your token from: https://huggingface.co/settings/tokens

### Step 3: Create a New Space
```bash
huggingface-cli repo create deepfake-detector --type space --space_sdk gradio
```

### Step 4: Clone and Push
```bash
# Clone the space
git clone https://huggingface.co/spaces/YOUR_USERNAME/deepfake-detector hf-space
cd hf-space

# Copy files
copy ..\app_gradio.py app.py
copy ..\requirements_hf.txt requirements.txt
copy ..\README_HF.md README.md
xcopy ..\models models\ /E /I
xcopy ..\test_data test_data\ /E /I

# Push to Hugging Face
git add .
git commit -m "Initial deployment"
git push
```

### Step 5: Enable GPU (Optional - for faster inference)
1. Go to your Space settings
2. Under "Hardware", select "GPU basic" (free tier available)
3. Click "Apply"

## Files Needed for Deployment

```
hf-space/
├── app.py              (renamed from app_gradio.py)
├── requirements.txt    (from requirements_hf.txt)
├── README.md          (from README_HF.md - contains Space config)
├── models/
│   ├── deepfake_detector.pth
│   └── ai_generated_detector.pth
└── test_data/
    ├── real/
    └── fake/
```

## Alternative: One-Command Deploy

```bash
# From project directory
python deploy_to_hf.py --username YOUR_HF_USERNAME
```

## Your Space URL
After deployment: `https://huggingface.co/spaces/YOUR_USERNAME/deepfake-detector`

## Troubleshooting

### "Out of memory" error
- Reduce batch size or use CPU fallback
- Request GPU quota increase

### Models not loading
- Ensure .pth files are uploaded
- Check file paths in app.py

### Slow inference
- Enable GPU in Space settings
- Models load on first request (cold start)
