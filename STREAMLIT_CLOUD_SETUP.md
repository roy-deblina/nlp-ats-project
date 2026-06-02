# 🚀 Streamlit Cloud Deployment Guide

This document contains all the steps needed to deploy the ATS Validation Tool on Streamlit Cloud.

## Quick Start (5 minutes)

### 1. Push Code to GitHub

```bash
# Make sure you're on the deployment branch
git branch

# Add all changes
git add .

# Commit
git commit -m "chore: prepare for streamlit cloud deployment

- Add cloud-compatible database path handling
- Update model loading to support both local and HF Hub
- Optimize llama.cpp settings for cloud CPU
- Add .streamlit/config.toml for cloud configuration
"

# Push to GitHub (master branch)
git push origin streamlit-cloud-deployment
```

### 2. Create Streamlit Cloud Account & Connect Repository

1. Go to https://streamlit.io/cloud
2. Sign up with GitHub
3. Click "New app" → "From existing repo"
4. Select: `roy-deblina/nlp-ats-project`
5. Set:
   - **Branch**: `streamlit-cloud-deployment`
   - **Main file path**: `app.py`
   - **App URL** (optional): e.g., `ats-validator`

### 3. Set Environment Variables (if needed)

In Streamlit Cloud dashboard → App settings → Secrets:

```toml
# Optional: only needed if using non-default paths
DATABASE_DIR = ".streamlit"
LLAMA_N_THREADS = "2"
```

### 4. Deploy

Click "Deploy" - Streamlit Cloud will:
- ✓ Install dependencies from `requirements.txt`
- ✓ Download embedding model from HuggingFace automatically
- ✓ Start the app
- ✓ Show your live URL

**Expected first load**: 30-60 seconds (models downloading)

---

## How Models Are Handled

### Embedding Model (MiniLM-L6-v2)

| Environment | Strategy | Details |
|-------------|----------|---------|
| **Local** | Use local files | `models/embeddings/all-MiniLM-L6-v2/` (Git excluded) |
| **Cloud** | HuggingFace Hub | Auto-downloads on first request, cached in `/tmp` |

**Code**: [ats_engine.py](ats_engine.py#L18-L22)

### LLM Model (Qwen2.5 GGUF)

| Environment | Strategy | Status |
|-------------|----------|--------|
| **Local** | Use local file | `models/llm/qwen2.5-1.5b-instruct-q4_k_m.gguf` |
| **Cloud** | ⚠️ Manual Upload | Need to add to repo or use cloud storage |

**Two Options:**

#### Option A: Upload Model to GitHub (Simpler)

```bash
# 1. Track LFS for large files
git lfs install
git lfs track "*.gguf"
git add .gitattributes
git commit -m "Enable Git LFS for GGUF models"

# 2. Add model
git add models/llm/qwen2.5-1.5b-instruct-q4_k_m.gguf
git commit -m "Add Qwen model for cloud deployment"
git push

# Note: GitHub LFS requires paid plan for >1GB
```

#### Option B: Download from Cloud Storage (Recommended)

Store model on AWS S3/Hugging Face and download on startup:

```python
# In analyzer.py (already flexible)
MODEL_PATH = os.environ.get(
    "QWEN_MODEL_PATH",
    "models/llm/qwen2.5-1.5b-instruct-q4_k_m.gguf"
)

# Add to Streamlit Cloud secrets:
# QWEN_MODEL_PATH = "https://huggingface.co/... or s3://..."
```

---

## Resource Limits & Performance

### Streamlit Cloud Free Tier

- **Memory**: 1 GB base + 512 MB per thread
- **CPU**: 1-2 vCPU
- **Timeout**: 30 seconds per interaction
- **Disk**: Limited (model caching in `/tmp`)

### App Configuration (Cloud-Optimized)

| Parameter | Local | Cloud | Reason |
|-----------|-------|-------|--------|
| LLM Context (`n_ctx`) | 2048 | 1024 | Less RAM |
| Batch Size | 128 | 64 | Memory constraints |
| GPU Layers | -1 (M1) | 0 (CPU) | No GPU on cloud |
| Threads | 4 | 2-4 | Adaptive |

**Files Updated**:
- ✓ [analyzer.py](analyzer.py) - Cloud-aware llama.cpp settings
- ✓ [ats_engine.py](ats_engine.py) - Fallback to HF Hub
- ✓ [database.py](database.py) - Cloud-friendly DB path

---

## Troubleshooting

### Issue: "Model not found"

```
FileNotFoundError: models/llm/qwen2.5-1.5b-instruct-q4_k_m.gguf
```

**Solution**: Add the model to cloud using Option A or B above.

### Issue: "Out of memory"

```
MemoryError or Killed (exit code 137)
```

**Solutions**:
1. Reduce `n_ctx` further (try 512)
2. Upgrade to Streamlit Cloud tier
3. Use HuggingFace Inference API instead of local LLM

### Issue: "Slow cold start"

First request takes 2+ minutes.

**Expected**: Yes, models download. Subsequent requests are fast (~5-10s).

### Issue: "Timeout"

App takes >30 seconds to respond.

**Solutions**:
1. Reduce context window
2. Reduce input document size
3. Cache more aggressively in Streamlit

---

## Monitoring & Logs

**View logs in Streamlit Cloud**:
1. Go to your app
2. Click "Manage app"
3. Click "View logs"

**Stream logs locally**:
```bash
streamlit run app.py --logger.level=debug
```

---

## Next Steps

1. **Test locally first**:
   ```bash
   streamlit run app.py
   ```

2. **Commit and push**:
   ```bash
   git add .
   git commit -m "streamlit cloud deployment"
   git push origin streamlit-cloud-deployment
   ```

3. **Deploy on Streamlit Cloud**

4. **Share your URL**: 
   ```
   https://share.streamlit.io/roy-deblina/nlp-ats-project/streamlit-cloud-deployment/app.py
   ```

---

## Alternative Cloud Platforms

| Platform | Models | Cost | Notes |
|----------|--------|------|-------|
| **Streamlit Cloud** | Local/HF | Free | Best for Streamlit apps |
| **Hugging Face Spaces** | Local/HF | Free | Good alternative |
| **Railway** | Local/Docker | $5/mo | More control |
| **Render** | Local/Docker | Free/tier | Good balance |
| **AWS/Azure** | Any | Pay-as-you-go | Overkill unless scaling |

---

## Contact & Support

- Issues? Check GitHub discussions
- Model questions? See [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md)
- Local setup help? See [README.md](README.md)

---

**Last Updated**: 2 June 2026  
**Branch**: `streamlit-cloud-deployment`  
**Status**: ✅ Ready for deployment
