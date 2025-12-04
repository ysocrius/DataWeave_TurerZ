# 🚀 Deployment Guide - DataWeave

## Render.com Deployment (Recommended)

### Prerequisites
- GitHub repository (✅ Already done!)
- Render.com account (free tier available)
- OpenAI API key

### Quick Deploy Steps

1. **Go to Render Dashboard**
   - Visit: https://render.com
   - Sign in with GitHub

2. **Create New Blueprint**
   - Click "New" → "Blueprint"
   - Connect your GitHub repository: `ysocrius/DataWeave_TurerZ`
   - Render will auto-detect `render.yaml`

3. **Set Environment Variables**
   
   **For Backend Service:**
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   MONGODB_ATLAS_URI=your_mongodb_uri (optional)
   ```

4. **Deploy**
   - Click "Apply"
   - Render will automatically deploy both services
   - Wait 5-10 minutes for initial build

5. **Access Your App**
   - Backend: `https://dataweave-backend.onrender.com`
   - Frontend: `https://dataweave-frontend.onrender.com`

---

## Manual Deployment (Alternative)

### Backend Deployment

**On Render:**
```yaml
Build Command: pip install -r Assignment_r/requirements.txt
Start Command: cd Assignment_r && uvicorn backend:app --host 0.0.0.0 --port $PORT
```

**Environment Variables:**
- `OPENAI_API_KEY` - Your OpenAI API key
- `PYTHON_VERSION` - 3.11.0
- `LLM_MODEL` - gpt-4o-mini
- `LLM_TEMPERATURE` - 0.0

### Frontend Deployment

**On Render:**
```yaml
Build Command: cd ai-doc-processor-frontend && npm install && npm run build
Start Command: cd ai-doc-processor-frontend && npm run preview -- --host 0.0.0.0 --port $PORT
```

**Environment Variables:**
- `VITE_API_URL` - URL of your deployed backend

---

## Local Testing Before Deploy

### Test Backend:
```bash
cd Assignment_r
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux
pip install -r requirements.txt
uvicorn backend:app --reload
```

### Test Frontend:
```bash
cd ai-doc-processor-frontend
npm install
npm run build
npm run preview
```

---

## Environment Variables Reference

### Required:
- `OPENAI_API_KEY` - OpenAI API key for GPT-4

### Optional:
- `MONGODB_ATLAS_URI` - MongoDB connection string (for learning system)
- `LLM_MODEL` - AI model to use (default: gpt-4o-mini)
- `LLM_TEMPERATURE` - Temperature setting (default: 0.0)
- `LEARNING_ENABLED` - Enable learning system (default: false)

---

## Troubleshooting

### Build Fails
- Check Python version is 3.11+
- Verify all dependencies in requirements.txt
- Check build logs in Render dashboard

### Backend Won't Start
- Verify OPENAI_API_KEY is set
- Check PORT environment variable is used
- Review application logs

### Frontend Can't Connect
- Verify VITE_API_URL points to backend
- Check CORS settings in backend
- Ensure backend is running

---

## Cost Estimate

**Render Free Tier:**
- ✅ Backend: Free (with 750 hours/month)
- ✅ Frontend: Free (with 750 hours/month)
- ⚠️ Services sleep after 15 min inactivity
- ⚠️ Cold start: ~30 seconds

**Render Paid Tier ($7/month per service):**
- ✅ Always on (no sleep)
- ✅ Faster performance
- ✅ Custom domains

**OpenAI API Costs:**
- ~$0.01-0.03 per document (GPT-4o-mini)
- ~$1-3 for 100 documents

---

## Production Checklist

- [ ] Set all environment variables
- [ ] Test with sample documents
- [ ] Monitor API costs
- [ ] Set up error tracking (Sentry)
- [ ] Configure custom domain (optional)
- [ ] Enable HTTPS (automatic on Render)
- [ ] Set up monitoring/alerts

---

## Support

For deployment issues:
- Render Docs: https://render.com/docs
- GitHub Issues: https://github.com/ysocrius/DataWeave_TurerZ/issues
