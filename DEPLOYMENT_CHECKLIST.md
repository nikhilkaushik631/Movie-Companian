# ✅ Deployment Checklist - Cinemizer

Use this checklist to ensure smooth deployment of both frontend and backend.

## 🎯 Quick Links

- **Full Guide**: See [FREE_DEPLOYMENT_GUIDE.md](./FREE_DEPLOYMENT_GUIDE.md) for detailed instructions
- **Railway**: https://railway.app
- **Vercel**: https://vercel.com
- **Render** (alternative): https://render.com

---

## 📋 Pre-Deployment Checklist

### Required API Keys (Get these first!)

- [ ] Google/Gemini API Key - https://aistudio.google.com/app/apikey
- [ ] Groq API Key - https://console.groq.com
- [ ] TMDB API Key - https://www.themoviedb.org/settings/api
- [ ] OMDB API Key - http://www.omdbapi.com/apikey.aspx
- [ ] Pinecone API Key - https://www.pinecone.io
- [ ] Deepgram API Key (optional) - https://console.deepgram.com
- [ ] HuggingFace Token (optional) - https://huggingface.co/settings/tokens

### Generate Secret Key

- [ ] Generate AUTH_SECRET_KEY:
  ```bash
  python -c "import secrets; print(secrets.token_urlsafe(32))"
  ```

---

## 🚀 Backend Deployment (Railway.app)

### Step 1: Setup Railway

- [ ] Create account at https://railway.app
- [ ] Connect GitHub account
- [ ] Create new project from GitHub repo

### Step 2: Configure Railway

- [ ] Verify `railway.json` is detected
- [ ] Verify `Dockerfile.backend` is being used
- [ ] Check build logs show successful build

### Step 3: Add Environment Variables

Copy these to Railway dashboard (Settings → Variables):

```bash
GOOGLE_API_KEY=your_key_here
GEMINI_API_KEY=your_key_here
GROQ_API_KEY=your_key_here
TMDB_API_KEY=your_key_here
OMDB_API_KEY=your_key_here
PINECONE_API_KEY=your_key_here
DEEPGRAM_API_KEY=your_key_here
HUGGINGFACE_API_TOKEN=your_token_here
DATABASE_URL=sqlite:///./app.db
AUTH_SECRET_KEY=your_generated_secret
```

- [ ] All required environment variables added
- [ ] SECRET_KEY is unique and secure
- [ ] No spaces or quotes around values

### Step 4: Deploy & Verify

- [ ] Railway deployment successful
- [ ] Copy backend URL (e.g., `https://cinemizer-production.up.railway.app`)
- [ ] Test health endpoint: `https://your-url/health`
  - Should return: `{"status": "healthy", "message": "..."}`
- [ ] Test API docs: `https://your-url/docs`
  - Should show FastAPI Swagger UI

---

## 🎨 Frontend Deployment (Vercel)

### Step 1: Setup Vercel

- [ ] Create account at https://vercel.com
- [ ] Connect GitHub account
- [ ] Import your repository

### Step 2: Configure Project

Project Settings:
- [ ] **Framework Preset**: Next.js
- [ ] **Root Directory**: `frontend-next`
- [ ] **Build Command**: `npm run build` (auto-detected)
- [ ] **Output Directory**: `.next` (auto-detected)
- [ ] **Install Command**: `npm install` (auto-detected)

### Step 3: Add Environment Variables

In Vercel → Settings → Environment Variables:

```bash
NEXT_PUBLIC_API_BASE_URL=https://your-railway-backend-url.up.railway.app
NEXT_PUBLIC_DEMO=0
```

⚠️ **Critical**: Replace `your-railway-backend-url` with your ACTUAL Railway URL!

- [ ] Environment variables added
- [ ] Backend URL is correct (no trailing slash)
- [ ] Variables start with `NEXT_PUBLIC_`

### Step 4: Deploy & Verify

- [ ] Click "Deploy" button
- [ ] Wait for build to complete (2-3 minutes)
- [ ] Copy Vercel URL (e.g., `https://cinemizer.vercel.app`)
- [ ] Visit your Vercel URL
- [ ] Homepage loads correctly
- [ ] Chat page accessible

---

## 🧪 Testing Deployment

### Backend Tests

- [ ] Health check works: `GET /health`
- [ ] API docs accessible: `GET /docs`
- [ ] Can create account: `POST /signup`
- [ ] Can login: `POST /token`
- [ ] Chat endpoint works: `POST /chat` (with auth token)

Test with curl:
```bash
# Health check
curl https://your-backend-url/health

# Create account
curl -X POST https://your-backend-url/signup \
  -H "Content-Type: application/json" \
  -d '{"email":"test@example.com","password":"testpass123","display_name":"Test User"}'
```

### Frontend Tests

- [ ] Homepage loads
- [ ] Can navigate to login page
- [ ] Can create account
- [ ] Can login
- [ ] Can access chat interface
- [ ] Can send messages
- [ ] Receives AI responses
- [ ] Recommendations display correctly
- [ ] No console errors

### Integration Tests

- [ ] Frontend can communicate with backend
- [ ] No CORS errors in browser console
- [ ] Authentication flow works end-to-end
- [ ] Chat messages persist in session
- [ ] Logout works correctly

---

## 🔄 Post-Deployment

### Configure Auto-Deploy

- [ ] Railway: Auto-deploy on push (enabled by default)
- [ ] Vercel: Auto-deploy on push (enabled by default)
- [ ] Test: Push a small change and verify auto-deployment

### Monitor Resources

Railway Dashboard:
- [ ] Check memory usage (should be < 512MB)
- [ ] Check CPU usage
- [ ] Monitor remaining free credit ($5/month)
- [ ] Set up usage alerts

Vercel Dashboard:
- [ ] Check build times
- [ ] Monitor bandwidth usage
- [ ] Review deployment logs

### Set Up Monitoring

- [ ] Add Railway health check endpoint to monitoring service
- [ ] Set up UptimeRobot or similar (free) for uptime monitoring
- [ ] Configure error tracking (optional: Sentry free tier)

---

## 📝 Update Documentation

- [ ] Update README.md with deployment URLs
- [ ] Update CLAUDE.md with deployment information
- [ ] Document any deployment-specific configuration
- [ ] Add troubleshooting notes if issues encountered

---

## 🎉 Final Verification

Before sharing your app:

- [ ] Test all major features
- [ ] Verify API keys are working
- [ ] Check error handling
- [ ] Test on mobile device
- [ ] Test on different browsers
- [ ] Verify no sensitive data in logs
- [ ] Confirm environment variables are secure (not in code)

---

## 🚨 Troubleshooting

### Backend Issues

**Deployment fails:**
- Check Railway logs for error messages
- Verify Dockerfile.backend builds locally
- Ensure all dependencies in requirements.txt

**500 errors:**
- Check environment variables are set
- View Railway logs for stack traces
- Verify API keys are valid

**Database errors:**
- Confirm DATABASE_URL is set correctly
- For Railway: Use SQLite path `sqlite:///./app.db`
- Check file permissions

### Frontend Issues

**Build fails:**
- Check Vercel build logs
- Verify package.json is correct
- Test build locally: `cd frontend-next && npm run build`

**API errors:**
- Verify NEXT_PUBLIC_API_BASE_URL is correct
- Check CORS configuration in backend
- Inspect network tab in browser dev tools

**Environment variables not working:**
- Must start with NEXT_PUBLIC_ for client-side
- Redeploy after changing env vars
- Clear cache and hard refresh

---

## 📞 Getting Help

If stuck:

1. Check logs:
   - Railway: Dashboard → Deployments → Logs
   - Vercel: Dashboard → Deployments → Build Logs

2. Review documentation:
   - [FREE_DEPLOYMENT_GUIDE.md](./FREE_DEPLOYMENT_GUIDE.md)
   - [DEPLOYMENT_GUIDE.md](./DEPLOYMENT_GUIDE.md)

3. Common issues documented in guides above

---

## ✨ Success!

Once all checkboxes are complete:

✅ Backend deployed and healthy
✅ Frontend deployed and accessible
✅ Integration working
✅ Monitoring set up

**Your app is live at:**
- Frontend: `https://your-app.vercel.app`
- Backend: `https://your-api.up.railway.app`

Share your deployment URLs and enjoy! 🎉
