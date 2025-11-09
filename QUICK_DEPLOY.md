# ⚡ Quick Deploy - Cinemizer (5 Minutes)

**Fastest way to deploy Cinemizer using 100% FREE services**

---

## 🎯 What You'll Get

- ✅ Frontend on Vercel (Free)
- ✅ Backend on Railway (Free $5/month credit)
- ✅ Total Cost: $0
- ✅ Auto-deploy on git push

---

## 🚀 Step 1: Deploy Backend (2 min)

### Railway.app

1. **Sign up**: https://railway.app → Login with GitHub
2. **New Project** → **Deploy from GitHub repo** → Select `cinemizer`
3. **Add Environment Variables** (Settings → Variables):

```bash
GOOGLE_API_KEY=
GEMINI_API_KEY=
GROQ_API_KEY=
TMDB_API_KEY=
OMDB_API_KEY=
PINECONE_API_KEY=
DATABASE_URL=sqlite:///./app.db
AUTH_SECRET_KEY=
```

Generate secret: `python -c "import secrets; print(secrets.token_urlsafe(32))"`

4. **Copy your backend URL**: `https://cinemizer-production.up.railway.app`

---

## 🎨 Step 2: Deploy Frontend (2 min)

### Vercel

1. **Sign up**: https://vercel.com → Login with GitHub
2. **Add New** → **Project** → Import `cinemizer`
3. **Configure**:
   - Framework: Next.js
   - Root Directory: `frontend-next`
   - Build Command: `npm run build` (auto)

4. **Add Environment Variables**:

```bash
NEXT_PUBLIC_API_BASE_URL=https://YOUR-RAILWAY-URL.up.railway.app
NEXT_PUBLIC_DEMO=0
```

⚠️ Replace with your actual Railway URL!

5. **Deploy** → Wait 2 minutes

---

## ✅ Step 3: Verify (1 min)

1. **Backend**: Visit `https://your-railway-url/health`
   - Should show: `{"status": "healthy"}`

2. **Frontend**: Visit `https://your-vercel-url`
   - Homepage should load
   - Try login/signup

3. **Test**: Send a chat message!

---

## 🎉 Done!

Your app is live!

**Frontend**: `https://cinemizer.vercel.app`
**Backend**: `https://cinemizer-production.up.railway.app`

---

## 🔧 Get API Keys (All Free!)

- **Gemini**: https://aistudio.google.com/app/apikey
- **Groq**: https://console.groq.com
- **TMDB**: https://www.themoviedb.org/settings/api
- **OMDB**: http://www.omdbapi.com/apikey.aspx
- **Pinecone**: https://www.pinecone.io

---

## 📚 Full Guides

- **Detailed**: [FREE_DEPLOYMENT_GUIDE.md](./FREE_DEPLOYMENT_GUIDE.md)
- **Checklist**: [DEPLOYMENT_CHECKLIST.md](./DEPLOYMENT_CHECKLIST.md)

---

## 🐛 Issues?

**Backend won't start:**
- Check environment variables are set
- View Railway logs

**Frontend can't connect:**
- Verify `NEXT_PUBLIC_API_BASE_URL` is correct
- Redeploy Vercel after changing env vars

**API keys not working:**
- Ensure no extra spaces
- Check keys are valid in provider dashboards
