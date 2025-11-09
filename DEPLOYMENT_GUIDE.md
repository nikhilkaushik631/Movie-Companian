# 🚀 Movie Companion Deployment Guide

This guide provides multiple deployment options for the Movie Companion application.

## 📋 Prerequisites

- Docker & Docker Compose (for local/VPS deployment)
- API Keys for external services:
  - Google/Gemini API keys
  - Groq API key  
  - TMDB API key
  - OMDB API key
  - Pinecone API key
  - Deepgram API key (optional)

## 🛠️ Quick Local Deployment

### Option 1: Docker Compose (Recommended)

```bash
# 1. Clone and navigate to project
git clone <your-repo-url>
cd Movie-Companian

# 2. Copy environment template
cp .env.template .env

# 3. Update .env with your API keys
nano .env

# 4. Deploy with one command
./deploy.sh

# 5. Access the application
# Frontend: http://localhost:3000
# Backend:  http://localhost:8000
# API Docs: http://localhost:8000/docs
```

### Option 2: Manual Development Setup

```bash
# Backend
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
pip install -r requirements.txt
python main.py

# Frontend (in new terminal)
cd frontend-next
npm install
npm run dev
```

## ☁️ Cloud Deployment Options

### 1. Railway (Backend) + Vercel (Frontend) - **Recommended for Production**

#### Backend on Railway:
1. Push code to GitHub
2. Connect Railway to your repo
3. Deploy using `railway.json` config
4. Add environment variables in Railway dashboard
5. Copy the Railway backend URL

#### Frontend on Vercel:
1. Connect Vercel to your GitHub repo
2. Set root directory to `frontend-next`
3. Add environment variable: `NEXT_PUBLIC_API_BASE_URL=https://your-railway-url.com`
4. Deploy automatically

### 2. Render.com (Full Stack)

1. Push code to GitHub
2. Connect Render to your repo
3. Create service using `render.yaml`
4. Add environment variables
5. Deploy automatically

### 3. Google Cloud Platform

```bash
# Using Cloud Run
gcloud run deploy movie-companion-backend \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated

gcloud run deploy movie-companion-frontend \
  --source ./frontend-next \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

### 4. AWS ECS with Fargate

```bash
# Build and push to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com
docker build -f Dockerfile.backend -t movie-companion-backend .
docker tag movie-companion-backend:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/movie-companion-backend:latest
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/movie-companion-backend:latest
```

### 5. DigitalOcean App Platform

```yaml
# app.yaml
name: movie-companion
services:
  - name: backend
    source_dir: /
    github:
      repo: your-username/movie-companion
      branch: main
    run_command: uvicorn main:app --host 0.0.0.0 --port $PORT
    environment_slug: python
    instance_count: 1
    instance_size_slug: basic-xxs
    
  - name: frontend
    source_dir: /frontend-next
    github:
      repo: your-username/movie-companion
      branch: main
    run_command: npm start
    environment_slug: node-js
    instance_count: 1
    instance_size_slug: basic-xxs
```

## 🔧 Environment Variables

Create a `.env` file with the following variables:

```bash
# Required API Keys
GOOGLE_API_KEY=your_google_api_key
GEMINI_API_KEY=your_gemini_api_key  
GROQ_API_KEY=your_groq_api_key
TMDB_API_KEY=your_tmdb_api_key
OMDB_API_KEY=your_omdb_api_key
PINECONE_API_KEY=your_pinecone_api_key

# Optional
DEEPGRAM_API_KEY=your_deepgram_api_key
HUGGINGFACE_API_TOKEN=your_hf_token

# Database
DATABASE_URL=sqlite:///./app.db

# Security
AUTH_SECRET_KEY=your_jwt_secret_key

# Paths (for local development)
PLOTS_PATH=path/to/plots.csv
DESCRIPTION_PATH=path/to/descriptions.pickle  
METADATA_PATH=path/to/metadata.cache
```

## 📊 Performance Optimizations Included

The deployed application includes:

- **Database Connection Pooling** (20 connections, 30 overflow)
- **LRU Caching** with TTL (prevents memory leaks)
- **API Rate Limiting** (prevents API exhaustion)
- **Response Caching** (5-minute TTL for external APIs)
- **Async Operations** (non-blocking I/O)
- **Load Balancing** across multiple LLM APIs

## 🔍 Monitoring & Health Checks

### Health Check Endpoints:
- Backend: `http://your-backend-url/health`
- Voice API: `http://your-backend-url/voice/health`

### Logs:
```bash
# Docker Compose
./deploy.sh development logs

# Individual services  
docker-compose logs -f backend
docker-compose logs -f frontend
```

## 🔒 Security Considerations

1. **Environment Variables**: Never commit API keys to version control
2. **HTTPS**: Use SSL certificates in production (included in nginx config)
3. **Rate Limiting**: Configured in Nginx (10 req/sec for API, 50 for static)
4. **CORS**: Properly configured for your domain
5. **Headers**: Security headers included (XSS, CSRF protection)

## 🚨 Troubleshooting

### Common Issues:

1. **API Key Errors**:
   ```bash
   # Check environment variables
   docker-compose exec backend env | grep API_KEY
   ```

2. **Database Connection**:
   ```bash
   # Check database file permissions
   ls -la app.db
   ```

3. **Frontend Build Issues**:
   ```bash
   # Clear Next.js cache
   cd frontend-next && rm -rf .next
   npm run build
   ```

4. **Memory Issues**:
   ```bash
   # Monitor container memory usage
   docker stats
   ```

### Performance Monitoring:

```bash
# Backend performance stats
curl http://localhost:8000/chat -X POST \
  -H "Content-Type: application/json" \
  -d '{"message": "stats"}'
```

## 📈 Scaling Considerations

- **Horizontal Scaling**: Use multiple backend instances behind load balancer
- **Database**: Migrate from SQLite to PostgreSQL for production
- **Caching**: Add Redis for distributed caching
- **CDN**: Use CloudFlare or AWS CloudFront for static assets

## 🎯 Deployment Checklist

- [ ] API keys configured
- [ ] Environment variables set
- [ ] Database initialized
- [ ] Health checks passing
- [ ] SSL certificates configured (production)
- [ ] Domain DNS configured
- [ ] Monitoring setup
- [ ] Backup strategy implemented

## 📞 Support

For deployment issues:
1. Check logs first: `./deploy.sh [env] logs`
2. Verify environment variables
3. Test individual components
4. Check external API connectivity

---

🎬 **Your Movie Companion is ready to help users discover amazing movies and TV shows!**