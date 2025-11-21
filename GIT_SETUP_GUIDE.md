# Git Repository Setup & Push Guide

## 📋 Pre-Push Checklist

### ✅ Files to Include in Repository

```
pricing-model-api/
├── .gitignore                 ✅ Created
├── README.md                  ✅ Created
│
├── deployment/
│   ├── app.py                 ✅ Ready
│   ├── requirements.txt       ✅ Ready
│   ├── Dockerfile            ✅ Ready
│   ├── docker-compose.yml    ✅ Ready
│   ├── test_api.py           ✅ Ready
│   ├── Jenkinsfile           ✅ Created
│   └── deployment_report.md  ✅ Ready
│
├── models/
│   ├── pricing_gradient_boosting_v1.pkl  ✅ Ready (check size)
│   └── model_metrics_v1.csv              ✅ Ready
│
└── data/
    └── used_or_will_be_used/
        ├── all_listings_clean.csv   ✅ Ready
        └── houses_data_eng.csv      ✅ Ready
```

### ⚠️ Files to EXCLUDE

- `__pycache__/` (Python cache)
- `.ipynb_checkpoints/` (Jupyter checkpoints)
- `.venv/`, `venv/`, `env/` (virtual environments)
- `*.log` (log files)
- `.DS_Store` (macOS files)
- Raw/temporary data files

---

## 🚀 Step-by-Step Push Instructions

### Step 1: Initialize Local Git Repository

```bash
cd /home/medgm/vsc/dApp-Ai

# Initialize git (if not already initialized)
git init

# Check current status
git status
```

### Step 2: Add Remote Repository

```bash
# Add remote repository
git remote add origin git@github.com:DApp-for-Real-Estate-Rental-on-Ethereum/pricing-model-api.git

# Verify remote is added
git remote -v
```

### Step 3: Stage Files for Commit

```bash
# Add .gitignore first
git add .gitignore

# Add README
git add README.md

# Add deployment files
git add deployment/app.py
git add deployment/requirements.txt
git add deployment/Dockerfile
git add deployment/docker-compose.yml
git add deployment/test_api.py
git add deployment/Jenkinsfile
git add deployment/deployment_report.md

# Add model files (check size first!)
ls -lh models/pricing_gradient_boosting_v1.pkl

# If model < 100MB, add it
git add models/pricing_gradient_boosting_v1.pkl
git add models/model_metrics_v1.csv

# Add data files
git add data/used_or_will_be_used/all_listings_clean.csv
git add data/used_or_will_be_used/houses_data_eng.csv

# Verify what's staged
git status
```

### Step 4: Create Initial Commit

```bash
# Commit with meaningful message
git commit -m "feat: Initial commit - Morocco Airbnb Dynamic Pricing API

- FastAPI microservice for price predictions
- GradientBoosting model (5.73% MAPE)
- Docker containerization with docker-compose
- Comprehensive test suite with pytest
- Jenkins CI/CD pipeline
- City-specific insights for 6 Moroccan markets
- Complete deployment documentation
- Training data (3,963 clean listings)
- Model artifacts and metrics"
```

### Step 5: Create Main Branch (if needed)

```bash
# Rename master to main (modern convention)
git branch -M main
```

### Step 6: Push to GitHub

```bash
# Push to remote repository
git push -u origin main

# If repository is not empty and you get errors:
# Option A: Force push (⚠️ USE WITH CAUTION - destroys remote history)
git push -u origin main --force

# Option B: Pull first, then merge
git pull origin main --allow-unrelated-histories
git push -u origin main
```

---

## 🔍 Troubleshooting

### Issue 1: Large File Error (>100MB)

**Error**: "remote: error: File ... is XXX MB; this exceeds GitHub's file size limit of 100 MB"

**Solution**: Use Git LFS for large model files

```bash
# Install Git LFS
git lfs install

# Track large files
git lfs track "models/*.pkl"

# Add .gitattributes
git add .gitattributes

# Commit and push
git commit -m "chore: Add Git LFS for large model files"
git push -u origin main
```

### Issue 2: Authentication Failed

**Error**: "Permission denied (publickey)"

**Solution**: Set up SSH key

```bash
# Generate SSH key (if not exists)
ssh-keygen -t ed25519 -C "your_email@example.com"

# Add SSH key to ssh-agent
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519

# Copy public key
cat ~/.ssh/id_ed25519.pub

# Add to GitHub: Settings → SSH and GPG keys → New SSH key
```

### Issue 3: Repository Not Empty

**Error**: "Updates were rejected because the remote contains work"

**Solution**: Pull and merge

```bash
git pull origin main --allow-unrelated-histories
git commit -m "merge: Merge remote changes"
git push -u origin main
```

---

## 📦 Optional: Create Release Tag

```bash
# Create and push a version tag
git tag -a v1.0.0 -m "Release v1.0.0: Production-ready pricing API"
git push origin v1.0.0
```

---

## 🔐 Setting Up Jenkins

### 1. Create Jenkins Pipeline Job

1. **Open Jenkins**: http://your-jenkins-server:8080
2. **New Item** → Enter name: `morocco-pricing-api` → **Pipeline** → OK

### 2. Configure Git Repository

**Pipeline Configuration:**
- Definition: `Pipeline script from SCM`
- SCM: `Git`
- Repository URL: `git@github.com:DApp-for-Real-Estate-Rental-on-Ethereum/pricing-model-api.git`
- Credentials: Add SSH key or GitHub token
- Branch: `*/main`
- Script Path: `deployment/Jenkinsfile`

### 3. Add Required Credentials

**Navigate**: Manage Jenkins → Manage Credentials → Global

**Add Credentials:**

1. **Docker Hub** (ID: `docker-hub-credentials`)
   - Kind: Username with password
   - Username: `your-dockerhub-username`
   - Password: `your-dockerhub-token`
   - ID: `docker-hub-credentials`

2. **GitHub SSH** (if needed)
   - Kind: SSH Username with private key
   - Username: `git`
   - Private Key: Paste your SSH private key

### 4. Configure Webhooks (Optional)

**GitHub Repository Settings:**
1. Go to Settings → Webhooks → Add webhook
2. Payload URL: `http://your-jenkins-server:8080/github-webhook/`
3. Content type: `application/json`
4. Events: `Just the push event`
5. Active: ✅

### 5. Install Required Jenkins Plugins

**Required Plugins:**
- Git Plugin
- Pipeline Plugin
- Docker Plugin
- Docker Pipeline Plugin
- JUnit Plugin
- HTML Publisher Plugin (for coverage reports)

**Install**: Manage Jenkins → Manage Plugins → Available

---

## 🎯 Verification Checklist

After pushing to GitHub, verify:

- [ ] All files are visible on GitHub
- [ ] .gitignore is working (no `__pycache__`, `.venv`, etc.)
- [ ] README.md renders correctly
- [ ] Model file size is acceptable (<100MB or using LFS)
- [ ] Jenkins can access the repository
- [ ] Jenkinsfile is recognized by Jenkins
- [ ] Docker Hub credentials are configured
- [ ] First build runs successfully

---

## 📝 Commit Message Conventions

Use conventional commits for better changelog generation:

```bash
# Feature
git commit -m "feat: Add batch prediction endpoint"

# Bug fix
git commit -m "fix: Correct price calculation for Marrakech"

# Documentation
git commit -m "docs: Update API usage examples"

# Performance
git commit -m "perf: Optimize model loading time"

# Refactor
git commit -m "refactor: Simplify feature preparation logic"

# Test
git commit -m "test: Add integration tests for city insights"

# CI/CD
git commit -m "ci: Add Docker image caching to pipeline"
```

---

## 🚀 Next Steps After Push

1. **Verify GitHub Repository**: Check all files are present
2. **Set Up Jenkins Job**: Configure pipeline as described above
3. **Run First Build**: Trigger manual build to test pipeline
4. **Monitor Build**: Check console output for any errors
5. **Deploy to Staging**: If build passes, deploy to staging environment
6. **Run Integration Tests**: Verify API functionality
7. **Deploy to Production**: After testing, promote to production

---

## 📞 Support

If you encounter issues:

1. Check `.gitignore` is excluding unnecessary files
2. Verify SSH keys are set up correctly
3. Ensure model file size is within limits
4. Check Jenkins logs for detailed error messages
5. Review Jenkinsfile syntax

**Common Commands:**

```bash
# Check what will be committed
git status

# See ignored files
git status --ignored

# Remove file from staging
git reset HEAD <file>

# Undo last commit (keep changes)
git reset --soft HEAD~1

# View commit history
git log --oneline --graph
```

---

**Ready to push? Run the commands in Steps 1-6 above!** 🚀
