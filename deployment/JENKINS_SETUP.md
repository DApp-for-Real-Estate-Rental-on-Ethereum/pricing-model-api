# Jenkins CI/CD Setup Guide
## Morocco Airbnb Dynamic Pricing API

This guide explains how to set up the Jenkins pipeline for the pricing API, following your existing Jenkins configuration patterns.

---

## 📋 Prerequisites

✅ Jenkins server running and accessible  
✅ Docker installed on Jenkins agent  
✅ SonarQube server running on `localhost:9000` (optional)  
✅ Local Docker registry on `localhost:5000` (optional)  
✅ Docker Hub account (for pushing images)  

---

## 🔧 Jenkins Configuration

### Step 1: Install Required Jenkins Plugins

Navigate to: **Manage Jenkins** → **Manage Plugins** → **Available**

Install these plugins:
- ✅ **Git Plugin** - For Git repository integration
- ✅ **Pipeline Plugin** - For Pipeline support
- ✅ **Docker Pipeline Plugin** - For Docker operations
- ✅ **JUnit Plugin** - For test result publishing
- ✅ **HTML Publisher Plugin** - For security scan reports
- ✅ **Credentials Plugin** - For managing secrets

### Step 2: Configure Credentials

Navigate to: **Manage Jenkins** → **Manage Credentials** → **Global**

#### 2.1 Docker Registry Credentials
- **Kind**: Username with password
- **ID**: `docker-registry-creds` (must match exactly)
- **Username**: Your Docker Hub username (e.g., `medgm`)
- **Password**: Your Docker Hub access token
- **Description**: Docker Hub credentials

#### 2.2 SonarQube Token (Optional)
- **Kind**: Secret text
- **ID**: `sonarqube-token` (must match exactly)
- **Secret**: Your SonarQube authentication token
- **Description**: SonarQube authentication token

### Step 3: Create Jenkins Pipeline Job

1. **Click**: New Item
2. **Enter name**: `morocco-pricing-api`
3. **Select**: Pipeline
4. **Click**: OK

### Step 4: Configure Pipeline

**General Settings:**
- ✅ Description: `Morocco Airbnb Dynamic Pricing API - ML Model Deployment`
- ✅ Discard old builds: Keep last 10 builds
- ✅ GitHub project: `https://github.com/DApp-for-Real-Estate-Rental-on-Ethereum/pricing-model-api`

**Build Triggers** (Optional):
- ✅ GitHub hook trigger for GITScm polling (if webhooks configured)
- ✅ Poll SCM: `H/5 * * * *` (every 5 minutes)

**Pipeline Definition:**
- **Definition**: `Pipeline script from SCM`
- **SCM**: `Git`
- **Repository URL**: `git@github.com:DApp-for-Real-Estate-Rental-on-Ethereum/pricing-model-api.git`
- **Credentials**: Select your GitHub SSH key
- **Branch Specifier**: `*/main`
- **Script Path**: `deployment/Jenkinsfile`

**Click**: Save

---

## 🚀 Pipeline Stages Overview

### Stage 1: Checkout
- Checks out code from Git repository
- Gets current commit hash

### Stage 2: Verify Project Layout
- Validates project structure
- Checks for required files:
  - `deployment/requirements.txt`
  - `models/pricing_gradient_boosting_v1.pkl`
- Cleans up previous build artifacts

### Stage 3: Build & Test
- Creates Python Docker container
- Installs dependencies from `requirements.txt`
- Runs pytest test suite
- Generates JUnit test results

### Stage 4: Publish Test Results
- Publishes test results to Jenkins
- Makes test reports available in build

### Stage 5: Code Quality Analysis
- Runs SonarQube analysis (if configured)
- Analyzes Python code quality
- Generates quality metrics

### Stage 6: Security Scan
- Runs `safety` to check for vulnerable dependencies
- Runs `bandit` for Python security linting
- Generates HTML security report

### Stage 7: Model Validation
- Validates ML model can be loaded
- Checks model file integrity
- Verifies predict method exists

### Stage 8: Build Docker Image
- Builds Docker image from `deployment/Dockerfile`
- Tags image with:
  - Build number: `medgm/morocco-pricing-api:${BUILD_NUMBER}`
  - Latest: `medgm/morocco-pricing-api:latest`
  - Git commit: `medgm/morocco-pricing-api:${GIT_COMMIT_SHORT}`

### Stage 9: Test Docker Container
- Runs container on port 8888
- Tests health endpoint: `/health`
- Tests prediction endpoint: `/predict`
- Validates API is working

### Stage 10: Push to Docker Hub
- Pushes all tagged images to Docker Hub
- Uses credentials from Jenkins

### Stage 11: Deploy to Local Registry
- Tags images for local registry
- Pushes to `localhost:5000`
- Makes images available locally

### Stage 12: Integration Tests
- Placeholder for future integration tests

---

## 📊 Expected Pipeline Duration

| Stage | Typical Duration |
|-------|-----------------|
| Checkout | 5-10 seconds |
| Verify Project Layout | 2-5 seconds |
| Build & Test | 30-60 seconds |
| Publish Test Results | 2-5 seconds |
| Code Quality Analysis | 20-40 seconds |
| Security Scan | 30-60 seconds |
| Model Validation | 15-30 seconds |
| Build Docker Image | 60-120 seconds |
| Test Docker Container | 15-30 seconds |
| Push to Docker Hub | 30-60 seconds |
| Deploy to Local Registry | 20-40 seconds |
| Integration Tests | 2-5 seconds |
| **Total** | **~5-8 minutes** |

---

## 🧪 Running Your First Build

### Manual Build:
1. Navigate to the pipeline job
2. Click **Build Now**
3. Watch the build progress in **Console Output**

### Expected Output:
```
Started by user admin
Running in Durability level: MAX_SURVIVABILITY
[Pipeline] Start of Pipeline
[Pipeline] node
Running on Jenkins in /var/jenkins_home/workspace/morocco-pricing-api
[Pipeline] {
[Pipeline] stage
[Pipeline] { (Checkout)
[Pipeline] script
[Pipeline] {
[Pipeline] sh
+ git rev-parse --short HEAD
abc1234
[Pipeline] }
[Pipeline] // script
[Pipeline] }
[Pipeline] // stage
...
Morocco Pricing API pipeline completed successfully! 🎉
[Pipeline] End of Pipeline
Finished: SUCCESS
```

---

## 🔍 Troubleshooting

### Issue 1: Model File Not Found

**Error**: `ERROR: Model file not found!`

**Solution**:
```bash
# Check model file exists
ls -la models/pricing_gradient_boosting_v1.pkl

# If missing, ensure it's committed to Git
git add models/pricing_gradient_boosting_v1.pkl
git commit -m "Add model file"
git push
```

### Issue 2: Docker Permission Denied

**Error**: `permission denied while trying to connect to the Docker daemon`

**Solution**:
```bash
# Add Jenkins user to docker group
sudo usermod -aG docker jenkins

# Restart Jenkins
sudo systemctl restart jenkins
```

### Issue 3: Tests Failing

**Error**: `pytest execution failed`

**Solution**:
```bash
# Run tests locally to debug
cd deployment
pytest test_api.py -v

# Fix any issues, commit, and push
git add test_api.py
git commit -m "fix: Resolve test failures"
git push
```

### Issue 4: Docker Hub Push Failed

**Error**: `denied: requested access to the resource is denied`

**Solution**:
1. Verify Docker Hub credentials in Jenkins
2. Check credential ID is exactly: `docker-registry-creds`
3. Ensure Docker Hub token has push permissions
4. Test credentials manually:
```bash
echo "YOUR_TOKEN" | docker login -u medgm --password-stdin
```

### Issue 5: SonarQube Connection Failed

**Error**: `Failed to connect to SonarQube server`

**Solution**:
- Ensure SonarQube is running: `curl http://localhost:9000`
- Check SonarQube token is valid
- Stage will continue even if it fails (`|| true`)

---

## 📈 Monitoring Build Results

### View Test Results:
1. Go to build page
2. Click **Test Result**
3. See detailed pytest results

### View Security Scan:
1. Go to build page
2. Click **Python Security Scan Report**
3. Review vulnerability findings

### View Docker Images:
```bash
# Check local images
docker images | grep morocco-pricing-api

# Check Docker Hub
# Visit: https://hub.docker.com/r/medgm/morocco-pricing-api
```

---

## 🔄 Updating the Pipeline

### Modify Jenkinsfile:
```bash
# Edit Jenkinsfile
vi deployment/Jenkinsfile

# Commit changes
git add deployment/Jenkinsfile
git commit -m "ci: Update pipeline configuration"
git push
```

Jenkins will automatically use the updated Jenkinsfile on next build.

---

## 🎯 Pipeline Best Practices

### 1. Keep Builds Fast
- Use Docker layer caching
- Parallelize independent stages
- Cache dependencies when possible

### 2. Fail Fast
- Run quick tests first
- Validate project structure early
- Stop on critical errors

### 3. Clean Up Resources
- Remove test containers in `post` block
- Prune old Docker images
- Archive only necessary artifacts

### 4. Security First
- Scan dependencies regularly
- Never commit credentials
- Use Jenkins credentials store

### 5. Monitor and Alert
- Set up email notifications
- Track build trends
- Review security scan reports

---

## 📧 Notifications (Optional)

### Email Notifications:
Add to `post` block in Jenkinsfile:
```groovy
post {
    success {
        emailext (
            subject: "✅ Morocco Pricing API Build #${BUILD_NUMBER} Succeeded",
            body: "Build completed successfully!",
            to: "your-email@example.com"
        )
    }
    failure {
        emailext (
            subject: "❌ Morocco Pricing API Build #${BUILD_NUMBER} Failed",
            body: "Build failed. Check console output.",
            to: "your-email@example.com"
        )
    }
}
```

### Slack Notifications:
```groovy
post {
    success {
        slackSend(
            color: 'good',
            message: "✅ Build #${BUILD_NUMBER} succeeded - Morocco Pricing API"
        )
    }
}
```

---

## 🚀 Next Steps

After successful pipeline setup:

1. ✅ **Test the pipeline**: Run manual build
2. ✅ **Configure webhooks**: Auto-trigger on push
3. ✅ **Set up monitoring**: Track build metrics
4. ✅ **Deploy to staging**: Test in staging environment
5. ✅ **Add integration tests**: Expand test coverage
6. ✅ **Configure alerts**: Get notified of failures
7. ✅ **Document procedures**: Create runbooks

---

## 📚 Additional Resources

- **Jenkins Pipeline Syntax**: https://www.jenkins.io/doc/book/pipeline/syntax/
- **Docker Pipeline Plugin**: https://plugins.jenkins.io/docker-workflow/
- **JUnit Plugin**: https://plugins.jenkins.io/junit/
- **HTML Publisher**: https://plugins.jenkins.io/htmlpublisher/

---

**Pipeline is ready! Click "Build Now" to start your first deployment.** 🎉
