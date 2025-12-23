# Deployment Guide

This guide covers deploying the Smartbox Anomaly Detection system to production environments.

---

## Quick Start

### Prerequisites

- Docker 20.10+ and Docker Compose 2.0+
- Access to VictoriaMetrics endpoint
- 2GB RAM minimum, 4GB recommended
- 10GB disk space for models and data

### Deploy with Docker Compose

```bash
# Clone and navigate to the project
cd smartbox-anomaly-detection

# Configure your environment
cp config.json config.production.json
# Edit config.production.json with your VictoriaMetrics endpoint

# Build the image
docker compose build

# Start the service
docker compose up -d

# View logs
docker compose logs -f
```

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    Docker Container                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐  │
│  │   Cron      │  │  Training   │  │   Inference     │  │
│  │  Scheduler  │──│  (main.py)  │──│  (inference.py) │  │
│  └─────────────┘  └─────────────┘  └─────────────────┘  │
│         │                │                   │           │
│         ▼                ▼                   ▼           │
│  ┌─────────────────────────────────────────────────┐    │
│  │              Shared Volumes                      │    │
│  │  ./smartbox_models  ./data  ./logs              │    │
│  └─────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
         │                                     │
         ▼                                     ▼
┌─────────────────┐                 ┌─────────────────────┐
│ VictoriaMetrics │                 │  Observability API  │
│   (metrics)     │                 │  (anomaly reports)  │
└─────────────────┘                 └─────────────────────┘
```

---

## Docker Deployment

### Building the Image

```bash
# Standard build
docker compose build

# Build for specific platform (e.g., from Mac M1/M2 for Linux server)
docker buildx build --platform linux/amd64 -t smartbox-anomaly:latest .
```

### docker-compose.yml Configuration

```yaml
version: "3.8"

services:
  yaga:
    build:
      context: .
      dockerfile: Dockerfile
      target: production
      platforms:
        - linux/amd64
    platform: linux/amd64
    container_name: smartbox-anomaly
    restart: unless-stopped

    environment:
      - TZ=UTC
      - CONFIG_PATH=/app/config.json
      # Schedule configuration (cron syntax)
      - TRAIN_SCHEDULE=0 2 * * *         # Daily at 2 AM
      - INFERENCE_SCHEDULE=*/10 * * * *  # Every 10 minutes

    volumes:
      # Trained models (persistent)
      - ./smartbox_models:/app/smartbox_models
      # SQLite database for state (persistent)
      - ./data:/app/data
      # Logs
      - ./logs:/app/logs

    healthcheck:
      test: ["CMD", "python", "-c", "import smartbox_anomaly; print('OK')"]
      interval: 60s
      timeout: 10s
      retries: 3
      start_period: 30s
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TZ` | `UTC` | Timezone for scheduling |
| `CONFIG_PATH` | `/app/config.json` | Path to configuration file |
| `TRAIN_SCHEDULE` | `0 2 * * *` | Cron schedule for training (daily 2 AM) |
| `INFERENCE_SCHEDULE` | `*/10 * * * *` | Cron schedule for inference (every 10 min) |
| `VM_ENDPOINT` | from config | VictoriaMetrics URL |
| `OBSERVABILITY_URL` | from config | Observability API URL |
| `OBSERVABILITY_ENABLED` | `true` | Enable/disable API reporting |

### Running Commands

```bash
# Start with scheduling (default)
docker compose up -d

# Run training manually
docker compose run --rm yaga train

# Run inference manually
docker compose run --rm yaga inference

# Run inference with verbose output
docker compose run --rm yaga inference --verbose

# Run both once (for testing)
docker compose run --rm yaga once

# Get a shell inside the container
docker compose run --rm yaga shell

# View scheduled job logs
docker compose exec yaga tail -f /app/logs/inference.log
```

---

## Persistent Storage

### Volume Mounts

| Host Path | Container Path | Purpose |
|-----------|----------------|---------|
| `./smartbox_models` | `/app/smartbox_models` | Trained ML models |
| `./data` | `/app/data` | SQLite state database |
| `./logs` | `/app/logs` | Application logs |

### Backup Strategy

```bash
# Backup models and state
tar -czvf smartbox-backup-$(date +%Y%m%d).tar.gz \
    smartbox_models/ \
    data/

# Restore from backup
tar -xzvf smartbox-backup-20240115.tar.gz
```

### Storage Requirements

| Component | Size Estimate |
|-----------|---------------|
| Models (per service) | 5-15 MB |
| SQLite database | 50-500 MB (depends on incident volume) |
| Logs (with rotation) | 100 MB max |
| **Total (20 services)** | ~500 MB - 1 GB |

---

## Scheduling Configuration

### Default Schedules

- **Training**: Daily at 2 AM (`0 2 * * *`)
- **Inference**: Every 10 minutes (`*/10 * * * *`)

### Customizing Schedules

Edit `docker-compose.yml`:

```yaml
environment:
  # Run inference every 2 minutes
  - INFERENCE_SCHEDULE=*/2 * * * *

  # Train weekly on Sunday at 3 AM
  - TRAIN_SCHEDULE=0 3 * * 0
```

### Schedule Recommendations

| Scenario | Inference | Training |
|----------|-----------|----------|
| High-traffic production | Every 2-5 min | Daily |
| Standard monitoring | Every 10 min | Daily |
| Cost-sensitive | Every 15-30 min | Weekly |
| Development/testing | Manual | Manual |

---

## Resource Requirements

### Minimum Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| CPU | 1 core | 2 cores |
| RAM | 2 GB | 4 GB |
| Disk | 5 GB | 20 GB |

### Resource Usage by Operation

| Operation | CPU | RAM | Duration |
|-----------|-----|-----|----------|
| Training (per service) | High | 500 MB - 1 GB | 2-10 min |
| Inference (per service) | Low | 100-200 MB | 5-30 sec |
| Idle | Minimal | 50 MB | - |

### Scaling Guidelines

| Services Monitored | RAM | CPU |
|-------------------|-----|-----|
| 1-10 | 2 GB | 1 core |
| 10-30 | 4 GB | 2 cores |
| 30-50 | 6 GB | 2-4 cores |
| 50+ | 8+ GB | 4+ cores |

---

## Kubernetes Deployment

### Basic Deployment

```yaml
# kubernetes/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: smartbox-anomaly
  labels:
    app: smartbox-anomaly
spec:
  replicas: 1
  selector:
    matchLabels:
      app: smartbox-anomaly
  template:
    metadata:
      labels:
        app: smartbox-anomaly
    spec:
      containers:
      - name: smartbox
        image: smartbox-anomaly:latest
        command: ["docker-entrypoint.sh", "scheduler"]
        env:
        - name: TZ
          value: "UTC"
        - name: CONFIG_PATH
          value: "/app/config.json"
        - name: TRAIN_SCHEDULE
          value: "0 2 * * *"
        - name: INFERENCE_SCHEDULE
          value: "*/10 * * * *"
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        volumeMounts:
        - name: models
          mountPath: /app/smartbox_models
        - name: data
          mountPath: /app/data
        - name: config
          mountPath: /app/config.json
          subPath: config.json
        livenessProbe:
          exec:
            command: ["python", "-c", "import smartbox_anomaly"]
          initialDelaySeconds: 30
          periodSeconds: 60
        readinessProbe:
          exec:
            command: ["python", "-c", "import smartbox_anomaly"]
          initialDelaySeconds: 10
          periodSeconds: 30
      volumes:
      - name: models
        persistentVolumeClaim:
          claimName: smartbox-models-pvc
      - name: data
        persistentVolumeClaim:
          claimName: smartbox-data-pvc
      - name: config
        configMap:
          name: smartbox-config
```

### PersistentVolumeClaim

```yaml
# kubernetes/pvc.yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: smartbox-models-pvc
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 5Gi
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: smartbox-data-pvc
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 1Gi
```

### ConfigMap

```yaml
# kubernetes/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: smartbox-config
data:
  config.json: |
    {
      "victoria_metrics": {
        "endpoint": "http://victoria-metrics:8428"
      },
      "observability_api": {
        "base_url": "http://observability-api:8000",
        "enabled": true
      },
      "services": {
        "critical": ["booking", "search"],
        "standard": ["api-gateway"]
      }
    }
```

### CronJob for Training (Alternative)

If you prefer Kubernetes-native scheduling:

```yaml
# kubernetes/cronjob-train.yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: smartbox-train
spec:
  schedule: "0 2 * * *"
  concurrencyPolicy: Forbid
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: train
            image: smartbox-anomaly:latest
            command: ["python", "main.py"]
            volumeMounts:
            - name: models
              mountPath: /app/smartbox_models
            - name: config
              mountPath: /app/config.json
              subPath: config.json
          restartPolicy: OnFailure
          volumes:
          - name: models
            persistentVolumeClaim:
              claimName: smartbox-models-pvc
          - name: config
            configMap:
              name: smartbox-config
```

---

## Network Configuration

### Required Outbound Access

| Destination | Port | Purpose |
|-------------|------|---------|
| VictoriaMetrics | 8428 (typical) | Fetch metrics |
| Observability API | 8000 (typical) | Report anomalies |

### Firewall Rules

```bash
# Allow outbound to VictoriaMetrics
iptables -A OUTPUT -p tcp --dport 8428 -j ACCEPT

# Allow outbound to Observability API
iptables -A OUTPUT -p tcp --dport 8000 -j ACCEPT
```

---

## High Availability

### Single Instance (Recommended for Most Cases)

The system is designed to run as a **single instance**:
- SQLite database doesn't support concurrent writes
- Models are trained on the same schedule
- Inference is fast enough for single-threaded execution

### Horizontal Scaling (Advanced)

For very large deployments (100+ services), consider:

1. **Partition by service group**: Run separate instances for different service categories
2. **Separate training and inference**: Use Kubernetes CronJobs for training, Deployment for inference
3. **Shared storage**: Use network-attached storage (NFS, EFS) for model sharing

```yaml
# Example: Partitioned deployment
# Instance 1: Critical services
- SERVICES_FILTER=critical
# Instance 2: Standard services
- SERVICES_FILTER=standard
```

---

## Monitoring the Deployment

### Health Checks

```bash
# Check container health
docker compose ps

# Check if models exist
docker compose exec yaga ls -la /app/smartbox_models/

# Check recent inference runs
docker compose exec yaga tail -20 /app/logs/inference.log

# Check cron status
docker compose exec yaga crontab -l
```

### Log Locations

| Log | Path | Content |
|-----|------|---------|
| Training | `/app/logs/train.log` | Model training output |
| Inference | `/app/logs/inference.log` | Detection results |
| Container | `docker compose logs` | Combined stdout/stderr |

### Metrics to Monitor

| Metric | Alert Threshold | Description |
|--------|-----------------|-------------|
| Container restarts | > 3 in 1 hour | Crash loop |
| Inference duration | > 5 min | Performance issue |
| Training failures | Any | Model staleness risk |
| Disk usage | > 80% | Storage capacity |

---

## Upgrading

### Standard Upgrade

```bash
# Pull latest code
git pull

# Rebuild image
docker compose build

# Restart with new image
docker compose up -d
```

### Zero-Downtime Upgrade (Kubernetes)

```bash
# Update image
kubectl set image deployment/smartbox-anomaly \
  smartbox=smartbox-anomaly:v2.0.1

# Monitor rollout
kubectl rollout status deployment/smartbox-anomaly
```

### Model Compatibility

Models are forward-compatible within minor versions. After major version upgrades:

```bash
# Retrain all models after major upgrade
docker compose run --rm yaga train
```

---

## Troubleshooting Deployment

### Container Won't Start

```bash
# Check build logs
docker compose build --no-cache

# Check startup logs
docker compose logs smartbox

# Verify config file
docker compose run --rm yaga python -c "from smartbox_anomaly.core.config import get_config; print(get_config())"
```

### Cron Jobs Not Running

```bash
# Check cron is running
docker compose exec yaga ps aux | grep cron

# Check cron configuration
docker compose exec yaga cat /etc/cron.d/smartbox

# Check cron logs
docker compose exec yaga tail -f /app/logs/*.log
```

### Out of Memory

```yaml
# Increase memory limit in docker-compose.yml
deploy:
  resources:
    limits:
      memory: 4G
```

### Permission Issues

```bash
# Fix volume permissions
sudo chown -R 1000:1000 ./smartbox_models ./data ./logs
```

---

## Security Considerations

### Non-Root User

The container runs as non-root user `smartbox` (UID 1000) for security.

### Secrets Management

Avoid storing secrets in config files. Use environment variables:

```yaml
environment:
  - VM_AUTH_TOKEN=${VM_AUTH_TOKEN}
```

### Network Isolation

Consider running in an isolated network:

```yaml
networks:
  internal:
    driver: bridge
    internal: true
```

---

## Further Reading

- [CONFIGURATION.md](./CONFIGURATION.md) - All configuration options
- [OPERATIONS.md](./OPERATIONS.md) - Monitoring and troubleshooting
- [MACHINE_LEARNING.md](./MACHINE_LEARNING.md) - How the ML models work
