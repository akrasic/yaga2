# Operations Guide

This guide covers day-to-day operations, monitoring, troubleshooting, and maintenance of the Smartbox Anomaly Detection system.

---

## Daily Operations

### Health Check

```bash
# Quick health check
docker compose ps
docker compose exec yaga python -c "import smartbox_anomaly; print('OK')"

# Check last inference run
docker compose exec yaga tail -5 /app/logs/inference.log

# Check last training run
docker compose exec yaga tail -20 /app/logs/train.log
```

### Viewing Logs

```bash
# All container output
docker compose logs -f

# Just inference logs
docker compose exec yaga tail -f /app/logs/inference.log

# Just training logs
docker compose exec yaga tail -f /app/logs/train.log

# Last 100 lines with timestamps
docker compose exec yaga tail -100 /app/logs/inference.log
```

### Manual Runs

```bash
# Run inference manually (useful for testing)
docker compose run --rm yaga inference --verbose

# Run training manually (after adding new service)
docker compose run --rm yaga train

# Run both once
docker compose run --rm yaga once
```

---

## Monitoring

### Key Metrics to Track

| Metric | How to Check | Healthy Value |
|--------|--------------|---------------|
| Container status | `docker compose ps` | "Up" with healthy |
| Last inference | Check log timestamp | Within schedule interval |
| Last training | Check log timestamp | Within 24 hours |
| Model count | `ls smartbox_models/` | One dir per service |
| Disk usage | `du -sh smartbox_models/ data/` | < 80% capacity |
| Model drift | Check `drift_warning` in output | No severe drift (score < 5) |
| Validation warnings | Check `validation_warnings` in output | Minimal or none |

### Automated Monitoring Script

```bash
#!/bin/bash
# health-check.sh

# Check container is running
if ! docker compose ps | grep -q "Up"; then
    echo "CRITICAL: Container not running"
    exit 2
fi

# Check last inference (within last 15 minutes)
LAST_LOG=$(docker compose exec -T yaga stat -c %Y /app/logs/inference.log 2>/dev/null)
NOW=$(date +%s)
AGE=$((NOW - LAST_LOG))

if [ $AGE -gt 900 ]; then
    echo "WARNING: No inference in last 15 minutes"
    exit 1
fi

echo "OK: System healthy"
exit 0
```

### Log Rotation

Logs are automatically rotated by Docker's json-file driver:

```yaml
# In docker-compose.yml
logging:
  driver: "json-file"
  options:
    max-size: "10m"
    max-file: "3"
```

---

## Troubleshooting

### Common Issues

#### 1. No Anomalies Detected (When Expected)

**Symptoms**: Real incidents occurring but no alerts generated.

**Checks**:
```bash
# Verify models exist for the service
ls -la smartbox_models/<service-name>/

# Check if inference is running
docker compose exec yaga tail -20 /app/logs/inference.log

# Run inference manually with verbose output
docker compose run --rm yaga inference --verbose
```

**Common Causes**:
| Cause | Solution |
|-------|----------|
| No trained model | Run `docker compose run --rm yaga train` |
| Service not in config | Add to `config.json` services section |
| Contamination too high | Lower contamination for the service |
| Model stale | Retrain with fresh data |

#### 2. Model Drift Detected

**Symptoms**: `drift_warning` appearing in output, confidence scores reduced.

**Checks**:
```bash
# Check for drift warnings in recent output
docker compose exec yaga grep -r "drift_warning" /app/alerts/

# Check drift score in verbose output
docker compose run --rm yaga inference --verbose 2>&1 | grep -A5 "drift"
```

**Common Causes**:
| Cause | Solution |
|-------|----------|
| Seasonal traffic change | Retrain with recent data |
| Infrastructure change | Retrain models |
| New feature deployment | Expected - retrain after stabilization |
| Data quality issue | Check VictoriaMetrics data |

**Solutions**:

For **moderate drift** (score 3-5):
- Monitor for 24-48 hours to see if it stabilizes
- If persistent, schedule retraining

For **severe drift** (score > 5):
```bash
# Immediate retraining recommended
docker compose run --rm yaga train

# Or for a specific service
docker compose run --rm yaga train --service my-service
```

#### 3. Too Many False Positives

**Symptoms**: Alerts for normal behavior, alert fatigue.

**Checks**:
```bash
# Check contamination setting
grep -A5 "contamination" config.json

# Review recent alerts
docker compose exec yaga tail -100 /app/logs/inference.log | grep "ANOMAL"
```

**Solutions**:
| Cause | Solution |
|-------|----------|
| Contamination too low | Increase contamination (e.g., 0.03 → 0.06) |
| Recent behavior change | Retrain models |
| Wrong time period | Check timezone configuration |
| Noisy service | Move to `micro` category |

**Adjust contamination**:
```json
{
  "model": {
    "contamination_by_service": {
      "noisy-service": 0.08
    }
  }
}
```

#### 3. Training Fails

**Symptoms**: Models not updating, old timestamps on model files.

**Checks**:
```bash
# Check training logs
docker compose exec yaga cat /app/logs/train.log

# Check VictoriaMetrics connectivity
docker compose exec yaga curl -s "http://your-vm:8428/api/v1/status/buildinfo"

# Verify disk space
df -h
```

**Common Causes**:
| Cause | Solution |
|-------|----------|
| VictoriaMetrics unreachable | Check network, endpoint URL |
| Not enough historical data | Wait for 30 days of data collection |
| Disk full | Clean old models/logs, expand storage |
| Memory exhausted | Increase container memory limit |

#### 4. Container Keeps Restarting

**Symptoms**: `docker compose ps` shows restart count increasing.

**Checks**:
```bash
# Check container logs for errors
docker compose logs --tail 50

# Check exit code
docker compose ps -a
```

**Common Causes**:
| Cause | Solution |
|-------|----------|
| Config file missing | Ensure config.json exists |
| Invalid config | Validate JSON syntax |
| Permission issues | Fix ownership of mounted volumes |
| OOM killed | Increase memory limit |

#### 5. Fingerprinting Not Working

**Symptoms**: Duplicate alerts for same incident, no incident tracking.

**Checks**:
```bash
# Check database exists and is writable
docker compose exec yaga ls -la /app/data/

# Check database integrity
docker compose exec yaga sqlite3 /app/data/anomaly_state.db ".tables"
```

**Solutions**:
```bash
# Reset fingerprinting database (if corrupted)
docker compose exec yaga rm /app/data/anomaly_state.db
docker compose restart
```

#### 6. Input Validation Warnings

**Symptoms**: `validation_warnings` in output, metrics being capped or replaced.

**Checks**:
```bash
# Check for validation warnings
docker compose run --rm yaga inference --verbose 2>&1 | grep -A5 "validation"
```

**Common Causes**:
| Cause | Solution |
|-------|----------|
| VictoriaMetrics returning NaN | Check metric collection pipeline |
| Negative latencies | Investigate metric source |
| Extreme outliers in data | Check for data pipeline issues |

**Note**: Validation warnings are informational. The system will still run detection using sanitized values. However, frequent warnings may indicate data quality issues upstream.

#### 7. Wrong Time Period Detection

**Symptoms**: Business hours model used during night, or vice versa.

**Checks**:
```bash
# Check container timezone
docker compose exec yaga date

# Check time period in verbose output
docker compose run --rm yaga inference --verbose 2>&1 | grep "period"
```

**Solution**: Set timezone in docker-compose.yml:
```yaml
environment:
  - TZ=Europe/London
```

---

## Debugging

### Enable Verbose Logging

```bash
# Run inference with full debug output
docker compose run --rm yaga inference --verbose

# Enable debug logging in config
```

```json
{
  "logging": {
    "level": "DEBUG"
  }
}
```

### Inspect Model Details

```bash
# Check model metadata
docker compose exec yaga cat /app/smartbox_models/<service>/business_hours/metadata.json | python -m json.tool

# List all models
docker compose exec yaga find /app/smartbox_models -name "*.joblib" -ls
```

### Test Detection on Specific Service

```python
# Run interactively
docker compose run --rm yaga shell

# Inside container
python
>>> from inference import SmartboxMLInferencePipeline
>>> pipeline = SmartboxMLInferencePipeline(verbose=True)
>>> results = pipeline.run_inference(["booking"])
>>> print(results)
```

### Check VictoriaMetrics Queries

```bash
# Test metrics query
docker compose exec yaga curl -s \
  "http://your-vm:8428/api/v1/query?query=rate(requests_total[5m])" | python -m json.tool
```

---

## Maintenance Tasks

### Weekly Maintenance

1. **Review alert quality**
   - Check for recurring false positives
   - Identify services needing contamination adjustment

2. **Check disk usage**
   ```bash
   du -sh smartbox_models/ data/ logs/
   ```

3. **Verify all services have recent models**
   ```bash
   find smartbox_models/ -name "metadata.json" -exec grep -l "trained_at" {} \; | \
     xargs -I {} sh -c 'echo {} && grep trained_at {}'
   ```

### Monthly Maintenance

1. **Clean old incidents from database**
   ```bash
   # Incidents older than 30 days are auto-cleaned, but you can force:
   docker compose exec yaga sqlite3 /app/data/anomaly_state.db \
     "DELETE FROM incidents WHERE updated_at < datetime('now', '-30 days');"
   ```

2. **Archive old logs**
   ```bash
   tar -czvf logs-$(date +%Y%m).tar.gz logs/
   rm logs/*.log
   docker compose restart
   ```

3. **Review and update service categories**
   - New services may need to be added
   - Service criticality may have changed

### Backup Procedures

```bash
# Full backup
tar -czvf smartbox-backup-$(date +%Y%m%d).tar.gz \
  smartbox_models/ \
  data/ \
  config.json

# Restore
tar -xzvf smartbox-backup-20240115.tar.gz
docker compose restart
```

---

## Performance Tuning

### Inference Too Slow

| Symptom | Check | Solution |
|---------|-------|----------|
| > 30s per service | CPU usage | Increase CPU limit |
| Memory pressure | `docker stats` | Increase memory limit |
| Many services | Service count | Increase inference interval |

### Training Too Slow

| Symptom | Check | Solution |
|---------|-------|----------|
| > 10 min per service | Data volume | Reduce training window |
| Memory spikes | `docker stats` | Increase memory, reduce batch size |
| Disk I/O | `iostat` | Use SSD storage |

### Reduce Resource Usage

```yaml
# Limit resources in docker-compose.yml
deploy:
  resources:
    limits:
      memory: 2G
      cpus: '1.0'
    reservations:
      memory: 512M
      cpus: '0.5'
```

---

## Incident Response

### Anomaly Detection is Down

**Impact**: No anomaly alerts for any service.

**Immediate Actions**:
1. Check container status: `docker compose ps`
2. Check logs: `docker compose logs --tail 100`
3. Restart if needed: `docker compose restart`

**Recovery Verification**:
```bash
# Verify inference runs
docker compose run --rm yaga inference --verbose
```

### Missed Real Incident

**Root Cause Analysis**:
1. Was the model trained? Check `smartbox_models/<service>/`
2. Was inference running? Check log timestamps
3. Was contamination too high? Check config
4. Was it a novel failure pattern? Review ML limitations

**Corrective Actions**:
1. Add rule-based override if pattern is known
2. Adjust contamination if too lenient
3. Retrain models with recent data
4. Consider adding the service to `critical` category

### False Positive Storm

**Immediate Actions**:
1. Identify affected service(s) from logs
2. Check for legitimate infrastructure changes
3. Temporarily increase contamination if needed

**Long-term Fix**:
```json
{
  "model": {
    "contamination_by_service": {
      "affected-service": 0.10
    }
  }
}
```

Then retrain: `docker compose run --rm yaga train`

---

## Runbook: Adding a New Service

1. **Add to config.json**:
   ```json
   {
     "services": {
       "standard": [..., "new-service"]
     }
   }
   ```

2. **Rebuild container** (config is baked in):
   ```bash
   docker compose build
   docker compose up -d
   ```

3. **Train models**:
   ```bash
   docker compose run --rm yaga train
   ```

4. **Verify detection**:
   ```bash
   docker compose run --rm yaga inference --verbose
   ```

5. **Monitor for 1-2 weeks** and adjust contamination if needed.

---

## Runbook: Changing Detection Sensitivity

1. **Identify the service** showing too many or too few alerts.

2. **Adjust contamination** in config.json:
   ```json
   {
     "model": {
       "contamination_by_service": {
         "service-name": 0.05
       }
     }
   }
   ```

3. **Rebuild and restart**:
   ```bash
   docker compose build
   docker compose up -d
   ```

4. **Retrain the service**:
   ```bash
   docker compose run --rm yaga train
   ```

---

## Runbook: Enabling Drift Detection

1. **Update config.json** to enable drift checking:
   ```json
   {
     "inference": {
       "check_drift": true
     }
   }
   ```

2. **Retrain models** (required to compute drift baselines):
   ```bash
   docker compose run --rm yaga train
   ```

3. **Rebuild and restart**:
   ```bash
   docker compose build
   docker compose up -d
   ```

4. **Monitor drift output**:
   ```bash
   # Check for drift warnings in inference output
   docker compose run --rm yaga inference --verbose 2>&1 | grep -E "(drift|penalty)"
   ```

5. **Interpret drift results**:
   - Score < 3: Normal variance
   - Score 3-5: Monitor, consider retraining if persistent
   - Score > 5: Retrain soon

---

## Runbook: Recovering from Corrupted State

1. **Stop the container**:
   ```bash
   docker compose down
   ```

2. **Backup current state** (for investigation):
   ```bash
   cp -r data/ data-backup-$(date +%Y%m%d)/
   ```

3. **Remove corrupted database**:
   ```bash
   rm data/anomaly_state.db
   ```

4. **Restart** (new database will be created):
   ```bash
   docker compose up -d
   ```

5. **Verify**:
   ```bash
   docker compose run --rm yaga inference --verbose
   ```

---

## Contact and Escalation

| Issue | First Response | Escalation |
|-------|----------------|------------|
| Container down | Restart container | Check host resources |
| Training failures | Check VM connectivity | Review data quality |
| False positives | Adjust contamination | Review ML thresholds |
| Missed incidents | Check model coverage | Add rule-based overrides |

---

## Further Reading

- [DEPLOYMENT.md](./DEPLOYMENT.md) - Deployment and infrastructure
- [CONFIGURATION.md](./CONFIGURATION.md) - All configuration options
- [MACHINE_LEARNING.md](./MACHINE_LEARNING.md) - Understanding the ML system
- [KNOWN_ISSUES.md](./KNOWN_ISSUES.md) - Known limitations and workarounds
