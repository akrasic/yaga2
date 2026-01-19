#!/usr/bin/env python3
"""
Yaga2 - Anomaly Detection Control Center v2.1

A mission-control style admin interface for managing the ML anomaly detection system.

Features:
- Service management with search/filter
- SLO configuration with validation (latency, errors, database latency, request rate)
- Exception enrichment configuration
- Improvement signal filtering
- Dependency graph visualization
- Config export/import
- Unsaved changes tracking
- Audit logging

Run with: python admin_dashboard.py
Access at: http://localhost:8050
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any
import hashlib

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from starlette.middleware.base import BaseHTTPMiddleware
import uvicorn

# Configuration
CONFIG_PATH = Path(__file__).parent / "config.json"
AUDIT_LOG_PATH = Path(__file__).parent / ".config_audit.log"

app = FastAPI(title="Yaga2 Control Center", version="2.1.0")


# ============================================================================
# Request Logging Middleware (nginx combined log format)
# ============================================================================

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Request logging middleware using nginx combined log format."""

    async def dispatch(self, request: Request, call_next):
        start_time = time.perf_counter()

        # Process the request
        response = await call_next(request)

        # Calculate duration
        duration_secs = time.perf_counter() - start_time

        # Get request info
        client_host = request.client.host if request.client else "-"
        method = request.method
        path = request.url.path
        if request.url.query:
            path = f"{path}?{request.url.query}"
        http_version = request.scope.get("http_version", "1.1")
        status_code = response.status_code
        content_length = response.headers.get("content-length", "-")
        referer = request.headers.get("referer", "-")
        user_agent = request.headers.get("user-agent", "-")

        # Format timestamp like nginx: [16/Jan/2026:14:30:15 +0000]
        timestamp = datetime.now().strftime("[%d/%b/%Y:%H:%M:%S %z]").strip()
        if not timestamp.endswith("]"):
            # If no timezone info, add +0000
            timestamp = datetime.now().strftime("[%d/%b/%Y:%H:%M:%S +0000]")

        # nginx combined log format with response time extension
        # Format: $remote_addr - $remote_user [$time_local] "$request" $status $body_bytes_sent "$http_referer" "$http_user_agent" $request_time
        log_line = (
            f'{client_host} - - {timestamp} '
            f'"{method} {path} HTTP/{http_version}" '
            f'{status_code} {content_length} '
            f'"{referer}" "{user_agent}" '
            f'{duration_secs:.3f}'
        )
        print(log_line)

        return response


# Register middleware if enabled (works with uvicorn reload)
if os.environ.get("YAGA_ACCESS_LOG", "0") == "1":
    app.add_middleware(RequestLoggingMiddleware)


def load_config() -> dict[str, Any]:
    """Load configuration from JSON file."""
    if not CONFIG_PATH.exists():
        raise HTTPException(status_code=404, detail="Configuration file not found")
    with open(CONFIG_PATH) as f:
        return json.load(f)


def save_config(config: dict[str, Any], change_description: str = "Manual update") -> None:
    """Save configuration to JSON file with backup and audit logging."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create timestamped backup
    backup_dir = CONFIG_PATH.parent / ".config_backups"
    backup_dir.mkdir(exist_ok=True)

    if CONFIG_PATH.exists():
        backup_path = backup_dir / f"config_{timestamp}.json"
        with open(CONFIG_PATH) as f:
            backup_data = f.read()
        with open(backup_path, "w") as f:
            f.write(backup_data)

        # Keep only last 20 backups
        backups = sorted(backup_dir.glob("config_*.json"), reverse=True)
        for old_backup in backups[20:]:
            old_backup.unlink()

    # Save new config
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)

    # Audit log
    log_entry = f"{datetime.now().isoformat()} | {change_description}\n"
    with open(AUDIT_LOG_PATH, "a") as f:
        f.write(log_entry)


def validate_config(config: dict[str, Any]) -> list[str]:
    """Validate configuration and return list of issues."""
    issues = []

    # Validate SLO thresholds
    slos = config.get("slos", {})
    defaults = slos.get("defaults", {})

    if defaults:
        if defaults.get("latency_acceptable_ms", 0) >= defaults.get("latency_warning_ms", 0):
            issues.append("SLO: latency_acceptable_ms should be less than latency_warning_ms")
        if defaults.get("latency_warning_ms", 0) >= defaults.get("latency_critical_ms", 0):
            issues.append("SLO: latency_warning_ms should be less than latency_critical_ms")
        if defaults.get("error_rate_acceptable", 0) >= defaults.get("error_rate_warning", 0):
            issues.append("SLO: error_rate_acceptable should be less than error_rate_warning")
        if defaults.get("error_rate_warning", 0) >= defaults.get("error_rate_critical", 0):
            issues.append("SLO: error_rate_warning should be less than error_rate_critical")

    # Validate service SLOs
    for service, slo in slos.get("services", {}).items():
        if slo.get("latency_acceptable_ms", 0) >= slo.get("latency_critical_ms", float('inf')):
            issues.append(f"SLO [{service}]: latency_acceptable_ms >= latency_critical_ms")

    # Validate fingerprinting
    fp = config.get("fingerprinting", {})
    if fp.get("confirmation_cycles", 2) < 1:
        issues.append("Fingerprinting: confirmation_cycles must be at least 1")
    if fp.get("resolution_grace_cycles", 3) < 1:
        issues.append("Fingerprinting: resolution_grace_cycles must be at least 1")

    # Validate VictoriaMetrics
    vm = config.get("victoria_metrics", {})
    if not vm.get("endpoint"):
        issues.append("VictoriaMetrics: endpoint is required")
    if vm.get("timeout_seconds", 10) < 1:
        issues.append("VictoriaMetrics: timeout_seconds must be at least 1")

    return issues


def get_config_hash(config: dict) -> str:
    """Get hash of config for change detection."""
    return hashlib.md5(json.dumps(config, sort_keys=True).encode()).hexdigest()[:8]


# API Routes
@app.get("/api/config")
async def get_config():
    """Get full configuration with metadata."""
    config = load_config()
    return {
        "config": config,
        "hash": get_config_hash(config),
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/api/config")
async def update_config(request: Request):
    """Update full configuration with validation."""
    try:
        data = await request.json()
        config = data.get("config", data)
        description = data.get("description", "Manual update via dashboard")

        # Validate
        issues = validate_config(config)
        if issues and not data.get("force", False):
            return JSONResponse(
                status_code=400,
                content={"status": "validation_failed", "issues": issues}
            )

        save_config(config, description)
        return {
            "status": "success",
            "message": "Configuration saved",
            "hash": get_config_hash(config),
            "warnings": issues if issues else None
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/config/validate")
async def validate_current_config():
    """Validate current configuration."""
    config = load_config()
    issues = validate_config(config)
    return {
        "valid": len(issues) == 0,
        "issues": issues,
    }


@app.post("/api/config/validate")
async def validate_config_body(request: Request):
    """Validate configuration from request body."""
    config = await request.json()
    issues = validate_config(config)
    return {
        "valid": len(issues) == 0,
        "issues": issues,
    }


@app.get("/api/config/export")
async def export_config():
    """Export configuration as downloadable file."""
    if not CONFIG_PATH.exists():
        raise HTTPException(status_code=404, detail="Configuration file not found")
    return FileResponse(
        CONFIG_PATH,
        media_type="application/json",
        filename=f"yaga2_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )


@app.get("/api/config/backups")
async def list_backups():
    """List available config backups."""
    backup_dir = CONFIG_PATH.parent / ".config_backups"
    if not backup_dir.exists():
        return {"backups": []}

    backups = []
    for f in sorted(backup_dir.glob("config_*.json"), reverse=True):
        backups.append({
            "name": f.name,
            "timestamp": f.stem.replace("config_", ""),
            "size": f.stat().st_size,
        })
    return {"backups": backups[:20]}


@app.post("/api/config/restore/{backup_name}")
async def restore_backup(backup_name: str):
    """Restore configuration from backup."""
    backup_path = CONFIG_PATH.parent / ".config_backups" / backup_name
    if not backup_path.exists():
        raise HTTPException(status_code=404, detail="Backup not found")

    with open(backup_path) as f:
        config = json.load(f)

    save_config(config, f"Restored from backup: {backup_name}")
    return {"status": "success", "message": f"Restored from {backup_name}"}


@app.get("/api/audit-log")
async def get_audit_log():
    """Get recent audit log entries."""
    if not AUDIT_LOG_PATH.exists():
        return {"entries": []}

    with open(AUDIT_LOG_PATH) as f:
        lines = f.readlines()[-50:]  # Last 50 entries

    return {"entries": [line.strip() for line in reversed(lines)]}


@app.get("/api/services/summary")
async def get_services_summary():
    """Get summary of all services with their categories and SLOs."""
    config = load_config()
    services = config.get("services", {})
    slos = config.get("slos", {})
    contamination = config.get("model", {}).get("contamination_by_service", {})
    dependencies = config.get("dependencies", {}).get("graph", {})

    summary = []
    for category in ["critical", "standard", "micro", "admin", "core"]:
        for service in services.get(category, []):
            slo = slos.get("services", {}).get(service, slos.get("defaults", {}))
            deps = dependencies.get(service, [])
            summary.append({
                "name": service,
                "category": category,
                "contamination": contamination.get(service, config.get("model", {}).get("contamination_by_category", {}).get(category, 0.05)),
                "slo": slo,
                "has_custom_slo": service in slos.get("services", {}),
                "dependencies": deps,
                "dependents": [s for s, d in dependencies.items() if service in d],
            })

    return summary


@app.get("/api/dependencies/graph")
async def get_dependency_graph():
    """Get dependency graph data for visualization."""
    config = load_config()
    services = config.get("services", {})
    dependencies = config.get("dependencies", {}).get("graph", {})

    # Build nodes
    nodes = []
    all_services = set()
    for category in ["critical", "standard", "micro", "admin", "core"]:
        for service in services.get(category, []):
            all_services.add(service)
            nodes.append({
                "id": service,
                "category": category,
            })

    # Build edges
    edges = []
    for source, targets in dependencies.items():
        for target in targets:
            if source in all_services and target in all_services:
                edges.append({"source": source, "target": target})

    return {"nodes": nodes, "edges": edges}


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    config = load_config()
    issues = validate_config(config)
    return {
        "status": "healthy" if not issues else "degraded",
        "config_exists": CONFIG_PATH.exists(),
        "config_valid": len(issues) == 0,
        "validation_issues": len(issues),
        "timestamp": datetime.now().isoformat(),
    }


# Training Report API endpoints
try:
    from smartbox_anomaly.training import TrainingRunStorage
    _training_storage = TrainingRunStorage()
except ImportError:
    _training_storage = None


@app.get("/api/training/runs")
async def get_training_runs(service: str = None, limit: int = 50):
    """Get recent training runs, optionally filtered by service."""
    if _training_storage is None:
        raise HTTPException(status_code=500, detail="Training storage not available")

    if service:
        runs = _training_storage.get_runs_for_service(service, limit=limit)
    else:
        runs = _training_storage.get_recent_runs(limit=limit)

    return {
        "runs": [r.to_dict() for r in runs],
        "count": len(runs),
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/api/training/runs/{run_id}")
async def get_training_run(run_id: str):
    """Get details of a specific training run."""
    if _training_storage is None:
        raise HTTPException(status_code=500, detail="Training storage not available")

    run = _training_storage.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Training run not found")

    return run.to_dict()


@app.get("/api/training/latest")
async def get_latest_training_runs():
    """Get latest training run for each service."""
    if _training_storage is None:
        raise HTTPException(status_code=500, detail="Training storage not available")

    runs = _training_storage.get_latest_run_per_service()
    # Return as array sorted by service name for the UI
    runs_list = [r.to_dict() for r in runs.values()]
    runs_list.sort(key=lambda x: x.get("service_name", ""))
    return {
        "runs": runs_list,
        "count": len(runs_list),
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/api/training/summary")
async def get_training_summary():
    """Get summary statistics of training runs."""
    if _training_storage is None:
        raise HTTPException(status_code=500, detail="Training storage not available")

    stats = _training_storage.get_summary_stats()
    # Extract validation counts for the UI (expects passed, warnings, failed)
    by_validation = stats.get("by_validation", {})
    return {
        "total_runs": stats.get("total_runs", 0),
        "passed": by_validation.get("PASSED", 0),
        "warnings": by_validation.get("WARNING", 0),
        "failed": by_validation.get("FAILED", 0),
        "by_status": stats.get("by_status", {}),
        "recent_failures": stats.get("recent_failures", []),
        "timestamp": datetime.now().isoformat(),
    }


# Dashboard HTML
DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Yaga2 — Anomaly Detection Control Center</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-primary: #0a0e14;
            --bg-secondary: #0d1117;
            --bg-tertiary: #161b22;
            --bg-elevated: #1c2128;
            --bg-hover: #21262d;
            --text-primary: #e6edf3;
            --text-secondary: #8b949e;
            --text-muted: #6e7681;
            --status-critical: #f85149;
            --status-critical-bg: rgba(248, 81, 73, 0.15);
            --status-warning: #d29922;
            --status-warning-bg: rgba(210, 153, 34, 0.15);
            --status-success: #3fb950;
            --status-success-bg: rgba(63, 185, 80, 0.15);
            --status-info: #58a6ff;
            --status-info-bg: rgba(88, 166, 255, 0.15);
            --accent: #79c0ff;
            --accent-hover: #a5d6ff;
            --border-default: #30363d;
            --border-muted: #21262d;
            --space-xs: 4px;
            --space-sm: 8px;
            --space-md: 16px;
            --space-lg: 24px;
            --space-xl: 32px;
            --radius-sm: 4px;
            --radius-md: 6px;
            --radius-lg: 8px;
        }
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Space Grotesk', -apple-system, BlinkMacSystemFont, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.5;
            min-height: 100vh;
        }
        ::-webkit-scrollbar { width: 8px; height: 8px; }
        ::-webkit-scrollbar-track { background: var(--bg-secondary); }
        ::-webkit-scrollbar-thumb { background: var(--border-default); border-radius: 4px; }
        ::-webkit-scrollbar-thumb:hover { background: var(--text-muted); }

        .app-container {
            display: grid;
            grid-template-columns: 260px 1fr;
            grid-template-rows: 60px 1fr;
            min-height: 100vh;
        }
        .header {
            grid-column: 1 / -1;
            background: var(--bg-secondary);
            border-bottom: 1px solid var(--border-default);
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0 var(--space-lg);
        }
        .header-left { display: flex; align-items: center; gap: var(--space-md); }
        .logo { display: flex; align-items: center; gap: var(--space-sm); font-weight: 600; font-size: 15px; letter-spacing: -0.3px; }
        .logo-icon {
            width: 28px; height: 28px;
            background: linear-gradient(135deg, var(--status-info) 0%, var(--accent) 100%);
            border-radius: var(--radius-md);
            display: flex; align-items: center; justify-content: center;
            font-size: 14px;
        }
        .header-actions { display: flex; align-items: center; gap: var(--space-md); }
        .header-status {
            display: flex; align-items: center; gap: var(--space-lg);
            font-family: 'JetBrains Mono', monospace;
            font-size: 12px; color: var(--text-secondary);
        }
        .status-item { display: flex; align-items: center; gap: var(--space-xs); }
        .status-dot {
            width: 8px; height: 8px; border-radius: 50%;
            background: var(--status-success);
            animation: pulse 2s ease-in-out infinite;
        }
        .status-dot.warning { background: var(--status-warning); }
        .status-dot.error { background: var(--status-critical); }
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }

        .unsaved-indicator {
            display: none;
            align-items: center;
            gap: var(--space-xs);
            padding: var(--space-xs) var(--space-sm);
            background: var(--status-warning-bg);
            border: 1px solid var(--status-warning);
            border-radius: var(--radius-md);
            font-size: 12px;
            color: var(--status-warning);
        }
        .unsaved-indicator.active { display: flex; }

        .sidebar {
            background: var(--bg-secondary);
            border-right: 1px solid var(--border-default);
            padding: var(--space-md);
            display: flex;
            flex-direction: column;
            gap: var(--space-xs);
            overflow-y: auto;
        }
        .nav-section { margin-bottom: var(--space-md); }
        .nav-section-title {
            font-size: 10px; font-weight: 600;
            text-transform: uppercase; letter-spacing: 0.5px;
            color: var(--text-muted);
            padding: var(--space-sm) var(--space-sm);
            margin-bottom: var(--space-xs);
        }
        .nav-item {
            display: flex; align-items: center; gap: var(--space-sm);
            padding: var(--space-sm) var(--space-md);
            border-radius: var(--radius-md);
            cursor: pointer;
            transition: all 0.15s ease;
            font-size: 13px;
            color: var(--text-secondary);
            border: 1px solid transparent;
        }
        .nav-item:hover { background: var(--bg-hover); color: var(--text-primary); }
        .nav-item.active { background: var(--bg-tertiary); color: var(--text-primary); border-color: var(--border-default); }
        .nav-item-icon { width: 18px; height: 18px; opacity: 0.7; }
        .nav-item.active .nav-item-icon { opacity: 1; }
        .nav-badge {
            margin-left: auto;
            background: var(--bg-elevated);
            padding: 2px 6px;
            border-radius: 10px;
            font-size: 10px;
            font-family: 'JetBrains Mono', monospace;
        }

        .main-content { padding: var(--space-lg); overflow-y: auto; background: var(--bg-primary); }
        .page { display: none; animation: fadeIn 0.2s ease; }
        .page.active { display: block; }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(4px); } to { opacity: 1; transform: translateY(0); } }

        .page-header { margin-bottom: var(--space-lg); display: flex; align-items: flex-start; justify-content: space-between; }
        .page-header-text { flex: 1; }
        .page-title { font-size: 24px; font-weight: 600; letter-spacing: -0.5px; margin-bottom: var(--space-xs); }
        .page-description { color: var(--text-secondary); font-size: 14px; }
        .page-actions { display: flex; gap: var(--space-sm); }

        .card { background: var(--bg-secondary); border: 1px solid var(--border-default); border-radius: var(--radius-lg); overflow: hidden; }
        .card-header {
            padding: var(--space-md) var(--space-lg);
            border-bottom: 1px solid var(--border-default);
            display: flex; align-items: center; justify-content: space-between;
        }
        .card-title { font-size: 14px; font-weight: 600; display: flex; align-items: center; gap: var(--space-sm); }
        .card-body { padding: var(--space-lg); }

        .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: var(--space-md); margin-bottom: var(--space-lg); }
        .stat-card { background: var(--bg-secondary); border: 1px solid var(--border-default); border-radius: var(--radius-lg); padding: var(--space-lg); }
        .stat-label { font-size: 12px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: var(--space-xs); }
        .stat-value { font-family: 'JetBrains Mono', monospace; font-size: 28px; font-weight: 600; color: var(--text-primary); }
        .stat-detail { font-size: 12px; color: var(--text-secondary); margin-top: var(--space-xs); }

        .search-box {
            display: flex; align-items: center; gap: var(--space-sm);
            background: var(--bg-tertiary); border: 1px solid var(--border-default);
            border-radius: var(--radius-md); padding: var(--space-sm) var(--space-md);
            margin-bottom: var(--space-md);
        }
        .search-box input {
            flex: 1; background: none; border: none; color: var(--text-primary);
            font-family: inherit; font-size: 13px; outline: none;
        }
        .search-box input::placeholder { color: var(--text-muted); }
        .search-icon { color: var(--text-muted); }

        .services-table { width: 100%; border-collapse: collapse; }
        .services-table th {
            text-align: left; padding: var(--space-sm) var(--space-md);
            font-size: 11px; font-weight: 600; text-transform: uppercase;
            letter-spacing: 0.5px; color: var(--text-muted);
            border-bottom: 1px solid var(--border-default);
        }
        .services-table td { padding: var(--space-md); border-bottom: 1px solid var(--border-muted); font-size: 13px; }
        .services-table tr:hover { background: var(--bg-hover); }
        .services-table tr.hidden { display: none; }
        .service-name { font-family: 'JetBrains Mono', monospace; font-weight: 500; }

        .category-badge {
            display: inline-block; padding: 3px 8px; border-radius: 4px;
            font-size: 11px; font-weight: 500; text-transform: uppercase;
            background: rgba(139, 148, 158, 0.15); color: #8b949e; /* Default for custom categories */
        }
        .category-critical { background: var(--status-critical-bg); color: var(--status-critical); }
        .category-standard { background: var(--status-info-bg); color: var(--status-info); }
        .category-micro { background: var(--status-success-bg); color: var(--status-success); }
        .category-admin { background: var(--status-warning-bg); color: var(--status-warning); }
        .category-core { background: rgba(163, 113, 247, 0.15); color: #a371f7; }
        .category-background { background: rgba(110, 118, 129, 0.15); color: #6e7681; }
        .category-warning { background: var(--status-warning-bg); color: var(--status-warning); }

        .status-badge {
            display: inline-flex; align-items: center; gap: 4px;
            padding: 4px 10px; border-radius: 4px;
            font-size: 12px; font-weight: 500;
        }
        .status-passed { background: var(--status-success-bg); color: var(--status-success); }
        .status-warning { background: var(--status-warning-bg); color: var(--status-warning); }
        .status-failed { background: var(--status-critical-bg); color: var(--status-critical); }
        .status-unknown { background: var(--bg-tertiary); color: var(--text-muted); }

        .form-group { margin-bottom: var(--space-md); }
        .form-label { display: block; font-size: 12px; font-weight: 500; color: var(--text-secondary); margin-bottom: var(--space-xs); }
        .form-label-with-help { display: flex; align-items: center; gap: var(--space-xs); }
        .help-icon {
            width: 14px; height: 14px;
            display: inline-flex; align-items: center; justify-content: center;
            background: var(--bg-tertiary); border-radius: 50%;
            font-size: 10px; color: var(--text-muted); cursor: help;
            position: relative;
        }
        .help-icon:hover .tooltip { display: block; }
        .tooltip {
            display: none; position: absolute; bottom: calc(100% + 8px); left: 50%;
            transform: translateX(-50%);
            background: var(--bg-elevated); border: 1px solid var(--border-default);
            border-radius: var(--radius-md); padding: var(--space-sm) var(--space-md);
            font-size: 12px; color: var(--text-secondary);
            z-index: 100; max-width: 300px; white-space: normal;
        }
        .tooltip::after {
            content: ''; position: absolute; top: 100%; left: 50%;
            transform: translateX(-50%);
            border: 6px solid transparent; border-top-color: var(--border-default);
        }

        .form-input, .form-select {
            width: 100%; padding: var(--space-sm) var(--space-md);
            background: var(--bg-tertiary); border: 1px solid var(--border-default);
            border-radius: var(--radius-md); color: var(--text-primary);
            font-family: 'JetBrains Mono', monospace; font-size: 13px;
            transition: border-color 0.15s ease;
        }
        .form-input:focus, .form-select:focus { outline: none; border-color: var(--accent); }
        .form-input::placeholder { color: var(--text-muted); }
        .form-input.error { border-color: var(--status-critical); }

        .btn {
            display: inline-flex; align-items: center; gap: var(--space-xs);
            padding: var(--space-sm) var(--space-md);
            border-radius: var(--radius-md);
            font-size: 13px; font-weight: 500;
            cursor: pointer; transition: all 0.15s ease;
            border: 1px solid transparent;
        }
        .btn-primary { background: var(--accent); color: var(--bg-primary); }
        .btn-primary:hover { background: var(--accent-hover); }
        .btn-primary:disabled { opacity: 0.5; cursor: not-allowed; }
        .btn-secondary { background: var(--bg-tertiary); border-color: var(--border-default); color: var(--text-primary); }
        .btn-secondary:hover { background: var(--bg-hover); }
        .btn-danger { background: var(--status-critical-bg); color: var(--status-critical); border-color: var(--status-critical); }
        .btn-danger:hover { background: var(--status-critical); color: white; }
        .btn-sm { padding: var(--space-xs) var(--space-sm); font-size: 12px; }
        .btn-icon { padding: var(--space-sm); }

        .toggle { position: relative; width: 44px; height: 24px; }
        .toggle input { opacity: 0; width: 0; height: 0; }
        .toggle-slider {
            position: absolute; cursor: pointer; top: 0; left: 0; right: 0; bottom: 0;
            background: var(--bg-tertiary); border-radius: 12px;
            transition: 0.2s; border: 1px solid var(--border-default);
        }
        .toggle-slider:before {
            position: absolute; content: "";
            height: 18px; width: 18px; left: 2px; bottom: 2px;
            background: var(--text-muted); border-radius: 50%; transition: 0.2s;
        }
        .toggle input:checked + .toggle-slider { background: var(--status-success-bg); border-color: var(--status-success); }
        .toggle input:checked + .toggle-slider:before { transform: translateX(20px); background: var(--status-success); }

        .slo-bar { display: flex; height: 8px; border-radius: 4px; overflow: hidden; background: var(--bg-tertiary); margin-top: var(--space-xs); }
        .slo-segment { height: 100%; }
        .slo-segment-ok { background: var(--status-success); }
        .slo-segment-warning { background: var(--status-warning); }
        .slo-segment-critical { background: var(--status-critical); }

        .modal-overlay {
            display: none; position: fixed; inset: 0;
            background: rgba(0, 0, 0, 0.7); z-index: 1000;
            align-items: center; justify-content: center;
            backdrop-filter: blur(4px);
        }
        .modal-overlay.active { display: flex; }
        .modal {
            background: var(--bg-secondary); border: 1px solid var(--border-default);
            border-radius: var(--radius-lg); width: 100%; max-width: 600px;
            max-height: 80vh; overflow: hidden;
            animation: modalIn 0.2s ease;
        }
        @keyframes modalIn { from { opacity: 0; transform: scale(0.95); } to { opacity: 1; transform: scale(1); } }
        .modal-header {
            padding: var(--space-lg); border-bottom: 1px solid var(--border-default);
            display: flex; align-items: center; justify-content: space-between;
        }
        .modal-title { font-size: 16px; font-weight: 600; }
        .modal-close {
            background: none; border: none; color: var(--text-muted);
            cursor: pointer; padding: var(--space-xs); border-radius: var(--radius-sm);
        }
        .modal-close:hover { background: var(--bg-hover); color: var(--text-primary); }
        .modal-body { padding: var(--space-lg); overflow-y: auto; max-height: calc(80vh - 140px); }
        .modal-footer {
            padding: var(--space-md) var(--space-lg);
            border-top: 1px solid var(--border-default);
            display: flex; justify-content: flex-end; gap: var(--space-sm);
        }

        .toast-container {
            position: fixed; bottom: var(--space-lg); right: var(--space-lg);
            z-index: 2000; display: flex; flex-direction: column; gap: var(--space-sm);
        }
        .toast {
            background: var(--bg-elevated); border: 1px solid var(--border-default);
            border-radius: var(--radius-md); padding: var(--space-md) var(--space-lg);
            display: flex; align-items: center; gap: var(--space-sm);
            font-size: 13px; animation: toastIn 0.3s ease;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        }
        @keyframes toastIn { from { opacity: 0; transform: translateX(20px); } to { opacity: 1; transform: translateX(0); } }
        .toast.success { border-left: 3px solid var(--status-success); }
        .toast.error { border-left: 3px solid var(--status-critical); }
        .toast.warning { border-left: 3px solid var(--status-warning); }

        .grid-2 { display: grid; grid-template-columns: repeat(2, 1fr); gap: var(--space-md); }
        .grid-3 { display: grid; grid-template-columns: repeat(3, 1fr); gap: var(--space-md); }
        .grid-4 { display: grid; grid-template-columns: repeat(4, 1fr); gap: var(--space-md); }
        .section-divider { height: 1px; background: var(--border-default); margin: var(--space-lg) 0; }

        .code-block {
            background: var(--bg-tertiary); border: 1px solid var(--border-default);
            border-radius: var(--radius-md); padding: var(--space-md);
            font-family: 'JetBrains Mono', monospace; font-size: 12px;
            overflow-x: auto; white-space: pre;
        }

        .help-panel {
            background: var(--status-info-bg); border: 1px solid rgba(88, 166, 255, 0.3);
            border-radius: var(--radius-md); padding: var(--space-md); margin-bottom: var(--space-lg);
        }
        .help-panel-title {
            font-size: 13px; font-weight: 600; color: var(--status-info);
            margin-bottom: var(--space-xs);
            display: flex; align-items: center; gap: var(--space-xs);
        }
        .help-panel-text { font-size: 13px; color: var(--text-secondary); line-height: 1.6; }

        .validation-issues {
            background: var(--status-critical-bg); border: 1px solid rgba(248, 81, 73, 0.3);
            border-radius: var(--radius-md); padding: var(--space-md); margin-bottom: var(--space-lg);
        }
        .validation-issues-title { font-size: 13px; font-weight: 600; color: var(--status-critical); margin-bottom: var(--space-sm); }
        .validation-issues ul { margin: 0; padding-left: var(--space-lg); color: var(--text-secondary); font-size: 13px; }

        .busy-period-card {
            background: var(--bg-tertiary); border: 1px solid var(--border-default);
            border-radius: var(--radius-md); padding: var(--space-md); margin-bottom: var(--space-sm);
            display: flex; align-items: center; justify-content: space-between;
        }
        .busy-period-dates { font-family: 'JetBrains Mono', monospace; font-size: 13px; }

        .tabs {
            display: flex; gap: var(--space-xs); margin-bottom: var(--space-lg);
            border-bottom: 1px solid var(--border-default); padding-bottom: var(--space-sm);
        }
        .tab {
            padding: var(--space-sm) var(--space-md);
            border-radius: var(--radius-md) var(--radius-md) 0 0;
            cursor: pointer; font-size: 13px; color: var(--text-secondary);
            transition: all 0.15s ease;
        }
        .tab:hover { color: var(--text-primary); }
        .tab.active { color: var(--text-primary); background: var(--bg-tertiary); }

        .empty-state { text-align: center; padding: var(--space-xl); color: var(--text-muted); }

        .dep-graph {
            background: var(--bg-tertiary); border: 1px solid var(--border-default);
            border-radius: var(--radius-md); height: 400px; position: relative;
            overflow: hidden;
        }
        .dep-graph-node {
            position: absolute; padding: var(--space-xs) var(--space-sm);
            background: var(--bg-secondary); border: 1px solid var(--border-default);
            border-radius: var(--radius-sm); font-size: 11px;
            font-family: 'JetBrains Mono', monospace;
            cursor: pointer; transition: all 0.15s ease;
        }
        .dep-graph-node:hover { border-color: var(--accent); z-index: 10; }
        .dep-graph-node.critical { border-color: var(--status-critical); }
        .dep-graph-node.standard { border-color: var(--status-info); }

        /* Pipeline Flow Visualization */
        .pipeline-flow {
            display: flex; align-items: center; justify-content: center;
            gap: var(--space-md); padding: var(--space-md) 0;
            flex-wrap: wrap;
        }
        .pipeline-stage {
            display: flex; flex-direction: column; align-items: center;
            gap: var(--space-xs); min-width: 120px;
        }
        .pipeline-stage-icon {
            width: 48px; height: 48px; border-radius: 12px;
            display: flex; align-items: center; justify-content: center;
            border: 2px solid; transition: all 0.2s ease;
        }
        .pipeline-stage-icon.disabled {
            background: var(--bg-tertiary) !important;
            border-color: var(--border-default) !important;
            opacity: 0.5;
        }
        .pipeline-stage-label {
            font-size: 13px; font-weight: 600; color: var(--text-primary);
        }
        .pipeline-stage-desc {
            font-size: 11px; color: var(--text-muted);
            font-family: 'JetBrains Mono', monospace;
        }
        .pipeline-arrow {
            font-size: 20px; color: var(--text-muted);
            font-weight: 300;
        }

        /* Config Summary Grid */
        .config-summary-grid {
            display: grid; grid-template-columns: repeat(2, 1fr); gap: var(--space-sm);
        }
        .config-summary-item {
            display: flex; justify-content: space-between; align-items: center;
            padding: var(--space-xs) 0;
            border-bottom: 1px solid var(--border-muted);
        }
        .config-summary-item:last-child { border-bottom: none; }
        .config-summary-label { font-size: 12px; color: var(--text-secondary); }
        .config-summary-value {
            font-family: 'JetBrains Mono', monospace;
            font-size: 12px; font-weight: 500;
        }

        /* Improvement Badges */
        .improvement-metrics { display: flex; gap: var(--space-sm); flex-wrap: wrap; }
        .improvement-badge {
            display: inline-flex; align-items: center; gap: 4px;
            padding: 4px 10px; background: var(--status-success-bg);
            border: 1px solid var(--status-success);
            border-radius: 20px; font-size: 11px; font-weight: 500;
            font-family: 'JetBrains Mono', monospace;
            color: var(--status-success);
        }

        /* Database Latency Ratios */
        .ratio-grid {
            display: grid; grid-template-columns: repeat(4, 1fr); gap: var(--space-sm);
            margin-top: var(--space-md);
        }
        .ratio-item {
            text-align: center; padding: var(--space-sm);
            background: var(--bg-tertiary); border-radius: var(--radius-md);
        }
        .ratio-label { font-size: 10px; color: var(--text-muted); text-transform: uppercase; }
        .ratio-value {
            font-family: 'JetBrains Mono', monospace;
            font-size: 16px; font-weight: 600; margin-top: 2px;
        }
        .ratio-value.ok { color: var(--status-success); }
        .ratio-value.warning { color: var(--status-warning); }
        .ratio-value.high { color: #f0883e; }
        .ratio-value.critical { color: var(--status-critical); }

        /* Request Rate Config */
        .request-rate-grid { display: grid; grid-template-columns: 1fr 1fr; gap: var(--space-lg); }
        .request-rate-section { padding: var(--space-md); background: var(--bg-tertiary); border-radius: var(--radius-md); }
        .request-rate-title {
            font-size: 12px; font-weight: 600; color: var(--text-primary);
            margin-bottom: var(--space-sm); display: flex; align-items: center; gap: var(--space-xs);
        }
        .surge-icon { color: var(--status-critical); }
        .cliff-icon { color: var(--status-info); }

        .loading-overlay {
            position: fixed; inset: 0; background: rgba(10, 14, 20, 0.8);
            display: none; align-items: center; justify-content: center; z-index: 3000;
        }
        .loading-overlay.active { display: flex; }
        .loading-spinner {
            width: 40px; height: 40px; border: 3px solid var(--border-default);
            border-top-color: var(--accent); border-radius: 50%;
            animation: spin 1s linear infinite;
        }
        @keyframes spin { to { transform: rotate(360deg); } }

        @media (max-width: 1200px) { .grid-3, .grid-4 { grid-template-columns: repeat(2, 1fr); } }
        @media (max-width: 900px) {
            .app-container { grid-template-columns: 1fr; }
            .sidebar { display: none; }
            .grid-2, .grid-3, .grid-4 { grid-template-columns: 1fr; }
            .header { padding: 0 var(--space-md); }
            .page-header { flex-direction: column; gap: var(--space-md); }
        }
    </style>
</head>
<body>
    <div class="loading-overlay" id="loading-overlay">
        <div class="loading-spinner"></div>
    </div>

    <div class="app-container">
        <header class="header">
            <div class="header-left">
                <div class="logo">
                    <div class="logo-icon">🔮</div>
                    <span>Yaga2</span>
                </div>
                <div class="unsaved-indicator" id="unsaved-indicator">
                    <span>●</span> Unsaved changes
                </div>
            </div>
            <div class="header-actions">
                <button class="btn btn-secondary btn-sm" onclick="exportConfig()" title="Export config">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                        <polyline points="7 10 12 15 17 10"></polyline>
                        <line x1="12" y1="15" x2="12" y2="3"></line>
                    </svg>
                    Export
                </button>
                <button class="btn btn-secondary btn-sm" onclick="openImportModal()" title="Import config">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                        <polyline points="17 8 12 3 7 8"></polyline>
                        <line x1="12" y1="3" x2="12" y2="15"></line>
                    </svg>
                    Import
                </button>
            </div>
            <div class="header-status">
                <div class="status-item">
                    <div class="status-dot" id="health-dot"></div>
                    <span id="health-text">Loading...</span>
                </div>
                <div class="status-item" id="config-status">
                    Hash: <span id="config-hash">—</span>
                </div>
                <div class="status-item" id="clock"></div>
            </div>
        </header>

        <nav class="sidebar">
            <div class="nav-section">
                <div class="nav-section-title">Dashboard</div>
                <div class="nav-item" data-page="overview">
                    <svg class="nav-item-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <rect x="3" y="3" width="7" height="7"></rect>
                        <rect x="14" y="3" width="7" height="7"></rect>
                        <rect x="14" y="14" width="7" height="7"></rect>
                        <rect x="3" y="14" width="7" height="7"></rect>
                    </svg>
                    Overview
                </div>
                <div class="nav-item" data-page="training">
                    <svg class="nav-item-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"></path>
                        <path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"></path>
                        <line x1="12" y1="6" x2="12" y2="12"></line>
                        <line x1="9" y1="9" x2="15" y2="9"></line>
                    </svg>
                    Training Report
                </div>
            </div>
            <div class="nav-section">
                <div class="nav-section-title">Configuration</div>
                <div class="nav-item" data-page="categories">
                    <svg class="nav-item-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"></path>
                    </svg>
                    Categories
                    <span class="nav-badge" id="categories-count">-</span>
                </div>
                <div class="nav-item" data-page="services">
                    <svg class="nav-item-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M12 2L2 7l10 5 10-5-10-5z"></path>
                        <path d="M2 17l10 5 10-5"></path>
                        <path d="M2 12l10 5 10-5"></path>
                    </svg>
                    Services
                    <span class="nav-badge" id="services-count">-</span>
                </div>
                <div class="nav-item" data-page="slos">
                    <svg class="nav-item-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M22 12h-4l-3 9L9 3l-3 9H2"></path>
                    </svg>
                    SLO Thresholds
                </div>
                <div class="nav-item" data-page="dependencies">
                    <svg class="nav-item-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <circle cx="18" cy="5" r="3"></circle>
                        <circle cx="6" cy="12" r="3"></circle>
                        <circle cx="18" cy="19" r="3"></circle>
                        <line x1="8.59" y1="13.51" x2="15.42" y2="17.49"></line>
                        <line x1="15.41" y1="6.51" x2="8.59" y2="10.49"></line>
                    </svg>
                    Dependencies
                </div>
                <div class="nav-item" data-page="settings">
                    <svg class="nav-item-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <circle cx="12" cy="12" r="3"></circle>
                        <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"></path>
                    </svg>
                    System Settings
                </div>
            </div>
            <div class="nav-section">
                <div class="nav-section-title">Help</div>
                <div class="nav-item" data-page="docs">
                    <svg class="nav-item-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                        <polyline points="14 2 14 8 20 8"></polyline>
                        <line x1="16" y1="13" x2="8" y2="13"></line>
                        <line x1="16" y1="17" x2="8" y2="17"></line>
                    </svg>
                    Documentation
                </div>
                <div class="nav-item" data-page="audit">
                    <svg class="nav-item-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M12 20h9"></path>
                        <path d="M16.5 3.5a2.121 2.121 0 0 1 3 3L7 19l-4 1 1-4L16.5 3.5z"></path>
                    </svg>
                    Audit Log
                </div>
            </div>
        </nav>

        <main class="main-content">
            <!-- Overview Page -->
            <div class="page" id="page-overview">
                <div class="page-header">
                    <div class="page-header-text">
                        <h1 class="page-title">System Overview</h1>
                        <p class="page-description">Pipeline status and configuration summary</p>
                    </div>
                </div>
                <div id="validation-issues-container"></div>

                <!-- Pipeline Status -->
                <div class="card" style="margin-bottom: var(--space-lg);">
                    <div class="card-header">
                        <div class="card-title">
                            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline>
                            </svg>
                            Inference Pipeline
                        </div>
                    </div>
                    <div class="card-body">
                        <div class="pipeline-flow">
                            <div class="pipeline-stage">
                                <div class="pipeline-stage-icon" style="background: var(--status-info-bg); border-color: var(--status-info);">
                                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="var(--status-info)" stroke-width="2">
                                        <circle cx="11" cy="11" r="8"></circle>
                                        <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
                                    </svg>
                                </div>
                                <div class="pipeline-stage-label">Detection</div>
                                <div class="pipeline-stage-desc">ML pattern matching</div>
                            </div>
                            <div class="pipeline-arrow">→</div>
                            <div class="pipeline-stage">
                                <div class="pipeline-stage-icon" id="slo-pipeline-icon" style="background: var(--status-success-bg); border-color: var(--status-success);">
                                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="var(--status-success)" stroke-width="2">
                                        <path d="M22 12h-4l-3 9L9 3l-3 9H2"></path>
                                    </svg>
                                </div>
                                <div class="pipeline-stage-label">SLO Evaluation</div>
                                <div class="pipeline-stage-desc" id="slo-pipeline-status">Enabled</div>
                            </div>
                            <div class="pipeline-arrow">→</div>
                            <div class="pipeline-stage">
                                <div class="pipeline-stage-icon" id="enrichment-pipeline-icon" style="background: var(--status-warning-bg); border-color: var(--status-warning);">
                                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="var(--status-warning)" stroke-width="2">
                                        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                                        <polyline points="14 2 14 8 20 8"></polyline>
                                        <line x1="16" y1="13" x2="8" y2="13"></line>
                                        <line x1="16" y1="17" x2="8" y2="17"></line>
                                    </svg>
                                </div>
                                <div class="pipeline-stage-label">Exception Enrichment</div>
                                <div class="pipeline-stage-desc" id="enrichment-pipeline-status">On error breach</div>
                            </div>
                            <div class="pipeline-arrow">→</div>
                            <div class="pipeline-stage">
                                <div class="pipeline-stage-icon" id="service-graph-pipeline-icon" style="background: rgba(136, 192, 208, 0.15); border-color: #88c0d0;">
                                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#88c0d0" stroke-width="2">
                                        <circle cx="12" cy="12" r="3"></circle>
                                        <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"></path>
                                    </svg>
                                </div>
                                <div class="pipeline-stage-label">Service Graph</div>
                                <div class="pipeline-stage-desc" id="service-graph-pipeline-status">On latency breach</div>
                            </div>
                            <div class="pipeline-arrow">→</div>
                            <div class="pipeline-stage">
                                <div class="pipeline-stage-icon" style="background: rgba(163, 113, 247, 0.15); border-color: #a371f7;">
                                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#a371f7" stroke-width="2">
                                        <path d="M18 8A6 6 0 0 0 6 8c0 7-3 9-3 9h18s-3-2-3-9"></path>
                                        <path d="M13.73 21a2 2 0 0 1-3.46 0"></path>
                                    </svg>
                                </div>
                                <div class="pipeline-stage-label">Alert</div>
                                <div class="pipeline-stage-desc">Fingerprinted</div>
                            </div>
                        </div>
                    </div>
                </div>

                <div class="stats-grid" id="stats-grid"></div>

                <!-- Quick Config Summary -->
                <div class="grid-2" style="margin-bottom: var(--space-lg);">
                    <div class="card">
                        <div class="card-header">
                            <div class="card-title">SLO Thresholds (Defaults)</div>
                        </div>
                        <div class="card-body">
                            <div class="config-summary-grid" id="slo-summary"></div>
                        </div>
                    </div>
                    <div class="card">
                        <div class="card-header">
                            <div class="card-title">Improvement Filtering</div>
                        </div>
                        <div class="card-body">
                            <p style="color: var(--text-secondary); font-size: 13px; margin-bottom: var(--space-md);">
                                Lower-is-better metrics with improvements (direction=low) are automatically filtered out and don't trigger alerts.
                            </p>
                            <div class="improvement-metrics">
                                <span class="improvement-badge">dependency_latency ↓</span>
                                <span class="improvement-badge">database_latency ↓</span>
                                <span class="improvement-badge">error_rate ↓</span>
                            </div>
                        </div>
                    </div>
                </div>

                <div class="card">
                    <div class="card-header">
                        <div class="card-title">Services by Category</div>
                        <div class="search-box" style="margin: 0; width: 250px;">
                            <svg class="search-icon" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <circle cx="11" cy="11" r="8"></circle>
                                <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
                            </svg>
                            <input type="text" id="overview-search" placeholder="Search services..." oninput="filterOverviewServices()">
                        </div>
                    </div>
                    <div class="card-body" style="padding: 0;">
                        <table class="services-table" id="overview-services-table">
                            <thead>
                                <tr>
                                    <th>Service</th>
                                    <th>Category</th>
                                    <th>Contamination</th>
                                    <th>SLO Status</th>
                                    <th>Latency Threshold</th>
                                    <th>Dependencies</th>
                                </tr>
                            </thead>
                            <tbody id="overview-services-body"></tbody>
                        </table>
                    </div>
                </div>
            </div>

            <!-- Training Report Page -->
            <div class="page" id="page-training">
                <div class="page-header">
                    <div class="page-header-text">
                        <h1 class="page-title">Training Report</h1>
                        <p class="page-description">Model training history and validation status</p>
                    </div>
                    <button class="btn btn-secondary" onclick="loadTrainingData()">
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <polyline points="23 4 23 10 17 10"></polyline>
                            <path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"></path>
                        </svg>
                        Refresh
                    </button>
                </div>

                <!-- Training Summary Cards -->
                <div class="stats-grid" id="training-stats-grid">
                    <div class="stat-card">
                        <div class="stat-label">Total Runs</div>
                        <div class="stat-value" id="training-total-runs">-</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">Passed</div>
                        <div class="stat-value stat-success" id="training-passed">-</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">Warnings</div>
                        <div class="stat-value stat-warning" id="training-warnings">-</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">Failed</div>
                        <div class="stat-value stat-error" id="training-failed">-</div>
                    </div>
                </div>

                <!-- Latest Training Per Service -->
                <div class="card" style="margin-bottom: var(--space-lg);">
                    <div class="card-header">
                        <div class="card-title">
                            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <path d="M12 2L2 7l10 5 10-5-10-5z"></path>
                                <path d="M2 17l10 5 10-5"></path>
                                <path d="M2 12l10 5 10-5"></path>
                            </svg>
                            Latest Training Per Service
                        </div>
                    </div>
                    <div class="card-body" style="padding: 0;">
                        <table class="services-table" id="training-latest-table">
                            <thead>
                                <tr>
                                    <th>Service</th>
                                    <th>Status</th>
                                    <th>Validation</th>
                                    <th>Data Points</th>
                                    <th>Time Periods</th>
                                    <th>Duration</th>
                                    <th>Completed</th>
                                </tr>
                            </thead>
                            <tbody id="training-latest-body"></tbody>
                        </table>
                    </div>
                </div>

                <!-- Recent Training Runs -->
                <div class="card">
                    <div class="card-header">
                        <div class="card-title">
                            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <circle cx="12" cy="12" r="10"></circle>
                                <polyline points="12 6 12 12 16 14"></polyline>
                            </svg>
                            Recent Training Runs
                        </div>
                        <select class="form-select" id="training-filter-service" onchange="loadTrainingRuns()" style="width: 200px;">
                            <option value="">All Services</option>
                        </select>
                    </div>
                    <div class="card-body" style="padding: 0;">
                        <table class="services-table" id="training-runs-table">
                            <thead>
                                <tr>
                                    <th>Run ID</th>
                                    <th>Service</th>
                                    <th>Status</th>
                                    <th>Validation</th>
                                    <th>Passed/Warned/Failed</th>
                                    <th>Duration</th>
                                    <th>Started</th>
                                    <th>Details</th>
                                </tr>
                            </thead>
                            <tbody id="training-runs-body"></tbody>
                        </table>
                    </div>
                </div>

            </div>

            <!-- Categories Page -->
            <div class="page" id="page-categories">
                <div class="page-header">
                    <div class="page-header-text">
                        <h1 class="page-title">Service Categories</h1>
                        <p class="page-description">Manage categories and their default contamination rates</p>
                    </div>
                    <div class="page-actions">
                        <button class="btn btn-primary" onclick="openAddCategoryModal()">+ Add Category</button>
                    </div>
                </div>

                <div class="help-panel">
                    <div class="help-panel-title">ℹ️ About Categories</div>
                    <div class="help-panel-text">
                        Categories group services with similar characteristics. Each category has a default <strong>contamination rate</strong> - the expected proportion of anomalies.
                        Lower rates = stricter detection. Services inherit their category's rate unless overridden.
                    </div>
                </div>

                <div class="card">
                    <div class="card-body" style="padding: 0;">
                        <table class="services-table">
                            <thead>
                                <tr>
                                    <th>Category Name</th>
                                    <th>Description</th>
                                    <th>Default Contamination</th>
                                    <th>Services</th>
                                    <th>Actions</th>
                                </tr>
                            </thead>
                            <tbody id="categories-table-body"></tbody>
                        </table>
                    </div>
                </div>
            </div>

            <!-- Services Page -->
            <div class="page" id="page-services">
                <div class="page-header">
                    <div class="page-header-text">
                        <h1 class="page-title">Services Management</h1>
                        <p class="page-description">Manage services, categories, and contamination rates</p>
                    </div>
                    <div class="page-actions">
                        <button class="btn btn-primary" onclick="openAddServiceModal()">+ Add Service</button>
                    </div>
                </div>
                <div class="help-panel">
                    <div class="help-panel-title">ℹ️ About Services</div>
                    <div class="help-panel-text">
                        Each service belongs to a <strong>category</strong> which determines its default contamination rate.
                        You can override the rate per-service, or manage categories on the <a href="#" onclick="navigateTo('categories'); return false;" style="color: var(--accent-blue);">Categories page</a>.
                    </div>
                </div>

                <div class="search-box">
                    <svg class="search-icon" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <circle cx="11" cy="11" r="8"></circle>
                        <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
                    </svg>
                    <input type="text" id="services-search" placeholder="Search services..." oninput="filterServices()">
                </div>
                <div class="tabs" id="services-tabs">
                    <!-- Tabs populated dynamically by renderServicesTabs() -->
                </div>
                <div class="card">
                    <div class="card-body" style="padding: 0;">
                        <table class="services-table">
                            <thead>
                                <tr>
                                    <th>Service Name</th>
                                    <th>Category</th>
                                    <th>Contamination Rate</th>
                                    <th>Custom SLO</th>
                                    <th>Actions</th>
                                </tr>
                            </thead>
                            <tbody id="services-table-body"></tbody>
                        </table>
                    </div>
                </div>
            </div>

            <!-- SLOs Page -->
            <div class="page" id="page-slos">
                <div class="page-header">
                    <div class="page-header-text">
                        <h1 class="page-title">SLO Thresholds</h1>
                        <p class="page-description">Configure Service Level Objective thresholds for severity evaluation</p>
                    </div>
                    <div class="page-actions">
                        <button class="btn btn-primary" onclick="saveSLOConfig()">Save Changes</button>
                    </div>
                </div>
                <div class="help-panel">
                    <div class="help-panel-title">ℹ️ How SLO Evaluation Works</div>
                    <div class="help-panel-text">
                        SLO evaluation adds operational context to ML-detected anomalies. An anomaly within acceptable thresholds becomes <strong>low</strong> severity (logged but not urgent), while SLO breaches are escalated to <strong>critical</strong> regardless of ML severity.
                        <br><br><strong>Severity Matrix:</strong> Anomaly + OK → <span style="color: var(--status-success)">low</span> | Anomaly + Warning → <span style="color: var(--status-warning)">high</span> | SLO Breach → <span style="color: var(--status-critical)">critical</span>
                    </div>
                </div>
                <div class="grid-2">
                    <div class="card">
                        <div class="card-header">
                            <div class="card-title">SLO Configuration</div>
                            <label class="toggle">
                                <input type="checkbox" id="slo-enabled" onchange="markUnsaved()">
                                <span class="toggle-slider"></span>
                            </label>
                        </div>
                        <div class="card-body">
                            <div class="form-group">
                                <label class="form-label">Allow Downgrade to Low Severity</label>
                                <label class="toggle">
                                    <input type="checkbox" id="slo-allow-downgrade" onchange="markUnsaved()">
                                    <span class="toggle-slider"></span>
                                </label>
                                <p style="font-size: 10px; color: var(--text-muted); margin-top: 4px;">When enabled, anomalies within acceptable SLO thresholds are downgraded to "low" severity</p>
                            </div>
                            <div class="form-group">
                                <label class="form-label">Require SLO Breach for Critical</label>
                                <label class="toggle">
                                    <input type="checkbox" id="slo-require-breach" onchange="markUnsaved()">
                                    <span class="toggle-slider"></span>
                                </label>
                                <p style="font-size: 10px; color: var(--text-muted); margin-top: 4px;">When enabled, "critical" severity only assigned when SLO is actually breached</p>
                            </div>
                        </div>
                    </div>
                    <div class="card">
                        <div class="card-header">
                            <div class="card-title">Default Thresholds</div>
                        </div>
                        <div class="card-body">
                            <div class="grid-3">
                                <div class="form-group">
                                    <label class="form-label">Latency OK (ms)</label>
                                    <input type="number" class="form-input" id="slo-default-latency-acceptable" oninput="markUnsaved(); validateSLOInputs()">
                                </div>
                                <div class="form-group">
                                    <label class="form-label">Latency Warn (ms)</label>
                                    <input type="number" class="form-input" id="slo-default-latency-warning" oninput="markUnsaved(); validateSLOInputs()">
                                </div>
                                <div class="form-group">
                                    <label class="form-label">Latency Crit (ms)</label>
                                    <input type="number" class="form-input" id="slo-default-latency-critical" oninput="markUnsaved(); validateSLOInputs()">
                                </div>
                            </div>
                            <div class="slo-bar" id="slo-bar-latency">
                                <div class="slo-segment slo-segment-ok" style="width: 50%"></div>
                                <div class="slo-segment slo-segment-warning" style="width: 30%"></div>
                                <div class="slo-segment slo-segment-critical" style="width: 20%"></div>
                            </div>
                            <div class="grid-4" style="margin-top: var(--space-md);">
                                <div class="form-group">
                                    <label class="form-label">Error OK (%)</label>
                                    <input type="number" step="0.01" class="form-input" id="slo-default-error-acceptable" oninput="markUnsaved()">
                                </div>
                                <div class="form-group">
                                    <label class="form-label">Error Warn (%)</label>
                                    <input type="number" step="0.01" class="form-input" id="slo-default-error-warning" oninput="markUnsaved()">
                                </div>
                                <div class="form-group">
                                    <label class="form-label">Error Crit (%)</label>
                                    <input type="number" step="0.01" class="form-input" id="slo-default-error-critical" oninput="markUnsaved()">
                                </div>
                                <div class="form-group">
                                    <label class="form-label">Error Floor (%)</label>
                                    <input type="number" step="0.01" class="form-input" id="slo-default-error-floor" oninput="markUnsaved()">
                                </div>
                            </div>
                            <p style="font-size: 10px; color: var(--text-muted); margin-top: 4px;">
                                <strong>Error Floor:</strong> Suppress error anomalies when rate is below this (0 = use OK threshold). Filters operationally insignificant alerts.
                            </p>
                            <div class="grid-2" style="margin-top: var(--space-md);">
                                <div class="form-group">
                                    <label class="form-label">Min Traffic (req/s)</label>
                                    <input type="number" step="0.1" class="form-input" id="slo-default-min-traffic" oninput="markUnsaved()">
                                </div>
                                <div class="form-group">
                                    <label class="form-label">Busy Period Factor</label>
                                    <input type="number" step="0.1" class="form-input" id="slo-default-busy-factor" oninput="markUnsaved()">
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                <div class="section-divider"></div>
                <div class="card">
                    <div class="card-header">
                        <div class="card-title">Service-Specific SLOs</div>
                        <button class="btn btn-primary btn-sm" onclick="openAddSLOModal()">+ Add Service SLO</button>
                    </div>
                    <div class="card-body" style="padding: 0;">
                        <table class="services-table">
                            <thead>
                                <tr>
                                    <th>Service</th>
                                    <th>Latency (OK/Warn/Crit)</th>
                                    <th>Error Rate (OK/Warn/Crit)</th>
                                    <th>Actions</th>
                                </tr>
                            </thead>
                            <tbody id="slo-services-body"></tbody>
                        </table>
                    </div>
                </div>
                <div class="section-divider"></div>
                <div class="card">
                    <div class="card-header">
                        <div class="card-title">Busy Periods</div>
                        <button class="btn btn-primary btn-sm" onclick="openAddBusyPeriodModal()">+ Add Busy Period</button>
                    </div>
                    <div class="card-body" id="busy-periods-container"></div>
                </div>

                <div class="section-divider"></div>
                <div class="grid-2">
                    <!-- Database Latency Ratios -->
                    <div class="card">
                        <div class="card-header">
                            <div class="card-title">
                                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                    <ellipse cx="12" cy="5" rx="9" ry="3"></ellipse>
                                    <path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"></path>
                                    <path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"></path>
                                </svg>
                                Database Latency Ratios
                            </div>
                        </div>
                        <div class="card-body">
                            <p style="color: var(--text-secondary); font-size: 12px; margin-bottom: var(--space-md);">
                                Severity based on ratio to training baseline. Values below floor are always OK.
                            </p>
                            <div class="grid-4" style="gap: var(--space-sm);">
                                <div class="form-group">
                                    <label class="form-label" style="color: var(--status-success)">Info (×)</label>
                                    <input type="number" step="0.1" class="form-input" id="db-ratio-info" oninput="markUnsaved()" style="text-align: center;">
                                </div>
                                <div class="form-group">
                                    <label class="form-label" style="color: var(--status-warning)">Warning (×)</label>
                                    <input type="number" step="0.1" class="form-input" id="db-ratio-warning" oninput="markUnsaved()" style="text-align: center;">
                                </div>
                                <div class="form-group">
                                    <label class="form-label" style="color: #ff9500">High (×)</label>
                                    <input type="number" step="0.1" class="form-input" id="db-ratio-high" oninput="markUnsaved()" style="text-align: center;">
                                </div>
                                <div class="form-group">
                                    <label class="form-label" style="color: var(--status-critical)">Critical (×)</label>
                                    <input type="number" step="0.1" class="form-input" id="db-ratio-critical" oninput="markUnsaved()" style="text-align: center;">
                                </div>
                            </div>
                            <div class="form-group" style="margin-top: var(--space-md);">
                                <label class="form-label">Floor (minimum ms)</label>
                                <input type="number" step="0.1" class="form-input" id="db-latency-floor" oninput="markUnsaved()" style="width: 100px;">
                                <p style="font-size: 10px; color: var(--text-muted); margin-top: 4px;">Latency below this value is always considered OK (noise filter)</p>
                            </div>
                        </div>
                    </div>

                    <!-- Request Rate SLO -->
                    <div class="card">
                        <div class="card-header">
                            <div class="card-title">
                                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                    <polyline points="23 6 13.5 15.5 8.5 10.5 1 18"></polyline>
                                    <polyline points="17 6 23 6 23 12"></polyline>
                                </svg>
                                Request Rate Detection
                            </div>
                        </div>
                        <div class="card-body">
                            <div class="request-rate-grid">
                                <div class="request-rate-section">
                                    <div class="request-rate-title">
                                        <span class="surge-icon">📈</span> Surge Detection
                                    </div>
                                    <div class="form-group">
                                        <label class="form-label">Surge Threshold (× baseline)</label>
                                        <input type="number" step="0.1" class="form-input" id="surge-threshold" value="2.0" oninput="markUnsaved()">
                                    </div>
                                    <p style="font-size: 11px; color: var(--text-muted);">
                                        Surge alone → informational<br>
                                        Surge + latency breach → warning<br>
                                        Surge + error breach → high
                                    </p>
                                </div>
                                <div class="request-rate-section">
                                    <div class="request-rate-title">
                                        <span class="cliff-icon">📉</span> Cliff Detection
                                    </div>
                                    <div class="form-group">
                                        <label class="form-label">Cliff Threshold (× baseline)</label>
                                        <input type="number" step="0.1" class="form-input" id="cliff-threshold" value="0.3" oninput="markUnsaved()">
                                    </div>
                                    <p style="font-size: 11px; color: var(--text-muted);">
                                        Cliff off-peak → warning<br>
                                        Cliff peak hours → high<br>
                                        Cliff + errors → critical
                                    </p>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Dependencies Page -->
            <div class="page" id="page-dependencies">
                <div class="page-header">
                    <div class="page-header-text">
                        <h1 class="page-title">Service Dependencies</h1>
                        <p class="page-description">Visualize and manage service dependency graph for cascade detection</p>
                    </div>
                </div>
                <div class="help-panel">
                    <div class="help-panel-title">ℹ️ Cascade Detection</div>
                    <div class="help-panel-text">
                        When a service has latency issues and its dependencies also show anomalies, the system detects this as a cascade pattern.
                        This helps identify root causes vs symptoms in complex service meshes.
                    </div>
                </div>
                <div class="card">
                    <div class="card-header">
                        <div class="card-title">Dependency Graph</div>
                    </div>
                    <div class="card-body">
                        <div class="dep-graph" id="dep-graph"></div>
                    </div>
                </div>
                <div class="section-divider"></div>
                <div class="card">
                    <div class="card-header">
                        <div class="card-title">Dependencies Table</div>
                    </div>
                    <div class="card-body" style="padding: 0;">
                        <table class="services-table">
                            <thead>
                                <tr>
                                    <th>Service</th>
                                    <th>Depends On</th>
                                    <th>Depended By</th>
                                </tr>
                            </thead>
                            <tbody id="dependencies-table-body"></tbody>
                        </table>
                    </div>
                </div>
            </div>

            <!-- Settings Page -->
            <div class="page" id="page-settings">
                <div class="page-header">
                    <div class="page-header-text">
                        <h1 class="page-title">System Settings</h1>
                        <p class="page-description">Configure VictoriaMetrics, fingerprinting, and inference settings</p>
                    </div>
                    <div class="page-actions">
                        <button class="btn btn-primary" onclick="saveSettings()">Save Settings</button>
                    </div>
                </div>
                <div class="grid-2">
                    <div class="card">
                        <div class="card-header"><div class="card-title">VictoriaMetrics Connection</div></div>
                        <div class="card-body">
                            <div class="form-group">
                                <label class="form-label">Endpoint URL</label>
                                <input type="text" class="form-input" id="vm-endpoint" oninput="markUnsaved()">
                            </div>
                            <div class="grid-2">
                                <div class="form-group">
                                    <label class="form-label">Timeout (seconds)</label>
                                    <input type="number" class="form-input" id="vm-timeout" oninput="markUnsaved()">
                                </div>
                                <div class="form-group">
                                    <label class="form-label">Max Retries</label>
                                    <input type="number" class="form-input" id="vm-retries" oninput="markUnsaved()">
                                </div>
                            </div>
                            <div class="grid-2">
                                <div class="form-group">
                                    <label class="form-label">Circuit Breaker Threshold</label>
                                    <input type="number" class="form-input" id="vm-cb-threshold" oninput="markUnsaved()">
                                </div>
                                <div class="form-group">
                                    <label class="form-label">CB Timeout (seconds)</label>
                                    <input type="number" class="form-input" id="vm-cb-timeout" oninput="markUnsaved()">
                                </div>
                            </div>
                        </div>
                    </div>
                    <div class="card">
                        <div class="card-header"><div class="card-title">Observability API</div></div>
                        <div class="card-body">
                            <div class="form-group">
                                <label class="form-label">Base URL</label>
                                <input type="text" class="form-input" id="obs-url" oninput="markUnsaved()">
                            </div>
                            <div class="grid-2">
                                <div class="form-group">
                                    <label class="form-label">Request Timeout (s)</label>
                                    <input type="number" class="form-input" id="obs-timeout" oninput="markUnsaved()">
                                </div>
                                <div class="form-group">
                                    <label class="form-label">API Enabled</label>
                                    <label class="toggle">
                                        <input type="checkbox" id="obs-enabled" onchange="markUnsaved()">
                                        <span class="toggle-slider"></span>
                                    </label>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                <div class="section-divider"></div>
                <div class="grid-2">
                    <div class="card">
                        <div class="card-header"><div class="card-title">Fingerprinting (Incident Tracking)</div></div>
                        <div class="card-body">
                            <div class="form-group">
                                <label class="form-label">Database Path</label>
                                <input type="text" class="form-input" id="fp-db-path" oninput="markUnsaved()">
                            </div>
                            <div class="grid-2">
                                <div class="form-group">
                                    <label class="form-label">Confirmation Cycles</label>
                                    <input type="number" class="form-input" id="fp-confirm-cycles" oninput="markUnsaved()">
                                </div>
                                <div class="form-group">
                                    <label class="form-label">Resolution Grace Cycles</label>
                                    <input type="number" class="form-input" id="fp-grace-cycles" oninput="markUnsaved()">
                                </div>
                            </div>
                            <div class="grid-2">
                                <div class="form-group">
                                    <label class="form-label">Incident Separation (min)</label>
                                    <input type="number" class="form-input" id="fp-separation" oninput="markUnsaved()">
                                </div>
                                <div class="form-group">
                                    <label class="form-label">Cleanup Age (hours)</label>
                                    <input type="number" class="form-input" id="fp-cleanup" oninput="markUnsaved()">
                                </div>
                            </div>
                        </div>
                    </div>
                    <div class="card">
                        <div class="card-header"><div class="card-title">Inference Settings</div></div>
                        <div class="card-body">
                            <div class="form-group">
                                <label class="form-label">Alerts Directory</label>
                                <input type="text" class="form-input" id="inf-alerts-dir" oninput="markUnsaved()">
                            </div>
                            <div class="grid-2">
                                <div class="form-group">
                                    <label class="form-label">Max Workers</label>
                                    <input type="number" class="form-input" id="inf-workers" oninput="markUnsaved()">
                                </div>
                                <div class="form-group">
                                    <label class="form-label">Inter-Service Delay (s)</label>
                                    <input type="number" step="0.1" class="form-input" id="inf-delay" oninput="markUnsaved()">
                                </div>
                            </div>
                            <div class="form-group">
                                <label class="form-label">Enable Drift Detection</label>
                                <label class="toggle">
                                    <input type="checkbox" id="inf-drift" onchange="markUnsaved()">
                                    <span class="toggle-slider"></span>
                                </label>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Documentation Page -->
            <div class="page" id="page-docs">
                <div class="page-header">
                    <div class="page-header-text">
                        <h1 class="page-title">Documentation</h1>
                        <p class="page-description">Inference pipeline, configuration reference, and system behavior</p>
                    </div>
                </div>

                <!-- Inference Pipeline -->
                <div class="card" style="margin-bottom: var(--space-lg);">
                    <div class="card-header"><div class="card-title">🔮 Inference Pipeline</div></div>
                    <div class="card-body">
                        <p style="color: var(--text-secondary); margin-bottom: var(--space-md);">
                            The inference pipeline runs every few minutes for each monitored service. Each stage transforms or enriches the detection result.
                        </p>
                        <div class="code-block" style="font-size: 11px;">┌───────────────┐    ┌────────────────┐    ┌──────────────────┐    ┌────────────────┐    ┌─────────┐
│   DETECTION   │───►│ SLO EVALUATION │───►│ EXCEPTION ENRICH │───►│ SERVICE GRAPH  │───►│  ALERT  │
│ (ML Patterns) │    │  (Thresholds)  │    │ (On Error Breach)│    │(On Lat Breach) │    │(Finger) │
└───────────────┘    └────────────────┘    └──────────────────┘    └────────────────┘    └─────────┘
        │                    │                      │                       │
        ▼                    ▼                      ▼                       ▼
• Time-aware models    • Latency SLO         • events_total query    • service_graph metrics
• Pattern matching     • Error Rate SLO      • exception_type        • client→server routes
• Training baselines   • DB Latency Ratios   • Breakdown by rate     • Latency per route
• Improvement filter   • Request Rate SLO    • API enrichment        • Top traffic routes</div>
                        <div style="margin-top: var(--space-md); font-size: 12px; color: var(--text-secondary);">
                            <strong>Stage Details:</strong>
                            <ul style="margin-top: var(--space-xs); padding-left: 20px;">
                                <li><strong>Detection:</strong> Isolation Forest ML models + named pattern matching identify anomalies</li>
                                <li><strong>SLO Evaluation:</strong> Adjusts severity based on operational thresholds (can downgrade or upgrade)</li>
                                <li><strong>Exception Enrichment:</strong> Queries exception types when error SLO is breached</li>
                                <li><strong>Service Graph:</strong> Queries downstream dependencies when latency SLO is breached</li>
                                <li><strong>Fingerprinting:</strong> Tracks incident lifecycle (SUSPECTED → OPEN → RECOVERING → CLOSED)</li>
                            </ul>
                        </div>
                    </div>
                </div>

                <!-- Core Metrics -->
                <div class="card" style="margin-bottom: var(--space-lg);">
                    <div class="card-header"><div class="card-title">📈 Core Metrics</div></div>
                    <div class="card-body" style="padding: 0;">
                        <table class="services-table">
                            <thead><tr><th>Metric</th><th>Unit</th><th>Description</th><th>Lower is Better?</th></tr></thead>
                            <tbody>
                                <tr>
                                    <td><code>request_rate</code></td>
                                    <td>req/s</td>
                                    <td>Incoming requests per second. Measures traffic volume.</td>
                                    <td style="color: var(--text-muted)">No</td>
                                </tr>
                                <tr>
                                    <td><code>application_latency</code></td>
                                    <td>ms</td>
                                    <td>Server-side processing time. Total time to handle request.</td>
                                    <td style="color: var(--text-muted)">Context-dependent*</td>
                                </tr>
                                <tr>
                                    <td><code>dependency_latency</code></td>
                                    <td>ms</td>
                                    <td>Time spent waiting on external services/dependencies.</td>
                                    <td style="color: var(--status-success)">Yes</td>
                                </tr>
                                <tr>
                                    <td><code>database_latency</code></td>
                                    <td>ms</td>
                                    <td>Time spent on database operations.</td>
                                    <td style="color: var(--status-success)">Yes</td>
                                </tr>
                                <tr>
                                    <td><code>error_rate</code></td>
                                    <td>ratio</td>
                                    <td>Fraction of failed requests (0.05 = 5%).</td>
                                    <td style="color: var(--status-success)">Yes</td>
                                </tr>
                            </tbody>
                        </table>
                        <p style="color: var(--text-muted); font-size: 11px; padding: var(--space-sm) var(--space-md);">
                            * <code>application_latency</code> is not marked "lower is better" because very low latency + high errors indicates fast-fail patterns (circuit breaker, rate limiting).
                        </p>
                    </div>
                </div>

                <!-- Time Periods -->
                <div class="card" style="margin-bottom: var(--space-lg);">
                    <div class="card-header"><div class="card-title">🕐 Time Periods</div></div>
                    <div class="card-body">
                        <p style="color: var(--text-secondary); margin-bottom: var(--space-md);">
                            Services behave differently at different times. Yaga2 trains separate ML models for each time period to reduce false positives.
                        </p>
                        <div class="grid-2">
                            <table class="services-table" style="margin: 0;">
                                <thead><tr><th>Period</th><th>Hours</th><th>Days</th></tr></thead>
                                <tbody>
                                    <tr><td><code>business_hours</code></td><td>08:00 - 18:00</td><td>Mon-Fri</td></tr>
                                    <tr><td><code>evening_hours</code></td><td>18:00 - 22:00</td><td>Mon-Fri</td></tr>
                                    <tr><td><code>night_hours</code></td><td>22:00 - 06:00</td><td>Mon-Fri</td></tr>
                                    <tr><td><code>weekend_day</code></td><td>08:00 - 22:00</td><td>Sat-Sun</td></tr>
                                    <tr><td><code>weekend_night</code></td><td>22:00 - 08:00</td><td>Sat-Sun</td></tr>
                                </tbody>
                            </table>
                            <div style="font-size: 12px; color: var(--text-secondary);">
                                <strong>Why time-aware models?</strong>
                                <ul style="margin-top: var(--space-xs); padding-left: 20px;">
                                    <li>Traffic at 3 AM is naturally lower than 3 PM</li>
                                    <li>Weekend patterns differ from weekdays</li>
                                    <li>Prevents false "traffic cliff" alerts at night</li>
                                    <li>Each period has its own baseline statistics</li>
                                </ul>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Training Baselines -->
                <div class="card" style="margin-bottom: var(--space-lg);">
                    <div class="card-header"><div class="card-title">📊 Training Baselines</div></div>
                    <div class="card-body">
                        <p style="color: var(--text-secondary); margin-bottom: var(--space-md);">
                            Each time-period model stores training statistics from 30 days of historical data. These baselines are used for anomaly detection and SLO evaluation.
                        </p>
                        <div class="grid-2">
                            <div>
                                <strong style="font-size: 12px;">Statistics stored per metric:</strong>
                                <ul style="font-size: 12px; color: var(--text-secondary); margin-top: var(--space-xs); padding-left: 20px;">
                                    <li><code>mean</code> - Average value during training</li>
                                    <li><code>std</code> - Standard deviation</li>
                                    <li><code>p50, p90, p95, p99</code> - Percentiles</li>
                                    <li><code>min, max</code> - Range boundaries</li>
                                </ul>
                            </div>
                            <div>
                                <strong style="font-size: 12px;">Used for:</strong>
                                <ul style="font-size: 12px; color: var(--text-secondary); margin-top: var(--space-xs); padding-left: 20px;">
                                    <li>Database latency ratio thresholds (current/baseline)</li>
                                    <li>Request rate surge/cliff detection</li>
                                    <li>Contextual severity adjustment</li>
                                    <li>Pattern explanation generation</li>
                                    <li>Percentile position in alerts</li>
                                </ul>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Service Categories & Contamination -->
                <div class="grid-2" style="margin-bottom: var(--space-lg);">
                    <div class="card">
                        <div class="card-header"><div class="card-title">🏷️ Service Categories</div></div>
                        <div class="card-body" style="padding: 0;">
                            <table class="services-table">
                                <thead><tr><th>Category</th><th>Contamination</th><th>Use For</th></tr></thead>
                                <tbody>
                                    <tr><td><span class="category-badge category-critical">Critical</span></td><td><code>3%</code></td><td>Revenue-impacting (booking, checkout)</td></tr>
                                    <tr><td><span class="category-badge category-standard">Standard</span></td><td><code>5%</code></td><td>Production services</td></tr>
                                    <tr><td><span class="category-badge category-core">Core</span></td><td><code>4%</code></td><td>Platform infrastructure</td></tr>
                                    <tr><td><span class="category-badge category-admin">Admin</span></td><td><code>6%</code></td><td>Admin interfaces</td></tr>
                                    <tr><td><span class="category-badge category-micro">Micro</span></td><td><code>8%</code></td><td>Low-traffic utilities</td></tr>
                                </tbody>
                            </table>
                        </div>
                        <div style="padding: var(--space-sm) var(--space-md); font-size: 11px; color: var(--text-muted); border-top: 1px solid var(--border-subtle);">
                            <strong>Contamination</strong> = expected % of anomalies in training data. Lower = stricter detection, fewer false positives. Higher = more tolerant of variance.
                        </div>
                    </div>
                    <div class="card">
                        <div class="card-header"><div class="card-title">⚡ SLO Severity Matrix</div></div>
                        <div class="card-body" style="padding: 0;">
                            <table class="services-table">
                                <thead><tr><th></th><th>SLO OK</th><th>SLO Warning</th><th>SLO Breach</th></tr></thead>
                                <tbody>
                                    <tr><td><strong>Anomaly Detected</strong></td><td style="color: var(--status-success)">low</td><td style="color: var(--status-warning)">high</td><td style="color: var(--status-critical)">critical</td></tr>
                                    <tr><td><strong>No Anomaly</strong></td><td style="color: var(--text-muted)">—</td><td style="color: var(--status-warning)">warning</td><td style="color: var(--status-critical)">critical</td></tr>
                                </tbody>
                            </table>
                        </div>
                        <div style="padding: var(--space-sm) var(--space-md); font-size: 11px; color: var(--text-muted); border-top: 1px solid var(--border-subtle);">
                            ML detects statistical anomalies. SLO evaluation determines operational impact. An anomaly within acceptable SLO is <strong>low</strong> severity (logged but not urgent).
                        </div>
                    </div>
                </div>

                <!-- SLO Thresholds -->
                <div class="card" style="margin-bottom: var(--space-lg);">
                    <div class="card-header"><div class="card-title">🎯 SLO Thresholds</div></div>
                    <div class="card-body">
                        <p style="color: var(--text-secondary); margin-bottom: var(--space-md);">
                            SLO thresholds define operational boundaries. These can be customized per service in the SLOs page.
                        </p>
                        <div class="grid-2">
                            <div>
                                <strong style="font-size: 12px;">Latency Thresholds (default):</strong>
                                <table class="services-table" style="margin-top: var(--space-xs);">
                                    <tbody>
                                        <tr><td>Acceptable</td><td><code>&lt; 500ms</code></td><td style="color: var(--status-success)">OK</td></tr>
                                        <tr><td>Warning</td><td><code>500-800ms</code></td><td style="color: var(--status-warning)">Warning</td></tr>
                                        <tr><td>Critical</td><td><code>&gt; 800ms</code></td><td style="color: var(--status-critical)">Breach</td></tr>
                                    </tbody>
                                </table>
                            </div>
                            <div>
                                <strong style="font-size: 12px;">Error Rate Thresholds (default):</strong>
                                <table class="services-table" style="margin-top: var(--space-xs);">
                                    <tbody>
                                        <tr><td>Acceptable</td><td><code>&lt; 0.5%</code></td><td style="color: var(--status-success)">OK</td></tr>
                                        <tr><td>Warning</td><td><code>0.5-1%</code></td><td style="color: var(--status-warning)">Warning</td></tr>
                                        <tr><td>Critical</td><td><code>&gt; 1%</code></td><td style="color: var(--status-critical)">Breach</td></tr>
                                    </tbody>
                                </table>
                            </div>
                        </div>
                        <div style="margin-top: var(--space-md);">
                            <strong style="font-size: 12px;">Database Latency (ratio-based):</strong>
                            <p style="font-size: 11px; color: var(--text-muted); margin-top: var(--space-xs);">
                                Database latency uses ratio thresholds relative to training baseline mean:
                                <code>1.5x</code> = info, <code>2x</code> = warning, <code>3x</code> = high, <code>5x</code> = critical.
                                Values below noise floor (default 1ms) are always OK.
                            </p>
                        </div>
                    </div>
                </div>

                <!-- Named Patterns -->
                <div class="card" style="margin-bottom: var(--space-lg);">
                    <div class="card-header"><div class="card-title">🔍 Named Detection Patterns</div></div>
                    <div class="card-body" style="padding: 0;">
                        <table class="services-table">
                            <thead><tr><th>Pattern</th><th>Severity</th><th>Conditions</th><th>Meaning</th></tr></thead>
                            <tbody>
                                <tr><td><code>traffic_surge_failing</code></td><td style="color: var(--status-critical)">critical</td><td>High traffic + high latency + high errors</td><td>Service overwhelmed, users affected</td></tr>
                                <tr><td><code>traffic_cliff</code></td><td style="color: var(--status-critical)">critical</td><td>Very low traffic (sudden drop)</td><td>Upstream issue or service unreachable</td></tr>
                                <tr><td><code>error_rate_critical</code></td><td style="color: var(--status-critical)">critical</td><td>Very high errors, normal traffic/latency</td><td>Major failure affecting many requests</td></tr>
                                <tr><td><code>fast_rejection</code></td><td style="color: var(--status-critical)">critical</td><td>Very low latency + very high errors</td><td>Requests rejected before processing</td></tr>
                                <tr><td><code>latency_spike_recent</code></td><td style="color: var(--status-warning)">high</td><td>High latency, normal traffic/errors</td><td>Recent change caused slowdown</td></tr>
                                <tr><td><code>database_bottleneck</code></td><td style="color: var(--status-warning)">high</td><td>High DB latency, DB &gt;70% of total</td><td>Database is primary constraint</td></tr>
                                <tr><td><code>downstream_cascade</code></td><td style="color: var(--status-warning)">high</td><td>High dependency latency &gt;60% of total</td><td>External dependency is bottleneck</td></tr>
                                <tr><td><code>internal_latency_issue</code></td><td style="color: var(--status-warning)">high</td><td>High app latency, deps healthy</td><td>Issue is internal to service</td></tr>
                                <tr><td><code>traffic_surge_healthy</code></td><td style="color: var(--status-success)">low</td><td>High traffic, normal latency/errors</td><td>System handling load well</td></tr>
                            </tbody>
                        </table>
                    </div>
                </div>

                <!-- Exception Enrichment -->
                <div class="card" style="margin-bottom: var(--space-lg);">
                    <div class="card-header"><div class="card-title">🔍 Exception Enrichment</div></div>
                    <div class="card-body">
                        <p style="color: var(--text-secondary); margin-bottom: var(--space-md);">
                            When SLO evaluation confirms error rate is above threshold, Yaga2 queries VictoriaMetrics
                            for exception breakdown to enrich the alert with actionable context.
                        </p>
                        <div class="code-block" style="font-size: 11px;">Query: sum(rate(events_total{service_name="<SERVICE>",
                 deployment_environment_name=~"production"}[5m])) by (exception_type)

Trigger conditions:
  ✓ Anomaly severity = HIGH or CRITICAL
  ✓ Pattern is error-related (error, failure, outage)
  ✓ SLO error_rate_evaluation.status ≠ "ok"

Result: exception_context added to API payload with:
  • Top exception types by rate
  • Percentage breakdown
  • Short names for readability</div>
                    </div>
                </div>

                <!-- Service Graph Enrichment -->
                <div class="card" style="margin-bottom: var(--space-lg);">
                    <div class="card-header"><div class="card-title">🌐 Service Graph Enrichment</div></div>
                    <div class="card-body">
                        <p style="color: var(--text-secondary); margin-bottom: var(--space-md);">
                            When SLO evaluation confirms latency is above threshold, Yaga2 queries OpenTelemetry
                            service graph metrics to show which downstream services and routes are being called.
                        </p>
                        <div class="code-block" style="font-size: 11px;">Request Rate Query:
  sum(rate(traces_service_graph_request_total{client="<SERVICE>"}[5m]))
      by (client, server, server_http_route)

Latency Query:
  sum(rate(traces_service_graph_request_server_seconds_sum{client="<SERVICE>"}[5m]))
      by (client, server, server_http_route)
  / sum(rate(traces_service_graph_request_server_seconds_count{client="<SERVICE>"}[5m]))
      by (client, server, server_http_route)

Trigger conditions:
  ✓ SLO latency_evaluation.status ≠ "ok" (dependency_latency above threshold)
  ✓ Service is a client calling downstream services

Result: service_graph_context added to API payload with:
  • Downstream servers being called
  • HTTP routes with request rates
  • Average latency per route
  • Top route by traffic
  • Slowest route by latency</div>
                    </div>
                </div>

                <!-- Improvement Filtering -->
                <div class="card" style="margin-bottom: var(--space-lg);">
                    <div class="card-header"><div class="card-title">✨ Improvement Signal Filtering</div></div>
                    <div class="card-body">
                        <p style="color: var(--text-secondary); margin-bottom: var(--space-md);">
                            Lower-is-better metrics that show improvement (direction=low) are automatically filtered
                            and don't trigger alerts. This prevents false positives when performance improves.
                        </p>
                        <div class="improvement-metrics">
                            <span class="improvement-badge">dependency_latency ↓ = OK</span>
                            <span class="improvement-badge">database_latency ↓ = OK</span>
                            <span class="improvement-badge">error_rate ↓ = OK</span>
                        </div>
                        <p style="color: var(--text-muted); font-size: 11px; margin-top: var(--space-md);">
                            Example: Dependency latency drops from 50ms mean to 20ms → No alert (this is an improvement)
                        </p>
                    </div>
                </div>

                <div class="card" style="margin-bottom: var(--space-lg);">
                    <div class="card-header"><div class="card-title">🔄 Incident Lifecycle</div></div>
                    <div class="card-body">
                        <p style="color: var(--text-secondary); margin-bottom: var(--space-md);">
                            Incidents follow a state machine to reduce alert noise. First detection creates a SUSPECTED incident (no alert).
                            After confirmation, it becomes OPEN (alerts sent). Grace period prevents flapping.
                        </p>
                        <div class="code-block" style="font-size: 11px;">┌─────────────┐     confirmed     ┌──────────┐
│  SUSPECTED  │ ─────────────────►│   OPEN   │
│ (no alert)  │   (2 cycles)      │ (alerts) │
└─────────────┘                   └──────────┘
      ▲                                 │
      │                                 │ not detected (1-2 cycles)
      │           ┌─────────────┐       │
      └───────────│ RECOVERING  │◄──────┘
     re-detected  │  (waiting)  │
                  └─────────────┘
                        │
                        │ grace period (3 cycles)
                        ▼
                  ┌──────────┐
                  │  CLOSED  │ → Resolution sent to API
                  └──────────┘</div>
                        <div class="grid-2" style="margin-top: var(--space-md);">
                            <div>
                                <strong style="font-size: 12px;">State Descriptions:</strong>
                                <ul style="font-size: 12px; color: var(--text-secondary); margin-top: var(--space-xs); padding-left: 20px;">
                                    <li><strong>SUSPECTED:</strong> First detection, waiting for confirmation. No alert sent yet.</li>
                                    <li><strong>OPEN:</strong> Confirmed incident. Alerts are being sent to API.</li>
                                    <li><strong>RECOVERING:</strong> Not detected recently. May resolve or return.</li>
                                    <li><strong>CLOSED:</strong> Resolved. Resolution notification sent.</li>
                                </ul>
                            </div>
                            <div>
                                <strong style="font-size: 12px;">Default Timing:</strong>
                                <ul style="font-size: 12px; color: var(--text-secondary); margin-top: var(--space-xs); padding-left: 20px;">
                                    <li><strong>Confirmation:</strong> 2 consecutive detections (~4-6 min)</li>
                                    <li><strong>Grace period:</strong> 3 cycles without detection (~6-9 min)</li>
                                    <li><strong>Staleness:</strong> 30 min gap → new incident created</li>
                                    <li><strong>Cleanup:</strong> Closed incidents removed after 72 hours</li>
                                </ul>
                            </div>
                        </div>
                        <p style="color: var(--text-muted); font-size: 11px; margin-top: var(--space-md);">
                            <strong>Why confirmation?</strong> Prevents single-point false positives from triggering alerts.
                            <strong>Why grace period?</strong> Prevents flapping alerts when anomaly is intermittent.
                        </p>
                    </div>
                </div>

                <!-- API Payload Overview -->
                <div class="card">
                    <div class="card-header"><div class="card-title">📦 API Payload Structure</div></div>
                    <div class="card-body">
                        <p style="color: var(--text-secondary); margin-bottom: var(--space-md);">
                            Each inference cycle produces a JSON payload for the observability API. Key fields:
                        </p>
                        <div class="code-block" style="font-size: 11px;">{
  "alert_type": "anomaly_detected",     // or "no_anomaly", "incident_resolved"
  "service_name": "booking",
  "timestamp": "2026-01-15T10:30:00Z",
  "time_period": "business_hours",
  "overall_severity": "high",           // Adjusted by SLO evaluation
  "anomaly_count": 1,

  "anomalies": {
    "latency_spike_recent": {           // Named pattern
      "severity": "high",
      "confidence": 0.85,
      "description": "...",
      "root_metric": "application_latency",
      "fingerprint_id": "anomaly_abc123",
      "incident_id": "incident_xyz789",
      "status": "OPEN"
    }
  },

  "current_metrics": { ... },           // Raw metric values
  "slo_evaluation": { ... },            // SLO thresholds and status
  "exception_context": { ... },         // Exception breakdown (if error breach)
  "service_graph_context": { ... },     // Downstream deps (if latency breach)
  "fingerprinting": { ... }             // Incident lifecycle metadata
}</div>
                        <p style="color: var(--text-muted); font-size: 11px; margin-top: var(--space-md);">
                            Full payload specification: <code>docs/INFERENCE_API_PAYLOAD.md</code>
                        </p>
                    </div>
                </div>
            </div>

            <!-- Audit Log Page -->
            <div class="page" id="page-audit">
                <div class="page-header">
                    <div class="page-header-text">
                        <h1 class="page-title">Audit Log</h1>
                        <p class="page-description">Recent configuration changes</p>
                    </div>
                    <div class="page-actions">
                        <button class="btn btn-secondary" onclick="loadAuditLog()">Refresh</button>
                    </div>
                </div>
                <div class="card">
                    <div class="card-body">
                        <div class="code-block" id="audit-log-content" style="max-height: 500px; overflow-y: auto;">Loading...</div>
                    </div>
                </div>
            </div>
        </main>
    </div>

    <div class="modal-overlay" id="modal-overlay">
        <div class="modal" id="modal">
            <div class="modal-header">
                <h3 class="modal-title" id="modal-title">Modal Title</h3>
                <button class="modal-close" onclick="closeModal()">✕</button>
            </div>
            <div class="modal-body" id="modal-body"></div>
            <div class="modal-footer" id="modal-footer">
                <button class="btn btn-secondary" onclick="closeModal()">Cancel</button>
                <button class="btn btn-primary" id="modal-save">Save</button>
            </div>
        </div>
    </div>

    <div class="toast-container" id="toast-container"></div>

    <script>
        let config = {};
        let originalConfigHash = '';
        let currentCategory = 'all';
        let hasUnsavedChanges = false;
        const initialPage = '{{INITIAL_PAGE}}';

        // Category constants - must be defined before functions that use them
        const CATEGORY_DESCRIPTIONS = {
            critical: 'Revenue-critical, high-traffic services requiring strict detection',
            standard: 'Normal production services with balanced detection',
            core: 'Platform infrastructure that other services depend on',
            admin: 'Administrative and back-office tools',
            micro: 'Low-traffic microservices and utilities',
            background: 'Background workers, jobs, and queue processors'
        };

        const DEFAULT_CONTAMINATION = {
            critical: 0.03, standard: 0.05, core: 0.04, admin: 0.06, micro: 0.08, background: 0.08
        };

        document.addEventListener('DOMContentLoaded', async () => {
            await loadConfig();
            setupNavigation();
            // Show the initial page based on URL
            showPage(initialPage);
            // Load page-specific data
            if (initialPage === 'audit') loadAuditLog();
            if (initialPage === 'training') loadTrainingData();

            updateClock();
            setInterval(updateClock, 1000);
            checkHealth();
            setInterval(checkHealth, 30000);

            window.addEventListener('beforeunload', (e) => {
                if (hasUnsavedChanges) {
                    e.preventDefault();
                    e.returnValue = '';
                }
            });
        });

        function showLoading() { document.getElementById('loading-overlay').classList.add('active'); }
        function hideLoading() { document.getElementById('loading-overlay').classList.remove('active'); }

        async function loadConfig() {
            showLoading();
            try {
                const response = await fetch('/api/config');
                const data = await response.json();
                config = data.config;
                originalConfigHash = data.hash;
                document.getElementById('config-hash').textContent = data.hash;
                hasUnsavedChanges = false;
                updateUnsavedIndicator();
                renderAll();
            } catch (error) {
                showToast('Failed to load configuration', 'error');
            } finally {
                hideLoading();
            }
        }

        async function checkHealth() {
            try {
                const response = await fetch('/api/health');
                const data = await response.json();
                const dot = document.getElementById('health-dot');
                const text = document.getElementById('health-text');
                if (data.config_valid) {
                    dot.className = 'status-dot';
                    text.textContent = 'Healthy';
                } else {
                    dot.className = 'status-dot warning';
                    text.textContent = `${data.validation_issues} issues`;
                }
            } catch (error) {
                document.getElementById('health-dot').className = 'status-dot error';
                document.getElementById('health-text').textContent = 'Error';
            }
        }

        function markUnsaved() {
            hasUnsavedChanges = true;
            updateUnsavedIndicator();
        }

        function updateUnsavedIndicator() {
            const indicator = document.getElementById('unsaved-indicator');
            indicator.classList.toggle('active', hasUnsavedChanges);
        }

        function renderAll() {
            renderStats();
            renderPipelineStatus();
            renderSLOSummary();
            renderOverviewServices();
            renderServicesTable();
            renderCategories();
            renderSLOConfig();
            renderSettings();
            renderDependencies();
        }

        function renderPipelineStatus() {
            const sloEnabled = config.slos?.enabled ?? true;
            const sloIcon = document.getElementById('slo-pipeline-icon');
            const sloStatus = document.getElementById('slo-pipeline-status');

            if (sloEnabled) {
                sloIcon.style.background = 'var(--status-success-bg)';
                sloIcon.style.borderColor = 'var(--status-success)';
                sloStatus.textContent = 'Enabled';
            } else {
                sloIcon.classList.add('disabled');
                sloStatus.textContent = 'Disabled';
            }
        }

        function renderSLOSummary() {
            const defaults = config.slos?.defaults || {};
            const container = document.getElementById('slo-summary');

            container.innerHTML = `
                <div class="config-summary-item">
                    <span class="config-summary-label">Latency OK</span>
                    <span class="config-summary-value" style="color: var(--status-success)">${defaults.latency_acceptable_ms || 500}ms</span>
                </div>
                <div class="config-summary-item">
                    <span class="config-summary-label">Latency Crit</span>
                    <span class="config-summary-value" style="color: var(--status-critical)">${defaults.latency_critical_ms || 1000}ms</span>
                </div>
                <div class="config-summary-item">
                    <span class="config-summary-label">Error OK</span>
                    <span class="config-summary-value" style="color: var(--status-success)">${((defaults.error_rate_acceptable || 0.005) * 100).toFixed(1)}%</span>
                </div>
                <div class="config-summary-item">
                    <span class="config-summary-label">Error Crit</span>
                    <span class="config-summary-value" style="color: var(--status-critical)">${((defaults.error_rate_critical || 0.02) * 100).toFixed(1)}%</span>
                </div>
                <div class="config-summary-item">
                    <span class="config-summary-label">Min Traffic</span>
                    <span class="config-summary-value">${defaults.min_traffic_rps || 5} req/s</span>
                </div>
                <div class="config-summary-item">
                    <span class="config-summary-label">Busy Factor</span>
                    <span class="config-summary-value">${defaults.busy_period_factor || 1.5}×</span>
                </div>
            `;
        }

        function renderStats() {
            const services = config.services || {};
            const totalServices = Object.values(services).filter(v => Array.isArray(v)).reduce((acc, arr) => acc + arr.length, 0);
            const sloServices = Object.keys(config.slos?.services || {}).length;
            const deps = config.dependencies?.graph || {};
            const depsCount = Object.values(deps).reduce((acc, arr) => acc + arr.length, 0);

            document.getElementById('stats-grid').innerHTML = `
                <div class="stat-card">
                    <div class="stat-label">Total Services</div>
                    <div class="stat-value">${totalServices}</div>
                    <div class="stat-detail">Across ${Object.keys(services).filter(k => !k.includes('pattern')).length} categories</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Custom SLOs</div>
                    <div class="stat-value">${sloServices}</div>
                    <div class="stat-detail">${Math.round(sloServices/totalServices*100) || 0}% coverage</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">SLO Evaluation</div>
                    <div class="stat-value" style="color: ${config.slos?.enabled ? 'var(--status-success)' : 'var(--text-muted)'}">${config.slos?.enabled ? 'Enabled' : 'Disabled'}</div>
                    <div class="stat-detail">${config.slos?.busy_periods?.length || 0} busy periods</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Dependencies</div>
                    <div class="stat-value">${depsCount}</div>
                    <div class="stat-detail">${Object.keys(deps).length} services with deps</div>
                </div>
            `;
            document.getElementById('services-count').textContent = totalServices;

            // Validation issues
            const issuesContainer = document.getElementById('validation-issues-container');
            fetch('/api/config/validate').then(r => r.json()).then(data => {
                if (data.issues.length > 0) {
                    issuesContainer.innerHTML = `
                        <div class="validation-issues">
                            <div class="validation-issues-title">⚠️ Configuration Issues</div>
                            <ul>${data.issues.map(i => `<li>${i}</li>`).join('')}</ul>
                        </div>
                    `;
                } else {
                    issuesContainer.innerHTML = '';
                }
            });
        }

        function renderOverviewServices() {
            const services = config.services || {};
            const slos = config.slos || {};
            const contamination = config.model?.contamination_by_service || {};
            const categoryContamination = config.model?.contamination_by_category || {};
            const deps = config.dependencies?.graph || {};

            let html = '';
            for (const category of getCategories()) {
                for (const service of (services[category] || [])) {
                    const cont = contamination[service] || categoryContamination[category] || DEFAULT_CONTAMINATION[category] || 0.05;
                    const hasSlo = service in (slos.services || {});
                    const slo = slos.services?.[service] || slos.defaults || {};
                    const serviceDeps = deps[service] || [];

                    html += `
                        <tr data-service="${service.toLowerCase()}">
                            <td class="service-name">${service}</td>
                            <td><span class="category-badge category-${category}">${category}</span></td>
                            <td>${(cont * 100).toFixed(1)}%</td>
                            <td>${hasSlo ? '<span style="color: var(--status-success)">✓ Custom</span>' : '<span style="color: var(--text-muted)">Default</span>'}</td>
                            <td>
                                <span style="color: var(--status-success)">${slo.latency_acceptable_ms || 500}</span> /
                                <span style="color: var(--status-warning)">${slo.latency_warning_ms || 800}</span> /
                                <span style="color: var(--status-critical)">${slo.latency_critical_ms || 1000}</span> ms
                            </td>
                            <td>${serviceDeps.length > 0 ? serviceDeps.join(', ') : '—'}</td>
                        </tr>
                    `;
                }
            }
            document.getElementById('overview-services-body').innerHTML = html;
        }

        function filterOverviewServices() {
            const query = document.getElementById('overview-search').value.toLowerCase();
            document.querySelectorAll('#overview-services-body tr').forEach(row => {
                const service = row.dataset.service || '';
                row.classList.toggle('hidden', !service.includes(query));
            });
        }

        function filterServices() {
            const query = document.getElementById('services-search').value.toLowerCase();
            document.querySelectorAll('#services-table-body tr').forEach(row => {
                const service = row.dataset.service || '';
                row.classList.toggle('hidden', !service.includes(query));
            });
        }

        function renderServicesTabs() {
            const categories = getCategories();
            let html = `<div class="tab ${currentCategory === 'all' ? 'active' : ''}" data-category="all">All Services</div>`;
            for (const cat of categories) {
                html += `<div class="tab ${currentCategory === cat ? 'active' : ''}" data-category="${cat}">${cat.charAt(0).toUpperCase() + cat.slice(1)}</div>`;
            }
            const container = document.getElementById('services-tabs');
            container.innerHTML = html;

            // Re-attach tab click handlers
            container.querySelectorAll('.tab').forEach(tab => {
                tab.addEventListener('click', () => {
                    container.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
                    tab.classList.add('active');
                    currentCategory = tab.dataset.category;
                    renderServicesTableContent();
                });
            });
        }

        function renderServicesTableContent() {
            const services = config.services || {};
            const slos = config.slos || {};
            const contamination = config.model?.contamination_by_service || {};
            const categoryContamination = config.model?.contamination_by_category || {};

            let html = '';
            for (const category of getCategories()) {
                for (const service of (services[category] || [])) {
                    if (currentCategory !== 'all' && currentCategory !== category) continue;
                    const cont = contamination[service] || categoryContamination[category] || DEFAULT_CONTAMINATION[category] || 0.05;
                    const hasSlo = service in (slos.services || {});

                    html += `
                        <tr data-service="${service.toLowerCase()}">
                            <td class="service-name">${service}</td>
                            <td><span class="category-badge category-${category}">${category}</span></td>
                            <td>${(cont * 100).toFixed(1)}%</td>
                            <td>${hasSlo ? '<span style="color: var(--status-success)">✓</span>' : '—'}</td>
                            <td>
                                <button class="btn btn-secondary btn-sm" onclick="editService('${service}', '${category}')">Edit</button>
                                <button class="btn btn-danger btn-sm" onclick="deleteService('${service}', '${category}')">Remove</button>
                            </td>
                        </tr>
                    `;
                }
            }
            document.getElementById('services-table-body').innerHTML = html || '<tr><td colspan="5" class="empty-state">No services in this category</td></tr>';
        }

        function renderServicesTable() {
            renderServicesTabs();
            renderServicesTableContent();
        }

        function getCategories() {
            // Get all unique categories from services and contamination config
            // Filter to only include keys that have array values (exclude pattern_detection, etc.)
            const categories = new Set();
            const services = config.services || {};
            Object.keys(services).filter(k => Array.isArray(services[k])).forEach(cat => categories.add(cat));
            Object.keys(config.model?.contamination_by_category || {}).forEach(cat => categories.add(cat));
            // Ensure default categories exist
            ['critical', 'standard', 'core', 'admin', 'micro', 'background'].forEach(cat => categories.add(cat));
            return Array.from(categories).sort();
        }

        function renderCategories() {
            const categories = getCategories();
            const services = config.services || {};
            const contamination = config.model?.contamination_by_category || {};

            let html = '';
            for (const cat of categories) {
                const serviceCount = (services[cat] || []).length;
                const cont = contamination[cat] || DEFAULT_CONTAMINATION[cat] || 0.05;
                const desc = CATEGORY_DESCRIPTIONS[cat] || 'Custom category';
                const isBuiltIn = ['critical', 'standard', 'core', 'admin', 'micro', 'background'].includes(cat);

                html += `
                    <tr>
                        <td>
                            <span class="category-badge category-${cat}">${cat}</span>
                            ${isBuiltIn ? '' : '<span style="font-size: 10px; color: var(--text-muted); margin-left: 8px;">custom</span>'}
                        </td>
                        <td style="color: var(--text-secondary); font-size: 13px;">${desc}</td>
                        <td><strong>${(cont * 100).toFixed(1)}%</strong></td>
                        <td>${serviceCount} service${serviceCount !== 1 ? 's' : ''}</td>
                        <td>
                            <button class="btn btn-secondary btn-sm" onclick="editCategory('${cat}')">Edit</button>
                            ${!isBuiltIn && serviceCount === 0 ? `<button class="btn btn-danger btn-sm" onclick="deleteCategory('${cat}')">Delete</button>` : ''}
                        </td>
                    </tr>
                `;
            }
            document.getElementById('categories-table-body').innerHTML = html;
            document.getElementById('categories-count').textContent = categories.length;
        }

        function openAddCategoryModal() {
            const body = `
                <div class="form-group">
                    <label class="form-label">Category Name</label>
                    <input type="text" class="form-input" id="new-category-name" placeholder="e.g., gateway" pattern="^[a-z][a-z0-9_-]*$">
                    <p style="font-size: 10px; color: var(--text-muted); margin-top: 4px;">Lowercase letters, numbers, hyphens, underscores only</p>
                </div>
                <div class="form-group">
                    <label class="form-label">Description</label>
                    <input type="text" class="form-input" id="new-category-desc" placeholder="e.g., API gateway services">
                </div>
                <div class="form-group">
                    <label class="form-label">Default Contamination Rate (%)</label>
                    <input type="number" step="0.1" min="0.5" max="20" class="form-input" id="new-category-cont" value="5.0">
                    <p style="font-size: 10px; color: var(--text-muted); margin-top: 4px;">Lower = stricter detection. Typical: critical 3%, standard 5%, micro 8%</p>
                </div>
            `;
            openModal('Add Category', body, addCategory);
        }

        async function addCategory() {
            const name = document.getElementById('new-category-name').value.trim().toLowerCase();
            const desc = document.getElementById('new-category-desc').value.trim();
            const cont = parseFloat(document.getElementById('new-category-cont').value) / 100;

            if (!name) { showToast('Category name is required', 'error'); return; }
            if (!/^[a-z][a-z0-9_-]*$/.test(name)) { showToast('Invalid category name format', 'error'); return; }
            if (getCategories().includes(name)) { showToast('Category already exists', 'error'); return; }

            // Initialize category
            if (!config.services) config.services = {};
            if (!config.services[name]) config.services[name] = [];
            if (!config.model) config.model = {};
            if (!config.model.contamination_by_category) config.model.contamination_by_category = {};
            config.model.contamination_by_category[name] = cont;

            // Store description in a new config section if provided
            if (desc) {
                if (!config.category_descriptions) config.category_descriptions = {};
                config.category_descriptions[name] = desc;
                CATEGORY_DESCRIPTIONS[name] = desc;
            }

            await saveFullConfig(`Added category: ${name}`);
            closeModal();
            renderAll();
        }

        function editCategory(name) {
            const cont = (config.model?.contamination_by_category?.[name] || DEFAULT_CONTAMINATION[name] || 0.05) * 100;
            const desc = config.category_descriptions?.[name] || CATEGORY_DESCRIPTIONS[name] || '';
            const isBuiltIn = ['critical', 'standard', 'core', 'admin', 'micro', 'background'].includes(name);

            const body = `
                <div class="form-group">
                    <label class="form-label">Category Name</label>
                    <input type="text" class="form-input" value="${name}" readonly style="opacity: 0.7">
                </div>
                <div class="form-group">
                    <label class="form-label">Description</label>
                    <input type="text" class="form-input" id="edit-category-desc" value="${desc}" ${isBuiltIn ? 'readonly style="opacity: 0.7"' : ''}>
                    ${isBuiltIn ? '<p style="font-size: 10px; color: var(--text-muted); margin-top: 4px;">Built-in category description cannot be changed</p>' : ''}
                </div>
                <div class="form-group">
                    <label class="form-label">Default Contamination Rate (%)</label>
                    <input type="number" step="0.1" min="0.5" max="20" class="form-input" id="edit-category-cont" value="${cont.toFixed(1)}">
                    <p style="font-size: 10px; color: var(--text-muted); margin-top: 4px;">This affects all services in this category without custom overrides</p>
                </div>
            `;
            openModal(`Edit Category: ${name}`, body, () => updateCategory(name));
        }

        async function updateCategory(name) {
            const cont = parseFloat(document.getElementById('edit-category-cont').value) / 100;
            const desc = document.getElementById('edit-category-desc').value.trim();
            const isBuiltIn = ['critical', 'standard', 'core', 'admin', 'micro', 'background'].includes(name);

            if (!config.model) config.model = {};
            if (!config.model.contamination_by_category) config.model.contamination_by_category = {};
            config.model.contamination_by_category[name] = cont;

            if (!isBuiltIn && desc) {
                if (!config.category_descriptions) config.category_descriptions = {};
                config.category_descriptions[name] = desc;
                CATEGORY_DESCRIPTIONS[name] = desc;
            }

            await saveFullConfig(`Updated category: ${name}`);
            closeModal();
            renderAll();
        }

        async function deleteCategory(name) {
            const serviceCount = (config.services?.[name] || []).length;
            if (serviceCount > 0) {
                showToast(`Cannot delete category with ${serviceCount} services. Move services first.`, 'error');
                return;
            }
            if (!confirm(`Delete category "${name}"? This cannot be undone.`)) return;

            if (config.services?.[name]) delete config.services[name];
            if (config.model?.contamination_by_category?.[name]) delete config.model.contamination_by_category[name];
            if (config.category_descriptions?.[name]) delete config.category_descriptions[name];

            await saveFullConfig(`Deleted category: ${name}`);
            renderAll();
        }

        function getCategoryOptions(selected = '') {
            return getCategories().map(c =>
                `<option value="${c}" ${c === selected ? 'selected' : ''}>${c.charAt(0).toUpperCase() + c.slice(1)}</option>`
            ).join('');
        }

        function renderSLOConfig() {
            const slos = config.slos || {};
            document.getElementById('slo-enabled').checked = slos.enabled || false;
            document.getElementById('slo-allow-downgrade').checked = slos.allow_downgrade_to_informational !== false;
            document.getElementById('slo-require-breach').checked = slos.require_slo_breach_for_critical !== false;

            const defaults = slos.defaults || {};
            document.getElementById('slo-default-latency-acceptable').value = defaults.latency_acceptable_ms || 500;
            document.getElementById('slo-default-latency-warning').value = defaults.latency_warning_ms || 800;
            document.getElementById('slo-default-latency-critical').value = defaults.latency_critical_ms || 1000;
            document.getElementById('slo-default-error-acceptable').value = (defaults.error_rate_acceptable || 0.005) * 100;
            document.getElementById('slo-default-error-warning').value = (defaults.error_rate_warning || 0.01) * 100;
            document.getElementById('slo-default-error-critical').value = (defaults.error_rate_critical || 0.02) * 100;
            document.getElementById('slo-default-error-floor').value = (defaults.error_rate_floor || 0) * 100;
            document.getElementById('slo-default-min-traffic').value = defaults.min_traffic_rps || 1.0;
            document.getElementById('slo-default-busy-factor').value = defaults.busy_period_factor || 1.5;

            // Database latency ratios
            const dbRatios = defaults.database_latency_ratios || {};
            document.getElementById('db-ratio-info').value = dbRatios.info || 1.5;
            document.getElementById('db-ratio-warning').value = dbRatios.warning || 2.0;
            document.getElementById('db-ratio-high').value = dbRatios.high || 3.0;
            document.getElementById('db-ratio-critical').value = dbRatios.critical || 5.0;
            document.getElementById('db-latency-floor').value = defaults.database_latency_floor_ms || 1.0;

            // Request rate thresholds
            document.getElementById('surge-threshold').value = defaults.request_rate_surge_threshold || 2.0;
            document.getElementById('cliff-threshold').value = defaults.request_rate_cliff_threshold || 0.3;

            updateSLOBar();

            let sloHtml = '';
            for (const [service, slo] of Object.entries(slos.services || {})) {
                sloHtml += `
                    <tr>
                        <td class="service-name">${service}</td>
                        <td>
                            <span style="color: var(--status-success)">${slo.latency_acceptable_ms || 'def'}</span> /
                            <span style="color: var(--status-warning)">${slo.latency_warning_ms || 'def'}</span> /
                            <span style="color: var(--status-critical)">${slo.latency_critical_ms || 'def'}</span> ms
                        </td>
                        <td>
                            <span style="color: var(--status-success)">${((slo.error_rate_acceptable || 0.005) * 100).toFixed(2)}%</span> /
                            <span style="color: var(--status-warning)">${((slo.error_rate_warning || 0.01) * 100).toFixed(2)}%</span> /
                            <span style="color: var(--status-critical)">${((slo.error_rate_critical || 0.02) * 100).toFixed(2)}%</span>
                        </td>
                        <td>
                            <button class="btn btn-secondary btn-sm" onclick="editSLO('${service}')">Edit</button>
                            <button class="btn btn-danger btn-sm" onclick="deleteSLO('${service}')">Remove</button>
                        </td>
                    </tr>
                `;
            }
            document.getElementById('slo-services-body').innerHTML = sloHtml || '<tr><td colspan="4" class="empty-state">No service-specific SLOs configured</td></tr>';

            let bpHtml = '';
            for (let i = 0; i < (slos.busy_periods || []).length; i++) {
                const bp = slos.busy_periods[i];
                bpHtml += `
                    <div class="busy-period-card">
                        <div class="busy-period-dates">
                            <strong>${new Date(bp.start).toLocaleDateString()}</strong> → <strong>${new Date(bp.end).toLocaleDateString()}</strong>
                        </div>
                        <button class="btn btn-danger btn-sm" onclick="deleteBusyPeriod(${i})">Remove</button>
                    </div>
                `;
            }
            document.getElementById('busy-periods-container').innerHTML = bpHtml || '<div class="empty-state">No busy periods configured</div>';
        }

        function validateSLOInputs() {
            const ok = parseInt(document.getElementById('slo-default-latency-acceptable').value);
            const warn = parseInt(document.getElementById('slo-default-latency-warning').value);
            const crit = parseInt(document.getElementById('slo-default-latency-critical').value);

            const okEl = document.getElementById('slo-default-latency-acceptable');
            const warnEl = document.getElementById('slo-default-latency-warning');
            const critEl = document.getElementById('slo-default-latency-critical');

            okEl.classList.toggle('error', ok >= warn);
            warnEl.classList.toggle('error', warn >= crit || ok >= warn);
            critEl.classList.toggle('error', warn >= crit);

            updateSLOBar();
        }

        function updateSLOBar() {
            const ok = parseInt(document.getElementById('slo-default-latency-acceptable').value) || 500;
            const warn = parseInt(document.getElementById('slo-default-latency-warning').value) || 800;
            const crit = parseInt(document.getElementById('slo-default-latency-critical').value) || 1000;

            const total = crit;
            const okPct = (ok / total) * 100;
            const warnPct = ((warn - ok) / total) * 100;
            const critPct = ((crit - warn) / total) * 100;

            const bar = document.getElementById('slo-bar-latency');
            bar.innerHTML = `
                <div class="slo-segment slo-segment-ok" style="width: ${okPct}%"></div>
                <div class="slo-segment slo-segment-warning" style="width: ${warnPct}%"></div>
                <div class="slo-segment slo-segment-critical" style="width: ${critPct}%"></div>
            `;
        }

        function renderSettings() {
            const vm = config.victoria_metrics || {};
            document.getElementById('vm-endpoint').value = vm.endpoint || '';
            document.getElementById('vm-timeout').value = vm.timeout_seconds || 10;
            document.getElementById('vm-retries').value = vm.max_retries || 3;
            document.getElementById('vm-cb-threshold').value = vm.circuit_breaker_threshold || 5;
            document.getElementById('vm-cb-timeout').value = vm.circuit_breaker_timeout_seconds || 300;

            const obs = config.observability_api || {};
            document.getElementById('obs-url').value = obs.base_url || '';
            document.getElementById('obs-timeout').value = obs.request_timeout_seconds || 5;
            document.getElementById('obs-enabled').checked = obs.enabled !== false;

            const fp = config.fingerprinting || {};
            document.getElementById('fp-db-path').value = fp.db_path || './anomaly_state.db';
            document.getElementById('fp-confirm-cycles').value = fp.confirmation_cycles || 2;
            document.getElementById('fp-grace-cycles').value = fp.resolution_grace_cycles || 3;
            document.getElementById('fp-separation').value = fp.incident_separation_minutes || 30;
            document.getElementById('fp-cleanup').value = fp.cleanup_max_age_hours || 72;

            const inf = config.inference || {};
            document.getElementById('inf-alerts-dir').value = inf.alerts_directory || './alerts/';
            document.getElementById('inf-workers').value = inf.max_workers || 3;
            document.getElementById('inf-delay').value = inf.inter_service_delay_seconds || 0.2;
            document.getElementById('inf-drift').checked = inf.check_drift || false;
        }

        function renderDependencies() {
            const deps = config.dependencies?.graph || {};
            const services = getAllServices();

            // Table
            let html = '';
            for (const service of services) {
                const dependsOn = deps[service] || [];
                const dependedBy = Object.entries(deps).filter(([s, d]) => d.includes(service)).map(([s]) => s);

                html += `
                    <tr>
                        <td class="service-name">${service}</td>
                        <td>${dependsOn.length > 0 ? dependsOn.map(d => `<span class="category-badge category-standard">${d}</span>`).join(' ') : '—'}</td>
                        <td>${dependedBy.length > 0 ? dependedBy.map(d => `<span class="category-badge category-warning">${d}</span>`).join(' ') : '—'}</td>
                    </tr>
                `;
            }
            document.getElementById('dependencies-table-body').innerHTML = html;

            // Simple graph visualization
            const graphEl = document.getElementById('dep-graph');
            const nodes = services.slice(0, 15); // Limit for visual clarity
            const nodePositions = {};

            graphEl.innerHTML = '';
            nodes.forEach((service, i) => {
                const angle = (i / nodes.length) * 2 * Math.PI;
                const x = 180 + Math.cos(angle) * 150;
                const y = 180 + Math.sin(angle) * 150;
                nodePositions[service] = { x, y };

                const category = getServiceCategory(service);
                const node = document.createElement('div');
                node.className = `dep-graph-node ${category}`;
                node.style.left = x + 'px';
                node.style.top = y + 'px';
                node.textContent = service;
                node.title = `${service}\\nDeps: ${(deps[service] || []).join(', ') || 'none'}`;
                graphEl.appendChild(node);
            });
        }

        function getServiceCategory(service) {
            const services = config.services || {};
            for (const cat of ['critical', 'standard', 'core', 'admin', 'micro', 'background']) {
                if ((services[cat] || []).includes(service)) return cat;
            }
            return 'standard';
        }

        function setupNavigation() {
            document.querySelectorAll('.nav-item').forEach(item => {
                item.addEventListener('click', (e) => {
                    e.preventDefault();
                    const page = item.dataset.page;
                    if (page) {
                        navigateTo(page);
                    }
                });
            });
            // Services tabs are rendered dynamically by renderServicesTabs()
        }

        // Show page without URL navigation (used on initial load)
        function showPage(page) {
            document.querySelectorAll('.nav-item').forEach(i => i.classList.remove('active'));
            document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
            document.querySelector(`.nav-item[data-page="${page}"]`)?.classList.add('active');
            document.getElementById(`page-${page}`)?.classList.add('active');
        }

        // Navigate to page via URL (for clicks)
        function navigateTo(page) {
            // Map page names to URL paths
            let url;
            if (page === 'overview') {
                url = '/';
            } else if (page === 'docs') {
                url = '/documentation';
            } else {
                url = '/' + page;
            }
            window.location.href = url;
        }

        async function loadAuditLog() {
            try {
                const response = await fetch('/api/audit-log');
                const data = await response.json();
                document.getElementById('audit-log-content').textContent = data.entries.join('\\n') || 'No audit entries yet.';
            } catch (error) {
                document.getElementById('audit-log-content').textContent = 'Failed to load audit log.';
            }
        }

        function openModal(title, body, onSave) {
            document.getElementById('modal-title').textContent = title;
            document.getElementById('modal-body').innerHTML = body;
            document.getElementById('modal-save').onclick = onSave;
            document.getElementById('modal-overlay').classList.add('active');
        }

        function closeModal() {
            document.getElementById('modal-overlay').classList.remove('active');
        }

        // Service CRUD
        function openAddServiceModal() {
            const body = `
                <div class="form-group">
                    <label class="form-label">Service Name</label>
                    <input type="text" class="form-input" id="new-service-name" placeholder="e.g., my-new-service">
                </div>
                <div class="form-group">
                    <label class="form-label">Category</label>
                    <select class="form-select" id="new-service-category">
                        ${getCategoryOptions('standard')}
                    </select>
                </div>
                <div class="form-group">
                    <label class="form-label">Custom Contamination Rate (%)</label>
                    <input type="number" step="0.1" class="form-input" id="new-service-contamination" placeholder="Leave empty for category default">
                </div>
            `;
            openModal('Add Service', body, addService);
        }

        async function addService() {
            const name = document.getElementById('new-service-name').value.trim();
            const category = document.getElementById('new-service-category').value;
            const contamination = document.getElementById('new-service-contamination').value;

            if (!name) { showToast('Service name is required', 'error'); return; }

            if (!config.services[category]) config.services[category] = [];
            if (!config.services[category].includes(name)) {
                config.services[category].push(name);
            }

            if (contamination) {
                if (!config.model.contamination_by_service) config.model.contamination_by_service = {};
                config.model.contamination_by_service[name] = parseFloat(contamination) / 100;
            }

            await saveFullConfig(`Added service: ${name} to ${category}`);
            closeModal();
            renderAll();
        }

        function editService(name, category) {
            const cont = (config.model?.contamination_by_service?.[name] || config.model?.contamination_by_category?.[category] || 0.05) * 100;
            const body = `
                <div class="form-group">
                    <label class="form-label">Service Name</label>
                    <input type="text" class="form-input" value="${name}" readonly style="opacity: 0.7">
                </div>
                <div class="form-group">
                    <label class="form-label">Category</label>
                    <select class="form-select" id="edit-service-category">
                        ${getCategoryOptions(category)}
                    </select>
                </div>
                <div class="form-group">
                    <label class="form-label">Contamination Rate (%)</label>
                    <input type="number" step="0.1" class="form-input" id="edit-service-contamination" value="${cont.toFixed(1)}">
                </div>
            `;
            openModal('Edit Service', body, () => updateService(name, category));
        }

        async function updateService(oldName, oldCategory) {
            const newCategory = document.getElementById('edit-service-category').value;
            const contamination = document.getElementById('edit-service-contamination').value;

            if (newCategory !== oldCategory) {
                config.services[oldCategory] = config.services[oldCategory].filter(s => s !== oldName);
                if (!config.services[newCategory]) config.services[newCategory] = [];
                config.services[newCategory].push(oldName);
            }

            if (!config.model.contamination_by_service) config.model.contamination_by_service = {};
            config.model.contamination_by_service[oldName] = parseFloat(contamination) / 100;

            await saveFullConfig(`Updated service: ${oldName}`);
            closeModal();
            renderAll();
        }

        async function deleteService(name, category) {
            if (!confirm(`Remove ${name} from ${category} services?`)) return;
            config.services[category] = config.services[category].filter(s => s !== name);
            if (config.model?.contamination_by_service?.[name]) delete config.model.contamination_by_service[name];
            await saveFullConfig(`Deleted service: ${name}`);
            renderAll();
        }

        // SLO CRUD
        function openAddSLOModal() {
            const services = getAllServices();
            const existingSLOs = Object.keys(config.slos?.services || {});
            const available = services.filter(s => !existingSLOs.includes(s));

            if (available.length === 0) { showToast('All services already have custom SLOs', 'warning'); return; }

            const body = `
                <div class="form-group">
                    <label class="form-label">Service</label>
                    <select class="form-select" id="new-slo-service">
                        ${available.map(s => `<option value="${s}">${s}</option>`).join('')}
                    </select>
                </div>
                <div class="grid-3">
                    <div class="form-group"><label class="form-label">Latency OK (ms)</label><input type="number" class="form-input" id="new-slo-lat-accept" value="500"></div>
                    <div class="form-group"><label class="form-label">Latency Warn (ms)</label><input type="number" class="form-input" id="new-slo-lat-warn" value="800"></div>
                    <div class="form-group"><label class="form-label">Latency Crit (ms)</label><input type="number" class="form-input" id="new-slo-lat-crit" value="1000"></div>
                </div>
                <div class="grid-3">
                    <div class="form-group"><label class="form-label">Error OK (%)</label><input type="number" step="0.1" class="form-input" id="new-slo-err-accept" value="0.5"></div>
                    <div class="form-group"><label class="form-label">Error Warn (%)</label><input type="number" step="0.1" class="form-input" id="new-slo-err-warn" value="1.0"></div>
                    <div class="form-group"><label class="form-label">Error Crit (%)</label><input type="number" step="0.1" class="form-input" id="new-slo-err-crit" value="2.0"></div>
                </div>
            `;
            openModal('Add Service SLO', body, addSLO);
        }

        async function addSLO() {
            const service = document.getElementById('new-slo-service').value;
            if (!config.slos) config.slos = { enabled: true, services: {} };
            if (!config.slos.services) config.slos.services = {};

            config.slos.services[service] = {
                latency_acceptable_ms: parseInt(document.getElementById('new-slo-lat-accept').value),
                latency_warning_ms: parseInt(document.getElementById('new-slo-lat-warn').value),
                latency_critical_ms: parseInt(document.getElementById('new-slo-lat-crit').value),
                error_rate_acceptable: parseFloat(document.getElementById('new-slo-err-accept').value) / 100,
                error_rate_warning: parseFloat(document.getElementById('new-slo-err-warn').value) / 100,
                error_rate_critical: parseFloat(document.getElementById('new-slo-err-crit').value) / 100,
            };

            await saveFullConfig(`Added SLO for: ${service}`);
            closeModal();
            renderSLOConfig();
        }

        function editSLO(service) {
            const slo = config.slos?.services?.[service] || {};
            const body = `
                <div class="form-group"><label class="form-label">Service</label><input type="text" class="form-input" value="${service}" readonly style="opacity: 0.7"></div>
                <div class="grid-3">
                    <div class="form-group"><label class="form-label">Latency OK (ms)</label><input type="number" class="form-input" id="edit-slo-lat-accept" value="${slo.latency_acceptable_ms || 500}"></div>
                    <div class="form-group"><label class="form-label">Latency Warn (ms)</label><input type="number" class="form-input" id="edit-slo-lat-warn" value="${slo.latency_warning_ms || 800}"></div>
                    <div class="form-group"><label class="form-label">Latency Crit (ms)</label><input type="number" class="form-input" id="edit-slo-lat-crit" value="${slo.latency_critical_ms || 1000}"></div>
                </div>
                <div class="grid-3">
                    <div class="form-group"><label class="form-label">Error OK (%)</label><input type="number" step="0.1" class="form-input" id="edit-slo-err-accept" value="${((slo.error_rate_acceptable || 0.005) * 100).toFixed(2)}"></div>
                    <div class="form-group"><label class="form-label">Error Warn (%)</label><input type="number" step="0.1" class="form-input" id="edit-slo-err-warn" value="${((slo.error_rate_warning || 0.01) * 100).toFixed(2)}"></div>
                    <div class="form-group"><label class="form-label">Error Crit (%)</label><input type="number" step="0.1" class="form-input" id="edit-slo-err-crit" value="${((slo.error_rate_critical || 0.02) * 100).toFixed(2)}"></div>
                </div>
            `;
            openModal('Edit Service SLO', body, () => updateSLO(service));
        }

        async function updateSLO(service) {
            config.slos.services[service] = {
                latency_acceptable_ms: parseInt(document.getElementById('edit-slo-lat-accept').value),
                latency_warning_ms: parseInt(document.getElementById('edit-slo-lat-warn').value),
                latency_critical_ms: parseInt(document.getElementById('edit-slo-lat-crit').value),
                error_rate_acceptable: parseFloat(document.getElementById('edit-slo-err-accept').value) / 100,
                error_rate_warning: parseFloat(document.getElementById('edit-slo-err-warn').value) / 100,
                error_rate_critical: parseFloat(document.getElementById('edit-slo-err-crit').value) / 100,
            };
            await saveFullConfig(`Updated SLO for: ${service}`);
            closeModal();
            renderSLOConfig();
        }

        async function deleteSLO(service) {
            if (!confirm(`Remove custom SLO for ${service}?`)) return;
            delete config.slos.services[service];
            await saveFullConfig(`Deleted SLO for: ${service}`);
            renderSLOConfig();
        }

        function openAddBusyPeriodModal() {
            const body = `
                <div class="form-group"><label class="form-label">Start Date</label><input type="datetime-local" class="form-input" id="new-bp-start"></div>
                <div class="form-group"><label class="form-label">End Date</label><input type="datetime-local" class="form-input" id="new-bp-end"></div>
            `;
            openModal('Add Busy Period', body, addBusyPeriod);
        }

        async function addBusyPeriod() {
            const start = document.getElementById('new-bp-start').value;
            const end = document.getElementById('new-bp-end').value;
            if (!start || !end) { showToast('Dates are required', 'error'); return; }

            if (!config.slos.busy_periods) config.slos.busy_periods = [];
            config.slos.busy_periods.push({ start: new Date(start).toISOString(), end: new Date(end).toISOString() });

            await saveFullConfig('Added busy period');
            closeModal();
            renderSLOConfig();
        }

        async function deleteBusyPeriod(index) {
            if (!confirm('Remove this busy period?')) return;
            config.slos.busy_periods.splice(index, 1);
            await saveFullConfig('Deleted busy period');
            renderSLOConfig();
        }

        async function saveSLOConfig() {
            config.slos = {
                ...config.slos,
                enabled: document.getElementById('slo-enabled').checked,
                allow_downgrade_to_informational: document.getElementById('slo-allow-downgrade').checked,
                require_slo_breach_for_critical: document.getElementById('slo-require-breach').checked,
                defaults: {
                    latency_acceptable_ms: parseInt(document.getElementById('slo-default-latency-acceptable').value),
                    latency_warning_ms: parseInt(document.getElementById('slo-default-latency-warning').value),
                    latency_critical_ms: parseInt(document.getElementById('slo-default-latency-critical').value),
                    error_rate_acceptable: parseFloat(document.getElementById('slo-default-error-acceptable').value) / 100,
                    error_rate_warning: parseFloat(document.getElementById('slo-default-error-warning').value) / 100,
                    error_rate_critical: parseFloat(document.getElementById('slo-default-error-critical').value) / 100,
                    error_rate_floor: parseFloat(document.getElementById('slo-default-error-floor').value) / 100,
                    min_traffic_rps: parseFloat(document.getElementById('slo-default-min-traffic').value),
                    busy_period_factor: parseFloat(document.getElementById('slo-default-busy-factor').value),
                    database_latency_floor_ms: parseFloat(document.getElementById('db-latency-floor').value),
                    database_latency_ratios: {
                        info: parseFloat(document.getElementById('db-ratio-info').value),
                        warning: parseFloat(document.getElementById('db-ratio-warning').value),
                        high: parseFloat(document.getElementById('db-ratio-high').value),
                        critical: parseFloat(document.getElementById('db-ratio-critical').value),
                    },
                    request_rate_surge_threshold: parseFloat(document.getElementById('surge-threshold').value),
                    request_rate_cliff_threshold: parseFloat(document.getElementById('cliff-threshold').value),
                }
            };
            await saveFullConfig('Updated SLO configuration');
        }

        async function saveSettings() {
            config.victoria_metrics = { ...config.victoria_metrics, endpoint: document.getElementById('vm-endpoint').value, timeout_seconds: parseInt(document.getElementById('vm-timeout').value), max_retries: parseInt(document.getElementById('vm-retries').value), circuit_breaker_threshold: parseInt(document.getElementById('vm-cb-threshold').value), circuit_breaker_timeout_seconds: parseInt(document.getElementById('vm-cb-timeout').value) };
            config.observability_api = { ...config.observability_api, base_url: document.getElementById('obs-url').value, request_timeout_seconds: parseInt(document.getElementById('obs-timeout').value), enabled: document.getElementById('obs-enabled').checked };
            config.fingerprinting = { ...config.fingerprinting, db_path: document.getElementById('fp-db-path').value, confirmation_cycles: parseInt(document.getElementById('fp-confirm-cycles').value), resolution_grace_cycles: parseInt(document.getElementById('fp-grace-cycles').value), incident_separation_minutes: parseInt(document.getElementById('fp-separation').value), cleanup_max_age_hours: parseInt(document.getElementById('fp-cleanup').value) };
            config.inference = { ...config.inference, alerts_directory: document.getElementById('inf-alerts-dir').value, max_workers: parseInt(document.getElementById('inf-workers').value), inter_service_delay_seconds: parseFloat(document.getElementById('inf-delay').value), check_drift: document.getElementById('inf-drift').checked };
            await saveFullConfig('Updated system settings');
        }

        async function saveFullConfig(description = 'Manual update') {
            showLoading();
            try {
                const response = await fetch('/api/config', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ config, description })
                });
                const data = await response.json();

                if (response.ok) {
                    originalConfigHash = data.hash;
                    document.getElementById('config-hash').textContent = data.hash;
                    hasUnsavedChanges = false;
                    updateUnsavedIndicator();
                    showToast('Configuration saved', 'success');
                    checkHealth();
                } else if (data.issues) {
                    showToast(`Validation failed: ${data.issues[0]}`, 'error');
                } else {
                    throw new Error('Save failed');
                }
            } catch (error) {
                showToast('Failed to save configuration', 'error');
            } finally {
                hideLoading();
            }
        }

        function exportConfig() {
            window.location.href = '/api/config/export';
        }

        function openImportModal() {
            const body = `
                <div class="form-group">
                    <label class="form-label">Paste configuration JSON or upload file</label>
                    <textarea class="form-input" id="import-json" rows="10" style="font-family: 'JetBrains Mono', monospace; font-size: 12px;"></textarea>
                </div>
                <div class="form-group">
                    <input type="file" id="import-file" accept=".json" onchange="handleFileImport(event)">
                </div>
            `;
            openModal('Import Configuration', body, importConfig);
        }

        function handleFileImport(event) {
            const file = event.target.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = (e) => {
                    document.getElementById('import-json').value = e.target.result;
                };
                reader.readAsText(file);
            }
        }

        async function importConfig() {
            try {
                const json = document.getElementById('import-json').value;
                const newConfig = JSON.parse(json);
                config = newConfig;
                await saveFullConfig('Imported configuration');
                closeModal();
                renderAll();
            } catch (error) {
                showToast('Invalid JSON format', 'error');
            }
        }

        function getAllServices() {
            const services = config.services || {};
            const all = [];
            for (const cat of ['critical', 'standard', 'core', 'admin', 'micro', 'background']) {
                all.push(...(services[cat] || []));
            }
            return all;
        }

        function showToast(message, type = 'info') {
            const container = document.getElementById('toast-container');
            const toast = document.createElement('div');
            toast.className = `toast ${type}`;
            toast.innerHTML = message;
            container.appendChild(toast);
            setTimeout(() => { toast.style.opacity = '0'; setTimeout(() => toast.remove(), 300); }, 3000);
        }

        function updateClock() {
            document.getElementById('clock').textContent = new Date().toLocaleTimeString();
        }

        // Training Report Functions
        async function loadTrainingData() {
            try {
                // Load summary stats
                const summaryResponse = await fetch('/api/training/summary');
                if (summaryResponse.ok) {
                    const summary = await summaryResponse.json();
                    document.getElementById('training-total-runs').textContent = summary.total_runs || 0;
                    document.getElementById('training-passed').textContent = summary.passed || 0;
                    document.getElementById('training-warnings').textContent = summary.warnings || 0;
                    document.getElementById('training-failed').textContent = summary.failed || 0;
                }

                // Load latest runs per service
                const latestResponse = await fetch('/api/training/latest');
                if (latestResponse.ok) {
                    const latest = await latestResponse.json();
                    renderLatestTrainingTable(latest.runs || []);
                    populateServiceFilter(latest.runs || []);
                }

                // Load recent runs
                await loadTrainingRuns();
            } catch (error) {
                console.error('Failed to load training data:', error);
                showToast('Failed to load training data', 'error');
            }
        }

        function renderLatestTrainingTable(runs) {
            const tbody = document.getElementById('training-latest-body');
            if (!tbody) return;

            if (runs.length === 0) {
                tbody.innerHTML = '<tr><td colspan="7" style="text-align: center; color: var(--text-muted);">No training runs recorded yet</td></tr>';
                return;
            }

            tbody.innerHTML = runs.map(run => {
                const runStatusClass = getRunStatusClass(run.status);
                const validationClass = getValidationStatusClass(run.validation_status);
                const validationIcon = getValidationStatusIcon(run.validation_status);
                const timePeriods = run.time_periods_trained || 0;
                const dataPoints = formatNumber(run.total_data_points || 0);
                const duration = formatDuration(run.duration_seconds);
                const date = formatDateTime(run.completed_at || run.started_at);

                return `
                    <tr onclick="showTrainingDetails('${run.run_id}')" style="cursor: pointer;">
                        <td class="service-name">${escapeHtml(run.service_name)}</td>
                        <td><span class="status-badge ${runStatusClass}">${run.status || 'UNKNOWN'}</span></td>
                        <td><span class="status-badge ${validationClass}">${validationIcon} ${run.validation_status || 'N/A'}</span></td>
                        <td>${dataPoints}</td>
                        <td>${timePeriods}</td>
                        <td>${duration}</td>
                        <td>${date}</td>
                    </tr>
                `;
            }).join('');
        }

        async function loadTrainingRuns(serviceFilter = null) {
            try {
                // Get filter from select element if not provided
                if (serviceFilter === null) {
                    const select = document.getElementById('training-filter-service');
                    serviceFilter = select ? select.value : '';
                }

                let url = '/api/training/runs?limit=50';
                if (serviceFilter) {
                    url += `&service=${encodeURIComponent(serviceFilter)}`;
                }

                const response = await fetch(url);
                if (response.ok) {
                    const data = await response.json();
                    renderRecentTrainingTable(data.runs || []);
                }
            } catch (error) {
                console.error('Failed to load training runs:', error);
            }
        }

        function renderRecentTrainingTable(runs) {
            const tbody = document.getElementById('training-runs-body');
            if (!tbody) return;

            if (runs.length === 0) {
                tbody.innerHTML = '<tr><td colspan="8" style="text-align: center; color: var(--text-muted);">No training runs found</td></tr>';
                return;
            }

            tbody.innerHTML = runs.map(run => {
                const validationClass = getValidationStatusClass(run.validation_status);
                const validationIcon = getValidationStatusIcon(run.validation_status);
                const runStatusClass = getRunStatusClass(run.status);
                const duration = formatDuration(run.duration_seconds);
                const date = formatDateTime(run.started_at);
                const passedWarnedFailed = `${run.periods_passed || 0}/${run.periods_warned || 0}/${run.periods_failed || 0}`;
                const shortRunId = run.run_id ? run.run_id.substring(0, 12) : '—';

                return `
                    <tr onclick="showTrainingDetails('${run.run_id}')" style="cursor: pointer;">
                        <td style="font-family: var(--font-mono); font-size: 11px;">${shortRunId}</td>
                        <td class="service-name">${escapeHtml(run.service_name)}</td>
                        <td><span class="status-badge ${runStatusClass}">${run.status || 'UNKNOWN'}</span></td>
                        <td><span class="status-badge ${validationClass}">${validationIcon} ${run.validation_status || 'N/A'}</span></td>
                        <td style="font-family: var(--font-mono);">${passedWarnedFailed}</td>
                        <td>${duration}</td>
                        <td>${date}</td>
                        <td>
                            <button class="btn btn-sm" onclick="event.stopPropagation(); showTrainingDetails('${run.run_id}')">
                                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                    <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"></path>
                                    <circle cx="12" cy="12" r="3"></circle>
                                </svg>
                            </button>
                        </td>
                    </tr>
                `;
            }).join('');
        }

        function populateServiceFilter(runs) {
            const select = document.getElementById('training-filter-service');
            if (!select) return;

            const services = [...new Set(runs.map(r => r.service_name))].sort();
            select.innerHTML = '<option value="">All Services</option>' +
                services.map(s => `<option value="${escapeHtml(s)}">${escapeHtml(s)}</option>`).join('');
        }

        function filterTrainingRuns() {
            const select = document.getElementById('training-filter-service');
            const service = select ? select.value : '';
            loadTrainingRuns(service);
        }

        function showTrainingDetails(runId) {
            // Navigate to dedicated training run details page
            window.location.href = '/training/' + runId;
        }

        // Training helper functions
        function getValidationStatusClass(status) {
            switch (status) {
                case 'PASSED': return 'status-passed';
                case 'WARNING': return 'status-warning';
                case 'FAILED': return 'status-failed';
                default: return 'status-unknown';
            }
        }

        function getValidationStatusIcon(status) {
            switch (status) {
                case 'PASSED': return '✓';
                case 'WARNING': return '⚠';
                case 'FAILED': return '✗';
                default: return '?';
            }
        }

        function getRunStatusClass(status) {
            switch (status) {
                case 'COMPLETED': return 'standard';
                case 'RUNNING': return 'warning';
                case 'FAILED': return 'critical';
                default: return 'standard';
            }
        }

        function formatNumber(num) {
            if (num >= 1000000) return (num / 1000000).toFixed(1) + 'M';
            if (num >= 1000) return (num / 1000).toFixed(1) + 'K';
            return num.toString();
        }

        function formatDuration(seconds) {
            if (!seconds) return '—';
            if (seconds < 60) return `${Math.round(seconds)}s`;
            if (seconds < 3600) return `${Math.round(seconds / 60)}m`;
            return `${Math.round(seconds / 3600)}h ${Math.round((seconds % 3600) / 60)}m`;
        }

        function formatDate(isoString) {
            if (!isoString) return '—';
            const date = new Date(isoString);
            return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
        }

        function formatDateTime(isoString) {
            if (!isoString) return '—';
            const date = new Date(isoString);
            return date.toLocaleString('en-US', { month: 'short', day: 'numeric', year: 'numeric', hour: '2-digit', minute: '2-digit', second: '2-digit' });
        }

        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }

        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') closeModal();
            if ((e.ctrlKey || e.metaKey) && e.key === 's') { e.preventDefault(); saveFullConfig('Saved via keyboard shortcut'); }
        });
    </script>
</body>
</html>
"""


TRAINING_RUN_PAGE_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Training Run Details — Yaga2</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-primary: #0a0e14;
            --bg-secondary: #0d1117;
            --bg-tertiary: #161b22;
            --bg-elevated: #1c2128;
            --text-primary: #e6edf3;
            --text-secondary: #8b949e;
            --text-muted: #6e7681;
            --border-color: #30363d;
            --status-critical: #f85149;
            --status-critical-bg: rgba(248, 81, 73, 0.15);
            --status-warning: #d29922;
            --status-warning-bg: rgba(210, 153, 34, 0.15);
            --status-success: #3fb950;
            --status-success-bg: rgba(63, 185, 80, 0.15);
            --status-info: #58a6ff;
            --status-info-bg: rgba(88, 166, 255, 0.15);
            --accent: #79c0ff;
        }

        * { box-sizing: border-box; margin: 0; padding: 0; }

        body {
            font-family: 'Space Grotesk', -apple-system, BlinkMacSystemFont, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.6;
            min-height: 100vh;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 32px 24px;
        }

        /* Back link */
        .back-link {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            color: var(--text-muted);
            text-decoration: none;
            font-size: 14px;
            margin-bottom: 24px;
            transition: color 0.2s;
        }
        .back-link:hover { color: var(--accent); }

        /* Hero section */
        .hero {
            background: linear-gradient(135deg, var(--bg-secondary) 0%, var(--bg-tertiary) 100%);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 32px;
            margin-bottom: 32px;
        }

        .hero-header {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            margin-bottom: 24px;
        }

        .hero-title {
            font-size: 32px;
            font-weight: 700;
            letter-spacing: -0.02em;
        }

        .hero-subtitle {
            color: var(--text-muted);
            font-size: 14px;
            margin-top: 4px;
        }

        .status-pill {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 8px 16px;
            border-radius: 24px;
            font-weight: 600;
            font-size: 14px;
        }

        .status-pill.passed {
            background: var(--status-success-bg);
            color: var(--status-success);
            border: 1px solid var(--status-success);
        }

        .status-pill.warning {
            background: var(--status-warning-bg);
            color: var(--status-warning);
            border: 1px solid var(--status-warning);
        }

        .status-pill.failed {
            background: var(--status-critical-bg);
            color: var(--status-critical);
            border: 1px solid var(--status-critical);
        }

        .status-pill.running {
            background: var(--status-info-bg);
            color: var(--status-info);
            border: 1px solid var(--status-info);
        }

        /* Stats row */
        .stats-row {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 24px;
        }

        .stat-item {
            text-align: center;
        }

        .stat-value {
            font-size: 28px;
            font-weight: 700;
            font-family: 'JetBrains Mono', monospace;
            color: var(--text-primary);
        }

        .stat-label {
            font-size: 12px;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-top: 4px;
        }

        /* Section */
        .section {
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            margin-bottom: 24px;
            overflow: hidden;
        }

        .section-header {
            padding: 16px 24px;
            border-bottom: 1px solid var(--border-color);
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .section-title {
            font-size: 16px;
            font-weight: 600;
        }

        .section-badge {
            font-size: 12px;
            padding: 4px 10px;
            border-radius: 12px;
            background: var(--bg-tertiary);
            color: var(--text-muted);
        }

        /* Time period rows */
        .period-row {
            display: grid;
            grid-template-columns: 200px 1fr 140px;
            gap: 24px;
            padding: 20px 24px;
            border-bottom: 1px solid var(--border-color);
            align-items: center;
            transition: background 0.2s;
        }

        .period-row:last-child { border-bottom: none; }
        .period-row:hover { background: var(--bg-tertiary); }

        .period-name {
            font-weight: 600;
            font-size: 15px;
        }

        .period-details {
            display: flex;
            gap: 32px;
            font-size: 13px;
            color: var(--text-secondary);
        }

        .period-detail {
            display: flex;
            align-items: center;
            gap: 6px;
        }

        .period-detail-label { color: var(--text-muted); }
        .period-detail-value { color: var(--text-primary); font-family: 'JetBrains Mono', monospace; }

        .period-status {
            text-align: right;
        }

        .period-badge {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 6px 12px;
            border-radius: 6px;
            font-size: 12px;
            font-weight: 600;
        }

        .period-badge.passed {
            background: var(--status-success-bg);
            color: var(--status-success);
        }

        .period-badge.failed {
            background: var(--status-critical-bg);
            color: var(--status-critical);
        }

        .period-badge.warning {
            background: var(--status-warning-bg);
            color: var(--status-warning);
        }

        .period-badge.skipped {
            background: var(--bg-tertiary);
            color: var(--text-muted);
        }

        /* Failure details */
        .failure-details {
            grid-column: 1 / -1;
            background: var(--status-critical-bg);
            border: 1px solid rgba(248, 81, 73, 0.3);
            border-radius: 8px;
            padding: 16px;
            margin-top: 12px;
        }

        .failure-title {
            font-size: 12px;
            font-weight: 600;
            color: var(--status-critical);
            margin-bottom: 8px;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .failure-list {
            list-style: none;
            padding: 0;
            margin: 0;
        }

        .failure-list li {
            font-size: 13px;
            color: var(--text-primary);
            padding: 6px 0;
            padding-left: 20px;
            position: relative;
        }

        .failure-list li::before {
            content: "×";
            position: absolute;
            left: 0;
            color: var(--status-critical);
            font-weight: bold;
        }

        /* Info grid */
        .info-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 24px;
            padding: 24px;
        }

        .info-item label {
            display: block;
            font-size: 12px;
            color: var(--text-muted);
            margin-bottom: 4px;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .info-item span {
            font-size: 14px;
            color: var(--text-primary);
        }

        /* Error section */
        .error-section {
            background: var(--status-critical-bg);
            border: 1px solid rgba(248, 81, 73, 0.3);
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 24px;
        }

        .error-title {
            font-size: 16px;
            font-weight: 600;
            color: var(--status-critical);
            margin-bottom: 12px;
        }

        .error-message {
            background: var(--bg-primary);
            border-radius: 8px;
            padding: 16px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 13px;
            color: var(--text-primary);
            white-space: pre-wrap;
            overflow-x: auto;
        }

        .error-traceback {
            margin-top: 16px;
        }

        .error-traceback summary {
            cursor: pointer;
            color: var(--text-muted);
            font-size: 13px;
        }

        .error-traceback pre {
            background: var(--bg-primary);
            border-radius: 8px;
            padding: 16px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 11px;
            color: var(--text-muted);
            white-space: pre-wrap;
            overflow-x: auto;
            margin-top: 8px;
        }

        /* Loading */
        .loading {
            text-align: center;
            padding: 80px 24px;
            color: var(--text-muted);
        }

        .loading-spinner {
            width: 40px;
            height: 40px;
            border: 3px solid var(--border-color);
            border-top-color: var(--accent);
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin: 0 auto 16px;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        /* Data quality section */
        .quality-bar {
            height: 8px;
            background: var(--bg-tertiary);
            border-radius: 4px;
            overflow: hidden;
            margin-top: 8px;
        }

        .quality-fill {
            height: 100%;
            border-radius: 4px;
            transition: width 0.3s;
        }

        .quality-fill.good { background: var(--status-success); }
        .quality-fill.warning { background: var(--status-warning); }
        .quality-fill.poor { background: var(--status-critical); }

        /* Timeline */
        .timeline {
            padding: 24px;
        }

        .timeline-item {
            display: flex;
            gap: 16px;
            padding-bottom: 20px;
            position: relative;
        }

        .timeline-item:last-child { padding-bottom: 0; }

        .timeline-item::before {
            content: "";
            position: absolute;
            left: 11px;
            top: 28px;
            bottom: 0;
            width: 2px;
            background: var(--border-color);
        }

        .timeline-item:last-child::before { display: none; }

        .timeline-dot {
            width: 24px;
            height: 24px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 12px;
            flex-shrink: 0;
            z-index: 1;
        }

        .timeline-dot.success {
            background: var(--status-success-bg);
            color: var(--status-success);
            border: 2px solid var(--status-success);
        }

        .timeline-dot.info {
            background: var(--status-info-bg);
            color: var(--status-info);
            border: 2px solid var(--status-info);
        }

        .timeline-content {
            flex: 1;
        }

        .timeline-title {
            font-weight: 600;
            font-size: 14px;
        }

        .timeline-desc {
            font-size: 13px;
            color: var(--text-muted);
            margin-top: 2px;
        }

        @media (max-width: 768px) {
            .stats-row { grid-template-columns: repeat(2, 1fr); }
            .period-row { grid-template-columns: 1fr; gap: 12px; }
            .period-status { text-align: left; }
            .info-grid { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>
    <div class="container">
        <a href="/#training" class="back-link">
            ← Back to Dashboard
        </a>

        <div id="content">
            <div class="loading">
                <div class="loading-spinner"></div>
                <div>Loading training run details...</div>
            </div>
        </div>
    </div>

    <script>
        const RUN_ID = '{run_id}';

        function formatNumber(n) {
            if (n === null || n === undefined) return '—';
            if (n >= 1000000) return (n / 1000000).toFixed(1) + 'M';
            if (n >= 1000) return (n / 1000).toFixed(1) + 'K';
            return n.toLocaleString();
        }

        function formatDuration(seconds) {
            if (!seconds) return '—';
            if (seconds < 60) return Math.round(seconds) + 's';
            if (seconds < 3600) return Math.round(seconds / 60) + 'm ' + Math.round(seconds % 60) + 's';
            return Math.round(seconds / 3600) + 'h ' + Math.round((seconds % 3600) / 60) + 'm';
        }

        function formatDateTime(isoStr) {
            if (!isoStr) return '—';
            const d = new Date(isoStr);
            return d.toLocaleString('en-US', {
                month: 'short', day: 'numeric', year: 'numeric',
                hour: '2-digit', minute: '2-digit'
            });
        }

        function getStatusClass(status) {
            const s = (status || '').toUpperCase();
            if (s === 'PASSED' || s === 'COMPLETED') return 'passed';
            if (s === 'WARNING') return 'warning';
            if (s === 'FAILED') return 'failed';
            if (s === 'RUNNING') return 'running';
            return 'skipped';
        }

        function getStatusIcon(status) {
            const s = (status || '').toUpperCase();
            if (s === 'PASSED' || s === 'COMPLETED') return '✓';
            if (s === 'WARNING') return '⚠';
            if (s === 'FAILED') return '✗';
            if (s === 'RUNNING') return '◌';
            return '—';
        }

        function escapeHtml(str) {
            if (!str) return '';
            return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
        }

        function getFailureReasons(result) {
            const reasons = [];
            const checks = result.validation_checks || {};

            if (checks.anomaly_rate_acceptable === false) {
                const rate = ((result.normal_anomaly_rate || 0) * 100).toFixed(1);
                const threshold = ((result.threshold_used || 0) * 100).toFixed(1);
                reasons.push('Anomaly rate ' + rate + '% exceeds threshold ' + threshold + '%');
            }
            if (checks.sufficient_tests === false) {
                reasons.push('Insufficient test samples (minimum 10 required)');
            }
            if (checks.low_error_rate === false) {
                reasons.push('Test errors during validation');
            }
            if (checks.model_functional === false) {
                reasons.push('Model failed basic functionality check');
            }

            const explainChecks = result.explainability_checks || {};
            if (!result.explainability_passed) {
                if (explainChecks.has_training_statistics === false) {
                    reasons.push('Missing training statistics');
                }
                if (explainChecks.has_percentiles === false) {
                    reasons.push('Missing percentile data');
                }
            }

            return reasons;
        }

        function renderPeriodRow(periodName, result) {
            const displayName = periodName.replace(/_/g, ' ').replace(/\\b\\w/g, c => c.toUpperCase());

            if (result.status === 'insufficient_data') {
                return `
                    <div class="period-row">
                        <div class="period-name">${displayName}</div>
                        <div class="period-details">
                            <div class="period-detail">
                                <span class="period-detail-label">Samples:</span>
                                <span class="period-detail-value">${result.samples || 0}</span>
                            </div>
                            <div class="period-detail">
                                <span class="period-detail-label">Required:</span>
                                <span class="period-detail-value">${result.min_required || 'unknown'}</span>
                            </div>
                        </div>
                        <div class="period-status">
                            <span class="period-badge warning">⚠ Insufficient Data</span>
                        </div>
                    </div>
                `;
            }

            const passed = result.enhanced_passed || result.passed;
            const statusClass = passed ? 'passed' : 'failed';
            const statusIcon = passed ? '✓' : '✗';
            const statusText = passed ? 'Passed' : 'Failed';

            const samples = result.samples_tested || 0;
            const anomalyRate = result.normal_anomaly_rate !== undefined
                ? (result.normal_anomaly_rate * 100).toFixed(1) + '%'
                : '—';
            const threshold = result.threshold_used !== undefined
                ? (result.threshold_used * 100).toFixed(1) + '%'
                : '—';

            const failureReasons = !passed ? getFailureReasons(result) : [];

            let failureHtml = '';
            if (failureReasons.length > 0) {
                failureHtml = `
                    <div class="failure-details">
                        <div class="failure-title">Why it failed</div>
                        <ul class="failure-list">
                            ${failureReasons.map(r => '<li>' + escapeHtml(r) + '</li>').join('')}
                        </ul>
                    </div>
                `;
            }

            return `
                <div class="period-row" style="${failureReasons.length > 0 ? 'flex-wrap: wrap;' : ''}">
                    <div class="period-name">${displayName}</div>
                    <div class="period-details">
                        <div class="period-detail">
                            <span class="period-detail-label">Samples:</span>
                            <span class="period-detail-value">${formatNumber(samples)}</span>
                        </div>
                        <div class="period-detail">
                            <span class="period-detail-label">Anomaly Rate:</span>
                            <span class="period-detail-value">${anomalyRate}</span>
                        </div>
                        <div class="period-detail">
                            <span class="period-detail-label">Threshold:</span>
                            <span class="period-detail-value">${threshold}</span>
                        </div>
                    </div>
                    <div class="period-status">
                        <span class="period-badge ${statusClass}">${statusIcon} ${statusText}</span>
                    </div>
                    ${failureHtml}
                </div>
            `;
        }

        function renderPage(run) {
            const statusClass = getStatusClass(run.validation_status || run.status);
            const statusIcon = getStatusIcon(run.validation_status || run.status);
            const statusText = run.validation_status || run.status || 'Unknown';

            // Build time period rows
            const details = run.validation_details || {};
            const periods = ['business_hours', 'evening_hours', 'night_hours', 'weekend_day', 'weekend_night'];
            let periodsHtml = '';

            for (const period of periods) {
                const result = details[period];
                if (result) {
                    periodsHtml += renderPeriodRow(period, result);
                }
            }

            if (!periodsHtml) {
                periodsHtml = '<div style="padding: 40px; text-align: center; color: var(--text-muted);">No validation data available</div>';
            }

            // Error section
            let errorHtml = '';
            if (run.error_message) {
                errorHtml = `
                    <div class="error-section">
                        <div class="error-title">⚠ Training Error</div>
                        <div class="error-message">${escapeHtml(run.error_message)}</div>
                        ${run.error_traceback ? `
                            <details class="error-traceback">
                                <summary>Show stack trace</summary>
                                <pre>${escapeHtml(run.error_traceback)}</pre>
                            </details>
                        ` : ''}
                    </div>
                `;
            }

            // Data quality section
            let qualityHtml = '';
            const quality = run.data_quality_report || {};
            if (Object.keys(quality).length > 0) {
                const coverage = quality.coverage_percent || 100;
                const qualityClass = coverage >= 90 ? 'good' : coverage >= 70 ? 'warning' : 'poor';
                qualityHtml = `
                    <div class="section">
                        <div class="section-header">
                            <span class="section-title">Data Quality</span>
                            <span class="section-badge">${coverage.toFixed(0)}% coverage</span>
                        </div>
                        <div class="info-grid">
                            <div class="info-item">
                                <label>Coverage</label>
                                <div class="quality-bar">
                                    <div class="quality-fill ${qualityClass}" style="width: ${coverage}%"></div>
                                </div>
                            </div>
                            <div class="info-item">
                                <label>Data Points</label>
                                <span>${formatNumber(quality.total_points || run.total_data_points)}</span>
                            </div>
                            ${quality.gaps_detected ? `
                                <div class="info-item">
                                    <label>Gaps Detected</label>
                                    <span style="color: var(--status-warning)">${quality.gap_count || 0} gaps</span>
                                </div>
                            ` : ''}
                        </div>
                    </div>
                `;
            }

            // Timeline
            const timelineHtml = `
                <div class="section">
                    <div class="section-header">
                        <span class="section-title">Training Timeline</span>
                    </div>
                    <div class="timeline">
                        <div class="timeline-item">
                            <div class="timeline-dot info">▸</div>
                            <div class="timeline-content">
                                <div class="timeline-title">Training Started</div>
                                <div class="timeline-desc">${formatDateTime(run.started_at)}</div>
                            </div>
                        </div>
                        ${run.training_start_date ? `
                            <div class="timeline-item">
                                <div class="timeline-dot info">◆</div>
                                <div class="timeline-content">
                                    <div class="timeline-title">Data Collection Period</div>
                                    <div class="timeline-desc">${run.training_start_date} to ${run.training_end_date}</div>
                                </div>
                            </div>
                        ` : ''}
                        ${run.completed_at ? `
                            <div class="timeline-item">
                                <div class="timeline-dot ${run.status === 'COMPLETED' ? 'success' : 'info'}">
                                    ${run.status === 'COMPLETED' ? '✓' : '✗'}
                                </div>
                                <div class="timeline-content">
                                    <div class="timeline-title">${run.status === 'COMPLETED' ? 'Training Completed' : 'Training Failed'}</div>
                                    <div class="timeline-desc">${formatDateTime(run.completed_at)} (${formatDuration(run.duration_seconds)})</div>
                                </div>
                            </div>
                        ` : ''}
                    </div>
                </div>
            `;

            document.getElementById('content').innerHTML = `
                <div class="hero">
                    <div class="hero-header">
                        <div>
                            <h1 class="hero-title">${run.service_name}</h1>
                            <div class="hero-subtitle">${run.model_variant || 'baseline'} model • Run ${run.run_id.substring(0, 8)}</div>
                        </div>
                        <span class="status-pill ${statusClass}">${statusIcon} ${statusText}</span>
                    </div>
                    <div class="stats-row">
                        <div class="stat-item">
                            <div class="stat-value">${run.time_periods_trained || 0}</div>
                            <div class="stat-label">Time Periods</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-value">${formatNumber(run.total_data_points)}</div>
                            <div class="stat-label">Data Points</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-value">${formatDuration(run.duration_seconds)}</div>
                            <div class="stat-label">Duration</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-value">${run.explainability_metrics || 0}</div>
                            <div class="stat-label">Explainability Metrics</div>
                        </div>
                    </div>
                </div>

                ${errorHtml}

                <div class="section">
                    <div class="section-header">
                        <span class="section-title">Time Period Models</span>
                        <span class="section-badge">${run.periods_passed || 0} passed • ${run.periods_warned || 0} warned • ${run.periods_failed || 0} failed</span>
                    </div>
                    ${periodsHtml}
                </div>

                ${qualityHtml}
                ${timelineHtml}
            `;

            document.title = run.service_name + ' — Training Run Details — Yaga2';
        }

        async function loadRun() {
            try {
                const response = await fetch('/api/training/runs/' + RUN_ID);
                if (!response.ok) {
                    throw new Error('Training run not found');
                }
                const run = await response.json();
                renderPage(run);
            } catch (error) {
                document.getElementById('content').innerHTML = `
                    <div class="error-section">
                        <div class="error-title">Failed to load training run</div>
                        <div class="error-message">${error.message}</div>
                    </div>
                `;
            }
        }

        loadRun();
    </script>
</body>
</html>
"""


@app.get("/training/{run_id}", response_class=HTMLResponse)
async def training_run_page(run_id: str):
    """Serve the training run details page."""
    return TRAINING_RUN_PAGE_HTML.replace("{run_id}", run_id)


def serve_dashboard(initial_page: str = "overview") -> str:
    """Serve the dashboard HTML with the specified initial page."""
    return DASHBOARD_HTML.replace("{{INITIAL_PAGE}}", initial_page)


# Page routes - each serves the same SPA with different initial page
@app.get("/", response_class=HTMLResponse)
async def dashboard_root():
    """Serve the dashboard HTML at root (overview page)."""
    return serve_dashboard("overview")


@app.get("/overview", response_class=HTMLResponse)
async def dashboard_overview():
    """Serve the dashboard HTML (overview page)."""
    return serve_dashboard("overview")


@app.get("/training", response_class=HTMLResponse)
async def dashboard_training():
    """Serve the dashboard HTML (training page)."""
    return serve_dashboard("training")


@app.get("/services", response_class=HTMLResponse)
async def dashboard_services():
    """Serve the dashboard HTML (services page)."""
    return serve_dashboard("services")


@app.get("/slos", response_class=HTMLResponse)
async def dashboard_slos():
    """Serve the dashboard HTML (SLOs page)."""
    return serve_dashboard("slos")


@app.get("/dependencies", response_class=HTMLResponse)
async def dashboard_dependencies():
    """Serve the dashboard HTML (dependencies page)."""
    return serve_dashboard("dependencies")


@app.get("/settings", response_class=HTMLResponse)
async def dashboard_settings():
    """Serve the dashboard HTML (settings page)."""
    return serve_dashboard("settings")


@app.get("/documentation", response_class=HTMLResponse)
async def dashboard_docs():
    """Serve the dashboard HTML (docs page)."""
    return serve_dashboard("docs")


@app.get("/audit", response_class=HTMLResponse)
async def dashboard_audit():
    """Serve the dashboard HTML (audit page)."""
    return serve_dashboard("audit")


@app.get("/categories", response_class=HTMLResponse)
async def dashboard_categories():
    """Serve the dashboard HTML (categories page)."""
    return serve_dashboard("categories")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Yaga2 Admin Dashboard")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload on file changes")
    parser.add_argument("--port", type=int, default=8050, help="Port to run on (default: 8050)")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)")
    parser.add_argument("--no-access-log", action="store_true", help="Disable request logging")
    args = parser.parse_args()

    # Set environment variable for access logging (persists through reload)
    os.environ["YAGA_ACCESS_LOG"] = "0" if args.no_access_log else "1"

    print("\n" + "="*60)
    print("  Smartbox Anomaly Detection - Configuration Dashboard v2.1")
    print("="*60)
    print(f"\n  Dashboard URL: http://localhost:{args.port}")
    print(f"  Config file:   {CONFIG_PATH}")
    print("\n  Features:")
    print("    • Service search/filter")
    print("    • Config validation")
    print("    • Export/Import")
    print("    • Audit logging")
    print("    • Unsaved changes tracking")
    print("    • Dependency visualization")
    print("    • Training run history")
    if args.reload:
        print("\n  🔄 Auto-reload ENABLED - watching for file changes")
    if not args.no_access_log:
        print("  📝 Request logging ENABLED")
    print("\n  Press Ctrl+C to stop\n")

    # Print separator before request logs
    if not args.no_access_log:
        print("─" * 80)

    if args.reload:
        # For reload to work, we need to pass the app as a string import path
        uvicorn.run(
            "admin_dashboard:app",
            host=args.host,
            port=args.port,
            reload=True,
            reload_dirs=[str(Path(__file__).parent)],
            log_level="warning",  # Reduce uvicorn noise
            access_log=False,     # Use our custom logging instead
        )
    else:
        # Need to manually add middleware when not using reload
        # (since module is already loaded and env var check already ran)
        if not args.no_access_log and not any(
            isinstance(m.cls, type) and m.cls.__name__ == "RequestLoggingMiddleware"
            for m in app.user_middleware
        ):
            app.add_middleware(RequestLoggingMiddleware)

        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            log_level="warning",
            access_log=False,  # Use our custom logging instead
        )
