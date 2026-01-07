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
from datetime import datetime
from pathlib import Path
from typing import Any
import hashlib

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
import uvicorn

# Configuration
CONFIG_PATH = Path(__file__).parent / "config.json"
AUDIT_LOG_PATH = Path(__file__).parent / ".config_audit.log"
app = FastAPI(title="Yaga2 Control Center", version="2.1.0")


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
        }
        .category-critical { background: var(--status-critical-bg); color: var(--status-critical); }
        .category-standard { background: var(--status-info-bg); color: var(--status-info); }
        .category-micro { background: var(--status-success-bg); color: var(--status-success); }
        .category-admin { background: var(--status-warning-bg); color: var(--status-warning); }
        .category-core { background: rgba(163, 113, 247, 0.15); color: #a371f7; }

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

        @media (max-width: 1200px) { .grid-3 { grid-template-columns: repeat(2, 1fr); } }
        @media (max-width: 900px) {
            .app-container { grid-template-columns: 1fr; }
            .sidebar { display: none; }
            .grid-2, .grid-3 { grid-template-columns: 1fr; }
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
                <div class="nav-item active" data-page="overview">
                    <svg class="nav-item-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <rect x="3" y="3" width="7" height="7"></rect>
                        <rect x="14" y="3" width="7" height="7"></rect>
                        <rect x="14" y="14" width="7" height="7"></rect>
                        <rect x="3" y="14" width="7" height="7"></rect>
                    </svg>
                    Overview
                </div>
            </div>
            <div class="nav-section">
                <div class="nav-section-title">Configuration</div>
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
            <div class="page active" id="page-overview">
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
                                <span class="improvement-badge">client_latency ↓</span>
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
                    <div class="help-panel-title">ℹ️ About Contamination Rates</div>
                    <div class="help-panel-text">
                        <strong>Contamination</strong> is the expected proportion of anomalies in the data. Lower values = stricter detection (fewer false positives).
                        Critical services typically use 2-3%, standard services 5%, and low-traffic services 8%.
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
                    <div class="tab active" data-category="all">All Services</div>
                    <div class="tab" data-category="critical">Critical</div>
                    <div class="tab" data-category="standard">Standard</div>
                    <div class="tab" data-category="core">Core</div>
                    <div class="tab" data-category="admin">Admin</div>
                    <div class="tab" data-category="micro">Micro</div>
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
                        SLO evaluation adds operational context to ML-detected anomalies. An anomaly within acceptable thresholds becomes "informational" (logged but not alerted), while SLO breaches are escalated to "critical" regardless of ML severity.
                        <br><br><strong>Severity Matrix:</strong> Anomaly + OK → informational | Anomaly + Warning → high | SLO Breach → critical
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
                                <label class="form-label">Allow Downgrade to Informational</label>
                                <label class="toggle">
                                    <input type="checkbox" id="slo-allow-downgrade" onchange="markUnsaved()">
                                    <span class="toggle-slider"></span>
                                </label>
                            </div>
                            <div class="form-group">
                                <label class="form-label">Require SLO Breach for Critical</label>
                                <label class="toggle">
                                    <input type="checkbox" id="slo-require-breach" onchange="markUnsaved()">
                                    <span class="toggle-slider"></span>
                                </label>
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
                            <div class="grid-3" style="margin-top: var(--space-md);">
                                <div class="form-group">
                                    <label class="form-label">Error OK (%)</label>
                                    <input type="number" step="0.1" class="form-input" id="slo-default-error-acceptable" oninput="markUnsaved()">
                                </div>
                                <div class="form-group">
                                    <label class="form-label">Error Warn (%)</label>
                                    <input type="number" step="0.1" class="form-input" id="slo-default-error-warning" oninput="markUnsaved()">
                                </div>
                                <div class="form-group">
                                    <label class="form-label">Error Crit (%)</label>
                                    <input type="number" step="0.1" class="form-input" id="slo-default-error-critical" oninput="markUnsaved()">
                                </div>
                            </div>
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
                                Thresholds are based on ratio to training baseline (floor + ratio × mean)
                            </p>
                            <div class="ratio-grid">
                                <div class="ratio-item">
                                    <div class="ratio-label">OK</div>
                                    <div class="ratio-value ok" id="db-ratio-ok">1.5×</div>
                                </div>
                                <div class="ratio-item">
                                    <div class="ratio-label">Warning</div>
                                    <div class="ratio-value warning" id="db-ratio-warning">2.0×</div>
                                </div>
                                <div class="ratio-item">
                                    <div class="ratio-label">High</div>
                                    <div class="ratio-value high" id="db-ratio-high">3.0×</div>
                                </div>
                                <div class="ratio-item">
                                    <div class="ratio-label">Critical</div>
                                    <div class="ratio-value critical" id="db-ratio-critical">5.0×</div>
                                </div>
                            </div>
                            <div class="form-group" style="margin-top: var(--space-md);">
                                <label class="form-label">Floor (minimum ms)</label>
                                <input type="number" class="form-input" id="db-latency-floor" value="5" oninput="markUnsaved()" style="width: 100px;">
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
                    </div>
                </div>

                <!-- Training Baselines -->
                <div class="card" style="margin-bottom: var(--space-lg);">
                    <div class="card-header"><div class="card-title">📊 Training Baselines</div></div>
                    <div class="card-body">
                        <p style="color: var(--text-secondary); margin-bottom: var(--space-md);">
                            Each time-period model stores training statistics that are used for SLO evaluation:
                        </p>
                        <div class="grid-2">
                            <div>
                                <strong style="font-size: 12px;">Metrics with training means:</strong>
                                <ul style="font-size: 12px; color: var(--text-secondary); margin-top: var(--space-xs); padding-left: 20px;">
                                    <li><code>application_latency_mean</code></li>
                                    <li><code>client_latency_mean</code></li>
                                    <li><code>database_latency_mean</code></li>
                                    <li><code>error_rate_mean</code></li>
                                    <li><code>request_rate_mean</code></li>
                                </ul>
                            </div>
                            <div>
                                <strong style="font-size: 12px;">Used for:</strong>
                                <ul style="font-size: 12px; color: var(--text-secondary); margin-top: var(--space-xs); padding-left: 20px;">
                                    <li>Database latency ratio thresholds</li>
                                    <li>Request rate surge/cliff detection</li>
                                    <li>Contextual severity adjustment</li>
                                    <li>Pattern explanation generation</li>
                                </ul>
                            </div>
                        </div>
                    </div>
                </div>

                <div class="grid-2" style="margin-bottom: var(--space-lg);">
                    <div class="card">
                        <div class="card-header"><div class="card-title">Service Categories</div></div>
                        <div class="card-body" style="padding: 0;">
                            <table class="services-table">
                                <thead><tr><th>Category</th><th>Contamination</th><th>Use For</th></tr></thead>
                                <tbody>
                                    <tr><td><span class="category-badge category-critical">Critical</span></td><td><code>3%</code></td><td>Revenue-impacting</td></tr>
                                    <tr><td><span class="category-badge category-standard">Standard</span></td><td><code>5%</code></td><td>Production services</td></tr>
                                    <tr><td><span class="category-badge category-core">Core</span></td><td><code>4%</code></td><td>Infrastructure</td></tr>
                                    <tr><td><span class="category-badge category-admin">Admin</span></td><td><code>6%</code></td><td>Admin interfaces</td></tr>
                                    <tr><td><span class="category-badge category-micro">Micro</span></td><td><code>8%</code></td><td>Utility services</td></tr>
                                </tbody>
                            </table>
                        </div>
                    </div>
                    <div class="card">
                        <div class="card-header"><div class="card-title">SLO Severity Matrix</div></div>
                        <div class="card-body" style="padding: 0;">
                            <table class="services-table">
                                <thead><tr><th></th><th>OK</th><th>Warning</th><th>Breach</th></tr></thead>
                                <tbody>
                                    <tr><td><strong>Anomaly</strong></td><td style="color: var(--status-success)">info</td><td style="color: var(--status-warning)">high</td><td style="color: var(--status-critical)">critical</td></tr>
                                    <tr><td><strong>No Anomaly</strong></td><td style="color: var(--text-muted)">—</td><td style="color: var(--status-warning)">warn</td><td style="color: var(--status-critical)">critical</td></tr>
                                </tbody>
                            </table>
                        </div>
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
  ✓ SLO latency_evaluation.status ≠ "ok" (client_latency above threshold)
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
                            <span class="improvement-badge">client_latency ↓ = OK</span>
                            <span class="improvement-badge">database_latency ↓ = OK</span>
                            <span class="improvement-badge">error_rate ↓ = OK</span>
                        </div>
                        <p style="color: var(--text-muted); font-size: 11px; margin-top: var(--space-md);">
                            Example: Client latency drops from 50ms mean to 20ms → No alert (this is an improvement)
                        </p>
                    </div>
                </div>

                <div class="card">
                    <div class="card-header"><div class="card-title">Incident Lifecycle</div></div>
                    <div class="card-body">
                        <div class="code-block">┌─────────────┐     confirmed     ┌──────────┐
│  SUSPECTED  │ ─────────────────►│   OPEN   │
└─────────────┘   (N cycles)      └──────────┘
      ▲                                 │
      │                                 │ not detected
      │           ┌─────────────┐       │
      └───────────│ RECOVERING  │◄──────┘
     re-detected  └─────────────┘
                        │
                        │ grace period expired
                        ▼
                  ┌──────────┐
                  │  CLOSED  │
                  └──────────┘</div>
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

        document.addEventListener('DOMContentLoaded', async () => {
            await loadConfig();
            setupNavigation();
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
            for (const category of ['critical', 'standard', 'core', 'admin', 'micro']) {
                for (const service of (services[category] || [])) {
                    const cont = contamination[service] || categoryContamination[category] || 0.05;
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

        function renderServicesTable() {
            const services = config.services || {};
            const slos = config.slos || {};
            const contamination = config.model?.contamination_by_service || {};
            const categoryContamination = config.model?.contamination_by_category || {};

            let html = '';
            for (const category of ['critical', 'standard', 'core', 'admin', 'micro']) {
                for (const service of (services[category] || [])) {
                    if (currentCategory !== 'all' && currentCategory !== category) continue;
                    const cont = contamination[service] || categoryContamination[category] || 0.05;
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
            document.getElementById('slo-default-min-traffic').value = defaults.min_traffic_rps || 1.0;
            document.getElementById('slo-default-busy-factor').value = defaults.busy_period_factor || 1.5;

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
            for (const cat of ['critical', 'standard', 'core', 'admin', 'micro']) {
                if ((services[cat] || []).includes(service)) return cat;
            }
            return 'standard';
        }

        function setupNavigation() {
            document.querySelectorAll('.nav-item').forEach(item => {
                item.addEventListener('click', () => {
                    const page = item.dataset.page;
                    if (page) {
                        navigateTo(page);
                        if (page === 'audit') loadAuditLog();
                    }
                });
            });

            document.querySelectorAll('#services-tabs .tab').forEach(tab => {
                tab.addEventListener('click', () => {
                    document.querySelectorAll('#services-tabs .tab').forEach(t => t.classList.remove('active'));
                    tab.classList.add('active');
                    currentCategory = tab.dataset.category;
                    renderServicesTable();
                });
            });
        }

        function navigateTo(page) {
            document.querySelectorAll('.nav-item').forEach(i => i.classList.remove('active'));
            document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
            document.querySelector(`.nav-item[data-page="${page}"]`)?.classList.add('active');
            document.getElementById(`page-${page}`)?.classList.add('active');
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
                        <option value="critical">Critical</option>
                        <option value="standard" selected>Standard</option>
                        <option value="core">Core</option>
                        <option value="admin">Admin</option>
                        <option value="micro">Micro</option>
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
                        ${['critical', 'standard', 'core', 'admin', 'micro'].map(c => `<option value="${c}" ${c === category ? 'selected' : ''}>${c.charAt(0).toUpperCase() + c.slice(1)}</option>`).join('')}
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
                    min_traffic_rps: parseFloat(document.getElementById('slo-default-min-traffic').value),
                    busy_period_factor: parseFloat(document.getElementById('slo-default-busy-factor').value),
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
            for (const cat of ['critical', 'standard', 'core', 'admin', 'micro']) {
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

        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') closeModal();
            if ((e.ctrlKey || e.metaKey) && e.key === 's') { e.preventDefault(); saveFullConfig('Saved via keyboard shortcut'); }
        });
    </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Serve the dashboard HTML."""
    return DASHBOARD_HTML


if __name__ == "__main__":
    print("\\n" + "="*60)
    print("  Smartbox Anomaly Detection - Configuration Dashboard v2.0")
    print("="*60)
    print(f"\\n  Dashboard URL: http://localhost:8050")
    print(f"  Config file:   {CONFIG_PATH}")
    print("\\n  Features:")
    print("    • Service search/filter")
    print("    • Config validation")
    print("    • Export/Import")
    print("    • Audit logging")
    print("    • Unsaved changes tracking")
    print("    • Dependency visualization")
    print("\\n  Press Ctrl+C to stop\\n")

    uvicorn.run(app, host="0.0.0.0", port=8050, log_level="warning")
