#!/bin/bash
set -e

# =============================================================================
# Smartbox Anomaly Detection - Docker Entrypoint
# =============================================================================

# Default schedule configuration (can be overridden via environment variables)
TRAIN_SCHEDULE="${TRAIN_SCHEDULE:-0 2 * * *}"        # Daily at 2 AM
INFERENCE_SCHEDULE="${INFERENCE_SCHEDULE:-*/10 * * * *}"  # Every 10 minutes

# Model directories (can be overridden via environment variables)
STAGING_DIR="${STAGING_DIR:-/app/smartbox_models_staging}"
MODELS_DIR="${MODELS_DIR:-/app/smartbox_models}"

# Log directory
LOG_DIR="/app/logs"
mkdir -p "$LOG_DIR" "$STAGING_DIR"

# Function to set up cron jobs
setup_cron() {
    echo "Setting up cron schedules..."
    echo "  Training: $TRAIN_SCHEDULE"
    echo "  Inference: $INFERENCE_SCHEDULE"
    echo "  Staging dir: $STAGING_DIR"
    echo "  Models dir: $MODELS_DIR"

    # Create cron file
    cat > /etc/cron.d/smartbox << EOF
# Smartbox Anomaly Detection Cron Jobs
SHELL=/bin/bash
PATH=/usr/local/bin:/usr/bin:/bin
PYTHONUNBUFFERED=1
CONFIG_PATH=${CONFIG_PATH:-/app/config.json}

# Training schedule (uses staging directory, auto-promotes on validation success)
$TRAIN_SCHEDULE root cd /app && python main.py --staging-dir $STAGING_DIR --models-dir $MODELS_DIR --parallel 8 --warm-cache >> $LOG_DIR/train.log 2>&1

# Inference schedule (uses production models directory)
$INFERENCE_SCHEDULE root cd /app && python inference.py >> $LOG_DIR/inference.log 2>&1

EOF

    # Set proper permissions
    chmod 0644 /etc/cron.d/smartbox
    crontab /etc/cron.d/smartbox

    echo "Cron jobs configured successfully"
}

# Function to run training
run_train() {
    echo "Starting training pipeline..."
    echo "  Staging dir: $STAGING_DIR"
    echo "  Models dir: $MODELS_DIR"
    echo "  Parallel workers: 8"
    exec python main.py --staging-dir "$STAGING_DIR" --models-dir "$MODELS_DIR" --parallel 8 --warm-cache "$@"
}

# Function to run inference
run_inference() {
    echo "Starting inference pipeline..."
    exec python inference.py "$@"
}

# Function to run scheduler (cron in foreground)
run_scheduler() {
    setup_cron

    echo "Starting cron scheduler..."
    echo "Logs available at: $LOG_DIR"

    # Touch log files so tail doesn't fail
    touch "$LOG_DIR/train.log" "$LOG_DIR/inference.log"

    # Start cron in foreground
    cron -f &
    CRON_PID=$!

    # Tail logs in background for container output
    tail -F "$LOG_DIR/train.log" "$LOG_DIR/inference.log" 2>/dev/null &

    # Wait for cron
    wait $CRON_PID
}

# Function to run both once (for testing)
run_once() {
    echo "Running training once..."
    echo "  Staging dir: $STAGING_DIR"
    echo "  Models dir: $MODELS_DIR"
    echo "  Parallel workers: 8"
    python main.py --staging-dir "$STAGING_DIR" --models-dir "$MODELS_DIR" --parallel 8 --warm-cache

    echo "Running inference once..."
    python inference.py
}

# Function to run admin dashboard
run_dashboard() {
    echo "Starting admin dashboard on port 8050..."
    exec python admin_dashboard.py "$@"
}

# Main entrypoint logic
case "${1:-scheduler}" in
    train)
        shift
        run_train "$@"
        ;;
    inference)
        shift
        run_inference "$@"
        ;;
    scheduler)
        run_scheduler
        ;;
    once)
        run_once
        ;;
    dashboard)
        shift
        run_dashboard "$@"
        ;;
    shell)
        exec /bin/bash
        ;;
    *)
        # If command doesn't match, execute it directly
        exec "$@"
        ;;
esac
