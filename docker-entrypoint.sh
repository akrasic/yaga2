#!/bin/bash
set -e

# =============================================================================
# Smartbox Anomaly Detection - Docker Entrypoint
# =============================================================================

# Default schedule configuration (can be overridden via environment variables)
TRAIN_SCHEDULE="${TRAIN_SCHEDULE:-0 2 * * *}"        # Daily at 2 AM
INFERENCE_SCHEDULE="${INFERENCE_SCHEDULE:-*/10 * * * *}"  # Every 10 minutes

# Log directory
LOG_DIR="/app/logs"
mkdir -p "$LOG_DIR"

# Function to set up cron jobs
setup_cron() {
    echo "Setting up cron schedules..."
    echo "  Training: $TRAIN_SCHEDULE"
    echo "  Inference: $INFERENCE_SCHEDULE"

    # Create cron file
    cat > /etc/cron.d/smartbox << EOF
# Smartbox Anomaly Detection Cron Jobs
SHELL=/bin/bash
PATH=/usr/local/bin:/usr/bin:/bin
PYTHONUNBUFFERED=1
CONFIG_PATH=${CONFIG_PATH:-/app/config.json}

# Training schedule
$TRAIN_SCHEDULE root cd /app && python main.py >> $LOG_DIR/train.log 2>&1

# Inference schedule
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
    exec python main.py "$@"
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
    python main.py

    echo "Running inference once..."
    python inference.py
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
    shell)
        exec /bin/bash
        ;;
    *)
        # If command doesn't match, execute it directly
        exec "$@"
        ;;
esac
