# Inference Module Architecture

**Last Updated:** 2026-01-14
**Version:** 2.0.0

This document describes the architecture of the inference module (`smartbox_anomaly/inference/`) after the January 2026 refactoring.

---

## Overview

The inference module provides production-grade anomaly detection with:
- Two-pass detection with dependency context
- SLO-aware severity evaluation
- Exception and service graph enrichment
- Time-aware model selection

## Module Structure

```
smartbox_anomaly/inference/
├── __init__.py              # Public exports
├── pipeline.py              # Main orchestrator (SmartboxMLInferencePipeline)
├── detection_runner.py      # Two-pass detection logic (DetectionRunner)
├── enrichment_runner.py     # Post-detection enrichment (EnrichmentRunner)
├── detection_engine.py      # ML detection engine
├── model_manager.py         # Model loading and caching
├── time_aware.py            # Time-aware detection
├── results_processor.py     # Alert formatting and API posting
└── models.py                # Data models (AnomalyResult, ServiceInferenceResult)
```

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      SmartboxMLInferencePipeline                            │
│                         (Orchestrator Layer)                                │
│                                                                             │
│  Responsibilities:                                                          │
│  - Initialize all components                                                │
│  - Coordinate detection and enrichment phases                               │
│  - Provide public API (run_inference, run_enhanced_time_aware_inference)    │
└─────────────────────────────────────────────────────────────────────────────┘
                │                                    │
                ▼                                    ▼
┌───────────────────────────────────┐  ┌───────────────────────────────────────┐
│        DetectionRunner            │  │         EnrichmentRunner              │
│     (Detection Layer)             │  │       (Enrichment Layer)              │
│                                   │  │                                       │
│  Responsibilities:                │  │  Responsibilities:                    │
│  - Metrics collection             │  │  - SLO-aware severity evaluation      │
│  - Input validation               │  │  - Exception context enrichment       │
│  - Pass 1: Initial detection      │  │  - Service graph enrichment           │
│  - Pass 2: Dependency-aware       │  │  - Result logging and summary         │
│  - Dependency context building    │  │                                       │
└───────────────────────────────────┘  └───────────────────────────────────────┘
                │                                    │
                ▼                                    ▼
┌───────────────────────────────────┐  ┌───────────────────────────────────────┐
│  EnhancedTimeAwareDetector        │  │  SLOEvaluator                         │
│  EnhancedModelManager             │  │  ExceptionEnrichmentService           │
│  VictoriaMetricsClient            │  │  ServiceGraphEnrichmentService        │
└───────────────────────────────────┘  └───────────────────────────────────────┘
```

## Component Details

### SmartboxMLInferencePipeline

**File:** `pipeline.py` (452 lines)

The main entry point and orchestrator. Uses composition to delegate work to specialized runners.

```python
class SmartboxMLInferencePipeline:
    def __init__(self, ...):
        # Initialize core components
        self.vm_client = VictoriaMetricsClient(vm_endpoint)
        self.model_manager = EnhancedModelManager(models_directory)
        # ...

        # Initialize runners (composition)
        self._detection_runner = DetectionRunner(...)
        self._enrichment_runner = EnrichmentRunner(...)
```

**Key Methods:**

| Method | Description |
|--------|-------------|
| `run_enhanced_time_aware_inference()` | Full pipeline: metrics → detection → enrichment |
| `run_inference(services)` | Simplified interface returning structured results |
| `get_system_status()` | Health check and system status |

### DetectionRunner

**File:** `detection_runner.py` (516 lines)

Handles two-pass anomaly detection with dependency context.

```python
class DetectionRunner:
    def __init__(
        self,
        vm_client: VictoriaMetricsClient,
        model_manager: EnhancedModelManager,
        time_aware_detector: EnhancedTimeAwareDetector,
        dependency_graph: Dict[str, List[str]],
        check_drift: bool = False,
        verbose: bool = False,
    ):
```

**Key Methods:**

| Method | Description |
|--------|-------------|
| `validate_metrics()` | Sanitize inputs (NaN, negative values, extremes) |
| `collect_metrics_for_services()` | Batch metric collection from VictoriaMetrics |
| `run_pass1_detection()` | Initial detection without dependency context |
| `run_pass2_detection()` | Re-analyze with dependency context for latency anomalies |
| `build_dependency_context()` | Build upstream dependency state from Pass 1 results |
| `has_latency_anomaly()` | Check if result contains latency-related anomalies |

**Two-Pass Detection Flow:**

```
Pass 1: Detect anomalies for all services (no dependency context)
           │
           ▼
     Identify services with latency anomalies
           │
           ▼
Pass 2: Re-run detection for latency anomalies with dependency context
        (adds cascade_analysis to identify root cause service)
```

### EnrichmentRunner

**File:** `enrichment_runner.py` (314 lines)

Handles post-detection enrichment including SLO evaluation and context enrichment.

```python
class EnrichmentRunner:
    def __init__(
        self,
        slo_evaluator: Optional[SLOEvaluator],
        exception_enrichment: ExceptionEnrichmentService,
        service_graph_enrichment: ServiceGraphEnrichmentService,
        verbose: bool = False,
    ):
```

**Key Methods:**

| Method | Description |
|--------|-------------|
| `apply_slo_evaluation()` | Adjust severity based on SLO thresholds |
| `apply_exception_enrichment()` | Add exception breakdown for error anomalies |
| `apply_service_graph_enrichment()` | Add downstream service calls for latency anomalies |
| `enrich_with_exceptions()` | Single-service exception enrichment |
| `process_and_log_results()` | Log summary and statistics |

**Enrichment Flow:**

```
Detection Results
       │
       ▼
┌──────────────────┐
│  SLO Evaluation  │  Adjust severity based on operational thresholds
└────────┬─────────┘
         ▼
┌──────────────────┐
│ Exception Enrich │  Add exception breakdown if error SLO breached
└────────┬─────────┘
         ▼
┌──────────────────┐
│ Service Graph    │  Add downstream calls if latency SLO breached
└────────┬─────────┘
         ▼
   Enriched Results
```

## Inference Pipeline Flow

The complete inference pipeline runs in 6 phases:

```
Phase 1: Collect Metrics
         ├── Query VictoriaMetrics for all services
         └── Cache results for both detection passes

Phase 2: Pass 1 Detection
         ├── Run time-aware detection for each service
         └── No dependency context (detect in isolation)

Phase 3: Pass 2 Detection
         ├── Identify services with latency anomalies
         ├── Build dependency context from Pass 1 results
         └── Re-run detection with cascade analysis

Phase 4: SLO Evaluation
         ├── Compare metrics against SLO thresholds
         └── Adjust severity (e.g., critical → low if within SLO)

Phase 5: Exception Enrichment
         ├── Check if error SLO breached
         └── Query exception types from OpenTelemetry metrics

Phase 6: Service Graph Enrichment
         ├── Check if latency SLO breached
         └── Query downstream service calls from service graph metrics
```

## Public API

### Main Pipeline

```python
from smartbox_anomaly.inference import SmartboxMLInferencePipeline

# Initialize
pipeline = SmartboxMLInferencePipeline(
    vm_endpoint="https://otel-metrics.production.smartbox.com",
    verbose=True,
)

# Run full inference
results = pipeline.run_enhanced_time_aware_inference()

# Run for specific services
results = pipeline.run_inference(["booking", "search"])

# Health check
status = pipeline.get_system_status()
```

### Direct Runner Access

For advanced use cases, runners can be accessed directly:

```python
from smartbox_anomaly.inference import DetectionRunner, EnrichmentRunner

# Runners are initialized internally but can be used for testing
detection_runner = pipeline._detection_runner
enrichment_runner = pipeline._enrichment_runner

# Validate metrics
validated, warnings = detection_runner.validate_metrics(metrics, "booking")

# Apply SLO evaluation
enriched_results = enrichment_runner.apply_slo_evaluation(results)
```

## Design Decisions

### Why Composition Over Inheritance?

The refactoring uses **composition** (runners as separate objects) rather than mixins or inheritance because:

1. **Clearer ownership** - Each runner owns specific functionality
2. **Easier testing** - Runners can be unit tested independently
3. **Better encapsulation** - Implementation details hidden behind runner interfaces
4. **Simpler dependencies** - Each runner declares its own dependencies

### Why Two-Pass Detection?

Two-pass detection is necessary for accurate cascade analysis:

1. **Pass 1** detects anomalies for all services independently
2. **Pass 2** uses Pass 1 results to understand which upstream services are affected
3. This enables identifying root cause in dependency chains

Without two passes, a service would need to know its dependencies' state before detection completes, creating a circular dependency.

### Backward Compatibility

The refactoring maintains backward compatibility through delegation methods in `SmartboxMLInferencePipeline`:

```python
# These methods delegate to detection_runner
def _validate_metrics(self, metrics_dict, service_name):
    return self._detection_runner.validate_metrics(metrics_dict, service_name)

def _has_latency_anomaly(self, result):
    return self._detection_runner.has_latency_anomaly(result)

def _build_dependency_context(self, service_name, all_results):
    return self._detection_runner.build_dependency_context(service_name, all_results)
```

---

## Refactoring History

### January 2026 Refactoring (v2.0.0)

**Problem:** `pipeline.py` had grown to 1,114 lines with multiple responsibilities mixed together.

**Solution:** Split into three focused modules using composition pattern.

| Before | After |
|--------|-------|
| `pipeline.py` (1,114 lines) | `pipeline.py` (452 lines) - orchestration |
| | `detection_runner.py` (516 lines) - detection logic |
| | `enrichment_runner.py` (314 lines) - enrichment logic |

**Benefits:**
- Each file under 600 lines
- Clear separation of concerns
- Improved testability
- All 439 tests continue to pass

---

## Related Documentation

- [DETECTION_SIGNALS.md](./DETECTION_SIGNALS.md) - Detection methods (IF, pattern matching)
- [CONFIGURATION.md](./CONFIGURATION.md) - SLO and service configuration
- [FINGERPRINTING.md](./FINGERPRINTING.md) - Incident lifecycle tracking
- [INFERENCE_API_PAYLOAD.md](./INFERENCE_API_PAYLOAD.md) - Output payload format
