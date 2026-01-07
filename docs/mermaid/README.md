# Architecture Diagrams

This directory contains Mermaid diagrams documenting the inference engine and SLO evaluation architecture.

## Diagrams

| File | Description |
|------|-------------|
| `01-inference-pipeline.mmd` | High-level data flow from VictoriaMetrics through ML models to final output |
| `02-slo-evaluation-detail.mmd` | Detailed view of how each metric type is evaluated against SLO thresholds |
| `03-request-rate-correlation.mmd` | Request rate (surge/cliff) correlation-based severity logic |
| `04-inference-sequence.mmd` | Sequence diagram showing operations during a typical inference run |
| `05-severity-matrix.mmd` | How SLO status maps to final severity adjustment |

## Viewing

These `.mmd` files can be rendered using:

- **VS Code**: Install the "Mermaid Preview" extension
- **GitHub**: Rename to `.md` and wrap in ` ```mermaid ` code blocks
- **Online**: Use [mermaid.live](https://mermaid.live)
- **CLI**: Use `mmdc` (mermaid-cli) to generate PNG/SVG

## Quick Reference

### SLO Evaluation Flow

```
Metrics → ML Detection → SLO Evaluation → Severity Adjustment → Output
                              ↓
                    ┌─────────┼─────────┐
                    ↓         ↓         ↓
               Latency    Errors    DB Latency   Request Rate
                    ↓         ↓         ↓              ↓
                    └─────────┴─────────┘      (correlation-based)
                              ↓
                     Combined SLO Status
                              ↓
                     Adjusted Severity
```

### Request Rate Severity Matrix

| Type  | Condition              | Severity        |
|-------|------------------------|-----------------|
| Surge | Standalone             | informational   |
| Surge | + Latency breach       | warning         |
| Surge | + Error breach         | high            |
| Cliff | Off-peak               | warning         |
| Cliff | Peak hours             | high            |
| Cliff | + Errors               | critical        |
