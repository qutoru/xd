"""Observability — operator-facing projections of a completed pipeline cycle.

Read-only and computes nothing: it reads the objects a cycle already produced
(PipelineResult and the reports it carries) and reshapes them into one structured
summary for logging/diagnostics. It has no runtime dependency on the pipeline,
performs no PnL math, and is venue-agnostic.
"""
