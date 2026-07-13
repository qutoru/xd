"""Vendored neutral quant kernel — production's own copy.

A faithful, standalone reimplementation of the neutral market-neutral portfolio
machinery. It intentionally duplicates (does not import) the research engine so
that production is fully autonomous: frozen research keeps its copy immutable for
reproducibility, and production is free to evolve this one. No strategy lives
here — only pure portfolio/metric mechanics.
"""
