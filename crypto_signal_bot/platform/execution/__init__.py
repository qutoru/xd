"""Execution Layer — broker-agnostic abstractions and orchestration.

Turns a ready TargetBook into abstract OrderRequests and routes them through a
Broker, returning OrderResults. Stage 5 is abstractions only: no real orders.

Boundaries:
- The core (domain, broker, engine, fake_broker) knows nothing about Bybit,
  pybit, ALX, funding, momentum, ML or research.
- The Broker knows nothing about signals, alphas or portfolios — only orders.
- The ExecutionEngine is the only bridge that consumes a TargetBook.
- BybitBroker is a thin adapter placeholder with execution disabled.
"""
