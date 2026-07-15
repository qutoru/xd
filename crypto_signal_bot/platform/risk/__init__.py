"""Risk Layer — turns portfolio weights into risk-sized target notionals.

A standalone layer between Portfolio and IntentBuilder:

    TargetBook(weights) + NAV + RiskConfig -> RiskManager -> {symbol: target_notional}

It is deliberately independent: it imports no Execution, Broker, Bybit, Research or
Strategy code — only the portfolio TargetBook it consumes and pandas.
"""
