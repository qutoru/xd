"""Presentation layer — renders platform entities into user-facing messages.

Kept separate from the execution package so TradeIntent stays a pure data model
and display logic (Telegram layout, formatting) lives here:

    TradeIntent -> TelegramFormatter -> message
"""
