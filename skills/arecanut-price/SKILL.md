---
name: arecanut-price
description: "Use when user asks arecanut/betelnut/supari mandi prices, market rate, lowest-highest-average price, or latest price update by state/market. Trigger for: arecanut price, supari rate, betelnut mandi, karnataka market price, yellapur price."
argument-hint: "User question about arecanut mandi prices; may include state and market"
user-invocable: true
disable-model-invocation: false
---

# Arecanut Price Skill

## Purpose
Fetch live arecanut mandi prices from commodityonline route pages and answer with price summary plus source URL.

## When To Use
- User asks for arecanut/betelnut/supari mandi prices.
- User asks latest price in a specific state/market.
- User asks lowest, highest, average arecanut mandi rates.
- User asks for the latest price update timestamp.

## Do Not Use
- General document questions that need uploaded-file grounding.
- Casual conversation and social chat.
- Non-arecanut commodity requests (unless the tool is extended).

## Procedure
1. Detect arecanut price intent.
2. Resolve state and market from query.
3. If missing fields, ask follow-up question for state then market.
4. Fetch commodityonline state/market route.
5. Extract `Price updated`, average, lowest, and highest price.
6. Return concise answer with source URL always included.

## Integration Notes
- Gateway router: `services/whatsapp_gateway/app/supervisor.py`
- Tool implementation: `services/whatsapp_gateway/app/arecanut_price_tool.py`
- Message dispatch: `services/whatsapp_gateway/app/main.py`
- Thinking indicator should not be sent for this route.

## References
- [Code map](./references/code-map.md)
