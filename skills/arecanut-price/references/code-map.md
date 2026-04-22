# Arecanut Price Skill Code Map

## WhatsApp Gateway

- `services/whatsapp_gateway/app/supervisor.py`
  - Adds `arecanut_price` route type in LLM routing policy.
- `services/whatsapp_gateway/app/arecanut_price_tool.py`
  - State/market slot filling.
  - CommodityOnline scraping + parsing.
  - Always includes source URL in output.
- `services/whatsapp_gateway/app/main.py`
  - Dispatches arecanut price route without RAG/thinking indicator.

## Tests

- `services/whatsapp_gateway/tests/test_arecanut_price_tool.py`
  - Normalization, parsing, and source-inclusion checks.
- `services/whatsapp_gateway/tests/test_supervisor.py`
  - Route classification acceptance for `arecanut_price`.
- `services/whatsapp_gateway/tests/test_main_routing.py`
  - Verifies dispatch path and no thinking indicator for this route.

## Runtime Expectations

- Should use direct commodityonline route scraping for deterministic results.
- Should ask follow-up for missing state/market.
- Should include `Source:` in every arecanut skill response.
