---
name: news-search
description: "Use when user asks latest news, headlines, breaking updates, or current-events by topic/location. Trigger for: latest news, top headlines, today news, technology news, political updates, world news."
argument-hint: "User query asking for latest web news"
user-invocable: true
disable-model-invocation: false
---

# News Search Skill

## Purpose
Fetch fresh web news using DuckDuckGo News search, then return a concise answer in the same language the user asked.
Also supports reusable candidate-list mode for downstream flows (for example meme-generation news selection before image creation).

## When To Use
- User asks for latest news or headlines.
- User asks current updates by topic, company, country, domain, or person.
- User expects web-news style answers rather than document-grounded retrieval.

## Do Not Use
- Uploaded-document questions that need RAG citations.
- Arecanut mandi price/rate queries.
- Pure social chat without a news request.

## Procedure
1. Detect news intent.
2. Rewrite the query with LLM for stronger news retrieval.
3. Dynamically choose top-k (default 5, but not fixed).
4. Fetch results from DuckDuckGo News.
5. Synthesize a concise response in the user's language.

## Integration Notes
- Skill implementation: `services/whatsapp_gateway/app/skills/news_search_skill.py`
- Supervisor route type: `news_search`
- Gateway dispatch: `services/whatsapp_gateway/app/main.py`
- Thinking indicator should not be sent for this route.
- Meme integration: `meme_generation` route reuses this skill to fetch selectable news when user asks for trending/news-based memes.

## References
- [Code map](./references/code-map.md)
