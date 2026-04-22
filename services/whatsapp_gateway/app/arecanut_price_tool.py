import asyncio
import html
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo

import httpx


COMMODITY_SLUG = "arecanut-betelnutsupari"
BASE_SOURCE_URL = "https://www.commodityonline.com/mandiprices"
LANG_EN = "en"
LANG_KN = "kn"

AVAILABLE_STATES = [
    "all",
    "assam",
    "goa",
    "karnataka",
    "kerala",
    "maharashtra",
    "meghalaya",
    "nagaland",
    "odisha",
    "tamil-nadu",
    "uttar-pradesh",
]

AVAILABLE_MARKETS = [
    "arasikere",
    "bangalore",
    "bantwala",
    "belthangdi",
    "bhadravathi",
    "chamaraj nagar",
    "channagiri",
    "chikkamagalore",
    "chitradurga",
    "davangere",
    "gonikappal",
    "gowribidanoor",
    "gubbi",
    "gundlupet",
    "haliyala",
    "hiriyur",
    "holalkere",
    "honnali",
    "honnali apmc",
    "honnavar",
    "hosadurga",
    "hosanagar",
    "hubli (amaragol)",
    "hunsur",
    "k.r. pet",
    "k.r.nagar",
    "karkala",
    "koppa",
    "kumta",
    "kundapura",
    "madhugiri",
    "madikeri",
    "malur",
    "mangalore",
    "moodigere",
    "pavagada",
    "piriya pattana",
    "puttur",
    "sagar",
    "shikaripura",
    "shimoga",
    "shimoga(theertahalli)",
    "siddapur",
    "sira",
    "sirsi",
    "somvarpet",
    "sorabha",
    "srirangapattana",
    "sulya",
    "tarikere",
    "thirthahalli",
    "tumkur",
    "turvekere",
    "yellapur",
]

STATE_ALIASES = {
    "tamil nadu": "tamil-nadu",
    "uttar pradesh": "uttar-pradesh",
    "ಕರ್ನಾಟಕ": "karnataka",
    "ಕೇರಳ": "kerala",
    "ತಮಿಳುನಾಡು": "tamil-nadu",
    "ಮಹಾರಾಷ್ಟ್ರ": "maharashtra",
    "ಒಡಿಶಾ": "odisha",
    "ಅಸ್ಸಾಂ": "assam",
}

MARKET_ALIASES = {
    "kr pet": "k.r. pet",
    "k r pet": "k.r. pet",
    "krnagar": "k.r.nagar",
    "k r nagar": "k.r.nagar",
    "therthahalli": "thirthahalli",
    "theerthahalli": "thirthahalli",
    "theertahalli": "thirthahalli",
    "shimoga theertahalli": "shimoga(theertahalli)",
    "hubli amaragol": "hubli (amaragol)",
    "ಯಲ್ಲಾಪುರ": "yellapur",
    "ಸಿರಸಿ": "sirsi",
    "ಸಿದ್ಧಾಪುರ": "siddapur",
    "ಸಾಗರ": "sagar",
}

STATE_DISPLAY_KN = {
    "karnataka": "ಕರ್ನಾಟಕ",
    "kerala": "ಕೇರಳ",
    "tamil-nadu": "ತಮಿಳುನಾಡು",
    "maharashtra": "ಮಹಾರಾಷ್ಟ್ರ",
    "odisha": "ಒಡಿಶಾ",
    "uttar-pradesh": "ಉತ್ತರ ಪ್ರದೇಶ",
    "goa": "ಗೋವಾ",
    "assam": "ಅಸ್ಸಾಂ",
    "meghalaya": "ಮೆಘಾಲಯ",
    "nagaland": "ನಾಗಾಲ್ಯಾಂಡ್",
}

MARKET_DISPLAY_KN = {
    "arasikere": "ಅರಸೀಕೆರೆ",
    "bangalore": "ಬೆಂಗಳೂರು",
    "bantwala": "ಬಂಟ್ವಾಳ",
    "belthangdi": "ಬೆಳ್ತಂಗಡಿ",
    "bhadravathi": "ಭದ್ರಾವತಿ",
    "chamaraj nagar": "ಚಾಮರಾಜನಗರ",
    "channagiri": "ಚನ್ನಗಿರಿ",
    "chikkamagalore": "ಚಿಕ್ಕಮಗಳೂರು",
    "chitradurga": "ಚಿತ್ರದುರ್ಗ",
    "davangere": "ದಾವಣಗೆರೆ",
    "gonikappal": "ಗೋಣಿಕೊಪ್ಪಲು",
    "gowribidanoor": "ಗೌರಿಬಿದನೂರು",
    "gubbi": "ಗುಬ್ಬಿ",
    "gundlupet": "ಗುಂಡ್ಲುಪೇಟೆ",
    "haliyala": "ಹಳಿಯಾಳ",
    "hiriyur": "ಹಿರಿಯೂರು",
    "holalkere": "ಹೊಳಲ್ಕೆರೆ",
    "honnali": "ಹೊನ್ನಾಳಿ",
    "honnali apmc": "ಹೊನ್ನಾಳಿ ಎಪಿಎಂಸಿ",
    "honnavar": "ಹೊನ್ನಾವರ",
    "hosadurga": "ಹೊಸದುರ್ಗ",
    "hosanagar": "ಹೊಸನಗರ",
    "hubli (amaragol)": "ಹುಬ್ಬಳ್ಳಿ (ಅಮರಗೋಳ)",
    "hunsur": "ಹುನ್ಸೂರು",
    "k.r. pet": "ಕೆ.ಆರ್. ಪೇಟೆ",
    "k.r.nagar": "ಕೆ.ಆರ್. ನಗರ",
    "karkala": "ಕಾರ್ಕಳ",
    "koppa": "ಕೊಪ್ಪ",
    "kumta": "ಕುಮಟಾ",
    "kundapura": "ಕುಂದಾಪುರ",
    "madhugiri": "ಮಧುಗಿರಿ",
    "madikeri": "ಮಡಿಕೇರಿ",
    "malur": "ಮಾಲೂರು",
    "mangalore": "ಮಂಗಳೂರು",
    "moodigere": "ಮೂಡಿಗೆರೆ",
    "pavagada": "ಪಾವಗಡ",
    "piriya pattana": "ಪಿರಿಯಾಪಟ್ಟಣ",
    "puttur": "ಪುತ್ತೂರು",
    "sagar": "ಸಾಗರ",
    "shikaripura": "ಶಿಕಾರಿಪುರ",
    "shimoga": "ಶಿವಮೊಗ್ಗ",
    "shimoga(theertahalli)": "ಶಿವಮೊಗ್ಗ (ತೀರ್ಥಹಳ್ಳಿ)",
    "siddapur": "ಸಿದ್ಧಾಪುರ",
    "sira": "ಸಿರಾ",
    "sirsi": "ಸಿರಸಿ",
    "somvarpet": "ಸೋಮವಾರಪೇಟೆ",
    "sorabha": "ಸೊರಬ",
    "srirangapattana": "ಶ್ರೀರಂಗಪಟ್ಟಣ",
    "sulya": "ಸುಳ್ಯ",
    "tarikere": "ತಾರಿಕೆರೆ",
    "thirthahalli": "ತೀರ್ಥಹಳ್ಳಿ",
    "tumkur": "ತುಮಕೂರು",
    "turvekere": "ತುರುವೇಕೆರೆ",
    "yellapur": "ಯಲ್ಲಾಪುರ",
}


@dataclass
class CacheItem:
    answer_text: str
    expires_at: float


class ArecanutPriceTool:
    def __init__(self) -> None:
        self.timeout_seconds = float(os.getenv("ARECANUT_PRICE_TIMEOUT_SECONDS", "15"))
        self.cache_ttl_seconds = int(os.getenv("ARECANUT_PRICE_CACHE_TTL_SECONDS", "300"))
        self.max_retries = int(os.getenv("ARECANUT_PRICE_MAX_RETRIES", "2"))
        self.max_table_rows = int(os.getenv("ARECANUT_PRICE_MAX_TABLE_ROWS", "200"))
        self.max_display_rows = int(os.getenv("ARECANUT_PRICE_MAX_DISPLAY_ROWS", "10"))
        self.enabled = self._as_bool(os.getenv("ARECANUT_PRICE_ENABLED"), True)
        self.enable_jina_fallback = self._as_bool(os.getenv("ARECANUT_PRICE_ENABLE_JINA_FALLBACK"), True)

        self._cache: Dict[str, CacheItem] = {}
        self._state_aliases = self._build_aliases(AVAILABLE_STATES, STATE_ALIASES)
        self._market_aliases = self._build_aliases(AVAILABLE_MARKETS, MARKET_ALIASES)
        self._sorted_state_aliases = sorted(self._state_aliases.items(), key=lambda pair: len(pair[0]), reverse=True)
        self._sorted_market_aliases = sorted(self._market_aliases.items(), key=lambda pair: len(pair[0]), reverse=True)

    async def ask(self, user_query: str, conversation_history: Optional[List[Dict[str, str]]] = None) -> str:
        language = self._resolve_language(user_query=user_query, conversation_history=conversation_history)
        if not self.enabled:
            return (
                f"{self._msg(language, 'feature_disabled')}\n"
                f"{self._msg(language, 'source', source=self._source_template_url())}"
            )

        state, market = self._extract_state_market(user_query)
        if conversation_history and (not state or not market):
            history_state, history_market = self._extract_state_market_from_history(conversation_history)
            state = state or history_state
            market = market or history_market

        if not state or state == "all":
            return self._state_follow_up_message(language)

        if not market:
            return self._market_follow_up_message(state, language)

        return await self._fetch_and_format_response(state=state, market=market, language=language)

    def get_fetching_message(
        self,
        user_query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        language = self._resolve_language(user_query=user_query, conversation_history=conversation_history)
        return self._msg(language, "fetching_message")

    async def _fetch_and_format_response(self, state: str, market: str, language: str) -> str:
        cache_key = f"{state}|{market}|{language}"
        now = time.time()

        cached = self._cache.get(cache_key)
        if cached and cached.expires_at > now:
            return cached.answer_text

        state_slug = self._to_slug(state)
        market_candidates = self._candidate_market_slugs(market)

        last_error: Optional[str] = None
        for market_slug in market_candidates:
            url = f"{BASE_SOURCE_URL}/{COMMODITY_SLUG}/{state_slug}/{market_slug}"
            html_text, err, fetch_mode = await self._fetch_html(url)
            if err:
                last_error = err
                continue

            parsed = self._parse_market_page(html_text)
            if not parsed["has_any_price"]:
                last_error = "No price fields found in parsed response"
                continue

            answer = self._format_success_answer(
                state=state,
                market=market,
                source_url=url,
                fetch_mode=fetch_mode,
                price_updated=parsed["price_updated"],
                avg_price=parsed["average_price"],
                lowest_price=parsed["lowest_price"],
                highest_price=parsed["highest_price"],
                table_rows=parsed["table_rows"],
                language=language,
            )
            self._cache[cache_key] = CacheItem(answer_text=answer, expires_at=now + self.cache_ttl_seconds)
            return answer

        source_hint = f"{BASE_SOURCE_URL}/{COMMODITY_SLUG}/{state_slug}/{self._to_slug(market)}"
        reason = last_error or "Unable to parse commodity page"
        return (
            f"{self._msg(language, 'fetch_failed', market=self._display_market(market, language), state=self._display_state(state, language))}\n"
            f"{self._msg(language, 'reason', reason=reason)}\n"
            f"{self._msg(language, 'source', source=source_hint)}"
        )

    async def _fetch_html(self, url: str) -> tuple[Optional[str], Optional[str], str]:
        headers = {
            "User-Agent": "RagsApp-ArecanutSkill/1.0 (+https://github.com/negativenagesh/RagsApp)",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }

        for attempt in range(1, self.max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self.timeout_seconds, follow_redirects=True) as client:
                    response = await client.get(url, headers=headers)
                if response.status_code == 404:
                    return None, "Market page not found", "direct"
                if response.status_code == 403:
                    return await self._fetch_via_jina(url)
                response.raise_for_status()
                if not response.text:
                    return None, "Empty response body", "direct"
                return response.text, None, "direct"
            except Exception as exc:
                if attempt >= self.max_retries:
                    if self.enable_jina_fallback:
                        return await self._fetch_via_jina(url)
                    return None, str(exc), "direct"
                await asyncio.sleep(0.4 * attempt)

        return None, "Unknown fetch failure", "direct"

    async def _fetch_via_jina(self, source_url: str) -> tuple[Optional[str], Optional[str], str]:
        if not self.enable_jina_fallback:
            return None, "Direct source blocked and fallback disabled", "direct"

        url_without_scheme = source_url.split("://", 1)[-1]
        jina_url = f"https://r.jina.ai/http://{url_without_scheme}"
        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds, follow_redirects=True) as client:
                response = await client.get(jina_url)
            response.raise_for_status()
            if not response.text:
                return None, "Empty fallback response body", "jina"
            return response.text, None, "jina"
        except Exception as exc:
            return None, f"Fallback fetch failed: {exc}", "jina"

    def _parse_market_page(self, html_text: str) -> Dict[str, Optional[str]]:
        compact = re.sub(r"\s+", " ", html_text)
        table_rows = self._extract_table_rows(html_text)

        price_updated = self._extract_price_updated_value(html_text) or self._extract_price_updated_value(compact)
        average_price = self._extract_metric(compact, "Average Price")
        lowest_price = self._extract_metric(compact, "Lowest Market Price")
        highest_price = self._extract_metric(compact, "Costliest Market Price")

        price_updated = self._clean_plain_text(price_updated) if price_updated else None
        average_price = self._clean_plain_text(average_price) if average_price else None
        lowest_price = self._clean_plain_text(lowest_price) if lowest_price else None
        highest_price = self._clean_plain_text(highest_price) if highest_price else None

        if table_rows:
            derived_avg, derived_low, derived_high = self._derive_metrics_from_rows(table_rows)
            average_price = average_price or derived_avg
            lowest_price = lowest_price or derived_low
            highest_price = highest_price or derived_high

        return {
            "price_updated": price_updated,
            "average_price": average_price,
            "lowest_price": lowest_price,
            "highest_price": highest_price,
            "table_rows": table_rows,
            "has_any_price": any([average_price, lowest_price, highest_price, table_rows]),
        }

    def _extract_price_updated_value(self, text: str) -> Optional[str]:
        search_text = html.unescape(text or "")
        patterns = [
            r"Price\s*updated\s*:\s*([0-9]{1,2}\s+[A-Za-z]{3}\s*[\'\u2019`]?\d{2},\s*[0-9]{1,2}:[0-9]{2}\s*(?:am|pm))",
            r"Last\s*price\s*updated\s*:\s*([0-9]{1,2}\s+[A-Za-z]{3}\s*[\'\u2019`]?\d{2},\s*[0-9]{1,2}:[0-9]{2}\s*(?:am|pm))",
            r"Price\s*updated\s*:\s*([^\r\n<]+?)(?=\s+(?:Average Price|Lowest Market Price|Costliest Market Price)\b|$)",
            r"Last\s*price\s*updated\s*:\s*([^\r\n<]+?)(?=\s+(?:Average Price|Lowest Market Price|Costliest Market Price)\b|$)",
        ]

        for pattern in patterns:
            value = self._extract_with_regex(search_text, pattern)
            if value:
                return self._clean_plain_text(value)

        return None

    def _format_success_answer(
        self,
        state: str,
        market: str,
        source_url: str,
        fetch_mode: str,
        price_updated: Optional[str],
        avg_price: Optional[str],
        lowest_price: Optional[str],
        highest_price: Optional[str],
        table_rows: List[Dict[str, str]],
        language: str,
    ) -> str:
        now_ist = datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%Y-%m-%d %H:%M:%S IST")
        display_state = self._display_state(state, language)
        display_market = self._display_market(market, language)

        lines = [self._msg(language, "title", market=display_market, state=display_state)]
        lines.append(
            self._msg(
                language,
                "price_updated",
                value=price_updated or self._msg(language, "not_found_on_page"),
            )
        )
        lines.append(self._format_price_line(self._msg(language, "average"), avg_price, language))
        lines.append(self._format_price_line(self._msg(language, "lowest"), lowest_price, language))
        lines.append(self._format_price_line(self._msg(language, "highest"), highest_price, language))
        if table_rows:
            display_rows = table_rows[: max(1, self.max_display_rows)]
            lines.append("")
            lines.append(self._msg(language, "mandi_rows_numbered"))
            lines.extend(self._format_rows_list(display_rows, language))
            if len(table_rows) > len(display_rows):
                lines.append(self._msg(language, "more_rows", count=len(table_rows) - len(display_rows)))
        if fetch_mode == "jina":
            lines.append(self._msg(language, "fetch_mode_jina"))
        lines.append(self._msg(language, "fetched_at", ts=now_ist))
        lines.append(self._msg(language, "source", source=source_url))
        return "\n".join(lines)

    def _format_rows_list(self, rows: List[Dict[str, str]], language: str) -> List[str]:
        output: List[str] = []
        for index, row in enumerate(rows, start=1):
            output.append(
                f"{index}. {row.get('arrival_date', '')} | "
                f"{self._msg(language, 'variety')}: {row.get('variety', '')} | "
                f"{self._msg(language, 'market')}: {self._display_market(row.get('market', ''), language)} | "
                f"{self._msg(language, 'min')}: {row.get('min_price', '')} | "
                f"{self._msg(language, 'max')}: {row.get('max_price', '')} | "
                f"{self._msg(language, 'avg')}: {row.get('avg_price', '')}"
            )
        return output

    def _format_price_line(self, label: str, amount_text: Optional[str], language: str) -> str:
        if not amount_text:
            return f"{label}: {self._msg(language, 'not_available')}"

        value = self._parse_float(amount_text)
        if value is None:
            return f"{label}: Rs {amount_text} / Quintal"

        per_kg = value / 100.0
        return f"{label}: Rs {value:,.2f} / Quintal (Rs {per_kg:,.2f} / kg)"

    def _extract_state_market(self, text: str) -> tuple[Optional[str], Optional[str]]:
        normalized = self._normalize(text)
        state = self._find_best_match(normalized, self._sorted_state_aliases)
        market = self._find_best_match(normalized, self._sorted_market_aliases)
        return state, market

    def _extract_state_market_from_history(
        self,
        conversation_history: List[Dict[str, str]],
    ) -> tuple[Optional[str], Optional[str]]:
        state: Optional[str] = None
        market: Optional[str] = None

        for turn in reversed(conversation_history):
            content = str(turn.get("content", "")).strip()
            if not content:
                continue

            detected_state, detected_market = self._extract_state_market(content)
            if not state and detected_state:
                state = detected_state
            if not market and detected_market:
                market = detected_market

            if not state:
                # Helps when assistant follow-up was: "Please tell me the market in Karnataka."
                followup_state = self._extract_followup_state(content)
                if followup_state:
                    state = followup_state

            if state and market:
                break

        return state, market

    def _extract_followup_state(self, text: str) -> Optional[str]:
        normalized = self._normalize(text)
        match = re.search(r"market in ([a-z0-9 ]+)", normalized)
        if not match:
            return None

        candidate = self._find_best_match(match.group(1), self._sorted_state_aliases)
        if candidate and candidate != "all":
            return candidate
        return None

    def _state_follow_up_message(self, language: str) -> str:
        states = [self._display_state(s, language) for s in AVAILABLE_STATES if s != "all"]
        return (
            f"{self._msg(language, 'ask_state')}\n"
            f"{self._msg(language, 'available_states', states=', '.join(states))}\n"
            f"{self._msg(language, 'source', source=self._source_template_url())}"
        )

    def _market_follow_up_message(self, state: str, language: str) -> str:
        state_slug = self._to_slug(state)
        state_source = f"{BASE_SOURCE_URL}/{COMMODITY_SLUG}/{state_slug}"
        display_state = self._display_state(state, language)
        display_markets = [self._display_market(market, language) for market in AVAILABLE_MARKETS]
        return (
            f"{self._msg(language, 'ask_market', state=display_state)}\n"
            f"{self._msg(language, 'available_markets', markets=', '.join(display_markets))}\n"
            f"{self._msg(language, 'source', source=state_source)}"
        )

    def _source_template_url(self) -> str:
        return f"{BASE_SOURCE_URL}/{COMMODITY_SLUG}"

    def _candidate_market_slugs(self, market: str) -> list[str]:
        slugs = [self._to_slug(market)]
        if market == "shimoga(theertahalli)":
            slugs.append("shimoga-theertahalli")
        if market == "thirthahalli":
            slugs.append("thirthahalli")
            slugs.append("theertahalli")
        return list(dict.fromkeys(slugs))

    def _extract_metric(self, text: str, label: str) -> Optional[str]:
        pattern = rf"{re.escape(label)}[\s\S]{{0,250}}?(?:₹|Rs\.?\s*)\s*([0-9,]+(?:\.[0-9]+)?)\s*/\s*Quintal"
        return self._extract_with_regex(text, pattern)

    def _extract_table_rows(self, page_text: str) -> List[Dict[str, str]]:
        rows = self._extract_rows_from_html_table(page_text)
        if not rows:
            rows = self._extract_rows_from_tabular_text(page_text)
        if not rows:
            rows = self._extract_rows_from_markdown_table(page_text)
        if not rows:
            return []

        deduped: List[Dict[str, str]] = []
        seen = set()
        for row in rows:
            key = (
                row.get("arrival_date", ""),
                row.get("variety", ""),
                row.get("market", ""),
                row.get("min_price", ""),
                row.get("max_price", ""),
                row.get("avg_price", ""),
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(row)

        return deduped[: self.max_table_rows]

    def _extract_rows_from_html_table(self, page_text: str) -> List[Dict[str, str]]:
        table_match = re.search(
            r"<table[^>]*id=[\"']main-table2[\"'][^>]*>([\s\S]*?)</table>",
            page_text,
            flags=re.IGNORECASE,
        )
        if not table_match:
            return []

        table_html = table_match.group(1)
        row_blocks = re.findall(r"<tr[^>]*>([\s\S]*?)</tr>", table_html, flags=re.IGNORECASE)
        rows: List[Dict[str, str]] = []
        for row_html in row_blocks:
            cells = re.findall(r"<t[dh][^>]*>([\s\S]*?)</t[dh]>", row_html, flags=re.IGNORECASE)
            clean_cells = [self._clean_cell(cell) for cell in cells]
            if len(clean_cells) < 9:
                continue
            if not re.match(r"\d{2}/\d{2}/\d{4}", clean_cells[1]):
                continue

            rows.append(
                {
                    "arrival_date": clean_cells[1],
                    "variety": clean_cells[2],
                    "market": clean_cells[5],
                    "min_price": clean_cells[6],
                    "max_price": clean_cells[7],
                    "avg_price": clean_cells[8],
                }
            )
        return rows

    def _extract_rows_from_tabular_text(self, page_text: str) -> List[Dict[str, str]]:
        rows: List[Dict[str, str]] = []
        for raw_line in page_text.splitlines():
            line = raw_line.strip()
            if not line or "\t" not in line:
                continue

            parts = [segment.strip() for segment in line.split("\t") if segment.strip()]
            if len(parts) < 9:
                continue
            if not re.match(r"\d{2}/\d{2}/\d{4}", parts[1]):
                continue

            rows.append(
                {
                    "arrival_date": parts[1],
                    "variety": parts[2],
                    "market": parts[5],
                    "min_price": parts[6],
                    "max_price": parts[7],
                    "avg_price": parts[8],
                }
            )
        return rows

    def _extract_rows_from_markdown_table(self, page_text: str) -> List[Dict[str, str]]:
        rows: List[Dict[str, str]] = []
        for raw_line in page_text.splitlines():
            line = raw_line.strip()
            if not line.startswith("|"):
                continue
            if "Arrival Date" in line or re.match(r"^\|\s*-+", line):
                continue

            parts = [segment.strip() for segment in line.split("|")[1:-1]]
            if len(parts) < 9:
                continue

            date_value = self._clean_markdown_cell(parts[1])
            if not re.match(r"\d{2}/\d{2}/\d{4}", date_value):
                continue

            rows.append(
                {
                    "arrival_date": date_value,
                    "variety": self._clean_markdown_cell(parts[2]),
                    "market": self._clean_markdown_cell(parts[5]),
                    "min_price": self._clean_markdown_cell(parts[6]),
                    "max_price": self._clean_markdown_cell(parts[7]),
                    "avg_price": self._clean_markdown_cell(parts[8]),
                }
            )
        return rows

    def _derive_metrics_from_rows(self, rows: List[Dict[str, str]]) -> tuple[Optional[str], Optional[str], Optional[str]]:
        min_values: List[float] = []
        max_values: List[float] = []
        avg_values: List[float] = []

        for row in rows:
            min_value = self._extract_price_value(row.get("min_price", ""))
            max_value = self._extract_price_value(row.get("max_price", ""))
            avg_value = self._extract_price_value(row.get("avg_price", ""))

            if min_value is not None:
                min_values.append(min_value)
            if max_value is not None:
                max_values.append(max_value)
            if avg_value is not None:
                avg_values.append(avg_value)

        derived_avg = f"{(sum(avg_values) / len(avg_values)):.2f}" if avg_values else None
        derived_min = f"{min(min_values):.2f}" if min_values else None
        derived_max = f"{max(max_values):.2f}" if max_values else None
        return derived_avg, derived_min, derived_max

    def _clean_cell(self, raw: str) -> str:
        no_tags = re.sub(r"<[^>]+>", " ", raw)
        unescaped = html.unescape(no_tags)
        return re.sub(r"\s+", " ", unescaped).strip()

    def _clean_markdown_cell(self, raw: str) -> str:
        text = raw.strip()
        text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
        text = re.sub(r"[*_`]+", "", text)
        text = html.unescape(text)
        return re.sub(r"\s+", " ", text).strip()

    def _clean_plain_text(self, raw: str) -> str:
        unescaped = html.unescape(raw or "")
        return re.sub(r"\s+", " ", unescaped).strip()

    def _extract_price_value(self, text: str) -> Optional[float]:
        match = re.search(r"([0-9][0-9,]*(?:\.[0-9]+)?)", text or "")
        if not match:
            return None
        return self._parse_float(match.group(1))

    def _extract_with_regex(self, text: str, pattern: str) -> Optional[str]:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            return None

        groups = [g for g in match.groups() if g is not None] if match.groups() else []
        if groups:
            return groups[0].strip()
        return match.group(1).strip() if match.lastindex else None

    def _find_best_match(self, normalized_text: str, pairs: list[tuple[str, str]]) -> Optional[str]:
        bounded = f" {normalized_text} "
        for alias, canonical in pairs:
            if alias == "all":
                continue
            token = f" {alias} "
            if token in bounded:
                return canonical
        return None

    def _build_aliases(self, canonical_items: list[str], extras: Dict[str, str]) -> Dict[str, str]:
        aliases: Dict[str, str] = {}
        for item in canonical_items:
            aliases[self._normalize(item)] = item
            aliases[self._normalize(item.replace("-", " "))] = item
            aliases[self._normalize(item.replace(".", " "))] = item
            aliases[self._normalize(item.replace("(", " ").replace(")", " "))] = item

        for alias, canonical in extras.items():
            aliases[self._normalize(alias)] = canonical

        return aliases

    def _to_slug(self, value: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "-", value.lower())
        return slug.strip("-")

    def _normalize(self, value: str) -> str:
        lowered = value.lower().replace("&", " and ")
        lowered = re.sub(r"[^\w]+", " ", lowered, flags=re.UNICODE)
        return re.sub(r"\s+", " ", lowered).strip()

    def _resolve_language(
        self,
        user_query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        language = self._detect_language(user_query)
        if language == LANG_KN:
            return LANG_KN

        if not conversation_history:
            return LANG_EN

        for turn in reversed(conversation_history[-6:]):
            content = str(turn.get("content", "")).strip()
            if not content:
                continue
            if self._detect_language(content) == LANG_KN:
                return LANG_KN

        return LANG_EN

    def _detect_language(self, text: str) -> str:
        if re.search(r"[\u0C80-\u0CFF]", text or ""):
            return LANG_KN
        return LANG_EN

    def _display_state(self, state: str, language: str) -> str:
        if language == LANG_KN:
            return STATE_DISPLAY_KN.get(state, state)
        return state

    def _display_market(self, market: str, language: str) -> str:
        normalized_market = (market or "").strip().lower()
        if language == LANG_KN:
            return MARKET_DISPLAY_KN.get(normalized_market, market)
        return market

    def _msg(self, language: str, key: str, **kwargs) -> str:
        en = {
            "feature_disabled": "Arecanut price feature is currently disabled.",
            "source": "Source: {source}",
            "fetching_message": "Fetching latest arecanut mandi prices... Just a moment.",
            "fetch_failed": "I could not fetch live arecanut mandi prices for {market}, {state} right now.",
            "reason": "Reason: {reason}",
            "title": "Arecanut mandi price in {market}, {state}",
            "price_updated": "Price updated: {value}",
            "not_found_on_page": "Not found on page",
            "average": "Average",
            "lowest": "Lowest",
            "highest": "Highest",
            "not_available": "Not available",
            "mandi_rows_numbered": "Mandi rows (numbered):",
            "more_rows": "... and {count} more row(s).",
            "fetch_mode_jina": "Fetch mode: text-mirror fallback (source is automation-protected)",
            "fetched_at": "Fetched at: {ts}",
            "variety": "Variety",
            "market": "Market",
            "min": "Min",
            "max": "Max",
            "avg": "Avg",
            "ask_state": "Please tell me the state to fetch arecanut mandi price.",
            "available_states": "Available states: {states}",
            "ask_market": "Please tell me the market in {state}.",
            "available_markets": "Available markets: {markets}",
        }
        kn = {
            "feature_disabled": "ಅಡಕೆ ಬೆಲೆ ವೈಶಿಷ್ಟ್ಯವನ್ನು ಈಗ ನಿಷ್ಕ್ರಿಯಗೊಳಿಸಲಾಗಿದೆ.",
            "source": "ಮೂಲ: {source}",
            "fetching_message": "ಇತ್ತೀಚಿನ ಅಡಕೆ ಮಾರುಕಟ್ಟೆ ಬೆಲೆಗಳನ್ನು ಪಡೆಯಲಾಗುತ್ತಿದೆ... ಒಂದು ಕ್ಷಣ.",
            "fetch_failed": "{market}, {state}ಗಾಗಿ ಲೈವ್ ಅಡಕೆ ಮಾರುಕಟ್ಟೆ ಬೆಲೆ ಪಡೆಯಲಾಗಲಿಲ್ಲ.",
            "reason": "ಕಾರಣ: {reason}",
            "title": "{market}, {state}ನಲ್ಲಿ ಅಡಕೆ ಮಾರುಕಟ್ಟೆ ಬೆಲೆ",
            "price_updated": "ಬೆಲೆ ನವೀಕರಣ: {value}",
            "not_found_on_page": "ಪುಟದಲ್ಲಿ ಕಂಡುಬಂದಿಲ್ಲ",
            "average": "ಸರಾಸರಿ",
            "lowest": "ಕನಿಷ್ಠ",
            "highest": "ಅತ್ಯಧಿಕ",
            "not_available": "ಲಭ್ಯವಿಲ್ಲ",
            "mandi_rows_numbered": "ಮಾರುಕಟ್ಟೆ ಸಾಲುಗಳು (ಕ್ರಮ ಸಂಖ್ಯೆ):",
            "more_rows": "... ಇನ್ನೂ {count} ಸಾಲು(ಗಳು) ಇವೆ.",
            "fetch_mode_jina": "ಫೆಚ್ ಮೋಡ್: ಟೆಕ್ಸ್ಟ್-ಮಿರರ್ ಫಾಲ್ಬ್ಯಾಕ್ (ಮೂಲ ತಾಣ ಸ್ವಯಂಚಾಲಿತ ಪ್ರವೇಶವನ್ನು ನಿರ್ಬಂಧಿಸಿದೆ)",
            "fetched_at": "ಪಡೆಯಲಾದ ಸಮಯ: {ts}",
            "variety": "ವೈವಿಧ್ಯ",
            "market": "ಮಾರುಕಟ್ಟೆ",
            "min": "ಕನಿಷ್ಠ",
            "max": "ಗರಿಷ್ಠ",
            "avg": "ಸರಾಸರಿ",
            "ask_state": "ಅಡಕೆ ಮಾರುಕಟ್ಟೆ ಬೆಲೆ ಪಡೆಯಲು ದಯವಿಟ್ಟು ರಾಜ್ಯವನ್ನು ತಿಳಿಸಿ.",
            "available_states": "ಲಭ್ಯ ರಾಜ್ಯಗಳು: {states}",
            "ask_market": "ದಯವಿಟ್ಟು {state} ರಾಜ್ಯದ ಮಾರುಕಟ್ಟೆ ಹೆಸರನ್ನು ತಿಳಿಸಿ.",
            "available_markets": "ಲಭ್ಯ ಮಾರುಕಟ್ಟೆಗಳು: {markets}",
        }
        template = kn.get(key) if language == LANG_KN else en.get(key)
        if not template:
            template = en.get(key, key)
        return template.format(**kwargs)

    def _parse_float(self, value: str) -> Optional[float]:
        try:
            return float(value.replace(",", ""))
        except Exception:
            return None

    @staticmethod
    def _as_bool(value: Optional[str], default: bool) -> bool:
        if value is None:
            return default
        return value.strip().lower() in {"1", "true", "yes", "on"}
