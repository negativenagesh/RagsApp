import os
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock

SERVICE_ROOT = Path(__file__).resolve().parents[1]
if str(SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVICE_ROOT))

from app.arecanut_price_tool import ArecanutPriceTool


class ArecanutPriceToolTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._previous_env = dict(os.environ)
        os.environ["ARECANUT_PRICE_ENABLED"] = "true"
        os.environ["ARECANUT_PRICE_CACHE_TTL_SECONDS"] = "60"

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._previous_env)

    async def test_missing_state_returns_follow_up_with_source(self):
        tool = ArecanutPriceTool()

        result = await tool.ask("what is arecanut mandi price now")

        self.assertIn("Available states", result)
        self.assertIn("Source:", result)

    async def test_missing_state_returns_kannada_follow_up_for_kannada_query(self):
        tool = ArecanutPriceTool()

        result = await tool.ask("ಅಡಕೆ ಬೆಲೆ ಏನು?")

        self.assertIn("ಲಭ್ಯ ರಾಜ್ಯಗಳು", result)
        self.assertIn("ಮೂಲ:", result)

    async def test_missing_market_returns_follow_up_with_source(self):
        tool = ArecanutPriceTool()

        result = await tool.ask("arecanut price in karnataka")

        self.assertIn("Available markets", result)
        self.assertIn("Source:", result)

    async def test_market_only_follow_up_uses_state_from_history(self):
        tool = ArecanutPriceTool()
        tool._fetch_and_format_response = AsyncMock(return_value="ok")
        history = [
            {"role": "user", "content": "what is arecanut price today"},
            {"role": "assistant", "content": "Please tell me the market in Karnataka."},
        ]

        result = await tool.ask("yellapur", conversation_history=history)

        self.assertEqual(result, "ok")
        tool._fetch_and_format_response.assert_awaited_once_with(
            state="karnataka",
            market="yellapur",
            language="en",
        )

    def test_parse_market_page_extracts_metrics(self):
        tool = ArecanutPriceTool()
        sample_html = (
            "Price updated : 14 Apr &#039;26, 10:05 am "
            "Average Price ₹41765.83/Quintal "
            "Lowest Market Price ₹13109.00/Quintal "
            "Costliest Market Price ₹59375.00/Quintal"
        )

        parsed = tool._parse_market_page(sample_html)

        self.assertEqual(parsed["price_updated"], "14 Apr '26, 10:05 am")
        self.assertEqual(parsed["average_price"], "41765.83")
        self.assertEqual(parsed["lowest_price"], "13109.00")
        self.assertEqual(parsed["highest_price"], "59375.00")

    def test_parse_market_page_extracts_table_rows_from_html(self):
        tool = ArecanutPriceTool()
        sample_html = """
        <div class="col-lg-4"><p>Price updated : 14 Apr '26, 10:05 am</p></div>
        <table id="main-table2">
            <thead>
                <tr><th>Commodity</th><th>Arrival Date</th><th>Variety</th><th>State</th><th>District</th><th>Market</th><th>Min Price</th><th>Max Price</th><th>Avg price</th></tr>
            </thead>
            <tbody>
                <tr>
                    <td>Arecanut(Betelnut/Supari)</td><td>13/04/2026</td><td>Cqca</td><td>Karnataka</td><td>Karwar(Uttar Kannad)</td><td>Yellapur</td><td>Rs 13109 / Quintal</td><td>Rs 34899 / Quintal</td><td>Rs 31309 / Quintal</td>
                </tr>
                <tr>
                    <td>Arecanut(Betelnut/Supari)</td><td>13/04/2026</td><td>Rashi</td><td>Karnataka</td><td>Karwar(Uttar Kannad)</td><td>Yellapur</td><td>Rs 50719 / Quintal</td><td>Rs 59375 / Quintal</td><td>Rs 54689 / Quintal</td>
                </tr>
            </tbody>
        </table>
        """

        parsed = tool._parse_market_page(sample_html)

        self.assertEqual(parsed["price_updated"], "14 Apr '26, 10:05 am")
        self.assertEqual(len(parsed["table_rows"]), 2)
        self.assertEqual(parsed["table_rows"][0]["arrival_date"], "13/04/2026")
        self.assertEqual(parsed["table_rows"][0]["variety"], "Cqca")
        self.assertEqual(parsed["table_rows"][0]["market"], "Yellapur")
        self.assertEqual(parsed["table_rows"][0]["min_price"], "Rs 13109 / Quintal")
        self.assertEqual(parsed["table_rows"][0]["max_price"], "Rs 34899 / Quintal")
        self.assertEqual(parsed["table_rows"][0]["avg_price"], "Rs 31309 / Quintal")

    def test_format_success_answer_includes_rows_table(self):
        tool = ArecanutPriceTool()
        answer = tool._format_success_answer(
            state="karnataka",
            market="yellapur",
            source_url="https://example.com",
            fetch_mode="direct",
            price_updated="14 Apr '26, 10:05 am",
            avg_price="41765.83",
            lowest_price="13109.00",
            highest_price="59375.00",
            table_rows=[
                {
                    "arrival_date": "13/04/2026",
                    "variety": "Cqca",
                    "market": "Yellapur",
                    "min_price": "Rs 13109 / Quintal",
                    "max_price": "Rs 34899 / Quintal",
                    "avg_price": "Rs 31309 / Quintal",
                }
            ],
            language="en",
        )

        self.assertIn("Mandi rows (numbered):", answer)
        self.assertIn(
            "1. 13/04/2026 | Variety: Cqca | Market: Yellapur | Min: Rs 13109 / Quintal | Max: Rs 34899 / Quintal | Avg: Rs 31309 / Quintal",
            answer,
        )

    def test_format_success_answer_in_kannada_for_kannada_query_language(self):
        tool = ArecanutPriceTool()
        answer = tool._format_success_answer(
            state="karnataka",
            market="sirsi",
            source_url="https://example.com",
            fetch_mode="direct",
            price_updated="14 Apr '26, 10:05 am",
            avg_price="41765.83",
            lowest_price="13109.00",
            highest_price="59375.00",
            table_rows=[
                {
                    "arrival_date": "13/04/2026",
                    "variety": "Rashi",
                    "market": "Sirsi",
                    "min_price": "Rs 50719 / Quintal",
                    "max_price": "Rs 59375 / Quintal",
                    "avg_price": "Rs 54689 / Quintal",
                }
            ],
            language="kn",
        )

        self.assertIn("ಅಡಕೆ ಮಾರುಕಟ್ಟೆ ಬೆಲೆ", answer)
        self.assertIn("ಬೆಲೆ ನವೀಕರಣ", answer)
        self.assertIn("ಮಾರುಕಟ್ಟೆ ಸಾಲುಗಳು", answer)
        self.assertIn("ಮಾರುಕಟ್ಟೆ: ಸಿರಸಿ", answer)

    def test_get_fetching_message_uses_query_language(self):
        tool = ArecanutPriceTool()

        self.assertEqual(
            tool.get_fetching_message("what is arecanut price in sirsi"),
            "Fetching latest arecanut mandi prices... Just a moment.",
        )
        self.assertEqual(
            tool.get_fetching_message("ಸಿರಸಿ ಅಡಕೆ ಬೆಲೆ"),
            "ಇತ್ತೀಚಿನ ಅಡಕೆ ಮಾರುಕಟ್ಟೆ ಬೆಲೆಗಳನ್ನು ಪಡೆಯಲಾಗುತ್ತಿದೆ... ಒಂದು ಕ್ಷಣ.",
        )

    async def test_state_selected_in_english_keeps_kannada_from_context(self):
        tool = ArecanutPriceTool()
        history = [
            {"role": "user", "content": "ಇಂದಿನ ಅಡಿಕೆ ಬೆಲೆ"},
            {"role": "assistant", "content": "ಅಡಕೆ ಮಾರುಕಟ್ಟೆ ಬೆಲೆ ಪಡೆಯಲು ದಯವಿಟ್ಟು ರಾಜ್ಯವನ್ನು ತಿಳಿಸಿ."},
        ]

        result = await tool.ask("Karnataka", conversation_history=history)

        self.assertIn("ದಯವಿಟ್ಟು ಕರ್ನಾಟಕ ರಾಜ್ಯದ ಮಾರುಕಟ್ಟೆ ಹೆಸರನ್ನು ತಿಳಿಸಿ.", result)
        self.assertIn("ಲಭ್ಯ ಮಾರುಕಟ್ಟೆಗಳು", result)
        self.assertIn("ಬೆಂಗಳೂರು", result)
        self.assertIn("ಅರಸೀಕೆರೆ", result)
        self.assertNotIn("bangalore", result.lower())
        self.assertIn("ಮೂಲ:", result)

    def test_fetching_message_uses_kannada_from_context(self):
        tool = ArecanutPriceTool()
        history = [
            {"role": "user", "content": "ಇಂದಿನ ಅಡಿಕೆ ಬೆಲೆ"},
            {"role": "assistant", "content": "ಅಡಕೆ ಮಾರುಕಟ್ಟೆ ಬೆಲೆ ಪಡೆಯಲು ದಯವಿಟ್ಟು ರಾಜ್ಯವನ್ನು ತಿಳಿಸಿ."},
        ]

        result = tool.get_fetching_message("Karnataka", conversation_history=history)

        self.assertEqual(result, "ಇತ್ತೀಚಿನ ಅಡಕೆ ಮಾರುಕಟ್ಟೆ ಬೆಲೆಗಳನ್ನು ಪಡೆಯಲಾಗುತ್ತಿದೆ... ಒಂದು ಕ್ಷಣ.")

    def test_parse_market_page_extracts_rows_from_tabular_text(self):
        tool = ArecanutPriceTool()
        sample_text = (
            "Price updated : 14 Apr '26, 10:05 am\n"
            "Commodity\tArrival Date\tVariety\tState\tDistrict\tMarket\tMin Price\tMax Price\tAvg price\n"
            "Arecanut(Betelnut/Supari)\t13/04/2026\tBette\tKarnataka\tKarwar(Uttar Kannad)\tYellapur\tRs 35699 / Quintal\tRs 44789 / Quintal\tRs 43299 / Quintal\n"
        )

        parsed = tool._parse_market_page(sample_text)

        self.assertEqual(parsed["price_updated"], "14 Apr '26, 10:05 am")
        self.assertEqual(len(parsed["table_rows"]), 1)
        self.assertEqual(parsed["table_rows"][0]["variety"], "Bette")
        self.assertEqual(parsed["table_rows"][0]["market"], "Yellapur")

    def test_parse_market_page_extracts_rows_from_markdown_table(self):
        tool = ArecanutPriceTool()
        sample_text = (
            "Price updated : 14 Apr '26, 10:05 am\n"
            "| Commodity | Arrival Date | Variety | State | District | Market | Min Price | Max Price | Avg price | Mobile App |\n"
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n"
            "| [Arecanut(Betelnut/Supari)](https://www.commodityonline.com/mandiprices/arecanut-betelnutsupari) | 13/04/2026 | Cqca | [Karnataka](https://example.com/state) | [Karwar(Uttar Kannad)](https://example.com/district) | [Yellapur](https://example.com/market) | Rs 13109 / Quintal | Rs 34899 / Quintal | Rs 31309 / Quintal | <button>View</button> |\n"
        )

        parsed = tool._parse_market_page(sample_text)

        self.assertEqual(parsed["price_updated"], "14 Apr '26, 10:05 am")
        self.assertEqual(len(parsed["table_rows"]), 1)
        self.assertEqual(parsed["table_rows"][0]["arrival_date"], "13/04/2026")
        self.assertEqual(parsed["table_rows"][0]["variety"], "Cqca")
        self.assertEqual(parsed["table_rows"][0]["market"], "Yellapur")
        self.assertEqual(parsed["table_rows"][0]["min_price"], "Rs 13109 / Quintal")

    def test_parse_market_page_extracts_price_updated_with_curly_apostrophe(self):
        tool = ArecanutPriceTool()
        sample_text = (
            "Price updated : 15 Apr ’26, 10:05 am\n"
            "Average Price ₹41765.83/Quintal\n"
            "Lowest Market Price ₹13109.00/Quintal\n"
            "Costliest Market Price ₹59375.00/Quintal\n"
        )

        parsed = tool._parse_market_page(sample_text)

        self.assertEqual(parsed["price_updated"], "15 Apr ’26, 10:05 am")

    def test_format_success_answer_uses_ist_timezone_label(self):
        tool = ArecanutPriceTool()
        answer = tool._format_success_answer(
            state="karnataka",
            market="yellapur",
            source_url="https://example.com",
            fetch_mode="direct",
            price_updated="15 Apr '26, 10:05 am",
            avg_price="41765.83",
            lowest_price="13109.00",
            highest_price="59375.00",
            table_rows=[],
            language="kn",
        )

        self.assertIn("IST", answer)
        self.assertNotIn("UTC", answer)

if __name__ == "__main__":
    unittest.main()
