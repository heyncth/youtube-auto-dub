"""Google Translate client with RPC and scrape fallback."""

import asyncio
import json
import re
from typing import List

import httpx
from bs4 import BeautifulSoup

from youtube_auto_dub.models import (
    TRANSLATE_API_URL,
    TRANSLATE_SCRAPE_URL,
    TRANSLATE_TIMEOUT,
    TRANSLATE_TOKEN_URL,
    TRANSLATE_USER_AGENT,
)
from youtube_auto_dub.ui import console

BATCH_CHUNK = 25
BATCH_CONCURRENCY = 4


class GoogleTranslator:
    """Google Translate client with RPC (primary) and web scrape (fallback)."""

    def __init__(self):
        self.client = httpx.AsyncClient(timeout=TRANSLATE_TIMEOUT, follow_redirects=True)
        self.base_url_rpc = TRANSLATE_API_URL
        self.base_url_scrape = TRANSLATE_SCRAPE_URL
        self.headers = {"User-Agent": TRANSLATE_USER_AGENT}
        self.bl = None

    async def _refresh_rpc_token(self):
        try:
            response = await self.client.get(TRANSLATE_TOKEN_URL, headers=self.headers)
            bl_match = re.search(r'"cfb2h":"(.*?)"', response.text)
            if bl_match:
                self.bl = bl_match.group(1)
            else:
                self.bl = "boq_translate-webserver_20251215.06_p0"
        except Exception as e:
            console.warning(f"Token refresh failed: {e}. Using fallback.")
            self.bl = "boq_translate-webserver_20251215.06_p0"

    async def _parse_rpc_response(self, raw_text):
        try:
            match = re.search(r'\["wrb.fr","MkEWBc","(.*?)",null,null,null,"generic"\]', raw_text, re.DOTALL)
            if not match:
                raise ValueError("Could not find translation data in RPC response.")
            inner_json_str = match.group(1).replace('\\"', '"').replace("\\\\", "\\")
            data = json.loads(inner_json_str)
            translation_parts = data[1][0][0][5]
            final_text = " ".join([part[0] for part in translation_parts if part[0]])
            return final_text
        except Exception as e:
            raise ValueError(f"RPC Parse Error: {e}")

    async def _translate_rpc(self, text, source, target):
        """Method 1: fake browser API requests."""
        if not self.bl:
            await self._refresh_rpc_token()

        rpc_arg = json.dumps([[text, source, target, True, [1]]], ensure_ascii=False)
        f_req = json.dumps([["MkEWBc", rpc_arg, None, "generic"]])

        params = {"rpcids": "MkEWBc", "bl": self.bl, "hl": "en", "rt": "c"}

        response = await self.client.post(
            self.base_url_rpc,
            headers=self.headers,
            params=params,
            data={"f.req": f_req},
        )

        if response.status_code != 200:
            raise Exception(f"RPC HTTP Error: {response.status_code}")

        return await self._parse_rpc_response(response.text)

    async def _translate_scrape(self, text, source, target):
        """Method 2: Web scraping. Simple fallback."""
        params = {"sl": source, "tl": target, "q": text}

        response = await self.client.get(self.base_url_scrape, params=params, headers=self.headers)

        if response.status_code == 429:
            raise Exception("Too Many Requests (429)")
        if response.status_code != 200:
            raise Exception(f"Scrape HTTP Error: {response.text}")

        soup = BeautifulSoup(response.text, "html.parser")
        element = soup.find("div", {"class": "t0"})
        if not element:
            element = soup.find("div", {"class": "result-container"})
        if not element:
            raise Exception("Could not find translation element in HTML.")

        return element.get_text(strip=True)

    async def translate(self, text, source="auto", target="vi"):
        """Main interface. Tries RPC first, falls back to scraping.

        Raises RuntimeError when both methods fail: returning the input text
        here would render an untranslated video as if it succeeded.
        """
        if not text:
            return ""

        # Source matches target — skip translation
        if source != "auto" and source == target:
            return text

        errors = []

        # Try RPC API first
        try:
            return await self._translate_rpc(text, source, target)
        except Exception as e:
            errors.append(f"rpc: {e}")

        # Try scrape fallback
        try:
            return await self._translate_scrape(text, source, target)
        except Exception as e:
            errors.append(f"scrape: {e}")

        console.error(f"All translation methods failed: {'; '.join(errors)}")
        raise RuntimeError(f"Translation failed for target {target} ({'; '.join(errors)})")

    async def translate_batch(self, texts: List[str], target: str, source: str = "auto") -> List[str]:
        """Translate texts in small chunks (large GET queries get HTTP 400)."""
        delimiter = "\n\n|||\n\n"
        out: List[str] = []

        for i in range(0, len(texts), BATCH_CHUNK):
            chunk = texts[i:i + BATCH_CHUNK]
            combined = delimiter.join([t if t.strip() else " " for t in chunk])
            translated = await self.translate(combined, source=source, target=target)
            results = [t.strip() for t in translated.split("|||")]
            if len(results) == len(chunk):
                out.extend(results)
                continue

            # Delimiter lost in translation — fall back per text, bounded concurrency
            sem = asyncio.Semaphore(BATCH_CONCURRENCY)

            async def one(t: str) -> str:
                async with sem:
                    return (await self.translate(t, source=source, target=target)).strip()

            out.extend(await asyncio.gather(*[one(t) for t in chunk]))

        return out

    async def close(self):
        await self.client.aclose()
