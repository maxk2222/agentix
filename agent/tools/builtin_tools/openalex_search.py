"""OpenAlex search builtin tool."""

from __future__ import annotations

import asyncio
import logging
import re
import urllib.parse
from io import BytesIO
from typing import Any, Dict, List, Optional

import aiohttp

from .common import get_headers

logger = logging.getLogger(__name__)


class OpenAlexSearchTool:
    """Search the OpenAlex academic graph API for scholarly entities."""

    name = "openalex_search"
    description = (
        "Search the OpenAlex academic graph for scholarly works, authors, "
        "institutions, concepts, venues, funders, or publishers. "
        "Optionally fetches and extracts text from public PDFs for open-access works. "
        "An optional API key can be configured in the tool settings for premium access."
    )

    @property
    def tool_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query.",
                        },
                        "entity": {
                            "type": "string",
                            "description": (
                                "Entity type to search: works, authors, institutions, "
                                "concepts, venues, funders, publishers. Default: works."
                            ),
                            "default": "works",
                        },
                        "filters": {
                            "type": "object",
                            "description": (
                                "Optional structured filters as key-value pairs, "
                                "e.g. {'publication_year': '2020-2024', 'is_oa': true}."
                            ),
                        },
                        "per_page": {
                            "type": "integer",
                            "description": "Number of results per page (default 5, max 10).",
                            "default": 5,
                        },
                        "sort": {
                            "type": "string",
                            "description": "Sort key, e.g. 'cited_by_count:desc' or 'relevance_score:desc'.",
                        },
                        "fetch_pdfs": {
                            "type": "boolean",
                            "description": (
                                "If true, automatically download and extract text from "
                                "public PDFs for open-access works. Default: false."
                            ),
                            "default": False,
                        },
                    },
                    "required": ["query"],
                },
            },
        }

    async def execute(
        self,
        query: str,
        entity: str = "works",
        filters: Optional[Dict[str, Any]] = None,
        per_page: int = 5,
        sort: Optional[str] = None,
        fetch_pdfs: bool = False,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        valid_entities = {
            "works",
            "authors",
            "institutions",
            "concepts",
            "venues",
            "funders",
            "publishers",
        }
        if entity not in valid_entities:
            raise ValueError(
                f"Invalid entity '{entity}'. Must be one of: {', '.join(sorted(valid_entities))}"
            )

        tool_config = (context or {}).get("tool_config", {})
        params: Dict[str, Any] = {
            "search": query,
            "per_page": per_page,
        }
        mailto = tool_config.get("mailto", "")
        if mailto:
            params["mailto"] = mailto
        if sort:
            params["sort"] = sort
        if filters:
            filter_parts = []
            for key, value in filters.items():
                if isinstance(value, bool):
                    filter_parts.append(f"{key}:{str(value).lower()}")
                else:
                    filter_parts.append(f"{key}:{value}")
            if filter_parts:
                params["filter"] = ",".join(filter_parts)

        api_key = tool_config.get("api_key")
        if api_key:
            params["api_key"] = api_key

        url = f"https://api.openalex.org/{entity}"
        timeout = aiohttp.ClientTimeout(total=60)
        last_error = ""
        async with aiohttp.ClientSession(timeout=timeout) as session:
            for attempt in range(4):
                async with session.get(url, params=params) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        break
                    last_error = await resp.text()
                    if resp.status in (429, 503) and attempt < 3:
                        await asyncio.sleep(2 ** attempt)
                        continue
                    if attempt < 3:
                        raise RuntimeError(
                            f"OpenAlex request failed with HTTP {resp.status}. "
                            f"This may be due to rate limiting or temporary outage. "
                            f"If it persists, try setting a mailto address or API key in the tool configuration to access the polite/premium pool."
                        )
            else:
                raise RuntimeError(
                    f"OpenAlex request failed after retries. "
                    f"This may be due to rate limiting or temporary outage. "
                    f"If it persists, try setting a mailto address or API key in the tool configuration to access the polite/premium pool."
                )

        results = data.get("results", [])
        if not results:
            return f"No results found for: {query} (entity: {entity})"

        output_lines: List[str] = [
            f"OpenAlex search results for: {query} (entity: {entity})",
            "",
        ]

        for idx, item in enumerate(results, 1):
            output_lines.append(self._format_item(item, entity, idx))
            output_lines.append("")

        if fetch_pdfs and entity == "works":
            output_lines.append("--- PDF Content ---")
            semaphore = asyncio.Semaphore(3)

            async def _fetch_pdf_for_work(item: Dict[str, Any]) -> str:
                oa_info = item.get("open_access", {})
                if not oa_info.get("is_oa"):
                    return ""
                oa_url = oa_info.get("oa_url")
                if not oa_url:
                    return ""
                async with semaphore:
                    try:
                        text = await self._download_and_extract_pdf(oa_url)
                        return f"\n--- PDF: {oa_url} ---\n{text}\n"
                    except Exception as exc:
                        return f"\n[PDF nicht lesbar: {oa_url} — {exc}]\n"

            pdf_tasks = [_fetch_pdf_for_work(item) for item in results]
            pdf_outputs = await asyncio.gather(*pdf_tasks)
            output_lines.extend([out for out in pdf_outputs if out])

        return "\n".join(output_lines)

    def _format_item(self, item: Dict[str, Any], entity: str, idx: int) -> str:
        lines: List[str] = []
        if entity == "works":
            lines.append(f"{idx}. {item.get('display_name', 'Untitled')}")
            authors = item.get("authorships", [])
            if authors:
                author_names = []
                for a in authors:
                    author = a.get("author", {})
                    name = author.get("display_name", "")
                    orcid = author.get("orcid", "")
                    if name:
                        if orcid:
                            author_names.append(f"{name} ({orcid})")
                        else:
                            author_names.append(name)
                if author_names:
                    lines.append(f"   Authors: {', '.join(author_names)}")
            abstract = item.get("abstract", "") or ""
            if abstract:
                truncated = abstract[:500] + "..." if len(abstract) > 500 else abstract
                lines.append(f"   Abstract: {truncated}")
            doi = item.get("doi", "")
            if doi:
                lines.append(f"   DOI: {doi}")
            pub_date = item.get("publication_date", "")
            if pub_date:
                lines.append(f"   Published: {pub_date}")
            work_type = item.get("type", "")
            if work_type:
                lines.append(f"   Type: {work_type}")
            cited = item.get("cited_by_count")
            if cited is not None:
                lines.append(f"   Cited by: {cited}")
            oa = item.get("open_access", {})
            if oa:
                is_oa = oa.get("is_oa", False)
                oa_status = oa.get("oa_status", "")
                oa_url = oa.get("oa_url", "")
                parts = []
                if is_oa:
                    parts.append("Open Access")
                if oa_status:
                    parts.append(f"Status: {oa_status}")
                if oa_url:
                    parts.append(f"URL: {oa_url}")
                if parts:
                    lines.append(f"   OA: {'; '.join(parts)}")
            topics = item.get("topics", [])
            if topics:
                topic_names = [
                    t.get("display_name", "") for t in topics if t.get("display_name")
                ]
                if topic_names:
                    lines.append(f"   Topics: {', '.join(topic_names[:5])}")
            host = item.get("host_venue", {}) or item.get("primary_location", {}).get("source", {})
            if host and host.get("display_name"):
                lines.append(f"   Venue: {host['display_name']}")

        elif entity == "authors":
            lines.append(f"{idx}. {item.get('display_name', 'Unknown')}")
            orcid = item.get("orcid", "")
            if orcid:
                lines.append(f"   ORCID: {orcid}")
            works_count = item.get("works_count")
            if works_count is not None:
                lines.append(f"   Works: {works_count}")
            cited = item.get("cited_by_count")
            if cited is not None:
                lines.append(f"   Cited by: {cited}")
            last_known = item.get("last_known_institution", {})
            if last_known and last_known.get("display_name"):
                lines.append(f"   Institution: {last_known['display_name']}")

        elif entity == "institutions":
            lines.append(f"{idx}. {item.get('display_name', 'Unknown')}")
            country = item.get("country_code", "")
            if country:
                lines.append(f"   Country: {country}")
            works_count = item.get("works_count")
            if works_count is not None:
                lines.append(f"   Works: {works_count}")
            cited = item.get("cited_by_count")
            if cited is not None:
                lines.append(f"   Cited by: {cited}")
            homepage = item.get("homepage_url", "")
            if homepage:
                lines.append(f"   Homepage: {homepage}")

        elif entity == "concepts":
            lines.append(f"{idx}. {item.get('display_name', 'Unknown')}")
            level = item.get("level")
            if level is not None:
                lines.append(f"   Level: {level}")
            works_count = item.get("works_count")
            if works_count is not None:
                lines.append(f"   Works: {works_count}")
            cited = item.get("cited_by_count")
            if cited is not None:
                lines.append(f"   Cited by: {cited}")
            description = item.get("description", "")
            if description:
                lines.append(f"   Description: {description}")

        elif entity == "venues":
            lines.append(f"{idx}. {item.get('display_name', 'Unknown')}")
            publisher = item.get("publisher", "")
            if publisher:
                lines.append(f"   Publisher: {publisher}")
            works_count = item.get("works_count")
            if works_count is not None:
                lines.append(f"   Works: {works_count}")
            cited = item.get("cited_by_count")
            if cited is not None:
                lines.append(f"   Cited by: {cited}")
            issn = item.get("issn_l", "") or item.get("issn", "")
            if issn:
                lines.append(f"   ISSN: {issn}")

        elif entity == "funders":
            lines.append(f"{idx}. {item.get('display_name', 'Unknown')}")
            country = item.get("country_code", "")
            if country:
                lines.append(f"   Country: {country}")
            works_count = item.get("works_count")
            if works_count is not None:
                lines.append(f"   Works: {works_count}")
            cited = item.get("cited_by_count")
            if cited is not None:
                lines.append(f"   Cited by: {cited}")
            description = item.get("description", "")
            if description:
                lines.append(f"   Description: {description}")

        elif entity == "publishers":
            lines.append(f"{idx}. {item.get('display_name', 'Unknown')}")
            works_count = item.get("works_count")
            if works_count is not None:
                lines.append(f"   Works: {works_count}")
            cited = item.get("cited_by_count")
            if cited is not None:
                lines.append(f"   Cited by: {cited}")
            homepage = item.get("homepage_url", "")
            if homepage:
                lines.append(f"   Homepage: {homepage}")

        else:
            lines.append(f"{idx}. {item.get('display_name', 'Unknown')}")

        return "\n".join(lines)

    def _resolve_pdf_url(self, url: str) -> str:
        """Convert common landing-page URLs to direct PDF links."""
        parsed = urllib.parse.urlparse(url)
        netloc = parsed.netloc.lower()
        path = parsed.path

        # arXiv abstract -> PDF
        if netloc in ("arxiv.org", "www.arxiv.org"):
            match = re.match(r"/abs/(\d+\.\d+)(v\d+)?$", path)
            if match:
                return f"https://arxiv.org/pdf/{match.group(1)}{match.group(2) or ''}.pdf"

        # DOI resolver -> keep as-is, let redirect follow
        if netloc in ("doi.org", "dx.doi.org", "hdl.handle.net"):
            return url

        # PubMed Central
        if netloc in ("pubmed.ncbi.nlm.nih.gov", "www.ncbi.nlm.nih.gov"):
            pmc_match = re.search(r"/pmc/articles/PMC(\d+)", path)
            if pmc_match:
                return f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmc_match.group(1)}/pdf"

        # Europe PMC
        if netloc in ("europepmc.org", "www.europepmc.org"):
            pmc_match = re.match(r"/articles/pmc(\d+)", path)
            if pmc_match:
                return f"https://europepmc.org/backend/ptpmcrender.fcgi?accid=PMC{pmc_match.group(1)}&blobtype=pdf"

        # HAL
        if netloc in ("hal.science", "hal.archives-ouvertes.fr"):
            hal_match = re.match(r"/hal-\d+/(\d+)", path)
            if hal_match:
                return f"https://{netloc}/hal-{hal_match.group(1)}/file/{hal_match.group(1)}.pdf"

        return url

    async def _download_and_extract_pdf(self, url: str) -> str:
        """Download a PDF and extract text using pypdf."""
        if not url.startswith(("http://", "https://")):
            raise ValueError("PDF URL must start with http:// or https://")

        resolved_url = self._resolve_pdf_url(url)

        max_size = 20 * 1024 * 1024  # 20 MB
        timeout = aiohttp.ClientTimeout(total=60)
        pdf_headers = {
            "User-Agent": "Mozilla/5.0 (compatible; Agentix-Bot/1.0; +https://github.com)",
            "Accept": "application/pdf,*/*;q=0.8",
        }

        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(
                resolved_url,
                headers=pdf_headers,
                allow_redirects=True,
                max_redirects=10,
            ) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"HTTP {resp.status}")
                content_length = resp.headers.get("Content-Length")
                if content_length and int(content_length) > max_size:
                    raise RuntimeError("PDF too large (>20 MB)")
                pdf_bytes = await resp.read()
                if len(pdf_bytes) > max_size:
                    raise RuntimeError("PDF too large (>20 MB)")
                content_type = resp.headers.get("Content-Type", "").lower()
                if content_type and not any(
                    ct in content_type for ct in ("application/pdf", "application/octet-stream", "binary/octet-stream")
                ):
                    # Some servers send wrong Content-Type; check PDF magic bytes
                    if not pdf_bytes.startswith(b"%PDF"):
                        raise RuntimeError(
                            f"Expected PDF but got Content-Type '{content_type}' and no PDF magic bytes"
                        )

        try:
            from pypdf import PdfReader

            reader = PdfReader(BytesIO(pdf_bytes))
            text_parts: List[str] = []
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
            full_text = "\n".join(text_parts)
            if len(full_text) > 5_000:
                full_text = full_text[:5_000] + "\n[PDF text truncated]"
            return full_text
        except Exception as exc:
            raise RuntimeError(f"Failed to parse PDF: {exc}")
