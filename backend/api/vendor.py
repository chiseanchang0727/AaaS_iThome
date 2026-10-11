"""Plotly for the charts the agent's scripts write, served from here instead of cdn.plot.ly.

    GET /api/vendor/plotly-<version>.min.js     that Plotly, fetched from cdn.plot.ly once and kept

A chart page (plotly's write_html with include_plotlyjs="cdn") loads Plotly,
about 5 MB, from cdn.plot.ly. Chart pages are shown as sandboxed frames, and
a sandboxed frame does not share the browser's cache: each one downloaded
its own copy, so a report of seven charts fetched 35 MB and stayed blank
until it arrived. When a chart page is served, its cdn.plot.ly address is
pointed here (`local_plotly`); the file as the script wrote it is unchanged.
The same version is fetched, so the chart draws as it did. If cdn.plot.ly
cannot be reached from here, the browser is sent there, as before.
"""

import asyncio
import logging
import re
from pathlib import Path

import httpx
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, RedirectResponse

log = logging.getLogger(__name__)

_CDN = re.compile(rb"https://cdn\.plot\.ly/(plotly-\d+\.\d+\.\d+\.min\.js)")
_NAME = re.compile(r"^plotly-\d+\.\d+\.\d+\.min\.js$")
CDN = "https://cdn.plot.ly/"


def local_plotly(page: bytes) -> bytes:
    """`page` loading Plotly from this server instead of cdn.plot.ly."""
    return _CDN.sub(rb"/api/vendor/\1", page)


def vendor_router(cache_dir: Path) -> APIRouter:
    router = APIRouter(prefix="/api/vendor")
    locks: dict[str, asyncio.Lock] = {}

    async def fetch(name: str, path: Path) -> None:
        async with httpx.AsyncClient(timeout=60, follow_redirects=True) as client:
            response = await client.get(CDN + name)
            response.raise_for_status()
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(".part")
        partial.write_bytes(response.content)
        partial.replace(path)

    @router.get("/{name}")
    async def plotly_js(name: str):
        if not _NAME.match(name):
            raise HTTPException(404)
        path = cache_dir / name
        if not path.is_file():
            async with locks.setdefault(name, asyncio.Lock()):  # a report asks for it seven times at once
                if not path.is_file():
                    try:
                        await fetch(name, path)
                    except Exception:
                        log.warning("could not fetch %s; sending the browser to the CDN", name, exc_info=True)
                        return RedirectResponse(CDN + name)
        # A version's file never changes. Chart pages load it with crossorigin="anonymous" (and its
        # integrity hash) from a sandboxed page, whose origin is null: a CORS request, as from the CDN.
        return FileResponse(path, media_type="text/javascript",
                            headers={"Cache-Control": "public, max-age=31536000, immutable",
                                     "Access-Control-Allow-Origin": "*"})

    return router
