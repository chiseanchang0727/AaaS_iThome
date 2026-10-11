"""Plotly served from here: chart pages point at it, and it is fetched from the CDN once."""

import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api import vendor
from api.vendor import local_plotly, vendor_router

PAGE = b'<script src="https://cdn.plot.ly/plotly-4.1.1.min.js"></script><div id="c"></div>'


def test_a_chart_page_loads_plotly_from_here():
    assert local_plotly(PAGE) == b'<script src="/api/vendor/plotly-4.1.1.min.js"></script><div id="c"></div>'
    other = b'<script src="https://example.com/plotly-4.1.1.min.js"></script>'
    assert local_plotly(other) == other


def app_with(tmp_path):
    app = FastAPI()
    app.include_router(vendor_router(tmp_path))
    return TestClient(app)


def test_it_is_fetched_once_then_served_from_disk(tmp_path, monkeypatch):
    fetched = []

    async def get(self, url, **kwargs):
        fetched.append(url)
        return httpx.Response(200, content=b"/* plotly */", request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx.AsyncClient, "get", get)
    client = app_with(tmp_path)
    for _ in range(2):
        response = client.get("/api/vendor/plotly-4.1.1.min.js")
        assert response.status_code == 200 and response.content == b"/* plotly */"
        assert "immutable" in response.headers["cache-control"]
        assert response.headers["access-control-allow-origin"] == "*"  # loaded with crossorigin from a null origin
    assert fetched == ["https://cdn.plot.ly/plotly-4.1.1.min.js"]
    assert (tmp_path / "plotly-4.1.1.min.js").read_bytes() == b"/* plotly */"


def test_without_the_cdn_the_browser_is_sent_there(tmp_path, monkeypatch):
    async def get(self, url, **kwargs):
        raise httpx.ConnectError("offline")

    monkeypatch.setattr(httpx.AsyncClient, "get", get)
    response = app_with(tmp_path).get("/api/vendor/plotly-4.1.1.min.js", follow_redirects=False)
    assert response.status_code == 307 and response.headers["location"] == vendor.CDN + "plotly-4.1.1.min.js"


def test_only_plotly_files(tmp_path):
    client = app_with(tmp_path)
    assert client.get("/api/vendor/evil.js").status_code == 404
    assert client.get("/api/vendor/plotly-latest.min.js").status_code == 404
