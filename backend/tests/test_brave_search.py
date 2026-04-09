import pytest

from app.services import brave_search
from app.services.brave_search import BraveSearchClient, _TokenBucket


def test_token_bucket_waits_for_refill(monkeypatch):
    timeline = {"now": 100.0}
    sleep_calls: list[float] = []

    def fake_monotonic() -> float:
        return timeline["now"]

    def fake_sleep(seconds: float) -> None:
        sleep_calls.append(seconds)
        timeline["now"] += seconds

    monkeypatch.setattr(brave_search.time, "monotonic", fake_monotonic)
    monkeypatch.setattr(brave_search.time, "sleep", fake_sleep)

    bucket = _TokenBucket(rate_per_second=2.0, capacity=1)

    bucket.acquire()
    bucket.acquire()

    assert sleep_calls == [pytest.approx(0.5)]


def test_brave_search_client_uses_rate_limiter(monkeypatch):
    acquire_calls: list[str] = []

    class FakeBucket:
        def acquire(self) -> None:
            acquire_calls.append("acquire")

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "web": {
                    "results": [
                        {
                            "title": "東雲ベイテラス",
                            "url": "https://example.com/p1",
                            "description": "江東区の1LDK",
                            "age": "1 day",
                            "extra_snippets": ["徒歩6分"],
                        }
                    ]
                }
            }

    class FakeClient:
        def __init__(self, *, timeout: int):
            self.timeout = timeout

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def get(self, url: str, *, headers: dict, params: dict) -> FakeResponse:
            assert url == "https://api.search.brave.com/res/v1/web/search"
            assert headers["X-Subscription-Token"] == "brave-key"
            assert params["q"] == "江東区 賃貸"
            return FakeResponse()

    monkeypatch.setattr(
        brave_search,
        "_get_brave_token_bucket",
        lambda **kwargs: FakeBucket(),
    )
    monkeypatch.setattr(brave_search.httpx, "Client", FakeClient)

    client = BraveSearchClient("brave-key", timeout_seconds=5)
    results = client.search("江東区 賃貸", count=1)

    assert acquire_calls == ["acquire"]
    assert results == [
        {
            "title": "東雲ベイテラス",
            "url": "https://example.com/p1",
            "description": "江東区の1LDK",
            "age": "1 day",
            "extra_snippets": ["徒歩6分"],
        }
    ]
