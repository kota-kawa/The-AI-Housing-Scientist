from app.stages.search_normalize import run_search_and_normalize


def test_search_normalize_dedup_group():
    items = [
        {
            "title": "サンプルマンション 1LDK 徒歩8分",
            "description": "家賃14.5万 35㎡",
            "url": "https://example.com/1",
            "extra_snippets": [],
        },
        {
            "title": "サンプルマンション 1LDK 徒歩8分",
            "description": "家賃14.8万 35㎡",
            "url": "https://example.com/2",
            "extra_snippets": [],
        },
    ]

    result = run_search_and_normalize(query="渋谷 賃貸", search_results=items)

    assert len(result["normalized_properties"]) == 2
    assert len(result["duplicate_groups"]) == 1
