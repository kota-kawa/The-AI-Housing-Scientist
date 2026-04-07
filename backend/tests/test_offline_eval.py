from pathlib import Path

from app.research.offline_eval import (
    evaluate_branch,
    evaluate_final_result,
    load_offline_eval_cases,
    run_offline_eval_suite,
    select_best_branch,
    summarize_branch_failures,
)


def test_offline_evaluator_prefers_high_quality_branch_and_summarizes_failures():
    strong_branch = evaluate_branch(
        branch_id="balanced",
        label="balanced",
        queries=["江東区 賃貸 1LDK", "江東区 徒歩7分 賃貸"],
        raw_results=[
            {"source_name": "catalog"},
            {"source_name": "brave"},
            {"source_name": "catalog"},
        ],
        normalized_properties=[
            {
                "rent": 118000,
                "station_walk_min": 6,
                "layout": "1LDK",
                "address_norm": "東京都江東区",
            },
            {
                "rent": 119000,
                "station_walk_min": 7,
                "layout": "1LDK",
                "address_norm": "東京都江東区",
            },
            {
                "rent": 115000,
                "station_walk_min": 5,
                "layout": "1LDK",
                "address_norm": "東京都江東区",
            },
        ],
        ranked_properties=[
            {"score": 92.0},
            {"score": 84.0},
            {"score": 78.0},
        ],
        duplicate_groups=[{"property_ids": ["a", "b"]}],
        search_summary={"detail_hit_count": 3},
    )

    weak_branch = evaluate_branch(
        branch_id="broad",
        label="broad",
        queries=["江東区 賃貸"],
        raw_results=[],
        normalized_properties=[],
        ranked_properties=[],
        duplicate_groups=[],
        search_summary={"detail_hit_count": 0},
    )

    selected = select_best_branch([weak_branch, strong_branch])
    assert selected is not None
    assert selected["branch_id"] == "balanced"

    failure_summary = summarize_branch_failures([weak_branch, strong_branch])
    assert "検索結果が取得できていない" in failure_summary["top_issues"]
    assert failure_summary["recommendations"]

    final_result = evaluate_final_result(
        selected_branch_summary=selected,
        visible_ranked_properties=[
            {"property_id_norm": "p1"},
            {"property_id_norm": "p2"},
            {"property_id_norm": "p3"},
        ],
        search_summary={"detail_hit_count": 3},
    )
    assert final_result["readiness"] == "high"
    assert "上位候補の契約条件確認へ進む" in final_result["recommendations"]


def test_offline_eval_fixture_suite_passes():
    fixture_path = Path(__file__).parent / "fixtures" / "offline_eval_cases.json"
    cases = load_offline_eval_cases(fixture_path)
    report = run_offline_eval_suite(cases)

    assert report["case_count"] == 2
    assert report["failed_count"] == 0
    assert all(case["passed"] for case in report["cases"])


def test_select_best_branch_returns_none_when_all_branches_failed():
    assert (
        select_best_branch(
            [
                {"branch_id": "strict", "status": "failed", "branch_score": 0.0},
                {"branch_id": "broad", "status": "failed", "branch_score": 0.0},
            ]
        )
        is None
    )
