from app.research.journal import ResearchJournal, ResearchNode
from app.stages.final_report import run_final_report


def test_selected_branch_nodes_include_selected_path_nodes():
    journal = ResearchJournal()
    journal.append(
        ResearchNode(
            id=1,
            stage="tree_search",
            node_type="search_candidate",
            status="completed",
            input_payload={},
            output_payload={"summary": "root"},
            reasoning="root reasoning",
            branch_id="node-1",
        )
    )
    journal.append(
        ResearchNode(
            id=2,
            stage="tree_search",
            node_type="search_candidate",
            status="completed",
            input_payload={},
            output_payload={"summary": "child"},
            reasoning="child reasoning",
            branch_id="node-2",
        )
    )
    journal.append(
        ResearchNode(
            id=3,
            stage="tree_search",
            node_type="search_selection",
            status="completed",
            input_payload={},
            output_payload={
                "selected_branch": {"branch_id": "node-2"},
                "selected_path": [
                    {"branch_id": "node-1", "label": "root"},
                    {"branch_id": "node-2", "label": "child"},
                ],
            },
            reasoning="selected",
            branch_id="node-2",
            selected=True,
        )
    )

    selected_nodes = journal.selected_branch_nodes()

    assert [node.id for node in selected_nodes] == [1, 2, 3]


def test_final_report_fallback_uses_branch_result_summary_and_selected_path():
    stage_nodes = [
        ResearchNode(
            id=10,
            stage="plan_finalize",
            node_type="stage",
            status="completed",
            input_payload={},
            output_payload={"summary": "条件を固定しました。"},
            reasoning="plan finalized",
        ),
        ResearchNode(
            id=11,
            stage="synthesize",
            node_type="stage",
            status="completed",
            input_payload={},
            output_payload={
                "offline_evaluation": {
                    "recommendations": ["管理費と契約条件を確認する"],
                },
                "failure_summary": {
                    "top_issues": ["詳細ページ補完率が低い"],
                },
                "research_summary": "東雲ベイテラスが最有力です。",
            },
            reasoning="synthesized",
        ),
    ]
    selected_branch_nodes = [
        ResearchNode(
            id=20,
            stage="tree_search",
            node_type="search_selection",
            status="completed",
            input_payload={},
            output_payload={
                "selected_branch": {
                    "branch_id": "node-2",
                    "label": "detail_first",
                    "branch_result_summary": {
                        "物件候補リスト": [
                            {
                                "building_name": "東雲ベイテラス",
                                "rent": 118000,
                                "layout": "1LDK",
                                "station_walk_min": 6,
                                "area_m2": 42.1,
                                "reason": "家賃と駅徒歩が条件内で、面積にも余裕があります。",
                            },
                            {
                                "building_name": "豊洲レジデンス",
                                "rent": 121000,
                                "layout": "1LDK",
                                "station_walk_min": 4,
                                "area_m2": 39.0,
                                "reason": "駅距離は良いが家賃がやや上振れです。",
                            },
                        ],
                        "共通リスク": ["管理費や初期費用の内訳が不足している"],
                        "未解決の調査項目": ["東雲ベイテラス の管理費・初期費用内訳"],
                    },
                },
                "selected_path": [
                    {
                        "branch_id": "node-1",
                        "label": "source_diversify",
                        "depth": 1,
                        "strategy_tags": ["source_diversify"],
                        "branch_score": 74.0,
                    },
                    {
                        "branch_id": "node-2",
                        "label": "detail_first",
                        "depth": 2,
                        "strategy_tags": ["detail_first"],
                        "branch_score": 88.0,
                    },
                ],
                "search_tree_summary": {
                    "executed_node_count": 4,
                    "termination_reason": "stable_high_readiness",
                },
            },
            reasoning="selected path",
            branch_id="node-2",
            selected=True,
            metrics={},
        )
    ]

    result = run_final_report(
        stage_nodes=stage_nodes,
        selected_branch_nodes=selected_branch_nodes,
        adapter=None,
    )

    report = result["report_markdown"]
    assert "## 探索経路" in report
    assert "source_diversify" in report
    assert "## 候補比較表" in report
    assert "| 東雲ベイテラス | 118,000円 | 1LDK | 6分 | 42.1㎡ |" in report
    assert "## リスク" in report
    assert "管理費や初期費用の内訳が不足している" in report
    assert "## 推奨物件と根拠" in report
    assert "東雲ベイテラス を推奨します" in report
    assert "## 追加調査の提案" in report
    assert "東雲ベイテラス の管理費・初期費用内訳" in report


def test_final_report_fallback_says_no_recommendation_without_candidates():
    stage_nodes = [
        ResearchNode(
            id=10,
            stage="synthesize",
            node_type="stage",
            status="completed",
            input_payload={},
            output_payload={
                "offline_evaluation": {
                    "readiness": "low",
                    "recommendations": ["strict条件で再探索する"],
                },
                "failure_summary": {
                    "top_issues": ["正規化後の候補が残っていない"],
                },
                "research_summary": "条件に合う推薦可能候補はありませんでした。",
            },
            reasoning="synthesized",
        )
    ]
    selected_branch_nodes = [
        ResearchNode(
            id=20,
            stage="tree_search",
            node_type="search_selection",
            status="completed",
            input_payload={},
            output_payload={
                "selected_branch": {
                    "branch_id": "node-1",
                    "label": "strict",
                    "branch_result_summary": {
                        "物件候補リスト": [],
                        "共通リスク": ["検索ヒットはあるが、整合性レビュー後に推薦可能候補が残っていない"],
                        "未解決の調査項目": ["対象エリア・間取り・must条件を固定した再検索"],
                    },
                },
                "selected_path": [
                    {
                        "branch_id": "node-1",
                        "label": "strict",
                        "depth": 1,
                        "strategy_tags": ["tighten_match"],
                        "branch_score": 8.0,
                    }
                ],
                "search_tree_summary": {
                    "executed_node_count": 1,
                    "termination_reason": "frontier_exhausted",
                },
            },
            reasoning="selected path",
            branch_id="node-1",
            selected=True,
            metrics={},
        )
    ]

    result = run_final_report(
        stage_nodes=stage_nodes,
        selected_branch_nodes=selected_branch_nodes,
        adapter=None,
    )

    report = result["report_markdown"]
    assert "推奨物件なし" in report
    assert "| 候補なし |" in report
    assert "東雲ベイテラス を推奨します" not in report
