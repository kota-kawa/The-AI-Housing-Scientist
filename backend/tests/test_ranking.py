import json

from app.llm.base import LLMAdapter
from app.stages.ranking import run_ranking


class FakeRankingAdapter(LLMAdapter):
    def __init__(self, payload: dict):
        self.payload = payload
        self.calls = 0
        self.last_user = ""

    def generate_text(self, *, system: str, user: str, temperature: float = 0.2) -> str:
        raise AssertionError("generate_text should not be called in ranking")

    def generate_structured(
        self,
        *,
        system: str,
        user: str,
        schema: dict,
        temperature: float = 0.2,
    ) -> dict:
        self.calls += 1
        self.last_user = user
        return self.payload

    def list_models(self) -> list[str]:
        return ["fake-ranking-model"]


def test_run_ranking_fallback_reflects_explicit_nice_to_have_keyword():
    result = run_ranking(
        normalized_properties=[
            {
                "property_id_norm": "p1",
                "building_name": "中野ワークレジデンス",
                "address": "東京都中野区",
                "area_name": "中野区",
                "nearest_station": "中野駅",
                "station_walk_min": 5,
                "layout": "1LDK",
                "area_m2": 32.0,
                "rent": 118000,
                "notes": "在宅ワーク向けの個室スペースあり。",
                "features": ["在宅ワーク向け", "独立洗面台"],
            },
            {
                "property_id_norm": "p2",
                "building_name": "中野ガーデン",
                "address": "東京都中野区",
                "area_name": "中野区",
                "nearest_station": "中野駅",
                "station_walk_min": 5,
                "layout": "1LDK",
                "area_m2": 32.0,
                "rent": 118000,
                "notes": "収納が多い物件です。",
                "features": ["独立洗面台"],
            },
        ],
        user_memory={
            "budget_max": 120000,
            "station_walk_max": 7,
            "layout_preference": "1LDK",
            "nice_to_have": ["在宅ワーク向け"],
        },
        adapter=None,
    )

    ranked = result["ranked_properties"]
    assert ranked[0]["property_id_norm"] == "p1"
    assert ranked[0]["score"] > ranked[1]["score"]
    assert "在宅ワーク向け" in ranked[0]["why_selected"]
    assert result["llm_reasoning_applied"] is False


def test_run_ranking_llm_adds_semantic_nice_to_have_bonus_and_reasoning():
    adapter = FakeRankingAdapter(
        {
            "assessments": [
                {
                    "property_id_norm": "p1",
                    "why_selected": "高速回線とワークスペースの記載があり、在宅勤務のイメージが持ちやすい候補です。",
                    "why_not_selected": "駅からは許容範囲ですが、周辺の静かさは内見時に確認したいです。",
                    "nice_to_have_assessments": [
                        {
                            "condition": "在宅ワーク向け",
                            "match_level": "strong",
                            "evidence": "高速回線とオンライン会議向けスペースの記載",
                        }
                    ],
                },
                {
                    "property_id_norm": "p2",
                    "why_selected": "基本条件は満たしていますが、特徴の打ち出しはやや無難です。",
                    "why_not_selected": "在宅ワーク向けと判断できる材料がなく、決め手はやや弱めです。",
                    "nice_to_have_assessments": [
                        {
                            "condition": "在宅ワーク向け",
                            "match_level": "none",
                            "evidence": "",
                        }
                    ],
                },
            ]
        }
    )

    result = run_ranking(
        normalized_properties=[
            {
                "property_id_norm": "p1",
                "building_name": "目黒リモートハウス",
                "address": "東京都目黒区",
                "area_name": "目黒区",
                "nearest_station": "目黒駅",
                "station_walk_min": 6,
                "layout": "1LDK",
                "area_m2": 31.0,
                "rent": 129000,
                "notes": "高速回線導入済み。オンライン会議向けのカウンタースペースあり。",
                "features": ["高速回線", "宅配ボックス"],
            },
            {
                "property_id_norm": "p2",
                "building_name": "目黒スタンダード",
                "address": "東京都目黒区",
                "area_name": "目黒区",
                "nearest_station": "目黒駅",
                "station_walk_min": 6,
                "layout": "1LDK",
                "area_m2": 31.0,
                "rent": 129000,
                "notes": "収納充実。",
                "features": ["宅配ボックス"],
            },
        ],
        user_memory={
            "budget_max": 130000,
            "station_walk_max": 7,
            "layout_preference": "1LDK",
            "nice_to_have": ["在宅ワーク向け"],
        },
        adapter=adapter,
    )

    ranked = result["ranked_properties"]
    assert adapter.calls == 1
    assert ranked[0]["property_id_norm"] == "p1"
    assert (
        ranked[0]["why_selected"]
        == "高速回線とワークスペースの記載があり、在宅勤務のイメージが持ちやすい候補です。"
    )
    assert (
        ranked[1]["why_not_selected"]
        == "在宅ワーク向けと判断できる材料がなく、決め手はやや弱めです。"
    )
    assert result["nice_to_have_assessments"]["p1"][0]["match_level"] == "strong"
    assert result["llm_reasoning_applied"] is True


def test_run_ranking_injects_two_prompt_examples_into_llm_payload():
    adapter = FakeRankingAdapter({"assessments": []})

    run_ranking(
        normalized_properties=[
            {
                "property_id_norm": "p1",
                "building_name": "中野ワークレジデンス",
                "address": "東京都中野区",
                "area_name": "中野区",
                "nearest_station": "中野駅",
                "station_walk_min": 5,
                "layout": "1LDK",
                "area_m2": 32.0,
                "rent": 118000,
                "notes": "在宅ワーク向けの個室スペースあり。",
                "features": ["在宅ワーク向け", "独立洗面台"],
            }
        ],
        user_memory={
            "budget_max": 120000,
            "station_walk_max": 7,
            "layout_preference": "1LDK",
            "nice_to_have": ["在宅ワーク向け"],
        },
        adapter=adapter,
    )

    payload = json.loads(adapter.last_user)
    assert len(payload["examples"]) == 2
    assert all("case_id" in item for item in payload["examples"])
    assert all("input" in item for item in payload["examples"])
    assert all("output" in item for item in payload["examples"])
