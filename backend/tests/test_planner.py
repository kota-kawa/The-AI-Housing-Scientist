import json

from app.llm.base import LLMAdapter
from app.stages.planner import run_planner


def make_planner_payload(
    *,
    intent: str = "search",
    target_area: str | None = "江東区",
    budget_max: int | None = 120000,
    station_walk_max: int | None = 7,
    layout_preference: str | None = "1LDK",
    move_in_date: str | None = None,
    must_conditions: list[str] | None = None,
    nice_to_have: list[str] | None = None,
    missing_slots: list[str] | None = None,
    follow_up_questions: list[dict] | None = None,
    next_action: str = "search_and_compare",
    seed_queries: list[str] | None = None,
    research_plan: dict | None = None,
    condition_reasons: dict[str, str] | None = None,
) -> dict:
    must_conditions = must_conditions or []
    nice_to_have = nice_to_have or []
    missing_slots = missing_slots or []
    follow_up_questions = follow_up_questions or []
    if seed_queries is None:
        seed_queries = (
            [
                "江東区 賃貸 12万円 1LDK",
                "江東区で家賃12万円以下の1LDK賃貸",
                "江東区 1LDK 賃貸",
            ]
            if next_action == "search_and_compare"
            else []
        )
    if research_plan is None:
        research_plan = {
            "summary": "条件に近い候補を比較します。",
            "goal": "問い合わせ候補を数件まで絞り込みます。",
            "strategy": [
                "条件に合う候補を広めに集めます。",
                "詳細ページで不足情報を確認します。",
                "一致度の高い候補から並べます。",
            ],
            "rationale": "最初に母集団を作ってから絞る方が条件差分を見やすいためです。",
        }
    default_condition_reasons = {
        "target_area": "",
        "budget_max": "",
        "station_walk_max": "",
        "move_in_date": "",
        "layout_preference": "",
        "must_conditions": "",
        "nice_to_have": "",
    }
    if condition_reasons is not None:
        default_condition_reasons.update(condition_reasons)

    return {
        "intent": intent,
        "user_memory": {
            "target_area": target_area,
            "budget_max": budget_max,
            "station_walk_max": station_walk_max,
            "layout_preference": layout_preference,
            "move_in_date": move_in_date,
            "must_conditions": must_conditions,
            "nice_to_have": nice_to_have,
        },
        "missing_slots": missing_slots,
        "follow_up_questions": follow_up_questions,
        "next_action": next_action,
        "seed_queries": seed_queries,
        "research_plan": research_plan,
        "condition_reasons": default_condition_reasons,
    }


class FakePlannerAdapter(LLMAdapter):
    def __init__(self, payload: dict):
        self.payload = payload
        self.last_user = ""

    def generate_text(self, *, system: str, user: str, temperature: float = 0.2) -> str:
        raise AssertionError("generate_text should not be called in planner tests")

    def generate_structured(
        self,
        *,
        system: str,
        user: str,
        schema: dict,
        temperature: float = 0.2,
    ) -> dict:
        self.last_user = user
        return self.payload

    def list_models(self) -> list[str]:
        return ["fake-planner-model"]


def test_planner_extracts_required_slots_via_llm():
    adapter = FakePlannerAdapter(
        make_planner_payload(
            target_area="渋谷",
            budget_max=150000,
            station_walk_max=10,
            layout_preference="1LDK",
        )
    )

    result = run_planner(
        message="渋谷で家賃15万以内、駅徒歩10分以内の1LDKを探したい",
        user_memory={},
        adapter=adapter,
    )

    assert result["missing_slots"] == []
    assert result["plan"]["target_area"] == "渋谷"
    assert result["plan"]["budget_max"] == 150000
    assert result["plan"]["station_walk_max"] == 10


def test_planner_extracts_bare_ward_name_as_target_area_via_llm():
    adapter = FakePlannerAdapter(
        make_planner_payload(
            target_area="江東区",
            budget_max=120000,
            station_walk_max=7,
            layout_preference="1LDK",
        )
    )

    result = run_planner(
        message="江東区で家賃12万円以下、駅徒歩7分以内の1LDKを探しています",
        user_memory={},
        adapter=adapter,
    )

    assert result["missing_slots"] == []
    assert result["plan"]["target_area"] == "江東区"


def test_planner_extracts_machida_and_rc_condition_via_llm():
    adapter = FakePlannerAdapter(
        make_planner_payload(
            target_area="町田",
            budget_max=100000,
            station_walk_max=None,
            layout_preference=None,
            must_conditions=["RC造"],
            seed_queries=[
                "町田 賃貸 10万円 RC造",
                "町田で家賃10万円以下、RC造を優先した賃貸",
            ],
        )
    )

    result = run_planner(
        message="町田に10万円以下でRCの物件に住みたい",
        user_memory={},
        adapter=adapter,
    )

    assert result["next_action"] == "search_and_compare"
    assert result["plan"]["target_area"] == "町田"
    assert result["plan"]["budget_max"] == 100000
    assert result["user_memory"]["must_conditions"] == ["RC造"]


def test_planner_starts_search_even_when_target_area_is_missing():
    adapter = FakePlannerAdapter(
        make_planner_payload(
            target_area=None,
            budget_max=150000,
            station_walk_max=10,
            layout_preference="1LDK",
            follow_up_questions=[
                {
                    "slot": "target_area",
                    "label": "希望エリア",
                    "question": "どのエリアを優先して探しますか？",
                    "examples": ["江東区", "吉祥寺", "横浜駅周辺"],
                }
            ],
        )
    )

    result = run_planner(
        message="家賃15万円以内、駅徒歩10分以内の1LDKを比較したい",
        user_memory={},
        adapter=adapter,
    )

    assert result["missing_slots"] == []
    assert result["next_action"] == "search_and_compare"
    assert result["plan"]["target_area"] is None
    assert result["follow_up_questions"][0]["slot"] == "target_area"


def test_planner_asks_follow_up_questions_for_generic_request():
    adapter = FakePlannerAdapter(
        make_planner_payload(
            target_area=None,
            budget_max=None,
            station_walk_max=None,
            layout_preference=None,
            next_action="missing_slots_question",
            missing_slots=["target_area", "budget_max", "station_walk_max"],
            follow_up_questions=[
                {
                    "slot": "target_area",
                    "label": "希望エリア",
                    "question": "どのエリアで探しますか？",
                    "examples": ["江東区", "吉祥寺", "横浜駅周辺"],
                },
                {
                    "slot": "budget_max",
                    "label": "家賃上限",
                    "question": "家賃の上限はいくらですか？",
                    "examples": ["10万円まで", "12万円以内", "15万円まで"],
                },
                {
                    "slot": "station_walk_max",
                    "label": "駅徒歩",
                    "question": "駅から徒歩何分以内を希望しますか？",
                    "examples": ["徒歩7分以内", "徒歩10分まで", "駅近だとうれしい"],
                },
            ],
        )
    )

    result = run_planner(
        message="おすすめの賃貸を探したい",
        user_memory={},
        adapter=adapter,
    )

    assert result["next_action"] == "missing_slots_question"
    assert result["missing_slots"] == ["target_area", "budget_max", "station_walk_max"]
    assert [item["slot"] for item in result["follow_up_questions"]] == [
        "target_area",
        "budget_max",
        "station_walk_max",
    ]


def test_planner_uses_llm_generated_questions_queries_and_plan():
    adapter = FakePlannerAdapter(
        make_planner_payload(
            target_area="江東区",
            budget_max=120000,
            station_walk_max=7,
            layout_preference="1LDK",
            must_conditions=["ペット可"],
            nice_to_have=["在宅ワーク向け"],
            follow_up_questions=[
                {
                    "slot": "move_in_date",
                    "label": "入居時期",
                    "question": "ペットと一緒に住む前提で、いつ頃から入居したいですか？",
                    "examples": ["できるだけ早く", "来月中", "6月ごろ"],
                }
            ],
            seed_queries=[
                "江東区 賃貸 12万円 1LDK ペット可",
                "江東区で家賃12万円以下、ペット可で在宅ワークしやすい賃貸",
                "江東区 ペット可 1LDK 在宅ワーク 賃貸",
            ],
            research_plan={
                "summary": "江東区でペット可かつ在宅ワークしやすい1LDKを優先して調べます。",
                "goal": "早めに問い合わせできる候補を3件前後まで絞り込む",
                "strategy": [
                    "ペット可物件は希少なので江東区内で候補を広めに集めます。",
                    "在宅ワーク向け設備や回線条件は詳細ページで重点確認します。",
                    "家賃と駅距離の両立度を見て問い合わせ候補を上位化します。",
                ],
                "rationale": "希少条件があるため、最初に母集団を確保してから詳細条件で絞る方が安全です。",
            },
            condition_reasons={
                "target_area": "生活圏を固定したい条件なので検索の起点にします。",
                "budget_max": "毎月の上限を超える候補を早めに外すためです。",
                "station_walk_max": "徒歩負担が大きい候補を除外するためです。",
                "layout_preference": "仕事スペースを確保しやすい間取りかを見極めるためです。",
                "must_conditions": "ペット可は候補数が限られるので最優先で見ます。",
                "nice_to_have": "在宅ワーク条件は詳細ページで差が出やすいため比較軸にします。",
            },
        )
    )

    result = run_planner(
        message="江東区で家賃12万円以下、ペット可、在宅ワークしやすい1LDKを探したい",
        user_memory={},
        adapter=adapter,
    )

    assert result["intent"] == "search"
    assert result["next_action"] == "search_and_compare"
    assert result["seed_queries"][0] == "江東区 賃貸 12万円 1LDK ペット可"
    assert (
        result["research_plan"]["strategy"][1]
        == "在宅ワーク向け設備や回線条件は詳細ページで重点確認します。"
    )
    assert result["follow_up_questions"][0]["slot"] == "move_in_date"
    assert "ペットと一緒に住む前提" in result["follow_up_questions"][0]["question"]
    assert (
        result["condition_reasons"]["must_conditions"]
        == "ペット可は候補数が限られるので最優先で見ます。"
    )


def test_planner_keeps_up_to_eight_seed_queries():
    adapter = FakePlannerAdapter(
        make_planner_payload(
            seed_queries=[f"query-{index}" for index in range(10)],
        )
    )

    result = run_planner(
        message="江東区で家賃12万円以下の1LDKを探したい",
        user_memory={},
        adapter=adapter,
    )

    assert result["seed_queries"] == [f"query-{index}" for index in range(8)]


def test_planner_uses_llm_intent_for_natural_search_request_without_structured_slots():
    adapter = FakePlannerAdapter(
        make_planner_payload(
            target_area=None,
            budget_max=None,
            station_walk_max=None,
            layout_preference=None,
            move_in_date="asap",
            next_action="missing_slots_question",
            missing_slots=["target_area", "budget_max", "layout_preference"],
            follow_up_questions=[
                {
                    "slot": "target_area",
                    "label": "希望エリア",
                    "question": "来月の引っ越し先として、どのエリアを優先したいですか？",
                    "examples": ["江東区", "川崎駅周辺", "西荻窪"],
                },
                {
                    "slot": "budget_max",
                    "label": "家賃上限",
                    "question": "家賃はどこまで見ますか？",
                    "examples": ["10万円まで", "12万円以内", "管理費込みで14万円以下"],
                },
                {
                    "slot": "layout_preference",
                    "label": "間取り",
                    "question": "一人暮らし向けの間取り希望はありますか？",
                    "examples": ["1K", "1DK", "1LDK"],
                },
            ],
        )
    )

    result = run_planner(
        message="来月引っ越すんだけど、いい感じの部屋を探したい",
        user_memory={},
        adapter=adapter,
    )

    assert result["intent"] == "search"
    assert result["next_action"] == "missing_slots_question"
    assert result["missing_slots"] == ["target_area", "budget_max", "layout_preference"]


def test_planner_uses_llm_follow_up_questions_as_is_when_subset_selected():
    adapter = FakePlannerAdapter(
        make_planner_payload(
            target_area=None,
            budget_max=None,
            station_walk_max=None,
            layout_preference=None,
            move_in_date=None,
            next_action="missing_slots_question",
            missing_slots=["target_area"],
            follow_up_questions=[
                {
                    "slot": "target_area",
                    "label": "希望エリア",
                    "question": "まずはどのエリアを優先したいですか？",
                    "examples": ["中野", "横浜駅周辺", "江東区"],
                }
            ],
        )
    )

    result = run_planner(
        message="部屋を探したい",
        user_memory={},
        adapter=adapter,
    )

    assert result["next_action"] == "missing_slots_question"
    assert result["missing_slots"] == ["target_area"]
    assert [item["slot"] for item in result["follow_up_questions"]] == ["target_area"]


def test_planner_uses_llm_condition_reasons_as_is_without_default_backfill():
    adapter = FakePlannerAdapter(
        make_planner_payload(
            target_area="江東区",
            budget_max=120000,
            station_walk_max=7,
            layout_preference="1LDK",
            must_conditions=["ペット可"],
            condition_reasons={},
        )
    )

    result = run_planner(
        message="江東区で家賃12万円以下、駅徒歩7分以内の1LDKを探したい",
        user_memory={},
        adapter=adapter,
    )

    assert result["next_action"] == "search_and_compare"
    assert result["condition_reasons"] == {
        "budget_max": "",
        "target_area": "",
        "station_walk_max": "",
        "move_in_date": "",
        "layout_preference": "",
        "must_conditions": "",
        "nice_to_have": "",
    }


def test_planner_injects_two_prompt_examples_into_llm_payload():
    adapter = FakePlannerAdapter(make_planner_payload())

    run_planner(
        message="中野で部屋を探したい",
        user_memory={},
        adapter=adapter,
    )

    payload = json.loads(adapter.last_user)
    assert payload["examples_instruction"]
    assert len(payload["examples"]) == 2
    assert all("case_id" in item for item in payload["examples"])
    assert all("input" in item for item in payload["examples"])
    assert all("output" in item for item in payload["examples"])
    assert any("非網羅" in rule or "固定候補" in rule for rule in payload["decision_rules"])
    assert any("近隣エリア" in rule or "沿線違い" in rule for rule in payload["decision_rules"])
    assert any("必須条件を外した比較用" in rule for rule in payload["decision_rules"])
    assert "固定してはいけません" in payload["examples_instruction"]
