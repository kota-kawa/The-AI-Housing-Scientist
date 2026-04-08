from app.llm.base import LLMAdapter
from app.stages.planner import run_planner


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


def test_planner_extracts_required_slots():
    message = "渋谷で家賃15万以内、駅徒歩10分以内の1LDKを探したい"

    result = run_planner(message=message, user_memory={}, adapter=None)

    assert result["missing_slots"] == []
    assert result["plan"]["target_area"] == "渋谷"
    assert result["plan"]["budget_max"] == 150000
    assert result["plan"]["station_walk_max"] == 10


def test_planner_extracts_bare_ward_name_as_target_area():
    message = "江東区で家賃12万円以下、駅徒歩7分以内の1LDKを探しています"

    result = run_planner(message=message, user_memory={}, adapter=None)

    assert result["missing_slots"] == []
    assert result["plan"]["target_area"] == "江東区"


def test_planner_starts_search_even_when_target_area_is_missing():
    message = "家賃15万円以内、駅徒歩10分以内の1LDKを比較したい"

    result = run_planner(message=message, user_memory={}, adapter=None)

    assert result["missing_slots"] == []
    assert result["next_action"] == "search_and_compare"
    assert result["plan"]["target_area"] is None
    assert result["follow_up_questions"][0]["slot"] == "target_area"


def test_planner_asks_follow_up_questions_for_generic_request():
    message = "おすすめの賃貸を探したい"

    result = run_planner(message=message, user_memory={}, adapter=None)

    assert result["next_action"] == "missing_slots_question"
    assert result["missing_slots"] == ["target_area", "budget_max", "station_walk_max"]
    assert [item["slot"] for item in result["follow_up_questions"]] == [
        "target_area",
        "budget_max",
        "station_walk_max",
    ]


def test_planner_uses_llm_generated_questions_queries_and_plan():
    adapter = FakePlannerAdapter(
        {
            "intent": "search",
            "extracted_slots": {
                "target_area": "江東区",
                "budget_max": 120000,
                "station_walk_max": 7,
                "layout_preference": "1LDK",
                "move_in_date": None,
                "must_conditions": ["ペット可"],
                "nice_to_have": ["在宅ワーク向け"],
            },
            "follow_up_questions": [
                {
                    "slot": "move_in_date",
                    "question": "ペットと一緒に住む前提で、いつ頃から入居したいですか？",
                    "examples": ["できるだけ早く", "来月中", "6月ごろ"],
                }
            ],
            "seed_queries": [
                "江東区 賃貸 12万円 1LDK ペット可",
                "江東区で家賃12万円以下、ペット可で在宅ワークしやすい賃貸",
                "江東区 ペット可 1LDK 在宅ワーク 賃貸",
            ],
            "research_plan": {
                "summary": "江東区でペット可かつ在宅ワークしやすい1LDKを優先して調べます。",
                "goal": "早めに問い合わせできる候補を3件前後まで絞り込む",
                "strategy": [
                    "ペット可物件は希少なので江東区内で候補を広めに集めます。",
                    "在宅ワーク向け設備や回線条件は詳細ページで重点確認します。",
                    "家賃と駅距離の両立度を見て問い合わせ候補を上位化します。",
                ],
                "rationale": "希少条件があるため、最初に母集団を確保してから詳細条件で絞る方が安全です。",
            },
            "condition_reasons": {
                "target_area": "生活圏を固定したい条件なので検索の起点にします。",
                "budget_max": "毎月の上限を超える候補を早めに外すためです。",
                "station_walk_max": "徒歩負担が大きい候補を除外するためです。",
                "move_in_date": "",
                "layout_preference": "仕事スペースを確保しやすい間取りかを見極めるためです。",
                "must_conditions": "ペット可は候補数が限られるので最優先で見ます。",
                "nice_to_have": "在宅ワーク条件は詳細ページで差が出やすいため比較軸にします。",
            },
        }
    )

    result = run_planner(
        message="江東区で家賃12万円以下、ペット可、在宅ワークしやすい1LDKを探したい",
        user_memory={},
        adapter=adapter,
    )

    assert result["intent"] == "search"
    assert result["next_action"] == "search_and_compare"
    assert result["seed_queries"][0] == "江東区 賃貸 12万円 1LDK ペット可"
    assert result["research_plan"]["strategy"][1] == "在宅ワーク向け設備や回線条件は詳細ページで重点確認します。"
    assert result["follow_up_questions"][0]["slot"] == "move_in_date"
    assert "ペットと一緒に住む前提" in result["follow_up_questions"][0]["question"]
    assert result["condition_reasons"]["must_conditions"] == "ペット可は候補数が限られるので最優先で見ます。"


def test_planner_uses_llm_intent_for_natural_search_request_without_structured_slots():
    adapter = FakePlannerAdapter(
        {
            "intent": "search",
            "extracted_slots": {
                "target_area": None,
                "budget_max": None,
                "station_walk_max": None,
                "layout_preference": None,
                "move_in_date": "asap",
                "must_conditions": [],
                "nice_to_have": [],
            },
            "follow_up_questions": [
                {
                    "slot": "target_area",
                    "question": "来月の引っ越し先として、どのエリアを優先したいですか？",
                    "examples": ["江東区", "川崎駅周辺", "西荻窪"],
                },
                {
                    "slot": "budget_max",
                    "question": "家賃はどこまで見ますか？",
                    "examples": ["10万円まで", "12万円以内", "管理費込みで14万円以下"],
                },
                {
                    "slot": "layout_preference",
                    "question": "一人暮らし向けの間取り希望はありますか？",
                    "examples": ["1K", "1DK", "1LDK"],
                },
            ],
            "seed_queries": [],
            "research_plan": {
                "summary": "",
                "goal": "",
                "strategy": [],
                "rationale": "",
            },
            "condition_reasons": {
                "target_area": "",
                "budget_max": "",
                "station_walk_max": "",
                "move_in_date": "",
                "layout_preference": "",
                "must_conditions": "",
                "nice_to_have": "",
            },
        }
    )

    result = run_planner(
        message="来月引っ越すんだけど、いい感じの部屋を探したい",
        user_memory={},
        adapter=adapter,
    )

    assert result["intent"] == "search"
    assert result["next_action"] == "missing_slots_question"
    assert result["missing_slots"] == ["target_area", "budget_max", "layout_preference"]


def test_planner_does_not_backfill_default_follow_up_questions_when_llm_selects_subset():
    adapter = FakePlannerAdapter(
        {
            "intent": "search",
            "extracted_slots": {
                "target_area": None,
                "budget_max": None,
                "station_walk_max": None,
                "layout_preference": None,
                "move_in_date": None,
                "must_conditions": [],
                "nice_to_have": [],
            },
            "follow_up_questions": [
                {
                    "slot": "target_area",
                    "question": "まずはどのエリアを優先したいですか？",
                    "examples": ["中野", "横浜駅周辺", "江東区"],
                }
            ],
            "seed_queries": [],
            "research_plan": {
                "summary": "",
                "goal": "",
                "strategy": [],
                "rationale": "",
            },
            "condition_reasons": {
                "target_area": "",
                "budget_max": "",
                "station_walk_max": "",
                "move_in_date": "",
                "layout_preference": "",
                "must_conditions": "",
                "nice_to_have": "",
            },
        }
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
        {
            "intent": "search",
            "extracted_slots": {
                "target_area": "江東区",
                "budget_max": 120000,
                "station_walk_max": 7,
                "layout_preference": "1LDK",
                "move_in_date": None,
                "must_conditions": ["ペット可"],
                "nice_to_have": [],
            },
            "follow_up_questions": [],
            "seed_queries": [],
            "research_plan": {
                "summary": "",
                "goal": "",
                "strategy": [],
                "rationale": "",
            },
            "condition_reasons": {
                "target_area": "",
                "budget_max": "",
                "station_walk_max": "",
                "move_in_date": "",
                "layout_preference": "",
                "must_conditions": "",
                "nice_to_have": "",
            },
        }
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
