import json
from pathlib import Path

from app.config import Settings
from app.db import Database
from app.llm.base import LLMAdapter
from app.main import app, create_session as create_session_endpoint
from app.models import CreateSessionRequest
from app.orchestrator import HousingOrchestrator


def build_settings(database_path: str) -> Settings:
    return Settings(
        app_env="test",
        database_path=database_path,
        llm_default_provider="openai",
        model_strict_mode=False,
        run_preflight_on_startup=False,
        preflight_fail_fast=False,
        llm_timeout_seconds=3,
        llm_max_retries=0,
        openai_api_key="",
        gemini_api_key="",
        groq_api_key="",
        claude_api_key="",
        brave_search_api_key="",
        openai_model="gpt-5.4-mini",
        gemini_model="gemini-3-flash",
        groq_model_primary="openai/gpt-oss-120b",
        groq_model_secondary="qwen/qwen3-32b",
        claude_model="claude-sonnet-4-6",
    )


class FakeResearchSummaryAdapter(LLMAdapter):
    def __init__(self, summary: str, final_report: str | None = None):
        self.summary = summary
        self.final_report = final_report or (
            "# 最終レポート\n\n"
            "## 探索経路\n"
            "1. 条件を固定して探索を開始しました。\n\n"
            "## 候補比較表\n"
            "| 物件 | 家賃 | 間取り | 駅徒歩 | 面積 | 根拠 |\n"
            "| --- | --- | --- | --- | --- | --- |\n"
            "| サンプル候補 | 要確認 | 要確認 | 要確認 | 要確認 | 要約レポートです。 |\n\n"
            "## リスク\n"
            "- 詳細条件は再確認が必要です。\n\n"
            "## 推奨物件と根拠\n"
            "サンプル候補を暫定候補とします。\n\n"
            "## 追加調査の提案\n"
            "- 掲載条件の最新性を確認してください。"
        )
        self.last_text_user = ""
        self.last_report_user = ""

    def generate_text(self, *, system: str, user: str, temperature: float = 0.2) -> str:
        if "Write a final markdown report" in user:
            self.last_report_user = user
            return self.final_report
        self.last_text_user = user
        return self.summary

    def generate_structured(
        self,
        *,
        system: str,
        user: str,
        schema: dict,
        temperature: float = 0.2,
    ) -> dict:
        properties = schema.get("properties", {})
        if "branches" in properties:
            return {
                "branches": [
                    {
                        "branch_id": "balanced",
                        "query_suggestions": [],
                        "description_hint": "バランス重視",
                    },
                    {
                        "branch_id": "strict",
                        "query_suggestions": [],
                        "description_hint": "厳しめ条件",
                    },
                    {
                        "branch_id": "broad",
                        "query_suggestions": [],
                        "description_hint": "広めに収集",
                    },
                ],
                "summary": "3本の探索分岐を維持します。",
            }
        if "reviews" in properties:
            return {"reviews": []}
        if "assessments" in properties:
            return {"assessments": []}
        raise AssertionError(f"unexpected schema keys: {list(properties.keys())}")

    def list_models(self) -> list[str]:
        return ["fake-research-model"]


class FakePlannerRouteAdapter(LLMAdapter):
    def __init__(self, payload: dict, presentation_payload: dict | None = None):
        self.payload = payload
        self.presentation_payload = presentation_payload or {}

    def generate_text(self, *, system: str, user: str, temperature: float = 0.2) -> str:
        raise AssertionError("generate_text should not be called in planner route tests")

    def generate_structured(
        self,
        *,
        system: str,
        user: str,
        schema: dict,
        temperature: float = 0.2,
    ) -> dict:
        properties = schema.get("properties", {})
        if "assistant_message" in properties:
            return self.presentation_payload
        return self.payload

    def list_models(self) -> list[str]:
        return ["fake-planner-model"]


def make_planner_payload(
    *,
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
    reasons = {
        "target_area": "",
        "budget_max": "",
        "station_walk_max": "",
        "move_in_date": "",
        "layout_preference": "",
        "must_conditions": "",
        "nice_to_have": "",
    }
    if condition_reasons is not None:
        reasons.update(condition_reasons)
    return {
        "intent": "search",
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
        "condition_reasons": reasons,
    }


def make_risk_planner_payload() -> dict:
    return {
        "intent": "risk_check",
        "user_memory": {
            "target_area": None,
            "budget_max": None,
            "station_walk_max": None,
            "layout_preference": None,
            "move_in_date": None,
            "must_conditions": [],
            "nice_to_have": [],
        },
        "missing_slots": [],
        "follow_up_questions": [],
        "next_action": "risk_check",
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


def install_fake_route_adapters(
    orchestrator: HousingOrchestrator,
    *,
    planner_adapter: LLMAdapter | None = None,
    research_default_adapter: LLMAdapter | None = None,
) -> None:
    planner_adapter = planner_adapter or FakePlannerRouteAdapter(make_planner_payload())

    def _get_adapter_for_route(**kwargs):
        route_key = kwargs.get("route_key")
        if route_key == "planner":
            return planner_adapter
        if route_key == "research_default":
            return research_default_adapter
        return None

    orchestrator._get_adapter_for_route = _get_adapter_for_route


def test_orchestrator_stage_flow_is_interactive(tmp_path: Path):
    database_path = str(tmp_path / "housing.db")
    db = Database(database_path)
    db.init()
    orchestrator = HousingOrchestrator(settings=build_settings(database_path), db=db)
    install_fake_route_adapters(orchestrator)

    session_id, _ = db.create_session()

    search_response = orchestrator.process_user_message(
        session_id=session_id,
        message="江東区で家賃12万円以下、駅徒歩7分以内の1LDKを探しています",
        provider="openai",
    )

    assert search_response.status == "awaiting_plan_confirmation"
    assert search_response.next_action == "approve_research_plan"
    assert any(block.type == "plan" for block in search_response.blocks)

    queued_response = orchestrator.execute_action(
        session_id=session_id,
        action_type="approve_research_plan",
        payload={},
    )

    assert queued_response.status == "research_queued"
    assert any(block.type == "timeline" for block in queued_response.blocks)
    assert any(block.type == "tree" for block in queued_response.blocks)

    assert orchestrator.process_next_research_job() is True
    research_state = orchestrator.get_research_state(session_id)
    assert research_state.status == "completed"
    assert research_state.response is not None
    assert research_state.response.status == "research_completed"
    assert any(block.type == "cards" for block in research_state.response.blocks)
    assert any(block.type == "tree" for block in research_state.response.blocks)
    assert any(block.title == "最終レポート" for block in research_state.response.blocks)

    user_memory, task_memory = db.get_memories(session_id)
    selected_property_id = task_memory["last_ranked_properties"][0]["property_id_norm"]
    comparison_response = orchestrator.execute_action(
        session_id=session_id,
        action_type="compare_selected_properties",
        payload={
            "property_ids": [
                item["property_id_norm"] for item in task_memory["last_ranked_properties"][:2]
            ]
        },
    )

    assert comparison_response.status == "research_completed"
    assert any(block.title == "選択物件の比較表" for block in comparison_response.blocks)

    inquiry_response = orchestrator.execute_action(
        session_id=session_id,
        action_type="generate_inquiry",
        payload={"property_id": selected_property_id},
    )

    assert inquiry_response.status == "inquiry_draft_ready"
    assert inquiry_response.pending_confirmation is True
    assert any(block.title == "問い合わせ下書き" for block in inquiry_response.blocks)

    contract_prompt = orchestrator.execute_action(
        session_id=session_id,
        action_type="start_contract_review",
        payload={"property_id": selected_property_id},
    )

    assert contract_prompt.status == "awaiting_contract_text"
    assert contract_prompt.next_action == "paste_contract_text"

    install_fake_route_adapters(
        orchestrator,
        planner_adapter=FakePlannerRouteAdapter(make_risk_planner_payload()),
    )

    risk_response = orchestrator.process_user_message(
        session_id=session_id,
        message="更新料1ヶ月。短期解約違約金あり。解約予告2か月前。保証会社加入必須。",
        provider="openai",
    )

    assert risk_response.status == "risk_check_completed"
    assert any(block.title == "契約リスク一覧" for block in risk_response.blocks)
    _, updated_task_memory = db.get_memories(session_id)
    assert updated_task_memory["awaiting_contract_text"] is False
    assert updated_task_memory["risk_items"]


def test_orchestrator_uses_llm_generated_plan_content(tmp_path: Path):
    database_path = str(tmp_path / "housing.db")
    db = Database(database_path)
    db.init()
    orchestrator = HousingOrchestrator(settings=build_settings(database_path), db=db)
    planner_adapter = FakePlannerRouteAdapter(
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
                "rationale": "希少条件があるため、母集団を先に確保してから絞り込む進め方を取ります。",
            },
            condition_reasons={
                "target_area": "生活圏を固定したい条件なので検索の起点にします。",
                "budget_max": "毎月の予算上限なので厳格に見ます。",
                "station_walk_max": "通勤や移動負担に直結するためです。",
                "move_in_date": "",
                "layout_preference": "仕事スペースを確保できるかの判断軸にします。",
                "must_conditions": "ペット可は候補数が少ないため優先度が高い条件です。",
                "nice_to_have": "在宅ワーク条件は比較の差が出やすいので加点軸にします。",
            },
        )
    )
    install_fake_route_adapters(orchestrator, planner_adapter=planner_adapter)

    session_id, _ = db.create_session()
    response = orchestrator.process_user_message(
        session_id=session_id,
        message="江東区で家賃12万円以下、ペット可、在宅ワークしやすい1LDKを探したい",
        provider="openai",
    )

    assert response.status == "awaiting_plan_confirmation"
    plan_block = next(block for block in response.blocks if block.type == "plan")
    assert plan_block.content["summary"] == "江東区でペット可かつ在宅ワークしやすい1LDKを優先して調べます。"
    assert plan_block.content["rationale"] == "希少条件があるため、母集団を先に確保してから絞り込む進め方を取ります。"
    assert plan_block.content["seed_queries"][1] == "江東区で家賃12万円以下、ペット可で在宅ワークしやすい賃貸"
    must_condition = next(
        item for item in plan_block.content["conditions"] if item["label"] == "必須条件"
    )
    assert must_condition["reason"] == "ペット可は候補数が少ないため優先度が高い条件です。"


def test_profile_memory_is_available_across_sessions(tmp_path: Path):
    database_path = str(tmp_path / "housing.db")
    db = Database(database_path)
    db.init()
    orchestrator = HousingOrchestrator(settings=build_settings(database_path), db=db)
    install_fake_route_adapters(orchestrator)

    profile_id, _ = db.get_or_create_profile("local-profile-1")
    first_session_id, _ = db.create_session(profile_id=profile_id)

    orchestrator.process_user_message(
        session_id=first_session_id,
        message="江東区で家賃12万円以下、駅徒歩7分以内の1LDKを探しています",
        provider="openai",
    )
    orchestrator.execute_action(
        session_id=first_session_id,
        action_type="approve_research_plan",
        payload={},
    )
    assert orchestrator.process_next_research_job() is True
    _, first_task_memory = db.get_memories(first_session_id)
    selected_property_id = first_task_memory["last_ranked_properties"][0]["property_id_norm"]

    reaction_response = orchestrator.execute_action(
        session_id=first_session_id,
        action_type="record_property_reaction",
        payload={"property_id": selected_property_id, "reaction": "favorite"},
    )

    assert reaction_response.status == "research_completed"

    second_session_id, _ = db.create_session(profile_id=profile_id)
    initial_response = orchestrator.build_session_initial_response(second_session_id)

    assert initial_response is not None
    assert initial_response.status == "awaiting_profile_resume"
    assert initial_response.assistant_message == "前回の条件を引き継ぎますか？"
    assert any("江東区" in str(block.content.get("body", "")) for block in initial_response.blocks)


def test_orchestrator_uses_llm_plan_presentation_for_plan_copy(tmp_path: Path):
    database_path = str(tmp_path / "housing.db")
    db = Database(database_path)
    db.init()
    orchestrator = HousingOrchestrator(settings=build_settings(database_path), db=db)
    planner_adapter = FakePlannerRouteAdapter(
        payload=make_planner_payload(
            target_area="町田",
            budget_max=100000,
            station_walk_max=None,
            layout_preference=None,
            must_conditions=["RC造"],
            follow_up_questions=[
                {
                    "slot": "station_walk_max",
                    "label": "駅徒歩",
                    "question": "駅からの距離はどこまで許容できますか？",
                    "examples": ["徒歩7分以内", "徒歩10分まで", "少し遠くても可"],
                }
            ],
            seed_queries=[
                "町田 賃貸 10万円 RC造",
                "町田で家賃10万円以下、RC造を優先した賃貸",
            ],
            research_plan={"summary": "", "goal": "", "strategy": [], "rationale": ""},
            condition_reasons={
                "target_area": "町田で探したい意思が明確なので検索の起点にします。",
                "budget_max": "予算超過の候補を早めに除外するためです。",
                "station_walk_max": "",
                "move_in_date": "",
                "layout_preference": "",
                "must_conditions": "RC造は建物構造の優先条件として確認します。",
                "nice_to_have": "",
            },
        ),
        presentation_payload={
            "assistant_message": (
                "町田で家賃10万円以下、RC造を優先する前提で調査計画をまとめました。"
                "駅距離が分かればさらに絞り込みやすいので、分かる範囲で追加できます。"
            ),
            "summary": "町田で予算内に収まるRC造の賃貸を先に広めに集め、条件が揃う順に比較します。",
            "goal": "内見や問い合わせに進める候補を、構造条件と予算の両面から数件まで絞り込みます。",
            "rationale": "RC造は候補数が限られやすいため、まず母集団を確保してから駅距離などで整理する方が取りこぼしを防げます。",
            "strategy": [
                "町田エリアで家賃10万円以下かつRC造の募集を横断的に集めます。",
                "構造表記が曖昧な掲載は詳細ページでRC造かどうかを優先確認します。",
                "駅距離や管理費込みの実支払額を見て、問い合わせ候補を上位化します。",
            ],
            "open_questions": ["駅から徒歩何分くらいまで許容できるか"],
        },
    )
    install_fake_route_adapters(orchestrator, planner_adapter=planner_adapter)

    session_id, _ = db.create_session()
    response = orchestrator.process_user_message(
        session_id=session_id,
        message="町田に10万円以下でRCの物件に住みたい",
        provider="openai",
    )

    assert response.status == "awaiting_plan_confirmation"
    assert response.assistant_message.startswith("町田で家賃10万円以下、RC造を優先する前提")
    plan_block = next(block for block in response.blocks if block.type == "plan")
    assert plan_block.content["summary"] == "町田で予算内に収まるRC造の賃貸を先に広めに集め、条件が揃う順に比較します。"
    assert plan_block.content["strategy"][0] == "町田エリアで家賃10万円以下かつRC造の募集を横断的に集めます。"
    assert plan_block.content["open_questions"] == ["駅から徒歩何分くらいまで許容できるか"]


def test_build_research_queries_adds_nearby_line_and_relaxed_variants(tmp_path: Path):
    database_path = str(tmp_path / "housing.db")
    db = Database(database_path)
    db.init()
    orchestrator = HousingOrchestrator(settings=build_settings(database_path), db=db)

    queries = orchestrator._build_research_queries(
        {
            "target_area": "町田",
            "budget_max": 100000,
            "station_walk_max": 10,
            "layout_preference": None,
            "move_in_date": None,
            "must_conditions": ["RC造"],
            "nice_to_have": [],
        },
        seed_queries=[
            "町田 賃貸 10万円以下 RC造",
            "町田で家賃10万円以下のRC造賃貸",
            "町田 RC造 賃貸 10万円",
            "町田 鉄筋コンクリート 賃貸 10万円以下",
        ],
    )

    assert len(queries) <= 8
    assert queries[0] == "町田 賃貸 10万円以下 RC造"
    assert any(any(area in query for area in ["相模原", "橋本", "南町田"]) for query in queries)
    assert any("小田急線" in query or "横浜線" in query for query in queries)
    assert any("11万円以下" in query for query in queries)
    assert any(
        "町田" in query and "RC造" not in query and "鉄筋コンクリート" not in query
        for query in queries
    )


def test_research_completed_response_prefers_llm_summary(tmp_path: Path):
    database_path = str(tmp_path / "housing.db")
    db = Database(database_path)
    db.init()
    orchestrator = HousingOrchestrator(settings=build_settings(database_path), db=db)
    summary = (
        "江東区で家賃12万円以内・駅徒歩7分以内・1LDKの条件では3件を比較できました。"
        "最上位候補は駅徒歩4分で条件に最も近く、まずは問い合わせに進める候補です。"
    )
    adapter = FakeResearchSummaryAdapter(summary)
    install_fake_route_adapters(orchestrator, research_default_adapter=adapter)

    session_id, _ = db.create_session()
    orchestrator.process_user_message(
        session_id=session_id,
        message="江東区で家賃12万円以下、駅徒歩7分以内の1LDKを探しています",
        provider="openai",
    )
    orchestrator.execute_action(
        session_id=session_id,
        action_type="approve_research_plan",
        payload={},
    )

    assert orchestrator.process_next_research_job() is True

    research_state = orchestrator.get_research_state(session_id)
    assert research_state.latest_summary == summary
    assert research_state.response is not None
    assert research_state.response.assistant_message == summary
    summary_block = next(
        block for block in research_state.response.blocks if block.title == "調査サマリー"
    )
    assert summary_block.content["body"] == summary
    report_block = next(
        block for block in research_state.response.blocks if block.title == "最終レポート"
    )
    assert report_block.content["body"].startswith("# 最終レポート")

    _, task_memory = db.get_memories(session_id)
    assert task_memory["last_research_summary"] == summary
    assert task_memory["last_final_report"].startswith("# 最終レポート")


def test_research_summary_prompt_includes_branch_tradeoffs_and_followups(tmp_path: Path):
    database_path = str(tmp_path / "housing.db")
    db = Database(database_path)
    db.init()
    orchestrator = HousingOrchestrator(settings=build_settings(database_path), db=db)
    adapter = FakeResearchSummaryAdapter(
        "結論: 問い合わせ候補があります。\n"
        "理由: 条件一致度が最も高い分岐を採用しました。\n"
        "懸念: 一部の条件は再確認が必要です。\n"
        "次の確認事項: 初期費用と契約条件を確認してください。"
    )
    install_fake_route_adapters(orchestrator, research_default_adapter=adapter)

    session_id, _ = db.create_session()
    orchestrator.process_user_message(
        session_id=session_id,
        message="江東区で家賃12万円以下、駅徒歩7分以内の1LDKを探しています",
        provider="openai",
    )
    orchestrator.execute_action(
        session_id=session_id,
        action_type="approve_research_plan",
        payload={},
    )

    assert orchestrator.process_next_research_job() is True

    payload = json.loads(adapter.last_text_user)
    assert payload["selected_branch"]["branch_id"]
    assert payload["alternative_branches"]
    assert payload["failure_summary"]["recommendations"]
    assert payload["confirmation_items"]
    assert "候補比較表" in adapter.last_report_user
    assert "selected_path" in adapter.last_report_user


def test_fresh_start_session_skips_profile_resume_prompt(tmp_path: Path):
    database_path = str(tmp_path / "housing.db")
    db = Database(database_path)
    db.init()
    orchestrator = HousingOrchestrator(settings=build_settings(database_path), db=db)
    install_fake_route_adapters(
        orchestrator,
        planner_adapter=FakePlannerRouteAdapter(
            make_planner_payload(
                target_area="吉祥寺",
                budget_max=120000,
                station_walk_max=7,
                layout_preference=None,
                nice_to_have=["在宅ワーク向け"],
            )
        ),
    )

    profile_id, _ = db.get_or_create_profile("local-profile-2")
    first_session_id, _ = db.create_session(profile_id=profile_id)
    orchestrator.process_user_message(
        session_id=first_session_id,
        message="吉祥寺で家賃12万円以下、駅徒歩7分以内、在宅ワーク向けで探したい",
        provider="openai",
    )
    orchestrator.execute_action(
        session_id=first_session_id,
        action_type="approve_research_plan",
        payload={},
    )
    assert orchestrator.process_next_research_job() is True

    app.state.db = db
    app.state.orchestrator = orchestrator

    response = create_session_endpoint(
        CreateSessionRequest(profile_id=profile_id, fresh_start=True)
    )

    assert response.profile_id == profile_id
    assert response.initial_response is None


def test_retry_and_reaction_feed_strategy_memory(tmp_path: Path):
    database_path = str(tmp_path / "housing.db")
    db = Database(database_path)
    db.init()
    orchestrator = HousingOrchestrator(settings=build_settings(database_path), db=db)
    install_fake_route_adapters(orchestrator)

    profile_id, _ = db.get_or_create_profile("strategy-profile-1")
    session_id, _ = db.create_session(profile_id=profile_id)

    orchestrator.process_user_message(
        session_id=session_id,
        message="江東区で家賃12万円以下、駅徒歩7分以内の1LDKを探しています",
        provider="openai",
    )
    orchestrator.execute_action(
        session_id=session_id,
        action_type="approve_research_plan",
        payload={},
    )
    assert orchestrator.process_next_research_job() is True

    _, task_memory = db.get_memories(session_id)
    selected_property_id = task_memory["last_ranked_properties"][0]["property_id_norm"]
    orchestrator.execute_action(
        session_id=session_id,
        action_type="record_property_reaction",
        payload={"property_id": selected_property_id, "reaction": "favorite"},
    )

    profile = db.get_profile(profile_id)
    assert profile is not None
    strategy_memory = profile["profile_memory"].get("strategy_memory", {})
    assert strategy_memory["preferred_strategy_tags"]

    retry_response = orchestrator.execute_action(
        session_id=session_id,
        action_type="retry_research_job",
        payload={},
    )
    assert retry_response.status == "research_queued"

    latest_job = db.get_latest_research_job(session_id)
    assert latest_job is not None
    retry_context = latest_job["approved_plan"].get("retry_context", {})
    assert retry_context["selected_path"]
    assert retry_context["top_issues"] == task_memory["failure_summary"]["top_issues"]
