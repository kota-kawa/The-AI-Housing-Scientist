from __future__ import annotations

from typing import Any

from app.config import ProviderName
from app.db import utc_now_iso
from app.llm_config import route_config_for
from app.models import ChatMessageResponse, UIBlock
from app.stages import run_communication
from app.stages.planner import REQUIRED_PLANNING_SLOTS, detect_search_signal, run_planner
from app.stages.risk_check import looks_like_contract_text


class OrchestratorActionsMixin:
    # JP: process user messageを処理する。
    # EN: Process process user message.
    def process_user_message(
        self,
        *,
        session_id: str,
        message: str,
        planner_answers: list[dict[str, Any]] | None = None,
        provider: ProviderName | None,
    ) -> ChatMessageResponse:
        user_memory, task_memory, llm_config = self._ensure_session_llm_config(session_id)
        contract_like = looks_like_contract_text(message)
        active_job_id = str(task_memory.get("active_research_job_id") or "").strip()
        active_job = self.db.get_research_job(active_job_id) if active_job_id else None

        if active_job and active_job["status"] in {"queued", "running"}:
            response = self._build_research_running_response(active_job)
            self.db.add_message(session_id, "assistant", response.model_dump())
            self.db.set_session_status(session_id, response.status)
            return response

        adapter = self._get_adapter_for_route(
            llm_config=llm_config,
            route_key="planner",
            session_id=session_id,
            job_id=active_job_id or None,
            interaction_type="planner",
        )
        planner_result = run_planner(
            message=message,
            user_memory=user_memory,
            adapter=adapter,
            profile_memory=self._get_profile_memory_for_session(session_id),
            planner_answers=planner_answers,
        )
        search_signal = detect_search_signal(message, planner_result=planner_result)

        if task_memory.get("awaiting_contract_text") and not search_signal:
            return self._process_contract_text(session_id=session_id, source_text=message)

        if contract_like and not search_signal:
            return self._process_contract_text(session_id=session_id, source_text=message)

        if not search_signal:
            return self._annotate_response_labels(
                self._build_guidance_response(
                    session_id=session_id,
                    task_memory=task_memory,
                    message=message,
                )
            )

        return self._annotate_response_labels(
            self._process_search_message(
                session_id=session_id,
                message=message,
                adapter=adapter,
                llm_config=llm_config,
                planner_result=planner_result,
                user_memory=user_memory,
                task_memory=task_memory,
            )
        )

    # JP: actionを実行する。
    # EN: Execute action.
    def execute_action(
        self,
        *,
        session_id: str,
        action_type: str,
        payload: dict[str, Any],
    ) -> ChatMessageResponse:
        session = self.db.get_session(session_id)
        if session is None:
            raise RuntimeError("session not found")

        user_memory, task_memory, llm_config = self._ensure_session_llm_config(session_id)
        normalized_properties = task_memory.get("last_normalized_properties", [])
        ranked_properties = task_memory.get("last_ranked_properties", [])
        property_reactions = self._get_property_reactions(task_memory)
        latest_job_id = str(task_memory.get("last_research_job_id") or "")
        latest_job = self.db.get_research_job(latest_job_id) if latest_job_id else None

        if action_type == "resume_profile_memory":
            task_memory["profile_resume_pending"] = False
            task_memory["status"] = "awaiting_plan_inputs"
            self.db.update_memories(session_id, {}, task_memory)

            message = "前回の条件は引き継がず、新しい条件で住まい探しを始めます。"
            response = ChatMessageResponse(
                status="awaiting_plan_inputs",
                assistant_message=message,
                missing_slots=[],
                next_action="await_search_input",
                blocks=[UIBlock(type="text", title="新しい検索を開始", content={"body": message})],
                pending_confirmation=False,
                pending_action=None,
            )
            self.db.set_session_status(session_id, "awaiting_plan_inputs")
            self.db.add_message(session_id, "assistant", response.model_dump())
            return response

        if action_type == "dismiss_profile_resume":
            task_memory["profile_resume_pending"] = False
            task_memory["status"] = "awaiting_plan_inputs"
            self.db.update_memories(session_id, {}, task_memory)

            message = "新しい条件で住まい探しを始めます。まずは物件種別、希望エリア、予算、間取りを入力してください。"
            response = ChatMessageResponse(
                status="awaiting_plan_inputs",
                assistant_message=message,
                missing_slots=[],
                next_action="await_search_input",
                blocks=[UIBlock(type="text", title="新しい検索を開始", content={"body": message})],
                pending_confirmation=False,
                pending_action=None,
            )
            self.db.set_session_status(session_id, "awaiting_plan_inputs")
            self.db.add_message(session_id, "assistant", response.model_dump())
            return response

        if action_type == "approve_research_plan":
            active_job_id = str(task_memory.get("active_research_job_id") or "").strip()
            if active_job_id:
                active_job = self.db.get_research_job(active_job_id)
                if active_job is not None and active_job["status"] in {"queued", "running"}:
                    response = self._build_research_running_response(active_job)
                    self.db.set_session_status(session_id, response.status)
                    self.db.add_message(session_id, "assistant", response.model_dump())
                    return response

            draft_plan = task_memory.get("draft_research_plan")
            if not isinstance(draft_plan, dict) or not draft_plan:
                raise RuntimeError("承認できる調査計画がありません")

            approved_llm_config = self._normalize_llm_config(task_memory.get("draft_llm_config"))
            research_route = route_config_for(approved_llm_config, "research_default")
            approved_plan = {**draft_plan, "approved_at": utc_now_iso()}
            job_id, _ = self.db.create_research_job(
                session_id=session_id,
                provider=self._resolve_provider_for_model(str(research_route["model"])),
                llm_config=approved_llm_config,
                approved_plan=approved_plan,
            )
            task_memory["status"] = "research_queued"
            task_memory["approved_research_plan"] = approved_plan
            task_memory["approved_llm_config"] = approved_llm_config
            task_memory["active_research_job_id"] = job_id
            task_memory["last_research_job_id"] = job_id
            task_memory["awaiting_contract_text"] = False
            task_memory["selected_property_id"] = None
            task_memory["property_reactions"] = {}
            task_memory["comparison_property_ids"] = []
            self.db.set_pending_action(session_id, None)
            self.db.update_memories(session_id, user_memory, task_memory)
            job = self.db.get_research_job(job_id)
            if job is None:
                raise RuntimeError("research job creation failed")
            response = self._build_research_running_response(job)
            self.db.set_session_status(session_id, response.status)
            self.db.add_message(session_id, "assistant", response.model_dump())
            return response

        if action_type == "revise_research_plan":
            current_plan = task_memory.get("draft_research_plan") or {}
            profile_memory = self._get_profile_memory_for_session(session_id)
            task_memory["status"] = "awaiting_plan_inputs"
            self.db.update_memories(session_id, user_memory, task_memory)
            required_questions = self._build_planning_questions(
                user_memory=user_memory,
                slots=list(REQUIRED_PLANNING_SLOTS),
                required=True,
                profile_memory=profile_memory,
            )
            optional_questions = self._build_planning_questions(
                user_memory=user_memory,
                slots=[
                    "layout_preference",
                    "station_walk_max",
                    "move_in_date",
                    "must_conditions",
                    "nice_to_have",
                ],
                required=False,
                profile_memory=profile_memory,
            )
            message = "条件を選び直すと、計画を更新してから再度確認できます。"
            blocks = []
            if current_plan:
                blocks.append(self._build_plan_block(current_plan))
            blocks.append(
                self._build_question_block(
                    questions=required_questions,
                    optional=False,
                )
            )
            blocks.append(
                self._build_question_block(
                    questions=optional_questions,
                    optional=True,
                )
            )
            response = ChatMessageResponse(
                status="awaiting_plan_inputs",
                assistant_message=message,
                missing_slots=[],
                next_action="await_search_input",
                blocks=blocks,
                pending_confirmation=False,
                pending_action=None,
            )
            self.db.set_session_status(session_id, "awaiting_plan_inputs")
            self.db.add_message(session_id, "assistant", response.model_dump())
            return response

        if action_type == "retry_research_job":
            approved_plan = task_memory.get("approved_research_plan") or (
                latest_job["approved_plan"] if latest_job is not None else None
            )
            if not isinstance(approved_plan, dict) or not approved_plan:
                raise RuntimeError("再実行できる調査計画がありません")

            draft_llm_config = self._normalize_llm_config(task_memory.get("draft_llm_config"))
            research_route = route_config_for(draft_llm_config, "research_default")
            retry_plan = {
                **approved_plan,
                "retry_context": {
                    "selected_branch_id": str(task_memory.get("selected_branch_id") or ""),
                    "selected_path": task_memory.get("selected_path") or [],
                    "search_tree_summary": task_memory.get("search_tree_summary") or {},
                    "top_issues": (task_memory.get("failure_summary") or {}).get("top_issues", [])
                    or [],
                },
            }
            job_id, _ = self.db.create_research_job(
                session_id=session_id,
                provider=self._resolve_provider_for_model(str(research_route["model"])),
                llm_config=draft_llm_config,
                approved_plan=retry_plan,
            )
            task_memory["status"] = "research_queued"
            task_memory["approved_llm_config"] = draft_llm_config
            task_memory["approved_research_plan"] = retry_plan
            task_memory["active_research_job_id"] = job_id
            task_memory["last_research_job_id"] = job_id
            self.db.update_memories(session_id, user_memory, task_memory)
            job = self.db.get_research_job(job_id)
            if job is None:
                raise RuntimeError("research job creation failed")
            response = self._build_research_running_response(job)
            self.db.set_session_status(session_id, response.status)
            self.db.add_message(session_id, "assistant", response.model_dump())
            return response

        if action_type == "compare_selected_properties":
            property_ids = [
                str(item).strip()
                for item in payload.get("property_ids", []) or []
                if str(item).strip()
            ]
            task_memory["comparison_property_ids"] = property_ids
            self.db.update_memories(session_id, user_memory, task_memory)

            response = ChatMessageResponse(
                status="research_completed",
                assistant_message=f"選択した{len(property_ids)}件を比較しました。",
                missing_slots=[],
                next_action="select_property",
                blocks=(
                    (
                        self._build_research_progress_blocks(
                            latest_job,
                            task_memory=task_memory,
                        )
                        if latest_job
                        else []
                    )
                    + self._build_compare_blocks(
                        property_ids=property_ids,
                        ranked_properties=self._visible_ranked_properties(
                            ranked_properties, task_memory
                        ),
                        normalized_properties=normalized_properties,
                        property_reactions=property_reactions,
                    )
                ),
                pending_confirmation=False,
                pending_action=None,
            )
            self.db.set_session_status(session_id, "research_completed")
            self.db.add_message(session_id, "assistant", response.model_dump())
            return response

        if action_type == "record_property_reaction":
            property_id = str(payload.get("property_id") or "").strip()
            reaction = str(payload.get("reaction") or "").strip()
            if not property_id or reaction not in {"favorite", "exclude", "clear"}:
                raise RuntimeError("property_id and valid reaction are required")

            property_snapshot = next(
                (
                    item
                    for item in normalized_properties
                    if item.get("property_id_norm") == property_id
                ),
                None,
            )
            if property_snapshot is None:
                raise RuntimeError("property not found")

            updated_reactions = dict(property_reactions)
            if reaction == "clear":
                updated_reactions.pop(property_id, None)
            else:
                updated_reactions[property_id] = reaction

            task_memory["property_reactions"] = updated_reactions
            if reaction == "exclude" and task_memory.get("selected_property_id") == property_id:
                task_memory["selected_property_id"] = None

            self.db.update_memories(session_id, user_memory, task_memory)

            visible_ranked_properties = self._visible_ranked_properties(
                ranked_properties, task_memory
            )
            reaction_label = {
                "favorite": "気になる",
                "exclude": "除外",
                "clear": "解除",
            }[reaction]
            property_name = self._find_property_name(task_memory, property_id)
            response = ChatMessageResponse(
                status="research_completed",
                assistant_message=f"{property_name}を「{reaction_label}」として記録しました。",
                missing_slots=[],
                next_action="select_property",
                blocks=self._build_research_result_blocks(
                    research_summary=task_memory.get("last_research_summary", ""),
                    final_report_markdown=task_memory.get("last_final_report", ""),
                    ranked_properties=visible_ranked_properties,
                    normalized_properties=normalized_properties,
                    search_summary=task_memory.get("last_search_summary", {}),
                    source_items=task_memory.get("last_source_items", []) or [],
                    task_memory=task_memory,
                    job_id=latest_job_id or None,
                ),
                pending_confirmation=False,
                pending_action=None,
            )
            self.db.set_session_status(session_id, "research_completed")
            self.db.add_message(session_id, "assistant", response.model_dump())
            return response

        if action_type == "generate_inquiry":
            property_id = str(payload.get("property_id") or "")
            if not property_id:
                raise RuntimeError("property_id is required")

            adapter = self._get_adapter_for_route(
                llm_config=llm_config,
                route_key="communication",
                session_id=session_id,
                job_id=latest_job_id or None,
                interaction_type="communication",
            )

            communication_result = run_communication(
                ranked_properties=ranked_properties,
                normalized_properties=normalized_properties,
                user_memory=user_memory,
                selected_property_id=property_id,
                adapter=adapter,
            )
            self.db.add_audit_event(
                session_id,
                "communication",
                {"property_id": property_id},
                communication_result,
                "選択物件の問い合わせ文を生成",
            )

            task_memory["status"] = "inquiry_draft_ready"
            task_memory["awaiting_contract_text"] = False
            task_memory["selected_property_id"] = property_id

            pending_action = communication_result["pending_action"]
            self.db.set_pending_action(session_id, pending_action)
            self.db.update_memories(session_id, user_memory, task_memory)

            property_name = self._find_property_name(task_memory, property_id)
            response = ChatMessageResponse(
                status="inquiry_draft_ready",
                assistant_message=f"{property_name}の問い合わせ文を作成しました。必要ならそのまま契約書チェックにも進めます。",
                missing_slots=[],
                next_action="confirm_before_send",
                blocks=self._build_inquiry_blocks(
                    ranked_properties=self._visible_ranked_properties(
                        ranked_properties, task_memory
                    ),
                    normalized_properties=normalized_properties,
                    communication=communication_result,
                    selected_property_id=property_id,
                ),
                pending_confirmation=pending_action is not None,
                pending_action=pending_action,
            )
            self.db.set_session_status(session_id, "inquiry_draft_ready")
            self.db.add_message(session_id, "assistant", response.model_dump())
            return response

        if action_type == "start_contract_review":
            property_id = str(
                payload.get("property_id") or task_memory.get("selected_property_id") or ""
            )
            task_memory["status"] = "awaiting_contract_text"
            task_memory["awaiting_contract_text"] = True
            if property_id:
                task_memory["selected_property_id"] = property_id
            self.db.update_memories(session_id, user_memory, task_memory)

            property_name = self._find_property_name(task_memory, property_id)
            response = ChatMessageResponse(
                status="awaiting_contract_text",
                assistant_message="契約書チェックモードに切り替えました。文面を貼り付けてください。",
                missing_slots=[],
                next_action="paste_contract_text",
                blocks=self._build_contract_prompt_blocks(property_name),
                pending_confirmation=session.get("pending_action") is not None,
                pending_action=session.get("pending_action"),
            )
            self.db.set_session_status(session_id, "awaiting_contract_text")
            self.db.add_message(session_id, "assistant", response.model_dump())
            return response

        raise RuntimeError(f"unsupported action_type: {action_type}")

    # JP: confirm actionを処理する。
    # EN: Process confirm action.
    def confirm_action(
        self,
        *,
        session_id: str,
        action_type: str,
        approved: bool,
    ) -> ChatMessageResponse:
        session = self.db.get_session(session_id)
        if session is None:
            raise RuntimeError("session not found")

        pending_action = session.get("pending_action")
        if pending_action is None or pending_action.get("action_type") != action_type:
            raise RuntimeError("no matching pending action")

        user_memory, task_memory = self.db.get_memories(session_id)

        if approved:
            task_memory["status"] = "inquiry_marked_as_sent"
            task_memory["last_action"] = {"action_type": action_type, "approved": True}
            message = "確認済みとして処理しました。実送信は外部連携時に実行されます。"
        else:
            task_memory["status"] = "inquiry_cancelled"
            task_memory["last_action"] = {"action_type": action_type, "approved": False}
            message = "送信操作をキャンセルしました。内容を修正して再確認できます。"

        self.db.set_pending_action(session_id, None)
        self.db.update_memories(session_id, user_memory, task_memory)

        response = ChatMessageResponse(
            status="completed",
            assistant_message=message,
            missing_slots=[],
            next_action="await_next_input",
            blocks=[UIBlock(type="text", title="操作結果", content={"body": message})],
            pending_confirmation=False,
            pending_action=None,
        )
        self.db.set_session_status(session_id, "completed")
        self.db.add_message(session_id, "assistant", response.model_dump())
        return response
