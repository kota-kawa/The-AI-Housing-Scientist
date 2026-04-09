import { FormEvent, KeyboardEvent, useEffect, useMemo, useRef, useState } from "react";

import StructuredBlock from "./components/StructuredBlock";
import {
  ActionDescriptor,
  ChatMessageResponse,
  LLMCapabilities,
  LLMConfig,
  LLMRouteKey,
  SessionLLMConfig,
  UIBlock,
  confirmAction,
  createSession,
  fetchLlmCapabilities,
  fetchPreflight,
  fetchResearchState,
  fetchSessionLlmConfig,
  runAction,
  saveSessionLlmConfig,
  sendMessage,
} from "./lib/api";

type Message = {
  id: string;
  role: "user" | "assistant";
  text: string;
  status?: string;
  blocks?: UIBlock[];
  pendingAction?: ActionDescriptor | null;
};

type ChecklistEntry = {
  label: string;
  checked: boolean;
};

type QuestionEntry = {
  label?: string;
  question?: string;
  examples?: string[];
  selected_example?: string;
};

type CardEntry = {
  id?: string;
  compare_selected?: boolean;
};

const PROFILE_STORAGE_KEY = "housing_scientist_profile_id";

const SAMPLE_PROMPTS: string[] = [
  "江東区で家賃12万円以下、駅徒歩7分以内の1LDKを探しています",
  "ペット可・新宿まで30分以内・2LDK・南向きで比較してほしい",
  "築20年以内、保証人不要、家賃8万円台で安心して住める物件を教えて",
  "在宅ワーク向けに、書斎が取れる広めの間取りを3件比較したい",
];

function toAssistantMessage(payload: ChatMessageResponse): Message {
  return {
    id: crypto.randomUUID(),
    role: "assistant",
    text: payload.assistant_message,
    status: payload.status,
    blocks: payload.blocks,
    pendingAction: payload.pending_action,
  };
}

function cloneLlmConfig(config: LLMConfig): LLMConfig {
  return {
    preset: config.preset,
    routes: {
      planner: { ...config.routes.planner },
      research_default: { ...config.routes.research_default },
      communication: { ...config.routes.communication },
      risk_check: { ...config.routes.risk_check },
    },
  };
}

function formatPresetLabel(preset: string): string {
  if (preset === "default") {
    return "Default";
  }
  if (preset === "custom") {
    return "Custom";
  }
  return preset;
}

function toStatusLabel(payload: ChatMessageResponse): string {
  if (payload.status_label) {
    return payload.status_label;
  }
  if (payload.status === "awaiting_profile_resume") {
    return "前回条件の引き継ぎ確認";
  }
  if (payload.status === "awaiting_user_input") {
    return "追加条件の回答待ち";
  }
  if (payload.status === "awaiting_plan_confirmation") {
    return "調査計画の承認待ち";
  }
  if (payload.status === "research_queued") {
    return "調査キュー登録済み";
  }
  if (payload.status === "research_running") {
    return "調査進行中";
  }
  if (payload.status === "research_completed" || payload.status === "search_results_ready") {
    return "調査完了・物件選択待ち";
  }
  if (payload.status === "research_failed") {
    return "調査エラー";
  }
  if (payload.status === "inquiry_draft_ready") {
    return "問い合わせ文の確認待ち";
  }
  if (payload.status === "awaiting_contract_text") {
    return "契約書入力待ち";
  }
  if (payload.status === "risk_check_completed") {
    return "契約リスク確認完了";
  }
  if (payload.pending_confirmation) {
    return "確認待ち";
  }
  return "処理完了";
}

function ChevronIcon({ open }: { open: boolean }) {
  return (
    <svg
      viewBox="0 0 20 20"
      fill="none"
      aria-hidden="true"
      className={`h-4 w-4 transition ${open ? "rotate-180" : ""}`}
    >
      <path
        d="M5 7.5L10 12.5L15 7.5"
        stroke="currentColor"
        strokeWidth="1.8"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

function SendIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" aria-hidden="true" className="h-5 w-5">
      <path
        d="M4.5 12H19.5M19.5 12L13 5.5M19.5 12L13 18.5"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

function NewSessionIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" aria-hidden="true" className="h-5 w-5">
      {/* ノート部分 */}
      <path
        d="M12 5H7a2 2 0 0 0-2 2v10a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2v-5"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      {/* 鉛筆部分 */}
      <path
        d="M17.5 3.5a2 2 0 0 1 2.83 2.83L11 15.5l-3.5.5.5-3.5L17.5 3.5z"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

function SparkleIcon({ className = "h-4 w-4" }: { className?: string }) {
  return (
    <svg viewBox="0 0 24 24" fill="none" aria-hidden="true" className={className}>
      <path
        d="M12 3L13.6 8.4L19 10L13.6 11.6L12 17L10.4 11.6L5 10L10.4 8.4L12 3Z"
        stroke="currentColor"
        strokeWidth="1.6"
        strokeLinejoin="round"
      />
      <path
        d="M19 16L19.7 18.3L22 19L19.7 19.7L19 22L18.3 19.7L16 19L18.3 18.3L19 16Z"
        stroke="currentColor"
        strokeWidth="1.4"
        strokeLinejoin="round"
      />
    </svg>
  );
}

function HouseLogoIcon({ className = "h-5 w-5" }: { className?: string }) {
  return (
    <svg viewBox="0 0 24 24" fill="none" aria-hidden="true" className={className}>
      <path
        d="M4 11.5L12 5L20 11.5V19.5C20 20.0523 19.5523 20.5 19 20.5H15V14.5H9V20.5H5C4.44772 20.5 4 20.0523 4 19.5V11.5Z"
        stroke="currentColor"
        strokeWidth="1.8"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

function UserAvatarIcon({ className = "h-5 w-5" }: { className?: string }) {
  return (
    <svg viewBox="0 0 24 24" fill="none" aria-hidden="true" className={className}>
      <circle cx="12" cy="9" r="3.5" stroke="currentColor" strokeWidth="1.8" />
      <path
        d="M5 19.5C5 16.4624 8.13401 14 12 14C15.866 14 19 16.4624 19 19.5"
        stroke="currentColor"
        strokeWidth="1.8"
        strokeLinecap="round"
      />
    </svg>
  );
}

export default function App() {
  const [sessionId, setSessionId] = useState<string>("");
  const [input, setInput] = useState<string>("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [status, setStatus] = useState<string>("初期化中...");
  const [responseState, setResponseState] = useState<string>("");
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string>("");
  const [preflightSummary, setPreflightSummary] = useState<string>("");
  const [activeResearchMessageId, setActiveResearchMessageId] = useState<string>("");
  const [llmCapabilities, setLlmCapabilities] = useState<LLMCapabilities | null>(null);
  const [llmConfig, setLlmConfig] = useState<SessionLLMConfig | null>(null);
  const [llmDraft, setLlmDraft] = useState<LLMConfig | null>(null);
  const [llmEditorOpen, setLlmEditorOpen] = useState<boolean>(false);
  const [llmSaving, setLlmSaving] = useState<boolean>(false);
  const [openModelDropdown, setOpenModelDropdown] = useState<string | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);
  const messagesEndRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const bootstrap = async () => {
      try {
        const storedProfileId = window.localStorage.getItem(PROFILE_STORAGE_KEY) ?? crypto.randomUUID();
        const [session, preflight, capabilities] = await Promise.all([
          createSession(storedProfileId),
          fetchPreflight(),
          fetchLlmCapabilities(),
        ]);
        const sessionLlmConfig = await fetchSessionLlmConfig(session.session_id);
        setSessionId(session.session_id);
        window.localStorage.setItem(PROFILE_STORAGE_KEY, session.profile_id);
        setLlmCapabilities(capabilities);
        setLlmConfig(sessionLlmConfig);
        setLlmDraft(cloneLlmConfig(sessionLlmConfig));

        const providerStates = Object.entries(preflight.providers)
          .map(([name, report]) => `${name}:${report.model_valid ? "OK" : "NG"}`)
          .join(" / ");

        setPreflightSummary(
          `Strict=${preflight.strict_mode ? "ON" : "OFF"} | Brave=${
            preflight.brave_reachable ? "OK" : "NG"
          } | ${providerStates}`
        );

        if (session.initial_response) {
          setMessages([toAssistantMessage(session.initial_response)]);
          setStatus(toStatusLabel(session.initial_response));
          setResponseState(session.initial_response.status);
        } else {
          setStatus("準備完了");
          setResponseState("ready");
        }
      } catch (e) {
        setError(e instanceof Error ? e.message : "初期化に失敗しました");
        setStatus("初期化失敗");
      }
    };

    void bootstrap();
  }, []);

  useEffect(() => {
    if (!openModelDropdown) return;
    const handle = () => setOpenModelDropdown(null);
    document.addEventListener("mousedown", handle);
    return () => document.removeEventListener("mousedown", handle);
  }, [openModelDropdown]);

  useEffect(() => {
    const frame = window.requestAnimationFrame(() => {
      messagesEndRef.current?.scrollIntoView({
        behavior: "smooth",
        block: "end",
      });
    });

    return () => window.cancelAnimationFrame(frame);
  }, [messages, loading]);

  useEffect(() => {
    if (!sessionId || !activeResearchMessageId) {
      return;
    }

    if (!["research_queued", "research_running"].includes(responseState)) {
      return;
    }

    let cancelled = false;
    let timerId: number | undefined;

    const poll = async () => {
      try {
        const researchState = await fetchResearchState(sessionId);
        if (cancelled || !researchState.response) {
          return;
        }

        const updatedMessage: Message = {
          ...toAssistantMessage(researchState.response),
          id: activeResearchMessageId,
        };

        setMessages((prev) =>
          prev.map((message) => (message.id === activeResearchMessageId ? updatedMessage : message))
        );
        setResponseState(researchState.response.status);
        setStatus(toStatusLabel(researchState.response));

        if (["research_completed", "research_failed"].includes(researchState.response.status)) {
          setActiveResearchMessageId("");
          return;
        }
      } catch (e) {
        if (!cancelled) {
          setError(e instanceof Error ? e.message : "進捗取得に失敗しました");
        }
      }

      if (!cancelled) {
        timerId = window.setTimeout(() => void poll(), 2000);
      }
    };

    timerId = window.setTimeout(() => void poll(), 1200);
    return () => {
      cancelled = true;
      if (timerId) {
        window.clearTimeout(timerId);
      }
    };
  }, [activeResearchMessageId, responseState, sessionId]);

  const inputPlaceholder = useMemo(() => {
    if (responseState === "awaiting_contract_text") {
      return "契約書・重要事項説明・初期費用表の文面を貼り付けてください…";
    }
    if (responseState === "awaiting_plan_confirmation") {
      return "条件を追加すると、調査計画を更新できます…";
    }
    if (responseState === "research_queued" || responseState === "research_running") {
      return "調査中です。完了すると結果と参照ソースがここに表示されます…";
    }
    return "希望条件や気になる物件、契約条項を入力してください…";
  }, [responseState]);

  const isResearchBusy = responseState === "research_queued" || responseState === "research_running";

  const pendingAction = useMemo(() => {
    for (let i = messages.length - 1; i >= 0; i -= 1) {
      const item = messages[i];
      if (item.role === "assistant" && "pendingAction" in item) {
        return item.pendingAction ?? null;
      }
    }
    return null;
  }, [messages]);

  const statusTone = useMemo(() => {
    if (error) {
      return "border-rose-200 bg-rose-50 text-rose-700";
    }
    if (loading) {
      return "border-sky-200 bg-sky-50 text-sky-700";
    }
    if (isResearchBusy) {
      return "border-teal-200 bg-teal-50 text-teal-700";
    }
    if (pendingAction) {
      return "border-cyan-200 bg-cyan-50 text-cyan-700";
    }
    if (status === "準備完了") {
      return "border-sky-200 bg-sky-50 text-sky-700";
    }
    return "border-sky-100 bg-sky-50/80 text-sky-700";
  }, [error, isResearchBusy, loading, pendingAction, status]);

  const llmEditable = Boolean(llmConfig?.editable) && !isResearchBusy;

  const llmSummary = useMemo(() => {
    if (!llmConfig || !llmCapabilities) {
      return "設定を読み込み中";
    }
    return llmCapabilities.route_definitions
      .map((route) => `${route.label}:${llmConfig.routes[route.key].model}`)
      .join(" / ");
  }, [llmCapabilities, llmConfig]);

  const appendAssistantResponse = (payload: ChatMessageResponse) => {
    const assistantMessage = toAssistantMessage(payload);
    setMessages((prev) => [...prev, assistantMessage]);
    if (payload.status === "research_queued" || payload.status === "research_running") {
      setActiveResearchMessageId(assistantMessage.id);
    } else {
      setActiveResearchMessageId("");
    }
  };

  const submitMessage = async (messageText: string) => {
    if (!sessionId || !messageText.trim() || loading || isResearchBusy) {
      return;
    }

    const userText = messageText.trim();
    setInput("");
    setError("");
    setLoading(true);
    setStatus(responseState === "awaiting_contract_text" ? "契約条項を確認中..." : "条件を整理中...");

    const userMessage: Message = {
      id: crypto.randomUUID(),
      role: "user",
      text: userText,
    };
    setMessages((prev) => [...prev, userMessage]);

    try {
      const response = await sendMessage(sessionId, userText);
      appendAssistantResponse(response);
      setResponseState(response.status);
      setStatus(toStatusLabel(response));
    } catch (e) {
      setError(e instanceof Error ? e.message : "送信に失敗しました");
      setStatus("エラー");
    } finally {
      setLoading(false);
    }
  };

  const handleLlmRouteModelChange = (routeKey: LLMRouteKey, model: string) => {
    if (!llmDraft) {
      return;
    }
    setLlmDraft({
      ...llmDraft,
      preset: "custom",
      routes: {
        ...llmDraft.routes,
        [routeKey]: { model },
      },
    });
  };

  const handleResetLlmConfig = () => {
    if (!llmCapabilities) {
      return;
    }
    setLlmDraft(cloneLlmConfig(llmCapabilities.default_config));
  };

  const handleSaveLlmConfig = async () => {
    if (!sessionId || !llmDraft || !llmConfig) {
      return;
    }
    setLlmSaving(true);
    setError("");
    try {
      const saved = await saveSessionLlmConfig(sessionId, llmDraft);
      setLlmConfig(saved);
      setLlmDraft(cloneLlmConfig(saved));
      setLlmEditorOpen(false);
    } catch (e) {
      setError(e instanceof Error ? e.message : "LLM設定の保存に失敗しました");
    } finally {
      setLlmSaving(false);
    }
  };

  const handleActionExecute = async (action: ActionDescriptor) => {
    if (!sessionId || loading) {
      return;
    }
    setLoading(true);
    setError("");
    setStatus(action.action_type === "approve_research_plan" ? "調査ジョブを登録中..." : "操作を実行中...");

    try {
      const response = await runAction(sessionId, action.action_type, action.payload ?? {});
      appendAssistantResponse(response);
      setResponseState(response.status);
      setStatus(toStatusLabel(response));
    } catch (e) {
      setError(e instanceof Error ? e.message : "操作に失敗しました");
      setStatus("エラー");
    } finally {
      setLoading(false);
    }
  };

  const submitInput = async () => {
    await submitMessage(input);
  };

  const onSubmit = (event: FormEvent) => {
    event.preventDefault();
    void submitInput();
  };

  const onTextareaKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
      event.preventDefault();
      void submitInput();
    }
  };

  const handleConfirm = async (approved: boolean) => {
    if (!sessionId || !pendingAction || loading) {
      return;
    }
    setLoading(true);
    setStatus("確認処理中...");
    setError("");

    try {
      const response = await confirmAction(sessionId, pendingAction.action_type, approved);
      appendAssistantResponse(response);
      setResponseState(response.status);
      setStatus(toStatusLabel(response));
    } catch (e) {
      setError(e instanceof Error ? e.message : "確認処理に失敗しました");
      setStatus("エラー");
    } finally {
      setLoading(false);
    }
  };

  const handlePromptPick = (prompt: string) => {
    setInput(prompt);
    textareaRef.current?.focus();
  };

  const handleNewSession = async () => {
    if (loading) {
      return;
    }

    setLoading(true);
    setError("");
    setStatus("新しいセッションを準備中...");
    setLlmEditorOpen(false);
    setActiveResearchMessageId("");

    try {
      const profileId = window.localStorage.getItem(PROFILE_STORAGE_KEY) ?? undefined;
      const session = await createSession(profileId, true);
      const sessionLlmConfig = await fetchSessionLlmConfig(session.session_id);
      setSessionId(session.session_id);
      window.localStorage.setItem(PROFILE_STORAGE_KEY, session.profile_id);
      setInput("");
      setLlmConfig(sessionLlmConfig);
      setLlmDraft(cloneLlmConfig(sessionLlmConfig));

      if (session.initial_response) {
        setMessages([toAssistantMessage(session.initial_response)]);
        setResponseState(session.initial_response.status);
        setStatus(toStatusLabel(session.initial_response));
      } else {
        setMessages([]);
        setResponseState("ready");
        setStatus("準備完了");
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "新しいセッションの作成に失敗しました");
      setStatus("エラー");
    } finally {
      setLoading(false);
    }
  };

  const handleQuestionSuggestionToggle = (
    messageId: string,
    blockIndex: number,
    itemIndex: number,
    example: string
  ) => {
    setMessages((prev) =>
      prev.map((message) => {
        if (message.id !== messageId || !message.blocks) {
          return message;
        }

        const nextBlocks = message.blocks.map((block, currentBlockIndex) => {
          if (currentBlockIndex !== blockIndex || block.type !== "question") {
            return block;
          }

          const items = Array.isArray(block.content.items)
            ? (block.content.items as QuestionEntry[])
            : [];

          return {
            ...block,
            content: {
              ...block.content,
              items: items.map((item, currentItemIndex) => {
                if (currentItemIndex !== itemIndex) {
                  return item;
                }
                return {
                  ...item,
                  selected_example: item.selected_example === example ? undefined : example,
                };
              }),
            },
          };
        });

        return { ...message, blocks: nextBlocks };
      })
    );
  };

  const handleQuestionExecute = (messageId: string, blockIndex: number) => {
    const message = messages.find((item) => item.id === messageId);
    const block = message?.blocks?.[blockIndex];
    if (!block || block.type !== "question") {
      return;
    }

    const items = Array.isArray(block.content.items) ? (block.content.items as QuestionEntry[]) : [];
    const selectedAnswers = items
      .map((item) => item.selected_example?.trim())
      .filter((item): item is string => Boolean(item));

    if (selectedAnswers.length === 0) {
      return;
    }

    void submitMessage(selectedAnswers.join("、"));
  };

  const handleChecklistToggle = (messageId: string, blockIndex: number, itemIndex: number) => {
    setMessages((prev) =>
      prev.map((message) => {
        if (message.id !== messageId || !message.blocks) {
          return message;
        }

        const nextBlocks = message.blocks.map((block, currentBlockIndex) => {
          if (currentBlockIndex !== blockIndex || block.type !== "checklist") {
            return block;
          }

          const items = Array.isArray(block.content.items)
            ? (block.content.items as ChecklistEntry[])
            : [];

          return {
            ...block,
            content: {
              ...block.content,
              items: items.map((item, currentItemIndex) =>
                currentItemIndex === itemIndex ? { ...item, checked: !item.checked } : item
              ),
            },
          };
        });

        return { ...message, blocks: nextBlocks };
      })
    );
  };

  const handleCardCompareToggle = (messageId: string, blockIndex: number, itemIndex: number) => {
    setMessages((prev) =>
      prev.map((message) => {
        if (message.id !== messageId || !message.blocks) {
          return message;
        }

        const nextBlocks = message.blocks.map((block, currentBlockIndex) => {
          if (currentBlockIndex !== blockIndex || block.type !== "cards") {
            return block;
          }

          const items = Array.isArray(block.content.items) ? (block.content.items as CardEntry[]) : [];

          return {
            ...block,
            content: {
              ...block.content,
              items: items.map((item, currentItemIndex) =>
                currentItemIndex === itemIndex
                  ? { ...item, compare_selected: !item.compare_selected }
                  : item
              ),
            },
          };
        });

        return { ...message, blocks: nextBlocks };
      })
    );
  };

  const handleCompareExecute = async (messageId: string, blockIndex: number) => {
    if (!sessionId || loading) {
      return;
    }

    const message = messages.find((item) => item.id === messageId);
    const block = message?.blocks?.[blockIndex];
    if (!block || block.type !== "cards") {
      return;
    }

    const items = Array.isArray(block.content.items) ? (block.content.items as CardEntry[]) : [];
    const selectedIds = items
      .filter((item) => item.compare_selected && item.id)
      .map((item) => String(item.id));

    if (selectedIds.length < 2) {
      setError("比較するには2件以上選択してください");
      return;
    }

    setError("");
    setLoading(true);
    setStatus("比較を更新中...");

    try {
      const response = await runAction(sessionId, "compare_selected_properties", {
        property_ids: selectedIds,
      });
      appendAssistantResponse(response);
      setResponseState(response.status);
      setStatus(toStatusLabel(response));
    } catch (e) {
      setError(e instanceof Error ? e.message : "比較に失敗しました");
      setStatus("エラー");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="relative min-h-screen overflow-x-clip pb-48 text-ink">
      <div className="relative mx-auto flex min-h-screen w-full max-w-5xl flex-col px-4 pb-8 pt-6 sm:px-6">
        {/* ===== Header ===== */}
        <header className="mb-8 flex flex-wrap items-center justify-between gap-4">
          <div className="flex items-center gap-3 rounded-[24px] border border-white/85 bg-white/88 px-4 py-3 shadow-[0_10px_28px_rgba(15,23,42,0.12)] backdrop-blur-xl">
            <div className="relative flex h-12 w-12 items-center justify-center rounded-2xl bg-accent-gradient text-white shadow-floating">
              <HouseLogoIcon />
              <span className="absolute -bottom-1 -right-1 inline-flex h-5 w-5 items-center justify-center rounded-full bg-white text-accent shadow-card">
                <SparkleIcon className="h-3 w-3" />
              </span>
            </div>
            <div>
              <p className="font-display text-xl font-bold tracking-tight text-slate-950 sm:text-2xl">
                AI Housing Scientist
              </p>
              <p className="mt-0.5 text-xs font-medium text-slate-700 sm:text-sm">
                対話で計画を固めてから進める、住まい探しの深掘り調査
              </p>
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-2 text-xs">
            <span
              title={preflightSummary || undefined}
              className={`inline-flex items-center gap-2 rounded-full border bg-white/85 px-3 py-1.5 font-semibold shadow-card backdrop-blur ${statusTone}`}
            >
              <span className="relative flex h-2 w-2">
                <span className="absolute inset-0 animate-pulseSoft rounded-full bg-current opacity-60" />
                <span className="relative h-2 w-2 rounded-full bg-current" />
              </span>
              {status}
            </span>
            {pendingAction && (
              <span className="inline-flex items-center rounded-full border border-cyan-200 bg-cyan-50/85 px-3 py-1.5 text-cyan-700 shadow-card backdrop-blur">
                確認待ち
              </span>
            )}
          </div>
        </header>

        {error && (
          <p className="mb-4 rounded-2xl border border-rose-200 bg-rose-50/90 px-4 py-3 text-sm text-rose-700 shadow-card backdrop-blur">
            {error}
          </p>
        )}

        {/* ===== Main ===== */}
        <main className="flex-1 space-y-5">
          {messages.length === 0 && (
            <section className="animate-rise rounded-[28px] border border-white/80 bg-white/85 p-7 shadow-card backdrop-blur-xl sm:p-9">
              <div className="flex items-start gap-4">
                <div className="flex h-11 w-11 shrink-0 items-center justify-center rounded-2xl bg-accent-soft-gradient text-accent shadow-card">
                  <SparkleIcon className="h-5 w-5" />
                </div>
                <div className="flex-1">
                  <h2 className="font-display text-lg font-bold tracking-tight text-ink sm:text-xl">
                    住まい探しを、もっと賢く。
                  </h2>
                  <p className="mt-2 text-sm leading-relaxed text-inkMuted sm:text-[15px]">
                    まず希望条件から調査計画を作り、承認後に時間をかけて候補を集めます。
                    結果の比較、問い合わせ文の作成、契約前チェックまで会話の流れに沿って段階的に進めます。
                  </p>

                  <p className="mt-6 text-[11px] font-semibold uppercase tracking-[0.12em] text-inkSubtle">
                    試しに聞いてみる
                  </p>
                  <div className="mt-3 flex flex-wrap gap-2">
                    {SAMPLE_PROMPTS.map((prompt) => (
                      <button
                        key={prompt}
                        type="button"
                        onClick={() => handlePromptPick(prompt)}
                        className="group inline-flex max-w-full items-center gap-2 rounded-full border border-hairline bg-white/80 px-3.5 py-2 text-xs text-ink shadow-sm transition hover:-translate-y-0.5 hover:border-accent/40 hover:bg-white hover:shadow-card sm:text-[13px]"
                      >
                        <span className="text-accent transition group-hover:translate-x-0.5">
                          ›
                        </span>
                        <span className="truncate">{prompt}</span>
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            </section>
          )}

          {messages.map((message) => {
            const isUser = message.role === "user";
            return (
              <article
                key={message.id}
                className={`flex animate-rise items-start gap-3 ${isUser ? "flex-row-reverse" : ""}`}
              >
                <div
                  className={`flex h-9 w-9 shrink-0 items-center justify-center rounded-2xl text-white shadow-card ${
                    isUser ? "bg-sky-700" : "bg-accent-gradient"
                  }`}
                  aria-hidden="true"
                >
                  {isUser ? <UserAvatarIcon className="h-5 w-5" /> : <HouseLogoIcon className="h-5 w-5" />}
                </div>

                <div
                  className={`max-w-[min(880px,calc(100%-3rem))] rounded-3xl border p-4 shadow-card sm:p-5 ${
                    isUser
                      ? "rounded-tr-md border-sky-300/90 bg-[linear-gradient(135deg,rgba(255,255,255,0.96)_0%,rgba(224,242,254,0.98)_46%,rgba(186,230,253,0.98)_100%)] text-slate-950 shadow-[0_12px_32px_rgba(14,116,144,0.18)] backdrop-blur-xl"
                      : "rounded-tl-md border-white/80 bg-white/92 backdrop-blur-xl"
                  }`}
                >
                  <p className={`whitespace-pre-wrap text-[15px] leading-7 ${isUser ? "font-medium text-slate-950" : "text-ink"}`}>
                    {message.text}
                  </p>
                  {message.blocks && message.blocks.length > 0 && (
                    <div className="mt-4 space-y-3">
                      {message.blocks.map((block, idx) => (
                        <StructuredBlock
                          key={`${message.id}-${idx}`}
                          block={block}
                          disabled={loading || isResearchBusy}
                          onActionExecute={(action) => void handleActionExecute(action)}
                          onCompareExecute={() => void handleCompareExecute(message.id, idx)}
                          onCompareToggle={(itemIndex) =>
                            handleCardCompareToggle(message.id, idx, itemIndex)
                          }
                          onChecklistToggle={(itemIndex) =>
                            handleChecklistToggle(message.id, idx, itemIndex)
                          }
                          onQuestionExecute={() => handleQuestionExecute(message.id, idx)}
                          onQuestionSuggestionToggle={(itemIndex, example) =>
                            handleQuestionSuggestionToggle(message.id, idx, itemIndex, example)
                          }
                        />
                      ))}
                    </div>
                  )}
                </div>
              </article>
            );
          })}

          {loading && (
            <div className="flex animate-fadeIn items-center gap-3">
              <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-2xl bg-accent-gradient text-white shadow-card">
                <HouseLogoIcon className="h-5 w-5" />
              </div>
              <div className="inline-flex items-center gap-1.5 rounded-full border border-white/80 bg-white/90 px-4 py-2.5 shadow-card backdrop-blur">
                <span className="h-2 w-2 animate-pulseSoft rounded-full bg-accent" />
                <span className="h-2 w-2 animate-pulseSoft rounded-full bg-accent [animation-delay:120ms]" />
                <span className="h-2 w-2 animate-pulseSoft rounded-full bg-accent [animation-delay:240ms]" />
              </div>
            </div>
          )}
          <div ref={messagesEndRef} aria-hidden="true" className="h-px w-full" />
        </main>
      </div>

      {/* ===== Floating input bar ===== */}
      <form
        onSubmit={onSubmit}
        className="fixed inset-x-0 bottom-0 z-20 px-4 pb-[calc(env(safe-area-inset-bottom)+14px)] pt-6 sm:px-6"
      >
        <div className="mx-auto max-w-5xl rounded-[28px] border border-sky-100 bg-white p-3 shadow-floating sm:p-4">
          <div className="flex flex-col gap-3">
            {pendingAction && (
              <div className="flex flex-wrap items-center justify-end gap-2">
                <button
                  type="button"
                  onClick={() => void handleConfirm(false)}
                  disabled={loading}
                  className="rounded-full border border-sky-200 bg-sky-50/80 px-4 py-2 text-sm font-medium text-sky-800 transition hover:border-sky-300 hover:bg-sky-100 disabled:opacity-50"
                >
                  キャンセル
                </button>
                <button
                  type="button"
                  onClick={() => void handleConfirm(true)}
                  disabled={loading}
                  className="rounded-full border border-sky-500/20 bg-accent px-4 py-2 text-sm font-medium text-white shadow-card transition hover:bg-accentDeep hover:shadow-glow disabled:opacity-50"
                >
                  実行する
                </button>
              </div>
            )}

            <div className="overflow-visible rounded-[26px] bg-sky-50 transition">
              {llmEditorOpen && llmDraft && llmCapabilities && (
                <div className="border-b border-sky-100/90 px-4 pb-4 pt-4 sm:px-5">
                  <div className="rounded-[24px] border border-sky-100 bg-white p-4 shadow-card">
                    <div className="flex flex-wrap items-start justify-between gap-3">
                      <div>
                        <p className="text-[11px] font-semibold uppercase tracking-[0.12em] text-inkSubtle">
                          LLM設定
                        </p>
                        <p className="mt-1 text-sm font-semibold text-ink">
                          実行前に段階ごとのモデルを調整できます
                        </p>
                        <p className="mt-1 text-xs leading-6 text-inkMuted">
                          調査ジョブは開始時に設定を固定します。問い合わせ文と契約チェックは実行時の設定を使います。
                        </p>
                      </div>
                      <button
                        type="button"
                        onClick={() => setLlmEditorOpen(false)}
                        className="rounded-full border border-sky-100 bg-white px-3 py-1.5 text-xs font-medium text-ink transition hover:border-sky-300 hover:bg-sky-50"
                      >
                        閉じる
                      </button>
                    </div>

                    <div className="mt-4 space-y-3">
                      {llmCapabilities.route_definitions.map((route) => {
                        const routeConfig = llmDraft.routes[route.key];
                        return (
                          <div
                            key={route.key}
                            className="grid gap-3 rounded-2xl border border-sky-100 bg-sky-50/45 p-3 sm:grid-cols-[160px_minmax(0,1fr)]"
                          >
                            <div>
                              <p className="text-sm font-semibold text-ink">{route.label}</p>
                              <p className="mt-1 text-xs leading-5 text-inkMuted">{route.description}</p>
                            </div>

                            <div className="space-y-1">
                              <span className="text-[11px] font-semibold uppercase tracking-[0.12em] text-inkSubtle">
                                Model
                              </span>
                              <div className="relative">
                                <button
                                  type="button"
                                  disabled={!llmEditable || llmSaving}
                                  onMouseDown={(e) => {
                                    e.stopPropagation();
                                    setOpenModelDropdown(openModelDropdown === route.key ? null : route.key);
                                  }}
                                  className="flex w-full items-center justify-between gap-2 rounded-xl border border-sky-100 bg-white px-3 py-2 text-left text-sm text-ink shadow-sm transition hover:border-sky-300 hover:shadow focus:outline-none disabled:cursor-not-allowed disabled:bg-slate-50 disabled:text-slate-400"
                                >
                                  <span className="truncate">{routeConfig.model}</span>
                                  <svg
                                    className={`h-4 w-4 flex-shrink-0 text-inkSubtle transition-transform duration-150 ${openModelDropdown === route.key ? "rotate-180" : ""}`}
                                    fill="none"
                                    viewBox="0 0 24 24"
                                    stroke="currentColor"
                                    strokeWidth={2}
                                  >
                                    <path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" />
                                  </svg>
                                </button>

                                {openModelDropdown === route.key && (
                                  <div
                                    className="absolute z-50 mt-1 max-h-60 w-full overflow-y-auto rounded-xl border border-sky-100 bg-white py-1 shadow-lg"
                                    onMouseDown={(e) => e.stopPropagation()}
                                  >
                                    {llmCapabilities.models.map((item) => {
                                      const isSelected = item.model === routeConfig.model;
                                      return (
                                        <button
                                          key={`${route.key}-${item.model}`}
                                          type="button"
                                          onClick={() => {
                                            handleLlmRouteModelChange(route.key, item.model);
                                            setOpenModelDropdown(null);
                                          }}
                                          className={`flex w-full items-center gap-2 px-3 py-2 text-left text-sm transition hover:bg-sky-50 ${
                                            isSelected
                                              ? "bg-sky-50/60 font-medium text-accent"
                                              : "text-ink"
                                          }`}
                                        >
                                          <span className={`flex h-4 w-4 flex-shrink-0 items-center justify-center ${isSelected ? "text-accent" : ""}`}>
                                            {isSelected && (
                                              <svg viewBox="0 0 16 16" fill="currentColor" className="h-3.5 w-3.5">
                                                <path d="M13.78 4.22a.75.75 0 0 1 0 1.06l-7.25 7.25a.75.75 0 0 1-1.06 0L2.22 9.28a.75.75 0 0 1 1.06-1.06L6 10.94l6.72-6.72a.75.75 0 0 1 1.06 0Z" />
                                              </svg>
                                            )}
                                          </span>
                                          <span className="truncate">{item.model}</span>
                                        </button>
                                      );
                                    })}
                                  </div>
                                )}
                              </div>
                              <p className="text-xs text-inkMuted">
                                選択可能モデル {llmCapabilities.models.length} 件
                              </p>
                            </div>
                          </div>
                        );
                      })}
                    </div>

                    <div className="mt-4 flex flex-wrap items-center justify-between gap-3">
                      <p className="text-xs text-inkMuted">
                        現在のプリセット: {formatPresetLabel(llmDraft.preset)}
                        {!llmEditable ? " / 調査中は編集できません" : ""}
                      </p>
                      <div className="flex flex-wrap items-center gap-2">
                        <button
                          type="button"
                          onClick={handleResetLlmConfig}
                          disabled={!llmEditable || llmSaving}
                          className="rounded-full border border-sky-100 bg-white px-4 py-2 text-sm font-medium text-ink transition hover:border-sky-300 hover:bg-sky-50 disabled:cursor-not-allowed disabled:opacity-60"
                        >
                          デフォルトに戻す
                        </button>
                        <button
                          type="button"
                          onClick={() => void handleSaveLlmConfig()}
                          disabled={!llmEditable || llmSaving}
                          className="rounded-full border border-sky-500/20 bg-accent px-4 py-2 text-sm font-medium text-white shadow-card transition hover:bg-accentDeep hover:shadow-glow disabled:cursor-not-allowed disabled:opacity-60"
                        >
                          {llmSaving ? "保存中..." : "設定を保存"}
                        </button>
                      </div>
                    </div>
                  </div>
                </div>
              )}

              <textarea
                ref={textareaRef}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={onTextareaKeyDown}
                disabled={isResearchBusy}
                rows={1}
                placeholder={inputPlaceholder}
                className="auto-resize max-h-48 min-h-[76px] w-full resize-none bg-transparent px-5 pb-3 pt-4 text-[15px] leading-7 text-ink outline-none placeholder:text-inkSubtle disabled:cursor-not-allowed disabled:opacity-70"
              />

              <div className="flex items-center justify-between gap-3 px-4 pb-3 pt-2">
                <div className="min-w-0">
                  <button
                    type="button"
                    disabled={!llmConfig || llmSaving}
                    onClick={() => setLlmEditorOpen((prev) => !prev)}
                    className="inline-flex max-w-full items-center gap-2 rounded-full border border-sky-100 bg-white/90 px-3.5 py-2 text-sm text-ink shadow-sm transition hover:border-sky-300 hover:bg-white hover:shadow-card disabled:cursor-not-allowed disabled:opacity-60"
                  >
                    <span className="text-[11px] font-semibold uppercase tracking-[0.12em] text-inkSubtle">
                      LLM
                    </span>
                    <span className="font-medium">{llmConfig ? formatPresetLabel(llmConfig.preset) : "Loading"}</span>
                    <span className="text-slate-400">
                      <ChevronIcon open={llmEditorOpen} />
                    </span>
                  </button>
                </div>

                <div className="flex items-center gap-2">
                  <button
                    type="button"
                    onClick={() => void handleNewSession()}
                    disabled={loading}
                    aria-label="新しいセッションを作成"
                    className="inline-flex h-11 w-11 shrink-0 items-center justify-center rounded-2xl border border-sky-500/20 bg-accent text-white shadow-floating transition hover:-translate-y-0.5 hover:bg-accentDeep hover:shadow-glow disabled:cursor-not-allowed disabled:border-sky-200 disabled:bg-sky-100 disabled:text-sky-400 disabled:opacity-100 disabled:shadow-none"
                  >
                    <NewSessionIcon />
                  </button>

                  <button
                    type="submit"
                    aria-label="送信"
                    disabled={loading || isResearchBusy || !sessionId || !input.trim()}
                    className="inline-flex h-11 w-11 shrink-0 items-center justify-center rounded-2xl border border-sky-500/20 bg-accent text-white shadow-floating transition hover:-translate-y-0.5 hover:bg-accentDeep hover:shadow-glow disabled:cursor-not-allowed disabled:border-sky-200 disabled:bg-sky-100 disabled:text-sky-400 disabled:opacity-100 disabled:shadow-none"
                  >
                    <SendIcon />
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      </form>
    </div>
  );
}
