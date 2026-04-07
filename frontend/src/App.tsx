import { FormEvent, KeyboardEvent, useEffect, useMemo, useRef, useState } from "react";

import StructuredBlock from "./components/StructuredBlock";
import {
  ActionDescriptor,
  ChatMessageResponse,
  Provider,
  UIBlock,
  confirmAction,
  createSession,
  fetchPreflight,
  runAction,
  sendMessage,
} from "./lib/api";

type Message = {
  id: string;
  role: "user" | "assistant";
  text: string;
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

const PROVIDERS: Provider[] = ["openai", "gemini", "groq", "claude"];

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
    blocks: payload.blocks,
    pendingAction: payload.pending_action,
  };
}

function toStatusLabel(payload: ChatMessageResponse): string {
  if (payload.status === "awaiting_user_input") {
    return "追加条件の回答待ち";
  }
  if (payload.status === "search_results_ready") {
    return "物件選択待ち";
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
  const [provider, setProvider] = useState<Provider>("openai");
  const [providerMenuOpen, setProviderMenuOpen] = useState<boolean>(false);
  const [messages, setMessages] = useState<Message[]>([]);
  const [status, setStatus] = useState<string>("初期化中...");
  const [responseState, setResponseState] = useState<string>("");
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string>("");
  const [preflightSummary, setPreflightSummary] = useState<string>("");
  const providerMenuRef = useRef<HTMLDivElement | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);
  const messagesEndRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const bootstrap = async () => {
      try {
        const [session, preflight] = await Promise.all([createSession(), fetchPreflight()]);
        setSessionId(session.session_id);

        const providerStates = Object.entries(preflight.providers)
          .map(([name, report]) => `${name}:${report.model_valid ? "OK" : "NG"}`)
          .join(" / ");

        setPreflightSummary(
          `Strict=${preflight.strict_mode ? "ON" : "OFF"} | Brave=${
            preflight.brave_reachable ? "OK" : "NG"
          } | ${providerStates}`
        );

        setStatus("準備完了");
        setResponseState("ready");
      } catch (e) {
        setError(e instanceof Error ? e.message : "初期化に失敗しました");
        setStatus("初期化失敗");
      }
    };

    void bootstrap();
  }, []);

  useEffect(() => {
    if (!providerMenuOpen) {
      return;
    }

    const handlePointerDown = (event: MouseEvent) => {
      if (providerMenuRef.current && !providerMenuRef.current.contains(event.target as Node)) {
        setProviderMenuOpen(false);
      }
    };

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setProviderMenuOpen(false);
      }
    };

    document.addEventListener("mousedown", handlePointerDown);
    document.addEventListener("keydown", handleKeyDown as unknown as EventListener);

    return () => {
      document.removeEventListener("mousedown", handlePointerDown);
      document.removeEventListener("keydown", handleKeyDown as unknown as EventListener);
    };
  }, [providerMenuOpen]);

  useEffect(() => {
    const frame = window.requestAnimationFrame(() => {
      messagesEndRef.current?.scrollIntoView({
        behavior: "smooth",
        block: "end",
      });
    });

    return () => window.cancelAnimationFrame(frame);
  }, [messages, loading]);

  const inputPlaceholder = useMemo(() => {
    if (responseState === "awaiting_contract_text") {
      return "契約書・重要事項説明・初期費用表の文面を貼り付けてください…";
    }
    return "希望条件や気になる物件、契約条項を入力してください…";
  }, [responseState]);

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
    if (pendingAction) {
      return "border-cyan-200 bg-cyan-50 text-cyan-700";
    }
    if (status === "準備完了") {
      return "border-sky-200 bg-sky-50 text-sky-700";
    }
    return "border-sky-100 bg-sky-50/80 text-sky-700";
  }, [error, loading, pendingAction, status]);

  const submitMessage = async (messageText: string) => {
    if (!sessionId || !messageText.trim() || loading) {
      return;
    }

    const userText = messageText.trim();
    setInput("");
    setError("");
    setLoading(true);
    setStatus(responseState === "awaiting_contract_text" ? "契約条項を確認中..." : "検索・比較を実行中...");

    const userMessage: Message = {
      id: crypto.randomUUID(),
      role: "user",
      text: userText,
    };
    setMessages((prev) => [...prev, userMessage]);

    try {
      const response = await sendMessage(sessionId, userText, provider);
      setMessages((prev) => [...prev, toAssistantMessage(response)]);
      setResponseState(response.status);
      setStatus(toStatusLabel(response));
    } catch (e) {
      setError(e instanceof Error ? e.message : "送信に失敗しました");
      setStatus("エラー");
    } finally {
      setLoading(false);
    }
  };

  const handleActionExecute = async (action: ActionDescriptor) => {
    if (!sessionId || loading) {
      return;
    }
    setLoading(true);
    setError("");
    setStatus("操作を実行中...");

    try {
      const response = await runAction(sessionId, action.action_type, action.payload ?? {});
      setMessages((prev) => [...prev, toAssistantMessage(response)]);
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
      setMessages((prev) => [...prev, toAssistantMessage(response)]);
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

  return (
    <div className="relative min-h-screen overflow-x-clip pb-48 text-ink">
      <div className="relative mx-auto flex min-h-screen w-full max-w-5xl flex-col px-4 pb-8 pt-6 sm:px-6">
        {/* ===== Header ===== */}
        <header className="mb-8 flex flex-wrap items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="relative flex h-12 w-12 items-center justify-center rounded-2xl bg-accent-gradient text-white shadow-floating">
              <HouseLogoIcon />
              <span className="absolute -bottom-1 -right-1 inline-flex h-5 w-5 items-center justify-center rounded-full bg-white text-accent shadow-card">
                <SparkleIcon className="h-3 w-3" />
              </span>
            </div>
            <div>
              <p className="font-display text-xl font-bold tracking-tight text-ink sm:text-2xl">
                AI Housing Scientist
              </p>
              <p className="mt-0.5 text-xs text-inkMuted sm:text-sm">
                対話で進める、住まい探しの条件整理と比較
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
                    希望条件を入れて比較し、気になる物件を選び、最後に契約書の文面を確認できます。
                    検索から問い合わせ、契約前チェックまで会話の流れに沿って段階的に進めます。
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
                      ? "rounded-tr-md border-sky-100 bg-user-bubble text-ink"
                      : "rounded-tl-md border-white/80 bg-white/92 backdrop-blur-xl"
                  }`}
                >
                  <p className="whitespace-pre-wrap text-[15px] leading-7 text-ink">
                    {message.text}
                  </p>
                  {message.blocks && message.blocks.length > 0 && (
                    <div className="mt-4 space-y-3">
                      {message.blocks.map((block, idx) => (
                        <StructuredBlock
                          key={`${message.id}-${idx}`}
                          block={block}
                          disabled={loading}
                          onActionExecute={(action) => void handleActionExecute(action)}
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
        <div className="mx-auto max-w-5xl rounded-[28px] bg-white/92 p-3 shadow-floating backdrop-blur-2xl sm:p-4">
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

            <div className="overflow-visible rounded-[26px] bg-sky-50/70 transition">
              <textarea
                ref={textareaRef}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={onTextareaKeyDown}
                rows={1}
                placeholder={inputPlaceholder}
                className="auto-resize max-h-48 min-h-[76px] w-full resize-none bg-transparent px-5 pb-3 pt-4 text-[15px] leading-7 text-ink outline-none placeholder:text-inkSubtle"
              />

              <div className="flex items-center justify-between gap-3 px-4 pb-3 pt-2">
                <div ref={providerMenuRef} className="relative">
                  <button
                    type="button"
                    onClick={() => setProviderMenuOpen((prev) => !prev)}
                    className="inline-flex items-center gap-2 rounded-full border border-sky-100 bg-white/90 px-3.5 py-2 text-sm text-ink shadow-sm transition hover:border-sky-300 hover:bg-white hover:shadow-card"
                  >
                    <span className="text-[11px] font-semibold uppercase tracking-[0.12em] text-inkSubtle">
                      Provider
                    </span>
                    <span className="font-medium capitalize">{provider}</span>
                    <span className="text-slate-400">
                      <ChevronIcon open={providerMenuOpen} />
                    </span>
                  </button>

                  {providerMenuOpen && (
                    <div className="absolute bottom-[calc(100%+12px)] left-0 z-30 min-w-[200px] overflow-hidden rounded-2xl border border-hairline bg-white/95 p-1.5 shadow-floating backdrop-blur-xl">
                      {PROVIDERS.map((item) => {
                        const selected = item === provider;
                        return (
                          <button
                            key={item}
                            type="button"
                            onClick={() => {
                              setProvider(item);
                              setProviderMenuOpen(false);
                            }}
                            className={`flex w-full items-center justify-between rounded-xl border px-3 py-2.5 text-left text-sm transition ${
                              selected
                                ? "border-sky-200 bg-sky-100 text-sky-900 shadow-sm"
                                : "border-transparent text-ink hover:border-sky-100 hover:bg-sky-50"
                            }`}
                          >
                            <span className="capitalize">{item}</span>
                            <span
                              className={`h-2 w-2 rounded-full ${
                                selected ? "bg-sky-600" : "bg-sky-200"
                              }`}
                            />
                          </button>
                        );
                      })}
                    </div>
                  )}
                </div>

                <button
                  type="submit"
                  aria-label="送信"
                  disabled={loading || !sessionId || !input.trim()}
                  className="inline-flex h-11 w-11 shrink-0 items-center justify-center rounded-2xl border border-sky-500/20 bg-accent text-white shadow-floating transition hover:-translate-y-0.5 hover:bg-accentDeep hover:shadow-glow disabled:cursor-not-allowed disabled:border-sky-200 disabled:bg-sky-100 disabled:text-sky-400 disabled:opacity-100 disabled:shadow-none"
                >
                  <SendIcon />
                </button>
              </div>
            </div>
          </div>
        </div>
      </form>
    </div>
  );
}
