import { FormEvent, useEffect, useMemo, useRef, useState } from "react";

import StructuredBlock from "./components/StructuredBlock";
import {
  ChatMessageResponse,
  Provider,
  UIBlock,
  confirmAction,
  createSession,
  fetchPreflight,
  sendMessage,
} from "./lib/api";

type Message = {
  id: string;
  role: "user" | "assistant";
  text: string;
  blocks?: UIBlock[];
  pendingAction?: { action_type: string; label: string } | null;
};

const PROVIDERS: Provider[] = ["openai", "gemini", "groq", "claude"];

function toAssistantMessage(payload: ChatMessageResponse): Message {
  return {
    id: crypto.randomUUID(),
    role: "assistant",
    text: payload.assistant_message,
    blocks: payload.blocks,
    pendingAction: payload.pending_action,
  };
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
    <svg viewBox="0 0 20 20" fill="none" aria-hidden="true" className="h-4 w-4">
      <path
        d="M4.16675 10H15.8334"
        stroke="currentColor"
        strokeWidth="1.8"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <path
        d="M10.8333 5L15.8333 10L10.8333 15"
        stroke="currentColor"
        strokeWidth="1.8"
        strokeLinecap="round"
        strokeLinejoin="round"
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
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string>("");
  const [preflightSummary, setPreflightSummary] = useState<string>("");
  const providerMenuRef = useRef<HTMLDivElement | null>(null);

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
    document.addEventListener("keydown", handleKeyDown);

    return () => {
      document.removeEventListener("mousedown", handlePointerDown);
      document.removeEventListener("keydown", handleKeyDown);
    };
  }, [providerMenuOpen]);

  const pendingAction = useMemo(() => {
    for (let i = messages.length - 1; i >= 0; i -= 1) {
      const item = messages[i];
      if (item.role === "assistant" && item.pendingAction) {
        return item.pendingAction;
      }
    }
    return null;
  }, [messages]);

  const statusTone = useMemo(() => {
    if (error) {
      return "border-rose-200 bg-rose-50 text-rose-700";
    }
    if (loading) {
      return "border-amber-200 bg-amber-50 text-amber-700";
    }
    if (pendingAction) {
      return "border-orange-200 bg-orange-50 text-orange-700";
    }
    if (status === "準備完了") {
      return "border-emerald-200 bg-emerald-50 text-emerald-700";
    }
    return "border-slate-200 bg-slate-50 text-slate-600";
  }, [error, loading, pendingAction, status]);

  const onSubmit = async (event: FormEvent) => {
    event.preventDefault();
    if (!sessionId || !input.trim() || loading) {
      return;
    }

    const userText = input.trim();
    setInput("");
    setError("");
    setLoading(true);
    setStatus("検索・比較を実行中...");

    const userMessage: Message = {
      id: crypto.randomUUID(),
      role: "user",
      text: userText,
    };
    setMessages((prev) => [...prev, userMessage]);

    try {
      const response = await sendMessage(sessionId, userText, provider);
      setMessages((prev) => [...prev, toAssistantMessage(response)]);
      setStatus(response.pending_confirmation ? "確認待ち" : "処理完了");
    } catch (e) {
      setError(e instanceof Error ? e.message : "送信に失敗しました");
      setStatus("エラー");
    } finally {
      setLoading(false);
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
      setStatus("確認処理完了");
    } catch (e) {
      setError(e instanceof Error ? e.message : "確認処理に失敗しました");
      setStatus("エラー");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="relative min-h-screen overflow-x-clip pb-44 text-ink">
      <div className="pointer-events-none absolute inset-x-0 top-0 h-[420px] bg-[radial-gradient(68%_58%_at_0%_0%,rgba(37,99,235,0.16),transparent_72%),radial-gradient(64%_58%_at_100%_0%,rgba(20,184,166,0.12),transparent_78%)]" />

      <div className="relative mx-auto flex min-h-screen w-full max-w-5xl flex-col px-4 pb-8 pt-5 sm:px-6">
        <header className="mb-6 flex flex-wrap items-center justify-between gap-3">
          <div>
            <p className="font-display text-lg font-semibold text-ink sm:text-xl">AI Housing Scientist</p>
            <p className="text-sm text-inkMuted">条件整理と比較</p>
          </div>

          <div className="flex flex-wrap items-center gap-2 text-xs">
            <span
              title={preflightSummary || undefined}
              className={`inline-flex items-center gap-2 rounded-full border bg-white/80 px-3 py-1 font-semibold shadow-sm backdrop-blur-sm ${statusTone}`}
            >
              <span className="h-1.5 w-1.5 rounded-full bg-current" />
              {status}
            </span>
            {pendingAction && (
              <span className="inline-flex items-center rounded-full border border-orange-200 bg-white/80 px-3 py-1 text-orange-700 shadow-sm backdrop-blur-sm">
                確認待ち
              </span>
            )}
            {!sessionId && !error && (
              <span className="inline-flex items-center rounded-full border border-slate-200 bg-white/80 px-3 py-1 text-slate-500 shadow-sm backdrop-blur-sm">
                準備中
              </span>
            )}
          </div>
        </header>

        {error && (
          <p className="mb-4 rounded-2xl border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-700">
            {error}
          </p>
        )}

        <main className="flex-1 space-y-4">
          {messages.length === 0 && (
            <section className="animate-rise rounded-[26px] border border-white/70 bg-white/72 px-5 py-6 text-sm text-inkMuted shadow-soft backdrop-blur-xl">
              条件や物件情報を入力してください。
            </section>
          )}

          {messages.map((message) => {
            const isUser = message.role === "user";
            return (
              <article
                key={message.id}
                className={`animate-rise rounded-[24px] border p-4 shadow-soft sm:p-5 ${
                  isUser
                    ? "ml-auto max-w-3xl border-slate-200 bg-slate-100/92"
                    : "mr-auto max-w-4xl border-white/70 bg-white/82 backdrop-blur-lg"
                }`}
              >
                <div className="mb-3 flex items-center gap-2 text-[11px] font-medium text-inkMuted">
                  <span className={`h-2 w-2 rounded-full ${isUser ? "bg-accent" : "bg-slate-400"}`} />
                  <span>{isUser ? "入力" : "結果"}</span>
                </div>

                <p className="mb-3 whitespace-pre-wrap text-sm leading-relaxed text-ink">{message.text}</p>
                {message.blocks && message.blocks.length > 0 && (
                  <div className="space-y-3">
                    {message.blocks.map((block, idx) => (
                      <StructuredBlock key={`${message.id}-${idx}`} block={block} />
                    ))}
                  </div>
                )}
              </article>
            );
          })}
        </main>
      </div>

      <form
        onSubmit={onSubmit}
        className="fixed inset-x-0 bottom-0 z-20 px-4 pb-[calc(env(safe-area-inset-bottom)+12px)] pt-3 sm:px-6"
      >
        <div className="mx-auto max-w-5xl rounded-[26px] border border-white/70 bg-white/82 p-3 shadow-soft backdrop-blur-2xl sm:p-4">
          <div className="flex flex-col gap-3">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div ref={providerMenuRef} className="relative">
                <button
                  type="button"
                  onClick={() => setProviderMenuOpen((prev) => !prev)}
                  className="inline-flex items-center gap-2 rounded-full border border-slate-200 bg-white px-3 py-1.5 text-sm text-ink shadow-sm transition hover:border-slate-300"
                >
                  <span className="text-xs text-inkMuted">Provider</span>
                  <span className="font-medium capitalize">{provider}</span>
                  <span className="text-slate-400">
                    <ChevronIcon open={providerMenuOpen} />
                  </span>
                </button>

                {providerMenuOpen && (
                  <div className="absolute bottom-[calc(100%+10px)] left-0 z-30 min-w-[180px] overflow-hidden rounded-2xl border border-slate-200 bg-white p-1.5 shadow-soft">
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
                          className={`flex w-full items-center justify-between rounded-xl px-3 py-2 text-left text-sm transition ${
                            selected ? "bg-slate-900 text-white" : "text-ink hover:bg-slate-100"
                          }`}
                        >
                          <span className="capitalize">{item}</span>
                          <span className={`h-2 w-2 rounded-full ${selected ? "bg-white" : "bg-slate-300"}`} />
                        </button>
                      );
                    })}
                  </div>
                )}
              </div>

              {pendingAction && (
                <div className="flex flex-wrap items-center gap-2">
                  <button
                    type="button"
                    onClick={() => void handleConfirm(false)}
                    disabled={loading}
                    className="rounded-full border border-slate-300 bg-white px-4 py-2 text-sm font-medium text-slate-600 transition hover:border-slate-400 disabled:opacity-50"
                  >
                    キャンセル
                  </button>
                  <button
                    type="button"
                    onClick={() => void handleConfirm(true)}
                    disabled={loading}
                    className="rounded-full bg-warm px-4 py-2 text-sm font-medium text-white transition hover:bg-warm/90 disabled:opacity-50"
                  >
                    実行
                  </button>
                </div>
              )}
            </div>

            <div className="relative">
              <textarea
                value={input}
                onChange={(e) => setInput(e.target.value)}
                rows={2}
                placeholder="条件や物件情報、契約条項を入力してください"
                className="w-full resize-y rounded-[22px] border border-slate-200 bg-white px-4 py-3 pr-16 text-sm text-ink outline-none transition focus:border-accent focus:ring-4 focus:ring-accent/10"
              />

              <button
                type="submit"
                aria-label="送信"
                disabled={loading || !sessionId || !input.trim()}
                className="absolute bottom-3 right-3 inline-flex h-10 w-10 items-center justify-center rounded-full bg-accent text-white shadow-sm transition hover:bg-accent/90 disabled:cursor-not-allowed disabled:bg-slate-300"
              >
                <SendIcon />
              </button>
            </div>
          </div>
        </div>
      </form>
    </div>
  );
}
