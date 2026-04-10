export type Provider = "openai" | "gemini" | "groq" | "claude";
export type LLMRouteKey = "planner" | "research_default" | "communication" | "risk_check";

export type LLMRouteConfig = {
  model: string;
};

export type LLMConfig = {
  preset: string;
  routes: Record<LLMRouteKey, LLMRouteConfig>;
};

export type SessionLLMConfig = LLMConfig & {
  session_id: string;
  editable: boolean;
  active_job_id: string | null;
};

export type LLMRouteDefinition = {
  key: LLMRouteKey;
  label: string;
  description: string;
};

export type LLMProviderCapability = {
  key_present: boolean;
  reachable: boolean;
  default_model: string;
  models: string[];
  details: string;
};

export type LLMModelOption = {
  model: string;
  provider: Provider;
  available: boolean;
};

export type LLMCapabilities = {
  route_definitions: LLMRouteDefinition[];
  providers: Record<Provider, LLMProviderCapability>;
  models: LLMModelOption[];
  default_config: LLMConfig;
};

export type ActionDescriptor = {
  action_type: string;
  label: string;
  payload?: Record<string, unknown>;
};

export type UIBlock = {
  type:
    | "text"
    | "table"
    | "checklist"
    | "cards"
    | "warning"
    | "question"
    | "actions"
    | "plan"
    | "timeline"
    | "tree"
    | "sources";
  title: string;
  content: Record<string, unknown>;
  display_label?: string;
};

export type ChatMessageResponse = {
  status: string;
  assistant_message: string;
  missing_slots: string[];
  next_action: string;
  blocks: UIBlock[];
  pending_confirmation: boolean;
  pending_action: ActionDescriptor | null;
  status_label?: string;
};

export type SessionState = {
  session_id: string;
  profile_id: string;
  status: string;
  pending_action: Record<string, unknown> | null;
  user_memory: Record<string, unknown>;
  task_memory: Record<string, unknown>;
  messages: Array<{ role: string; content: Record<string, unknown>; created_at: string }>;
};

export type CreateSessionResponse = {
  session_id: string;
  profile_id: string;
  initial_response: ChatMessageResponse | null;
};

export type PreflightReport = {
  strict_mode: boolean;
  brave_reachable: boolean;
  providers: Record<
    string,
    { key_present: boolean; reachable: boolean; model_valid: boolean; details: string }
  >;
};

export type ResearchState = {
  session_id: string;
  job_id: string | null;
  status: string;
  current_stage: string;
  progress_percent: number;
  latest_summary: string;
  response: ChatMessageResponse | null;
};

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

/**
 * 日本語: APIリクエストを共通化し、失敗時はdetail優先のエラーを投げます。
 * English: Sends a shared API request and throws a detail-first error on failures.
 */
async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });

  if (!response.ok) {
    const fallback = `${response.status} ${response.statusText}`;
    try {
      const payload = (await response.json()) as { detail?: string };
      throw new Error(payload.detail ?? fallback);
    } catch {
      throw new Error(fallback);
    }
  }

  return (await response.json()) as T;
}

/**
 * 日本語: チャットセッションを作成し、必要なら前回プロフィールを引き継ぎます。
 * English: Creates a chat session and optionally resumes with a previous profile.
 */
export async function createSession(
  profileId?: string,
  freshStart: boolean = false
): Promise<CreateSessionResponse> {
  return request<CreateSessionResponse>("/api/chat/sessions", {
    method: "POST",
    body: JSON.stringify({ profile_id: profileId, fresh_start: freshStart }),
  });
}

/**
 * 日本語: 指定セッションの状態と履歴を取得します。
 * English: Fetches state and message history for the given session.
 */
export async function fetchSession(sessionId: string): Promise<SessionState> {
  return request<SessionState>(`/api/chat/sessions/${sessionId}`);
}

/**
 * 日本語: ユーザーメッセージを送信して、最新のアシスタント応答を取得します。
 * English: Sends a user message and returns the latest assistant response.
 */
export async function sendMessage(
  sessionId: string,
  message: string,
  provider?: Provider
): Promise<ChatMessageResponse> {
  return request<ChatMessageResponse>(`/api/chat/sessions/${sessionId}/messages`, {
    method: "POST",
    body: JSON.stringify({ message, provider }),
  });
}

/**
 * 日本語: 保留中アクションの承認/却下を送信します。
 * English: Submits approval or rejection for a pending action.
 */
export async function confirmAction(
  sessionId: string,
  actionType: string,
  approved: boolean
): Promise<ChatMessageResponse> {
  return request<ChatMessageResponse>(`/api/chat/sessions/${sessionId}/actions/confirm`, {
    method: "POST",
    body: JSON.stringify({ action_type: actionType, approved }),
  });
}

/**
 * 日本語: 指定アクションを任意ペイロード付きで実行します。
 * English: Executes the specified action with an optional payload.
 */
export async function runAction(
  sessionId: string,
  actionType: string,
  payload: Record<string, unknown> = {}
): Promise<ChatMessageResponse> {
  return request<ChatMessageResponse>(`/api/chat/sessions/${sessionId}/actions`, {
    method: "POST",
    body: JSON.stringify({ action_type: actionType, payload }),
  });
}

/**
 * 日本語: システム事前チェック結果（接続性・鍵状態）を取得します。
 * English: Retrieves preflight health details such as connectivity and key status.
 */
export async function fetchPreflight(): Promise<PreflightReport> {
  return request<PreflightReport>("/api/system/preflight");
}

/**
 * 日本語: 利用可能なLLMモデルとルート定義を取得します。
 * English: Fetches available LLM models and route definitions.
 */
export async function fetchLlmCapabilities(): Promise<LLMCapabilities> {
  return request<LLMCapabilities>("/api/system/llm-capabilities");
}

/**
 * 日本語: セッション単位のLLM設定を読み込みます。
 * English: Loads session-scoped LLM configuration.
 */
export async function fetchSessionLlmConfig(sessionId: string): Promise<SessionLLMConfig> {
  return request<SessionLLMConfig>(`/api/chat/sessions/${sessionId}/llm-config`);
}

/**
 * 日本語: セッションのLLM設定を保存し、確定済み設定を返します。
 * English: Saves session LLM configuration and returns the persisted settings.
 */
export async function saveSessionLlmConfig(
  sessionId: string,
  payload: LLMConfig
): Promise<SessionLLMConfig> {
  return request<SessionLLMConfig>(`/api/chat/sessions/${sessionId}/llm-config`, {
    method: "PUT",
    body: JSON.stringify(payload),
  });
}

/**
 * 日本語: 調査ジョブの進捗と最新応答を取得します。
 * English: Retrieves research job progress and the latest response.
 */
export async function fetchResearchState(sessionId: string): Promise<ResearchState> {
  return request<ResearchState>(`/api/chat/sessions/${sessionId}/research`);
}
