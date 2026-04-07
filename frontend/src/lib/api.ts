export type Provider = "openai" | "gemini" | "groq" | "claude";

export type ActionDescriptor = {
  action_type: string;
  label: string;
  payload?: Record<string, unknown>;
};

export type UIBlock = {
  type: "text" | "table" | "checklist" | "cards" | "warning" | "question" | "actions";
  title: string;
  content: Record<string, unknown>;
};

export type ChatMessageResponse = {
  status: string;
  assistant_message: string;
  missing_slots: string[];
  next_action: string;
  blocks: UIBlock[];
  pending_confirmation: boolean;
  pending_action: ActionDescriptor | null;
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
  providers: Record<string, { key_present: boolean; reachable: boolean; model_valid: boolean; details: string }>;
};

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

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

export async function createSession(profileId?: string): Promise<CreateSessionResponse> {
  return request<CreateSessionResponse>("/api/chat/sessions", {
    method: "POST",
    body: JSON.stringify({ profile_id: profileId }),
  });
}

export async function fetchSession(sessionId: string): Promise<SessionState> {
  return request<SessionState>(`/api/chat/sessions/${sessionId}`);
}

export async function sendMessage(
  sessionId: string,
  message: string,
  provider: Provider
): Promise<ChatMessageResponse> {
  return request<ChatMessageResponse>(`/api/chat/sessions/${sessionId}/messages`, {
    method: "POST",
    body: JSON.stringify({ message, provider }),
  });
}

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

export async function fetchPreflight(): Promise<PreflightReport> {
  return request<PreflightReport>("/api/system/preflight");
}
