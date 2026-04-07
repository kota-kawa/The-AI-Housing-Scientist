# The-AI-Housing-Scientist (PoC)

`要件定義書.md` に基づいて実装した、チャット中心の賃貸意思決定支援PoCです。

## Stack

- Backend: Python + FastAPI + SQLite
- Frontend: React + TypeScript + Tailwind CSS
- Search: Brave Search API
- Property Data: SQLite seeded mock catalog + detail-page parsing
- LLM Providers: OpenAI / Gemini / Groq / Claude (共通Adapter)

## Quick Start

1. `.env.example` を `.env` としてコピーし、各APIキーを設定
2. 起動

```bash
docker compose up --build
```

- Frontend: `http://localhost:5173`
- Backend: `http://localhost:8000`
- Preflight: `GET /api/system/preflight`

## Main Endpoints

- `POST /api/chat/sessions`
- `POST /api/chat/sessions/{session_id}/messages`
- `POST /api/chat/sessions/{session_id}/actions`
- `GET /api/chat/sessions/{session_id}`
- `POST /api/chat/sessions/{session_id}/actions/confirm`
- `GET /api/audit/sessions/{session_id}`

## Notes

- `AI-Scientist-v2` 配下は参照専用で未変更です。
- 検索は Brave の結果に加えて、SQLite に投入した架空物件カタログの詳細ページを URL 単位で解析して比較に使います。
- 本番経路でLLM生成コードは実行しません。
- 厳格モード(`MODEL_STRICT_MODE=true`)では、指定モデルIDが利用不可の場合に失敗します。
