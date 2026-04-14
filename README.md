# The AI Housing Scientist

チャットで希望条件を整理し、賃貸物件の探索・比較・問い合わせ文作成・契約リスク確認までを支援するPoCです。

ユーザーは自然文で条件を入力し、アプリは不足条件の質問、調査計画の提示、候補物件の比較、問い合わせ文の下書き、契約条項のチェックを段階的に進めます。バックエンドはFastAPI、フロントエンドはReact/Viteで構成されています。

## 主な機能

- 希望エリア、家賃、間取り、駅徒歩、入居時期などの条件整理
- 調査計画の生成とユーザー承認
- Brave Search APIとモック物件カタログを使った候補収集
- 物件情報の正規化、重複検出、整合性レビュー、ランキング
- 複数候補の比較表示とユーザー反応の記録
- 問い合わせ文の自動生成と送信確認フロー
- 契約文面からの更新料、短期解約違約金、解約予告、保証会社などのリスク抽出
- セッション単位のLLMルーティング設定
- 監査ログ、LLM呼び出しログ、事前接続チェック

## 技術スタック

| 領域 | 使用技術 |
| --- | --- |
| Frontend | React 19, TypeScript, Vite, Tailwind CSS |
| Backend | Python 3.11, FastAPI, Pydantic, SQLite |
| Search | Brave Search API |
| LLM | OpenAI, Gemini, Groq, Claude互換アダプタ |
| DevOps | Docker Compose, GitHub Actions |

## ディレクトリ構成

```text
.
├── backend/
│   ├── app/
│   │   ├── main.py                 # FastAPIエントリポイント
│   │   ├── orchestrator.py         # 会話・調査フローの統合
│   │   ├── orchestrator_modules/   # アクション、計画、調査、表示の分割実装
│   │   ├── research/               # 探索ツリー、評価、ジョブ管理
│   │   ├── stages/                 # 条件整理、正規化、ランキング、リスク確認など
│   │   └── services/               # Brave検索、物件カタログ、画像解決
│   └── tests/
├── frontend/
│   └── src/
│       ├── App.tsx                 # チャットUI
│       ├── components/             # 構造化レスポンス表示
│       └── lib/api.ts              # Backend APIクライアント
├── docker-compose.yml
└── README.md
```

## セットアップ

### 1. 環境変数を用意する

```bash
cp .env.example .env
```

最低限、使用するLLMプロバイダーのAPIキーとBrave Search APIキーを設定します。

```env
OPENAI_API_KEY=
GEMINI_API_KEY=
CLAUDE_API_KEY=
GROQ_API_KEY=
BRAVE_SEARCH_API=
```

すべてのLLMキーが必須ではありません。`MODEL_STRICT_MODE=false`にすると、利用可能なプロバイダーだけで起動・検証できます。

### 2. Docker Composeで起動する

```bash
docker compose up --build
```

起動後、以下にアクセスします。

- Frontend: `http://localhost:5173`
- Backend: `http://localhost:8000`
- Preflight: `http://localhost:8000/api/system/preflight`

ポートを変更する場合は、`.env`に以下を追加します。

```env
WEB_PORT=5173
API_PORT=8000
```

## ローカルで個別起動する場合

Dockerを使わずに起動する場合の手順です。

### Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
mkdir -p data
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Frontend

```bash
cd frontend
npm install
npm run dev -- --host 0.0.0.0 --port 5173
```

必要に応じて、フロントエンドのAPI接続先を指定します。

```env
VITE_API_BASE_URL=http://localhost:8000
```

## 使い方

1. `http://localhost:5173`を開きます。
2. チャット欄に希望条件を入力します。
   - 例: `江東区で家賃12万円以下、駅徒歩7分以内の1LDKを探しています`
3. 不足条件があれば、画面上の質問に回答します。
4. 調査計画を確認し、承認するとバックグラウンド調査が開始されます。
5. 候補物件の比較、問い合わせ文の生成、契約文面のリスク確認を画面上のアクションから進めます。

調査中はセッションのLLM設定がロックされます。調査が完了または失敗すると再編集できます。

## 主要API

| Method | Path | 用途 |
| --- | --- | --- |
| `GET` | `/health` | ヘルスチェック |
| `GET` | `/api/system/preflight` | APIキー、モデル、Brave接続の事前チェック |
| `GET` | `/api/system/llm-capabilities` | 利用可能なLLMモデルとルート定義の取得 |
| `POST` | `/api/chat/sessions` | チャットセッション作成 |
| `GET` | `/api/chat/sessions/{session_id}` | セッション状態と履歴取得 |
| `POST` | `/api/chat/sessions/{session_id}/messages` | ユーザーメッセージ送信 |
| `POST` | `/api/chat/sessions/{session_id}/actions` | 調査承認、比較、問い合わせ生成などのアクション実行 |
| `POST` | `/api/chat/sessions/{session_id}/actions/confirm` | 保留中アクションの承認・却下 |
| `GET` | `/api/chat/sessions/{session_id}/research` | バックグラウンド調査ジョブの状態取得 |
| `GET` | `/api/chat/sessions/{session_id}/llm-config` | セッションのLLM設定取得 |
| `PUT` | `/api/chat/sessions/{session_id}/llm-config` | セッションのLLM設定更新 |
| `GET` | `/api/audit/sessions/{session_id}` | ステージ別の監査ログ取得 |
| `GET` | `/api/audit/sessions/{session_id}/llm-calls` | LLM呼び出しログ取得 |

## LLM設定

LLMは用途別のルートで使い分けます。

| ルート | 用途 |
| --- | --- |
| `planner` | 条件抽出、追加質問、調査計画作成 |
| `research_default` | 検索クエリ展開、調査フローの補助 |
| `communication` | 問い合わせ文の下書き生成 |
| `risk_check` | 契約リスク抽出 |

主な環境変数は以下です。

| 変数 | 既定値 | 説明 |
| --- | --- | --- |
| `LLM_DEFAULT_PROVIDER` | `openai` | 既定のプロバイダー |
| `MODEL_STRICT_MODE` | `true` | モデルやキーが無効な場合に厳格に失敗させる |
| `RUN_PREFLIGHT_ON_STARTUP` | `true` | 起動時に事前チェックを実行する |
| `PREFLIGHT_FAIL_FAST` | `false` | 事前チェック失敗時に起動を止める |
| `OPENAI_MODEL` | `gpt-5.4-mini` | OpenAIの設定候補モデル |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Geminiの設定候補モデル |
| `GROQ_MODEL_PRIMARY` | `openai/gpt-oss-120b` | 全ルートの既定値として使う主要モデル |
| `GROQ_MODEL_SECONDARY` | `qwen/qwen3-32b` | Groqの予備モデル |
| `CLAUDE_MODEL` | `claude-sonnet-4-6` | Claudeの設定候補モデル |
| `DATABASE_PATH` | `data/housing_agent.db` | SQLite DBの保存先 |

## テストと品質チェック

### Backend

```bash
cd backend
pip install -r requirements.txt
pytest --cov=app --cov-report=term-missing
```

Ruffを使う場合:

```bash
ruff check app tests
ruff format --check app tests
```

### Frontend

```bash
cd frontend
npm install
npm run lint
npm run format:check
npm run build
```

CIではPython 3.11、Node.js 22を前提に、バックエンドテスト、フロントエンドビルド、lint、Docker build、依存関係監査を実行します。

## データと検索

バックエンド起動時にSQLite DBを初期化し、モック物件カタログを投入します。検索ではBrave Search APIの結果に加えて、`https://mock-housing.local/properties/...`形式のモック詳細ページを解析し、候補物件の比較に使います。

調査フローでは以下の処理を組み合わせます。

- 検索クエリの生成
- Brave検索とモックカタログ検索
- 物件詳細ページの補完
- 物件情報の正規化
- 重複候補のグルーピング
- 整合性レビュー
- 条件一致度ランキング
- 最終レポート作成

## よくあるトラブル

### 起動直後にpreflightが失敗する

`.env`のAPIキー、モデル名、Brave Search APIキーを確認してください。開発中に一部プロバイダーだけで動かす場合は、以下を設定します。

```env
MODEL_STRICT_MODE=false
PREFLIGHT_FAIL_FAST=false
```

### フロントエンドからAPIに接続できない

`VITE_API_BASE_URL`と`API_PORT`が一致しているか確認してください。Docker Composeでは既定で`http://localhost:8000`に接続します。

### 調査中にLLM設定を変更できない

調査ジョブが`queued`または`running`の間は、セッションのLLM設定をロックします。調査完了後、または失敗後に再度変更してください。

## 注意事項

- このアプリはPoCです。物件情報、契約リスク、問い合わせ文は最終判断の前に必ず一次情報で確認してください。
- モック物件カタログには架空データが含まれます。
- 本番経路でLLM生成コードを実行する設計ではありません。
- APIキーを`.env`以外に書き込んだり、Gitにコミットしたりしないでください。
