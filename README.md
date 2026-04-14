> 日本語版は一番下にあります

# 🌟 AI Housing Scientist

![Backend](https://img.shields.io/badge/Backend-FastAPI-009688)
![Frontend](https://img.shields.io/badge/Frontend-React%20%2B%20TypeScript-61DAFB)
![Bundler](https://img.shields.io/badge/Bundler-Vite-646CFF)
![DB](https://img.shields.io/badge/DB-SQLite-003B57)
![Search](https://img.shields.io/badge/Search-Brave%20Search-FB542B)
![AI](https://img.shields.io/badge/AI-Multi--LLM-orange)
![Container](https://img.shields.io/badge/Container-Docker%20Compose-2496ED)

**AI Housing Scientist** is a chat-based web app that helps users look for rental housing more smoothly.
Instead of searching everything manually, users can talk with the assistant, organize their needs, compare options, draft inquiry messages, and review contract points in one flow.

This project was built as an end-to-end product demo with a simple conversational experience and a full-stack implementation behind it.
It was created as an internship assignment with a focus on product thinking, usability, and practical AI-assisted decision support.

## UI Preview

<p align="center">
  <img src="assets/images/ai-housing-scientist-preview.png" alt="AI Housing Scientist UI Preview" width="1100">
</p>

## 🎬 Demo Video

Click the image below to watch the demo on YouTube.

| Demo Video |
| --- |
| <a href="https://youtu.be/1nitw6KUBF0?si=Z2c0vfOylqkSezgq"><img src="assets/images/demo-video-thumbnail.jpg" alt="Demo Video" width="100%"></a> |

## ✨ What It Does

- Helps users describe what kind of home they want in natural language
- Asks follow-up questions when important details are missing
- Collects and compares housing candidates in one place
- Creates draft inquiry messages for landlords or agents
- Points out contract details that may deserve a closer look

## 🎯 What I Focused On

- Making housing search feel less overwhelming by turning it into a guided conversation
- Prioritizing usable results over just showing many results
- Supporting the next real action, not only research

## 👀 Demo Flow To Watch

1. The user enters a rough housing request in natural language
2. The app asks follow-up questions to fill in missing details
3. It proposes a research plan before starting the search
4. It gathers and compares candidates from multiple search directions
5. It moves from comparison to inquiry drafting and contract-point review

## 💡 Design Choices

- Instead of relying on one search path, the app tries multiple search directions such as strict matching, slightly relaxed matching, and nearby-area exploration
- It filters out weak or inconsistent candidates before comparison, so the final list is more practical
- The experience is designed to continue past search and help with inquiry writing and pre-contract checks

## 🛠 Challenges And How I Handled Them

- Housing information is often incomplete or inconsistent, so I added a review step before ranking candidates
- Strict conditions can easily lead to zero results, so I allowed broader search directions to recover options
- Search results often include pages that are not useful enough, so I prioritized pages with enough detail for real comparison

## 🚀 Future Improvements

- Connect to more real housing data sources and live listing APIs
- Improve candidate evaluation with better weighting and domain-specific scoring
- Add features such as saved favorites, sharing, and clearer comparison history

## 🧰 Tech Stack

- **Frontend**: React, TypeScript, Vite
- **Backend**: FastAPI, Python
- **Data**: SQLite
- **External services**: Brave Search API, LLM APIs
- **Infra**: Docker Compose

## ▶️ Quick Start

> **Prerequisite:** Docker Desktop or Docker Engine with Docker Compose

1. Create your environment file.

```bash
cp .env.example .env
```

2. Add the keys you want to use in `.env`.

```env
OPENAI_API_KEY=
GEMINI_API_KEY=
CLAUDE_API_KEY=
GROQ_API_KEY=
BRAVE_SEARCH_API=
```

3. Start the app.

```bash
docker compose up --build
```

4. Open it in your browser.

- **Frontend**: http://localhost:5173
- **Backend API**: http://localhost:8000

5. Stop the app when you are done.

```bash
docker compose down
```

## 🧪 Tests

Backend tests:

```bash
cd backend
pytest --cov=app --cov-report=term-missing
```

Frontend checks:

```bash
cd frontend
npm install
npm run lint
npm run build
```

## 📝 Note

This is a PoC project. Housing details, inquiry text, and contract checks should always be verified with original sources before making a real decision.

## 📜 License

Apache License 2.0. See `LICENSE` for details.

---

<details>
<summary>日本語版（クリックして開く）</summary>

# 🌟 AI Housing Scientist

**AI Housing Scientist** は、賃貸物件探しをチャット形式でサポートするWebアプリです。
条件整理、候補比較、問い合わせ文の下書き、契約時の注意点チェックまでを、ひとつの流れで進められるようにしています。

会話しながら進められる使いやすさと、フロントエンドからバックエンドまで一通り動くプロダクトとしての完成度を意識して作ったデモです。
インターン課題として、プロダクト設計、使いやすさ、AIを使った意思決定支援が伝わることを重視して作成しました。

## UI プレビュー

<p align="center">
  <img src="assets/images/ai-housing-scientist-preview.png" alt="AI Housing Scientist UI Preview" width="1100">
</p>

## 🎬 デモ動画

画像をクリックすると YouTube で動画を開きます。

| デモ動画 |
| --- |
| <a href="https://youtu.be/1nitw6KUBF0?si=Z2c0vfOylqkSezgq"><img src="assets/images/demo-video-thumbnail.jpg" alt="デモ動画" width="100%"></a> |

## ✨ できること

- 自然な文章で希望条件を入力できる
- 条件が足りないときは追加で質問してくれる
- 候補物件を集めて比較しやすくしてくれる
- 問い合わせ文のたたき台を作れる
- 契約前に気をつけたいポイントを確認できる

## 🎯 この課題で重視したこと

- 物件探しの複雑さを、会話で整理できる体験にすること
- 候補数の多さよりも、実際に比較しやすく使える候補を出すこと
- 調査で終わらず、次の行動につながること

## 👀 デモで見てほしい流れ

1. ユーザーがざっくりした希望条件を自然文で入力する
2. システムが追加質問をして条件を整理する
3. 調査計画を提示してから探索を始める
4. 複数の探し方で候補を集めて比較する
5. 問い合わせ文作成や契約確認までつなげる

## 💡 工夫した点

- ひとつの検索だけに頼らず、条件厳守、条件緩和、近隣拡張など複数の探索方針を試すようにしたこと
- 比較の前に、情報不足や矛盾が大きい候補を落とす流れを入れたこと
- 候補一覧を見せて終わりではなく、問い合わせ文や契約確認まで一連の体験にしたこと

## 🛠 苦労した点と対応

- 物件情報は不足や表記ゆれが多いため、順位付けの前に整合性を確認する段階を入れた
- 条件を厳しくすると候補が0件になりやすいため、少し条件を広げる探索も同時に行う設計にした
- 検索結果には比較に使いにくいページも多いため、詳細情報が十分に取れるページを優先するようにした

## 🚀 今後の改善案

- 実在の物件データや外部APIとの連携を増やす
- 候補評価の重み付けや順位付けをさらに改善する
- お気に入り保存、共有、比較履歴の見やすさを強化する

## 🧰 技術スタック

- **フロントエンド**: React, TypeScript, Vite
- **バックエンド**: FastAPI, Python
- **データ**: SQLite
- **外部サービス**: Brave Search API, 各種LLM API
- **インフラ**: Docker Compose

## ▶️ はじめ方

> **前提:** Docker Desktop または Docker Engine + Docker Compose

1. `.env` を作成します。

```bash
cp .env.example .env
```

2. `.env` に使いたいAPIキーを設定します。

```env
OPENAI_API_KEY=
GEMINI_API_KEY=
CLAUDE_API_KEY=
GROQ_API_KEY=
BRAVE_SEARCH_API=
```

3. アプリを起動します。

```bash
docker compose up --build
```

4. ブラウザで開きます。

- **フロントエンド**: http://localhost:5173
- **バックエンド API**: http://localhost:8000

5. 終了するときは停止します。

```bash
docker compose down
```

## 🧪 テスト

バックエンド:

```bash
cd backend
pytest --cov=app --cov-report=term-missing
```

フロントエンド:

```bash
cd frontend
npm install
npm run lint
npm run build
```

## 📝 補足

このアプリは PoC です。物件情報、問い合わせ文、契約内容は、実際に利用する前に必ず一次情報で確認してください。

## 📜 ライセンス

Apache License 2.0。詳細は `LICENSE` を参照してください。

</details>
