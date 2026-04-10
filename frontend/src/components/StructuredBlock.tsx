import type { ComponentType, SVGProps } from "react";

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import type { ActionDescriptor, UIBlock } from "../lib/api";

type Props = {
  block: UIBlock;
  disabled?: boolean;
  onCompareExecute?: () => void;
  onCompareToggle?: (itemIndex: number) => void;
  onChecklistToggle?: (itemIndex: number) => void;
  onQuestionExecute?: () => void;
  onQuestionInputChange?: (itemIndex: number, value: string) => void;
  onQuestionSuggestionToggle?: (itemIndex: number, prompt: string) => void;
  onActionExecute?: (action: ActionDescriptor) => void;
};

type IconProps = { className?: string };

type BlockTone = {
  label: string;
  iconBg: string;
  iconColor: string;
  accentTitle: string;
  accentLabel: string;
  border: string;
  surface: string;
  badgeBg: string;
  badgeText: string;
};

const TONES: Record<UIBlock["type"], BlockTone> = {
  text: {
    label: "メモ",
    iconBg: "bg-sky-50",
    iconColor: "text-sky-600",
    accentTitle: "text-sky-700",
    accentLabel: "text-sky-500",
    border: "border-sky-100",
    surface: "bg-white",
    badgeBg: "bg-sky-50",
    badgeText: "text-sky-700",
  },
  cards: {
    label: "候補物件",
    iconBg: "bg-cyan-50",
    iconColor: "text-cyan-600",
    accentTitle: "text-cyan-700",
    accentLabel: "text-cyan-500",
    border: "border-cyan-100",
    surface: "bg-white",
    badgeBg: "bg-cyan-50",
    badgeText: "text-cyan-700",
  },
  table: {
    label: "比較表",
    iconBg: "bg-blue-50",
    iconColor: "text-blue-600",
    accentTitle: "text-blue-700",
    accentLabel: "text-blue-500",
    border: "border-blue-100",
    surface: "bg-white",
    badgeBg: "bg-blue-50",
    badgeText: "text-blue-700",
  },
  checklist: {
    label: "チェック",
    iconBg: "bg-sky-50",
    iconColor: "text-sky-600",
    accentTitle: "text-sky-700",
    accentLabel: "text-sky-500",
    border: "border-sky-100",
    surface: "bg-white",
    badgeBg: "bg-sky-50",
    badgeText: "text-sky-700",
  },
  warning: {
    label: "注意",
    iconBg: "bg-amber-100",
    iconColor: "text-amber-700",
    accentTitle: "text-amber-800",
    accentLabel: "text-amber-600",
    border: "border-amber-200",
    surface: "bg-amber-50/70",
    badgeBg: "bg-amber-100",
    badgeText: "text-amber-800",
  },
  question: {
    label: "質問",
    iconBg: "bg-emerald-100",
    iconColor: "text-emerald-700",
    accentTitle: "text-emerald-800",
    accentLabel: "text-emerald-600",
    border: "border-emerald-200",
    surface: "bg-emerald-50/60",
    badgeBg: "bg-emerald-100",
    badgeText: "text-emerald-800",
  },
  actions: {
    label: "操作",
    iconBg: "bg-violet-100",
    iconColor: "text-violet-700",
    accentTitle: "text-violet-800",
    accentLabel: "text-violet-600",
    border: "border-violet-200",
    surface: "bg-violet-50/55",
    badgeBg: "bg-violet-100",
    badgeText: "text-violet-800",
  },
  plan: {
    label: "計画",
    iconBg: "bg-indigo-100",
    iconColor: "text-indigo-700",
    accentTitle: "text-indigo-800",
    accentLabel: "text-indigo-600",
    border: "border-indigo-200",
    surface: "bg-indigo-50/55",
    badgeBg: "bg-indigo-100",
    badgeText: "text-indigo-800",
  },
  timeline: {
    label: "進捗",
    iconBg: "bg-slate-100",
    iconColor: "text-slate-700",
    accentTitle: "text-slate-800",
    accentLabel: "text-slate-600",
    border: "border-slate-200",
    surface: "bg-slate-50/65",
    badgeBg: "bg-slate-100",
    badgeText: "text-slate-800",
  },
  tree: {
    label: "探索",
    iconBg: "bg-teal-100",
    iconColor: "text-teal-700",
    accentTitle: "text-teal-800",
    accentLabel: "text-teal-600",
    border: "border-teal-200",
    surface: "bg-[linear-gradient(180deg,rgba(240,253,250,0.96)_0%,rgba(255,255,255,0.96)_100%)]",
    badgeBg: "bg-teal-100",
    badgeText: "text-teal-800",
  },
  sources: {
    label: "根拠",
    iconBg: "bg-teal-100",
    iconColor: "text-teal-700",
    accentTitle: "text-teal-800",
    accentLabel: "text-teal-600",
    border: "border-teal-200",
    surface: "bg-teal-50/60",
    badgeBg: "bg-teal-100",
    badgeText: "text-teal-800",
  },
};

/* ---------- Icons ---------- */

/**
 * 日本語: テキストブロック用アイコンを描画します。
 * English: Renders the icon for text blocks.
 */
function TextIcon({ className = "h-4 w-4" }: IconProps) {
  return (
    <svg viewBox="0 0 20 20" fill="none" aria-hidden="true" className={className}>
      <path
        d="M5 6H15M5 10H15M5 14H11"
        stroke="currentColor"
        strokeWidth="1.7"
        strokeLinecap="round"
      />
    </svg>
  );
}

/**
 * 日本語: 物件カードブロック用アイコンを描画します。
 * English: Renders the icon for property card blocks.
 */
function CardsIcon({ className = "h-4 w-4" }: IconProps) {
  return (
    <svg viewBox="0 0 20 20" fill="none" aria-hidden="true" className={className}>
      <path
        d="M3 9L10 4L17 9V16C17 16.5523 16.5523 17 16 17H13V12H7V17H4C3.44772 17 3 16.5523 3 16V9Z"
        stroke="currentColor"
        strokeWidth="1.7"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

/**
 * 日本語: テーブルブロック用アイコンを描画します。
 * English: Renders the icon for table blocks.
 */
function TableIcon({ className = "h-4 w-4" }: IconProps) {
  return (
    <svg viewBox="0 0 20 20" fill="none" aria-hidden="true" className={className}>
      <rect x="3" y="4" width="14" height="12" rx="1.6" stroke="currentColor" strokeWidth="1.7" />
      <path d="M3 8.5H17M3 12.5H17M8.5 4V16M13 4V16" stroke="currentColor" strokeWidth="1.4" />
    </svg>
  );
}

/**
 * 日本語: チェックリストブロック用アイコンを描画します。
 * English: Renders the icon for checklist blocks.
 */
function ChecklistIcon({ className = "h-4 w-4" }: IconProps) {
  return (
    <svg viewBox="0 0 20 20" fill="none" aria-hidden="true" className={className}>
      <rect x="3" y="3" width="14" height="14" rx="3" stroke="currentColor" strokeWidth="1.7" />
      <path
        d="M6.8 10.6L9 12.8L13.4 8.2"
        stroke="currentColor"
        strokeWidth="1.7"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

/**
 * 日本語: 警告ブロック用アイコンを描画します。
 * English: Renders the icon for warning blocks.
 */
function WarningIcon({ className = "h-4 w-4" }: IconProps) {
  return (
    <svg viewBox="0 0 20 20" fill="none" aria-hidden="true" className={className}>
      <path
        d="M10 3.2L18 17H2L10 3.2Z"
        stroke="currentColor"
        strokeWidth="1.7"
        strokeLinejoin="round"
      />
      <path
        d="M10 8.5V11.8M10 14.2V14.4"
        stroke="currentColor"
        strokeWidth="1.8"
        strokeLinecap="round"
      />
    </svg>
  );
}

/**
 * 日本語: 質問ブロック用アイコンを描画します。
 * English: Renders the icon for question blocks.
 */
function QuestionIcon({ className = "h-4 w-4" }: IconProps) {
  return (
    <svg viewBox="0 0 20 20" fill="none" aria-hidden="true" className={className}>
      <circle cx="10" cy="10" r="7.2" stroke="currentColor" strokeWidth="1.7" />
      <path
        d="M8.4 7.5C8.4 6.45 9.2 5.7 10.3 5.7C11.36 5.7 12.1 6.36 12.1 7.32C12.1 8.15 11.62 8.68 10.72 9.22C9.92 9.7 9.55 10.16 9.55 11.1"
        stroke="currentColor"
        strokeWidth="1.55"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <circle cx="9.95" cy="13.65" r="0.85" fill="currentColor" />
    </svg>
  );
}

/**
 * 日本語: アクションブロック用アイコンを描画します。
 * English: Renders the icon for action blocks.
 */
function ActionIcon({ className = "h-4 w-4" }: IconProps) {
  return (
    <svg viewBox="0 0 20 20" fill="none" aria-hidden="true" className={className}>
      <path
        d="M4 10H16M11 5L16 10L11 15"
        stroke="currentColor"
        strokeWidth="1.7"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

/**
 * 日本語: 計画ブロック用アイコンを描画します。
 * English: Renders the icon for plan blocks.
 */
function PlanIcon({ className = "h-4 w-4" }: IconProps) {
  return (
    <svg viewBox="0 0 20 20" fill="none" aria-hidden="true" className={className}>
      <path
        d="M5 4.5H15M5 8.5H12.5M5 12.5H11"
        stroke="currentColor"
        strokeWidth="1.7"
        strokeLinecap="round"
      />
      <rect
        x="3"
        y="2.8"
        width="14"
        height="14.4"
        rx="2.2"
        stroke="currentColor"
        strokeWidth="1.5"
      />
    </svg>
  );
}

/**
 * 日本語: タイムラインブロック用アイコンを描画します。
 * English: Renders the icon for timeline blocks.
 */
function TimelineIcon({ className = "h-4 w-4" }: IconProps) {
  return (
    <svg viewBox="0 0 20 20" fill="none" aria-hidden="true" className={className}>
      <path
        d="M6 4V16M6 6H14M6 10H14M6 14H11"
        stroke="currentColor"
        strokeWidth="1.7"
        strokeLinecap="round"
      />
      <circle cx="6" cy="4" r="1.4" fill="currentColor" />
      <circle cx="6" cy="10" r="1.4" fill="currentColor" />
      <circle cx="6" cy="16" r="1.4" fill="currentColor" />
    </svg>
  );
}

/**
 * 日本語: 探索ツリーブロック用アイコンを描画します。
 * English: Renders the icon for tree exploration blocks.
 */
function TreeIcon({ className = "h-4 w-4" }: IconProps) {
  return (
    <svg viewBox="0 0 20 20" fill="none" aria-hidden="true" className={className}>
      <path
        d="M10 4.5V15.5M10 7.2H5.2M10 10H14.8M10 13H6.8"
        stroke="currentColor"
        strokeWidth="1.7"
        strokeLinecap="round"
      />
      <circle cx="10" cy="4.5" r="1.45" fill="currentColor" />
      <circle cx="5.2" cy="7.2" r="1.35" fill="currentColor" />
      <circle cx="14.8" cy="10" r="1.35" fill="currentColor" />
      <circle cx="6.8" cy="13" r="1.35" fill="currentColor" />
      <circle cx="10" cy="15.5" r="1.45" fill="currentColor" />
    </svg>
  );
}

/**
 * 日本語: 参照ソースブロック用アイコンを描画します。
 * English: Renders the icon for evidence/source blocks.
 */
function SourceIcon({ className = "h-4 w-4" }: IconProps) {
  return (
    <svg viewBox="0 0 20 20" fill="none" aria-hidden="true" className={className}>
      <path
        d="M8 12L12 8M7 6.5H5.8C4.25 6.5 3 7.75 3 9.3V14.2C3 15.75 4.25 17 5.8 17H10.7C12.25 17 13.5 15.75 13.5 14.2V13"
        stroke="currentColor"
        strokeWidth="1.6"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <path
        d="M10.5 3H17V9.5M17 3L9.2 10.8"
        stroke="currentColor"
        strokeWidth="1.6"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

/**
 * 日本語: チェック済み状態を示すマークアイコンを描画します。
 * English: Renders the checkmark icon for completed states.
 */
function CheckmarkIcon({ className = "h-3.5 w-3.5" }: IconProps) {
  return (
    <svg viewBox="0 0 20 20" fill="none" aria-hidden="true" className={className}>
      <path
        d="M5 10.5L8.5 14L15 7"
        stroke="currentColor"
        strokeWidth="2.2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

/**
 * 日本語: 徒歩情報を示す人物アイコンを描画します。
 * English: Renders the walking icon used for station distance.
 */
function WalkIcon({ className = "h-3.5 w-3.5" }: IconProps) {
  return (
    <svg viewBox="0 0 20 20" fill="none" aria-hidden="true" className={className}>
      <circle cx="12" cy="4.4" r="1.5" stroke="currentColor" strokeWidth="1.4" />
      <path
        d="M11 7.5L9 11L11 13.5L10 17.5M11 7.5L13 9.7L15.5 10.3M11 7.5L8 9.6L6 13"
        stroke="currentColor"
        strokeWidth="1.4"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

/**
 * 日本語: スコア表示用の星アイコンを描画します。
 * English: Renders the star icon used for score badges.
 */
function StarIcon({ className = "h-3.5 w-3.5" }: IconProps) {
  return (
    <svg viewBox="0 0 20 20" fill="currentColor" aria-hidden="true" className={className}>
      <path d="M10 2.2L12.36 7.06L17.6 7.84L13.78 11.62L14.72 16.96L10 14.4L5.28 16.96L6.22 11.62L2.4 7.84L7.64 7.06L10 2.2Z" />
    </svg>
  );
}

/**
 * 日本語: ポジティブ理由を示すプラス円アイコンを描画します。
 * English: Renders the plus-circle icon for positive reasons.
 */
function PlusCircleIcon({ className = "h-3.5 w-3.5" }: IconProps) {
  return (
    <svg viewBox="0 0 20 20" fill="none" aria-hidden="true" className={className}>
      <circle cx="10" cy="10" r="7.6" stroke="currentColor" strokeWidth="1.6" />
      <path d="M10 7V13M7 10H13" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" />
    </svg>
  );
}

/**
 * 日本語: ネガティブ理由を示すマイナス円アイコンを描画します。
 * English: Renders the minus-circle icon for negative reasons.
 */
function MinusCircleIcon({ className = "h-3.5 w-3.5" }: IconProps) {
  return (
    <svg viewBox="0 0 20 20" fill="none" aria-hidden="true" className={className}>
      <circle cx="10" cy="10" r="7.6" stroke="currentColor" strokeWidth="1.6" />
      <path d="M7 10H13" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" />
    </svg>
  );
}

const TYPE_ICONS: Record<UIBlock["type"], ComponentType<IconProps & SVGProps<SVGSVGElement>>> = {
  text: TextIcon,
  cards: CardsIcon,
  table: TableIcon,
  checklist: ChecklistIcon,
  warning: WarningIcon,
  question: QuestionIcon,
  actions: ActionIcon,
  plan: PlanIcon,
  timeline: TimelineIcon,
  tree: TreeIcon,
  sources: SourceIcon,
};

/* ---------- Helpers ---------- */

type TreeNodeItem = {
  id: number;
  parent_id: number | null;
  branch_id: string;
  kind: string;
  status: string;
  node_type: string;
  label: string;
  description: string;
  summary: string;
  depth: number;
  query_count: number;
  queries: string[];
  strategy_tags: string[];
  branch_score: number | null;
  frontier_score: number | null;
  detail_coverage: number | null;
  structured_ratio: number | null;
  selected: boolean;
  prune_reasons: string[];
  created_at: string;
  parent_label?: string;
  is_selected?: boolean;
  is_on_selected_path?: boolean;
};

type TreeStats = {
  executed_node_count: number;
  failed_node_count: number;
  pruned_node_count: number;
  frontier_remaining: number;
  running_node_count: number;
  max_depth_reached: number;
  termination_reason: string;
  termination_label: string;
  node_count: number;
};

type TreeFocusBranch = {
  branch_id: string;
  label: string;
  depth: number;
  branch_score: number | null;
  detail_coverage: number | null;
  frontier_score: number | null;
  summary: string;
};

type TreePruneSummary = {
  count: number;
  labels: string[];
  reasons: string[];
};

type TreeLayoutNode = TreeNodeItem & {
  x: number;
  y: number;
  radius: number;
};

type TreeLayoutEdge = {
  key: string;
  path: string;
  highlighted: boolean;
  running: boolean;
  queued: boolean;
  failed: boolean;
};

/**
 * 日本語: 任意値を画面表示用の文字列へ正規化します。
 * English: Normalizes unknown values into display-safe strings.
 */
function toDisplayText(value: unknown): string {
  if (value === null || value === undefined) {
    return "";
  }
  return String(value);
}

/**
 * 日本語: 任意値を有限の数値へ変換し、失敗時はnullを返します。
 * English: Converts unknown values to finite numbers, or null if invalid.
 */
function toDisplayNumber(value: unknown): number | null {
  const num = Number(value);
  return Number.isFinite(num) ? num : null;
}

/**
 * 日本語: 任意値をレコード配列へ安全に変換します。
 * English: Safely coerces unknown values into a record array.
 */
function toRecordArray(value: unknown): Array<Record<string, unknown>> {
  return Array.isArray(value) ? (value as Array<Record<string, unknown>>) : [];
}

/**
 * 日本語: 任意配列を空文字除外済みの文字列配列へ変換します。
 * English: Converts an unknown array into a filtered string array.
 */
function toStringArray(value: unknown): string[] {
  return Array.isArray(value) ? value.map((item) => toDisplayText(item)).filter(Boolean) : [];
}

/**
 * 日本語: 0-1スケールの値を百分率ラベルへ整形します。
 * English: Formats a 0-1 value into a percent label.
 */
function toPercentLabel(value: number | null, digits: number = 0): string {
  if (value === null) {
    return "";
  }
  return `${(value * 100).toFixed(digits)}%`;
}

/**
 * 日本語: 質問ラベルに応じた入力補助プレースホルダーを返します。
 * English: Returns a contextual placeholder for question input fields.
 */
function getQuestionInputPlaceholder(label: string): string {
  if (label.includes("徒歩")) {
    return "例: 徒歩12分まで、バス便でも可";
  }
  if (label.includes("エリア") || label.includes("駅")) {
    return "例: 中野駅周辺、東横線沿線、勤務先まで30分圏内";
  }
  if (label.includes("家賃")) {
    return "例: 管理費込みで13万円以内";
  }
  return "選択肢にない条件があれば自由に入力";
}

/**
 * 日本語: 数値配列の平均値を返し、空配列は0にします。
 * English: Returns the arithmetic mean, defaulting to 0 for empty arrays.
 */
function average(values: number[]): number {
  if (values.length === 0) {
    return 0;
  }
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

/**
 * 日本語: 文字列から再現可能な擬似ハッシュ値を生成します。
 * English: Generates a deterministic hash-like number from text.
 */
function hashText(value: string): number {
  let hash = 0;
  for (let index = 0; index < value.length; index += 1) {
    hash = (hash * 31 + value.charCodeAt(index)) >>> 0;
  }
  return hash;
}

/**
 * 日本語: SVG内に埋め込む文字列をエスケープします。
 * English: Escapes text for safe SVG embedding.
 */
function escapeSvgText(value: string): string {
  return value
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&apos;");
}

/**
 * 日本語: プレースホルダー画像用にテキストを短い行へ分割します。
 * English: Splits text into short lines for placeholder image captions.
 */
function splitTextForPlaceholder(value: string, maxLength: number = 14): string[] {
  const normalized = value.trim();
  if (!normalized) {
    return ["Mock Housing"];
  }
  const chunks: string[] = [];
  for (let index = 0; index < normalized.length; index += maxLength) {
    chunks.push(normalized.slice(index, index + maxLength));
  }
  return chunks.slice(0, 2);
}

/**
 * 日本語: 物件情報からフォールバック用SVG画像データURLを生成します。
 * English: Builds a fallback SVG data URL from property metadata.
 */
function buildPropertyPlaceholderImage({
  title,
  layout,
  station,
}: {
  title: string;
  layout: string;
  station: string;
}): string {
  const seed = hashText(`${title}|${layout}|${station}`);
  const hue = seed % 360;
  const accent = (hue + 42) % 360;
  const titleLines = splitTextForPlaceholder(title);
  const subtitle = [layout, station].filter(Boolean).join(" / ") || "おすすめ物件";
  const subtitleLabel = escapeSvgText(subtitle);
  const titleMarkup = titleLines
    .map((line, index) => {
      const dy = index === 0 ? "0" : "28";
      return `<tspan x="56" dy="${dy}">${escapeSvgText(line)}</tspan>`;
    })
    .join("");
  const svg = `
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 720" role="img" aria-label="${escapeSvgText(title)}">
      <defs>
        <linearGradient id="bg" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stop-color="hsl(${hue} 74% 92%)" />
          <stop offset="100%" stop-color="hsl(${accent} 78% 82%)" />
        </linearGradient>
        <linearGradient id="tower" x1="0%" y1="0%" x2="0%" y2="100%">
          <stop offset="0%" stop-color="hsl(${hue} 34% 34%)" />
          <stop offset="100%" stop-color="hsl(${accent} 38% 20%)" />
        </linearGradient>
      </defs>
      <rect width="1200" height="720" fill="url(#bg)" />
      <circle cx="1060" cy="124" r="72" fill="rgba(255,255,255,0.44)" />
      <path d="M0 580C132 534 220 552 332 590C458 632 554 644 688 600C808 562 914 530 1200 590V720H0Z" fill="rgba(255,255,255,0.36)" />
      <rect x="112" y="220" width="240" height="328" rx="18" fill="url(#tower)" opacity="0.9" />
      <rect x="308" y="168" width="292" height="384" rx="22" fill="url(#tower)" />
      <rect x="548" y="248" width="188" height="300" rx="18" fill="url(#tower)" opacity="0.92" />
      <rect x="704" y="296" width="150" height="252" rx="16" fill="url(#tower)" opacity="0.82" />
      <g fill="rgba(255,255,255,0.76)">
        <rect x="170" y="270" width="32" height="38" rx="6" />
        <rect x="222" y="270" width="32" height="38" rx="6" />
        <rect x="170" y="332" width="32" height="38" rx="6" />
        <rect x="222" y="332" width="32" height="38" rx="6" />
        <rect x="366" y="226" width="34" height="40" rx="6" />
        <rect x="420" y="226" width="34" height="40" rx="6" />
        <rect x="474" y="226" width="34" height="40" rx="6" />
        <rect x="366" y="288" width="34" height="40" rx="6" />
        <rect x="420" y="288" width="34" height="40" rx="6" />
        <rect x="474" y="288" width="34" height="40" rx="6" />
        <rect x="604" y="294" width="30" height="34" rx="6" />
        <rect x="652" y="294" width="30" height="34" rx="6" />
        <rect x="604" y="348" width="30" height="34" rx="6" />
        <rect x="652" y="348" width="30" height="34" rx="6" />
        <rect x="750" y="340" width="28" height="32" rx="6" />
        <rect x="796" y="340" width="28" height="32" rx="6" />
      </g>
      <rect x="56" y="64" width="560" height="154" rx="28" fill="rgba(255,255,255,0.74)" />
      <text x="56" y="132" fill="#0f172a" font-family="'Hiragino Sans','Noto Sans JP',sans-serif" font-size="52" font-weight="700">${titleMarkup}</text>
      <text x="56" y="190" fill="#334155" font-family="'Hiragino Sans','Noto Sans JP',sans-serif" font-size="28" font-weight="500">${subtitleLabel}</text>
    </svg>
  `.trim();
  return `data:image/svg+xml;charset=UTF-8,${encodeURIComponent(svg)}`;
}

/**
 * 日本語: 物件画像URLを返し、未設定なら自動生成プレースホルダーを返します。
 * English: Resolves a property image URL or falls back to generated placeholder art.
 */
function resolvePropertyImage(item: Record<string, unknown>): string {
  const imageUrl = toDisplayText(item.image_url).trim();
  if (imageUrl) {
    return imageUrl;
  }
  return buildPropertyPlaceholderImage({
    title: toDisplayText(item.title) || "候補物件",
    layout: toDisplayText(item.layout),
    station: toDisplayText(item.station),
  });
}

/**
 * 日本語: Markdown本文を統一スタイルで描画します。
 * English: Renders markdown content with a consistent component style set.
 */
function MarkdownBody({ body }: { body: string }) {
  return (
    <div className="markdown-body text-[14px] leading-7 text-ink">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          h1: ({ children }) => (
            <h1 className="mt-1 text-xl font-semibold text-slate-950 first:mt-0">{children}</h1>
          ),
          h2: ({ children }) => (
            <h2 className="mt-6 border-b border-slate-200 pb-2 text-lg font-semibold text-slate-900 first:mt-0">
              {children}
            </h2>
          ),
          h3: ({ children }) => (
            <h3 className="mt-5 text-base font-semibold text-slate-900">{children}</h3>
          ),
          p: ({ children }) => (
            <p className="mt-3 text-[14px] leading-7 text-ink first:mt-0">{children}</p>
          ),
          ul: ({ children }) => (
            <ul className="mt-3 list-disc space-y-1 pl-5 text-[14px] text-ink">{children}</ul>
          ),
          ol: ({ children }) => (
            <ol className="mt-3 list-decimal space-y-1 pl-5 text-[14px] text-ink">{children}</ol>
          ),
          li: ({ children }) => <li className="leading-7">{children}</li>,
          strong: ({ children }) => (
            <strong className="font-semibold text-slate-950">{children}</strong>
          ),
          blockquote: ({ children }) => (
            <blockquote className="mt-4 rounded-r-2xl border-l-4 border-sky-300 bg-sky-50/70 px-4 py-3 text-slate-700">
              {children}
            </blockquote>
          ),
          code: ({ children }) =>
            String(children).includes("\n") ? (
              <code className="block text-[13px] leading-6 text-slate-50">{children}</code>
            ) : (
              <code className="rounded bg-slate-100 px-1.5 py-0.5 text-[13px] text-slate-800">
                {children}
              </code>
            ),
          pre: ({ children }) => <pre className="mt-4 overflow-x-auto">{children}</pre>,
          hr: () => <hr className="my-6 border-slate-200" />,
          table: ({ children }) => (
            <div className="mt-4 overflow-x-auto rounded-2xl border border-slate-200">
              <table className="min-w-full border-collapse text-left text-[13px]">{children}</table>
            </div>
          ),
          thead: ({ children }) => <thead className="bg-slate-50 text-slate-700">{children}</thead>,
          th: ({ children }) => (
            <th className="border-b border-slate-200 px-3 py-2 font-semibold">{children}</th>
          ),
          td: ({ children }) => (
            <td className="border-b border-slate-100 px-3 py-2 align-top text-slate-700">
              {children}
            </td>
          ),
        }}
      >
        {body}
      </ReactMarkdown>
    </div>
  );
}

/**
 * 日本語: 剪定ノードを親ノード単位で集約し、要約情報を作成します。
 * English: Aggregates pruned nodes by parent and builds summary metadata.
 */
function buildTreePruneSummary(nodes: TreeNodeItem[]): Map<number, TreePruneSummary> {
  const summaryByParent = new Map<number, TreePruneSummary>();
  for (const node of nodes) {
    if (node.kind !== "pruned" || node.parent_id === null) {
      continue;
    }
    const summary = summaryByParent.get(node.parent_id) ?? { count: 0, labels: [], reasons: [] };
    summary.count += 1;
    const label = node.label || node.branch_id;
    if (label && !summary.labels.includes(label)) {
      summary.labels.push(label);
    }
    for (const reason of node.prune_reasons) {
      if (reason && !summary.reasons.includes(reason)) {
        summary.reasons.push(reason);
      }
    }
    summaryByParent.set(node.parent_id, summary);
  }
  return summaryByParent;
}

/* ---------- Section header ---------- */

/**
 * 日本語: ブロック種別に応じた見出しと件数バッジを描画します。
 * English: Renders a block header with tone, icon, and optional count badge.
 */
function SectionHeader({ block, count }: { block: UIBlock; count?: number }) {
  const tone = TONES[block.type];
  const Icon = TYPE_ICONS[block.type];

  return (
    <header className={`flex items-center gap-3 border-b px-4 py-3 ${tone.border}`}>
      <span
        className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-xl ${tone.iconBg} ${tone.iconColor}`}
      >
        <Icon className="h-4 w-4" />
      </span>
      <div className="min-w-0 flex-1">
        <p className={`text-[10px] font-semibold uppercase tracking-[0.14em] ${tone.accentLabel}`}>
          {block.display_label || tone.label}
        </p>
        <h3 className={`truncate text-sm font-semibold ${tone.accentTitle}`}>{block.title}</h3>
      </div>
      {count !== undefined && count > 0 && (
        <span
          className={`shrink-0 rounded-full px-2 py-0.5 text-[11px] font-semibold ${tone.badgeBg} ${tone.badgeText}`}
        >
          {count}
        </span>
      )}
    </header>
  );
}

/* ---------- Property card ---------- */

/**
 * 日本語: 単一物件カードを描画し、比較/アクション操作を受け付けます。
 * English: Renders a property card with compare and action controls.
 */
function PropertyCard({
  item,
  index,
  disabled = false,
  compareEnabled = true,
  onCompareToggle,
  onActionExecute,
}: {
  item: Record<string, unknown>;
  index: number;
  disabled?: boolean;
  compareEnabled?: boolean;
  onCompareToggle?: () => void;
  onActionExecute?: (action: ActionDescriptor) => void;
}) {
  const rent = toDisplayNumber(item.rent);
  const score = toDisplayText(item.score);
  const title = toDisplayText(item.title) || `候補物件 ${index + 1}`;
  const walk = toDisplayText(item.station_walk_min);
  const station = toDisplayText(item.station);
  const address = toDisplayText(item.address);
  const layout = toDisplayText(item.layout);
  const area = toDisplayText(item.area);
  const whySelected = toDisplayText(item.why_selected);
  const whyNotSelected = toDisplayText(item.why_not_selected);
  const action = (item.action as ActionDescriptor | undefined) ?? undefined;
  const secondaryActions = Array.isArray(item.secondary_actions)
    ? (item.secondary_actions as ActionDescriptor[])
    : [];
  const featureTags = Array.isArray(item.feature_tags)
    ? item.feature_tags.map((tag) => toDisplayText(tag)).filter(Boolean)
    : [];
  const compareSelected = Boolean(item.compare_selected);
  const reactionState = toDisplayText(item.reaction_state);
  const imageUrl = resolvePropertyImage(item);
  const reactionLabel =
    reactionState === "favorite" ? "気になる" : reactionState === "exclude" ? "除外済み" : "";
  const reactionTone =
    reactionState === "favorite"
      ? "bg-amber-100 text-amber-800"
      : reactionState === "exclude"
        ? "bg-slate-200 text-slate-700"
        : "";

  return (
    <article className="group relative overflow-hidden rounded-2xl border border-hairline bg-white p-4 shadow-card transition hover:-translate-y-0.5 hover:border-sky-200 hover:shadow-cardHover">
      <div className="relative mb-4 overflow-hidden rounded-2xl border border-slate-100 bg-slate-100">
        <img
          src={imageUrl}
          alt={`${title} の物件画像`}
          loading="lazy"
          className="h-44 w-full object-cover transition duration-500 group-hover:scale-[1.02]"
        />
        <div className="pointer-events-none absolute inset-x-0 bottom-0 h-24 bg-gradient-to-t from-slate-950/55 to-transparent" />
        <div className="absolute bottom-3 left-3 right-3 flex items-end justify-between gap-3">
          <div className="min-w-0">
            <p className="truncate text-[11px] font-medium tracking-[0.12em] text-white/80">
              RESULT PROPERTY
            </p>
            <p className="truncate text-sm font-semibold text-white">{title}</p>
          </div>
          {score && (
            <span className="inline-flex items-center gap-1 rounded-full bg-white/88 px-2 py-1 text-[11px] font-semibold text-sky-800 shadow-sm">
              <StarIcon className="h-3 w-3" />
              {score}
            </span>
          )}
        </div>
      </div>

      <header className="mb-3 flex items-start justify-between gap-3">
        <div className="min-w-0 flex-1">
          <h4 className="font-display text-[13px] font-semibold leading-snug text-ink line-clamp-2">
            {title}
          </h4>
          <p className="mt-1.5 flex items-baseline gap-1 font-display">
            <span className="text-xl font-bold tracking-tight text-ink">
              {rent !== null ? rent.toLocaleString() : "—"}
            </span>
            <span className="text-[11px] font-medium text-inkMuted">円 / 月</span>
          </p>
        </div>
        <div className="flex shrink-0 flex-col items-end gap-1.5">
          {reactionLabel && (
            <span
              className={`inline-flex rounded-full px-2 py-1 text-[11px] font-semibold ${reactionTone}`}
            >
              {reactionLabel}
            </span>
          )}
        </div>
      </header>

      <dl className="grid grid-cols-2 gap-x-3 gap-y-1.5 border-t border-hairline pt-3 text-[11px] text-inkMuted">
        <div className="flex items-center gap-1.5">
          <WalkIcon className="h-3.5 w-3.5 text-inkSubtle" />
          <span>徒歩 {walk || "—"} 分</span>
        </div>
        {station && (
          <div className="flex items-center gap-1.5">
            <span className="inline-flex h-3.5 w-3.5 items-center justify-center text-inkSubtle">
              <svg viewBox="0 0 20 20" fill="none" aria-hidden="true" className="h-3.5 w-3.5">
                <path
                  d="M10 17C13 13.5 15.5 11 15.5 8.2C15.5 5.05 13.04 2.6 10 2.6C6.96 2.6 4.5 5.05 4.5 8.2C4.5 11 7 13.5 10 17Z"
                  stroke="currentColor"
                  strokeWidth="1.4"
                  strokeLinejoin="round"
                />
                <circle cx="10" cy="8.2" r="1.7" stroke="currentColor" strokeWidth="1.4" />
              </svg>
            </span>
            <span className="truncate">{station}</span>
          </div>
        )}
        {layout && (
          <div className="flex items-center gap-1.5">
            <span className="inline-flex h-3.5 w-3.5 items-center justify-center text-inkSubtle">
              <svg viewBox="0 0 20 20" fill="none" aria-hidden="true" className="h-3.5 w-3.5">
                <rect
                  x="3"
                  y="3"
                  width="14"
                  height="14"
                  rx="1.5"
                  stroke="currentColor"
                  strokeWidth="1.4"
                />
                <path d="M3 11H12M12 3V17" stroke="currentColor" strokeWidth="1.4" />
              </svg>
            </span>
            <span>{layout}</span>
          </div>
        )}
        {area && (
          <div className="flex items-center gap-1.5">
            <span className="inline-flex h-3.5 w-3.5 items-center justify-center text-inkSubtle">
              <svg viewBox="0 0 20 20" fill="none" aria-hidden="true" className="h-3.5 w-3.5">
                <path
                  d="M3 13V17H7M17 7V3H13M3 7V3H7M17 13V17H13"
                  stroke="currentColor"
                  strokeWidth="1.5"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              </svg>
            </span>
            <span>{area}</span>
          </div>
        )}
      </dl>

      {address && <p className="mt-3 text-[11px] leading-5 text-inkMuted">{address}</p>}

      {featureTags.length > 0 && (
        <div className="mt-3 flex flex-wrap gap-1.5">
          {featureTags.map((tag) => (
            <span
              key={`${title}-${tag}`}
              className="rounded-full bg-sky-50 px-2 py-1 text-[11px] font-medium text-sky-700"
            >
              {tag}
            </span>
          ))}
        </div>
      )}

      {(whySelected || whyNotSelected) && (
        <div className="mt-3 space-y-1.5">
          {whySelected && (
            <p className="flex items-start gap-1.5 rounded-lg bg-sky-50 px-2.5 py-1.5 text-[11px] leading-relaxed text-sky-800">
              <PlusCircleIcon className="mt-0.5 h-3.5 w-3.5 shrink-0 text-sky-600" />
              <span>{whySelected}</span>
            </p>
          )}
          {whyNotSelected && (
            <p className="flex items-start gap-1.5 rounded-lg bg-cyan-50 px-2.5 py-1.5 text-[11px] leading-relaxed text-cyan-800">
              <MinusCircleIcon className="mt-0.5 h-3.5 w-3.5 shrink-0 text-cyan-600" />
              <span>{whyNotSelected}</span>
            </p>
          )}
        </div>
      )}

      <div className="mt-4 space-y-2">
        {compareEnabled && (
          <button
            type="button"
            disabled={disabled}
            onClick={onCompareToggle}
            className={`inline-flex w-full items-center justify-center rounded-xl border px-3 py-2 text-sm font-medium transition ${
              compareSelected
                ? "border-sky-700 bg-sky-700 text-white"
                : "border-sky-200 bg-sky-50 text-sky-800 hover:border-sky-300 hover:bg-sky-100"
            } disabled:cursor-not-allowed disabled:opacity-60`}
          >
            {compareSelected ? "比較対象から外す" : "比較に追加する"}
          </button>
        )}

        {action && (
          <button
            type="button"
            disabled={disabled}
            onClick={() => onActionExecute?.(action)}
            className="inline-flex w-full items-center justify-center rounded-xl border border-sky-500/20 bg-accent px-3 py-2 text-sm font-medium text-white shadow-sm transition hover:bg-accentDeep disabled:cursor-not-allowed disabled:border-sky-200 disabled:bg-sky-100 disabled:text-sky-400"
          >
            {action.label}
          </button>
        )}

        {secondaryActions.length > 0 && (
          <div className="grid grid-cols-2 gap-2">
            {secondaryActions.map((secondaryAction) => (
              <button
                key={`${title}-${secondaryAction.label}`}
                type="button"
                disabled={disabled}
                onClick={() => onActionExecute?.(secondaryAction)}
                className="inline-flex items-center justify-center rounded-xl border border-slate-200 bg-white px-3 py-2 text-xs font-medium text-slate-700 transition hover:border-slate-300 hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-60"
              >
                {secondaryAction.label}
              </button>
            ))}
          </div>
        )}
      </div>
    </article>
  );
}

/**
 * 日本語: ツリーノード状態に対応する配色と表示ラベルを返します。
 * English: Maps tree node status to visual tokens and display label.
 */
function treeStatusMeta(status: string) {
  if (status === "completed") {
    return {
      label: "完了",
      fill: "#10b981",
      stroke: "#047857",
      edge: "#6ee7b7",
      halo: "#a7f3d0",
    };
  }
  if (status === "running") {
    return {
      label: "実行中",
      fill: "#38bdf8",
      stroke: "#0369a1",
      edge: "#7dd3fc",
      halo: "#bae6fd",
    };
  }
  if (status === "failed") {
    return {
      label: "失敗",
      fill: "#fda4af",
      stroke: "#be123c",
      edge: "#fda4af",
      halo: "#ffe4e6",
    };
  }
  if (status === "pruned") {
    return {
      label: "剪定",
      fill: "#ffffff",
      stroke: "#c2410c",
      edge: "#fdba74",
      halo: "#ffedd5",
    };
  }
  if (status === "queued") {
    return {
      label: "待機",
      fill: "#e0f2fe",
      stroke: "#0284c7",
      edge: "#7dd3fc",
      halo: "#dbeafe",
    };
  }
  return {
    label: status || "pending",
    fill: "#ffffff",
    stroke: "#94a3b8",
    edge: "#cbd5e1",
    halo: "#e2e8f0",
  };
}

/**
 * 日本語: ノード種別・状態・スコアに応じて描画半径を決定します。
 * English: Computes node radius from kind, status, and score context.
 */
function treeNodeRadius(node: TreeNodeItem): number {
  if (node.kind === "root") {
    return 30;
  }
  if (node.kind === "pruned") {
    return 18;
  }
  let radius = 20;
  if (node.status === "running") {
    radius += 3;
  }
  if (node.is_selected) {
    radius += 5;
  } else if (node.is_on_selected_path) {
    radius += 2;
  }
  const score = node.branch_score ?? node.frontier_score ?? 0;
  if (score > 0) {
    radius += Math.max(0, Math.min(6, Math.round((score - 45) / 12)));
  }
  return radius;
}

/**
 * 日本語: ツリーノード群から座標・辺情報を計算してダイアグラムレイアウトを構築します。
 * English: Builds diagram layout coordinates and edges from tree nodes.
 */
function buildTreeDiagramLayout(nodes: TreeNodeItem[]): {
  nodes: TreeLayoutNode[];
  edges: TreeLayoutEdge[];
  width: number;
  height: number;
  depths: number[];
} {
  if (nodes.length === 0) {
    return { nodes: [], edges: [], width: 760, height: 360, depths: [] };
  }

  const sortedNodes = [...nodes].sort((a, b) => a.depth - b.depth || a.id - b.id);
  const nodeIds = new Set(sortedNodes.map((node) => node.id));
  const childrenByParent = new Map<number | null, TreeNodeItem[]>();
  for (const node of sortedNodes) {
    const parentKey =
      node.parent_id !== null && nodeIds.has(node.parent_id) ? node.parent_id : null;
    const bucket = childrenByParent.get(parentKey) ?? [];
    bucket.push(node);
    childrenByParent.set(parentKey, bucket);
  }
  for (const bucket of childrenByParent.values()) {
    bucket.sort((a, b) => a.id - b.id);
  }

  const roots = childrenByParent.get(null) ?? [sortedNodes[0]];
  const maxDepth = Math.max(...sortedNodes.map((node) => node.depth));
  const yStep = nodes.length >= 10 ? 78 : nodes.length >= 6 ? 90 : 108;
  const xStep = 210;
  const marginX = 96;
  const marginY = 88;
  let leafIndex = 0;
  const positioned = new Map<number, TreeLayoutNode>();

  /**
   * 日本語: 子ノードを再帰配置し、現在ノードのY座標を返します。
   * English: Recursively places children and returns the node's Y coordinate.
   */
  const placeNode = (node: TreeNodeItem): number => {
    const children = childrenByParent.get(node.id) ?? [];
    let y = marginY + leafIndex * yStep;
    if (children.length === 0) {
      leafIndex += 1;
    } else {
      const childYs = children.map((child) => placeNode(child));
      y = average(childYs);
    }
    const radius = treeNodeRadius(node);
    positioned.set(node.id, {
      ...node,
      x: marginX + node.depth * xStep,
      y,
      radius,
    });
    return y;
  };

  roots.forEach((root, index) => {
    if (index > 0) {
      leafIndex += 1;
    }
    placeNode(root);
  });

  for (const node of sortedNodes) {
    if (!positioned.has(node.id)) {
      placeNode(node);
    }
  }

  const layoutNodes = sortedNodes
    .map((node) => positioned.get(node.id))
    .filter((node): node is TreeLayoutNode => Boolean(node));
  const byId = new Map(layoutNodes.map((node) => [node.id, node]));
  const edges: TreeLayoutEdge[] = [];

  for (const node of layoutNodes) {
    if (node.parent_id === null) {
      continue;
    }
    const parent = byId.get(node.parent_id);
    if (!parent) {
      continue;
    }
    const startX = parent.x + parent.radius + 8;
    const endX = node.x - node.radius - 8;
    const dx = Math.max(60, endX - startX);
    const bend = Math.max(34, dx * 0.42);
    const highlighted =
      Boolean(node.is_on_selected_path || node.is_selected) &&
      (parent.kind === "root" || Boolean(parent.is_on_selected_path || parent.is_selected));
    edges.push({
      key: `${parent.id}-${node.id}`,
      path: `M ${startX} ${parent.y} C ${startX + bend} ${parent.y}, ${endX - bend} ${node.y}, ${endX} ${node.y}`,
      highlighted,
      running: node.status === "running",
      queued: node.status === "queued",
      failed: node.status === "failed",
    });
  }

  const heights = layoutNodes.map((node) => node.y + node.radius);
  const width = marginX * 2 + maxDepth * xStep + 160;
  const height = Math.max(360, Math.max(...heights) + marginY);
  const depths = Array.from(new Set(layoutNodes.map((node) => node.depth))).sort((a, b) => a - b);

  return { nodes: layoutNodes, edges, width, height, depths };
}

/**
 * 日本語: ノード詳細をホバー用ツールチップ文へ整形します。
 * English: Formats node details into a hover tooltip string.
 */
function buildTreeNodeTooltip(node: TreeNodeItem, pruneSummary?: TreePruneSummary): string {
  const status = treeStatusMeta(node.status).label;
  const parts = [node.label || node.branch_id || "探索ノード", `${status} / depth ${node.depth}`];
  if (node.summary) {
    parts.push(node.summary);
  }
  if (node.prune_reasons.length > 0) {
    parts.push(`剪定理由: ${node.prune_reasons.join(" / ")}`);
  }
  if (node.queries[0]) {
    parts.push(`query: ${node.queries[0]}`);
  }
  if (pruneSummary && pruneSummary.count > 0) {
    parts.push(`剪定: ${pruneSummary.count}件`);
    if (pruneSummary.reasons.length > 0) {
      parts.push(`剪定理由: ${pruneSummary.reasons.slice(0, 3).join(" / ")}`);
    } else if (pruneSummary.labels.length > 0) {
      const labels = pruneSummary.labels.slice(0, 3);
      const suffix = pruneSummary.labels.length > labels.length ? " / ..." : "";
      parts.push(`中止した候補: ${labels.join(" / ")}${suffix}`);
    }
  }
  return parts.join("\n");
}

/**
 * 日本語: ツリー図のノード凡例アイテムを描画します。
 * English: Renders a node legend item for the tree diagram.
 */
function TreeLegendItem({
  color,
  accent,
  label,
  count,
  doubleRing = false,
  dashed = false,
  cut = false,
}: {
  color: string;
  accent: string;
  label: string;
  count: number;
  doubleRing?: boolean;
  dashed?: boolean;
  cut?: boolean;
}) {
  return (
    <div className="inline-flex items-center gap-2 rounded-full border border-white/80 bg-white/88 px-3 py-1.5 text-[11px] font-medium text-slate-700 shadow-sm">
      <svg viewBox="0 0 20 20" aria-hidden="true" className="h-4 w-4">
        {doubleRing && (
          <circle
            cx="10"
            cy="10"
            r="8.2"
            fill="none"
            stroke={accent}
            strokeWidth="2.2"
            opacity="0.45"
          />
        )}
        <circle
          cx="10"
          cy="10"
          r="5.3"
          fill={color}
          stroke={accent}
          strokeWidth="1.8"
          strokeDasharray={dashed ? "2.6 2.6" : undefined}
        />
        {cut && (
          <>
            <path
              d="M 6.7 11.9 L 11.9 6.7"
              fill="none"
              stroke={accent}
              strokeWidth="1.8"
              strokeLinecap="round"
            />
            <path
              d="M 9.4 13.4 L 13.4 9.4"
              fill="none"
              stroke={accent}
              strokeWidth="1.4"
              strokeLinecap="round"
              opacity="0.55"
            />
          </>
        )}
      </svg>
      <span>{label}</span>
      <span className="rounded-full bg-slate-100 px-1.5 py-0.5 text-[10px] font-semibold text-slate-600">
        {count}
      </span>
    </div>
  );
}

/**
 * 日本語: ツリー図のエッジ凡例アイテムを描画します。
 * English: Renders an edge legend item for the tree diagram.
 */
function TreeEdgeLegendItem({
  color,
  label,
  dashed,
}: {
  color: string;
  label: string;
  dashed: string;
}) {
  return (
    <div className="inline-flex items-center gap-2 rounded-full border border-white/80 bg-white/82 px-3 py-1.5 text-[11px] font-medium text-slate-600 shadow-sm">
      <svg viewBox="0 0 24 12" aria-hidden="true" className="h-3 w-6">
        <path
          d="M 2 6 C 7 6, 8 3, 12 3 S 17 9, 22 9"
          fill="none"
          stroke={color}
          strokeWidth="1.9"
          strokeLinecap="round"
          strokeDasharray={dashed}
        />
      </svg>
      <span>{label}</span>
    </div>
  );
}

/**
 * 日本語: 探索ツリー全体の可視化パネルを描画します。
 * English: Renders the full exploration tree visualization panel.
 */
function TreeDiagram({
  nodes,
  stats,
  currentStage,
  isLive,
  summary,
  focusKind,
  focusBranch,
}: {
  nodes: TreeNodeItem[];
  stats: TreeStats;
  currentStage: string;
  isLive: boolean;
  summary: string;
  focusKind: string;
  focusBranch: TreeFocusBranch | null;
}) {
  const visibleNodes = nodes.filter((node) => node.kind !== "pruned");
  const pruneSummaryByParent = buildTreePruneSummary(nodes);
  const layout = buildTreeDiagramLayout(visibleNodes);
  const selectedCount = visibleNodes.filter((node) => node.is_selected).length;
  const pathCount = visibleNodes.filter((node) => node.is_on_selected_path).length;

  if (visibleNodes.length === 0) {
    return (
      <div className="rounded-[28px] border border-dashed border-teal-200 bg-white/85 p-8 text-center shadow-card">
        <div className="mx-auto flex w-fit items-center gap-2 rounded-full bg-teal-50 px-3 py-1.5 text-[11px] font-semibold uppercase tracking-[0.16em] text-teal-700">
          <span className="h-2 w-2 rounded-full bg-teal-500" />
          frontier standby
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex flex-wrap items-center gap-2">
          {currentStage && (
            <span className="rounded-full border border-teal-100 bg-teal-50/80 px-3 py-1.5 text-[11px] font-semibold text-teal-700 shadow-sm">
              {currentStage}
            </span>
          )}
          {isLive && (
            <span className="inline-flex items-center gap-2 rounded-full border border-cyan-100 bg-cyan-50/90 px-3 py-1.5 text-[11px] font-semibold text-cyan-800 shadow-sm">
              <span className="h-2 w-2 rounded-full bg-cyan-500 animate-pulseSoft" />
              live
            </span>
          )}
          <span className="rounded-full border border-slate-200 bg-white/90 px-3 py-1.5 text-[11px] font-semibold text-slate-700 shadow-sm">
            {stats.termination_label || "進行中"}
          </span>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <TreeLegendItem
            color="#ffffff"
            accent="#0f766e"
            label="採用"
            count={selectedCount}
            doubleRing
          />
          <TreeLegendItem color="#ffffff" accent="#0891b2" label="採用パス" count={pathCount} />
          <TreeLegendItem
            color="#38bdf8"
            accent="#0369a1"
            label="実行中"
            count={stats.running_node_count}
          />
          <TreeLegendItem
            color="#10b981"
            accent="#047857"
            label="完了"
            count={stats.executed_node_count}
          />
          <TreeLegendItem
            color="#e0f2fe"
            accent="#0284c7"
            label="待機"
            count={stats.frontier_remaining}
          />
          <TreeLegendItem
            color="#ffffff"
            accent="#c2410c"
            label="剪定"
            count={stats.pruned_node_count}
            dashed
            cut
          />
          <TreeLegendItem
            color="#fda4af"
            accent="#be123c"
            label="失敗"
            count={stats.failed_node_count}
          />
        </div>
      </div>

      <div className="flex flex-wrap items-center gap-2 text-[11px] font-medium text-slate-500">
        <TreeEdgeLegendItem color="#0284c7" label="青の点線: 待機中の枝" dashed="4 5" />
      </div>

      <div className="overflow-x-auto rounded-[30px] border border-teal-200/80 bg-[radial-gradient(circle_at_top_left,rgba(45,212,191,0.18),rgba(255,255,255,0.96)_52%),linear-gradient(135deg,rgba(255,255,255,0.98),rgba(236,253,245,0.88))] p-3 shadow-soft">
        <div style={{ minWidth: `${layout.width}px` }}>
          <svg
            viewBox={`0 0 ${layout.width} ${layout.height}`}
            role="img"
            aria-label="探索ツリーのダイアグラム"
            className="w-full"
            style={{ height: `${layout.height}px` }}
          >
            <rect
              x="0"
              y="0"
              width={layout.width}
              height={layout.height}
              rx="26"
              fill="rgba(255,255,255,0.26)"
            />

            {layout.depths.map((depth, index) => {
              const columnNodes = layout.nodes.filter((node) => node.depth === depth);
              if (columnNodes.length === 0) {
                return null;
              }
              const x = average(columnNodes.map((node) => node.x));
              return (
                <g key={`band-${depth}`}>
                  <rect
                    x={x - 72}
                    y={24}
                    width={144}
                    height={layout.height - 48}
                    rx={30}
                    fill={index % 2 === 0 ? "rgba(255,255,255,0.46)" : "rgba(236,253,245,0.42)"}
                  />
                  <line
                    x1={x}
                    y1={28}
                    x2={x}
                    y2={layout.height - 28}
                    stroke="rgba(15,118,110,0.08)"
                    strokeWidth="1"
                  />
                </g>
              );
            })}

            {layout.edges.map((edge) => {
              const highlightStroke = edge.running
                ? "#0ea5e9"
                : edge.failed
                  ? "#fb7185"
                  : edge.highlighted
                    ? "#0f766e"
                    : edge.queued
                      ? "#0284c7"
                      : "#cbd5e1";
              return (
                <g key={edge.key}>
                  <path
                    d={edge.path}
                    fill="none"
                    stroke={
                      edge.highlighted || edge.running
                        ? "rgba(45,212,191,0.16)"
                        : "rgba(148,163,184,0.12)"
                    }
                    strokeWidth={edge.highlighted || edge.running ? 12 : 8}
                    strokeLinecap="round"
                  />
                  <path
                    d={edge.path}
                    fill="none"
                    stroke={highlightStroke}
                    strokeWidth={edge.highlighted || edge.running ? 3.5 : 2.4}
                    strokeLinecap="round"
                    strokeDasharray={edge.queued ? "4 5" : undefined}
                    opacity={edge.failed ? 0.72 : 0.92}
                  />
                </g>
              );
            })}

            {layout.nodes.map((node) => {
              const status = treeStatusMeta(node.status);
              const isSelected = Boolean(node.is_selected);
              const isPath = Boolean(node.is_on_selected_path);
              const pruneSummary = pruneSummaryByParent.get(node.id);
              const tooltip = buildTreeNodeTooltip(node, pruneSummary);
              return (
                <g
                  key={node.id}
                  transform={`translate(${node.x} ${node.y})`}
                  className={node.status === "running" ? "animate-pulseSoft" : undefined}
                >
                  <title>{tooltip}</title>

                  {isPath && (
                    <circle
                      cx="0"
                      cy="0"
                      r={node.radius + (isSelected ? 12 : 8)}
                      fill="none"
                      stroke={isSelected ? "rgba(15,118,110,0.44)" : "rgba(8,145,178,0.28)"}
                      strokeWidth={isSelected ? 7 : 5}
                    />
                  )}

                  {node.kind === "root" && (
                    <>
                      <circle cx="0" cy="0" r={node.radius + 12} fill="rgba(15,118,110,0.10)" />
                      <circle
                        cx="0"
                        cy="0"
                        r={node.radius}
                        fill="#ffffff"
                        stroke="#0f766e"
                        strokeWidth="3"
                      />
                      <circle cx="0" cy="0" r={10} fill="#0f766e" opacity="0.92" />
                      <circle cx="-16" cy="-12" r={3} fill="#14b8a6" opacity="0.95" />
                      <circle cx="16" cy="-12" r={3} fill="#0ea5e9" opacity="0.95" />
                      <circle cx="0" cy="18" r={3} fill="#34d399" opacity="0.95" />
                    </>
                  )}

                  {node.kind !== "root" && node.kind !== "pruned" && (
                    <>
                      {node.status === "running" && (
                        <circle
                          cx="0"
                          cy="0"
                          r={node.radius + 10}
                          fill="none"
                          stroke="rgba(56,189,248,0.28)"
                          strokeWidth="5"
                        />
                      )}
                      <circle
                        cx="0"
                        cy="0"
                        r={node.radius}
                        fill={status.fill}
                        stroke={status.stroke}
                        strokeWidth={node.status === "queued" ? 2.6 : 3}
                      />
                      {node.status === "queued" && (
                        <>
                          <circle
                            cx="0"
                            cy="0"
                            r={node.radius + 6}
                            fill="none"
                            stroke="rgba(2,132,199,0.28)"
                            strokeWidth="2.4"
                            strokeDasharray="3 5"
                          />
                          <circle cx="0" cy="0" r={6.5} fill="#7dd3fc" />
                        </>
                      )}
                      {node.status === "completed" && (
                        <>
                          <circle cx="0" cy="0" r={7.2} fill="#ffffff" opacity="0.98" />
                          <circle cx="0" cy="0" r={2.8} fill={status.stroke} />
                        </>
                      )}
                      {node.status === "running" && (
                        <>
                          <circle cx="0" cy="0" r={6.4} fill="#ffffff" opacity="0.96" />
                          <circle cx="0" cy="0" r={3} fill="#0369a1" />
                          <circle
                            cx="0"
                            cy="0"
                            r={node.radius + 4}
                            fill="none"
                            stroke="#0369a1"
                            strokeWidth="1.5"
                            strokeDasharray="3 5"
                            opacity="0.6"
                          />
                        </>
                      )}
                      {node.status === "failed" && (
                        <>
                          <line
                            x1={-8}
                            y1={-8}
                            x2={8}
                            y2={8}
                            stroke="#881337"
                            strokeWidth="3"
                            strokeLinecap="round"
                          />
                          <line
                            x1={8}
                            y1={-8}
                            x2={-8}
                            y2={8}
                            stroke="#881337"
                            strokeWidth="3"
                            strokeLinecap="round"
                          />
                        </>
                      )}
                    </>
                  )}

                  {isSelected && node.kind !== "root" && (
                    <circle
                      cx="0"
                      cy="0"
                      r={node.radius + 11}
                      fill="none"
                      stroke="#0f766e"
                      strokeWidth="3"
                      strokeDasharray="1 0"
                      opacity="0.9"
                    />
                  )}

                  {pruneSummary && pruneSummary.count > 0 && (
                    <g transform={`translate(${node.radius * 0.88} ${-node.radius * 0.84})`}>
                      <title>
                        {`剪定 ${pruneSummary.count}件${
                          pruneSummary.reasons.length > 0
                            ? `\n${pruneSummary.reasons.slice(0, 3).join("\n")}`
                            : ""
                        }`}
                      </title>
                      <rect
                        x={pruneSummary.count >= 10 ? -13 : -11}
                        y="-8.5"
                        width={pruneSummary.count >= 10 ? 26 : 22}
                        height="17"
                        rx="8.5"
                        fill="#fff7ed"
                        stroke="#f59e0b"
                        strokeWidth="1.7"
                      />
                      <text
                        x="0"
                        y="3.7"
                        textAnchor="middle"
                        fontSize="7.2"
                        fontWeight="700"
                        fill="#c2410c"
                      >
                        {`×${pruneSummary.count > 99 ? "99+" : pruneSummary.count}`}
                      </text>
                    </g>
                  )}
                </g>
              );
            })}
          </svg>
        </div>
      </div>

      <div className="flex flex-wrap items-center gap-2 text-[11px] font-medium text-slate-500">
        <span className="rounded-full bg-white/88 px-3 py-1.5 shadow-sm">
          nodes {layout.nodes.length}
        </span>
        <span className="rounded-full bg-white/88 px-3 py-1.5 shadow-sm">
          depth {stats.max_depth_reached}
        </span>
        <span className="rounded-full bg-white/88 px-3 py-1.5 shadow-sm">hover for details</span>
      </div>

      {focusBranch && (
        <div className="rounded-xl border border-teal-100 bg-teal-50/60 px-4 py-3 text-[12px] text-teal-900">
          <div className="mb-1.5 flex flex-wrap items-center gap-2">
            <span className="font-semibold">
              {focusKind === "selected" ? "採用分岐" : "最有力分岐"}
            </span>
            {focusBranch.label && (
              <span className="rounded-full bg-teal-100 px-2 py-0.5 font-medium text-teal-800">
                {String(focusBranch.label)}
              </span>
            )}
            {focusBranch.branch_score != null && (
              <span className="rounded-full bg-white/80 px-2 py-0.5 text-slate-600">
                スコア {Number(focusBranch.branch_score).toFixed(1)}
              </span>
            )}
            {focusBranch.detail_coverage != null && (
              <span className="rounded-full bg-white/80 px-2 py-0.5 text-slate-600">
                詳細補完 {Number(focusBranch.detail_coverage).toFixed(1)}%
              </span>
            )}
          </div>
          {focusBranch.summary && (
            <p className="leading-6 text-teal-800">{String(focusBranch.summary)}</p>
          )}
        </div>
      )}

      {summary && (
        <div className="rounded-2xl border border-slate-200/80 bg-white/75 px-3 py-2.5 shadow-sm">
          <p className="whitespace-pre-wrap break-words text-[12px] leading-6 text-slate-600">
            {summary}
          </p>
        </div>
      )}
    </div>
  );
}

/**
 * 日本語: タイムライン段階の状態表示色とラベルを返します。
 * English: Returns color and label metadata for timeline stage status.
 */
function timelineStatusMeta(status: string) {
  if (status === "completed") {
    return {
      label: "完了",
      fill: "#14b8a6",
      stroke: "#0f766e",
      edge: "#14b8a6",
      halo: "rgba(20,184,166,0.18)",
    };
  }
  if (status === "running") {
    return {
      label: "実行中",
      fill: "#38bdf8",
      stroke: "#0369a1",
      edge: "#0ea5e9",
      halo: "rgba(56,189,248,0.18)",
    };
  }
  if (status === "failed") {
    return {
      label: "失敗",
      fill: "#fda4af",
      stroke: "#be123c",
      edge: "#fb7185",
      halo: "rgba(251,113,133,0.16)",
    };
  }
  return {
    label: "待機",
    fill: "#ffffff",
    stroke: "#94a3b8",
    edge: "#cbd5e1",
    halo: "rgba(148,163,184,0.10)",
  };
}

/**
 * 日本語: 段階インデックスとラベルからタイムライン種別を推定します。
 * English: Infers timeline stage kind from label and index.
 */
function timelineStageKind(index: number, label: string): "plan" | "tree" | "synthesize" {
  if (label.includes("探索")) {
    return "tree";
  }
  if (label.includes("要約") || label.includes("結果")) {
    return "synthesize";
  }
  if (index === 1) {
    return "tree";
  }
  if (index === 2) {
    return "synthesize";
  }
  return "plan";
}

/**
 * 日本語: タイムライン段階の中心グリフを状態付きで描画します。
 * English: Draws the central timeline stage glyph with state styling.
 */
function TimelineStageGlyph({
  kind,
  status,
  failed = false,
  running = false,
}: {
  kind: "plan" | "tree" | "synthesize";
  status: ReturnType<typeof timelineStatusMeta>;
  failed?: boolean;
  running?: boolean;
}) {
  return (
    <>
      {running && (
        <circle
          cx="0"
          cy="0"
          r={39}
          fill="none"
          stroke={status.edge}
          strokeWidth="5"
          strokeDasharray="5 7"
          opacity="0.35"
        />
      )}

      {kind === "plan" && (
        <>
          <path
            d="M-30 -20 L0 -34 L30 -20 L30 20 L0 34 L-30 20 Z"
            fill={status.fill}
            stroke={status.stroke}
            strokeWidth="3"
          />
          <path
            d="M-14 -10 H12"
            stroke="#ffffff"
            strokeWidth="3.4"
            strokeLinecap="round"
            opacity="0.96"
          />
          <path
            d="M-14 0 H18"
            stroke="#ffffff"
            strokeWidth="3.4"
            strokeLinecap="round"
            opacity="0.88"
          />
          <path
            d="M-14 10 H8"
            stroke="#ffffff"
            strokeWidth="3.4"
            strokeLinecap="round"
            opacity="0.78"
          />
        </>
      )}

      {kind === "tree" && (
        <>
          <circle cx="0" cy="0" r="28" fill={status.fill} stroke={status.stroke} strokeWidth="3" />
          <circle cx="0" cy="0" r="7.5" fill="#ffffff" opacity="0.96" />
          <circle cx="-16" cy="-12" r="5.5" fill="#ffffff" opacity="0.9" />
          <circle cx="16" cy="-12" r="5.5" fill="#ffffff" opacity="0.9" />
          <circle cx="0" cy="18" r="5.5" fill="#ffffff" opacity="0.9" />
          <path
            d="M-11 -8 L-4 -3 M11 -8 L4 -3 M0 10 L0 4"
            stroke={status.stroke}
            strokeWidth="2.4"
            strokeLinecap="round"
          />
        </>
      )}

      {kind === "synthesize" && (
        <>
          <rect
            x="-30"
            y="-26"
            width="60"
            height="52"
            rx="18"
            fill={status.fill}
            stroke={status.stroke}
            strokeWidth="3"
          />
          <path
            d="M-12 -8 H8 M-12 0 H14 M-12 8 H4"
            stroke="#ffffff"
            strokeWidth="3.4"
            strokeLinecap="round"
            opacity="0.92"
          />
          <circle cx="16" cy="0" r="4" fill="#ffffff" opacity="0.96" />
        </>
      )}

      {failed && (
        <>
          <line
            x1="-11"
            y1="-11"
            x2="11"
            y2="11"
            stroke="#7f1d1d"
            strokeWidth="4"
            strokeLinecap="round"
          />
          <line
            x1="11"
            y1="-11"
            x2="-11"
            y2="11"
            stroke="#7f1d1d"
            strokeWidth="4"
            strokeLinecap="round"
          />
        </>
      )}
    </>
  );
}

/**
 * 日本語: 調査進行の3段階タイムライン可視化を描画します。
 * English: Renders the three-step research progress timeline diagram.
 */
function TimelineDiagram({
  items,
  progress,
  currentStage,
  summary,
}: {
  items: Array<Record<string, unknown>>;
  progress: number;
  currentStage: string;
  summary: string;
}) {
  const fallbackItems = [
    { label: "計画確認", status: currentStage === "計画確認" ? "running" : "pending", detail: "" },
    { label: "動的探索", status: currentStage === "動的探索" ? "running" : "pending", detail: "" },
    { label: "結果要約", status: currentStage === "結果要約" ? "running" : "pending", detail: "" },
  ];
  const stages = (items.length > 0 ? items : fallbackItems).slice(0, 3).map((item, index) => ({
    label: toDisplayText(item.label) || `step-${index + 1}`,
    status: toDisplayText(item.status) || "pending",
    detail: toDisplayText(item.detail),
    x: 120 + index * 300,
    y: 116,
    kind: timelineStageKind(index, toDisplayText(item.label)),
  }));
  const width = stages.length > 1 ? stages[stages.length - 1].x + 120 : 840;
  const height = 250;
  const connectors = stages.slice(0, -1).map((stage, index) => {
    const next = stages[index + 1];
    const startX = stage.x + 46;
    const endX = next.x - 46;
    return { startX, endX, y: stage.y };
  });
  const normalizedProgress = Math.max(0, Math.min(100, progress));
  const segmentScale = connectors.length > 0 ? (normalizedProgress / 100) * connectors.length : 0;
  const completedCount = stages.filter((stage) => stage.status === "completed").length;
  const runningCount = stages.filter((stage) => stage.status === "running").length;
  const failedCount = stages.filter((stage) => stage.status === "failed").length;

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex flex-wrap items-center gap-2">
          <span className="rounded-full border border-slate-200 bg-white/90 px-3 py-1.5 text-[11px] font-semibold text-slate-700 shadow-sm">
            {currentStage || "待機中"}
          </span>
          <span className="rounded-full border border-teal-100 bg-teal-50/90 px-3 py-1.5 text-[11px] font-semibold text-teal-700 shadow-sm">
            {normalizedProgress}%
          </span>
        </div>
        <div className="flex flex-wrap items-center gap-2 text-[11px] font-medium text-slate-600">
          <span className="rounded-full bg-white/88 px-3 py-1.5 shadow-sm">
            完了 {completedCount}
          </span>
          <span className="rounded-full bg-white/88 px-3 py-1.5 shadow-sm">
            実行中 {runningCount}
          </span>
          <span className="rounded-full bg-white/88 px-3 py-1.5 shadow-sm">失敗 {failedCount}</span>
        </div>
      </div>

      <div className="rounded-[30px] border border-slate-200/90 bg-[radial-gradient(circle_at_top_left,rgba(14,165,233,0.16),rgba(255,255,255,0.98)_42%),linear-gradient(135deg,rgba(255,255,255,0.98),rgba(248,250,252,0.92))] p-3 shadow-card">
        <div>
          <svg
            viewBox={`0 0 ${width} ${height}`}
            role="img"
            aria-label="調査全体の進行ダイアグラム"
            className="w-full"
            style={{ height: `${height}px` }}
          >
            <rect x="0" y="0" width={width} height={height} rx="26" fill="rgba(255,255,255,0.22)" />

            {connectors.map((connector, index) => {
              const segmentProgress = Math.max(0, Math.min(1, segmentScale - index));
              const activeEndX =
                connector.startX + (connector.endX - connector.startX) * segmentProgress;
              return (
                <g key={`connector-${index}`}>
                  <line
                    x1={connector.startX}
                    y1={connector.y}
                    x2={connector.endX}
                    y2={connector.y}
                    stroke="rgba(148,163,184,0.26)"
                    strokeWidth="16"
                    strokeLinecap="round"
                  />
                  {segmentProgress > 0 && (
                    <>
                      <line
                        x1={connector.startX}
                        y1={connector.y}
                        x2={activeEndX}
                        y2={connector.y}
                        stroke="rgba(45,212,191,0.18)"
                        strokeWidth="18"
                        strokeLinecap="round"
                      />
                      <line
                        x1={connector.startX}
                        y1={connector.y}
                        x2={activeEndX}
                        y2={connector.y}
                        stroke="#0f766e"
                        strokeWidth="5"
                        strokeLinecap="round"
                      />
                    </>
                  )}
                </g>
              );
            })}

            {stages.map((stage, index) => {
              const meta = timelineStatusMeta(stage.status);
              const isRunning = stage.status === "running";
              const tooltip = [stage.label, meta.label, stage.detail].filter(Boolean).join("\n");
              return (
                <g key={stage.label} transform={`translate(${stage.x} ${stage.y})`}>
                  <title>{tooltip}</title>
                  <circle cx="0" cy="0" r="48" fill={meta.halo} />
                  <TimelineStageGlyph
                    kind={stage.kind}
                    status={meta}
                    failed={stage.status === "failed"}
                    running={isRunning}
                  />
                  {isRunning && <circle cx="0" cy="-44" r="4.5" fill="#0ea5e9" />}
                  <text
                    x="0"
                    y="76"
                    textAnchor="middle"
                    className="fill-slate-700"
                    style={{ fontSize: "13px", fontWeight: 700, letterSpacing: "0.02em" }}
                  >
                    {stage.label}
                  </text>
                  <text
                    x="0"
                    y="95"
                    textAnchor="middle"
                    className="fill-slate-400"
                    style={{ fontSize: "10px", fontWeight: 700, letterSpacing: "0.14em" }}
                  >
                    {meta.label}
                  </text>
                  {index < stages.length - 1 && (
                    <path
                      d={`M 72 0 L 58 -6 L 58 6 Z`}
                      fill={segmentScale > index ? "#0f766e" : "#cbd5e1"}
                      opacity={0.9}
                    />
                  )}
                </g>
              );
            })}
          </svg>
        </div>
      </div>

      {summary && (
        <div className="rounded-2xl border border-slate-200/80 bg-white/75 px-3 py-2.5 shadow-sm">
          <p className="whitespace-pre-wrap break-words text-[12px] leading-6 text-slate-600">
            {summary}
          </p>
        </div>
      )}
    </div>
  );
}

/* ---------- Block dispatcher ---------- */

/**
 * 日本語: ブロック種別ごとに適切なUIを選択して描画します。
 * English: Dispatches each block type to its corresponding UI renderer.
 */
export default function StructuredBlock({
  block,
  disabled = false,
  onCompareExecute,
  onCompareToggle,
  onChecklistToggle,
  onQuestionExecute,
  onQuestionInputChange,
  onQuestionSuggestionToggle,
  onActionExecute,
}: Props) {
  if (block.type === "text") {
    const tone = TONES.text;
    const body = toDisplayText(block.content.body);
    const format = toDisplayText(block.content.format).toLowerCase();
    return (
      <section
        className={`overflow-hidden rounded-2xl border ${tone.border} ${tone.surface} shadow-card`}
      >
        <SectionHeader block={block} />
        <div className="px-4 py-3.5">
          {format === "markdown" ? (
            <MarkdownBody body={body} />
          ) : (
            <p className="whitespace-pre-wrap text-[14px] leading-7 text-ink">{body}</p>
          )}
        </div>
      </section>
    );
  }

  if (block.type === "warning") {
    const tone = TONES.warning;
    return (
      <section
        className={`overflow-hidden rounded-2xl border ${tone.border} ${tone.surface} shadow-card`}
      >
        <SectionHeader block={block} />
        <div className="px-4 py-3.5">
          <p className="whitespace-pre-wrap text-[14px] leading-7 text-amber-900">
            {toDisplayText(block.content.body)}
          </p>
        </div>
      </section>
    );
  }

  if (block.type === "plan") {
    const tone = TONES.plan;
    const summary = toDisplayText(block.content.summary);
    const goal = toDisplayText(block.content.goal);
    const rationale = toDisplayText(block.content.rationale);
    const searchQuery = toDisplayText(block.content.search_query);
    const seedQueries = Array.isArray(block.content.seed_queries)
      ? block.content.seed_queries.map((item) => toDisplayText(item)).filter(Boolean)
      : [];
    const conditions = Array.isArray(block.content.conditions)
      ? (block.content.conditions as Array<Record<string, unknown>>)
      : [];
    const strategy = Array.isArray(block.content.strategy)
      ? block.content.strategy.map((item) => toDisplayText(item)).filter(Boolean)
      : [];
    const openQuestions = Array.isArray(block.content.open_questions)
      ? block.content.open_questions.map((item) => toDisplayText(item)).filter(Boolean)
      : [];

    return (
      <section
        className={`overflow-hidden rounded-2xl border ${tone.border} ${tone.surface} shadow-card`}
      >
        <SectionHeader block={block} />
        <div className="space-y-4 px-4 py-3.5">
          {(summary || goal || rationale) && (
            <div className="rounded-2xl border border-indigo-100 bg-white/85 p-4">
              {summary && <p className="text-sm font-semibold text-indigo-900">{summary}</p>}
              {goal && <p className="mt-2 text-[13px] leading-6 text-inkMuted">{goal}</p>}
              {rationale && (
                <p className="mt-2 text-[13px] leading-6 text-indigo-800/80">{rationale}</p>
              )}
            </div>
          )}

          {conditions.length > 0 && (
            <div className="rounded-2xl border border-indigo-100 bg-white/85 p-4">
              <p className="text-[11px] font-semibold uppercase tracking-[0.14em] text-indigo-600">
                条件
              </p>
              <dl className="mt-3 grid gap-2 sm:grid-cols-2">
                {conditions.map((condition, idx) => (
                  <div
                    key={`${toDisplayText(condition.label)}-${idx}`}
                    className="rounded-xl bg-indigo-50/70 px-3 py-2"
                  >
                    <dt className="text-[11px] font-semibold text-indigo-700">
                      {toDisplayText(condition.label)}
                    </dt>
                    <dd className="mt-1 text-sm text-ink">{toDisplayText(condition.value)}</dd>
                    {toDisplayText(condition.reason) && (
                      <dd className="mt-1 text-[12px] leading-5 text-inkMuted">
                        {toDisplayText(condition.reason)}
                      </dd>
                    )}
                  </div>
                ))}
              </dl>
            </div>
          )}

          {strategy.length > 0 && (
            <div className="rounded-2xl border border-indigo-100 bg-white/85 p-4">
              <p className="text-[11px] font-semibold uppercase tracking-[0.14em] text-indigo-600">
                調査の進め方
              </p>
              <ul className="mt-3 space-y-2 text-sm leading-6 text-ink">
                {strategy.map((item) => (
                  <li key={item} className="flex items-start gap-2">
                    <span className="mt-2 h-1.5 w-1.5 shrink-0 rounded-full bg-indigo-500" />
                    <span>{item}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {(searchQuery || seedQueries.length > 0 || openQuestions.length > 0) && (
            <div className="grid gap-3 sm:grid-cols-2">
              {(seedQueries.length > 0 || searchQuery) && (
                <div className="rounded-2xl border border-indigo-100 bg-white/85 p-4">
                  <p className="text-[11px] font-semibold uppercase tracking-[0.14em] text-indigo-600">
                    初期クエリ
                  </p>
                  {seedQueries.length > 0 ? (
                    <ul className="mt-2 space-y-2 text-sm leading-6 text-ink">
                      {seedQueries.map((item) => (
                        <li key={item}>{item}</li>
                      ))}
                    </ul>
                  ) : (
                    <p className="mt-2 text-sm leading-6 text-ink">{searchQuery}</p>
                  )}
                </div>
              )}
              {openQuestions.length > 0 && (
                <div className="rounded-2xl border border-indigo-100 bg-white/85 p-4">
                  <p className="text-[11px] font-semibold uppercase tracking-[0.14em] text-indigo-600">
                    追加で確認したい点
                  </p>
                  <ul className="mt-2 space-y-2 text-sm leading-6 text-inkMuted">
                    {openQuestions.map((item) => (
                      <li key={item}>{item}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          )}
        </div>
      </section>
    );
  }

  if (block.type === "timeline") {
    const tone = TONES.timeline;
    const progress = Math.max(
      0,
      Math.min(100, toDisplayNumber(block.content.progress_percent) ?? 0)
    );
    const currentStage = toDisplayText(block.content.current_stage);
    const summary = toDisplayText(block.content.summary);
    const items = Array.isArray(block.content.items)
      ? (block.content.items as Array<Record<string, unknown>>)
      : [];

    return (
      <section
        className={`overflow-hidden rounded-2xl border ${tone.border} ${tone.surface} shadow-card`}
      >
        <SectionHeader block={block} />
        <div className="px-4 py-3.5">
          <TimelineDiagram
            items={items}
            progress={progress}
            currentStage={currentStage}
            summary={summary}
          />
        </div>
      </section>
    );
  }

  if (block.type === "tree") {
    const tone = TONES.tree;
    const currentStage = toDisplayText(block.content.current_stage);
    const isLive = Boolean(block.content.is_live);
    const statsRecord =
      typeof block.content.stats === "object" && block.content.stats !== null
        ? (block.content.stats as Record<string, unknown>)
        : {};
    const stats: TreeStats = {
      executed_node_count: toDisplayNumber(statsRecord.executed_node_count) ?? 0,
      failed_node_count: toDisplayNumber(statsRecord.failed_node_count) ?? 0,
      pruned_node_count: toDisplayNumber(statsRecord.pruned_node_count) ?? 0,
      frontier_remaining: toDisplayNumber(statsRecord.frontier_remaining) ?? 0,
      running_node_count: toDisplayNumber(statsRecord.running_node_count) ?? 0,
      max_depth_reached: toDisplayNumber(statsRecord.max_depth_reached) ?? 0,
      termination_reason: toDisplayText(statsRecord.termination_reason),
      termination_label: toDisplayText(statsRecord.termination_label),
      node_count: toDisplayNumber(statsRecord.node_count) ?? 0,
    };
    const nodes = toRecordArray(block.content.nodes)
      .map<TreeNodeItem>((item) => ({
        id: toDisplayNumber(item.id) ?? 0,
        parent_id: toDisplayNumber(item.parent_id),
        branch_id: toDisplayText(item.branch_id),
        kind: toDisplayText(item.kind),
        status: toDisplayText(item.status),
        node_type: toDisplayText(item.node_type),
        label: toDisplayText(item.label),
        description: toDisplayText(item.description),
        summary: toDisplayText(item.summary),
        depth: toDisplayNumber(item.depth) ?? 0,
        query_count: toDisplayNumber(item.query_count) ?? 0,
        queries: toStringArray(item.queries),
        strategy_tags: toStringArray(item.strategy_tags),
        branch_score: toDisplayNumber(item.branch_score),
        frontier_score: toDisplayNumber(item.frontier_score),
        detail_coverage: toDisplayNumber(item.detail_coverage),
        structured_ratio: toDisplayNumber(item.structured_ratio),
        selected: Boolean(item.selected),
        prune_reasons: toStringArray(item.prune_reasons),
        created_at: toDisplayText(item.created_at),
        parent_label: toDisplayText(item.parent_label),
        is_selected: Boolean(item.is_selected),
        is_on_selected_path: Boolean(item.is_on_selected_path),
      }))
      .filter((item) => item.id > 0)
      .sort((a, b) => a.depth - b.depth || a.id - b.id);
    const treeSummary = toDisplayText(block.content.summary);
    const focusKind = toDisplayText(block.content.focus_kind);
    const focusBranchRecord =
      typeof block.content.focus_branch === "object" && block.content.focus_branch !== null
        ? (block.content.focus_branch as Record<string, unknown>)
        : null;
    const focusBranch: TreeFocusBranch | null = focusBranchRecord
      ? {
          branch_id: toDisplayText(focusBranchRecord.branch_id),
          label: toDisplayText(focusBranchRecord.label),
          depth: toDisplayNumber(focusBranchRecord.depth) ?? 0,
          branch_score: toDisplayNumber(focusBranchRecord.branch_score),
          detail_coverage: toDisplayNumber(focusBranchRecord.detail_coverage),
          frontier_score: toDisplayNumber(focusBranchRecord.frontier_score),
          summary: toDisplayText(focusBranchRecord.summary),
        }
      : null;

    return (
      <section
        className={`overflow-hidden rounded-2xl border ${tone.border} ${tone.surface} shadow-card`}
      >
        <SectionHeader
          block={block}
          count={nodes.filter((node) => node.kind !== "pruned").length}
        />
        <div className="px-4 py-3.5">
          <TreeDiagram
            nodes={nodes}
            stats={stats}
            currentStage={currentStage}
            isLive={isLive}
            summary={treeSummary}
            focusKind={focusKind}
            focusBranch={focusBranch}
          />
        </div>
      </section>
    );
  }

  if (block.type === "sources") {
    const tone = TONES.sources;
    const items = Array.isArray(block.content.items)
      ? (block.content.items as Array<Record<string, unknown>>)
      : [];
    return (
      <section
        className={`overflow-hidden rounded-2xl border ${tone.border} ${tone.surface} shadow-card`}
      >
        <SectionHeader block={block} count={items.length} />
        <div className="space-y-3 p-4">
          {items.map((item, idx) => {
            const queries = Array.isArray(item.queries)
              ? item.queries.map((query) => toDisplayText(query)).filter(Boolean)
              : [];
            const url = toDisplayText(item.url);
            return (
              <article
                key={`${toDisplayText(item.title)}-${idx}`}
                className="rounded-2xl border border-teal-100 bg-white/90 p-4"
              >
                <div className="flex items-start justify-between gap-3">
                  <div className="min-w-0 flex-1">
                    <p className="text-sm font-semibold text-teal-900">
                      {toDisplayText(item.title)}
                    </p>
                    {toDisplayText(item.matched_property) && (
                      <p className="mt-1 text-[12px] text-teal-700">
                        紐づく候補: {toDisplayText(item.matched_property)}
                      </p>
                    )}
                  </div>
                  {toDisplayText(item.source_name) && (
                    <span className="shrink-0 rounded-full bg-teal-100 px-2 py-1 text-[11px] font-semibold text-teal-800">
                      {toDisplayText(item.source_name)}
                    </span>
                  )}
                </div>

                {toDisplayText(item.reason) && (
                  <p className="mt-3 text-[13px] leading-6 text-ink">
                    {toDisplayText(item.reason)}
                  </p>
                )}

                {queries.length > 0 && (
                  <div className="mt-3 flex flex-wrap gap-2">
                    {queries.map((query) => (
                      <span
                        key={query}
                        className="rounded-full bg-teal-50 px-2 py-1 text-[11px] font-medium text-teal-700"
                      >
                        {query}
                      </span>
                    ))}
                  </div>
                )}

                {url && (
                  <a
                    href={url}
                    target="_blank"
                    rel="noreferrer"
                    className="mt-3 inline-flex items-center gap-2 text-[13px] font-medium text-teal-700 underline decoration-teal-300 underline-offset-4 transition hover:text-teal-900"
                  >
                    参照ページを開く
                  </a>
                )}
              </article>
            );
          })}
        </div>
      </section>
    );
  }

  if (block.type === "question") {
    const items = (block.content.items as Array<Record<string, unknown>>) ?? [];
    const intro = toDisplayText(block.content.intro);
    const selectedCount = items.filter((item) => {
      const answer = toDisplayText(item.free_text) || toDisplayText(item.selected_example);
      return Boolean(answer.trim());
    }).length;
    const tone = TONES.question;
    return (
      <section
        className={`overflow-hidden rounded-2xl border ${tone.border} ${tone.surface} shadow-card`}
      >
        <SectionHeader block={block} count={items.length} />
        <div className="space-y-3 px-4 py-3.5">
          {intro && <p className="text-[14px] leading-7 text-emerald-900">{intro}</p>}
          {items.map((item, idx) => {
            const label = toDisplayText(item.label) || `質問 ${idx + 1}`;
            const question = toDisplayText(item.question);
            const examples = Array.isArray(item.examples)
              ? item.examples.map((example) => toDisplayText(example)).filter(Boolean)
              : [];
            const selectedExample = toDisplayText(item.selected_example);
            const freeText = toDisplayText(item.free_text);
            const answerValue = freeText || selectedExample;

            return (
              <div
                key={`${label}-${idx}`}
                className="rounded-2xl border border-emerald-100 bg-white/90 p-3 shadow-sm"
              >
                <p className="text-[11px] font-semibold uppercase tracking-[0.14em] text-emerald-600">
                  {label}
                </p>
                <p className="mt-1 text-sm leading-6 text-ink">{question}</p>
                {examples.length > 0 && (
                  <div className="mt-3 flex flex-wrap gap-2">
                    {examples.map((example) => (
                      <button
                        key={example}
                        type="button"
                        aria-pressed={answerValue.trim() === example}
                        disabled={disabled}
                        onClick={() => onQuestionSuggestionToggle?.(idx, example)}
                        className={`rounded-full border px-3 py-1.5 text-xs font-medium transition disabled:cursor-not-allowed disabled:opacity-55 ${
                          answerValue.trim() === example
                            ? "border-emerald-700 bg-emerald-700 text-white shadow-sm"
                            : "border-emerald-200 bg-emerald-50 text-emerald-800 hover:border-emerald-300 hover:bg-emerald-100"
                        }`}
                      >
                        {example}
                      </button>
                    ))}
                  </div>
                )}
                <div className="mt-3">
                  <label className="mb-1 block text-[11px] font-semibold text-emerald-700">
                    選択肢にない場合は自由入力
                  </label>
                  <textarea
                    value={answerValue}
                    disabled={disabled}
                    rows={2}
                    onChange={(event) => onQuestionInputChange?.(idx, event.target.value)}
                    placeholder={getQuestionInputPlaceholder(label)}
                    className="min-h-[72px] w-full resize-y rounded-2xl border border-emerald-200 bg-emerald-50/45 px-3 py-2 text-sm leading-6 text-ink outline-none transition placeholder:text-emerald-400 focus:border-emerald-500 focus:bg-white focus:ring-2 focus:ring-emerald-200 disabled:cursor-not-allowed disabled:opacity-55"
                  />
                </div>
              </div>
            );
          })}
          <div className="flex flex-wrap items-center justify-between gap-3 rounded-2xl border border-emerald-100 bg-white/80 px-3 py-3">
            <p className="text-xs font-medium text-emerald-800">
              {selectedCount > 0
                ? `${selectedCount}件選択中です。内容を確認して実行してください。`
                : "候補選択または自由入力をすると、ここからまとめて送信できます。"}
            </p>
            <button
              type="button"
              disabled={disabled || selectedCount === 0}
              onClick={onQuestionExecute}
              className="rounded-full border border-emerald-700 bg-emerald-700 px-4 py-2 text-sm font-medium text-white transition hover:bg-emerald-800 disabled:cursor-not-allowed disabled:border-emerald-200 disabled:bg-emerald-100 disabled:text-emerald-400"
            >
              実行する
            </button>
          </div>
        </div>
      </section>
    );
  }

  if (block.type === "cards") {
    const items = (block.content.items as Array<Record<string, unknown>>) ?? [];
    const compareEnabled = block.content.compare_enabled !== false;
    const selectedCount = items.filter((item) => Boolean(item.compare_selected)).length;
    const tone = TONES.cards;
    return (
      <section
        className={`overflow-hidden rounded-2xl border ${tone.border} ${tone.surface} shadow-card`}
      >
        <SectionHeader block={block} count={items.length} />
        <div className="space-y-3 p-4">
          <div className="grid gap-3 sm:grid-cols-2">
            {items.map((item, idx) => (
              <PropertyCard
                key={toDisplayText(item.id) || `card-${idx}`}
                item={item}
                index={idx}
                disabled={disabled}
                compareEnabled={compareEnabled}
                onCompareToggle={() => onCompareToggle?.(idx)}
                onActionExecute={onActionExecute}
              />
            ))}
          </div>
          {compareEnabled && items.length > 0 && (
            <div className="flex flex-wrap items-center justify-between gap-3 rounded-2xl border border-cyan-100 bg-cyan-50/70 px-3 py-3">
              <p className="text-xs font-medium text-cyan-900">
                {selectedCount >= 2
                  ? `${selectedCount}件を選択中です。まとめて比較できます。`
                  : "比較したい候補を2件以上選ぶと、まとめて比較できます。"}
              </p>
              <button
                type="button"
                disabled={disabled || selectedCount < 2}
                onClick={onCompareExecute}
                className="rounded-full border border-cyan-700 bg-cyan-700 px-4 py-2 text-sm font-medium text-white transition hover:bg-cyan-800 disabled:cursor-not-allowed disabled:border-cyan-200 disabled:bg-cyan-100 disabled:text-cyan-400"
              >
                選択中の物件を比較
              </button>
            </div>
          )}
        </div>
      </section>
    );
  }

  if (block.type === "table") {
    const columns = (block.content.columns as string[]) ?? [];
    const rows = (block.content.rows as Array<Record<string, unknown>>) ?? [];
    const tone = TONES.table;
    return (
      <section
        className={`overflow-hidden rounded-2xl border ${tone.border} ${tone.surface} shadow-card`}
      >
        <SectionHeader block={block} count={rows.length} />
        <div className="overflow-x-auto">
          <table className="min-w-full text-xs">
            <thead className="bg-blue-50/60 text-[11px] uppercase tracking-wider text-blue-700">
              <tr>
                {columns.map((col) => (
                  <th key={col} className="px-4 py-2.5 text-left font-semibold">
                    {col}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 text-ink">
              {rows.map((row, idx) => (
                <tr key={idx} className="transition hover:bg-blue-50/40">
                  {columns.map((col) => (
                    <td key={col} className="whitespace-nowrap px-4 py-2.5">
                      {toDisplayText(row[col])}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    );
  }

  if (block.type === "checklist") {
    const items = (block.content.items as Array<{ label: string; checked: boolean }>) ?? [];
    const doneCount = items.filter((i) => i.checked).length;
    const tone = TONES.checklist;
    return (
      <section
        className={`overflow-hidden rounded-2xl border ${tone.border} ${tone.surface} shadow-card`}
      >
        <SectionHeader block={block} count={items.length} />
        {items.length > 0 && (
          <div className="border-b border-sky-100 bg-sky-50/50 px-4 py-2 text-[11px] font-semibold text-sky-700">
            完了 {doneCount} / {items.length}
          </div>
        )}
        <ul className="divide-y divide-slate-100 text-sm">
          {items.map((item, idx) => (
            <li key={idx}>
              <button
                type="button"
                aria-pressed={item.checked}
                disabled={disabled}
                onClick={() => onChecklistToggle?.(idx)}
                className="flex w-full items-start gap-3 px-4 py-2.5 text-left transition hover:bg-sky-50/60 disabled:cursor-not-allowed disabled:opacity-60"
              >
                <span
                  className={`mt-0.5 inline-flex h-5 w-5 shrink-0 items-center justify-center rounded-md transition ${
                    item.checked
                      ? "bg-accent text-white shadow-sm"
                      : "border border-hairline bg-white text-transparent"
                  }`}
                >
                  <CheckmarkIcon className="h-3 w-3" />
                </span>
                <span className={`leading-6 ${item.checked ? "text-ink" : "text-inkMuted"}`}>
                  {item.label}
                </span>
              </button>
            </li>
          ))}
        </ul>
      </section>
    );
  }

  if (block.type === "actions") {
    const items = (block.content.items as ActionDescriptor[]) ?? [];
    const tone = TONES.actions;
    return (
      <section
        className={`overflow-hidden rounded-2xl border ${tone.border} ${tone.surface} shadow-card`}
      >
        <SectionHeader block={block} count={items.length} />
        <div className="space-y-2 p-4">
          {items.map((item, idx) => (
            <button
              key={`${item.action_type}-${idx}`}
              type="button"
              disabled={disabled}
              onClick={() => onActionExecute?.(item)}
              className="flex w-full items-center justify-between rounded-2xl border border-violet-200 bg-white px-4 py-3 text-left text-sm font-medium text-violet-900 shadow-sm transition hover:border-violet-300 hover:bg-violet-50 disabled:cursor-not-allowed disabled:border-violet-100 disabled:bg-violet-50/50 disabled:text-violet-400"
            >
              <span>{item.label}</span>
              <ActionIcon className="h-4 w-4 text-violet-600" />
            </button>
          ))}
        </div>
      </section>
    );
  }

  return null;
}
