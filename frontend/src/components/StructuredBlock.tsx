import type { ComponentType, SVGProps } from "react";

import type { ActionDescriptor, UIBlock } from "../lib/api";

type Props = {
  block: UIBlock;
  disabled?: boolean;
  onCompareExecute?: () => void;
  onCompareToggle?: (itemIndex: number) => void;
  onChecklistToggle?: (itemIndex: number) => void;
  onQuestionExecute?: () => void;
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

function TableIcon({ className = "h-4 w-4" }: IconProps) {
  return (
    <svg viewBox="0 0 20 20" fill="none" aria-hidden="true" className={className}>
      <rect x="3" y="4" width="14" height="12" rx="1.6" stroke="currentColor" strokeWidth="1.7" />
      <path
        d="M3 8.5H17M3 12.5H17M8.5 4V16M13 4V16"
        stroke="currentColor"
        strokeWidth="1.4"
      />
    </svg>
  );
}

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

function PlanIcon({ className = "h-4 w-4" }: IconProps) {
  return (
    <svg viewBox="0 0 20 20" fill="none" aria-hidden="true" className={className}>
      <path
        d="M5 4.5H15M5 8.5H12.5M5 12.5H11"
        stroke="currentColor"
        strokeWidth="1.7"
        strokeLinecap="round"
      />
      <rect x="3" y="2.8" width="14" height="14.4" rx="2.2" stroke="currentColor" strokeWidth="1.5" />
    </svg>
  );
}

function TimelineIcon({ className = "h-4 w-4" }: IconProps) {
  return (
    <svg viewBox="0 0 20 20" fill="none" aria-hidden="true" className={className}>
      <path d="M6 4V16M6 6H14M6 10H14M6 14H11" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" />
      <circle cx="6" cy="4" r="1.4" fill="currentColor" />
      <circle cx="6" cy="10" r="1.4" fill="currentColor" />
      <circle cx="6" cy="16" r="1.4" fill="currentColor" />
    </svg>
  );
}

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

function StarIcon({ className = "h-3.5 w-3.5" }: IconProps) {
  return (
    <svg viewBox="0 0 20 20" fill="currentColor" aria-hidden="true" className={className}>
      <path d="M10 2.2L12.36 7.06L17.6 7.84L13.78 11.62L14.72 16.96L10 14.4L5.28 16.96L6.22 11.62L2.4 7.84L7.64 7.06L10 2.2Z" />
    </svg>
  );
}

function PlusCircleIcon({ className = "h-3.5 w-3.5" }: IconProps) {
  return (
    <svg viewBox="0 0 20 20" fill="none" aria-hidden="true" className={className}>
      <circle cx="10" cy="10" r="7.6" stroke="currentColor" strokeWidth="1.6" />
      <path
        d="M10 7V13M7 10H13"
        stroke="currentColor"
        strokeWidth="1.7"
        strokeLinecap="round"
      />
    </svg>
  );
}

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
  sources: SourceIcon,
};

/* ---------- Helpers ---------- */

function toDisplayText(value: unknown): string {
  if (value === null || value === undefined) {
    return "";
  }
  return String(value);
}

function toDisplayNumber(value: unknown): number | null {
  const num = Number(value);
  return Number.isFinite(num) ? num : null;
}

/* ---------- Section header ---------- */

function SectionHeader({
  block,
  count,
}: {
  block: UIBlock;
  count?: number;
}) {
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
        <p
          className={`text-[10px] font-semibold uppercase tracking-[0.14em] ${tone.accentLabel}`}
        >
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

function PropertyCard({
  item,
  index,
  disabled = false,
  onCompareToggle,
  onActionExecute,
}: {
  item: Record<string, unknown>;
  index: number;
  disabled?: boolean;
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
          {score && (
            <span className="inline-flex items-center gap-1 rounded-full bg-sky-100 px-2 py-1 text-[11px] font-semibold text-sky-700">
              <StarIcon className="h-3 w-3" />
              {score}
            </span>
          )}
          {reactionLabel && (
            <span className={`inline-flex rounded-full px-2 py-1 text-[11px] font-semibold ${reactionTone}`}>
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

/* ---------- Block dispatcher ---------- */

export default function StructuredBlock({
  block,
  disabled = false,
  onCompareExecute,
  onCompareToggle,
  onChecklistToggle,
  onQuestionExecute,
  onQuestionSuggestionToggle,
  onActionExecute,
}: Props) {
  if (block.type === "text") {
    const tone = TONES.text;
    return (
      <section className={`overflow-hidden rounded-2xl border ${tone.border} ${tone.surface} shadow-card`}>
        <SectionHeader block={block} />
        <div className="px-4 py-3.5">
          <p className="whitespace-pre-wrap text-[14px] leading-7 text-ink">
            {toDisplayText(block.content.body)}
          </p>
        </div>
      </section>
    );
  }

  if (block.type === "warning") {
    const tone = TONES.warning;
    return (
      <section className={`overflow-hidden rounded-2xl border ${tone.border} ${tone.surface} shadow-card`}>
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
    const searchQuery = toDisplayText(block.content.search_query);
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
      <section className={`overflow-hidden rounded-2xl border ${tone.border} ${tone.surface} shadow-card`}>
        <SectionHeader block={block} />
        <div className="space-y-4 px-4 py-3.5">
          {(summary || goal) && (
            <div className="rounded-2xl border border-indigo-100 bg-white/85 p-4">
              {summary && <p className="text-sm font-semibold text-indigo-900">{summary}</p>}
              {goal && <p className="mt-2 text-[13px] leading-6 text-inkMuted">{goal}</p>}
            </div>
          )}

          {conditions.length > 0 && (
            <div className="rounded-2xl border border-indigo-100 bg-white/85 p-4">
              <p className="text-[11px] font-semibold uppercase tracking-[0.14em] text-indigo-600">
                条件
              </p>
              <dl className="mt-3 grid gap-2 sm:grid-cols-2">
                {conditions.map((condition, idx) => (
                  <div key={`${toDisplayText(condition.label)}-${idx}`} className="rounded-xl bg-indigo-50/70 px-3 py-2">
                    <dt className="text-[11px] font-semibold text-indigo-700">{toDisplayText(condition.label)}</dt>
                    <dd className="mt-1 text-sm text-ink">{toDisplayText(condition.value)}</dd>
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

          {(searchQuery || openQuestions.length > 0) && (
            <div className="grid gap-3 sm:grid-cols-2">
              {searchQuery && (
                <div className="rounded-2xl border border-indigo-100 bg-white/85 p-4">
                  <p className="text-[11px] font-semibold uppercase tracking-[0.14em] text-indigo-600">
                    初期クエリ
                  </p>
                  <p className="mt-2 text-sm leading-6 text-ink">{searchQuery}</p>
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
    const progress = Math.max(0, Math.min(100, toDisplayNumber(block.content.progress_percent) ?? 0));
    const currentStage = toDisplayText(block.content.current_stage);
    const summary = toDisplayText(block.content.summary);
    const items = Array.isArray(block.content.items)
      ? (block.content.items as Array<Record<string, unknown>>)
      : [];

    return (
      <section className={`overflow-hidden rounded-2xl border ${tone.border} ${tone.surface} shadow-card`}>
        <SectionHeader block={block} />
        <div className="space-y-4 px-4 py-3.5">
          <div className="rounded-2xl border border-slate-200 bg-white/90 p-4">
            <div className="flex items-center justify-between gap-3">
              <div>
                <p className="text-[11px] font-semibold uppercase tracking-[0.14em] text-slate-500">
                  Current Stage
                </p>
                <p className="mt-1 text-sm font-semibold text-slate-900">{currentStage || "待機中"}</p>
              </div>
              <span className="rounded-full bg-slate-100 px-2.5 py-1 text-xs font-semibold text-slate-700">
                {progress}%
              </span>
            </div>
            <div className="mt-3 h-2 overflow-hidden rounded-full bg-slate-100">
              <div
                className="h-full rounded-full bg-[linear-gradient(90deg,#0f766e_0%,#0ea5e9_100%)] transition-all duration-500"
                style={{ width: `${progress}%` }}
              />
            </div>
            {summary && <p className="mt-3 text-sm leading-6 text-slate-700">{summary}</p>}
          </div>

          {items.length > 0 && (
            <div className="space-y-2">
              {items.map((item, idx) => {
                const itemStatus = toDisplayText(item.status);
                const badgeClass =
                  itemStatus === "completed"
                    ? "bg-emerald-100 text-emerald-800"
                    : itemStatus === "running"
                      ? "bg-sky-100 text-sky-800"
                      : itemStatus === "failed"
                        ? "bg-rose-100 text-rose-800"
                        : "bg-slate-100 text-slate-700";
                return (
                  <div key={`${toDisplayText(item.label)}-${idx}`} className="rounded-2xl border border-slate-200 bg-white/90 p-3">
                    <div className="flex items-center justify-between gap-3">
                      <p className="text-sm font-medium text-ink">{toDisplayText(item.label)}</p>
                      <span className={`rounded-full px-2.5 py-1 text-[11px] font-semibold ${badgeClass}`}>
                        {itemStatus || "pending"}
                      </span>
                    </div>
                    {toDisplayText(item.detail) && (
                      <p className="mt-2 text-[13px] leading-6 text-inkMuted">{toDisplayText(item.detail)}</p>
                    )}
                  </div>
                );
              })}
            </div>
          )}
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
      <section className={`overflow-hidden rounded-2xl border ${tone.border} ${tone.surface} shadow-card`}>
        <SectionHeader block={block} count={items.length} />
        <div className="space-y-3 p-4">
          {items.map((item, idx) => {
            const queries = Array.isArray(item.queries)
              ? item.queries.map((query) => toDisplayText(query)).filter(Boolean)
              : [];
            const url = toDisplayText(item.url);
            return (
              <article key={`${toDisplayText(item.title)}-${idx}`} className="rounded-2xl border border-teal-100 bg-white/90 p-4">
                <div className="flex items-start justify-between gap-3">
                  <div className="min-w-0 flex-1">
                    <p className="text-sm font-semibold text-teal-900">{toDisplayText(item.title)}</p>
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
                  <p className="mt-3 text-[13px] leading-6 text-ink">{toDisplayText(item.reason)}</p>
                )}

                {queries.length > 0 && (
                  <div className="mt-3 flex flex-wrap gap-2">
                    {queries.map((query) => (
                      <span key={query} className="rounded-full bg-teal-50 px-2 py-1 text-[11px] font-medium text-teal-700">
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
    const selectedCount = items.filter((item) => toDisplayText(item.selected_example)).length;
    const tone = TONES.question;
    return (
      <section className={`overflow-hidden rounded-2xl border ${tone.border} ${tone.surface} shadow-card`}>
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
                        aria-pressed={selectedExample === example}
                        disabled={disabled}
                        onClick={() => onQuestionSuggestionToggle?.(idx, example)}
                        className={`rounded-full border px-3 py-1.5 text-xs font-medium transition disabled:cursor-not-allowed disabled:opacity-55 ${
                          selectedExample === example
                            ? "border-emerald-700 bg-emerald-700 text-white shadow-sm"
                            : "border-emerald-200 bg-emerald-50 text-emerald-800 hover:border-emerald-300 hover:bg-emerald-100"
                        }`}
                      >
                        {example}
                      </button>
                    ))}
                  </div>
                )}
              </div>
            );
          })}
          <div className="flex flex-wrap items-center justify-between gap-3 rounded-2xl border border-emerald-100 bg-white/80 px-3 py-3">
            <p className="text-xs font-medium text-emerald-800">
              {selectedCount > 0
                ? `${selectedCount}件選択中です。内容を確認して実行してください。`
                : "候補を選択すると、ここからまとめて送信できます。"}
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
    const selectedCount = items.filter((item) => Boolean(item.compare_selected)).length;
    const tone = TONES.cards;
    return (
      <section className={`overflow-hidden rounded-2xl border ${tone.border} ${tone.surface} shadow-card`}>
        <SectionHeader block={block} count={items.length} />
        <div className="space-y-3 p-4">
          <div className="grid gap-3 sm:grid-cols-2">
            {items.map((item, idx) => (
              <PropertyCard
                key={toDisplayText(item.id) || `card-${idx}`}
                item={item}
                index={idx}
                disabled={disabled}
                onCompareToggle={() => onCompareToggle?.(idx)}
                onActionExecute={onActionExecute}
              />
            ))}
          </div>
          {items.length > 0 && (
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
      <section className={`overflow-hidden rounded-2xl border ${tone.border} ${tone.surface} shadow-card`}>
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
      <section className={`overflow-hidden rounded-2xl border ${tone.border} ${tone.surface} shadow-card`}>
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
      <section className={`overflow-hidden rounded-2xl border ${tone.border} ${tone.surface} shadow-card`}>
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
