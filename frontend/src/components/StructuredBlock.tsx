import type { UIBlock } from "../lib/api";

type Props = {
  block: UIBlock;
};

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

export default function StructuredBlock({ block }: Props) {
  if (block.type === "text") {
    return (
      <section className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm">
        <h3 className="mb-2 text-sm font-semibold text-accent">{block.title}</h3>
        <p className="whitespace-pre-wrap text-sm leading-relaxed text-ink">
          {toDisplayText(block.content.body)}
        </p>
      </section>
    );
  }

  if (block.type === "warning") {
    return (
      <section className="rounded-2xl border border-amber-200 bg-amber-50/80 p-4">
        <h3 className="mb-2 flex items-center gap-2 text-sm font-semibold text-amber-700">
          <span className="inline-flex h-5 w-5 items-center justify-center rounded-full bg-amber-200 text-[10px] font-bold text-amber-800">
            !
          </span>
          {block.title}
        </h3>
        <p className="whitespace-pre-wrap text-sm leading-relaxed text-amber-800">
          {toDisplayText(block.content.body)}
        </p>
      </section>
    );
  }

  if (block.type === "cards") {
    const items = (block.content.items as Array<Record<string, unknown>>) ?? [];
    return (
      <section className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm">
        <h3 className="mb-3 text-sm font-semibold text-accent">{block.title}</h3>
        <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
          {items.map((item, idx) => {
            const rent = toDisplayNumber(item.rent);
            return (
              <article
                key={toDisplayText(item.id) || `card-${idx}`}
                className="rounded-xl border border-slate-200 bg-slate-50/70 p-3"
              >
                <h4 className="mb-2 text-sm font-semibold text-ink">
                  {toDisplayText(item.title) || "候補物件"}
                </h4>

                <div className="space-y-1 text-xs text-inkMuted">
                  <p>スコア: {toDisplayText(item.score) || "-"}</p>
                  <p>家賃: {rent !== null ? `${rent.toLocaleString()}円` : "-"}</p>
                  <p>徒歩: {toDisplayText(item.station_walk_min) || "-"}分</p>
                </div>

                {toDisplayText(item.why_selected) && (
                  <p className="mt-2 rounded-lg bg-emerald-50 px-2 py-1 text-xs text-emerald-700">
                    {toDisplayText(item.why_selected)}
                  </p>
                )}

                {toDisplayText(item.why_not_selected) && (
                  <p className="mt-1 rounded-lg bg-amber-50 px-2 py-1 text-xs text-amber-700">
                    {toDisplayText(item.why_not_selected)}
                  </p>
                )}
              </article>
            );
          })}
        </div>
      </section>
    );
  }

  if (block.type === "table") {
    const columns = (block.content.columns as string[]) ?? [];
    const rows = (block.content.rows as Array<Record<string, unknown>>) ?? [];
    return (
      <section className="overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-sm">
        <h3 className="border-b border-slate-200 px-4 py-3 text-sm font-semibold text-accent">{block.title}</h3>
        <div className="overflow-x-auto">
          <table className="min-w-full text-xs">
            <thead className="bg-slate-100 text-inkMuted">
              <tr>
                {columns.map((col) => (
                  <th key={col} className="px-3 py-2 text-left font-semibold">
                    {col}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((row, idx) => (
                <tr
                  key={idx}
                  className={`border-t border-slate-100 text-ink ${idx % 2 === 0 ? "bg-white" : "bg-slate-50/50"}`}
                >
                  {columns.map((col) => (
                    <td key={col} className="px-3 py-2">
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
    return (
      <section className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm">
        <h3 className="mb-2 text-sm font-semibold text-accent">{block.title}</h3>
        <ul className="space-y-2 text-sm text-ink">
          {items.map((item, idx) => (
            <li key={idx} className="flex items-start gap-2 rounded-lg bg-slate-50/70 px-2 py-1.5">
              <span
                className={`mt-0.5 inline-flex h-4 w-4 items-center justify-center rounded-full text-[10px] ${
                  item.checked ? "bg-emerald-500 text-white" : "border border-slate-300 bg-white text-transparent"
                }`}
              >
                ✓
              </span>
              <span className={item.checked ? "text-ink" : "text-inkMuted"}>{item.label}</span>
            </li>
          ))}
        </ul>
      </section>
    );
  }

  return null;
}
