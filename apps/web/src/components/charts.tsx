/**
 * Small charts, drawn as inline SVG.
 *
 * No charting library. Three charts that each draw one series do not justify
 * 200KB of runtime and a second way of theming things — and a dependency here
 * would have to be kept in step with React's release cadence forever. SVG
 * inherits the page's colours through `currentColor` and CSS variables, so
 * these follow the dark-mode toggle without being told about it.
 *
 * Every chart renders nothing when it has no data, rather than an empty frame
 * with axes. An axis with no line on it reads as a broken component.
 */

"use client";

type Point = { label: string; value: number };

const PAD = { top: 8, right: 8, bottom: 22, left: 34 };

function niceBounds(values: number[], zeroBased: boolean) {
  const max = Math.max(...values);
  const min = zeroBased ? 0 : Math.min(...values);
  if (max === min) return { min: min - 0.5, max: max + 0.5 };
  const headroom = (max - min) * 0.12;
  return { min: zeroBased ? 0 : min - headroom, max: max + headroom };
}

/**
 * A metric over successive runs.
 *
 * `floor` draws the gate threshold, because the interesting question about a
 * quality metric is never its absolute value — it is how much room is left
 * before the build stops going out.
 */
export function LineChart({
  points,
  height = 160,
  floor,
  format = (v: number) => v.toFixed(3),
}: {
  points: Point[];
  height?: number;
  floor?: number;
  format?: (v: number) => string;
}) {
  const first = points[0];
  const last = points[points.length - 1];
  if (points.length < 2 || !first || !last) return null;

  const width = 520;
  const values = points.map((p) => p.value);
  const { min, max } = niceBounds(floor === undefined ? values : [...values, floor], false);

  const innerW = width - PAD.left - PAD.right;
  const innerH = height - PAD.top - PAD.bottom;
  const x = (i: number) => PAD.left + (i / (points.length - 1)) * innerW;
  const y = (v: number) => PAD.top + innerH - ((v - min) / (max - min)) * innerH;

  const path = points.map((p, i) => `${i === 0 ? "M" : "L"} ${x(i)} ${y(p.value)}`).join(" ");
  const area = `${path} L ${x(points.length - 1)} ${PAD.top + innerH} L ${x(0)} ${PAD.top + innerH} Z`;

  return (
    <svg
      viewBox={`0 0 ${width} ${height}`}
      className="w-full"
      role="img"
      aria-label={`Trend from ${format(first.value)} to ${format(last.value)}`}
    >
      {[0, 0.5, 1].map((t) => (
        <line
          key={t}
          x1={PAD.left}
          x2={width - PAD.right}
          y1={PAD.top + innerH * t}
          y2={PAD.top + innerH * t}
          className="stroke-line"
          strokeWidth={1}
        />
      ))}

      {floor !== undefined && floor >= min && floor <= max && (
        <>
          <line
            x1={PAD.left}
            x2={width - PAD.right}
            y1={y(floor)}
            y2={y(floor)}
            className="stroke-red-500/70"
            strokeWidth={1}
            strokeDasharray="4 3"
          />
          <text x={width - PAD.right} y={y(floor) - 4} textAnchor="end" className="fill-red-500 text-[9px]">
            gate floor {format(floor)}
          </text>
        </>
      )}

      <path d={area} className="fill-accent/10" />
      <path d={path} className="stroke-accent" strokeWidth={2} fill="none" strokeLinejoin="round" />

      {points.map((p, i) => (
        <circle key={p.label} cx={x(i)} cy={y(p.value)} r={3} className="fill-accent">
          <title>{`${p.label}: ${format(p.value)}`}</title>
        </circle>
      ))}

      <text x={2} y={PAD.top + 4} className="fill-muted text-[9px]">{format(max)}</text>
      <text x={2} y={PAD.top + innerH} className="fill-muted text-[9px]">{format(min)}</text>
      <text x={PAD.left} y={height - 6} className="fill-muted text-[9px]">{first.label}</text>
      <text x={width - PAD.right} y={height - 6} textAnchor="end" className="fill-muted text-[9px]">
        {last.label}
      </text>
    </svg>
  );
}

/** Bars, for a value per day. */
export function BarChart({
  points,
  height = 150,
  format = (v: number) => v.toFixed(2),
}: {
  points: Point[];
  height?: number;
  format?: (v: number) => string;
}) {
  const first = points[0];
  const last = points[points.length - 1];
  if (!first || !last) return null;

  const width = 520;
  const { max } = niceBounds(points.map((p) => p.value), true);
  const innerW = width - PAD.left - PAD.right;
  const innerH = height - PAD.top - PAD.bottom;
  const slot = innerW / points.length;
  const barW = Math.max(2, slot * 0.62);

  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="w-full" role="img" aria-label="Daily totals">
      <line
        x1={PAD.left}
        x2={width - PAD.right}
        y1={PAD.top + innerH}
        y2={PAD.top + innerH}
        className="stroke-line"
        strokeWidth={1}
      />
      {points.map((p, i) => {
        const h = max === 0 ? 0 : (p.value / max) * innerH;
        return (
          <rect
            key={p.label}
            x={PAD.left + i * slot + (slot - barW) / 2}
            y={PAD.top + innerH - h}
            width={barW}
            height={h}
            rx={2}
            className="fill-accent/75"
          >
            <title>{`${p.label}: ${format(p.value)}`}</title>
          </rect>
        );
      })}
      <text x={2} y={PAD.top + 6} className="fill-muted text-[9px]">{format(max)}</text>
      <text x={PAD.left} y={height - 6} className="fill-muted text-[9px]">{first.label}</text>
      <text x={width - PAD.right} y={height - 6} textAnchor="end" className="fill-muted text-[9px]">
        {last.label}
      </text>
    </svg>
  );
}

/**
 * A reliability diagram: stated confidence against observed accuracy.
 *
 * The diagonal is perfect calibration. Points below it are a judge claiming
 * more certainty than it earns, which is the failure mode that matters —
 * an overconfident judge is trusted more than it should be, and its score is
 * what gates the deployment.
 */
export function ReliabilityDiagram({
  bins,
  size = 190,
  label,
}: {
  bins: { confidence: number; accuracy: number; n: number }[];
  size?: number;
  label?: string;
}) {
  if (bins.length === 0) return null;

  const pad = 26;
  const inner = size - pad * 2;
  const x = (v: number) => pad + v * inner;
  const y = (v: number) => pad + inner - v * inner;
  const maxN = Math.max(...bins.map((b) => b.n));

  return (
    <svg
      viewBox={`0 0 ${size} ${size}`}
      className="w-full max-w-[190px]"
      role="img"
      aria-label={`Reliability diagram${label ? ` for ${label}` : ""}`}
    >
      <rect x={pad} y={pad} width={inner} height={inner} className="fill-none stroke-line" strokeWidth={1} />
      <line x1={x(0)} y1={y(0)} x2={x(1)} y2={y(1)} className="stroke-muted/50" strokeWidth={1} strokeDasharray="3 3" />

      <path
        d={bins.map((b, i) => `${i === 0 ? "M" : "L"} ${x(b.confidence)} ${y(b.accuracy)}`).join(" ")}
        className="stroke-accent"
        strokeWidth={1.5}
        fill="none"
      />
      {bins.map((b) => (
        <circle
          key={b.confidence}
          cx={x(b.confidence)}
          cy={y(b.accuracy)}
          r={3 + (b.n / maxN) * 2.5}
          className="fill-accent"
        >
          <title>{`confidence ${b.confidence.toFixed(2)} → accuracy ${b.accuracy.toFixed(2)} (n=${b.n})`}</title>
        </circle>
      ))}

      <text x={pad + inner / 2} y={size - 6} textAnchor="middle" className="fill-muted text-[9px]">
        stated confidence
      </text>
      <text
        x={10}
        y={pad + inner / 2}
        textAnchor="middle"
        transform={`rotate(-90 10 ${pad + inner / 2})`}
        className="fill-muted text-[9px]"
      >
        observed accuracy
      </text>
    </svg>
  );
}
