import { gradientForMeta } from "../lib/colors";
import { formatScientific } from "../lib/format";
import type { SnapshotFieldMeta } from "../lib/types";

interface ColorLegendProps {
  meta: SnapshotFieldMeta;
  accessible?: boolean;
}

export function ColorLegend({ meta, accessible = false }: ColorLegendProps) {
  const leftLabel = meta.scale === "symmetric" ? `-${formatScientific(Math.max(Math.abs(meta.min), Math.abs(meta.max)))}` : formatScientific(meta.min);
  const rightLabel = meta.scale === "symmetric" ? formatScientific(Math.max(Math.abs(meta.min), Math.abs(meta.max))) : formatScientific(meta.max);

  return (
    <section className="legend-panel">
      <div className="legend-header">
        <div>
          <p className="eyebrow">Color mapping</p>
          <h3>{meta.label}</h3>
        </div>
        <span className="legend-units">{meta.units}</span>
      </div>
      <div className="legend-bar" style={{ backgroundImage: gradientForMeta(meta, accessible) }} />
      <div className="legend-labels">
        <span>{leftLabel}</span>
        <span>{meta.scale}</span>
        <span>{rightLabel}</span>
      </div>
    </section>
  );
}
