import type { SnapshotFieldMeta } from "./types";

type Rgb = [number, number, number];

const COLOR_SCALES: Record<string, string[]> = {
  viridis: ["#440154", "#414487", "#2a788e", "#22a884", "#7ad151", "#fde725"],
  plasma: ["#0d0887", "#6a00a8", "#b12a90", "#e16462", "#fca636", "#f0f921"],
  hot: ["#0b0b0b", "#651000", "#c53a00", "#f39c12", "#ffe780", "#fffdf3"],
  inferno: ["#000004", "#320a5e", "#781c6d", "#bc3754", "#ed6925", "#fbb41a", "#fcffa4"],
  magma: ["#000004", "#221150", "#5f187f", "#982d80", "#d3436e", "#f67e4b", "#fddb7a"],
  rdbu: ["#053061", "#2166ac", "#92c5de", "#f7f7f7", "#f4a582", "#b2182b", "#67001f"],
  blueorange: ["#163b7a", "#4e7fc0", "#9fc4e5", "#f3f1eb", "#f4c27a", "#d77227", "#7f2704"],
  ylorred: ["#fff7bc", "#fee391", "#fec44f", "#fe9929", "#ec7014", "#cc4c02", "#8c2d04"],
  ylgnbu: ["#ffffd9", "#c7e9b4", "#7fcdbb", "#41b6c4", "#1d91c0", "#225ea8", "#0c2c84"],
  cividis: ["#00204c", "#2b5c8a", "#6c8f93", "#a4b97a", "#d4d96d", "#fee838"],
};

function resolveScaleName(scaleName: string, accessible = false) {
  if (!accessible) {
    return scaleName;
  }

  const accessibleMap: Record<string, string> = {
    viridis: "cividis",
    plasma: "magma",
    hot: "inferno",
    inferno: "inferno",
    magma: "magma",
    rdbu: "blueorange",
    ylorred: "inferno",
    ylgnbu: "cividis",
    cividis: "cividis",
    blueorange: "blueorange",
  };
  return accessibleMap[scaleName] ?? scaleName;
}

function hexToRgb(hex: string): Rgb {
  const normalized = hex.replace("#", "");
  const value = Number.parseInt(normalized, 16);
  return [(value >> 16) & 255, (value >> 8) & 255, value & 255];
}

function rgbToHex([r, g, b]: Rgb) {
  return `#${[r, g, b]
    .map((component) => component.toString(16).padStart(2, "0"))
    .join("")}`;
}

function interpolate(a: number, b: number, t: number) {
  return a + (b - a) * t;
}

function sampleScale(scaleName: string, t: number, accessible = false) {
  const resolved = resolveScaleName(scaleName, accessible);
  const stops = COLOR_SCALES[resolved] ?? COLOR_SCALES.viridis;
  if (t <= 0) {
    return stops[0];
  }
  if (t >= 1) {
    return stops[stops.length - 1];
  }
  const scaled = t * (stops.length - 1);
  const lowIndex = Math.floor(scaled);
  const highIndex = Math.min(lowIndex + 1, stops.length - 1);
  const localT = scaled - lowIndex;
  const low = hexToRgb(stops[lowIndex]);
  const high = hexToRgb(stops[highIndex]);
  return rgbToHex([
    Math.round(interpolate(low[0], high[0], localT)),
    Math.round(interpolate(low[1], high[1], localT)),
    Math.round(interpolate(low[2], high[2], localT)),
  ]);
}

function normalizeValue(value: number, meta: SnapshotFieldMeta) {
  if (!Number.isFinite(value)) {
    return 0.5;
  }

  if (meta.scale === "log") {
    const epsilon = Math.max(meta.min, 1e-20);
    const clamped = Math.max(value, epsilon);
    const minLog = Math.log10(epsilon);
    const maxLog = Math.log10(Math.max(meta.max, epsilon * 10));
    return (Math.log10(clamped) - minLog) / Math.max(maxLog - minLog, 1e-12);
  }

  if (meta.scale === "symmetric") {
    const maxAbs = Math.max(Math.abs(meta.min), Math.abs(meta.max), 1e-12);
    return (value + maxAbs) / (2 * maxAbs);
  }

  return (value - meta.min) / Math.max(meta.max - meta.min, 1e-12);
}

export function colorForValue(value: number, meta: SnapshotFieldMeta, accessible = false) {
  return sampleScale(meta.color_map, Math.min(Math.max(normalizeValue(value, meta), 0), 1), accessible);
}

export function gradientForMeta(meta: SnapshotFieldMeta, accessible = false) {
  const stops = Array.from({ length: 7 }, (_, index) => {
    const offset = index / 6;
    return `${sampleScale(meta.color_map, offset, accessible)} ${Math.round(offset * 100)}%`;
  });
  return `linear-gradient(90deg, ${stops.join(", ")})`;
}
