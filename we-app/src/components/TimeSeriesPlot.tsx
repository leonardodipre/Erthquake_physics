import Plot from "react-plotly.js";
import type { StationTimeseries } from "../lib/types";

interface TimeSeriesPlotProps {
  series: StationTimeseries;
  useCleaned: boolean;
  normalized: boolean;
  rangeStart: string;
  rangeEnd: string;
}

function normalize(values: Array<number | null>) {
  const first = values.find((value): value is number => typeof value === "number");
  if (first === undefined) {
    return values;
  }
  return values.map((value) => (typeof value === "number" ? value - first : null));
}

export function TimeSeriesPlot({
  series,
  useCleaned,
  normalized,
  rangeStart,
  rangeEnd,
}: TimeSeriesPlotProps) {
  const source = useCleaned && series.has_cleaned ? series.cleaned : series.raw;
  const filteredIndices = series.dates.reduce<number[]>((accumulator, date, index) => {
    const time = new Date(date).getTime();
    if (time >= new Date(rangeStart).getTime() && time <= new Date(rangeEnd).getTime()) {
      accumulator.push(index);
    }
    return accumulator;
  }, []);

  const dates = filteredIndices.map((index) => series.dates[index]);
  const east = filteredIndices.map((index) => source.E[index]);
  const north = filteredIndices.map((index) => source.N[index]);
  const up = filteredIndices.map((index) => source.U[index]);

  const traces = [
    {
      x: dates,
      y: normalized ? normalize(east) : east,
      type: "scattergl",
      mode: "lines",
      name: "East",
      line: { color: "#1d4ed8", width: 1.8 },
    },
    {
      x: dates,
      y: normalized ? normalize(north) : north,
      type: "scattergl",
      mode: "lines",
      name: "North",
      line: { color: "#e76f51", width: 1.8 },
    },
    {
      x: dates,
      y: normalized ? normalize(up) : up,
      type: "scattergl",
      mode: "lines",
      name: "Up",
      line: { color: "#2a9d8f", width: 1.8 },
    },
  ];

  return (
    <Plot
      data={traces}
      layout={{
        autosize: true,
        margin: { l: 56, r: 20, t: 32, b: 42 },
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "#f4efe4",
        font: { family: "IBM Plex Sans, Avenir Next, Segoe UI, sans-serif", color: "#1f2937" },
        xaxis: {
          title: "Date",
          gridcolor: "#d4cfc2",
          zeroline: false,
        },
        yaxis: {
          title: normalized ? "Displacement relative to first point [m]" : "Displacement [m]",
          gridcolor: "#d4cfc2",
          zerolinecolor: "#58626d",
        },
        legend: {
          orientation: "h",
          x: 0,
          y: 1.14,
        },
      }}
      config={{
        displaylogo: false,
        responsive: true,
        modeBarButtonsToRemove: ["lasso2d", "select2d", "autoScale2d"],
      }}
      style={{ width: "100%", height: "100%" }}
      useResizeHandler
    />
  );
}
