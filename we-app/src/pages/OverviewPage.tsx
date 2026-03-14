import { useDeferredValue, useEffect, useMemo, useState } from "react";
import { CircleMarker, GeoJSON, MapContainer, TileLayer, useMap } from "react-leaflet";
import L from "leaflet";
import { MetricCard } from "../components/MetricCard";
import { TimeSeriesPlot } from "../components/TimeSeriesPlot";
import {
  loadEvaluationMetrics,
  loadFaultGeoJSON,
  loadStations,
  loadStationTimeseries,
  loadTrainingHistory,
} from "../lib/data";
import { formatCompactNumber, formatDateLabel, overlapDateRanges } from "../lib/format";
import type {
  EvaluationMetricsData,
  StationEntry,
  StationsData,
  StationTimeseries,
  TrainingHistoryData,
} from "../lib/types";

const CATEGORY_COLORS: Record<string, string> = {
  accepted: "#2563eb",
  modified: "#b45309",
  acc_test: "#c2410c",
};

function FitToData({
  fault,
  stations,
}: {
  fault: GeoJSON.FeatureCollection | null;
  stations: StationEntry[];
}) {
  const map = useMap();

  useEffect(() => {
    const bounds = L.latLngBounds([]);

    if (fault) {
      bounds.extend(L.geoJSON(fault).getBounds());
    }

    stations.forEach((station) => {
      bounds.extend([station.latitude, station.longitude]);
    });

    if (bounds.isValid()) {
      map.fitBounds(bounds.pad(0.12), { animate: false });
    }
  }, [fault, map, stations]);

  return null;
}

export function OverviewPage() {
  const [stationsData, setStationsData] = useState<StationsData | null>(null);
  const [faultGeoJSON, setFaultGeoJSON] = useState<GeoJSON.FeatureCollection | null>(null);
  const [evaluation, setEvaluation] = useState<EvaluationMetricsData | null>(null);
  const [history, setHistory] = useState<TrainingHistoryData | null>(null);
  const [selectedStationId, setSelectedStationId] = useState<string>("AQUI00ITA");
  const [selectedSeries, setSelectedSeries] = useState<StationTimeseries | null>(null);
  const [seriesLoading, setSeriesLoading] = useState(false);
  const [search, setSearch] = useState("");
  const [rangeStart, setRangeStart] = useState("2020-01-01");
  const [rangeEnd, setRangeEnd] = useState("2025-03-25");
  const [useCleaned, setUseCleaned] = useState(true);
  const [normalized, setNormalized] = useState(false);
  const deferredSearch = useDeferredValue(search.trim().toUpperCase());

  useEffect(() => {
    void Promise.all([
      loadStations(),
      loadFaultGeoJSON(),
      loadEvaluationMetrics(),
      loadTrainingHistory(),
    ]).then(([stationsPayload, faultPayload, evaluationPayload, historyPayload]) => {
      setStationsData(stationsPayload);
      setFaultGeoJSON(faultPayload);
      setEvaluation(evaluationPayload);
      setHistory(historyPayload);
      if (!selectedStationId && stationsPayload.stations.length > 0) {
        setSelectedStationId(stationsPayload.stations[0].station_id);
      }
    });
  }, []);

  useEffect(() => {
    if (!selectedStationId) {
      return;
    }
    setSeriesLoading(true);
    void loadStationTimeseries(selectedStationId)
      .then((payload) => {
        setSelectedSeries(payload);
      })
      .finally(() => {
        setSeriesLoading(false);
      });
  }, [selectedStationId]);

  const filteredStations = useMemo(() => {
    if (!stationsData) {
      return [];
    }
    return stationsData.stations.filter((station) => {
      const matchesSearch =
        deferredSearch.length === 0 ||
        station.station_id.includes(deferredSearch) ||
        station.label.toUpperCase().includes(deferredSearch);
      const matchesRange = overlapDateRanges(
        station.available_date_range.start,
        station.available_date_range.end,
        rangeStart,
        rangeEnd,
      );
      return matchesSearch && matchesRange;
    });
  }, [deferredSearch, rangeEnd, rangeStart, stationsData]);

  const selectedStation = stationsData?.stations.find(
    (station) => station.station_id === selectedStationId,
  );

  return (
    <section className="page-grid overview-grid">
      <div className="hero-panel rise">
        <div className="hero-copy">
          <p className="eyebrow">2D overview</p>
          <h2>Fault trace, station coverage, and observed deformation</h2>
          <p>
            The map uses the real fault GeoJSON, repository station series, and station coordinates
            resolved from the local metadata JSON when available.
          </p>
        </div>

        <div className="metric-grid">
          <MetricCard
            label="Modeled stations"
            value={stationsData ? formatCompactNumber(stationsData.meta.station_count, 0) : "…"}
            hint="accepted + modified + holdout"
          />
          <MetricCard
            label="Real coordinates"
            value={
              stationsData?.meta.real_coordinate_count != null
                ? formatCompactNumber(stationsData.meta.real_coordinate_count, 0)
                : "…"
            }
            hint={stationsData?.meta.coordinate_source ?? "station metadata source"}
          />
          <MetricCard
            label="Median RMSE"
            value={
              evaluation?.summary.median_rmse_mm != null
                ? `${formatCompactNumber(evaluation.summary.median_rmse_mm)} mm`
                : "…"
            }
            hint="position misfit across 2020-2025 evaluation dates"
          />
          <MetricCard
            label="Velocity RMSE"
            value={
              evaluation?.summary.median_velocity_rmse_mm_per_day != null
                ? `${formatCompactNumber(evaluation.summary.median_velocity_rmse_mm_per_day, 3)} mm/day`
                : "…"
            }
            hint="median daily velocity misfit"
          />
          <MetricCard
            label="Training loss"
            value={
              history?.summary.final_total_loss != null
                ? history.summary.final_total_loss.toExponential(3)
                : "…"
            }
            hint="final total loss in 8_V5_alldata_bigmodel.history.csv"
          />
        </div>
      </div>

      <aside className="panel rise controls-panel">
        <div className="panel-header">
          <div>
            <p className="eyebrow">Station filters</p>
            <h3>Search and time window</h3>
          </div>
          <span className="pill-muted">{filteredStations.length} visible</span>
        </div>

        <label className="field-group">
          <span>Station search</span>
          <input
            className="text-input"
            type="search"
            placeholder="AQUI00ITA, ACQU01ITA, ..."
            value={search}
            onChange={(event) => setSearch(event.target.value)}
          />
        </label>

        <div className="field-row">
          <label className="field-group">
            <span>Range start</span>
            <input
              className="text-input"
              type="date"
              value={rangeStart}
              onChange={(event) => setRangeStart(event.target.value)}
            />
          </label>
          <label className="field-group">
            <span>Range end</span>
            <input
              className="text-input"
              type="date"
              value={rangeEnd}
              onChange={(event) => setRangeEnd(event.target.value)}
            />
          </label>
        </div>

        <div className="category-legend">
          {Object.entries(CATEGORY_COLORS).map(([category, color]) => (
            <div key={category} className="legend-item">
              <span className="legend-swatch" style={{ backgroundColor: color }} />
              <span>{category}</span>
            </div>
          ))}
        </div>

        <div className="station-list">
          {filteredStations.slice(0, 60).map((station) => (
            <button
              key={station.station_id}
              type="button"
              className={
                station.station_id === selectedStationId ? "station-button station-button-active" : "station-button"
              }
              onClick={() => setSelectedStationId(station.station_id)}
            >
              <span>{station.station_id}</span>
              <small>{station.category}</small>
            </button>
          ))}
          {filteredStations.length > 60 ? (
            <p className="panel-note">Showing the first 60 matches. Narrow the filter to reduce clutter.</p>
          ) : null}
        </div>
      </aside>

      <section className="panel rise map-panel overview-map-panel">
        <div className="panel-header">
          <div>
            <p className="eyebrow">Regional map</p>
            <h3>Fault trace and GNSS markers</h3>
          </div>
          <span className="pill-muted">EPSG:4326 view</span>
        </div>

        <MapContainer className="map-view" center={[42.35, 13.4]} zoom={8} scrollWheelZoom>
          <TileLayer
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          />

          {faultGeoJSON ? (
            <GeoJSON
              data={faultGeoJSON}
              style={{
                color: "#d97706",
                weight: 3,
                opacity: 0.95,
                fillColor: "#f59e0b",
                fillOpacity: 0.12,
              }}
            />
          ) : null}

          {filteredStations.map((station) => (
            <CircleMarker
              key={station.station_id}
              center={[station.latitude, station.longitude]}
              radius={station.station_id === selectedStationId ? 8 : 5}
              pathOptions={{
                color: "#fbfdff",
                weight: station.station_id === selectedStationId ? 2 : 1,
                fillColor: CATEGORY_COLORS[station.category] ?? "#475569",
                fillOpacity: station.station_id === selectedStationId ? 1 : 0.85,
              }}
              eventHandlers={{
                click: () => setSelectedStationId(station.station_id),
              }}
            />
          ))}

          <FitToData fault={faultGeoJSON} stations={filteredStations} />
        </MapContainer>
      </section>

      <section className="panel rise station-panel overview-station-panel">
        <div className="panel-header">
          <div>
            <p className="eyebrow">Station detail</p>
            <h3>{selectedStation?.station_id ?? "Select a station"}</h3>
          </div>
          {selectedStation ? <span className="pill-muted">{selectedStation.category}</span> : null}
        </div>

        {selectedStation ? (
          <div className="station-meta-grid">
            <div>
              <span className="meta-label">Available range</span>
              <strong>
                {formatDateLabel(selectedStation.available_date_range.start)} to{" "}
                {formatDateLabel(selectedStation.available_date_range.end)}
              </strong>
            </div>
            <div>
              <span className="meta-label">Reference system</span>
              <strong>{selectedStation.reference_system}</strong>
            </div>
            <div>
              <span className="meta-label">Coordinate source</span>
              <strong>{selectedStation.coordinates_source}</strong>
            </div>
            <div>
              <span className="meta-label">Samples</span>
              <strong>{formatCompactNumber(selectedStation.sample_count, 0)}</strong>
            </div>
            <div>
              <span className="meta-label">Networks</span>
              <strong>{selectedStation.networks || "n/a"}</strong>
            </div>
            <div>
              <span className="meta-label">Distance from L'Aquila</span>
              <strong>
                {selectedStation.distance_km_from_laquila != null
                  ? `${formatCompactNumber(selectedStation.distance_km_from_laquila, 1)} km`
                  : "n/a"}
              </strong>
            </div>
          </div>
        ) : null}

        <div className="toggle-row">
          <button
            type="button"
            className={useCleaned ? "toggle-chip toggle-chip-active" : "toggle-chip"}
            onClick={() => setUseCleaned(true)}
          >
            Cleaned series
          </button>
          <button
            type="button"
            className={!useCleaned ? "toggle-chip toggle-chip-active" : "toggle-chip"}
            onClick={() => setUseCleaned(false)}
          >
            Raw series
          </button>
          <button
            type="button"
            className={normalized ? "toggle-chip toggle-chip-active" : "toggle-chip"}
            onClick={() => setNormalized((current) => !current)}
          >
            Normalize to first sample
          </button>
        </div>

        <div className="chart-shell">
          {selectedSeries && !seriesLoading ? (
            <TimeSeriesPlot
              series={selectedSeries}
              useCleaned={useCleaned}
              normalized={normalized}
              rangeStart={rangeStart}
              rangeEnd={rangeEnd}
            />
          ) : (
            <div className="loading-state">Loading time series…</div>
          )}
        </div>
      </section>
    </section>
  );
}
