import { DoubleSide } from "three";
import { Canvas } from "@react-three/fiber";
import { Bounds, Line, OrbitControls, Text } from "@react-three/drei";
import { useEffect, useMemo, useState, useTransition } from "react";
import { ColorLegend } from "../components/ColorLegend";
import { MetricCard } from "../components/MetricCard";
import {
  loadFaultPatches,
  loadModelCatalog,
  loadSnapshot,
  loadSnapshotIndexForModel,
  loadStationSensitivity,
  loadStations,
} from "../lib/data";
import { colorForValue } from "../lib/colors";
import { formatCompactNumber, formatDateLabel, formatScientific } from "../lib/format";
import type {
  FaultPatch,
  FaultPatchesData,
  ModelCatalog,
  ModelCatalogEntry,
  SnapshotData,
  SnapshotFieldMeta,
  SnapshotIndex,
  StationSensitivityRecord,
  StationsData,
} from "../lib/types";

const STATION_COLORS: Record<string, string> = {
  accepted: "#2563eb",
  modified: "#b45309",
  acc_test: "#c2410c",
};

interface FieldGuide {
  summary: string;
  snapshotBehavior: string;
  interpretationHint: string;
}

function fieldGuide(fieldKey: string): FieldGuide | null {
  switch (fieldKey) {
    case "gnss_observability_m_per_m":
      return {
        summary:
          "This map shows how strongly each patch is seen by the GNSS network through the Green matrix. It is a data-coverage field, not a physical state variable of the fault.",
        snapshotBehavior:
          "Changing snapshot should not materially change this map, because the geometry and station layout stay the same. If you move the time slider, the pattern should remain stable.",
        interpretationHint:
          "Bright areas are better constrained by the GNSS network. Darker areas are less observable, so any physical pattern there should be interpreted with more caution.",
      };
    case "weighted_gnss_observability_m_per_m":
      return {
        summary:
          "This is the same GNSS observability map, but each station is weighted by its training sensitivity for the selected model. It blends network geometry with model-specific station importance.",
        snapshotBehavior:
          "Changing snapshot should leave this map essentially unchanged. If you change model checkpoint, the weighting can change because the training sensitivity ranking can change.",
        interpretationHint:
          "Bright areas are where the fault is both visible to the GNSS network and emphasized by the training process. It is useful for checking whether a model pattern is concentrated where the model had the strongest data support.",
      };
    case "slip_m":
      return {
        summary:
          "This field is the total slip on each patch at the selected time. It represents the cumulative modeled displacement on the fault plane, in meters.",
        snapshotBehavior:
          "As you move through snapshots, you should usually see slip accumulate or reorganize progressively rather than flicker randomly. Persistent bright zones indicate where the model concentrates most of the fault motion over time.",
        interpretationHint:
          "Use this as the main physical state map. Compare it against observability to distinguish a strong physical signal from a region that is simply better constrained by nearby stations.",
      };
    case "delta_slip_m":
      return {
        summary:
          "This field shows how much slip changed relative to the first exported snapshot. It isolates the evolution across the chosen time window rather than the total accumulated value.",
        snapshotBehavior:
          "Early snapshots should be close to zero. Later snapshots show where slip has grown most strongly since the start of the exported sequence.",
        interpretationHint:
          "This is the best field for asking where change is happening in time. If a patch stays bright while snapshots advance, that area is carrying much of the evolving signal.",
      };
    case "supported_delta_slip_m":
      return {
        summary:
          "This field is the absolute delta slip multiplied by normalized GNSS support. It highlights where the model changes strongly and the network also sees the fault well.",
        snapshotBehavior:
          "As snapshots advance, bright areas should follow the growth of delta slip, but only where support remains high. It is a time-evolving map, reweighted by static data coverage.",
        interpretationHint:
          "This is a confidence-oriented view of change. Bright areas are easier to defend physically because they combine model evolution with stronger observational support.",
      };
    case "unsupported_delta_slip_m":
      return {
        summary:
          "This field is the absolute delta slip multiplied by one minus normalized GNSS support. It highlights where the model changes strongly in places that the GNSS network constrains more weakly.",
        snapshotBehavior:
          "As snapshots advance, bright areas indicate where model evolution is growing in less-observed parts of the fault plane. It is useful to watch whether these zones persist or appear only briefly.",
        interpretationHint:
          "Treat bright patches here as caution zones. They are not necessarily wrong, but they are areas where the interpretation depends more on model structure and less on direct support from the data.",
      };
    case "delta_slip_over_support_m":
      return {
        summary:
          "This field divides absolute delta slip by normalized GNSS support. It asks where the model is changing a lot compared with how strongly that area is actually observed.",
        snapshotBehavior:
          "As snapshots advance, rising values indicate patches where change grows faster than support would justify. It is especially useful for spotting potentially over-interpreted regions.",
        interpretationHint:
          "Bright areas are important warning signs: they may contain real physics, but they deserve extra scrutiny because the change is large relative to data support.",
      };
    case "weighted_supported_delta_slip_m":
      return {
        summary:
          "This field is absolute delta slip multiplied by the training-weighted support map. It highlights changes that line up with both GNSS coverage and the stations that mattered most during training.",
        snapshotBehavior:
          "Through time, bright areas should track evolving delta slip in the parts of the fault that are most favored by the selected model's training sensitivity structure.",
        interpretationHint:
          "Use it to see where the model is changing in areas that it is most predisposed to trust. It is helpful for separating model-supported evolution from more weakly grounded change.",
      };
    case "weighted_unsupported_delta_slip_m":
      return {
        summary:
          "This field highlights absolute delta slip in areas that stay weakly supported even after station training utility is taken into account.",
        snapshotBehavior:
          "If bright regions persist across snapshots, the model is repeatedly placing change where even its favored stations do not strongly constrain the patch.",
        interpretationHint:
          "This is one of the strongest caution views in the app. Bright areas are the first places to question when you want to know whether a pattern is being driven more by model assumptions than by usable data.",
      };
    case "delta_slip_over_weighted_support_m":
      return {
        summary:
          "This field divides absolute delta slip by training-weighted support. It highlights where the model changes more than the training-favored GNSS support would suggest.",
        snapshotBehavior:
          "As snapshots advance, brightening indicates that a patch is gaining change faster than the model's own preferred data support would normally justify.",
        interpretationHint:
          "Use this as a model-aware stress test: bright patches are candidates for over-interpretation relative to the stations that carried the most training information.",
      };
    case "slip_rate_m_per_s":
      return {
        summary:
          "This is the modeled slip rate V on each patch, in meters per second. It indicates where the fault is currently moving fastest at the selected time rather than how much it has moved in total.",
        snapshotBehavior:
          "This field can change much more abruptly than total slip. As you move through snapshots, look for migrating hot spots, pulses, or concentration of activity along strike and down dip.",
        interpretationHint:
          "Use it to locate active transient behavior. If a region is bright in slip rate but not yet in total slip, it may be a zone of recent or ongoing acceleration.",
      };
    case "theta_s":
      return {
        summary:
          "This is the RSF state variable theta, in seconds. It is part of the internal fault constitutive state and helps describe the local frictional condition of each patch.",
        snapshotBehavior:
          "Theta often evolves more smoothly than slip rate. Through snapshots, look for broad reorganizations rather than only sharp local peaks.",
        interpretationHint:
          "Interpret it together with slip rate and a_minus_b. Strong contrasts can indicate where the fault state is evolving toward more locked or more active behavior.",
      };
    case "a":
    case "b":
    case "D_c_m":
      return {
        summary:
          "This field is a constitutive parameter of the rate-and-state friction model. It describes a material property assigned by the model rather than a directly observed displacement quantity.",
        snapshotBehavior:
          "In most interpretations these parameter fields should change little or not at all with snapshot. If you see large time changes, that reflects how this particular model architecture represents the field.",
        interpretationHint:
          "Use these maps mainly to understand which parts of the fault are mechanically different in the model, not as direct evidence of deformation by themselves.",
      };
    case "a_minus_b":
      return {
        summary:
          "This field is the difference a - b, a standard RSF diagnostic. It is often used to separate areas with different frictional tendencies across the fault plane.",
        snapshotBehavior:
          "You should generally expect its spatial pattern to stay fairly stable with time. The main information is where the field is positive, negative, or close to zero.",
        interpretationHint:
          "This is best read structurally rather than dynamically. Compare it with slip rate or stress to see whether active zones line up with particular frictional regimes.",
      };
    case "tau_elastic_pa":
      return {
        summary:
          "This field is the elastic shear stress on each patch, in pascals. It reflects how the modeled fault loading and interactions redistribute stress over time.",
        snapshotBehavior:
          "As snapshots advance, look for stress accumulation, release, or transfer between neighboring zones. The spatial pattern is often smoother than slip rate.",
        interpretationHint:
          "Use it to understand loading context. A bright or dark area becomes more meaningful when compared with slip rate, theta, or residual fields.",
      };
    case "tau_rsf_pa":
      return {
        summary:
          "This field is the RSF shear resistance term on each patch, in pascals. It represents the modeled frictional resistance associated with the constitutive law.",
        snapshotBehavior:
          "Through time, it should evolve with the fault state and slip conditions. Compare it against elastic stress to understand whether a region is close to balance or mismatch.",
        interpretationHint:
          "This is useful when read together with tau_elastic and the residual fields. Alone, it tells you about resistance; in comparison, it tells you about imbalance.",
      };
    case "tau_residual_over_sigma_n":
      return {
        summary:
          "This field measures the mismatch between elastic stress and RSF resistance, normalized by normal stress. It is a signed residual diagnostic rather than a direct physical observation.",
        snapshotBehavior:
          "As snapshots change, watch whether residuals concentrate, spread, or flip sign. Persistent extremes can indicate where the model is under the most constitutive tension.",
        interpretationHint:
          "Because this is a symmetric diagnostic, the central neutral tone is as important as the extremes. Focus on where the residual is close to zero versus strongly positive or negative.",
      };
    case "aging_residual":
      return {
        summary:
          "This field is the residual of the aging-law state evolution. It indicates how closely the model output matches the expected constitutive evolution for theta.",
        snapshotBehavior:
          "If the model behaves consistently, the field should remain relatively subdued. Strong localized values or persistent structures can flag patches where the constitutive balance is under tension.",
        interpretationHint:
          "Read this as a quality-of-dynamics diagnostic rather than a direct geology map. Bright extremes highlight where the internal RSF balance is least quiet.",
      };
    default:
      return null;
  }
}

function colorReading(meta: SnapshotFieldMeta, accessibleColors: boolean) {
  const paletteNote = accessibleColors
    ? "The accessible palette preserves the same value ordering while using hues that are easier to separate under color-vision deficiency."
    : "The palette matches the scientific diagnostic style selected for this field.";

  if (meta.scale === "log") {
    return `Dark-to-bright colors represent increasing values on a logarithmic scale. Equal visual steps do not mean equal absolute increments: they indicate multiplicative or order-of-magnitude changes. ${paletteNote}`;
  }

  if (meta.scale === "symmetric") {
    return `The central neutral tone is near zero, while the two opposite ends represent negative and positive values. What matters most is the sign and distance from the center, not only which side looks brighter. ${paletteNote}`;
  }

  return `Dark-to-bright colors represent increasing values on a linear scale. A moderate visual change corresponds to a moderate numerical change within the legend range. ${paletteNote}`;
}

function FaultMesh({
  fault,
  fieldMeta,
  fieldValues,
  accessibleColors,
  depthExaggeration,
  selectedPatchId,
  hoveredPatchId,
  onHoverPatch,
  onSelectPatch,
}: {
  fault: FaultPatchesData;
  fieldMeta: SnapshotFieldMeta;
  fieldValues: number[];
  accessibleColors: boolean;
  depthExaggeration: number;
  selectedPatchId: number | null;
  hoveredPatchId: number | null;
  onHoverPatch: (patchId: number | null) => void;
  onSelectPatch: (patchId: number | null) => void;
}) {
  const positions = useMemo(() => {
    const values = fault.patches.flatMap((patch) =>
      patch.triangles_local_xyz_m.flatMap((triangle) =>
        triangle.flatMap(([x, y, z]) => [x, y, z]),
      ),
    );
    return new Float32Array(values);
  }, [fault.patches]);

  const colors = useMemo(() => {
    const packed: number[] = [];

    fault.patches.forEach((patch) => {
      const color = colorForValue(fieldValues[patch.id], fieldMeta, accessibleColors);
      const normalized = color
        .replace("#", "")
        .match(/.{2}/g)!
        .map((component) => Number.parseInt(component, 16) / 255);

      const activePatch = patch.id === selectedPatchId || patch.id === hoveredPatchId;
      const tint = activePatch ? 0.18 : 0;
      const [r, g, b] = normalized.map((channel) => Math.min(channel + tint, 1));
      for (let index = 0; index < 6; index += 1) {
        packed.push(r, g, b);
      }
    });

    return new Float32Array(packed);
  }, [accessibleColors, fault.patches, fieldMeta, fieldValues, hoveredPatchId, selectedPatchId]);

  const highlightedPatch = hoveredPatchId ?? selectedPatchId;
  const overlayPositions = useMemo(() => {
    if (highlightedPatch == null) {
      return null;
    }
    const patch = fault.patches[highlightedPatch];
    const values = patch.triangles_local_xyz_m.flatMap((triangle) =>
      triangle.flatMap(([x, y, z]) => [x, y, z]),
    );
    return new Float32Array(values);
  }, [fault.patches, highlightedPatch]);

  return (
    <group scale={[1, depthExaggeration, 1]}>
      <mesh
        onPointerMove={(event) => {
          event.stopPropagation();
          const patchId = Math.floor((event.faceIndex ?? 0) / 2);
          onHoverPatch(patchId);
        }}
        onPointerOut={() => onHoverPatch(null)}
        onClick={(event) => {
          event.stopPropagation();
          const patchId = Math.floor((event.faceIndex ?? 0) / 2);
          onSelectPatch(patchId);
        }}
      >
        <bufferGeometry>
          <bufferAttribute attach="attributes-position" args={[positions, 3]} />
          <bufferAttribute attach="attributes-color" args={[colors, 3]} />
        </bufferGeometry>
        <meshStandardMaterial vertexColors side={DoubleSide} roughness={0.4} metalness={0.05} />
      </mesh>

      {overlayPositions ? (
        <mesh>
          <bufferGeometry>
            <bufferAttribute attach="attributes-position" args={[overlayPositions, 3]} />
          </bufferGeometry>
          <meshBasicMaterial color="#f8fafc" wireframe transparent opacity={1} side={DoubleSide} />
        </mesh>
      ) : null}
    </group>
  );
}

function SpatialContext({
  fault,
  stations,
  stationSensitivity,
  showSensitivityBars,
}: {
  fault: FaultPatchesData;
  stations: StationsData;
  stationSensitivity: Map<string, StationSensitivityRecord>;
  showSensitivityBars: boolean;
}) {
  const reference = stations.meta.reference_point;
  const mainshock = stations.meta.mainshock;
  const stationPoints = stations.stations.filter((station) => station.local_xyz_m != null);
  const extents = useMemo(() => {
    const values: number[] = [];
    stationPoints.forEach((station) => {
      values.push(Math.abs(station.local_xyz_m![0]), Math.abs(station.local_xyz_m![2]));
    });
    fault.patches.forEach((patch) => {
      values.push(Math.abs(patch.center_local_xyz_m[0]), Math.abs(patch.center_local_xyz_m[2]));
    });
    if (mainshock) {
      values.push(Math.abs(mainshock.local_xyz_m[0]), Math.abs(mainshock.local_xyz_m[2]));
    }
    return Math.max(...values, 45000) * 2.3;
  }, [fault.patches, mainshock, stationPoints]);
  const maxSensitivity = useMemo(() => {
    let max = 0;
    stationSensitivity.forEach((item) => {
      max = Math.max(max, item.sensitivity);
    });
    return max;
  }, [stationSensitivity]);

  return (
    <group>
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0, 0]}>
        <planeGeometry args={[extents, extents]} />
        <meshStandardMaterial
          color="#86a7b8"
          transparent
          opacity={0.14}
          side={DoubleSide}
          roughness={0.95}
        />
      </mesh>

      {fault.surface_trace_local_xyz_m && fault.surface_trace_local_xyz_m.length > 1 ? (
        <Line
          points={fault.surface_trace_local_xyz_m.map(([x, _y, z]) => [x, 35, z])}
          color="#d97706"
          lineWidth={2.2}
        />
      ) : null}

      {stationPoints.map((station) => (
        <group key={station.station_id} position={[station.local_xyz_m![0], 0, station.local_xyz_m![2]]}>
          {showSensitivityBars && stationSensitivity.has(station.station_id) ? (
            <mesh position={[0, 250 + Math.max(0.16, Math.sqrt(stationSensitivity.get(station.station_id)!.sensitivity / Math.max(maxSensitivity, 1e-12))) * 3500, 0]}>
              <cylinderGeometry
                args={[
                  110,
                  150,
                  500 + Math.max(0.16, Math.sqrt(stationSensitivity.get(station.station_id)!.sensitivity / Math.max(maxSensitivity, 1e-12))) * 7000,
                  10,
                ]}
              />
              <meshStandardMaterial
                color="#d97706"
                emissive="#8a4b08"
                emissiveIntensity={0.18}
                transparent
                opacity={0.72}
              />
            </mesh>
          ) : null}

          <mesh position={[0, 180, 0]}>
            <sphereGeometry args={[190, 10, 10]} />
            <meshStandardMaterial
              color={STATION_COLORS[station.category] ?? "#475569"}
              transparent
              opacity={0.88}
            />
          </mesh>
        </group>
      ))}

      {reference ? (
        <group position={[reference.local_xyz_m[0], 0, reference.local_xyz_m[2]]}>
          <mesh position={[0, 320, 0]}>
            <sphereGeometry args={[320, 18, 18]} />
            <meshStandardMaterial color="#f4b942" emissive="#8a5b00" emissiveIntensity={0.35} />
          </mesh>
          <Line points={[[0, 0, 0], [0, 1800, 0]]} color="#f4b942" lineWidth={1.6} />
          <Text
            position={[0, 2350, 0]}
            fontSize={1100}
            color="#5c3a05"
            anchorX="center"
            anchorY="middle"
          >
            L'Aquila
          </Text>
        </group>
      ) : null}

      {mainshock ? (
        <>
          <group position={[mainshock.local_xyz_m[0], 0, mainshock.local_xyz_m[2]]}>
            <mesh position={[0, 120, 0]}>
              <sphereGeometry args={[220, 16, 16]} />
              <meshStandardMaterial color="#dc2626" emissive="#7f1d1d" emissiveIntensity={0.28} />
            </mesh>
            <Line points={[[0, 0, 0], [0, mainshock.local_xyz_m[1], 0]]} color="#dc2626" lineWidth={1.6} />
            <Text
              position={[0, 2350, 0]}
              fontSize={950}
              color="#7f1d1d"
              anchorX="center"
              anchorY="middle"
            >
              2009 mainshock
            </Text>
          </group>

          <group position={[mainshock.local_xyz_m[0], mainshock.local_xyz_m[1], mainshock.local_xyz_m[2]]}>
            <mesh>
              <sphereGeometry args={[260, 18, 18]} />
              <meshStandardMaterial color="#dc2626" emissive="#7f1d1d" emissiveIntensity={0.38} />
            </mesh>
          </group>
        </>
      ) : null}

      <group position={[-extents * 0.34, 0, -extents * 0.34]}>
        <Line points={[[0, 0, 0], [7000, 0, 0]]} color="#b91c1c" lineWidth={1.7} />
        <Line points={[[0, 0, 0], [0, 0, 7000]]} color="#1d4ed8" lineWidth={1.7} />
        <Line points={[[0, 0, 0], [0, 5000, 0]]} color="#475569" lineWidth={1.7} />
        <Text position={[7800, 500, 0]} fontSize={900} color="#7f1d1d">
          E
        </Text>
        <Text position={[0, 500, 7800]} fontSize={900} color="#1e3a8a">
          N
        </Text>
        <Text position={[0, 5600, 0]} fontSize={900} color="#334155">
          Up
        </Text>
      </group>
    </group>
  );
}

export function FaultModelPage() {
  const [fault, setFault] = useState<FaultPatchesData | null>(null);
  const [stations, setStations] = useState<StationsData | null>(null);
  const [modelCatalog, setModelCatalog] = useState<ModelCatalog | null>(null);
  const [snapshotIndex, setSnapshotIndex] = useState<SnapshotIndex | null>(null);
  const [snapshot, setSnapshot] = useState<SnapshotData | null>(null);
  const [stationSensitivity, setStationSensitivity] = useState<Map<string, StationSensitivityRecord>>(new Map());
  const [selectedModelKey, setSelectedModelKey] = useState("");
  const [selectedSnapshotIndex, setSelectedSnapshotIndex] = useState(0);
  const [fieldKey, setFieldKey] = useState("slip_m");
  const [depthExaggeration, setDepthExaggeration] = useState(1.6);
  const [selectedPatchId, setSelectedPatchId] = useState<number | null>(null);
  const [hoveredPatchId, setHoveredPatchId] = useState<number | null>(null);
  const [showSensitivityBars, setShowSensitivityBars] = useState(true);
  const [accessibleColors, setAccessibleColors] = useState(true);
  const [sceneResetKey, setSceneResetKey] = useState(0);
  const [isPending, startTransition] = useTransition();

  useEffect(() => {
    void Promise.all([loadFaultPatches(), loadModelCatalog(), loadStations()]).then(([faultPayload, modelCatalogPayload, stationsPayload]) => {
      setFault(faultPayload);
      setModelCatalog(modelCatalogPayload);
      setStations(stationsPayload);
      setSelectedModelKey(modelCatalogPayload.default_model_key);
    });
  }, []);

  useEffect(() => {
    if (!selectedModelKey) {
      return;
    }
    setSnapshot(null);
    setSnapshotIndex(null);
    void loadSnapshotIndexForModel(selectedModelKey).then((payload) => {
      setSnapshotIndex(payload);
      setSelectedSnapshotIndex(0);
      const availableFieldKeys = Object.keys(payload.fields);
      if (!availableFieldKeys.includes(fieldKey) && availableFieldKeys.length > 0) {
        setFieldKey(availableFieldKeys[0]);
      }
    });
  }, [selectedModelKey]);

  useEffect(() => {
    if (!snapshotIndex) {
      return;
    }
    const availableFieldKeys = Object.keys(snapshotIndex.fields);
    if (!availableFieldKeys.includes(fieldKey) && availableFieldKeys.length > 0) {
      setFieldKey(availableFieldKeys[0]);
    }
  }, [fieldKey, snapshotIndex]);

  useEffect(() => {
    if (!snapshotIndex || !selectedModelKey) {
      return;
    }
    const descriptor = snapshotIndex.snapshots[selectedSnapshotIndex];
    if (!descriptor) {
      return;
    }
    void loadSnapshot(descriptor.date_key, selectedModelKey).then((payload) => {
      setSnapshot(payload);
    });
  }, [selectedModelKey, selectedSnapshotIndex, snapshotIndex]);

  const selectedSnapshotDescriptor = snapshotIndex?.snapshots[selectedSnapshotIndex] ?? null;
  const selectedModel: ModelCatalogEntry | null =
    modelCatalog?.models.find((model) => model.key === selectedModelKey) ?? null;
  const fieldMeta = snapshotIndex?.fields[fieldKey] ?? null;
  const fieldValues =
    snapshot?.fields[fieldKey] ?? snapshotIndex?.static_fields?.[fieldKey] ?? null;
  const selectedFieldGuide = fieldMeta ? fieldGuide(fieldKey) : null;
  const isStaticField = Boolean(snapshotIndex?.static_fields?.[fieldKey]);
  const activePatchId = hoveredPatchId ?? selectedPatchId;
  const activePatch: FaultPatch | null = fault && activePatchId != null ? fault.patches[activePatchId] : null;
  const activeValue =
    activePatch && fieldValues ? fieldValues[activePatch.id] ?? null : null;

  useEffect(() => {
    if (!selectedModel?.station_sensitivity_path) {
      setStationSensitivity(new Map());
      return;
    }
    void loadStationSensitivity(selectedModel.station_sensitivity_path).then((payload) => {
      const mapping = new Map<string, StationSensitivityRecord>();
      payload.forEach((item) => {
        mapping.set(item.station, item);
      });
      setStationSensitivity(mapping);
    });
  }, [selectedModel]);

  return (
    <section className="page-grid page-grid-3d">
      <div className="hero-panel rise">
        <div className="hero-copy">
          <p className="eyebrow">3D fault model</p>
          <h2>Patch-resolved geometry and exported model state</h2>
          <p>
            The surface is reconstructed from the real triangle mesh in{" "}
            <code>green_out/fault_mesh.npz</code>. Scalar values come from JSON snapshots exported
            from the checkpoint you select below, so you can switch model and timestep without
            touching the browser-side geometry.
          </p>
        </div>

        <div className="metric-grid">
          <MetricCard
            label="Patch cells"
            value={fault ? formatCompactNumber(fault.meta.patch_count, 0) : "…"}
            hint={`${fault?.meta.grid.nx ?? "…"} along strike × ${fault?.meta.grid.ny ?? "…"} down dip`}
          />
          <MetricCard
            label="Model"
            value={selectedModel?.label ?? "…"}
            hint={selectedModel?.checkpoint ?? "exported checkpoint"}
          />
          <MetricCard
            label="Depth range"
            value={
              fault
                ? `${formatCompactNumber(fault.meta.fault_meta.top_depth_m / 1000, 1)}-${formatCompactNumber(
                    fault.meta.fault_meta.bottom_depth_m / 1000,
                    1,
                  )} km`
                : "…"
            }
            hint="top to bottom of modeled fault plane"
          />
          <MetricCard
            label="Snapshots"
            value={snapshotIndex ? formatCompactNumber(snapshotIndex.meta.snapshot_count, 0) : "…"}
            hint={
              selectedModel?.time_range.start && selectedModel?.time_range.end
                ? `${formatDateLabel(selectedModel.time_range.start)} to ${formatDateLabel(selectedModel.time_range.end)}`
                : "snapshot time range"
            }
          />
          <MetricCard
            label="Top training station"
            value={selectedModel?.station_sensitivity_summary?.top_station ?? "n/a"}
            hint={
              selectedModel?.station_sensitivity_summary?.max_sensitivity != null
                ? `max sensitivity ${formatScientific(selectedModel.station_sensitivity_summary.max_sensitivity, 2)}`
                : "no station sensitivity ranking for this model"
            }
          />
          <MetricCard
            label="Mainshock"
            value={
              stations?.meta.mainshock
                ? `Mw ${formatCompactNumber(stations.meta.mainshock.magnitude_mw, 1)}`
                : "…"
            }
            hint={
              stations?.meta.mainshock
                ? `${formatCompactNumber(stations.meta.mainshock.latitude, 4)} N, ${formatCompactNumber(stations.meta.mainshock.longitude, 4)} E, depth ${formatCompactNumber(stations.meta.mainshock.depth_m / 1000, 1)} km`
                : "2009 L'Aquila hypocenter"
            }
          />
        </div>
      </div>

      <aside className="panel rise controls-panel">
        <div className="panel-header">
          <div>
            <p className="eyebrow">Controls</p>
            <h3>Snapshots and scalar fields</h3>
          </div>
          {selectedSnapshotDescriptor ? (
            <span className="pill-muted">{formatDateLabel(selectedSnapshotDescriptor.date)}</span>
          ) : null}
        </div>

        <label className="field-group">
          <span>Model checkpoint</span>
          <select
            className="text-input"
            value={selectedModelKey}
            onChange={(event) => {
              setSelectedPatchId(null);
              setHoveredPatchId(null);
              setSelectedModelKey(event.target.value);
            }}
          >
            {modelCatalog?.models.map((model) => (
              <option key={model.key} value={model.key}>
                {model.label}
              </option>
            ))}
          </select>
        </label>

        <label className="field-group">
          <span>Field</span>
          <select
            className="text-input"
            value={fieldKey}
            onChange={(event) => setFieldKey(event.target.value)}
          >
            {snapshotIndex
              ? Object.entries(snapshotIndex.fields).map(([key, meta]) => (
                  <option key={key} value={key}>
                    {meta.label} [{meta.units}]
                  </option>
                ))
              : null}
          </select>
        </label>

        <label className="field-group">
          <span>Snapshot date</span>
          <select
            className="text-input"
            value={selectedSnapshotDescriptor?.date_key ?? ""}
            onChange={(event) => {
              const nextIndex =
                snapshotIndex?.snapshots.findIndex(
                  (descriptor) => descriptor.date_key === event.target.value,
                ) ?? -1;
              if (nextIndex < 0) {
                return;
              }
              startTransition(() => {
                setSelectedSnapshotIndex(nextIndex);
              });
            }}
          >
            {snapshotIndex?.snapshots.map((descriptor, index) => (
              <option key={descriptor.date_key} value={descriptor.date_key}>
                {formatDateLabel(descriptor.date)} ({index + 1}/{snapshotIndex.snapshots.length})
              </option>
            ))}
          </select>
        </label>

        <label className="field-group">
          <span>
            Snapshot index{" "}
            {snapshotIndex ? `${selectedSnapshotIndex + 1} / ${snapshotIndex.snapshots.length}` : ""}
          </span>
          <input
            className="range-input"
            type="range"
            min={0}
            max={Math.max((snapshotIndex?.snapshots.length ?? 1) - 1, 0)}
            step={1}
            value={selectedSnapshotIndex}
            onChange={(event) => {
              const next = Number(event.target.value);
              startTransition(() => {
                setSelectedSnapshotIndex(next);
              });
            }}
          />
        </label>

        <label className="field-group">
          <span>Depth exaggeration: {depthExaggeration.toFixed(1)}x</span>
          <input
            className="range-input"
            type="range"
            min={0.7}
            max={4}
            step={0.1}
            value={depthExaggeration}
            onChange={(event) => setDepthExaggeration(Number(event.target.value))}
          />
        </label>

        <div className="toggle-row">
          <button type="button" className="toggle-chip" onClick={() => setSceneResetKey((value) => value + 1)}>
            Reset view
          </button>
          <button
            type="button"
            className={accessibleColors ? "toggle-chip toggle-chip-active" : "toggle-chip"}
            onClick={() => setAccessibleColors((value) => !value)}
          >
            Accessible colors
          </button>
          <button
            type="button"
            className={showSensitivityBars ? "toggle-chip toggle-chip-active" : "toggle-chip"}
            onClick={() => setShowSensitivityBars((value) => !value)}
          >
            Training utility bars
          </button>
          <button
            type="button"
            className="toggle-chip"
            onClick={() => {
              setSelectedPatchId(null);
              setHoveredPatchId(null);
            }}
          >
            Clear selection
          </button>
        </div>

        {fieldMeta ? <ColorLegend meta={fieldMeta} accessible={accessibleColors} /> : null}
        {fieldMeta && selectedFieldGuide ? (
          <section className="field-guide-panel">
            <div className="field-guide-header">
              <div>
                <p className="eyebrow">Field guide</p>
                <h4>{fieldMeta.label}</h4>
              </div>
              <span className="pill-muted">{isStaticField ? "Static field" : "Time-varying field"}</span>
            </div>

            <div className="field-guide-block">
              <span className="meta-label">What It Indicates</span>
              <p>{selectedFieldGuide.summary}</p>
            </div>

            <div className="field-guide-block">
              <span className="meta-label">How To Read Colors</span>
              <p>{colorReading(fieldMeta, accessibleColors)}</p>
            </div>

            <div className="field-guide-block">
              <span className="meta-label">What To Expect Across Snapshots</span>
              <p>{selectedFieldGuide.snapshotBehavior}</p>
            </div>

            <div className="field-guide-block">
              <span className="meta-label">How To Interpret It</span>
              <p>{selectedFieldGuide.interpretationHint}</p>
            </div>
          </section>
        ) : null}
        <p className="panel-note">
          {accessibleColors
            ? "Accessible palette is active: green-yellow and low-contrast green scales are remapped to colorblind-friendlier hues."
            : "Original diagnostic-style palette is active."}
        </p>
        {selectedModel?.metrics_summary?.median_rmse_mm != null ? (
          <p className="panel-note">
            Median RMSE: {formatCompactNumber(selectedModel.metrics_summary.median_rmse_mm)} mm
          </p>
        ) : null}
        {selectedModel?.station_sensitivity_summary ? (
          <p className="panel-note">
            Orange bar height scales with station sensitivity in the training diagnostics for this model.
          </p>
        ) : (
          <p className="panel-note">No station sensitivity ranking exported for this model.</p>
        )}
        {isPending ? <p className="panel-note">Loading snapshot…</p> : null}
      </aside>

      <section className="panel rise canvas-panel">
        <div className="panel-header">
          <div>
            <p className="eyebrow">3D scene</p>
            <h3>Orbit, zoom, pan, inspect</h3>
          </div>
          <span className="pill-muted">Local XYZ in meters</span>
        </div>

        <div className="canvas-shell">
          {fault && fieldMeta && stations && fieldValues ? (
            <Canvas
              key={`${sceneResetKey}-${selectedModelKey}`}
              camera={{ position: [0, 14000, 42000], fov: 34, near: 10, far: 300000 }}
              className="fault-canvas"
            >
              <color attach="background" args={["#edf0eb"]} />
              <ambientLight intensity={0.82} />
              <directionalLight position={[25000, 22000, 15000]} intensity={1.1} />
              <directionalLight position={[-25000, 15000, -12000]} intensity={0.45} />
              <gridHelper args={[120000, 24, "#48606f", "#a8b4bd"]} position={[0, -50, 0]} />
              <SpatialContext
                fault={fault}
                stations={stations}
                stationSensitivity={stationSensitivity}
                showSensitivityBars={showSensitivityBars}
              />

              <Bounds fit clip observe margin={1.4}>
                <FaultMesh
                  fault={fault}
                  fieldMeta={fieldMeta}
                  fieldValues={fieldValues}
                  accessibleColors={accessibleColors}
                  depthExaggeration={depthExaggeration}
                  selectedPatchId={selectedPatchId}
                  hoveredPatchId={hoveredPatchId}
                  onHoverPatch={setHoveredPatchId}
                  onSelectPatch={setSelectedPatchId}
                />
              </Bounds>

              <OrbitControls makeDefault />
            </Canvas>
          ) : (
            <div className="loading-state">Loading patch mesh and snapshot data…</div>
          )}
        </div>
      </section>

      <aside className="panel rise station-panel">
        <div className="panel-header">
          <div>
            <p className="eyebrow">Patch inspection</p>
            <h3>{activePatch ? `Patch ${activePatch.id}` : "Hover or click a patch"}</h3>
          </div>
          {activePatch ? <span className="pill-muted">{activePatch.row}:{activePatch.col}</span> : null}
        </div>

        {activePatch && fieldMeta && activeValue != null ? (
          <div className="inspection-stack">
            <div className="station-meta-grid">
              <div>
                <span className="meta-label">Model</span>
                <strong>{selectedModel?.label ?? "…"}</strong>
              </div>
              <div>
                <span className="meta-label">Snapshot</span>
                <strong>{snapshot ? formatDateLabel(snapshot.date) : "…"}</strong>
              </div>
              <div>
                <span className="meta-label">Field value</span>
                <strong>
                  {formatScientific(activeValue)} {fieldMeta.units}
                </strong>
              </div>
              <div>
                <span className="meta-label">Depth</span>
                <strong>{formatCompactNumber(activePatch.depth_m / 1000, 2)} km</strong>
              </div>
              <div>
                <span className="meta-label">Local center</span>
                <strong>
                  x {formatCompactNumber(activePatch.center_local_xyz_m[0], 0)}, y{" "}
                  {formatCompactNumber(activePatch.center_local_xyz_m[1], 0)}, z{" "}
                  {formatCompactNumber(activePatch.center_local_xyz_m[2], 0)}
                </strong>
              </div>
            </div>

            <div className="panel-note">
              Strike {fault?.meta.fault_meta.strike_deg.toFixed(1)}°, dip {fault?.meta.fault_meta.dip_deg.toFixed(1)}
              °, rake {fault?.meta.fault_meta.rake_deg.toFixed(1)}°. UTM EPSG {fault?.meta.utm_epsg}. Local axes:
              `x = East`, `y = Up`, `z = North`. Surface plane at `y = 0`, orange trace = projected
              fault trace, colored dots = GNSS, gold marker = L’Aquila, red marker/line = 2009
              mainshock hypocenter and its surface projection. Orange cylinders show how useful each
              station was for training, using the exported station-sensitivity ranking when available.
            </div>
          </div>
        ) : (
          <div className="loading-state">
            Use orbit controls to inspect the fault. Hover to probe scalar values, click to lock a
            patch.
          </div>
        )}
      </aside>
    </section>
  );
}
