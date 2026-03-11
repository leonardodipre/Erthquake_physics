#!/usr/bin/env python3
"""
Calcolo Matrici di Green per Dinamica di Faglie Sismiche
Green's Functions Matrices for Seismic Fault Dynamics
=========================================================

Calcola due matrici elastiche usando CUTDE (Okada, semispazio triangolare):
Computes two elastic matrices using CUTDE (Okada, triangular half-space):

  K_ij  (Nc × Nc)       — Trasferimento stress tra patch di faglia
                           Stress transfer between fault patches

  K_cd  (3·Nobs × Nc)   — Spostamento superficiale alle stazioni GNSS
                           Surface displacement at GNSS stations

Vedi docstring di build_K_ij() e build_K_cd() per algoritmo dettagliato.
See build_K_ij() and build_K_cd() docstrings for detailed algorithm.

Coordinate: (E, N, U) con U verso l'alto, profondità negativa.
Coordinates: (E, N, U) with U upward, depth negative.
Superficie libera a z = 0 / Free surface at z = 0.

Riferimenti / References:
  - Okada (1992), BSSA — Internal deformation in a half-space
  - CUTDE: https://github.com/tbenthompson/cutde
"""

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from pyproj import CRS, Transformer

import cutde.halfspace as HS
import cutde.fullspace as FS


# ──────────────────────────────────────────────────────────────
# I/O
# ──────────────────────────────────────────────────────────────

def read_fault_geojson(path: Path) -> Tuple[Dict[str, Any], np.ndarray]:
    """Carica poligono di faglia e proprietà da GeoJSON.
    Load fault polygon and properties from GeoJSON.

    Returns:
        props: proprietà della faglia (strike, dip, rake, profondità, ...)
        ring:  (N, 2) vertici poligonali [lon, lat]
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    feats = data.get("features", [])
    if not feats:
        raise RuntimeError("GeoJSON: 'features' mancante / missing")

    feat = feats[0]
    props = feat.get("properties", {}) or {}
    geom = feat.get("geometry", {}) or {}

    if geom.get("type") != "Polygon":
        raise RuntimeError(f"GeoJSON: atteso Polygon, trovato {geom.get('type')}")

    coords = geom.get("coordinates", [])
    if not coords or not coords[0]:
        raise RuntimeError("GeoJSON: coordinate poligono vuote")

    ring = np.asarray(coords[0], dtype=float)
    if ring.ndim != 2 or ring.shape[1] != 2:
        raise RuntimeError(f"GeoJSON: forma ring inattesa {ring.shape}")

    return props, ring


def load_stations(path: Path, fwd_utm: Transformer) -> Tuple[np.ndarray, np.ndarray]:
    """Carica stazioni GNSS e trasforma in UTM.
    Load GNSS stations and transform to UTM.

    Returns:
        ids: (Nobs,) stringhe ID stazione
        pts: (Nobs, 3) coordinate stazione in (E, N, U=0) metri
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    stations = data.get("stations", [])
    if not stations:
        raise RuntimeError("Stations JSON: lista 'stations' vuota")

    ids = np.array([str(s["marker"]) for s in stations], dtype=object)
    lat = np.array([float(s["aprioriNorth"]) for s in stations])
    lon = np.array([float(s["aprioriEast"]) for s in stations])

    E, N = fwd_utm.transform(lon, lat)
    pts = np.column_stack([np.asarray(E, float),
                           np.asarray(N, float),
                           np.zeros(len(E))])
    return ids, pts


# ──────────────────────────────────────────────────────────────
# Trasformazioni di Coordinate / Coordinate Transforms
# ──────────────────────────────────────────────────────────────

def utm_epsg_from_lonlat(lon: float, lat: float) -> int:
    """Determina codice EPSG UTM da un punto lon/lat."""
    zone = int(math.floor((lon + 180.0) / 6.0) + 1)
    return (32600 + zone) if lat >= 0 else (32700 + zone)


def lonlat_to_utm(lonlat: np.ndarray) -> Tuple[np.ndarray, CRS, Transformer]:
    """Converte array (lon, lat) in UTM (E, N).

    Returns:
        en:      (N, 2) array in metri UTM
        crs_utm: pyproj CRS della zona UTM
        fwd:     Transformer da WGS84 a UTM
    """
    lon0 = float(np.mean(lonlat[:, 0]))
    lat0 = float(np.mean(lonlat[:, 1]))
    epsg = utm_epsg_from_lonlat(lon0, lat0)

    crs_ll = CRS.from_epsg(4326)
    crs_utm = CRS.from_epsg(epsg)
    fwd = Transformer.from_crs(crs_ll, crs_utm, always_xy=True)

    x, y = fwd.transform(lonlat[:, 0], lonlat[:, 1])
    en = np.column_stack([np.asarray(x, float), np.asarray(y, float)])

    return en, crs_utm, fwd


# ──────────────────────────────────────────────────────────────
# Geometria della Faglia / Fault Geometry
# ──────────────────────────────────────────────────────────────

def _strike_vec_EN(strike_deg: float) -> np.ndarray:
    """Versore strike in (E, N). Strike unit vector in (E, N)."""
    th = math.radians(strike_deg)
    return np.array([math.sin(th), math.cos(th)])


def _dip_h_vec_EN(strike_deg: float) -> np.ndarray:
    """Componente orizzontale della direzione di dip in (E, N).
    Dip direction = strike + 90°."""
    th = math.radians(strike_deg + 90.0)
    return np.array([math.sin(th), math.cos(th)])


def fault_frame(
    strike_deg: float, dip_deg: float, rake_deg: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calcola il sistema di riferimento della faglia come vettori 3D (E, N, U).
    Compute fault reference frame as 3D unit vectors (E, N, U).

    Returns:
        s: direzione strike (orizzontale, lungo la traccia di faglia)
        d: direzione down-dip (nel piano di faglia, verso il basso)
        n: normale alla faglia (n = s × d, punta verso l'hanging wall)
        r: direzione rake/slip (nel piano di faglia)
    """
    s_EN = _strike_vec_EN(strike_deg)
    dh_EN = _dip_h_vec_EN(strike_deg)
    dip = math.radians(dip_deg)
    rake = math.radians(rake_deg)

    s = np.array([s_EN[0], s_EN[1], 0.0])
    d = np.array([dh_EN[0] * math.cos(dip),
                  dh_EN[1] * math.cos(dip),
                  -math.sin(dip)])

    s /= np.linalg.norm(s)
    d /= np.linalg.norm(d)
    n = np.cross(s, d)
    n /= np.linalg.norm(n)
    r = math.cos(rake) * s + math.sin(rake) * d
    r /= np.linalg.norm(r)

    return s, d, n, r


def estimate_fault_rect(
    props: Dict[str, Any], poly_en: np.ndarray,
) -> Dict[str, Any]:
    """Stima parametri rettangolari della faglia da proprietà GeoJSON e poligono.

    Estrae strike, dip, rake dalle proprietà. Calcola lunghezza dall'estensione
    del poligono lungo lo strike. Calcola larghezza dal range di profondità e dip.

    Returns dict con: strike_deg, dip_deg, rake_deg, top_depth_m, bottom_depth_m,
                      length_m, width_m, origin_en
    """
    strike = float(np.nanmean([props.get("strikemin", np.nan),
                                props.get("strikemax", np.nan)]))
    dip = float(np.nanmean([props.get("dipmin", np.nan),
                             props.get("dipmax", np.nan)]))
    rake = float(np.nanmean([props.get("rakemin", np.nan),
                              props.get("rakemax", np.nan)]))

    if not np.isfinite(strike) or not np.isfinite(dip):
        raise RuntimeError("GeoJSON faglia: strike o dip mancanti")
    if not np.isfinite(rake):
        rake = 0.0

    top_km = float(props.get("mindepth", props.get("mindepthq", 1.0)))
    bot_km = float(props.get("maxdepth", props.get("maxdepthq", 14.0)))
    top_m, bot_m = top_km * 1000.0, bot_km * 1000.0
    if bot_m <= top_m:
        raise RuntimeError("GeoJSON faglia: profondità bottom <= top")

    # Proietta poligono su coordinate strike/dip
    cen = np.mean(poly_en, axis=0)
    s_EN = _strike_vec_EN(strike)
    dh_EN = _dip_h_vec_EN(strike)
    delta = poly_en - cen[None, :]
    xs = delta @ s_EN  # coordinata lungo strike
    ys = delta @ dh_EN  # coordinata orizzontale dip

    length_m = max(1000.0, float(np.ptp(xs)))
    width_m = (bot_m - top_m) / math.sin(math.radians(dip))

    # Origine: centro-top del rettangolo di faglia
    x0 = 0.5 * (float(np.min(xs)) + float(np.max(xs)))
    y0 = float(np.min(ys))
    origin_en = cen + s_EN * x0 + dh_EN * y0

    return {
        "strike_deg": strike, "dip_deg": dip, "rake_deg": rake,
        "top_depth_m": top_m, "bottom_depth_m": bot_m,
        "length_m": length_m, "width_m": width_m,
        "origin_en": origin_en.astype(float),
    }


def build_triangle_mesh(
    origin_en: np.ndarray, strike_deg: float, dip_deg: float,
    top_depth_m: float, length_m: float, width_m: float,
    nx: int, ny: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Crea mesh triangolare della faglia rettangolare.

    Ogni patch rettangolare è diviso in 2 triangoli (alto-dx, basso-sx).
    Ordine patch row-major: iy (down-dip) varia lento, ix (strike) veloce.

    Returns:
        tris:    (2·Nc, 3, 3)  vertici triangoli in (E, N, U) metri
        centers: (Nc, 3)       centroidi patch
    """
    if nx <= 0 or ny <= 0:
        raise ValueError("nx e ny devono essere > 0")

    Nc = nx * ny
    s, d, n, _ = fault_frame(strike_deg, dip_deg, rake_deg=0.0)
    
    C_top = np.array([origin_en[0], origin_en[1], -top_depth_m])

    #lunghezza di variaizone per ogni blocco
    dl = length_m / nx
    dw = width_m / ny

    # Coordinate bordi griglia
    x_edges = -0.5 * length_m + np.arange(nx + 1) * dl
    w_edges = np.arange(ny + 1) * dw

  
    # Indici patch (row-major: iy lento, ix veloce)
    iy, ix = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    ix, iy = ix.ravel(), iy.ravel()

    # Vertici angolari per ogni patch: (Nc, 3)
    TL = C_top + np.outer(x_edges[ix], s) + np.outer(w_edges[iy], d)
    TR = C_top + np.outer(x_edges[ix + 1], s) + np.outer(w_edges[iy], d)
    BR = C_top + np.outer(x_edges[ix + 1], s) + np.outer(w_edges[iy + 1], d)
    BL = C_top + np.outer(x_edges[ix], s) + np.outer(w_edges[iy + 1], d)

    centers = 0.25 * (TL + TR + BR + BL)

    # 2 triangoli per patch interallacciati: (2·Nc, 3, 3)
    tris = np.empty((2 * Nc, 3, 3))
    tris[0::2, 0], tris[0::2, 1], tris[0::2, 2] = TL, TR, BR
    tris[1::2, 0], tris[1::2, 1], tris[1::2, 2] = TL, BR, BL

    # Assicura che le normali dei triangoli puntino nella direzione della normale di faglia
    e1 = tris[:, 1] - tris[:, 0]
    e2 = tris[:, 2] - tris[:, 0]
    tri_normals = np.cross(e1, e2)
    flip = (tri_normals @ n) < 0
    tris[flip, 1], tris[flip, 2] = tris[flip, 2].copy(), tris[flip, 1].copy()

    return tris, centers


# ──────────────────────────────────────────────────────────────
# Matrici di Green / Green's Functions Matrices
# ──────────────────────────────────────────────────────────────

def build_K_cd(
    obs_pts: np.ndarray,
    tris: np.ndarray,
    rake_deg: float,
    nu: float,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """
    Matrice di Green degli spostamenti K_cd.
    Displacement Green's function matrix K_cd.

    Collega lo slip sui patch di faglia agli spostamenti superficiali
    misurati dalle stazioni GNSS.

    Links fault patch slip to surface displacements at GNSS stations.

        K_cd[3·i + c, j] = spostamento (m) in direzione c alla stazione i
                            causato da 1 m di slip (lungo il rake) al patch j

    Componenti c: 0=Est, 1=Nord, 2=Verticale (Up)
    Ordine righe: [E₁, N₁, U₁, E₂, N₂, U₂, ...]

    Algoritmo passo per passo / Step-by-step algorithm:
    ───────────────────────────────────────────────────

    1. DIREZIONE DI SLIP / SLIP DIRECTION
       Vettore slip unitario nel sistema TDCS (Triangle Dislocation
       Coordinate System) di CUTDE:
           w = [cos(rake), sin(rake), 0]
       Componenti: (strike-slip, dip-slip, tensile).
       Esempio: rake=270° (faglia normale) → w = [0, -1, 0]
                = puro dip-slip verso il basso (hanging wall scende).

    2. MATRICE DI INFLUENZA SPOSTAMENTI / DISPLACEMENT INFLUENCE MATRIX
       [Chiamata CUTDE: HS.disp_matrix]
       D[i, c, t, s] = spostamento componente c alla stazione i
                        da slip unitario componente s sul triangolo t
       Forma: (Nobs, 3, Ntri, 3)

       Questo è il calcolo CUTDE principale, usando le soluzioni analitiche
       di Okada (1992) per elementi di dislocazione triangolare in semispazio.

    3. CONTRAZIONE RAKE / RAKE CONTRACTION
       Proietta le 3 componenti di slip sulla direzione del rake:
           D_rake[i, c, t] = Σ_s  D[i,c,t,s] · w[s]
       Forma: (Nobs, 3, Ntri)

       Dà lo spostamento per 1m di slip nella direzione del rake.

    4. SOMMA TRIANGOLI → PATCH / TRIANGLE-TO-PATCH SUMMATION
       Ogni patch rettangolare = 2 triangoli (indici 2j e 2j+1).
       Somma i contributi:
           K[i, c, j] = D_rake[i, c, 2j] + D_rake[i, c, 2j+1]
       Forma: (Nobs, 3, Npatch)

    5. RESHAPE FORMATO IMPILATO / RESHAPE TO STACKED FORMAT
       Appiattisce le dimensioni (stazione, componente):
           K_cd[3·i + c, j] = K[i, c, j]
       Forma finale: (3·Nobs, Npatch)
    """
    n_obs = obs_pts.shape[0]
    n_tri = tris.shape[0]
    n_patch = n_tri // 2

    # Controllo memoria / Memory check
    mem_gb = n_obs * 3 * n_tri * 3 * 8 / 1e9
    if mem_gb > 4.0:
        raise MemoryError(
            f"disp_matrix richiederebbe ~{mem_gb:.1f} GB. "
            f"Ridurre stazioni o risoluzione mesh."
        )

    # Passo 1: Direzione slip in TDCS
    rake = math.radians(rake_deg)
    w = np.array([math.cos(rake), math.sin(rake), 0.0])

    # Passo 2: Matrice influenza spostamenti completa
    D = HS.disp_matrix(obs_pts, tris, float(nu))  # (Nobs, 3, Ntri, 3)

    # Passo 3: Contrazione con direzione di slip
    Dw = np.einsum("octr,r->oct", D, w)  # (Nobs, 3, Ntri)
    del D

    # Passo 4: Somma 2 triangoli per patch
    Dp = Dw.reshape((n_obs, 3, n_patch, 2)).sum(axis=3)  # (Nobs, 3, Npatch)
    del Dw

    # Passo 5: Reshape formato impilato [E₁,N₁,U₁,E₂,N₂,U₂,...]
    return Dp.reshape((3 * n_obs, n_patch)).astype(dtype)


def build_K_ij(
    patch_centers: np.ndarray,
    tris: np.ndarray,
    strike_deg: float,
    dip_deg: float,
    rake_deg: float,
    nu: float,
    mu: float,
    receiver_offset_m: float,
    compute_sigN: bool = False,
    dtype: np.dtype = np.float32,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Matrice di Green per trasferimento stress K_ij.
    Stress transfer Green's function matrix K_ij.

    Descrive l'interazione elastica tra patch: come lo slip sul patch j
    cambia la trazione di taglio sul patch i.

    Describes elastic interaction between patches: how slip on patch j
    changes shear traction on patch i.

        K_ij[i, j] = trazione di taglio (Pa) al patch i lungo il rake
                     causata da 1 m di slip (lungo il rake) al patch j

    Opzionale: K_sigN[i, j] = trazione normale (compressione-positiva)

    Algoritmo passo per passo / Step-by-step algorithm:
    ───────────────────────────────────────────────────

    1. SISTEMA DI RIFERIMENTO FAGLIA / FAULT REFERENCE FRAME
       Calcola vettori unitari 3D dagli angoli strike/dip/rake:
         s = direzione strike (orizzontale, lungo la traccia di faglia)
         d = direzione down-dip (nel piano di faglia, verso profondità)
         n = normale alla faglia (n = s × d, punta verso l'hanging wall)
         r = direzione rake/slip (nel piano di faglia)

    2. OFFSET RICEVITORI / RECEIVER OFFSET
       Sposta i punti di osservazione leggermente fuori dal piano di faglia:
           obs = centri ± offset · n
       Evita la singolarità matematica quando sorgente e ricevitore
       coincidono sullo stesso piano.
       L'offset è piccolo (~5m) rispetto alla dimensione del patch (~1km).
       Se un punto uscirebbe dal semispazio (z > 0), inverte la direzione.

    3. MATRICE DI INFLUENZA STRAIN / STRAIN INFLUENCE MATRIX
       [Chiamata CUTDE: HS.strain_matrix]
       E[i, e, t, s] = componente strain e al ricevitore i
                        da slip unitario componente s sul triangolo t
       Forma: (Nc, 6, Ntri, 3)
       Componenti strain e: (εxx, εyy, εzz, εxy, εxz, εyz)

    4. CONTRAZIONE RAKE / RAKE CONTRACTION
       Proietta lo slip sulla direzione del rake:
           E_rake[i, e, t] = Σ_s  E[i,e,t,s] · w[s]
       Forma: (Nc, 6, Ntri)

    5. SOMMA TRIANGOLI → PATCH / TRIANGLE-TO-PATCH SUMMATION
       Somma i 2 triangoli per patch:
           E_patch[i, e, j] = E_rake[i, e, 2j] + E_rake[i, e, 2j+1]
       Forma: (Nc, 6, Nc)

    6. STRAIN → STRESS (Legge di Hooke / Hooke's Law)
       Applica elasticità lineare isotropa:
           σ_ij = λ · tr(ε) · δ_ij  +  2μ · ε_ij
       dove λ = 2μν / (1-2ν)  (primo parametro di Lamé)

       Input:  (Nc, Nc, 6) strain  →  Output: (Nc, Nc, 6) stress
       Componenti: (σxx, σyy, σzz, σxy, σxz, σyz)

    7. VETTORE TRAZIONE / TRACTION VECTOR (Formula di Cauchy)
       Calcola la trazione sul piano di faglia:
           t = σ · n
       Espanso:
           tₓ = σxx·nₓ + σxy·nᵧ + σxz·n_z
           tᵧ = σxy·nₓ + σyy·nᵧ + σyz·n_z
           t_z = σxz·nₓ + σyz·nᵧ + σzz·n_z

    8. TRAZIONE DI TAGLIO / SHEAR TRACTION (proiezione sul rake)
       Sforzo di taglio risolto che guida lo slip:
           τ = t · r = tₓ·rₓ + tᵧ·rᵧ + t_z·r_z
       Forma: (Nc, Nc)

    9. TRAZIONE NORMALE / NORMAL TRACTION (opzionale, compressione-positiva)
           σN = -(t · n)
       Il segno negativo converte dalla convenzione tensione-positiva
       di Cauchy alla convenzione compressione-positiva (meccanica rocce).
    """
    n_patch = patch_centers.shape[0]
    n_tri = tris.shape[0]

    # Controllo memoria / Memory check
    mem_gb = n_patch * 6 * n_tri * 3 * 8 / 1e9
    if mem_gb > 4.0:
        raise MemoryError(
            f"strain_matrix richiederebbe ~{mem_gb:.1f} GB. "
            f"Ridurre risoluzione mesh (attuale: Nc={n_patch})."
        )

    # Passo 1: Sistema di riferimento faglia
    _s, _d, n_vec, r = fault_frame(strike_deg, dip_deg, rake_deg)

    # Passo 2: Offset ricevitori fuori dal piano di faglia (per-punto)
    obs_pts = patch_centers + receiver_offset_m * n_vec[None, :]
    above = obs_pts[:, 2] > -1e-6
    if above.any():
        obs_pts[above] = patch_centers[above] - receiver_offset_m * n_vec[None, :]

    # Direzione slip in TDCS
    rake_rad = math.radians(rake_deg)
    w = np.array([math.cos(rake_rad), math.sin(rake_rad), 0.0])

    # Passo 3: Matrice influenza strain completa
    E = HS.strain_matrix(obs_pts, tris, float(nu))  # (Nc, 6, Ntri, 3)

    # Passo 4: Contrazione con direzione di slip
    Ew = np.einsum("oscr,r->osc", E, w)  # (Nc, 6, Ntri)
    del E

    # Passo 5: Somma 2 triangoli per patch
    Ep = Ew.reshape((n_patch, 6, n_patch, 2)).sum(axis=3)  # (Nc, 6, Nc)
    del Ew

    # Passo 6: Strain → Stress (Legge di Hooke)
    # Trasponi a (Nc, Nc, 6) per strain_to_stress, poi reshape
    Ep_t = np.transpose(Ep, (0, 2, 1))  # (Nc, Nc, 6)
    del Ep
    stress = FS.strain_to_stress(
        Ep_t.reshape((-1, 6)), float(mu), float(nu)
    ).reshape((n_patch, n_patch, 6))
    del Ep_t
    # stress[:,:,k] → k: 0=σxx, 1=σyy, 2=σzz, 3=σxy, 4=σxz, 5=σyz

    # Passo 7: Trazione sul piano di faglia (Cauchy: t = σ · n)
    nx, ny, nz = float(n_vec[0]), float(n_vec[1]), float(n_vec[2])
    tx = stress[:, :, 0] * nx + stress[:, :, 3] * ny + stress[:, :, 4] * nz
    ty = stress[:, :, 3] * nx + stress[:, :, 1] * ny + stress[:, :, 5] * nz
    tz = stress[:, :, 4] * nx + stress[:, :, 5] * ny + stress[:, :, 2] * nz

    # Passo 8: Trazione di taglio lungo il rake
    rx, ry, rz = float(r[0]), float(r[1]), float(r[2])
    K_tau = (tx * rx + ty * ry + tz * rz).astype(dtype)

    # Passo 9: Trazione normale (opzionale, compressione-positiva)
    K_sigN = None
    if compute_sigN:
        K_sigN = (-(tx * nx + ty * ny + tz * nz)).astype(dtype)

    return K_tau, K_sigN


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Calcola matrici di Green elastiche per faglia sismica.\n"
                    "Usa --config per leggere da file JSON, oppure argomenti CLI."
    )
    ap.add_argument("--config", default=None,
                    help="File JSON di configurazione (sezione 'green_function')")
    ap.add_argument("--fault_geojson", default=None,
                    help="File GeoJSON geometria faglia")
    ap.add_argument("--stations_json", default=None,
                    help="File JSON stazioni GNSS")
    ap.add_argument("--out_dir", default=None,
                    help="Directory output")
    ap.add_argument("--cell_km", type=float, default=None,
                    help="Dimensione patch in km")
    ap.add_argument("--nu", type=float, default=None,
                    help="Rapporto di Poisson")
    ap.add_argument("--mu", type=float, default=None,
                    help="Modulo di taglio (Pa)")
    ap.add_argument("--receiver_offset_m", type=float, default=None,
                    help="Offset per evitare singolarità K_ij (m)")
    ap.add_argument("--save_sigN", action="store_true",
                    help="Salva anche matrice trazione normale")
    cli = ap.parse_args()

    # --- Carica configurazione (config JSON + override CLI) ---
    cfg = {
        "fault_geojson": None,
        "stations_json": None,
        "out_dir": "green_out",
        "cell_km": 0.75,
        "nu": 0.25,
        "mu": 30e9,
        "receiver_offset_m": 5.0,
        "save_sigN": False,
    }

    if cli.config is not None:
        raw = json.loads(Path(cli.config).read_text(encoding="utf-8"))
        gc = raw.get("green_function", raw)
        cfg.update({k: v for k, v in gc.items() if not k.startswith("_")})

    # CLI override (valori espliciti sovrascrivono il config)
    for key in ("fault_geojson", "stations_json", "out_dir",
                "cell_km", "nu", "mu", "receiver_offset_m"):
        val = getattr(cli, key)
        if val is not None:
            cfg[key] = val
    if cli.save_sigN:
        cfg["save_sigN"] = True

    # Validazione parametri obbligatori
    if cfg["fault_geojson"] is None:
        ap.error("--fault_geojson obbligatorio (oppure specificare nel config)")
    if cfg["stations_json"] is None:
        ap.error("--stations_json obbligatorio (oppure specificare nel config)")

    print("=" * 60)
    print("  Green's Function - Configurazione")
    print("=" * 60)
    for k, v in cfg.items():
        print(f"  {k}: {v}")
    print("=" * 60)

    out = Path(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)

    # --- Carica geometria faglia ---
    props, ring_ll = read_fault_geojson(Path(cfg["fault_geojson"]))
    poly_en, crs_utm, fwd = lonlat_to_utm(ring_ll)
    meta = estimate_fault_rect(props, poly_en)

    # --- Discretizza ---
    cell_m = cfg["cell_km"] * 1000.0
    nx = max(1, round(meta["length_m"] / cell_m))
    ny = max(1, round(meta["width_m"] / cell_m))

    tris, centers = build_triangle_mesh(
        origin_en=meta["origin_en"],
        strike_deg=meta["strike_deg"], dip_deg=meta["dip_deg"],
        top_depth_m=meta["top_depth_m"],
        length_m=meta["length_m"], width_m=meta["width_m"],
        nx=nx, ny=ny,
    )

    Nc = centers.shape[0]
    epsg = int(crs_utm.to_epsg())

    print(f"[INFO] UTM EPSG: {epsg}")
    print(f"[INFO] Patch: Nc={Nc} (nx={nx}, ny={ny}), triangoli={tris.shape[0]}")
    print(f"[INFO] strike={meta['strike_deg']:.1f}  "
          f"dip={meta['dip_deg']:.1f}  rake={meta['rake_deg']:.1f}")
    print(f"[INFO] L={meta['length_m']/1e3:.1f}km  W={meta['width_m']/1e3:.1f}km  "
          f"profondita={meta['top_depth_m']/1e3:.1f}-{meta['bottom_depth_m']/1e3:.1f}km")

    # --- Salva mesh ---
    mesh_path = out / "fault_mesh.npz"
    np.savez(
        mesh_path,
        tris=tris.astype(np.float64),
        patch_centers=centers.astype(np.float64),
        utm_epsg=epsg,
        meta=json.dumps({
            k: (v.tolist() if isinstance(v, np.ndarray) else v)
            for k, v in meta.items()
        }),
    )
    print(f"[OK] Mesh salvata: {mesh_path}")

    # --- Calcola K_ij (trasferimento stress) ---
    print("[...] Calcolo K_ij (trasferimento stress)...")
    K_tau, K_sigN = build_K_ij(
        patch_centers=centers, tris=tris,
        strike_deg=meta["strike_deg"],
        dip_deg=meta["dip_deg"],
        rake_deg=meta["rake_deg"],
        nu=cfg["nu"], mu=cfg["mu"],
        receiver_offset_m=cfg["receiver_offset_m"],
        compute_sigN=cfg["save_sigN"],
    )

    tau_path = out / "K_ij_tau.npy"
    np.save(tau_path, K_tau)
    print(f"[OK] K_ij_tau: {tau_path}  shape={K_tau.shape}")

    if K_sigN is not None:
        sig_path = out / "K_ij_sigN.npy"
        np.save(sig_path, K_sigN)
        print(f"[OK] K_ij_sigN: {sig_path}  shape={K_sigN.shape}")

    # --- Calcola K_cd (spostamenti superficiali) ---
    st_ids, st_pts = load_stations(Path(cfg["stations_json"]), fwd)
    np.save(out / "station_ids.npy", st_ids)
    print(f"[INFO] Stazioni GNSS: Nobs={st_pts.shape[0]}")

    print("[...] Calcolo K_cd (spostamenti)...")
    K_cd = build_K_cd(
        obs_pts=st_pts, tris=tris,
        rake_deg=meta["rake_deg"], nu=cfg["nu"],
    )

    kcd_path = out / "K_cd_disp.npy"
    np.save(kcd_path, K_cd)
    print(f"[OK] K_cd: {kcd_path}  shape={K_cd.shape}")

    # --- Salva summary JSON ---
    summary = {
        "utm_epsg": epsg,
        "nx": nx, "ny": ny, "Nc": Nc,
        "fault_meta": {
            k: (v.tolist() if isinstance(v, np.ndarray) else v)
            for k, v in meta.items()
        },
        "elastic": {"nu": cfg["nu"], "mu": cfg["mu"]},
        "notes": {
            "K_ij_units": "Pa per meter of slip",
            "K_cd_units": "m per meter of slip",
            "K_cd_row_order": "[E1,N1,U1,E2,N2,U2,...]",
        },
    }
    summary_path = out / "green_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[OK] Summary: {summary_path}")


if __name__ == "__main__":
    main()
