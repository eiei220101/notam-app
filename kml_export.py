"""
NOTAM 由来の緯度経度・高度から KML（NOTAM ごとに頂点を結んだ押し出しポリゴン）を生成する。
"""

from __future__ import annotations

import html
import math
import re
from typing import Any, Optional

FT_TO_M = 0.3048
KML_NS = "http://www.opengis.net/kml/2.2"

# PSN : 393621.2N1414645.1E 形式（空白多少許容）
_PSN_PAIR_RE = re.compile(
    r"PSN\s*:\s*"
    r"(\d{2})(\d{2})(\d+(?:\.\d+)?)\s*([NS])\s*"
    r"(\d{3})(\d{2})(\d+(?:\.\d+)?)\s*([EW])",
    re.IGNORECASE,
)

# 連結度分秒（例 382346N1411242E = 38°23′46″N 141°12′42″E）。PSN: 無しの BOUNDED BY 等で使用。
_COMPACT_DMS_NE_RE = re.compile(
    r"\b(\d{2})(\d{2})(\d+(?:\.\d+)?)([NS])(\d{3})(\d{2})(\d+(?:\.\d+)?)([EW])\b",
    re.IGNORECASE,
)

# 同一行付近の HGT:648FT AMSL
_HGT_FT_AMSL_RE = re.compile(
    r"HGT\s*:\s*(\d+(?:\.\d+)?)\s*FT\s*AMSL",
    re.IGNORECASE,
)


def _dms_to_decimal(deg: str, minute: str, sec: str, hemi: str) -> float:
    v = float(deg) + float(minute) / 60.0 + float(sec) / 3600.0
    if hemi.upper() in ("S", "W"):
        v = -v
    return v


def _dms_parts_plausible(
    deg_s: str, min_s: str, sec_s: str, *, max_deg: int
) -> bool:
    try:
        d = int(deg_s)
        m = int(min_s)
        s = float(sec_s)
    except ValueError:
        return False
    if not (0 <= d <= max_deg and 0 <= m < 60 and 0.0 <= s < 60.0):
        return False
    return True


def parse_psn_compact(text: str) -> list[tuple[float, float]]:
    """PSN 行から緯度経度（度）のリストを返す（見つからなければ空）。"""
    return [(p["lat"], p["lon"]) for p in parse_psn_points_with_optional_hgt(text)]


def parse_psn_points_with_optional_hgt(text: str) -> list[dict[str, Any]]:
    """
    PSN 行および連結度分秒（382346N1411242E 等）から lat/lon を抽出する。
    同一マッチ付近の HGT FT AMSL があれば upper_ft_amsl を付与（PSN 行優先）。
    """
    out: list[dict[str, Any]] = []
    seen: set[tuple[float, float]] = set()

    def _append_point(lat: float, lon: float, tail_start: int) -> None:
        key = (round(lat, 6), round(lon, 6))
        if key in seen:
            return
        seen.add(key)
        tail = text[tail_start : tail_start + 120]
        hm = _HGT_FT_AMSL_RE.search(tail)
        row: dict[str, Any] = {"lat": lat, "lon": lon}
        if hm:
            try:
                row["upper_ft_amsl"] = float(hm.group(1))
            except ValueError:
                pass
        out.append(row)

    for m in _PSN_PAIR_RE.finditer(text):
        lat = _dms_to_decimal(m.group(1), m.group(2), m.group(3), m.group(4))
        lon = _dms_to_decimal(m.group(5), m.group(6), m.group(7), m.group(8))
        _append_point(lat, lon, m.end())

    for m in _COMPACT_DMS_NE_RE.finditer(text):
        ld, lm, ls, lh, gd, gm, gs, gh = m.groups()
        if not _dms_parts_plausible(ld, lm, ls, max_deg=90):
            continue
        if not _dms_parts_plausible(gd, gm, gs, max_deg=180):
            continue
        lat = _dms_to_decimal(ld, lm, ls, lh)
        lon = _dms_to_decimal(gd, gm, gs, gh)
        if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
            continue
        _append_point(lat, lon, m.end())

    return out


def _square_coords_lon_lat_alt(
    lat0: float, lon0: float, alt_m: float, half_size_m: float = 85.0
) -> str:
    """水平正方形リング（同一高度）。KML の lon,lat,alt 空白区切り。"""
    dlat = half_size_m / 111_320.0
    dlon = half_size_m / max(1e-6, 111_320.0 * math.cos(math.radians(lat0)))
    corners = [
        (lon0 - dlon, lat0 - dlat),
        (lon0 + dlon, lat0 - dlat),
        (lon0 + dlon, lat0 + dlat),
        (lon0 - dlon, lat0 + dlat),
        (lon0 - dlon, lat0 - dlat),
    ]
    return " ".join(f"{lo:.7f},{la:.7f},{alt_m:.2f}" for lo, la in corners)


def _segment_corridor_ring(
    la0: float,
    lo0: float,
    la1: float,
    lo1: float,
    alt_m: float,
    half_width_m: float = 45.0,
) -> str:
    """2 頂点を短辺でつないだ細長い水平四角形（同一高度）の閉じたリング。"""
    mid_lat = (la0 + la1) * 0.5
    cos_lat = max(1e-6, math.cos(math.radians(mid_lat)))
    dlat_m = (la1 - la0) * 111_320.0
    dlon_m = (lo1 - lo0) * 111_320.0 * cos_lat
    length = math.hypot(dlat_m, dlon_m)
    if length < 3.0:
        return _square_coords_lon_lat_alt(
            (la0 + la1) * 0.5, (lo0 + lo1) * 0.5, alt_m, half_size_m=max(half_width_m, 50.0)
        )
    pn_m = -dlon_m / length
    pe_m = dlat_m / length
    dlat_off = pn_m * (half_width_m / 111_320.0)
    dlon_off = pe_m * (half_width_m / (111_320.0 * cos_lat))
    corners = [
        (lo0 + dlon_off, la0 + dlat_off),
        (lo0 - dlon_off, la0 - dlat_off),
        (lo1 - dlon_off, la1 - dlat_off),
        (lo1 + dlon_off, la1 + dlat_off),
        (lo0 + dlon_off, la0 + dlat_off),
    ]
    return " ".join(f"{lo:.7f},{la:.7f},{alt_m:.2f}" for lo, la in corners)


def _dedupe_consecutive_lon_lat(
    lon_lat: list[tuple[float, float]], eps: float = 1e-6
) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for lo, la in lon_lat:
        if out and abs(out[-1][0] - lo) < eps and abs(out[-1][1] - la) < eps:
            continue
        out.append((lo, la))
    return out


def _centroid_lon_lat(lon_lat: list[tuple[float, float]]) -> tuple[float, float]:
    """頂点列の単純平均でピン位置（lon, lat）を求める。"""
    if not lon_lat:
        return 0.0, 0.0
    n = len(lon_lat)
    lo = sum(p[0] for p in lon_lat) / n
    la = sum(p[1] for p in lon_lat) / n
    return lo, la


def _esc(s: str) -> str:
    return html.escape(s or "", quote=True)


def augment_spatial_json_with_psn_regex(
    spatial: dict[str, Any],
    raw_text: str,
    domestic_list: list[str],
) -> dict[str, Any]:
    """
    原文の PSN / 連結度分秒から頂点を取り、KML 用 spatial を補完する。
    Gemini が has_positions でも誤った点列だけ返すことがあるため、
    原文から **3 点以上**取れたときはその頂点列を優先して差し替える。
    """
    pts_rows = parse_psn_points_with_optional_hgt(raw_text)
    if not pts_rows:
        return spatial

    def _features_empty_or_no_points() -> bool:
        feats = spatial.get("features")
        if not isinstance(feats, list) or not feats:
            return True
        for f in feats:
            if not isinstance(f, dict):
                continue
            p = f.get("points")
            if isinstance(p, list) and p:
                return False
        return True

    gemini_nonempty = spatial.get("has_positions") and not _features_empty_or_no_points()
    # 多角形級の座標が原文にあるのに、Gemini だけ信じるとズレるため上書きする
    prefer_raw_polygon = len(pts_rows) >= 3

    if gemini_nonempty and not prefer_raw_polygon:
        return spatial

    dom = (domestic_list[0] if domestic_list else "").strip()
    points: list[dict[str, Any]] = []
    for pr in pts_rows:
        la = float(pr["lat"])
        lo = float(pr["lon"])
        cell: dict[str, Any] = {
            "lat": la,
            "lon": lo,
            "comment": "原文（PSN/連結度分秒）より抽出",
        }
        if "upper_ft_amsl" in pr:
            cell["upper_ft_amsl"] = pr["upper_ft_amsl"]
        points.append(cell)
    return {
        "has_positions": True,
        "features": [
            {
                "notam_index": 1,
                "domestic_notam_number": dom,
                "points": points,
                "lower_ft_amsl": 0.0,
                "upper_ft_amsl": None,
                "notes": "自動補完（PSN / 連結度分秒 より抽出）",
            }
        ],
    }


def build_kml_bytes_from_spatial_json(
    spatial: dict[str, Any],
    *,
    document_title: str,
    fallback_domestic_by_index: list[str],
) -> Optional[bytes]:
    """
    spatial JSON から KML を生成する。NOTAM（feature）ごとに points を順に結び、
    1 件につき 1 つの水平押し出しポリゴンと、国内番号名のピン 1 本（頂点の重心・上端高度）を付ける。
    """
    if not spatial.get("has_positions"):
        return None
    features = spatial.get("features")
    if not isinstance(features, list) or not features:
        return None

    def _feat_sort_key(f: object) -> tuple[int, int]:
        if not isinstance(f, dict):
            return (999_999, 0)
        try:
            return (int(f.get("notam_index") or 0), 0)
        except (TypeError, ValueError):
            return (999_999, 0)

    features = sorted(
        (f for f in features if isinstance(f, dict)),
        key=_feat_sort_key,
    )

    parts: list[str] = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<kml xmlns="{KML_NS}">',
        "<Document>",
        f"<name>{_esc(document_title)}</name>",
        "<open>1</open>",
    ]

    for feat in features:
        if not isinstance(feat, dict):
            continue
        idx = int(feat.get("notam_index") or 0)
        name = (feat.get("domestic_notam_number") or "").strip()
        if 1 <= idx <= len(fallback_domestic_by_index):
            fb = (fallback_domestic_by_index[idx - 1] or "").strip()
            if fb:
                name = fb
        if not name:
            name = f"NOTAM-{idx}" if idx else "NOTAM"

        lower_ft = feat.get("lower_ft_amsl")
        upper_ft = feat.get("upper_ft_amsl")
        try:
            low_m_base = float(lower_ft) * FT_TO_M if lower_ft is not None else 0.0
        except (TypeError, ValueError):
            low_m_base = 0.0
        try:
            up_m_base = (
                float(upper_ft) * FT_TO_M if upper_ft is not None else low_m_base + 30.0
            )
        except (TypeError, ValueError):
            up_m_base = low_m_base + 30.0
        if up_m_base <= low_m_base:
            up_m_base = low_m_base + 15.0

        pts = feat.get("points")
        if not isinstance(pts, list) or not pts:
            continue

        vertices: list[tuple[float, float, float]] = []
        for p in pts:
            if not isinstance(p, dict):
                continue
            try:
                la = float(p["lat"])
                lo = float(p["lon"])
            except (KeyError, TypeError, ValueError):
                continue
            low_m = low_m_base
            up_m = up_m_base
            pt_upper = p.get("upper_ft_amsl")
            pt_lower = p.get("lower_ft_amsl")
            try:
                if pt_lower is not None:
                    low_m = float(pt_lower) * FT_TO_M
            except (TypeError, ValueError):
                pass
            try:
                if pt_upper is not None:
                    up_m = float(pt_upper) * FT_TO_M
            except (TypeError, ValueError):
                pass
            if up_m <= low_m:
                up_m = low_m + 15.0
            vertices.append((lo, la, up_m))

        if not vertices:
            continue

        unified_up = max(v[2] for v in vertices)
        lon_lat = _dedupe_consecutive_lon_lat([(v[0], v[1]) for v in vertices])
        n = len(lon_lat)
        if n < 1:
            continue

        if n >= 3:
            first = lon_lat[0]
            last = lon_lat[-1]
            if abs(first[0] - last[0]) < 1e-7 and abs(first[1] - last[1]) < 1e-7:
                ring_coords = lon_lat
            else:
                ring_coords = lon_lat + [first]
            ring = " ".join(
                f"{lo:.7f},{la:.7f},{unified_up:.2f}" for lo, la in ring_coords
            )
        elif n == 2:
            (lo0, la0), (lo1, la1) = lon_lat
            ring = _segment_corridor_ring(la0, lo0, la1, lo1, unified_up)
        else:
            lo0, la0 = lon_lat[0]
            ring = _square_coords_lon_lat_alt(la0, lo0, unified_up, half_size_m=90.0)

        desc_parts: list[str] = []
        notes = (feat.get("notes") or "").strip()
        if notes:
            desc_parts.append(notes)
        for p in pts:
            if isinstance(p, dict) and (p.get("comment") or "").strip():
                desc_parts.append(str(p["comment"]).strip())
        desc = " / ".join(dict.fromkeys(desc_parts)) if desc_parts else ""

        pin_lo, pin_la = _centroid_lon_lat(lon_lat)

        parts.append("<Placemark>")
        parts.append(f"<name>{_esc(name)} 区域</name>")
        if desc:
            parts.append(f"<description>{_esc(desc)}</description>")
        parts.append("<Style>")
        parts.append(
            "<PolyStyle><color>7d00ff00</color><outline>1</outline></PolyStyle>"
        )
        parts.append("</Style>")
        parts.append("<Polygon>")
        parts.append("<extrude>1</extrude>")
        parts.append("<altitudeMode>absolute</altitudeMode>")
        parts.append("<outerBoundaryIs><LinearRing>")
        parts.append(f"<coordinates>{ring}</coordinates>")
        parts.append("</LinearRing></outerBoundaryIs>")
        parts.append("</Polygon>")
        parts.append("</Placemark>")

        parts.append("<Placemark>")
        parts.append(f"<name>{_esc(name)}</name>")
        if desc:
            parts.append(f"<description>{_esc(desc)}</description>")
        parts.append("<Style>")
        parts.append(
            '<IconStyle><scale>1.1</scale><Icon><href>http://maps.google.com/mapfiles/kml/pushpin/grn-pushpin.png</href></Icon></IconStyle>'
        )
        parts.append("</Style>")
        parts.append("<Point>")
        parts.append(
            f"<coordinates>{pin_lo:.7f},{pin_la:.7f},{unified_up:.2f}</coordinates>"
        )
        parts.append("<altitudeMode>absolute</altitudeMode>")
        parts.append("</Point>")
        parts.append("</Placemark>")

    parts.append("</Document></kml>")
    xml = "\n".join(parts).encode("utf-8")
    if xml.count(b"<Placemark>") < 1:
        return None
    return xml
