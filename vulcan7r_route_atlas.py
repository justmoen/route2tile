import argparse
from collections import deque
import hashlib
import json
import math
import os
import re
import shutil
import struct
import subprocess
import time
import uuid

import cv2
import gpxpy
import mercantile
import numpy as np
from pyproj import Transformer
import rasterio
from rasterio.transform import from_origin
import requests
from shapely.geometry import LineString, box
from shapely.ops import transform, unary_union

# ==============================================================================
# CONFIGURATION (TUNED FOR B&G VULCAN 7R)
# ==============================================================================
DEFAULT_INPUT_FILE_PATH = "WaypointsRoutesTracks.usr"
OUTPUT_DIR = os.path.abspath("./imc_input_geotiffs")
SHARED_IMC_DIR = os.path.expanduser("~/route2tile_shared/imc_input_geotiffs")
TILE_CACHE_DIR = os.path.abspath("./tile_cache")
WINNER_CACHE_DIR = os.path.abspath("./winner_tile_cache")
DEFAULT_ATLAS_NAME = "Vulcan7R_Passage"
WINE_IMC_PATH = os.path.expanduser(
    "~/Applications/Wineskin/Insight Map Creator.app"
)

TARGET_ZOOM = 16  # Ideal detail-to-size balance for Vulcan 7R (800x480)
ATLAS_ZOOM_LEVELS = (14, 15, 16, 17, 18)
AT5_OUTPUT_DIR = os.path.expanduser("~/Downloads")
OPEN_WATER_BUFFER_NM = 0.5  # Narrow corridor in deep open water
COASTAL_BUFFER_NM = 1.5  # Wider corridor when near land or reefs

TILE_SOURCES = {
    "Esri_World_Imagery": (
        "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
    ),
    "Google_Satellite": "https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
    "Bing_VirtualEarth": "https://ecn.t3.tiles.virtualearth.net/tiles/a{q}.jpeg?g=1",
    "Allen Coral Atlas": "https://tiles.allencoralatlas.org/tiles/{z}/{x}/{y}.jpg",
    "Sentinel-2 Cloudless": "https://tiles.maps.eox.at/wmts/1.0.0/s2cloudless-2019_3857/default/g/{z}/{y}/{x}.jpg",
}

HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================
def quadkey(x, y, z):
    qk = ""
    for i in range(z, 0, -1):
        digit = 0
        mask = 1 << (i - 1)
        if (x & mask) != 0:
            digit += 1
        if (y & mask) != 0:
            digit += 2
        qk += str(digit)
    return qk


def fetch_tile_image(source_template, x, y, z, source_name=None):
    url = source_template.format(x=x, y=y, z=z, q=quadkey(x, y, z))
    cache_key = source_name or hashlib.sha256(source_template.encode("utf-8")).hexdigest()[:12]
    cache_key = re.sub(r"[^A-Za-z0-9_-]+", "_", cache_key).strip("_")
    cache_path = os.path.join(TILE_CACHE_DIR, cache_key, str(z), str(x), f"{y}.jpg")

    if os.path.isfile(cache_path):
        cached = cv2.imread(cache_path, cv2.IMREAD_COLOR)
        if cached is not None:
            return cached

    try:
        resp = requests.get(url, headers=HEADERS, timeout=6)
        if resp.status_code != 200 or len(resp.content) < 3500:
            return None
        arr = np.frombuffer(resp.content, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is not None:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            cv2.imwrite(cache_path, img)
        return img
    except Exception:
        return None


def has_land_or_obstacle(img):
    """Keeps tiles with land, reefs, or shallow shoals; discards uniform deep water."""
    if img is None:
        return False

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edge_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    hue_std = np.std(h)
    val_std = np.std(v)

    non_deep_water_ratio = np.mean((h < 85) | (h > 135) | (v > 140))

    if non_deep_water_ratio > 0.04:
        return True
    if edge_score > 120.0 and val_std > 12.0:
        return True
    if hue_std > 14.0:
        return True

    return False

def score_source_quality(img):
  """Scores coastal/underwater clarity while penalizing fog, glare, and turbid runoff."""
  if img is None:
    return -1.0

  hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
  h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

  # 1. Dense cloud / bright sun glare mask
  cloud_mask = (s < 35) & (v > 215)
  cloud_ratio = np.mean(cloud_mask)
  if cloud_ratio > 0.20:
    return 0.0

  # 2. Thin fog / coastal haze mask (low saturation, medium-high brightness)
  haze_mask = (s < 45) & (v > 145) & (v <= 215)
  haze_ratio = np.mean(haze_mask)
  if haze_ratio > 0.35:
    return 0.0

  # 3. Dark channel prior (standard dehazing metric: foggy pixels have high min(B,G,R))
  min_channel = np.min(img, axis=2)
  haze_intensity = np.mean(min_channel) / 255.0  # Higher = hazier/milkier

  # 4. Sharpness & tonal contrast (fog crushes standard deviation)
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
  contrast = np.std(gray) / 128.0  # Normalized contrast; foggy tiles have low std

  # Penalize haze heavily using exponential decay on dark-channel brightness
  haze_penalty = np.exp(-3.5 * haze_intensity)
  clean_ratio = max(0.0, 1.0 - (cloud_ratio * 2.0 + haze_ratio * 1.2))

  return sharpness * contrast * haze_penalty * clean_ratio


# ==============================================================================
# NATIVE LOWRANCE / B&G .USR PARSER (SUPPORTS USR v4, v5, v6)
# ==============================================================================
class BinaryReader:
    def __init__(self, data):
        self.data = data
        self.pos = 0

    def read(self, n):
        if n < 0 or self.pos + n > len(self.data):
            raise EOFError(
                f"Unexpected end of .usr file at byte {self.pos}/{len(self.data)} (requested {n} bytes)."
            )
        chunk = self.data[self.pos : self.pos + n]
        self.pos += n
        return chunk

    def u8(self):
        return struct.unpack("<B", self.read(1))[0]

    def i16(self):
        return struct.unpack("<h", self.read(2))[0]

    def u16(self):
        return struct.unpack("<H", self.read(2))[0]

    def i32(self):
        return struct.unpack("<i", self.read(4))[0]

    def u32(self):
        return struct.unpack("<I", self.read(4))[0]

    def u64(self):
        return struct.unpack("<Q", self.read(8))[0]

    def f32(self):
        return struct.unpack("<f", self.read(4))[0]

    def str_byte_len(self):
        """Reads a 4-byte length followed by that exact number of bytes (UTF-8/ASCII or UTF-16LE)."""
        byte_len = self.i32()
        if byte_len <= 0:
            return ""
        raw = self.read(byte_len)
        # Detect UTF-16LE (alternating null bytes on ASCII) vs UTF-8/Latin-1
        if byte_len >= 2 and byte_len % 2 == 0 and raw[1::2].count(0) > (byte_len // 4):
            return raw.decode("utf-16-le", errors="ignore").rstrip("\x00").strip()
        return raw.decode("utf-8", errors="ignore").rstrip("\x00").strip()


def _mm_to_lat_lon(lon_mm, lat_mm):
    """Converts Lowrance Mercator-millimeter integers into WGS84 (lon, lat)."""
    earth_r = 6356752.3142
    lon_deg = (lon_mm / earth_r) * (180.0 / math.pi)
    lat_rad = 2.0 * math.atan(math.exp(lat_mm / earth_r)) - (math.pi / 2.0)
    lat_deg = lat_rad * (180.0 / math.pi)
    return (lon_deg, lat_deg)


def _dump_hex_header(data, length=96):
    print("\n--- Diagnostic Hex Dump (First 96 Bytes of .usr File) ---")
    for offset in range(0, min(length, len(data)), 16):
        chunk = data[offset : offset + 16]
        hex_part = " ".join(f"{b:02x}" for b in chunk)
        ascii_part = "".join(chr(b) if 32 <= b < 127 else "." for b in chunk)
        print(f"  0x{offset:04x}: {hex_part:<47}  {ascii_part}")
    print("---------------------------------------------------------\n")


def parse_usr_routes_native(usr_path):
    """
    Parses routes directly from Lowrance / B&G .usr files (v4, v5, v6),
    stopping right after the routes section so USR v6 trails never cause an error.
    """
    with open(usr_path, "rb") as f:
        data = f.read()

    r = BinaryReader(data)
    fmt_ver = r.u32()
    stream_ver = r.u32()
    print(f"Detected USR format v{fmt_ver} (data stream v{stream_ver})")

    if fmt_ver not in (4, 5, 6):
        _dump_hex_header(data)
        raise ValueError(
            f"Unsupported USR format v{fmt_ver}. Export as GPX or USR v4/v5/v6 from your chartplotter."
        )

    try:
        # File Header
        _title = r.str_byte_len()
        _date_str = r.str_byte_len()
        _creation_date = r.u32()
        _creation_time = r.u32()
        _unknown_hdr = r.u8()
        _serial_num = r.u32()
        _file_desc = r.str_byte_len()

        # Waypoints Section
        num_waypoints = r.i32()
        waypoints_by_uuid = {}
        waypoints_by_uid = {}
        waypoints_by_idx = []

        for _ in range(num_waypoints):
            w_uuid = uuid.UUID(bytes_le=r.read(16))
            _uid_unit = r.u32()
            _uid_seq = r.u64()
            w_stream_ver = r.i16()
            w_name = r.str_byte_len()
            if fmt_ver >= 5:
                _uid_unit_dup = r.u32()
            lon_mm = r.i32()
            lat_mm = r.i32()
            _flags = r.u32()
            _icon = r.u16()
            _color = r.u16()
            _desc = r.str_byte_len()
            _alarm_radius = r.f32()
            _w_date = r.u32()
            _w_time = r.u32()
            _w_unk = r.u8()
            _depth = r.f32()
            _loran_gri = r.i32()
            _loran_tda = r.i32()
            _loran_tdb = r.i32()
            if fmt_ver >= 6 and w_stream_ver >= 4:
                _extra = r.read(4)

            coord = _mm_to_lat_lon(lon_mm, lat_mm)
            waypoints_by_uuid[w_uuid] = (w_name, coord)
            waypoints_by_uid[(_uid_unit, _uid_seq)] = (w_name, coord)
            waypoints_by_idx.append((w_name, coord))

        # Routes Section
        num_routes = r.i32()
        choices = []

        for idx in range(1, num_routes + 1):
            _r_uuid = uuid.UUID(bytes_le=r.read(16))
            _r_uid_unit = r.u32()
            _r_uid_seq = r.u64()
            _r_stream_ver = r.i16()
            r_name = r.str_byte_len()
            if fmt_ver >= 5:
                _r_uid_unit_dup = r.u32()
            num_legs = r.i32()
            if fmt_ver <= 4:
                _r_reversed = r.u8()

            pts = []
            for _ in range(num_legs):
                if fmt_ver <= 4:
                    _leg_unit = r.u32()
                    _leg_seq = r.u64()
                    if (_leg_unit, _leg_seq) in waypoints_by_uid:
                        pts.append(waypoints_by_uid[(_leg_unit, _leg_seq)][1])
                else:
                    leg_uuid = uuid.UUID(bytes_le=r.read(16))
                    if leg_uuid in waypoints_by_uuid:
                        pts.append(waypoints_by_uuid[leg_uuid][1])

            if fmt_ver >= 5:
                _route_mystery_u32_1 = r.u32()
                _route_mystery_u32_2 = r.u32()
                _route_mystery_u8 = r.u8()
            _route_mystery_tail = r.u8()

            if len(pts) >= 2:
                label = f"[Route] {r_name if r_name else f'Unnamed Route {idx}'}"
                choices.append((label, LineString(pts), pts))

        return choices

    except Exception as exc:
        _dump_hex_header(data)
        raise RuntimeError(
            f"Failed parsing USR v{fmt_ver} binary layout ({exc}). "
            "Tip: You can also export directly as GPX from your B&G Vulcan 7R."
        ) from exc


# ==============================================================================
# 1. PARSE INPUT FILE, SELECT ROUTE & BUILD ADAPTIVE CORRIDOR
# ==============================================================================
def parse_gpx_choices(gpx_path):
    with open(gpx_path, "r", encoding="utf-8") as f:
        gpx = gpxpy.parse(f)

    choices = []
    for idx, route in enumerate(gpx.routes, start=1):
        pts = [(pt.longitude, pt.latitude) for pt in route.points]
        if len(pts) >= 2:
            name = route.name.strip() if route.name else f"Unnamed Route {idx}"
            choices.append((f"[Route] {name}", LineString(pts), pts))

    for idx, track in enumerate(gpx.tracks, start=1):
        for seg_idx, seg in enumerate(track.segments, start=1):
            pts = [(pt.longitude, pt.latitude) for pt in seg.points]
            if len(pts) >= 2:
                base_name = track.name.strip() if track.name else f"Unnamed Track {idx}"
                label = (
                    f"[Track] {base_name} (Seg {seg_idx})"
                    if len(gpx.tracks[idx - 1].segments) > 1
                    else f"[Track] {base_name}"
                )
                choices.append((label, LineString(pts), pts))

    return choices


def select_route_interactively(choices):
    """Lists available routes/tracks and prompts the user to pick one."""
    if not choices:
        raise ValueError("File must contain at least one route or track with 2+ points.")

    if len(choices) == 1:
        label, line, pts = choices[0]
        print(f"Only 1 route found: {label} ({len(pts)} points)")
        return [line], pts, label, [(label, line, pts)]

    print("\nAvailable routes/tracks in file:")
    for i, (label, _, pts) in enumerate(choices, start=1):
        print(f"  {i}. {label} ({len(pts)} points)")
    print("  0. Process ALL routes/tracks")

    while True:
        raw = input(f"\nSelect a route number [0-{len(choices)}]: ").strip()
        if raw.isdigit():
            selection = int(raw)
            if selection == 0:
                all_pts = [pt for c in choices for pt in c[2]]
                return [c[1] for c in choices], all_pts, "All_Routes", choices
            elif 1 <= selection <= len(choices):
                chosen_label, chosen_line, chosen_pts = choices[selection - 1]
                print(f"Selected: {chosen_label}")
                return [chosen_line], chosen_pts, chosen_label, [
                    (chosen_label, chosen_line, chosen_pts)
                ]
        print("Invalid choice. Please enter a valid number from the list.")


def build_adaptive_corridor(file_path):
    if file_path.lower().endswith(".usr"):
        print(f"Parsing routes directly from {file_path}...")
        choices = parse_usr_routes_native(file_path)
    else:
        choices = parse_gpx_choices(file_path)

    lines, selected_coords, chosen_name, route_choices = select_route_interactively(choices)
    if chosen_name == "All_Routes":
        return None, selected_coords, chosen_name, route_choices

    to_meters = Transformer.from_crs(
        "EPSG:4326", "EPSG:3857", always_xy=True
    ).transform
    to_wgs84 = Transformer.from_crs(
        "EPSG:3857", "EPSG:4326", always_xy=True
    ).transform

    segment_length_m = 3704.0  # 2 NM
    buffered_segments = []

    for idx, route_line in enumerate(lines, start=1):
        line_m = transform(to_meters, route_line)
        num_segments = max(1, int(math.ceil(line_m.length / segment_length_m)))
        print(f"Analyzing route {idx}/{len(lines)} ({num_segments} segments)...")

        for i in range(num_segments):
            start_dist = i * segment_length_m
            end_dist = min((i + 1) * segment_length_m, line_m.length)
            sub_line_m = LineString([
                line_m.interpolate(d)
                for d in np.linspace(start_dist, end_dist, num=5)
            ])

            mid_wgs = transform(to_wgs84, sub_line_m.interpolate(0.5, normalized=True))
            probe_tile = mercantile.tile(mid_wgs.x, mid_wgs.y, 10)
            probe_img = fetch_tile_image(
                TILE_SOURCES["Esri_World_Imagery"],
                probe_tile.x,
                probe_tile.y,
                10,
                "Esri_World_Imagery",
            )

            near_coast = has_land_or_obstacle(probe_img)
            radius_nm = COASTAL_BUFFER_NM if near_coast else OPEN_WATER_BUFFER_NM
            buffered_segments.append(sub_line_m.buffer(radius_nm * 1852.0))

    return (
        transform(to_wgs84, unary_union(buffered_segments)),
        selected_coords,
        chosen_name,
        route_choices,
    )


# ==============================================================================
# 2. AUTOMATIC SOURCE SELECTION
# ==============================================================================
def select_best_source_for_tile(tile):
    """Compare every configured source for one tile and return its best image and name."""
    scored_sources = []

    for source_name, source_url in TILE_SOURCES.items():
        img = fetch_tile_image(source_url, tile.x, tile.y, tile.z, source_name)
        if img is None or not has_land_or_obstacle(img):
            continue

        quality_score = score_source_quality(img)
        if quality_score > 0:
            scored_sources.append((quality_score, source_name, img))

    if not scored_sources:
        return None, None, None

    quality_score, source_name, img = max(scored_sources, key=lambda item: item[0])
    return source_name, img, quality_score


# ==============================================================================
# 3. EXPORT GEOREFERENCED GEOTIFFS + TFW (USING RASTERIO)
# ==============================================================================
def write_tfw_file(tif_path, west, north, px_width, px_height):
    tfw_path = os.path.splitext(tif_path)[0] + ".tfw"
    with open(tfw_path, "w", encoding="utf-8") as f:
        f.write(f"{px_width:.12f}\n")
        f.write("0.0\n")
        f.write("0.0\n")
        f.write(f"{px_height:.12f}\n")
        f.write(f"{west + (px_width / 2.0):.12f}\n")
        f.write(f"{north + (px_height / 2.0):.12f}\n")


def save_geotiff_for_imc(img_bgr, tile, out_dir):
    bounds = mercantile.bounds(tile)
    height, width, _ = img_bgr.shape
    px_width = (bounds.east - bounds.west) / width
    px_height = (bounds.south - bounds.north) / height

    filename = os.path.join(out_dir, f"tile_{tile.z}_{tile.x}_{tile.y}.tif")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    # Transpose from (H, W, 3) to (3, H, W) for Rasterio
    img_bands = np.transpose(img_rgb, (2, 0, 1))

    geo_transform = from_origin(bounds.west, bounds.north, px_width, abs(px_height))

    with rasterio.open(
        filename,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=3,
        dtype=img_bands.dtype,
        crs="EPSG:4326",
        transform=geo_transform,
        compress="JPEG",
        jpeg_quality=80,
    ) as dst:
        dst.write(img_bands)

    write_tfw_file(filename, bounds.west, bounds.north, px_width, px_height)


def winner_cache_dir(source_name):
    safe_source = re.sub(r"[^A-Za-z0-9_-]+", "_", source_name).strip("_")
    return os.path.join(WINNER_CACHE_DIR, safe_source)


def winner_cache_paths(source_name, tile):
    cache_dir = winner_cache_dir(source_name)
    base_name = f"tile_{tile.z}_{tile.x}_{tile.y}"
    return (
        os.path.join(cache_dir, base_name + ".tif"),
        os.path.join(cache_dir, base_name + ".tfw"),
    )


def restore_cached_winner(source_name, tile, out_dir):
    cached_tif, cached_tfw = winner_cache_paths(source_name, tile)
    if not (os.path.isfile(cached_tif) and os.path.isfile(cached_tfw)):
        return False

    base_name = f"tile_{tile.z}_{tile.x}_{tile.y}"
    shutil.copy2(cached_tif, os.path.join(out_dir, base_name + ".tif"))
    shutil.copy2(cached_tfw, os.path.join(out_dir, base_name + ".tfw"))
    return True


def restore_cached_tile(tile, out_dir):
    base_name = f"tile_{tile.z}_{tile.x}_{tile.y}"
    manifest_path = os.path.join(WINNER_CACHE_DIR, base_name + ".json")
    if os.path.isfile(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as manifest_file:
                manifest = json.load(manifest_file)
                source_name = manifest.get("source")
        except (OSError, json.JSONDecodeError):
            source_name = None

        if manifest.get("status") == "skipped":
            return "__skipped__"

        if source_name in TILE_SOURCES and restore_cached_winner(source_name, tile, out_dir):
            return source_name

    for source_name in TILE_SOURCES:
        if restore_cached_winner(source_name, tile, out_dir):
            cache_winner_manifest(source_name, tile)
            return source_name
    return None


def cache_winner(source_name, tile):
    base_name = f"tile_{tile.z}_{tile.x}_{tile.y}"
    source_tif = os.path.join(OUTPUT_DIR, base_name + ".tif")
    source_tfw = os.path.join(OUTPUT_DIR, base_name + ".tfw")
    cached_tif, cached_tfw = winner_cache_paths(source_name, tile)

    os.makedirs(os.path.dirname(cached_tif), exist_ok=True)
    shutil.copy2(source_tif, cached_tif)
    shutil.copy2(source_tfw, cached_tfw)
    cache_winner_manifest(source_name, tile)


def cache_winner_manifest(source_name, tile):
    base_name = f"tile_{tile.z}_{tile.x}_{tile.y}"
    manifest_path = os.path.join(WINNER_CACHE_DIR, base_name + ".json")
    with open(manifest_path, "w", encoding="utf-8") as manifest_file:
        json.dump({"source": source_name, "z": tile.z, "x": tile.x, "y": tile.y}, manifest_file)


def cache_skipped_tile(tile):
    base_name = f"tile_{tile.z}_{tile.x}_{tile.y}"
    manifest_path = os.path.join(WINNER_CACHE_DIR, base_name + ".json")
    with open(manifest_path, "w", encoding="utf-8") as manifest_file:
        json.dump({"status": "skipped", "z": tile.z, "x": tile.x, "y": tile.y}, manifest_file)


def format_eta(seconds):
    if seconds is None:
        return "calculating..."
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m {seconds:02d}s"
    if minutes:
        return f"{minutes:d}m {seconds:02d}s"
    return f"{seconds:d}s"


# ==============================================================================
# 4. MACOS APPLESCRIPT AUTOMATION FOR INSIGHT MAP CREATOR IN WINE
# ==============================================================================
def to_wine_z_path(mac_path):
    r"""Converts a macOS POSIX path to Wine's Z:\ Windows path format."""
    win_slashes = mac_path.replace("/", "\\")
    return f"Z:{win_slashes}"

def to_z_path(mac_path):
    return f"Z:{mac_path}/"

def make_atlas_description(chosen_name, input_path):
    route_name = re.sub(r"^\[(?:Route|Track)\]\s*", "", chosen_name).strip()
    route_name = re.sub(r"\s*\(Seg\s+\d+\)\s*$", "", route_name).strip()

    if not route_name or route_name == "All_Routes":
        input_name = os.path.splitext(os.path.basename(input_path))[0]
        route_name = f"All Routes - {input_name or 'Route'}"

    route_name = re.sub(r"[^A-Za-z0-9 _().+-]+", "_", route_name).strip(" _")
    if not route_name:
        route_name = DEFAULT_ATLAS_NAME

    unique_key = f"{os.path.abspath(input_path)}:{chosen_name}"
    unique_suffix = hashlib.sha256(unique_key.encode("utf-8")).hexdigest()[:6].upper()
    return f"M={route_name}_{unique_suffix}"


def escape_applescript_string(value):
    return value.replace("\\", "\\\\").replace('"', '\\"')


def check_keystroke_permission():
    probe = 'tell application "System Events" to keystroke ""'
    result = subprocess.run(
        ["/usr/bin/osascript", "-e", probe],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        return True

    print("AppleScript keystroke permission is blocked.")
    print("  1. Open System Settings -> Privacy & Security -> Accessibility.")
    print("  2. Enable the app hosting the terminal: Terminal or Visual Studio Code.")
    print("  3. Also add /usr/bin/osascript if macOS lists it separately.")
    print("  4. Open Privacy & Security -> Automation -> System Events.")
    print("  5. Enable the app hosting the terminal.")
    print("  6. Fully quit and reopen Terminal or Visual Studio Code.")
    print("  7. Run this script again and approve any macOS prompt.")
    if result.stderr.strip():
        print(f"  macOS response: {result.stderr.strip()}")
    return False


def sync_shared_imc_folder(mac_input_dir):
    os.makedirs(SHARED_IMC_DIR, exist_ok=True)

    for filename in os.listdir(SHARED_IMC_DIR):
        if filename.lower().endswith((".tif", ".tfw")):
            os.remove(os.path.join(SHARED_IMC_DIR, filename))

    copied = 0
    for filename in os.listdir(mac_input_dir):
        if filename.lower().endswith((".tif", ".tfw")):
            shutil.copy2(
                os.path.join(mac_input_dir, filename),
                os.path.join(SHARED_IMC_DIR, filename),
            )
            copied += 1

    os.chmod(SHARED_IMC_DIR, 0o755)
    print(f"Shared IMC folder: {SHARED_IMC_DIR} ({copied} files)")


def automate_imc_in_wine(imc_exe_path, mac_input_dir, atlas_name):
    if not os.path.exists(imc_exe_path):
        print(f"IMC not found at: {imc_exe_path}")
        return

    if not check_keystroke_permission():
        print("IMC was not launched because macOS denied keystroke access.")
        return

    sync_shared_imc_folder(mac_input_dir)
    wine_input_path = to_wine_z_path(SHARED_IMC_DIR)
    if imc_exe_path.lower().endswith(".app"):
        wineskin_prefix = os.path.join(
            imc_exe_path,
            "Contents",
            "SharedSupport",
            "prefix",
            "drive_c",
        )
        wineskin_root = os.path.dirname(wineskin_prefix)
        dosdevices_dir = os.path.join(wineskin_root, "dosdevices")
        shared_drive = os.path.join(dosdevices_dir, "m:")
        os.makedirs(dosdevices_dir, exist_ok=True)
        if os.path.lexists(shared_drive):
            if os.path.islink(shared_drive) or os.path.isfile(shared_drive):
                os.unlink(shared_drive)
            else:
                raise IsADirectoryError(f"Cannot replace Wine drive mapping: {shared_drive}")
        os.symlink(SHARED_IMC_DIR, shared_drive)

        prefix_input_dir = os.path.join(wineskin_prefix, "imc_input_geotiffs")
        os.makedirs(prefix_input_dir, exist_ok=True)

        for filename in os.listdir(prefix_input_dir):
            if filename.lower().endswith((".tif", ".tfw")):
                os.remove(os.path.join(prefix_input_dir, filename))
        for filename in os.listdir(SHARED_IMC_DIR):
            if filename.lower().endswith((".tif", ".tfw")):
                shutil.copy2(
                    os.path.join(SHARED_IMC_DIR, filename),
                    os.path.join(prefix_input_dir, filename),
                )

        wine_input_path = "M:\\"

    print(f"Launching IMC in Wine with input path: {wine_input_path}")
    if imc_exe_path.lower().endswith(".app"):
        subprocess.Popen(["open", "-n", "-a", imc_exe_path])
    else:
        subprocess.Popen(["wine", imc_exe_path])

    time.sleep(6)

    atlas_name = escape_applescript_string(atlas_name)
    wine_input_path = escape_applescript_string(wine_input_path)
    os.makedirs(AT5_OUTPUT_DIR, exist_ok=True)
    save_work_directory = escape_applescript_string(
        to_z_path(AT5_OUTPUT_DIR)
    )
    applescript = f"""
    tell application "System Events"
        delay 1.0
        repeat with p in (every process whose background only is false)
            try
                repeat with w in (every window of p)
                    if name of w contains "Insight Map Creator" then
                        set frontmost of p to true
                        perform action "AXRaise" of w
                        exit repeat
                    end if
                end repeat
            end try
        end repeat

        -- Switch to Raster Mode: Alt+V (View) -> P (Processing Modes) -> R (Raster Mode)
        key down option
        keystroke "v"
        key up option
        delay 0.4
        keystroke "p"
        delay 0.4
        keystroke "r"
        delay 0.8

        -- Set Atlas Description to M=[Route] and keep Atlas Version 12.
        key down option
        keystroke "o"
        key up option
        delay 0.4
        key code 124
        delay 0.4
        keystroke tab
        keystroke tab
        keystroke return
        delay 0.4
        keystroke "{atlas_name}"
        delay 0.2
        key code 53
        delay 0.6

        -- Add C:/imc_input_geotiffs/ as the source folder.
        keystroke tab
        keystroke tab
        keystroke tab
        delay 0.2
        keystroke space
        delay 0.8
        keystroke "C:/imc_input_geotiffs/"
        delay 0.4
        key down option
        keystroke "c"
        key up option
        delay 0.8

        -- Set resolution minimum and maximum.
        keystroke tab
        keystroke tab
        keystroke tab
        keystroke "0"
        delay 0.4
        keystroke tab
        delay 0.4
        keystroke "3"
        delay 0.4

        -- Select Downloads as the work directory.
        keystroke tab
        keystroke "{save_work_directory}"
        delay 0.4

        -- Trigger Build.
        keystroke tab
        keystroke tab
        keystroke space
    end tell
    """

    try:
        subprocess.run(["/usr/bin/osascript", "-e", applescript], check=True)
        print("Automated IMC configuration and Build trigger sent.")
    except subprocess.CalledProcessError as e:
        print(f"AppleScript automation encountered an issue: {e}")
        print("Enable keystroke control for osascript in System Settings:")
        print("  Privacy & Security -> Accessibility")
        print("  Add/enable Terminal or Visual Studio Code")
        print("  Privacy & Security -> Automation -> System Events")
        print("Then fully quit and reopen the terminal app.")


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
def build_route_corridor(route_line):
    to_meters = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True).transform
    to_wgs84 = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True).transform
    segment_length_m = 3704.0
    buffered_segments = []
    line_m = transform(to_meters, route_line)
    num_segments = max(1, int(math.ceil(line_m.length / segment_length_m)))

    for i in range(num_segments):
        start_dist = i * segment_length_m
        end_dist = min((i + 1) * segment_length_m, line_m.length)
        sub_line_m = LineString([
            line_m.interpolate(d)
            for d in np.linspace(start_dist, end_dist, num=5)
        ])
        mid_wgs = transform(to_wgs84, sub_line_m.interpolate(0.5, normalized=True))
        probe_tile = mercantile.tile(mid_wgs.x, mid_wgs.y, 10)
        probe_img = fetch_tile_image(
            TILE_SOURCES["Esri_World_Imagery"],
            probe_tile.x,
            probe_tile.y,
            10,
            "Esri_World_Imagery",
        )
        near_coast = has_land_or_obstacle(probe_img)
        radius_nm = COASTAL_BUFFER_NM if near_coast else OPEN_WATER_BUFFER_NM
        buffered_segments.append(sub_line_m.buffer(radius_nm * 1852.0))

    return transform(to_wgs84, unary_union(buffered_segments))


def process_route_atlas(corridor_poly, chosen_name, input_path):
    dynamic_atlas_name = make_atlas_description(chosen_name, input_path)
    print(f"\nIMC Atlas Description: {dynamic_atlas_name}")
    print(f"IMC target zoom levels: {', '.join(map(str, ATLAS_ZOOM_LEVELS))}")

    for f in os.listdir(OUTPUT_DIR):
        if f.endswith((".tif", ".tfw")):
            os.remove(os.path.join(OUTPUT_DIR, f))

    minx, miny, maxx, maxy = corridor_poly.bounds
    candidate_tiles = list(
        mercantile.tiles(minx, miny, maxx, maxy, zooms=[TARGET_ZOOM])
    )
    intersecting_tiles = [
        t for t in candidate_tiles
        if corridor_poly.intersects(box(*mercantile.bounds(t)))
    ]
    print(f"Scanning {len(intersecting_tiles)} candidate tiles in corridor...")

    kept_count = 0
    skipped_water = 0
    cached_winner_count = 0
    source_counts = {source_name: 0 for source_name in TILE_SOURCES}
    uncached_tile_times = deque(maxlen=20)

    for processed_count, t in enumerate(intersecting_tiles, start=1):
        remaining_count = len(intersecting_tiles) - processed_count
        remaining_percent = (
            (remaining_count / len(intersecting_tiles) * 100.0)
            if intersecting_tiles else 0.0
        )

        cached_source_name = restore_cached_tile(t, OUTPUT_DIR)
        if cached_source_name == "__skipped__":
            skipped_water += 1
        elif cached_source_name is not None:
            kept_count += 1
            cached_winner_count += 1
            source_counts[cached_source_name] += 1
        else:
            uncached_start = time.monotonic()
            winner_source_name, winner_img, _ = select_best_source_for_tile(t)
            if winner_source_name is None:
                skipped_water += 1
                cache_skipped_tile(t)
            else:
                save_geotiff_for_imc(winner_img, t, OUTPUT_DIR)
                cache_winner(winner_source_name, t)
                kept_count += 1
                source_counts[winner_source_name] += 1
            uncached_tile_times.append(time.monotonic() - uncached_start)
            time.sleep(0.05)

        average_tile_seconds = (
            sum(uncached_tile_times) / len(uncached_tile_times)
            if uncached_tile_times else None
        )
        eta_text = format_eta(
            average_tile_seconds * remaining_count
            if average_tile_seconds is not None else None
        )
        print(
            f"\r  Processed {processed_count}/{len(intersecting_tiles)} files "
            f"({remaining_percent:.1f}% remaining) - ETA: {eta_text}",
            end="",
            flush=True,
        )

    print("\n\nTile Processing Summary:")
    print(f"  - Kept {kept_count} land/reef GeoTIFF tiles in: {OUTPUT_DIR}")
    print(f"  - Restored {cached_winner_count} winner tiles from cache.")
    print(f"  - Skipped {skipped_water} open-water tiles.")
    print("  - Winning source per tile:")
    for source_name, count in source_counts.items():
        print(f"      {source_name}: {count}")

    if kept_count > 0:
        automate_imc_in_wine(WINE_IMC_PATH, OUTPUT_DIR, dynamic_atlas_name)
    else:
        print("No land or reef tiles detected along this route; skipping IMC build.")


def main():
    parser = argparse.ArgumentParser(
        description="Build B&G Vulcan 7R satellite imagery overlays from a .usr or .gpx route file."
    )
    parser.add_argument(
        "input_file",
        nargs="?",
        default=DEFAULT_INPUT_FILE_PATH,
        help="Path to the input .usr or .gpx file (default: WaypointsRoutesTracks.usr)",
    )
    args = parser.parse_args()
    input_path = os.path.expanduser(args.input_file)

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(TILE_CACHE_DIR, exist_ok=True)
    os.makedirs(WINNER_CACHE_DIR, exist_ok=True)

    _, _, chosen_name, route_choices = build_adaptive_corridor(input_path)

    for index, (route_label, route_line, _) in enumerate(route_choices, start=1):
        print(f"\n=== Processing route {index}/{len(route_choices)}: {route_label} ===")
        corridor_poly = build_route_corridor(route_line)
        process_route_atlas(corridor_poly, route_label, input_path)


if __name__ == "__main__":
    main()
