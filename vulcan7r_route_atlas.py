import argparse
import math
import os
import re
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
DEFAULT_ATLAS_NAME = "Vulcan7R_Passage"
WINE_IMC_PATH = os.path.expanduser(
    "~/.wine/drive_c/Program Files (x86)/Navico/Insight Map Creator/InsightMapCreator.exe"
)

TARGET_ZOOM = 16  # Ideal detail-to-size balance for Vulcan 7R (800x480)
OPEN_WATER_BUFFER_NM = 0.5  # Narrow corridor in deep open water
COASTAL_BUFFER_NM = 2.5  # Wider corridor when near land or reefs

TILE_SOURCES = {
    "Esri_World_Imagery": (
        "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
    ),
    "Google_Satellite": "https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
    "Bing_VirtualEarth": "https://ecn.t3.tiles.virtualearth.net/tiles/a{q}.jpeg?g=1",
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


def fetch_tile_image(source_template, x, y, z):
    url = source_template.format(x=x, y=y, z=z, q=quadkey(x, y, z))
    try:
        resp = requests.get(url, headers=HEADERS, timeout=6)
        if resp.status_code != 200 or len(resp.content) < 3500:
            return None
        arr = np.frombuffer(resp.content, np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
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
    """Scores sharpness while penalizing clouds, glare, and low-res upscales."""
    if img is None:
        return -1.0

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    cloud_mask = (hsv[:, :, 1] < 30) & (hsv[:, :, 2] > 220)
    cloud_ratio = np.mean(cloud_mask)
    if cloud_ratio > 0.25:
        return 0.0

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    return sharpness * (1.0 - cloud_ratio)


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
        return [line], pts, label

    print("\nAvailable routes/tracks in file:")
    for i, (label, _, pts) in enumerate(choices, start=1):
        print(f"  {i}. {label} ({len(pts)} points)")
    print("  0. Process ALL routes/tracks")

    while True:
        raw = input(f"\nSelect a route number [0-{len(choices)}]: ").strip()
        if raw.isdigit():
            selection = int(raw)
            if selection == 0:
                all_lines = [c[1] for c in choices]
                all_pts = [pt for c in choices for pt in c[2]]
                return all_lines, all_pts, "All_Routes"
            elif 1 <= selection <= len(choices):
                chosen_label, chosen_line, chosen_pts = choices[selection - 1]
                print(f"Selected: {chosen_label}")
                return [chosen_line], chosen_pts, chosen_label
        print("Invalid choice. Please enter a valid number from the list.")


def build_adaptive_corridor(file_path):
    if file_path.lower().endswith(".usr"):
        print(f"Parsing routes directly from {file_path}...")
        choices = parse_usr_routes_native(file_path)
    else:
        choices = parse_gpx_choices(file_path)

    lines, selected_coords, chosen_name = select_route_interactively(choices)

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
            )

            near_coast = has_land_or_obstacle(probe_img)
            radius_nm = COASTAL_BUFFER_NM if near_coast else OPEN_WATER_BUFFER_NM
            buffered_segments.append(sub_line_m.buffer(radius_nm * 1852.0))

    safe_atlas_name = re.sub(r"[^A-Za-z0-9_-]+", "_", chosen_name).strip("_")
    if not safe_atlas_name:
        safe_atlas_name = DEFAULT_ATLAS_NAME

    return transform(to_wgs84, unary_union(buffered_segments)), selected_coords, safe_atlas_name


# ==============================================================================
# 2. AUTOMATIC SOURCE SELECTION
# ==============================================================================
def select_best_source(sample_coords, zoom):
    print("Evaluating satellite sources for clarity and cloud cover...")
    sample_indices = np.linspace(
        0, len(sample_coords) - 1, num=min(6, len(sample_coords)), dtype=int
    )
    scores = {name: [] for name in TILE_SOURCES}

    for idx in sample_indices:
        lon, lat = sample_coords[idx]
        t = mercantile.tile(lon, lat, zoom)
        for name, url_tmpl in TILE_SOURCES.items():
            img = fetch_tile_image(url_tmpl, t.x, t.y, zoom)
            s = score_source_quality(img)
            if s > 0:
                scores[name].append(s)

    best_source = None
    best_avg = -1.0
    for name, s_list in scores.items():
        avg_score = np.mean(s_list) if s_list else -1.0
        print(f"  - {name}: quality score = {avg_score:.1f}")
        if avg_score > best_avg:
            best_avg = avg_score
            best_source = name

    return best_source if (best_source and best_avg > 0) else "Esri_World_Imagery"


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


# ==============================================================================
# 4. MACOS APPLESCRIPT AUTOMATION FOR INSIGHT MAP CREATOR IN WINE
# ==============================================================================
def to_wine_z_path(mac_path):
    r"""Converts a macOS POSIX path to Wine's Z:\ Windows path format."""
    win_slashes = mac_path.replace("/", "\\")
    return f"Z:{win_slashes}"


def automate_imc_in_wine(imc_exe_path, mac_input_dir, atlas_name):
    if not os.path.exists(imc_exe_path):
        print(f"IMC not found at: {imc_exe_path}")
        return

    wine_input_path = to_wine_z_path(mac_input_dir)
    print(f"Launching IMC in Wine with input path: {wine_input_path}")
    subprocess.Popen(["wine", imc_exe_path])

    time.sleep(6)

    applescript = f"""
    tell application "System Events"
        set frontmost of (first process whose name contains "wine") to true
        delay 1.0

        -- Switch to Raster Mode: Alt+V (View) -> P (Processing Modes) -> R (Raster Mode)
        key down option
        keystroke "v"
        key up option
        delay 0.4
        keystroke "p"
        delay 0.4
        keystroke "r"
        delay 0.8

        -- Open Advanced Options: Alt+A (Advanced Options) -> A (Atlas Options)
        key down option
        keystroke "a"
        key up option
        delay 0.4
        keystroke "a"
        delay 0.6

        -- Set Atlas Description and Atlas Version 12
        keystroke "{atlas_name}"
        delay 0.2
        keystroke tab
        delay 0.2
        keystroke "12"
        delay 0.2
        keystroke return
        delay 0.6

        -- Open Raster Options to enable Shaded Relief for Vulcan 7R overlay
        key down option
        keystroke "a"
        key up option
        delay 0.4
        keystroke "r"
        delay 0.6
        keystroke space
        delay 0.2
        keystroke return
        delay 0.6

        -- Trigger Add Folder button (Alt+F)
        key down option
        keystroke "f"
        key up option
        delay 0.8

        -- Paste Wine Z: path into folder selection box and confirm
        keystroke "{wine_input_path}"
        delay 0.4
        keystroke return
        delay 0.8

        -- Trigger Build (Alt+B)
        key down option
        keystroke "b"
        key up option
    end tell
    """

    try:
        subprocess.run(["osascript", "-e", applescript], check=True)
        print("Automated IMC configuration and Build trigger sent.")
    except subprocess.CalledProcessError as e:
        print(f"AppleScript automation encountered an issue: {e}")
        print(
            "Check that Terminal has Accessibility permissions in macOS System Settings."
        )


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
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
    for f in os.listdir(OUTPUT_DIR):
        if f.endswith((".tif", ".tfw")):
            os.remove(os.path.join(OUTPUT_DIR, f))

    corridor_poly, route_coords, dynamic_atlas_name = build_adaptive_corridor(input_path)
    best_source_name = select_best_source(route_coords, TARGET_ZOOM)
    source_url = TILE_SOURCES[best_source_name]

    minx, miny, maxx, maxy = corridor_poly.bounds
    candidate_tiles = list(
        mercantile.tiles(minx, miny, maxx, maxy, zooms=[TARGET_ZOOM])
    )
    print(f"Scanning {len(candidate_tiles)} candidate tiles in corridor...")

    kept_count = 0
    skipped_water = 0

    for t in candidate_tiles:
        tile_poly = box(*mercantile.bounds(t))
        if not corridor_poly.intersects(tile_poly):
            continue

        img = fetch_tile_image(source_url, t.x, t.y, t.z)
        if img is None:
            continue

        if not has_land_or_obstacle(img):
            skipped_water += 1
            continue

        save_geotiff_for_imc(img, t, OUTPUT_DIR)
        kept_count += 1
        time.sleep(0.05)

    print("\nTile Processing Summary:")
    print(f"  - Kept {kept_count} land/reef GeoTIFF tiles in: {OUTPUT_DIR}")
    print(f"  - Skipped {skipped_water} open-water tiles.")

    if kept_count > 0:
        automate_imc_in_wine(WINE_IMC_PATH, OUTPUT_DIR, dynamic_atlas_name)
    else:
        print(
            "No land or reef tiles detected along this route; skipping IMC build."
        )


if __name__ == "__main__":
    main()
