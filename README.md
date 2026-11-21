# route2tile
Take usr or gpx export from chartplotter/app and convert to mbtiles or raster images

Currently support input of usr or gpx file.

Usr file will prompt a selection of routes contained in the usr file.
Output runs automatically through Insight Map Creator application.

Configuration is currently basic.  Modify the py file if you need to change paths.

Dependencies:
GPSBabel (for parsing usr file)
Insight Map Creator Wineskin app at `~/Applications/Wineskin/Insight Map Creator.app`

## Python setup

Create and activate the repository virtual environment, then install its Python dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Run the scripts with the virtual environment interpreter:

```bash
python vulcan7r_route_atlas.py ~/Downloads/WaypointsRoutesTracks.usr
```

At zoom 16, the script evaluates every candidate land or reef tile across all configured
satellite sources. It keeps the highest-scoring image for that individual tile, so the
result can mix Esri, Google, and Bing imagery along the route. Uniform deep-water tiles
are skipped. Winning tiles are cached separately for reuse on later runs.
Progress includes an estimated time remaining based on the rolling average of the most
recent uncached tile downloads.
The Insight Map Creator Atlas Description is set to `M=<route name>_<unique hash>`,
where `<route name>` is the selected route or track name and the hash prevents reuse of
the same description for another route or input file.
When `0. Process ALL routes/tracks` is selected, each route is built as a separate
atlas. Its corridor, GeoTIFF output folder, and `M=<route name>_<unique hash>` Atlas
Description are handled independently.

Atlas output is configured for zoom levels 16, 17, and 18, with project and AT5 files
targeted to `~/Downloads`.
IMC GeoTIFF input is staged in `~/route2tile_shared/imc_input_geotiffs`, which is
accessible to macOS and mirrored into the Wineskin `C:\imc_input_geotiffs` folder.

If `python` or `python3` resolves outside `.venv`, verify the environment with:

```bash
which python
python -c "import cv2, gpxpy, mercantile, numpy, pyproj, rasterio, requests, shapely; print('Dependencies OK')"
```
