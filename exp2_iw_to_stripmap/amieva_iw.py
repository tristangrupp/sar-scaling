"""Footprints of the paper's 10 Stripmap scenes (ASF) and same-direction IW RTC within 12 days
(Planetary Computer sentinel-1-rtc). Writes amieva_pairs.csv and amieva_footprints.geojson."""
import os, sys
os.environ.update(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_TIMEOUT="60")
from pathlib import Path
import pandas as pd, geopandas as gpd, requests, planetary_computer as pc
from shapely import wkt
from pystac_client import Client
ROOT = Path(__file__).resolve().parent
ASF = "https://api.daac.asf.alaska.edu/services/search/param"
sc = pd.read_csv(ROOT / "amieva_scenes.csv")
res = requests.get(ASF, params={"granule_list": ",".join(sc.granule), "output": "geojson"}, timeout=120).json()
fp = gpd.GeoDataFrame.from_features(res["features"], crs=4326)
fp = fp[fp.processingLevel == "SLC"][["sceneName", "flightDirection", "startTime", "geometry"]].rename(columns={"sceneName": "granule"})
fp.to_file(ROOT / "amieva_footprints.geojson", driver="GeoJSON")
c = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace)
rows = []
for r in fp.itertuples():
    t = pd.Timestamp(r.startTime)
    items = [i for i in c.search(collections=["sentinel-1-rtc"], intersects=r.geometry.__geo_interface__,
                                 datetime=f"{t - pd.Timedelta(days=12):%Y-%m-%d}/{t + pd.Timedelta(days=12):%Y-%m-%d}").items() if "vv" in i.assets]
    same = [i for i in items if (i.properties.get("sat:orbit_state") or "").upper() == r.flightDirection.upper()]
    best = sorted(same, key=lambda i: abs(pd.Timestamp(i.datetime) - t))
    cov = [round(i.geometry and __import__("shapely.geometry", fromlist=["shape"]).shape(i.geometry).intersection(r.geometry).area / r.geometry.area, 2) for i in best[:3]]
    p = r.geometry.representative_point()
    rows.append({"granule": r.granule, "start": r.startTime, "lon": p.x, "lat": p.y, "dir": r.flightDirection, "n_iw_any": len(items),
                 "n_iw_same": len(same), "iw_id": best[0].id if best else None,
                 "iw_dt_days": abs((pd.Timestamp(best[0].datetime) - t).days) if best else None,
                 "iw_ids_top3": ";".join(i.id for i in best[:3]), "iw_cover_top3": cov})
    print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(ROOT / "amieva_pairs.csv", index=False)
sys.stdout.flush(); os._exit(0)
