"""NISAR GCOV remote access helpers: ASF signed URLs and ranged HDF5 reads (no full download)."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "_pylib"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import h5py  # noqa: E402
import fsspec  # noqa: E402
import requests  # noqa: E402

from config import TOKEN_FILE  # noqa: E402
TOKEN = TOKEN_FILE.read_text().strip()
CMR = "https://cmr.earthdata.nasa.gov/search/granules.umm_json"
GCOV = "C2854338529-ASF"


def signed_url(url):
    for _ in range(6):
        r = requests.get(url, headers={"Authorization": f"Bearer {TOKEN}"}, allow_redirects=False,
                         timeout=60, stream=True)
        if r.status_code not in (301, 302, 303, 307):
            raise RuntimeError(f"ASF {r.status_code}: {r.text[:300]}")
        url = r.headers["Location"]
        if "asf.alaska.edu" not in url and "earthdatacloud.nasa.gov" not in url:
            return url
    raise RuntimeError("too many redirects")


def open_remote(url, block=8 << 20):
    fs = fsspec.filesystem("http")
    f = fs.open(signed_url(url), mode="rb", block_size=block, cache_type="blockcache")
    return h5py.File(f, "r")


def search(point=None, bbox=None, page_size=200, extra=None):
    p = {"collection_concept_id": GCOV, "page_size": page_size}
    if point:
        p["point"] = f"{point[0]},{point[1]}"
    if bbox:
        p["bounding_box"] = ",".join(map(str, bbox))
    if extra:
        p.update(extra)
    items = requests.get(CMR, params=p, timeout=120).json().get("items", [])
    out = []
    for it in items:
        u = it["umm"]
        aa = {a["Name"]: a["Values"] for a in u.get("AdditionalAttributes", [])}
        url = next(r["URL"] for r in u["RelatedUrls"] if r.get("Type") == "GET DATA" and r["URL"].startswith("https") and r["URL"].endswith(".h5"))
        out.append({"id": u["GranuleUR"], "url": url, "start": u["TemporalExtent"]["RangeDateTime"]["BeginningDateTime"],
                    "bwA": aa.get("FREQUENCY_A_RANGE_BANDWIDTH", [None])[0], "bwB": aa.get("FREQUENCY_B_RANGE_BANDWIDTH", [None])[0],
                    "polA": aa.get("FREQUENCY_A_POLARIZATION_CONCAT", [None])[0], "polB": aa.get("FREQUENCY_B_POLARIZATION_CONCAT", [None])[0],
                    "dir": aa.get("ASCENDING_DESCENDING", [None])[0], "track": aa.get("TRACK_NUMBER", [None])[0],
                    "frame": aa.get("FRAME_NUMBER", [None])[0],
                    "size": sum((a.get("SizeInBytes") or 0) for a in u.get("DataGranule", {}).get("ArchiveAndDistributionInformation", [])[:1])})
    return out
