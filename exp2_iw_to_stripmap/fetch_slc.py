"""Fetch the VV half of Sentinel-1 Stripmap SLC products from ASF as minimal SAFE folders.

The measurement TIFFs inside ASF zips are deflate-compressed, so they cannot be read by byte
range. This streams the zip through ranged HTTP reads and extracts only what sarsen needs for
VV: manifest.safe, the VV annotation, calibration and noise XMLs, and the VV measurement TIFF.
That skips the VH TIFF and roughly halves each download (about 2.3 GB instead of 4.7 GB).

  python fetch_slc.py scenes.csv E:/SAR_superresolution/exp2_iw_to_stripmap/data
scenes.csv needs columns: granule (CDSE GRD id), start, lon, lat
"""
from __future__ import annotations

import io
import os
import shutil
import sys
import time
import zipfile
from pathlib import Path

import pandas as pd
import requests

ASF = "https://api.daac.asf.alaska.edu/services/search/param"
TOKEN = Path(r"C:\Users\grupp\Desktop\earthdata.txt").read_text().strip()


def resolve_slc(start, lon, lat):
    """ASF SLC granule acquired within a minute of the Stripmap GRD start, at this point."""
    t = pd.Timestamp(start)
    p = {"platform": "SENTINEL-1", "processingLevel": "SLC", "intersectsWith": f"POINT({lon} {lat})",
         "start": (t - pd.Timedelta(minutes=1)).strftime("%Y-%m-%dT%H:%M:%SZ"),
         "end": (t + pd.Timedelta(minutes=1)).strftime("%Y-%m-%dT%H:%M:%SZ"), "output": "jsonlite"}
    res = requests.get(ASF, params=p, timeout=120).json()["results"]
    res = [r for r in res if r["beamMode"] not in ("IW", "EW", "WV")]
    return (res[0]["granuleName"], res[0]["downloadUrl"]) if res else (None, None)


class RangeFile(io.RawIOBase):
    def __init__(self, url):
        self.u, self.p = url, 0
        r = requests.get(url, headers={"Range": "bytes=0-0"}, timeout=60)
        self.n = int(r.headers["Content-Range"].split("/")[-1])

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.p

    def seek(self, o, w=0):
        self.p = o if w == 0 else self.p + o if w == 1 else self.n + o
        return self.p

    def readinto(self, b):
        if self.p >= self.n:
            return 0
        e = min(self.p + len(b), self.n) - 1
        for attempt in range(5):
            try:
                d = requests.get(self.u, headers={"Range": f"bytes={self.p}-{e}"}, timeout=300).content
                break
            except requests.RequestException:
                time.sleep(5 * (attempt + 1))
        b[:len(d)] = d
        self.p += len(d)
        return len(d)


def signed_url(url):
    """Follow ASF redirects (datapool -> sentinel1.asf -> signed CloudFront) with the token."""
    for _ in range(5):
        r = requests.get(url, headers={"Authorization": f"Bearer {TOKEN}"}, allow_redirects=False,
                         timeout=60, stream=True)
        if r.status_code not in (301, 302, 303, 307):
            raise RuntimeError(f"ASF {r.status_code}: {r.text[:200]}")
        url = r.headers["Location"]
        if "asf.alaska.edu" not in url:
            return url
    raise RuntimeError("too many ASF redirects")


def fetch(url, out_root: Path):
    z = zipfile.ZipFile(io.BufferedReader(RangeFile(signed_url(url)), 32 << 20))
    safe = z.namelist()[0].split("/")[0]
    dest = out_root / safe
    done = dest / ".complete"
    if done.exists():
        return dest
    want = [i for i in z.infolist() if not i.is_dir() and (
        i.filename.endswith("manifest.safe") or ("-vv-" in i.filename.split("/")[-1]))]
    for i in want:
        target = out_root / i.filename
        target.parent.mkdir(parents=True, exist_ok=True)
        with z.open(i) as src, open(target, "wb") as dst:
            shutil.copyfileobj(src, dst, 16 << 20)
    done.write_text("ok")
    return dest


def main():
    scenes = pd.read_csv(sys.argv[1])
    out_root = Path(sys.argv[2])
    out_root.mkdir(parents=True, exist_ok=True)
    log = []
    for n, s in enumerate(scenes.itertuples(), 1):
        t0 = time.time()
        try:
            if isinstance(getattr(s, "url", None), str):      # granule already resolved (e.g. paper's list)
                gran, url = s.granule, s.url
            else:
                gran, url = resolve_slc(s.start, s.lon, s.lat)
            if not url:
                log.append({"granule": s.granule, "status": "no_slc"})
                continue
            dest = fetch(url, out_root)
            log.append({"granule": s.granule, "slc": gran, "safe": str(dest), "status": "ok",
                        "sec": round(time.time() - t0)})
        except Exception as e:
            log.append({"granule": s.granule, "status": "error", "error": str(e)[:300]})
        print(f"[{n}/{len(scenes)}] {log[-1]}", flush=True)
        pd.DataFrame(log).to_csv(out_root / "fetch_log.csv", index=False)


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)
