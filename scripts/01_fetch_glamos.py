"""Stage 01: download and extract GLAMOS Swiss Glacier Inventories, write a layer report.

Output:
  raw/glamos/<inventory>/...                extracted shapefiles / GeoPackages
  manifest/glamos_layers.txt                every layer with CRS, feature count, columns
"""
import zipfile
from pathlib import Path

from common import http_get, list_vector_layers, load_config


def main():
    cfg = load_config()
    raw = Path(cfg["paths"]["raw_glamos"])
    report_lines = []

    for name, url in cfg["glamos"]["inventories"].items():
        dest = raw / name
        zpath = raw / f"{name}.zip"
        if dest.exists() and any(dest.iterdir()):
            print(f"[skip] {name} already extracted")
        else:
            print(f"[get ] {name} <- {url}")
            try:
                r = http_get(url, stream=True, timeout=300)
            except Exception as e:  # one bad URL must not stop the others
                print(f"[FAIL] {name}: {e}")
                report_lines.append(f"## {name}: DOWNLOAD FAILED ({e})")
                continue
            with open(zpath, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
            dest.mkdir(exist_ok=True)
            with zipfile.ZipFile(zpath) as z:
                z.extractall(dest)
            # GLAMOS zips sometimes nest further zips
            for inner in dest.rglob("*.zip"):
                with zipfile.ZipFile(inner) as z:
                    z.extractall(inner.parent / inner.stem)
            zpath.unlink()

        import pyogrio

        report_lines.append(f"## {name}")
        for path, layer in list_vector_layers(dest):
            info = pyogrio.read_info(path, layer=layer)
            cols = ", ".join(info["fields"])
            report_lines.append(
                f"{path.relative_to(raw)}:{layer or ''}  crs={info['crs']}  "
                f"n={info['features']}  geom={info['geometry_type']}\n    cols: {cols}"
            )

    out = Path(cfg["paths"]["manifest"]) / "glamos_layers.txt"
    out.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"\nLayer report: {out}")


if __name__ == "__main__":
    main()
