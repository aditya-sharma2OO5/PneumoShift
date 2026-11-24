#!/usr/bin/env python3
"""audit_predictions.py -- metadata/provenance/QC audit of oldckpt_real_v1 predictions.

Scope (deliberately narrow):
  * hashes, stamps, shapes, row counts, finiteness / extreme-logit counts
  * NEVER loads label arrays (header shape only), NEVER computes AUROC or any
    outcome-dependent statistic, so it cannot leak CheXpert results into Task E.
  * Never enables pickle. Anything it cannot match is reported NOT_VERIFIED,
    never guessed.

Usage (PowerShell, from repo root):
  python scripts\\audit_predictions.py --zip "C:\\path\\oldckpt_real_v1_predictions.zip" ^
      --plan ANALYSIS_PLAN_TRANSFER.md --manifest configs\\checkpoint_manifest.json ^
      --ref configs\\nih_test_reference_logits.npz --out outputs\\audit_predictions.txt
  (use --dir <folder> instead of --zip if already extracted)
Exit code 0 only if there are no FAIL results (NOT_VERIFIED does not hide: see summary).
"""
import argparse, hashlib, io, json, re, sys, tempfile, zipfile
from pathlib import Path

import numpy as np

EXP = {
    "run_id": "oldckpt_real_v1",
    "data_source": "real_inference",
    "plan_lf_sha256": "10baae5f06a6e4e33035f2acabaeae48b01ca8386d5ac318e8ddd8b41e465d48",
    "manifest_lf_sha256": "4ae0b9fc751c7453c1308ee5518ae4100ae018fbf922daa269d46ccf92cfbc1f",
    "ref_raw_sha256": "026fb68e71d0cbb944f446a9eb643ef3d7e0c5fb413627c1a6a92a12a246feb8",
    "commit": "d79bc756ba9a60d628674f2af1a683bbc25e2e6e",
    "n_chexpert": 191027,
}
# checkpoint SHA prefixes printed in the Kaggle log (prefix only -> prefix comparison only)
CKPT_PREFIX = {
    ("densenet121", 42): "6e77c6a6cc", ("densenet121", 1337): "c3f6afccac",
    ("densenet121", 2024): "494dd05fd5", ("efficientnet_b0", 42): "dc2b37375b",
    ("efficientnet_b0", 1337): "7772e387d3", ("efficientnet_b0", 2024): "8d8b6f8089",
}
FILE_RE = re.compile(r"(densenet121|efficientnet_b0)_(\d+)", re.I)
HEX64 = re.compile(r"^[0-9a-f]{64}$")
RESULTS, LINES = [], []


def say(s=""):
    print(s)
    LINES.append(s)


def rec(status, name, detail=""):
    RESULTS.append((status, name, detail))
    say(f"[{status}] {name}" + (f" -- {detail}" if detail else ""))


def sha_stream(f, chunk=1 << 20):
    h = hashlib.sha256()
    while True:
        b = f.read(chunk)
        if not b:
            break
        h.update(b)
    return h.hexdigest()


def sha_file(p, lf=False):
    if lf:
        return hashlib.sha256(Path(p).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    with open(p, "rb") as f:
        return sha_stream(f)


def npy_header(zf, name):
    with zf.open(name) as f:
        ver = np.lib.format.read_magic(f)
        if ver == (1, 0):
            shape, _, dt = np.lib.format.read_array_header_1_0(f)
        elif ver == (2, 0):
            shape, _, dt = np.lib.format.read_array_header_2_0(f)
        else:
            return None, None
    return shape, str(dt)


def flat_json(o, prefix=""):
    out = {}
    if isinstance(o, dict):
        for k, v in o.items():
            out.update(flat_json(v, f"{prefix}{k}."))
    elif isinstance(o, list) and len(o) <= 64 and all(not isinstance(x, (dict, list)) for x in o):
        out[prefix.rstrip(".")] = o
    elif isinstance(o, list):
        for i, v in enumerate(o):
            out.update(flat_json(v, f"{prefix}{i}."))
    else:
        out[prefix.rstrip(".")] = o
    return out


def sval(v):
    if isinstance(v, bytes):
        v = v.decode("utf-8", "replace")
    return str(v)


def pick(flat, *subs, anyof=()):
    """keys whose lowercase name contains all of subs (and any of anyof if given)."""
    res = {}
    for k, v in flat.items():
        kl = k.lower()
        if all(s in kl for s in subs) and (not anyof or any(a in kl for a in anyof)):
            res[k] = v
    return res


def check_expected(label, flat, expected, subs, anyof=(), prefix_ok=False):
    hits = pick(flat, *subs, anyof=anyof)
    if not hits:
        rec("NOT_VERIFIED", label, "no matching stamp key found")
        return
    bad = {}
    for k, v in hits.items():
        s = sval(v).strip()
        ok = (s == expected) or (prefix_ok and len(s) >= 7 and expected.startswith(s))
        if not ok:
            bad[k] = s[:80]
    if bad:
        rec("FAIL", label, f"mismatch in {bad}")
    else:
        rec("PASS", label, f"keys={list(hits)}")


def inspect_npz(path):
    info = {"shapes": {}, "stamps": {}, "qc": {}, "ids": {}, "extreme_rows": []}
    with zipfile.ZipFile(path) as zf:
        for n in zf.namelist():
            if n.endswith(".npy"):
                info["shapes"][n[:-4]] = npy_header(zf, n)
    npz = np.load(path, allow_pickle=False)
    cols = list(map(sval, npz["label_cols"].tolist())) if "label_cols" in npz.files else None
    img_ids = npz["image_ids"] if "image_ids" in npz.files else None
    meta_paths = npz["metadata_path"] if "metadata_path" in npz.files else None
    feats = None

    for key, (shape, dt) in info["shapes"].items():
        kl = key.lower()
        sensitive = any(t in kl for t in ("label", "y_true", "target")) and kl != "label_cols"
        size = int(np.prod(shape)) if shape is not None else 0
        is_logit = "logit" in kl
        is_idlike = any(t in kl for t in ("path", "patient", "image_id", "filename")) and (dt or "").lstrip("<>|")[:1] in ("U", "S", "O")
        try:
            if is_logit:
                z_raw = np.asarray(npz[key], dtype=np.float64)
                z = z_raw.ravel()
                fin = np.isfinite(z)
                zf_ = z[fin]
                info["qc"][key] = {
                    "n": int(z.size), "nan": int(np.isnan(z).sum()),
                    "inf": int(np.isinf(z).sum()),
                    "abs_gt_80": int((np.abs(zf_) > 80).sum()),
                    "lt_minus_88": int((zf_ < -88).sum()),
                    "min": float(zf_.min()) if zf_.size else None,
                    "max": float(zf_.max()) if zf_.size else None,
                }
                if z_raw.ndim == 2:
                    bad = np.argwhere(~np.isfinite(z_raw) | (np.abs(z_raw) > 80))
                    if bad.size > 0 and feats is None and "features" in npz.files:
                        feats = npz["features"]
                    for r, c in bad[:50]:
                        r_i, c_i = int(r), int(c)
                        fr = np.asarray(feats[r_i], dtype=np.float64) if feats is not None else None
                        info["extreme_rows"].append({
                            "key": key,
                            "row": r_i,
                            "col": cols[c_i] if cols and c_i < len(cols) else c_i,
                            "logit": float(z_raw[r_i, c_i]),
                            "id": sval(img_ids[r_i]) if img_ids is not None else None,
                            "path": sval(meta_paths[r_i]) if meta_paths is not None else None,
                            "feat_finite": bool(np.isfinite(fr).all()) if fr is not None else None,
                            "feat_absmax": float(np.abs(fr).max()) if fr is not None else None,
                            "feat_all_zero": bool((fr == 0).all()) if fr is not None else None,
                        })
            elif is_idlike and not sensitive:
                arr = npz[key]
                info["ids"][key] = hashlib.sha256("\n".join(map(sval, arr.ravel())).encode()).hexdigest()
            elif (not sensitive) and size <= 64 and not any(t in kl for t in ("score", "feature")):
                arr = npz[key]
                info["stamps"][key] = arr.item() if arr.ndim == 0 else arr.tolist()
        except ValueError as e:  # pickle-required arrays
            info["stamps"][key] = f"<UNREADABLE without pickle: {e}>"
    return info


def audit_token(path, plan_sha, manifest_sha):
    say(f"\n--- token / cohort JSON: {path.name} ---")
    flat = flat_json(json.loads(path.read_text(encoding="utf-8")))
    for k, v in flat.items():
        say(f"    {k} = {str(v)[:100]}")
    if "TOKEN" in path.name.upper() or "GATE" in path.name.upper():
        check_expected("token run_id", flat, EXP["run_id"], ("run",), anyof=("id",))
        check_expected("token plan hash", flat, plan_sha, ("plan",), anyof=("sha", "hash"))
        check_expected("token manifest hash", flat, manifest_sha, ("manifest",), anyof=("sha", "hash"))
        check_expected("token reference-bundle hash", flat, EXP["ref_raw_sha256"], ("ref",), anyof=("sha", "hash"))
        check_expected("token git commit", flat, EXP["commit"], ("commit",), prefix_ok=True)
        if not pick(flat, "gpu") or not pick(flat, "torch"):
            rec("NOT_VERIFIED", "token gpu/torch fields", "key(s) not found")
    return flat


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--zip")
    g.add_argument("--dir")
    ap.add_argument("--plan", default="ANALYSIS_PLAN_TRANSFER.md")
    ap.add_argument("--manifest", default="configs/checkpoint_manifest.json")
    ap.add_argument("--ref", default="configs/nih_test_reference_logits.npz")
    ap.add_argument("--out", default="outputs/audit_predictions.txt")
    a = ap.parse_args()

    say("=== audit_predictions.py (metadata/provenance only; no labels, no AUROC) ===")
    plan_sha = sha_file(a.plan, lf=True)
    man_sha = sha_file(a.manifest, lf=True)
    ref_sha = sha_file(a.ref)
    rec("PASS" if plan_sha == EXP["plan_lf_sha256"] else "FAIL", "local plan LF-SHA", plan_sha)
    rec("PASS" if man_sha == EXP["manifest_lf_sha256"] else "FAIL", "local manifest LF-SHA", man_sha)
    rec("PASS" if ref_sha == EXP["ref_raw_sha256"] else "FAIL", "local reference bundle raw SHA", ref_sha)
    man_text = Path(a.manifest).read_text(encoding="utf-8")
    manifest_hexes = {s for s in re.findall(r"[0-9a-f]{64}", man_text)}
    man_data = json.loads(man_text)
    sha_to_model = {
        str(v["sha256"]).lower(): (str(v["architecture"]).lower(), int(v["seed"]))
        for v in man_data.get("checkpoints", {}).values()
        if "sha256" in v and "architecture" in v and "seed" in v
    }

    tmp = tempfile.TemporaryDirectory()
    files = {}  # name -> Path
    if a.zip:
        zpath = Path(a.zip)
        rec("INFO", "zip SHA-256", f"{sha_file(zpath)}  ({zpath.stat().st_size} bytes)")
        with zipfile.ZipFile(zpath) as zf:
            bad = zf.testzip()
            rec("PASS" if bad is None else "FAIL", "zip CRC test", str(bad))
            for zi in zf.infolist():
                if zi.is_dir():
                    continue
                say(f"    member: {zi.filename}  {zi.file_size} bytes")
                if zi.filename.lower().endswith((".npz", ".json")):
                    dest = Path(tmp.name) / Path(zi.filename).name
                    with zf.open(zi) as src, open(dest, "wb") as dst:
                        while True:
                            b = src.read(1 << 20)
                            if not b:
                                break
                            dst.write(b)
                    files[Path(zi.filename).name] = dest
    else:
        for p in sorted(Path(a.dir).rglob("*")):
            if p.suffix.lower() in (".npz", ".json"):
                files[p.name] = p

    npzs = {n: p for n, p in files.items() if n.lower().endswith(".npz") and "chexpert" in n.lower()}
    other_npz = [n for n in files if n.lower().endswith(".npz") and n not in npzs]
    rec("PASS" if len(npzs) == 6 else "FAIL", "CheXpert prediction files found", f"{len(npzs)} (expected 6): {sorted(npzs)}")
    if other_npz:
        rec("INFO", "other npz members (not audited)", str(other_npz))

    per_file, stamp_common = {}, {}
    for name, p in sorted(npzs.items()):
        say(f"\n--- {name} ---")
        m = FILE_RE.search(name)
        arch, seed = (m.group(1).lower(), int(m.group(2))) if m else (None, None)
        say(f"    raw SHA-256: {sha_file(p)}  size={p.stat().st_size}")
        info = inspect_npz(p)
        per_file[name] = info
        for k, (sh, dt) in info["shapes"].items():
            say(f"    key {k!r}: shape={sh} dtype={dt}")
        stamps = {k: v for k, v in info["stamps"].items()}
        for k, v in stamps.items():
            say(f"    stamp {k} = {str(v)[:100]}")
        flat = {k: v for k, v in stamps.items()}
        check_expected(f"{name}: run_id", flat, EXP["run_id"], ("run",), anyof=("id",))
        check_expected(f"{name}: data_source", flat, EXP["data_source"], ("data_source",))
        check_expected(f"{name}: plan hash", flat, EXP["plan_lf_sha256"], ("plan",), anyof=("sha", "hash"))
        check_expected(f"{name}: manifest hash", flat, EXP["manifest_lf_sha256"], ("manifest",), anyof=("sha", "hash"))
        check_expected(f"{name}: git commit", flat, EXP["commit"], ("commit",), prefix_ok=True)
        # checkpoint identity & hash-to-seed mapping
        ck = [sval(v).strip().lower() for k, v in flat.items() if "checkpoint" in k.lower() and HEX64.match(sval(v).strip().lower())]
        if not ck:
            rec("NOT_VERIFIED", f"{name}: checkpoint hash stamp", "no 64-hex checkpoint stamp found")
        else:
            okm = all(c in manifest_hexes for c in ck)
            okp = (arch, seed) in CKPT_PREFIX and all(c.startswith(CKPT_PREFIX[(arch, seed)]) for c in ck)
            mapped = [sha_to_model.get(c) for c in ck]
            ok_map = all(m == (arch, seed) for m in mapped)
            rec("PASS" if (okm and okp and ok_map) else "FAIL", f"{name}: checkpoint hash in manifest and matches Kaggle-log prefix", f"manifest={okm} prefix={okp} map={mapped}")
        sk = pick(flat, "seed")
        if sk:
            ok = all(str(v) == str(seed) for v in sk.values())
            rec("PASS" if ok else "FAIL", f"{name}: seed stamp == filename seed {seed}", str(sk))
        elif ck and all(sha_to_model.get(c) == (arch, seed) for c in ck):
            rec("PASS", f"{name}: seed via checkpoint-hash-to-seed mapping == {seed}", f"sha={ck[0][:10]}...")
        else:
            rec("NOT_VERIFIED", f"{name}: seed stamp", "no seed key and no unique checkpoint-hash-to-seed match")
        # Pneumonia column index verification
        cols = flat.get("label_cols")
        pidx = flat.get("pneumonia_col_idx")
        if isinstance(cols, list) and isinstance(pidx, int) and 0 <= pidx < len(cols):
            rec("PASS" if cols[pidx] == "Pneumonia" else "FAIL", f"{name}: label_cols[pneumonia_col_idx] == 'Pneumonia'", f"idx={pidx} -> {cols[pidx]!r}")
        else:
            rec("NOT_VERIFIED", f"{name}: label_cols[pneumonia_col_idx] == 'Pneumonia'", f"cols={cols!r}, pidx={pidx!r}")
        for t in ("gpu", "torch"):
            for k, v in pick(flat, t).items():
                stamp_common.setdefault(t, {}).setdefault(sval(v), []).append(name)
        # failed images
        fk = pick(flat, "fail")
        say(f"    failed-image stamps: {fk if fk else 'NONE FOUND (NOT_VERIFIED)'}")
        if not fk:
            rec("NOT_VERIFIED", f"{name}: failed-image count", "no 'fail' stamp; plan's 0.1% rule cannot be checked from this file")
        # row counts (headers only for labels; skip label_cols metadata vector)
        rows = {k: sh[0] for k, (sh, dt) in info["shapes"].items() if sh and len(sh) >= 1 and k.lower() != "label_cols" and any(t in k.lower() for t in ("logit", "score", "label"))}
        bad_rows = {k: n for k, n in rows.items() if n != EXP["n_chexpert"]}
        rec("PASS" if rows and not bad_rows else "FAIL", f"{name}: row counts == {EXP['n_chexpert']}", str(bad_rows or rows))
        # finiteness / extremes
        for k, q in info["qc"].items():
            sev = "FAIL" if (q["nan"] or q["inf"]) else ("INFO" if not (q["abs_gt_80"] or q["lt_minus_88"]) else "WARN")
            rec(sev, f"{name}: logits QC [{k}]", json.dumps(q))
        for ex in info["extreme_rows"]:
            fmax = f"{ex['feat_absmax']:.2f}" if ex["feat_absmax"] is not None else "None"
            say(f"    extreme logit row={ex['row']} col={ex['col']} logit={ex['logit']:.2f} id={ex['id']} path={ex['path']} feat_finite={ex['feat_finite']} feat_absmax={fmax} feat_all_zero={ex['feat_all_zero']}")
        if not info["qc"]:
            rec("NOT_VERIFIED", f"{name}: logit QC", "no key containing 'logit'")

    for t, d in stamp_common.items():
        rec("PASS" if len(d) == 1 else "FAIL", f"{t} stamp identical across all files", str({k: len(v) for k, v in d.items()}))
    # row alignment across models (identifier arrays, if present)
    idkeys = {k for i in per_file.values() for k in i["ids"]}
    for k in sorted(idkeys):
        hs = {i["ids"][k] for i in per_file.values() if k in i["ids"]}
        rec("PASS" if len(hs) == 1 else "FAIL", f"row-alignment identifier array {k!r} identical across files", f"{len(hs)} distinct hash(es)")
    if not idkeys:
        rec("NOT_VERIFIED", "cross-model row alignment", "no path/patient/image identifier arrays found in the npz files")

    for name, p in sorted(files.items()):
        if name.lower().endswith(".json"):
            audit_token(p, plan_sha, man_sha)

    counts = {s: sum(1 for r in RESULTS if r[0] == s) for s in ("PASS", "FAIL", "WARN", "NOT_VERIFIED", "INFO")}
    say(f"\n=== SUMMARY: {counts} ===")
    for s, n, d in RESULTS:
        if s in ("FAIL", "WARN", "NOT_VERIFIED"):
            say(f"  {s}: {n} -- {d[:160]}")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(LINES) + "\n", encoding="utf-8")
    print(f"report written to {out}")
    sys.exit(1 if counts["FAIL"] else 0)


if __name__ == "__main__":
    main()