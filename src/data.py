import glob, os
import numpy as np
import pandas as pd

RAW = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "raw")
SEED = 4101

FAMILY = {
    "BENIGN": "Benign",
    "DoS Hulk": "DoS", "DoS GoldenEye": "DoS", "DoS slowloris": "DoS", "DoS Slowhttptest": "DoS",
    "DDoS": "DDoS",
    "PortScan": "PortScan",
    "FTP-Patator": "BruteForce", "SSH-Patator": "BruteForce",
    "Bot": "Bot",
    "Web Attack Brute Force": "WebAttack", "Web Attack XSS": "WebAttack", "Web Attack Sql Injection": "WebAttack",
    "Infiltration": None, "Heartbleed": None,
}


def load_raw(raw=RAW):
    frames = []
    for f in sorted(glob.glob(os.path.join(raw, "*.csv"))):
        df = pd.read_csv(f, encoding="latin-1", low_memory=False)
        df.columns = df.columns.str.strip()
        for c in df.columns:
            if c != "Label":
                df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
        df["Label"] = (df["Label"].str.replace(r"[^\x20-\x7E]", "", regex=True)
                       .str.replace(r"\s+", " ", regex=True).str.strip().astype("category"))
        df["session"] = os.path.basename(f).split("_pcap")[0]
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["Label"] = df["Label"].astype("category")
    df["session"] = df["session"].astype("category")
    return df


def feature_cols(df):
    return [c for c in df.columns if c not in ("Label", "family", "session")]


def finite(a):
    a = np.asarray(a, dtype=np.float64)
    return np.where(np.isinf(a), np.nan, a)


def row_hash(df, cols, clean_inf=False):
    h = np.zeros(len(df), dtype=np.uint64)
    for c in cols:
        v = pd.Series(finite(df[c])) if clean_inf else df[c]
        h = (h * np.uint64(0x100000001B3)) ^ pd.util.hash_pandas_object(v, index=False).to_numpy()
    return h


def near_identical(df, feats, sample=200_000, tol=1e-5, seed=SEED):
    S = df[feats].sample(min(sample, len(df)), random_state=seed).replace([np.inf, -np.inf], np.nan)
    S = S.loc[:, S.nunique() > 1]
    C = S.corr().abs()
    pairs = []
    for i, a in enumerate(C.columns):
        for b in C.columns[i + 1:]:
            if C.loc[a, b] > 0.99999:
                x, y = df[a].to_numpy(), df[b].to_numpy()
                ok = np.isfinite(x) & np.isfinite(y)
                d = np.abs(x[ok] - y[ok]) / np.maximum(1.0, np.abs(y[ok]))
                if d.max() <= tol:
                    pairs.append((b, a, float(d.max())))
    return pairs


def audit(df):
    feats = feature_cols(df)
    X = df[feats]
    out = {"rows": len(df), "features": len(feats)}
    out["label_counts"] = df["Label"].value_counts().to_dict()
    out["nan"] = {k: int(v) for k, v in X.isna().sum().items() if v}
    out["inf"] = {k: int(v) for k, v in np.isinf(X).sum().items() if v}
    out["negative"] = {k: int(v) for k, v in (X < 0).sum().items() if v}
    out["constant"] = [c for c in feats if X[c].nunique(dropna=False) <= 1]
    hx = row_hash(df, feats)
    hl = row_hash(df, ["Label"])
    out["exact_dup_rows"] = int(pd.Series(hx ^ hl).duplicated().sum())
    out["feature_dup_rows"] = int(pd.Series(hx).duplicated().sum())
    g = pd.DataFrame({"h": hx, "y": df["Label"].astype(str).to_numpy()}).groupby("h")["y"].nunique()
    out["conflicting_groups"] = int((g > 1).sum())
    colhash = {c: pd.util.hash_pandas_object(X[c], index=False).sum() for c in feats}
    seen, ident = {}, []
    for c, h in colhash.items():
        if h in seen and X[c].equals(X[seen[h]]):
            ident.append((c, seen[h]))
        else:
            seen.setdefault(h, c)
    out["identical"] = ident
    return out


def prepare(df, drop_cols):
    feats = [c for c in feature_cols(df) if c not in drop_cols]
    family = df["Label"].astype(str).map(FAMILY)
    fam_code = pd.DataFrame({"f": family.fillna("").to_numpy()})
    h = row_hash(df, feats, clean_inf=True) * np.uint64(0x100000001B3) ^ row_hash(fam_code, ["f"])
    keep = family.notna().to_numpy() & ~pd.Series(h).duplicated().to_numpy()
    out = pd.DataFrame({c: finite(df[c].to_numpy()[keep]).astype(np.float32) for c in feats})
    out["family"] = family.to_numpy()[keep]
    out["session"] = df["session"].to_numpy()[keep]
    out = out.loc[~pd.Series(row_hash(out, feats + ["family"])).duplicated().to_numpy()]
    return out.reset_index(drop=True), feats


def conflict_mask(df, feats):
    h = row_hash(df, feats)
    n = pd.Series(df["family"].to_numpy(), index=h).groupby(level=0).nunique()
    return pd.Series(h).map(n).to_numpy() > 1
