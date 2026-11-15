import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

PORT = "Destination Port"


def neg_to_nan(a):
    a = np.asarray(a, dtype=np.float32).copy()
    a[a < 0] = np.nan
    return a


def slog(a):
    return np.sign(a) * np.log1p(np.abs(a))


def preprocessor(feats, use_port=True):
    num = [c for c in feats if c != PORT]
    numeric = Pipeline([
        ("invalid", FunctionTransformer(neg_to_nan, feature_names_out="one-to-one")),
        ("impute", SimpleImputer(strategy="median", add_indicator=True, copy=False)),
        ("slog", FunctionTransformer(slog, feature_names_out="one-to-one")),
        ("scale", StandardScaler(copy=False)),
    ])
    parts = [("num", numeric, num)]
    if use_port and PORT in feats:
        parts.append(("port", OneHotEncoder(max_categories=32, handle_unknown="infrequent_if_exist",
                                            sparse_output=False, dtype=np.float32), [PORT]))
    return ColumnTransformer(parts, verbose_feature_names_out=False)
