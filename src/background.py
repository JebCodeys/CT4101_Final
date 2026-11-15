"""Session background ("noise") analysis and worst-case-background sample weights.

Every attack family in CIC-IDS2017 was staged on one day, against whatever benign
traffic happened to be on the network that day. The amount of benign traffic
differs a lot between sessions, so an attack staged on a quiet afternoon is
separated from benign by the model partly because its background was thin.
These helpers measure that background and re-weight benign flows so every attack
is trained against the heaviest *attack-free* background in the capture.
"""
import numpy as np
import pandas as pd

BENIGN = "Benign"


def short_session(s):
    return (str(s).replace("-WorkingHours", "").replace("-workingHours", "")
            .replace(".pcap_ISCX.csv", "").replace("Infilteration", "Infiltration"))


def background_table(family, session):
    """Per-session counts: benign, each attack family, and benign:attack ratio."""
    ct = pd.crosstab(pd.Series(session, name="session").map(short_session), pd.Series(family, name="family"))
    fam_cols = [c for c in ct.columns if c != BENIGN]
    ct["attacks"] = ct[fam_cols].sum(axis=1)
    ct["attack_family"] = [", ".join(c for c in fam_cols if ct.loc[i, c] > 0) or "(none)" for i in ct.index]
    ct["benign:attack"] = (ct[BENIGN] / ct["attacks"].replace(0, np.nan)).round(2)
    return ct[[BENIGN, "attacks", "attack_family", "benign:attack"] + fam_cols].sort_values(BENIGN, ascending=False)


def reference_background(family, session):
    """Benign flow count of the heaviest session that carries no attack family at all.

    Only attack-free sessions qualify so the reference is not confounded with an
    attack: on an attack day part of the "benign" traffic is response traffic
    to the attack (and, per Engelen et al. 2021, some attack appendix flows).
    """
    tab = background_table(family, session)
    clean = tab[tab["attacks"] == 0]
    ref = clean[BENIGN].idxmax()
    return ref, int(clean.loc[ref, BENIGN])


def worst_case_weights(family, session):
    """Sample weights that raise every session's benign traffic to the reference volume.

    Benign flow in session s gets weight N_ref / N_benign(s); attack flows get 1.
    After weighting, every attack family sits in a background as heavy as the
    heaviest attack-free day, instead of the (often much quieter) background of
    the afternoon it was staged in. Returns (weights, per-session table).
    """
    family = np.asarray(family)
    session = pd.Series(session).map(short_session).to_numpy()
    ref_name, n_ref = reference_background(family, session)
    benign = family == BENIGN
    n_s = pd.Series(session[benign]).value_counts()
    factor = (n_ref / n_s)
    w = np.ones(len(family), dtype=np.float64)
    w[benign] = factor.reindex(session[benign]).to_numpy()
    tab = pd.DataFrame({"benign_flows": n_s, "benign_weight": factor.round(2),
                        "effective_benign": (n_s * factor).round(0).astype(int)}).sort_values("benign_flows", ascending=False)
    tab.attrs["reference"] = (ref_name, n_ref)
    return w, tab


def scenario_weights(family, session, benign_volume, families=None):
    """Importance weights that re-express out-of-fold predictions under a traffic mix.

    benign_volume: number of benign flows in the scenario (e.g. one heavy day).
    families: attack families present (default: all) at their natural dev volume.
    Weighted metrics computed with these weights estimate what the same classifier
    would score on a day with that composition, without retraining.
    """
    family = np.asarray(family)
    w = np.zeros(len(family))
    benign = family == BENIGN
    w[benign] = benign_volume / benign.sum()
    for f in (families or [c for c in np.unique(family) if c != BENIGN]):
        w[family == f] = 1.0
    return w
