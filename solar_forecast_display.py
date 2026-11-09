"""
Solar energy generation forecasting
===================================
Horizon-based forecasts (1h, 3h, 6h, 12h, 24h ahead) of plant output with an
80% prediction interval.

Pipeline
  1. Data      : synthetic hourly plant + weather history (or your own CSV via --data)
  2. Features  : forecast weather at target time, solar geometry, calendar,
                 recent plant observations (known at issue time)
  3. Models    : one gradient-boosting quantile model per horizon (q10/q50/q90)
  4. Uncertainty: conformalized quantile regression (CQR) on a calibration block
  5. Evaluation: chronological split, compared with two persistence baselines

Run:  python solar_forecast.py                 (synthetic data)
      python solar_forecast.py --data my.csv   (real data, see README)
"""
import argparse

import matplotlib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance

import matplotlib.pyplot as plt  # noqa: E402

HORIZONS = [1, 3, 6, 12, 24]      # hours ahead
QS = (0.1, 0.5, 0.9)              # 80% central interval
TARGET_COVERAGE = 0.8
CLEAR_K = 0.95                    # clear-sky plant output ~ CLEAR_K * cap * GHI_cs/1000


# --------------------------------------------------------------------------- #
# Solar geometry
# --------------------------------------------------------------------------- #
def solar_geometry(idx, lat, lon, tz_hours=5.5):
    """cos(zenith) and Haurwitz clear-sky GHI (W/m2) for hourly timestamps (hour start)."""
    doy = idx.dayofyear.values
    hour = idx.hour.values + 0.5                         # mid-hour
    b = 2 * np.pi * (doy - 81) / 364
    eot = 9.87 * np.sin(2 * b) - 7.53 * np.cos(b) - 1.5 * np.sin(b)   # minutes
    solar_time = hour + (4 * (lon - 15 * tz_hours) + eot) / 60
    ha = np.radians(15 * (solar_time - 12))
    decl = np.radians(23.45) * np.sin(2 * np.pi * (284 + doy) / 365)
    phi = np.radians(lat)
    cosz = np.clip(np.sin(phi) * np.sin(decl) + np.cos(phi) * np.cos(decl) * np.cos(ha), 0, 1)
    ghi_cs = np.where(cosz > 0.01, 1098 * cosz * np.exp(-0.057 / np.maximum(cosz, 0.01)), 0.0)
    return cosz, ghi_cs


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def make_synthetic(lat, lon, cap_mw, start="2023-11-01", end="2026-10-31 23:00", seed=7):
    """Physically-motivated synthetic plant: cloud -> irradiance -> temperature-derated power."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, end, freq="h")
    n = len(idx)
    doy, hour = idx.dayofyear.values, idx.hour.values
    cosz, ghi_cs = solar_geometry(idx, lat, lon)

    # cloud cover (0-1): persistent AR(1) weather regimes + monsoon seasonality
    season = 0.9 * np.exp(-0.5 * ((doy - 300) / 35) ** 2) + 0.4 * np.exp(-0.5 * ((doy - 200) / 50) ** 2)
    x, e = np.zeros(n), rng.normal(0, 0.18, n)
    for i in range(1, n):
        x[i] = 0.96 * x[i - 1] + e[i]
    cloud = 1 / (1 + np.exp(-(1.6 * x + 1.6 * season - 0.9)))

    ghi = ghi_cs * np.clip(1 - 0.8 * cloud ** 2 + rng.normal(0, 0.06, n) * cloud, 0.05, 1.05)

    def ar(sd, phi=0.9):
        z, ee = np.zeros(n), rng.normal(0, sd, n)
        for i in range(1, n):
            z[i] = phi * z[i - 1] + ee[i]
        return z

    temp = (28 + 3.5 * np.sin(2 * np.pi * (doy - 50) / 365) + 4 * np.sin(2 * np.pi * (hour - 9) / 24)
            - 3 * cloud + ar(0.4))
    humidity = np.clip(80 - 1.8 * (temp - 28) + 15 * cloud + ar(1.5), 30, 100)
    wind = np.clip(3.5 + 1.5 * season + ar(0.5), 0.2, None)
    base_dir = np.where((doy > 150) & (doy < 270), 225, np.where((doy > 290) | (doy < 30), 45, 100))
    wdir = (base_dir + ar(8, 0.95)) % 360

    # DC power with temperature derating and AC clipping at plant capacity
    tcell = temp + 0.025 * ghi - 0.8 * wind * ghi / 1000
    p = 1.2 * cap_mw * ghi / 1000 * (1 - 0.004 * (tcell - 25)) * 0.9
    p = p * rng.normal(1.0, 0.02, n)
    p = np.clip(p, 0, cap_mw)
    for d in rng.choice(np.arange(n // 24), 10, replace=False):    # outages / curtailment
        s = d * 24 + rng.integers(8, 13)
        p[s:s + rng.integers(3, 7)] *= 0.0

    return pd.DataFrame(dict(gen_mw=p, cloud=cloud, temp=temp, humidity=humidity,
                             wind_speed=wind, wind_dir=wdir, ghi=ghi), index=idx)


def load_csv(path):
    """CSV columns: timestamp, gen_mw, cloud (0-1), temp, humidity, wind_speed, wind_dir
    Weather columns are treated as the *forecast* available for that hour."""
    df = pd.read_csv(path, parse_dates=["timestamp"]).set_index("timestamp").sort_index()
    return df.asfreq("h").interpolate(limit=3)


def add_solar_terms(df, lat, lon, cap_mw):
    df = df.copy()
    df["cosz"], df["ghi_cs"] = solar_geometry(df.index, lat, lon)
    df["clear_gen"] = np.minimum(cap_mw, CLEAR_K * cap_mw * df["ghi_cs"] / 1000)
    csi = (df["gen_mw"] / df["clear_gen"]).where(df["ghi_cs"] > 100).clip(0, 1.3)
    df["csi"] = csi.ffill().fillna(0.7)           # last observed clear-sky index (carried over night)
    return df


# --------------------------------------------------------------------------- #
# Features (one dataset per horizon)
# --------------------------------------------------------------------------- #
def build_features(df, h, noise, seed=0):
    """Row = one forecast issued at time t for target time T=t+h; indexed by T.
    Weather columns at T act as an NWP forecast: truth + noise that grows with h."""
    rng = np.random.default_rng(seed + h)
    iss, tg = df.iloc[:-h], df.iloc[h:]
    k = len(tg)
    nz = (lambda sd: rng.normal(0, sd, k)) if noise else (lambda sd: np.zeros(k))

    X = pd.DataFrame(index=tg.index)
    X["cloud_fc"] = np.clip(tg["cloud"].values + nz(0.04 + 0.012 * h), 0, 1)
    X["temp_fc"] = tg["temp"].values + nz(0.4 + 0.05 * h)
    X["hum_fc"] = np.clip(tg["humidity"].values + nz(3 + 0.3 * h), 0, 100)
    X["wind_fc"] = np.clip(tg["wind_speed"].values + nz(0.5 + 0.06 * h), 0, None)
    wd = np.radians(tg["wind_dir"].values + nz(15 + 2 * h))
    X["wdir_sin"], X["wdir_cos"] = np.sin(wd), np.cos(wd)
    X["ghi_fc"] = tg["ghi_cs"].values * (1 - 0.8 * X["cloud_fc"].values ** 2)   # irradiance implied by forecast
    X["cosz"], X["ghi_cs"] = tg["cosz"].values, tg["ghi_cs"].values
    X["clear_gen_tgt"] = tg["clear_gen"].values
    X["hour_sin"] = np.sin(2 * np.pi * tg.index.hour / 24)
    X["hour_cos"] = np.cos(2 * np.pi * tg.index.hour / 24)
    X["doy_sin"] = np.sin(2 * np.pi * tg.index.dayofyear / 365)
    X["doy_cos"] = np.cos(2 * np.pi * tg.index.dayofyear / 365)
    # what the plant has actually done up to issue time t
    X["gen_now"] = iss["gen_mw"].values
    X["csi_now"] = iss["csi"].values
    X["gen_mean3"] = df["gen_mw"].rolling(3).mean().iloc[:-h].values
    X["gen_lag24"] = df["gen_mw"].shift(24).iloc[h:].values        # same hour yesterday (<= t for h<=24)

    y = tg["gen_mw"]
    keep = (tg["ghi_cs"].values > 20) & ~X.isna().any(axis=1).values   # daytime, complete rows
    return X[keep], y[keep]


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
def fit_quantiles(X, y):
    return {q: HistGradientBoostingRegressor(
        loss="quantile", quantile=q, max_iter=250, learning_rate=0.06, max_leaf_nodes=15,
        min_samples_leaf=30, l2_regularization=1.0, random_state=0).fit(X, y) for q in QS}


def predict_q(models, X, cap):
    p = np.sort(np.column_stack([models[q].predict(X) for q in QS]), axis=1)   # no quantile crossing
    return np.clip(p, 0, cap).T


def cqr_offset(lo, hi, y, alpha=1 - TARGET_COVERAGE):
    """Conformalized quantile regression: widen/shrink [lo,hi] so calibration coverage is ~1-alpha."""
    s = np.maximum(lo - y, y - hi)
    k = int(np.ceil((len(y) + 1) * (1 - alpha)))
    return np.sort(s)[min(k, len(y)) - 1]


def pinball(y, qpred, q):
    d = y - qpred
    return np.mean(np.maximum(q * d, (q - 1) * d))


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="optional CSV with real plant+weather data")
    ap.add_argument("--lat", type=float, default=13.08)
    ap.add_argument("--lon", type=float, default=80.27)
    ap.add_argument("--cap", type=float, default=10.0, help="plant AC capacity (MW)")
    a = ap.parse_args()

    real = a.data is not None
    df = load_csv(a.data) if real else make_synthetic(a.lat, a.lon, a.cap)
    cap = df["gen_mw"].max() if real else a.cap
    df = add_solar_terms(df, a.lat, a.lon, cap)
    print(f"{len(df)} hourly rows, {df.index[0]} -> {df.index[-1]}, capacity {cap:.1f} MW")

    n = len(df)
    t_cal, t_test = df.index[int(n * 0.65)], df.index[int(n * 0.80)]     # chronological split
    rows, preds, models_h = [], {}, {}

    for h in HORIZONS:
        X, y = build_features(df, h, noise=not real)
        tr, ca, te = X.index < t_cal, (X.index >= t_cal) & (X.index < t_test), X.index >= t_test
        m = fit_quantiles(X[tr], y[tr])
        models_h[h] = (m, X[te], y[te])

        lo_c, _, hi_c = predict_q(m, X[ca], cap)
        off = cqr_offset(lo_c, hi_c, y[ca].values)
        lo, med, hi = predict_q(m, X[te], cap)
        lo_cal, hi_cal = np.clip(lo - off, 0, cap), np.clip(hi + off, 0, cap)
        yt = y[te].values
        base_sp = np.clip(X[te]["csi_now"].values * X[te]["clear_gen_tgt"].values, 0, cap)   # smart persistence
        base_d1 = X[te]["gen_lag24"].values                                                  # same hour yesterday

        mae = lambda p: float(np.mean(np.abs(yt - p)))
        rmse = lambda p: float(np.sqrt(np.mean((yt - p) ** 2)))
        rows.append(dict(
            horizon_h=h, n_test=int(te.sum()),
            MAE_model=mae(med), MAE_smart_persist=mae(base_sp), MAE_yesterday=mae(base_d1),
            RMSE_model=rmse(med), RMSE_smart_persist=rmse(base_sp),
            nMAE_model_pct=100 * mae(med) / cap, nMAE_smart_persist_pct=100 * mae(base_sp) / cap,
            skill_RMSE_vs_persist=1 - rmse(med) / rmse(base_sp),
            coverage_raw=float(np.mean((yt >= lo) & (yt <= hi))),
            coverage_cqr=float(np.mean((yt >= lo_cal) & (yt <= hi_cal))),
            width_raw_MW=float(np.mean(hi - lo)), width_cqr_MW=float(np.mean(hi_cal - lo_cal)),
            pinball_avg=float(np.mean([pinball(yt, p, q) for p, q in zip((lo, med, hi), QS)])),
            cqr_offset_MW=float(off)))
        preds[h] = pd.DataFrame(dict(y=yt, lo=lo_cal, med=med, hi=hi_cal, persist=base_sp), index=X[te].index)

    res = pd.DataFrame(rows).round(4)
    print(res[["horizon_h", "MAE_model", "MAE_smart_persist", "MAE_yesterday", "nMAE_model_pct",
               "skill_RMSE_vs_persist", "coverage_raw", "coverage_cqr", "width_cqr_MW"]].to_string(index=False))

    # ---------------- figures ----------------
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    C = dict(model="#1f6f8b", base="#c0c0c0", band="#8fc1d4", actual="#222222", warn="#d1495b")

    # 1) example forecasts over 4 test days
    t0 = df.index[-1].normalize() - pd.Timedelta(days=4)
    win = df.loc[t0:df.index[-1]]
    fig, axs = plt.subplots(2, 1, figsize=(7.5, 4.5), sharex=True)
    for ax, h in zip(axs, (24, 3)):
        p = preds[h].reindex(win.index).fillna(0.0)      # night hours: output is zero
        ax.fill_between(win.index, p["lo"], p["hi"], color=C["band"], alpha=.6, label="80% interval")
        ax.plot(win.index, p["med"], color=C["model"], lw=2, label="forecast (median)")
        ax.plot(win.index, win["gen_mw"], color=C["actual"], lw=1.3, ls="--", label="actual")
        ax.set_title(f"{h}-hour-ahead forecast", loc="left", fontweight="bold")
        ax.set_ylabel("MW")
    axs[0].legend(ncol=3, frameon=False, loc="upper left")
    fig.tight_layout()

    # 2) accuracy + calibration by horizon
    fig, axs = plt.subplots(1, 2, figsize=(7.5, 3))
    ax = axs[0]
    ax.plot(res.horizon_h, res.nMAE_model_pct, "o-", color=C["model"], lw=2, label="gradient boosting")
    ax.plot(res.horizon_h, res.nMAE_smart_persist_pct, "o-", color="#888888", lw=2, label="smart persistence")
    ax.plot(res.horizon_h, 100 * res.MAE_yesterday / cap, "o--", color=C["base"], lw=2, label="same hour yesterday")
    ax.set_xlabel("forecast horizon (h)")
    ax.set_ylabel("nMAE (% of capacity, daytime)")
    ax.set_xticks(HORIZONS)
    ax.set_ylim(0, 10)
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    ax.set_title("Error vs horizon", loc="left", fontweight="bold")
    ax = axs[1]
    w = 0.35
    xs = np.arange(len(res))
    ax.bar(xs - w / 2, 100 * res.coverage_raw, w, color=C["base"], label="raw quantiles")
    ax.bar(xs + w / 2, 100 * res.coverage_cqr, w, color=C["model"], label="after conformal calibration")
    ax.axhline(80, color=C["warn"], ls="--", lw=1.5)
    ax.text(len(res) - 0.5, 81, "target 80%", color=C["warn"], ha="right", fontsize=9)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{h}h" for h in res.horizon_h])
    ax.set_ylim(50, 110)
    ax.set_ylabel("test coverage (%)")
    ax.legend(frameon=False, loc="upper left", ncol=2, fontsize=9)
    ax.set_title("Interval calibration", loc="left", fontweight="bold")
    fig.tight_layout()

    # 3) what drives the day-ahead forecast
    m24, Xte, yte = models_h[24]
    pi = permutation_importance(m24[0.5], Xte, yte, n_repeats=3, random_state=0,
                                scoring="neg_mean_absolute_error")
    imp = pd.Series(pi.importances_mean, index=Xte.columns).sort_values().tail(8)
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.barh(imp.index, imp.values, color=C["model"])
    ax.set_xlabel("increase in MAE when shuffled (MW)")
    ax.set_title("Day-ahead model: top drivers", loc="left", fontweight="bold")
    fig.tight_layout()
    print("top drivers (24h):", ", ".join(imp.sort_values(ascending=False).index[:4]))
    plt.show()


if __name__ == "__main__":
    main()
