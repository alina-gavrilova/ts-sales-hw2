"""Функции для анализа и предобработки ряда продаж (ДЗ 1)."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import linregress
from statsmodels.tsa.stattools import adfuller, kpss, acf
from statsmodels.tsa.seasonal import seasonal_decompose, STL
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


def load_retail(path):
    """Читает xlsx и приводит типы."""
    df = pd.read_excel(path)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["InvoiceNo"] = df["InvoiceNo"].astype(str)
    df["StockCode"] = df["StockCode"].astype(str)
    return df


def select_product(df, stock_code):
    """Только настоящие продажи товара: без возвратов (счета на C) и без Quantity <= 0."""
    mask = (
        (df["StockCode"] == stock_code)
        & ~df["InvoiceNo"].str.startswith("C")
        & (df["Quantity"] > 0)
    )
    return df.loc[mask].copy()


def aggregate(product_df, freq, value="Quantity"):
    """Сумма Quantity по периодам. Период без транзакций остаётся NaN, а не нулём."""
    s = product_df.set_index("InvoiceDate")[value].resample(freq).sum(min_count=1)
    return s.rename(value)


def drop_partial_edges(s, data_start, data_end, period_freq):
    """Выкидывает неполные периоды по краям. period_freq: 'D', 'W-SUN' или 'M'."""
    per = s.index.to_period(period_freq)
    keep = (per.start_time >= pd.Timestamp(data_start).normalize()) & (
        per.end_time < pd.Timestamp(data_end).normalize()
    )
    return s[keep]


def fill_zero_if_open(s, open_days, period_freq):
    """Пустой период при работающем магазине считаем нулём, иначе оставляем NaN."""
    out = s.copy()
    per_open = pd.Series(1, index=open_days).groupby(
        open_days.to_period(period_freq)
    ).sum()
    for ts in out.index[out.isna()]:
        if per_open.get(ts.to_period(period_freq), 0) > 0:
            out.loc[ts] = 0
    return out


def trim_to_sales_span(s):
    """Обрезает ряд по первой и последней продаже."""
    return s.loc[s.first_valid_index():s.last_valid_index()]


def build_series(df, stock_code):
    """Весь путь от транзакций до трёх рядов {'D', 'W', 'MS'}."""
    start, end = df["InvoiceDate"].min(), df["InvoiceDate"].max()
    open_days = shop_open_days(df)
    prod = select_product(df, stock_code)
    out = {}
    for freq, pfreq in [("D", "D"), ("W", "W-SUN"), ("MS", "M")]:
        s = trim_to_sales_span(aggregate(prod, freq))
        s = drop_partial_edges(s, start, end, pfreq)
        if freq == "D":
            out[freq] = to_trading_calendar(s, open_days)
        else:
            s = fill_zero_if_open(s, open_days, pfreq)
            out[freq] = fill_missing(s, "interpolate") if s.isna().any() else s
    return out


def save_series(series, stock_code, folder):
    """Сохраняет ряды в csv: {'D': ..., 'W': ..., 'MS': ...}. Возвращает пути."""
    names = {"D": "daily", "W": "weekly", "MS": "monthly"}
    paths = {}
    for key, s in series.items():
        path = f"{folder}/sales_{stock_code}_{names[key]}.csv"
        s.to_csv(path, header=["quantity"], index_label="date")
        paths[key] = path
    return paths


def missing_report(s):
    """Сколько пропусков и где они лежат (подряд идущие отрезки)."""
    is_na = s.isna().to_numpy()
    runs, start = [], None
    for i, flag in enumerate(is_na):
        if flag and start is None:
            start = i
        if not flag and start is not None:
            runs.append((s.index[start], s.index[i - 1], i - start))
            start = None
    if start is not None:
        runs.append((s.index[start], s.index[-1], len(s) - start))
    return {
        "n_total": len(s),
        "n_missing": int(is_na.sum()),
        "share_missing": float(is_na.mean()),
        "runs": pd.DataFrame(runs, columns=["start", "end", "length"]),
    }


def shop_open_days(df):
    """Дни, когда была хотя бы одна транзакция по любому товару."""
    return pd.DatetimeIndex(df["InvoiceDate"].dt.normalize().unique()).sort_values()


def split_empty_days(s, open_days):
    """Пустые дни товара: (настоящие нули при работающем магазине, закрытые дни)."""
    empty = s.index[s.isna()]
    is_open = empty.isin(open_days)
    return empty[is_open], empty[~is_open]


def to_trading_calendar(s, open_days):
    """Нули в рабочие дни, закрытые дни выкидываем из ряда."""
    s = s.reindex(s.index.intersection(open_days.union(s.dropna().index)))
    return s.fillna(0)


def weekday_gaps(s):
    """Сколько пустых дней приходится на каждый день недели."""
    na = s[s.isna()]
    counts = na.index.dayofweek.value_counts().reindex(range(7), fill_value=0)
    counts.index = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"]
    return counts


def iqr_outliers(s, k=1.5):
    """Выбросы по правилу Тьюки, возвращает маску и границы."""
    x = s.dropna()
    q1, q3 = x.quantile([0.25, 0.75])
    iqr = q3 - q1
    lo, hi = q1 - k * iqr, q3 + k * iqr
    return (s < lo) | (s > hi), (lo, hi)


def mad_outliers(s, thr=3.5):
    """Выбросы по модифицированному z-score (медиана и MAD)."""
    x = s.dropna()
    med = x.median()
    mad = np.median(np.abs(x - med))
    z = 0.6745 * (s - med) / mad
    return z.abs() > thr, z


def stl_residual_outliers(s, period, thr=3.0):
    """Выбросы по остатку STL: больше thr робастных сигм."""
    r = decompose(s, period, method="stl", robust=True).resid
    sigma = 1.4826 * np.median(np.abs(r - np.median(r)))
    return r.abs() > thr * sigma, r


def inspect_day(product_df, date, top=5):
    """Самые крупные транзакции товара в один день."""
    d = product_df[product_df["InvoiceDate"].dt.normalize() == pd.Timestamp(date)]
    cols = [c for c in ["InvoiceNo", "Quantity", "UnitPrice", "CustomerID", "Country"]
            if c in d.columns]
    return d.sort_values("Quantity", ascending=False)[cols].head(top)


def fill_missing(s, how="zero"):
    """Заполнение пропусков: 'zero', 'interpolate' или 'ffill'."""
    if how == "zero":
        return s.fillna(0)
    if how == "interpolate":
        return s.interpolate("time")
    if how == "ffill":
        return s.ffill()
    raise ValueError(f"неизвестный способ: {how}")


def winsorize(s, lo=None, hi=None):
    """Обрезка значений по границам."""
    return s.clip(lower=lo, upper=hi)


def describe_series(s):
    """Описательные статистики плюс cv, доля нулей, ADI и CV² (по ненулевым значениям)."""
    x = s.dropna()
    d = x.describe().to_dict()
    d["skew"] = x.skew()
    d["kurtosis"] = x.kurt()
    d["cv"] = x.std() / x.mean() if x.mean() else np.nan
    d["share_zero"] = float((x == 0).mean())
    d["total"] = x.sum()
    # ADI и CV^2 нужны для классификации спроса (Syntetos-Boylan)
    nz = x[x > 0]
    d["adi"] = len(x) / len(nz) if len(nz) else np.nan
    d["cv2_nonzero"] = (nz.std() / nz.mean()) ** 2 if len(nz) > 1 else np.nan
    return pd.Series(d)


def classify_demand(adi, cv2):
    """Тип спроса по Syntetos-Boylan (пороги 1.32 и 0.49)."""
    if adi < 1.32 and cv2 < 0.49:
        return "гладкий"
    if adi < 1.32:
        return "нерегулярный, продажи почти каждый день, но объёмы скачут"
    if cv2 < 0.49:
        return "прерывистый"
    return "комковатый, редкие продажи и скачущие объёмы"


def rolling_stats(s, window):
    """Скользящие среднее, std и их отношение."""
    r = s.rolling(window, min_periods=max(2, window // 2))
    out = pd.DataFrame({"mean": r.mean(), "std": r.std()})
    out["cv"] = out["std"] / out["mean"]
    return out


def trend_slope(s):
    """Наклон линейного тренда по номеру периода."""
    x = s.dropna()
    res = linregress(np.arange(len(x)), x.to_numpy(dtype=float))
    return {"slope": res.slope, "pvalue": res.pvalue, "r2": res.rvalue ** 2}


def decompose(s, period, method="stl", model="additive", robust=True):
    """STL или классическая декомпозиция. Считаем по значениям, чтобы нерегулярный индекс не мешал."""
    y = s.to_numpy(dtype=float)
    if method == "stl":
        r = STL(y, period=period, robust=robust).fit()
    else:
        r = seasonal_decompose(y, model=model, period=period)
    mk = lambda a: pd.Series(np.asarray(a, dtype=float), index=s.index)
    return SimpleNamespace(
        observed=mk(r.observed), trend=mk(r.trend),
        seasonal=mk(r.seasonal), resid=mk(r.resid),
    )


def weekday_kruskal(s):
    """Краскел-Уоллис по дням недели (H0: распределения одинаковы)."""
    from scipy.stats import kruskal
    groups = [g.to_numpy() for _, g in s.groupby(s.index.dayofweek) if len(g) > 1]
    stat, p = kruskal(*groups)
    return stat, p


def season_verdict(n_signs, strength, weak=0.3):
    """Словесный вердикт по сезонности: сколько признаков из четырёх сошлось
    и насколько велика сила сезонности."""
    if n_signs >= 3 and strength >= weak:
        return "выражена"
    if n_signs >= 2:
        return "статистически значима, но слабая" if strength < weak else "умеренная"
    return "не подтверждена"


def seasonal_strength(res):
    """Сила сезонности и тренда по Хайндману: max(0, 1 - Var(R) / Var(R + компонента))."""
    resid = res.resid.dropna()
    seas = res.seasonal.loc[resid.index]
    trend = res.trend.loc[resid.index]
    fs = max(0.0, 1 - np.var(resid) / np.var(resid + seas))
    ft = max(0.0, 1 - np.var(resid) / np.var(resid + trend))
    return {"seasonal_strength": fs, "trend_strength": ft}


def acf_at_lags(s, lags):
    """ACF на выбранных лагах и граница 1.96 / sqrt(N)."""
    a = acf(s.dropna(), nlags=max(lags), fft=True)
    bound = 1.96 / np.sqrt(s.notna().sum())
    return pd.Series({l: a[l] for l in lags}, name="acf"), bound


def spectrum_peaks(s, top=5):
    """Самые высокие пики амплитудного спектра, период в отсчётах ряда."""
    y = s.dropna().to_numpy(dtype=float)
    y = y - y.mean()
    n = len(y)
    amp = np.abs(np.fft.rfft(y)) / n * 2
    f = np.fft.rfftfreq(n, d=1)
    amp, f = amp[1:], f[1:]
    idx = np.argsort(amp)[::-1][:top]
    return pd.DataFrame({"period": 1 / f[idx], "amplitude": amp[idx]})


def stationarity_tests(s, regression="c", name=""):
    """ADF (H0: единичный корень) и KPSS (H0: стационарность), вердикт словами."""
    x = s.dropna()
    adf_stat, adf_p, adf_lag, *_ = adfuller(x, regression=regression, autolag="AIC")
    kpss_stat, kpss_p, kpss_lag, _ = kpss(x, regression=regression, nlags="auto")
    adf_ok = adf_p < 0.05
    kpss_ok = kpss_p >= 0.05
    if adf_ok and kpss_ok:
        verdict = "стационарен (оба теста согласны)"
    elif not adf_ok and not kpss_ok:
        verdict = "нестационарен (оба теста согласны)"
    elif adf_ok and not kpss_ok:
        verdict = "тесты расходятся: ADF за стационарность, KPSS против"
    else:
        verdict = "тесты расходятся: ADF против стационарности, KPSS за"
    table = pd.DataFrame(
        {
            "stat": [adf_stat, kpss_stat],
            "p-value": [adf_p, kpss_p],
            "lags": [adf_lag, kpss_lag],
            "H0": ["единичный корень", "стационарность"],
        },
        index=[f"ADF {name}".strip(), f"KPSS {name}".strip()],
    )
    return table, verdict


def make_stationary(s, log=False, d=0, seasonal_lag=None):
    """log1p, разности порядка d, потом сезонная разность."""
    out = s.copy()
    if log:
        out = np.log1p(out)
    for _ in range(d):
        out = out.diff()
    if seasonal_lag:
        out = out.diff(seasonal_lag)
    return out.dropna()


def transform_name(kw):
    """Название преобразования для подписей на графиках."""
    parts = []
    if kw.get("log"):
        parts.append("log1p")
    if kw.get("d") == 1:
        parts.append("первая разность")
    elif kw.get("d"):
        parts.append(f"разность порядка {kw['d']}")
    if kw.get("seasonal_lag"):
        parts.append(f"сезонная разность (лаг {kw['seasonal_lag']})")
    return " + ".join(parts) or "без преобразования"


def choose_transformation(s, candidates=None, alpha=0.05):
    """Перебирает преобразования по порядку и берёт первое, где ADF и KPSS согласны."""
    if candidates is None:
        candidates = [
            dict(log=False, d=0),
            dict(log=True, d=0),
            dict(log=False, d=1),
            dict(log=True, d=1),
        ]
    rows, chosen = [], None
    for kw in candidates:
        z = make_stationary(s, **kw)
        tab, verdict = stationarity_tests(z)
        rows.append({
            **kw,
            "ADF p": tab["p-value"].iloc[0],
            "KPSS p": tab["p-value"].iloc[1],
            "вердикт": verdict,
        })
        ok = tab["p-value"].iloc[0] < alpha and tab["p-value"].iloc[1] >= alpha
        if ok and chosen is None:
            chosen = kw
    return chosen, pd.DataFrame(rows)


def plot_series(s, title, ylabel, label="продажи", marks=None, ax=None,
                figsize=(13, 4)):
    """График ряда. marks: список (серия, подпись, цвет) или (серия, подпись, цвет, 'ring')."""
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    ax.plot(s.index, s.values, lw=1, label=label)
    for pts, lab, color, *style in marks or []:
        pts = pts.dropna()
        if style and style[0] == "ring":
            ax.scatter(pts.index, pts.values, facecolors="none", edgecolors=color,
                       linewidths=1.5, zorder=4, s=90, label=lab)
        else:
            ax.scatter(pts.index, pts.values, color=color, zorder=3, s=25, label=lab)
    ax.set_title(title)
    ax.set_xlabel("Дата")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    ax.legend()
    return ax


def plot_decomposition(res, title, unit):
    """Ряд, тренд, сезонность и остаток на четырёх панелях."""
    fig, axes = plt.subplots(4, 1, figsize=(13, 9), sharex=True)
    parts = [
        ("Ряд", res.observed),
        ("Тренд", res.trend),
        ("Сезонность", res.seasonal),
        ("Остаток", res.resid),
    ]
    for ax, (name, data) in zip(axes, parts):
        ax.plot(data.index, data.values, lw=1)
        ax.set_ylabel(f"{name}, {unit}", fontsize=9)
        ax.grid(alpha=0.3)
    axes[0].set_title(title)
    axes[-1].set_xlabel("Дата")
    fig.tight_layout()
    return fig


def plot_acf_pacf(s, lags, suffix=""):
    """ACF и PACF рядом."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    plot_acf(s.dropna(), lags=lags, ax=axes[0])
    plot_pacf(s.dropna(), lags=lags, ax=axes[1], method="ywm")
    axes[0].set_title(f"ACF {suffix}".strip())
    axes[1].set_title(f"PACF {suffix}".strip())
    for ax in axes:
        ax.set_xlabel("Лаг")
        ax.set_ylabel("Корреляция")
    fig.tight_layout()
    return fig
