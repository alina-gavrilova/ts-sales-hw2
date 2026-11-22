"""Модели прогнозирования, скользящая проверка и метрики (ДЗ 2)."""
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingRegressor
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import kpss

# Параметры по частотам. Горизонты заданы в условии, остальное выбрано мной:
# n_test - сколько последних наблюдений уходит в тест, period - длина недели
# в торговых днях (из ДЗ 1), ma_window - окно скользящего среднего для бейзлайна.
CONFIG = {
    "D": dict(label="день", title="Дневной ряд", horizons=[1, 7, 31, 61], n_test=90,
              period=6, ma_window=6, lags=list(range(12)), windows=[6, 24], use_dow=True),
    "W": dict(label="неделя", title="Недельный ряд", horizons=[1, 4, 8], n_test=16,
              period=None, ma_window=4, lags=[0, 1, 2, 3], windows=[4, 8], use_dow=False),
    "MS": dict(label="месяц", title="Месячный ряд", horizons=[1, 2], n_test=4,
               period=None, ma_window=2, lags=[0, 1], windows=[2], use_dow=False),
}
MODEL_ORDER = ["Naive", "MA", "ETS", "SARIMA", "GBoost"]
COLORS = {"Naive": "gray", "MA": "tab:olive", "ETS": "tab:blue",
          "SARIMA": "tab:green", "GBoost": "tab:red"}


# ------------------------------------------------------------------ данные
def load_series(path):
    """Читает csv из ДЗ 1 (колонки date, quantity)."""
    s = pd.read_csv(path, index_col="date", parse_dates=True)["quantity"]
    return s.astype(float)


def check_series(s):
    """Быстрая проверка индекса: порядок, дубли, пропуски, шаги между датами."""
    gaps = s.index.to_series().diff().dropna().dt.days.value_counts().sort_index()
    return {
        "n": len(s),
        "начало": s.index[0].date(),
        "конец": s.index[-1].date(),
        "индекс возрастает": bool(s.index.is_monotonic_increasing),
        "дубликатов в индексе": int(s.index.duplicated().sum()),
        "пропусков в значениях": int(s.isna().sum()),
        "шаги между датами (дней: сколько раз)": {int(k): int(v) for k, v in gaps.head(6).items()},
    }


def split_info(n, key):
    """Размеры обучения и теста и число прогнозов на каждом горизонте."""
    cfg = CONFIG[key]
    hmax = max(cfg["horizons"])
    first = n - cfg["n_test"] - hmax
    return {
        "наблюдений": n,
        "тест": cfg["n_test"],
        "макс. горизонт": hmax,
        "обучение до первого прогноза": first + 1,
        "тест не короче макс. горизонта": cfg["n_test"] >= hmax,
        "прогнозов на каждый горизонт": cfg["n_test"],
    }


# ------------------------------------------------------------------ модели
class Naive:
    """Последнее наблюдение."""
    name = "Naive"

    def forecast(self, y, idx, hs):
        return {h: float(y[-1]) for h in hs}


class MovingAverage:
    """Среднее последних window наблюдений."""
    name = "MA"

    def __init__(self, window):
        self.window = window

    def forecast(self, y, idx, hs):
        m = float(np.mean(y[-self.window:]))
        return {h: m for h in hs}


def _fit_ets(y, cfg):
    kw = dict(trend=cfg["trend"], damped_trend=cfg["damped"], seasonal=cfg["seasonal"],
              seasonal_periods=cfg.get("period"))
    try:
        return ExponentialSmoothing(y, initialization_method="estimated", **kw).fit()
    except Exception:
        return ExponentialSmoothing(y, initialization_method="heuristic", **kw).fit()


def select_ets(y, period=None):
    """Выбор конфигурации ETS по AICc. y - только обучающая часть."""
    cands = [dict(name="SES", trend=None, damped=False, seasonal=None),
             dict(name="Holt", trend="add", damped=False, seasonal=None),
             dict(name="Holt (damped)", trend="add", damped=True, seasonal=None)]
    if period and len(y) >= 2 * period + 2:
        cands += [dict(name="Holt-Winters (add)", trend=None, damped=False,
                       seasonal="add", period=period),
                  dict(name="Holt-Winters (add, damped)", trend="add", damped=True,
                       seasonal="add", period=period)]
    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for c in cands:
            try:
                aicc = float(getattr(_fit_ets(y, c), "aicc", np.nan))
            except Exception:
                aicc = np.nan
            rows.append({"конфигурация": c["name"], "AICc": aicc})
    table = pd.DataFrame(rows)
    ok = table["AICc"].replace([np.inf, -np.inf], np.nan).dropna()
    best = cands[int(ok.idxmin())] if len(ok) else cands[0]
    return best, table


class ETS:
    """Экспоненциальное сглаживание (SES / Хольт / Хольт-Винтерс), параметры
    переоцениваются на каждом прогнозе, структура фиксирована заранее."""
    name = "ETS"

    def __init__(self, cfg):
        self.cfg = cfg
        self.failures = 0

    def forecast(self, y, idx, hs):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                path = np.asarray(_fit_ets(y, self.cfg).forecast(max(hs)), dtype=float)
            if not np.isfinite(path).all():
                raise ValueError("нечисловой прогноз")
            return {h: float(path[h - 1]) for h in hs}
        except Exception:
            self.failures += 1
            return Naive().forecast(y, idx, hs)


def choose_d(y):
    """Порядок разности по KPSS: если стационарность отвергается, берём d = 1."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p = kpss(y, regression="c", nlags="auto")[1]
        return 1 if p < 0.05 else 0
    except Exception:
        return 1


def _fit_sarima(y, order, seasonal):
    trend = "c" if order[1] == 0 else None
    return SARIMAX(y, order=order, seasonal_order=seasonal, trend=trend).fit(disp=False)


def select_arima(y, period=None, max_pq=2):
    """Порядок (p, d, q) и сезонная AR-часть по AICc, d фиксируется по KPSS."""
    d = choose_d(y)
    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for p in range(max_pq + 1):
            for q in range(max_pq + 1):
                for P in ((0, 1) if period else (0,)):
                    seasonal = (P, 0, 0, period) if P else (0, 0, 0, 0)
                    try:
                        aicc = float(getattr(_fit_sarima(y, (p, d, q), seasonal), "aicc", np.nan))
                    except Exception:
                        aicc = np.nan
                    rows.append({"p": p, "d": d, "q": q, "P": P, "AICc": aicc})
    table = pd.DataFrame(rows)
    ok = table["AICc"].replace([np.inf, -np.inf], np.nan).dropna()
    if len(ok):
        b = table.loc[ok.idxmin()]
        order, P = (int(b["p"]), d, int(b["q"])), int(b["P"])
    else:
        order, P = (0, d, 0), 0
    seasonal = (P, 0, 0, period) if P else (0, 0, 0, 0)
    return order, seasonal, table.sort_values("AICc").reset_index(drop=True)


class SARIMA:
    """SARIMA с фиксированным порядком, коэффициенты переоцениваются каждый раз."""
    name = "SARIMA"

    def __init__(self, order, seasonal):
        self.order = order
        self.seasonal = seasonal
        self.failures = 0

    def forecast(self, y, idx, hs):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                path = np.asarray(_fit_sarima(y, self.order, self.seasonal).forecast(max(hs)),
                                  dtype=float)
            if not np.isfinite(path).all():
                raise ValueError("нечисловой прогноз")
            return {h: float(path[h - 1]) for h in hs}
        except Exception:
            self.failures += 1
            return Naive().forecast(y, idx, hs)


def make_features(y, idx, lags, windows, use_dow):
    """Признаки в момент t строятся только по y[0..t]: лаги, скользящие
    среднее и std, отношение к скользящему среднему и день недели."""
    s = pd.Series(y)
    f = {}
    for k in lags:
        f[f"lag{k}"] = s.shift(k)
    for w in windows:
        r = s.rolling(w, min_periods=1)
        f[f"mean{w}"] = r.mean()
        f[f"std{w}"] = r.std().fillna(0)
        f[f"ratio{w}"] = s / (r.mean() + 1)
    if use_dow:
        f["dow"] = np.asarray(idx.dayofweek)
    return pd.DataFrame(f)


class GBoost:
    """Градиентный бустинг, прямая стратегия: отдельная модель на каждый горизонт.

    Деревья не умеют экстраполировать уровень, а в ДЗ 1 было видно, что среднее
    растёт, поэтому по умолчанию предсказывается отклонение от скользящего
    среднего (relative=True), а не сам уровень.
    """
    name = "GBoost"

    def __init__(self, lags, windows, use_dow, relative=True, n_estimators=150,
                 learning_rate=0.05, max_depth=2, seed=42):
        self.lags, self.windows, self.use_dow = lags, windows, use_dow
        self.relative = relative
        self.params = dict(n_estimators=n_estimators, learning_rate=learning_rate,
                           max_depth=max_depth, subsample=0.8, random_state=seed)
        self.fallbacks = 0

    def forecast(self, y, idx, hs):
        F = make_features(y, idx, self.lags, self.windows, self.use_dow)
        o = len(y) - 1
        t0 = max(self.lags)
        level = F[f"mean{self.windows[0]}"].to_numpy() if self.relative else np.zeros(len(y))
        out = {}
        for h in hs:
            rows = np.arange(t0, o - h + 1)       # t + h <= o: цель уже известна
            if len(rows) < 3:
                self.fallbacks += 1
                out[h] = float(y[-1])
                continue
            leaf = max(1, min(5, len(rows) // 10))
            m = GradientBoostingRegressor(min_samples_leaf=leaf, **self.params)
            m.fit(F.iloc[rows].to_numpy(), y[rows + h] - level[rows])
            out[h] = float(level[o] + m.predict(F.iloc[[o]].to_numpy())[0])
        return out


def build_models(series, key):
    """Подбирает структуру ETS и SARIMA по данным до первого прогноза
    и собирает список моделей. Возвращает модели и таблицы подбора."""
    cfg = CONFIG[key]
    first = len(series) - cfg["n_test"] - max(cfg["horizons"])
    y_sel = series.to_numpy(float)[:first + 1]
    ets_cfg, ets_tab = select_ets(y_sel, cfg["period"])
    order, seasonal, ar_tab = select_arima(y_sel, cfg["period"],
                                           max_pq=2 if len(y_sel) >= 60 else 1)
    models = [
        Naive(),
        MovingAverage(cfg["ma_window"]),
        ETS(ets_cfg),
        SARIMA(order, seasonal),
        GBoost(cfg["lags"], cfg["windows"], cfg["use_dow"]),
    ]
    chosen = {"ETS": ets_cfg["name"], "SARIMA": f"{order} x {seasonal}"}
    return models, chosen, {"ETS": ets_tab, "SARIMA": ar_tab}


# ------------------------------------------------------- скользящая проверка
def forecast_at(model, series, origin, hs):
    """Прогноз в момент origin (индекс последнего известного наблюдения).
    Модель получает только series[:origin + 1], прогнозы не бывают отрицательными."""
    y = series.to_numpy(float)
    preds = model.forecast(y[:origin + 1], series.index[:origin + 1], hs)
    return {h: max(float(v), 0.0) for h, v in preds.items()}


def rolling_backtest(model, series, key):
    """Расширяющееся окно. Для каждой цели из теста и каждого горизонта h
    прогноз делается из точки (цель - h) по данным до неё включительно, так что
    на всех горизонтах оцениваются одни и те же n_test целей."""
    cfg = CONFIG[key]
    n, hs_all, n_test = len(series), cfg["horizons"], cfg["n_test"]
    first = n - n_test - max(hs_all)
    rows = []
    for o in range(first, n - 1):
        hs = [h for h in hs_all if n - n_test <= o + h <= n - 1]
        if not hs:
            continue
        preds = forecast_at(model, series, o, hs)
        for h in hs:
            rows.append(dict(freq=key, model=model.name, horizon=h,
                             origin=series.index[o], target=series.index[o + h],
                             y_true=float(series.iloc[o + h]), y_pred=preds[h]))
    df = pd.DataFrame(rows)
    counts = df.groupby("horizon").size()
    assert (counts == n_test).all(), counts
    return df


def check_no_leakage(model, series, origin, hs, seed=0):
    """Заменяет всё, что после origin, мусором: прогноз не должен измениться."""
    rng = np.random.default_rng(seed)
    base = forecast_at(model, series, origin, hs)
    broken = series.copy()
    broken.iloc[origin + 1:] = rng.uniform(0, 5 * series.max(), len(series) - origin - 1)
    other = forecast_at(model, broken, origin, hs)
    return all(np.isclose(base[h], other[h]) for h in hs)


# ------------------------------------------------------------------ метрики
def naive_scale(y_train):
    """Средняя абсолютная ошибка наивного прогноза на шаг вперёд по обучению."""
    return float(np.mean(np.abs(np.diff(y_train))))


def compute_metrics(y_true, y_pred, scale):
    """MAE, RMSE, WAPE (%), MASE и ME = mean(факт - прогноз)."""
    e = np.asarray(y_true) - np.asarray(y_pred)
    mae = float(np.mean(np.abs(e)))
    return dict(MAE=mae,
                RMSE=float(np.sqrt(np.mean(e ** 2))),
                WAPE=float(100 * np.sum(np.abs(e)) / np.sum(np.abs(y_true))),
                MASE=mae / scale,
                ME=float(np.mean(e)))


def evaluate(fc, scale):
    """Метрики для каждой пары (модель, горизонт) одного ряда."""
    rows = []
    for (m, h), g in fc.groupby(["model", "horizon"]):
        rows.append(dict(freq=g["freq"].iloc[0], model=m, horizon=h, n=len(g),
                         **compute_metrics(g["y_true"], g["y_pred"], scale)))
    return pd.DataFrame(rows)


def sort_models(df):
    df = df.copy()
    df["freq"] = pd.Categorical(df["freq"], list(CONFIG), ordered=True)
    df["model"] = pd.Categorical(df["model"], MODEL_ORDER, ordered=True)
    return df.sort_values(["freq", "horizon", "model"]).reset_index(drop=True)


def best_models(metrics, metric="MASE"):
    """Лучшая модель на каждой частоте и горизонте и её выигрыш у Naive."""
    rows = []
    for (f, h), g in metrics.groupby(["freq", "horizon"], observed=True):
        b = g.loc[g[metric].idxmin()]
        naive = g.loc[g["model"] == "Naive", metric].iloc[0]
        rows.append(dict(freq=f, horizon=h, лучшая=b["model"], **{metric: b[metric]},
                         **{f"{metric} Naive": naive},
                         **{"выигрыш у Naive, %": 100 * (1 - b[metric] / naive)}))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ графики
def steps(h):
    """Склонение слова «шаг» для подписей: 1 шаг, 2 шага, 7 шагов."""
    if h % 10 == 1 and h % 100 != 11:
        return f"{h} шаг"
    if 2 <= h % 10 <= 4 and not 12 <= h % 100 <= 14:
        return f"{h} шага"
    return f"{h} шагов"


def plot_split(series_by_key):
    fig, axes = plt.subplots(len(series_by_key), 1, figsize=(13, 3.2 * len(series_by_key)))
    for ax, (key, s) in zip(np.atleast_1d(axes), series_by_key.items()):
        cfg = CONFIG[key]
        cut = len(s) - cfg["n_test"]
        ax.plot(s.index[:cut], s.values[:cut], lw=1, label="обучение")
        ax.plot(s.index[cut - 1:], s.values[cut - 1:], lw=1, color="tab:orange", label="тест")
        ax.axvline(s.index[cut], color="gray", ls="--", lw=1)
        ax.set_title(f"{cfg['title']}: разбиение на обучение и тест")
        ax.set_xlabel("Дата")
        ax.set_ylabel("Проданные единицы")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper left")
    fig.tight_layout()


def plot_forecasts(series, fc, key, tail=10):
    """Факт и прогнозы всех моделей, по панели на каждый горизонт."""
    cfg = CONFIG[key]
    hs = cfg["horizons"]
    fig, axes = plt.subplots(len(hs), 1, figsize=(13, 3.3 * len(hs)), sharex=True)
    shown = series.iloc[-(cfg["n_test"] + tail):]
    for ax, h in zip(np.atleast_1d(axes), hs):
        ax.plot(shown.index, shown.values, color="black", lw=1.5, label="факт")
        for m in MODEL_ORDER:
            g = fc[(fc["horizon"] == h) & (fc["model"] == m)].sort_values("target")
            ax.plot(g["target"], g["y_pred"], lw=1, marker="o", ms=2.5, color=COLORS[m], label=m)
        ax.set_title(f"{cfg['title']}: прогноз на {steps(h)} вперёд")
        ax.set_ylabel("Проданные единицы")
        ax.grid(alpha=0.3)
        ax.legend(ncol=6, fontsize=8, loc="upper left")
    np.atleast_1d(axes)[-1].set_xlabel("Дата (на какую дату сделан прогноз)")
    fig.tight_layout()


def plot_metric_by_horizon(metrics, metric, ylabel):
    keys = list(CONFIG)
    fig, axes = plt.subplots(1, len(keys), figsize=(15, 4))
    for ax, key in zip(axes, keys):
        sub = metrics[metrics["freq"] == key]
        for m in MODEL_ORDER:
            g = sub[sub["model"] == m].sort_values("horizon")
            ax.plot(g["horizon"], g[metric], marker="o", color=COLORS[m], label=m)
        ax.set_xticks(CONFIG[key]["horizons"])
        ax.set_title(f"{CONFIG[key]['title']}: {metric} по горизонтам")
        ax.set_xlabel(f"Горизонт, шагов ({CONFIG[key]['label']})")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()


def plot_heatmap(metrics, metric):
    keys = list(CONFIG)
    fig, axes = plt.subplots(1, len(keys), figsize=(15, 3.6),
                             gridspec_kw=dict(width_ratios=[len(CONFIG[k]["horizons"]) + 1 for k in keys]))
    vmax = float(metrics[metric].max())
    for ax, key in zip(axes, keys):
        pv = (metrics[metrics["freq"] == key]
              .pivot(index="model", columns="horizon", values=metric)
              .reindex(MODEL_ORDER))
        im = ax.imshow(pv.values, cmap="YlOrRd", vmin=0, vmax=vmax, aspect="auto")
        ax.set_xticks(range(pv.shape[1]), pv.columns)
        ax.set_yticks(range(pv.shape[0]), pv.index)
        for i in range(pv.shape[0]):
            for j in range(pv.shape[1]):
                v = pv.values[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if v > 0.6 * vmax else "black")
        if key != keys[0]:
            ax.set_yticklabels([])
        ax.set_title(f"{CONFIG[key]['title']}: {metric}")
        ax.set_xlabel(f"Горизонт, шагов ({CONFIG[key]['label']})")
    axes[0].set_ylabel("Модель")
    fig.colorbar(im, ax=axes, shrink=0.8, label=metric)


def plot_wape_overview(metrics):
    """WAPE всех моделей по всем частотам и горизонтам на одном графике."""
    combos = [(k, h) for k in CONFIG for h in CONFIG[k]["horizons"]]
    x = np.arange(len(combos))
    width = 0.8 / len(MODEL_ORDER)
    fig, ax = plt.subplots(figsize=(14, 4.5))
    for i, m in enumerate(MODEL_ORDER):
        vals = [float(metrics[(metrics["freq"] == k) & (metrics["horizon"] == h)
                              & (metrics["model"] == m)]["WAPE"].iloc[0]) for k, h in combos]
        ax.bar(x + (i - (len(MODEL_ORDER) - 1) / 2) * width, vals, width, color=COLORS[m], label=m)
    ax.set_xticks(x, [f"{CONFIG[k]['label']}, h={h}" for k, h in combos], rotation=30)
    ax.set_title("WAPE моделей на всех частотах и горизонтах")
    ax.set_xlabel("Частота ряда и горизонт прогноза")
    ax.set_ylabel("WAPE, %")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(ncol=5)
    fig.tight_layout()
