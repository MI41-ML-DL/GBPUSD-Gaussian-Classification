#!/usr/bin/env python3
"""Gaussian Classification Model — GBP/USD (той самий код, що в GBPUSD_Gaussian_Classification.ipynb).

Запуск (з активованим .venv):
    python gbpusd_gaussian.py            # таблиці в терміналі + графіки у вікнах (закрийте вікно — з'явиться наступне)
    python gbpusd_gaussian.py --no-show  # лише зберегти графіки в results/, вікна не відкривати

Усі графіки зберігаються в results/figures/, а весь текстовий вивід — у results/output.txt.
"""
import os
import sys
from pathlib import Path

os.chdir(Path(__file__).resolve().parent)          # щоб data/ знаходилась незалежно від того, звідки запущено
SHOW = "--no-show" not in sys.argv
import matplotlib
if not SHOW:
    matplotlib.use("Agg")
import matplotlib.pyplot as _plt
import pandas as _pd

RESULTS = Path("results"); FIG_DIR = RESULTS / "figures"; FIG_DIR.mkdir(parents=True, exist_ok=True)
for _old in FIG_DIR.glob("*.png"):
    _old.unlink()
_LOG = open(RESULTS / "output.txt", "w", encoding="utf-8")
_FIG_N = [0]


class _Tee:
    def __init__(self, *streams): self.streams = streams
    def write(self, s):
        for st in self.streams: st.write(s)
    def flush(self):
        for st in self.streams: st.flush()


sys.stdout = _Tee(sys.__stdout__, _LOG)


def section(title):
    print("\n" + "=" * 100 + f"\n{title}\n" + "=" * 100)


def show_result(obj):
    """Друкує результат, який у ноутбуці відображався б під клітинкою."""
    if hasattr(obj, "data") and obj.__class__.__name__ == "Styler":
        obj = obj.data
    if isinstance(obj, (_pd.DataFrame, _pd.Series)):
        with _pd.option_context("display.max_columns", None, "display.width", 250, "display.max_colwidth", 40):
            print(obj.to_string() if isinstance(obj, _pd.DataFrame) else obj.to_frame().to_string())
    elif obj is not None:
        print(obj)


def show_fig():
    """Зберігає поточний графік у results/figures/ і (якщо не --no-show) показує його у вікні."""
    fig = _plt.gcf()
    _FIG_N[0] += 1
    title = fig._suptitle.get_text() if fig._suptitle is not None else (fig.axes[0].get_title() if fig.axes else "")
    slug = "".join(ch if ch.isalnum() else "_" for ch in title)[:60].strip("_") or "figure"
    path = FIG_DIR / f"{_FIG_N[0]:02d}_{slug}.png"
    fig.savefig(path, bbox_inches="tight")
    print(f"[графік {_FIG_N[0]:02d} збережено: {path}]")
    if SHOW:
        _plt.show()
    _plt.close("all")


section('Gaussian Classification Model — напрямок денної зміни курсу GBP/USD')

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Ellipse, Patch
from matplotlib.lines import Line2D
import seaborn as sns
from scipy import stats
from scipy.linalg import cholesky, solve_triangular
from scipy.special import logsumexp

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.base import clone
from sklearn import metrics

warnings.filterwarnings("ignore", category=FutureWarning)
SEED = 42
np.random.seed(SEED)
pd.set_option("display.float_format", lambda v: f"{v:,.4f}")
pd.set_option("display.width", 160)

# Кольори: фіксовані ролі (↑ — синій, ↓ — помаранчевий, ≈ — бірюзовий)
C_UP, C_DOWN, C_FLAT = "#2a78d6", "#eb6834", "#1baf7a"
C_VIOLET, C_YELLOW, C_MAGENTA = "#6250d6", "#eda100", "#e87ba4"
C_GREY, C_INK, C_INK2 = "#8a8985", "#0b0b0b", "#52514e"
CLASS_COLORS = {0: C_DOWN, 1: C_UP}
CLASS_NAMES = {0: "↓ down", 1: "↑ up"}

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 150, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "axes.edgecolor": "#c9c8c3", "axes.labelcolor": C_INK2, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e5e0",
    "grid.linewidth": 0.6, "xtick.color": C_INK2, "ytick.color": C_INK2, "lines.linewidth": 2,
    "legend.frameon": False, "font.size": 10,
})
print("numpy", np.__version__, "| pandas", pd.__version__, "| sklearn", __import__("sklearn").__version__)


section('1. Отримання та очищення даних')

# Файл шукається в підпапці data/ або поруч із ноутбуком (зручно для Google Colab)
CANDIDATES = [Path("data/GBP_USD Historical Data.csv"), Path("GBP_USD Historical Data.csv")]
DATA_PATH = next((p for p in CANDIDATES if p.exists()), None)
assert DATA_PATH is not None, "Не знайдено 'GBP_USD Historical Data.csv' — покладіть його поруч із ноутбуком або в папку data/"

def load_investing_csv(path: Path) -> pd.DataFrame:
    """Читає CSV, експортований з investing.com, у впорядкований за датою DataFrame."""
    df = pd.read_csv(path, encoding="utf-8-sig", dtype=str)
    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")
    for col in ["Price", "Open", "High", "Low"]:
        df[col] = pd.to_numeric(df[col].str.replace(",", ""), errors="coerce")
    df["Change %"] = pd.to_numeric(df["Change %"].str.rstrip("%"), errors="coerce")
    df = df.drop(columns=["Vol."], errors="ignore")
    return df.sort_values("Date").reset_index(drop=True)

raw = load_investing_csv(DATA_PATH)
print(f"Рядків: {len(raw)},  період: {raw.Date.min():%d.%m.%Y} – {raw.Date.max():%d.%m.%Y}")
show_result(raw.tail())


# --- Перевірки якості даних ---
df = raw.copy()
report = {
    "пропуски (NaN)": int(df[["Price", "Open", "High", "Low"]].isna().sum().sum()),
    "дублікати дат": int(df.Date.duplicated().sum()),
    "дні 1 січня / 25 грудня (неторгові)": int(((df.Date.dt.month == 1) & (df.Date.dt.day == 1) |
                                              (df.Date.dt.month == 12) & (df.Date.dt.day == 25)).sum()),
    "вихідні дні (сб/нд)": int((df.Date.dt.dayofweek >= 5).sum()),
}
bad_hl = (df.High < df[["Open", "Price"]].max(axis=1)) | (df.Low > df[["Open", "Price"]].min(axis=1))
report["рядки з High/Low поза Open/Close"] = int(bad_hl.sum())
chg_check = (df.Price.pct_change() * 100 - df["Change %"]).abs().iloc[1:]
report["розбіжність Change % з Price (макс., п.п.)"] = round(float(chg_check.max()), 4)
show_result(pd.Series(report, name="значення", dtype=object).to_frame())


# --- Очищення ---
# 1) 1 січня та 25 грудня — міжбанківський ринок фактично закритий (мізерний діапазон, «шум»).
holiday = ((df.Date.dt.month == 1) & (df.Date.dt.day == 1)) | ((df.Date.dt.month == 12) & (df.Date.dt.day == 25))
df = df.loc[~holiday].copy()
# 2) Узгоджуємо High/Low з Open/Close (поодинокі помилки джерела): High = max(H,O,C), Low = min(L,O,C).
df["High"] = df[["High", "Open", "Price"]].max(axis=1)
df["Low"] = df[["Low", "Open", "Price"]].min(axis=1)
df = df.reset_index(drop=True)
print(f"Після очищення: {len(df)} торгових днів")


section('2. Ознаки та цільова змінна')

FEATURES = ["ret_0", "ret_1", "ret_2", "mom_5", "mom_20", "log_vol_10", "log_range", "ma_gap_20"]

def build_features(d: pd.DataFrame) -> pd.DataFrame:
    c = d["Price"]
    r = 100 * np.log(c).diff()
    f = pd.DataFrame({"Date": d["Date"]})
    f["ret_0"], f["ret_1"], f["ret_2"] = r, r.shift(1), r.shift(2)
    f["mom_5"] = 100 * np.log(c / c.shift(5))
    f["mom_20"] = 100 * np.log(c / c.shift(20))
    f["log_vol_10"] = np.log(r.rolling(10).std())
    f["log_range"] = np.log((100 * (d["High"] - d["Low"]) / c).clip(lower=0.01))
    f["ma_gap_20"] = 100 * (c / c.rolling(20).mean() - 1)
    f["next_ret"] = r.shift(-1)            # дохідність наступного дня → ціль
    f["target_date"] = d["Date"].shift(-1)
    return f

feat_all = build_features(df)
last_row = feat_all.dropna(subset=FEATURES).iloc[[-1]]          # останній день — без відомої цілі
data = feat_all.dropna(subset=FEATURES + ["next_ret"]).copy()
n_ties = int((data.next_ret == 0).sum())
data = data.loc[data.next_ret != 0].copy()
data["y"] = (data.next_ret > 0).astype(int)

SPLIT = pd.Timestamp("2023-01-01")
train = data.loc[data.target_date < SPLIT].copy()   # і ознаки, і ціль train-рядків — до 2023
test = data.loc[data.Date >= SPLIT].copy()

X_train_raw, y_train = train[FEATURES].to_numpy(), train["y"].to_numpy()
X_test_raw, y_test = test[FEATURES].to_numpy(), test["y"].to_numpy()

scaler = StandardScaler().fit(X_train_raw)          # статистики — лише з train
X_train, X_test = scaler.transform(X_train_raw), scaler.transform(X_test_raw)

print(f"Нульових змін виключено: {n_ties}")
print(f"Train: {len(train)} днів ({train.Date.min():%d.%m.%Y} – {train.Date.max():%d.%m.%Y}),  частка ↑ = {y_train.mean():.3f}")
print(f"Test : {len(test)} днів ({test.Date.min():%d.%m.%Y} – {test.Date.max():%d.%m.%Y}),  частка ↑ = {y_test.mean():.3f}")
show_result(data[FEATURES + ["next_ret", "y"]].describe().T)


section('3. Візуалізація розподілу даних')

# 3.1 Курс GBP/USD, розбиття train/test та ключові події
fig, ax = plt.subplots(figsize=(12, 4.2))
ax.plot(df.Date, df.Price, color=C_UP, lw=1.2)
ax.axvspan(SPLIT, df.Date.max(), color=C_GREY, alpha=0.12, lw=0)
ax.text(SPLIT + pd.Timedelta(days=40), df.Price.max() * 0.995, "TEST", color=C_INK2, fontweight="bold", va="top")
ax.text(pd.Timestamp("2010-03-01"), df.Price.max() * 0.995, "TRAIN", color=C_INK2, fontweight="bold", va="top")
events = {"2016-06-24": "Brexit-референдум", "2020-03-19": "COVID-19", "2022-09-26": "«Міні-бюджет»"}
for d_, label in events.items():
    d_ = pd.Timestamp(d_)
    p = df.loc[df.Date >= d_, "Price"].iloc[0]
    off = {"Brexit-референдум": (60, 40), "«Міні-бюджет»": (-75, -12)}.get(label, (0, -34))
    ax.annotate(label, (d_, p), xytext=off, textcoords="offset points", ha="center", fontsize=9, color=C_INK2,
                arrowprops=dict(arrowstyle="-", color=C_GREY, lw=0.8))
ax.set_title("GBP/USD — денна ціна закриття, 2010–2026"); ax.set_ylim(0.98, 1.76)
ax.set_ylabel("USD за 1 GBP")
ax.xaxis.set_major_locator(mdates.YearLocator(2)); ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
plt.tight_layout(); show_fig()


# 3.2 Розподіл денних дохідностей: гістограма + нормальний закон + QQ-plot
# Повний послідовний ряд дохідностей, до фільтрації за майбутньою ціллю.
r_all = (100 * np.log(df["Price"]).diff()).dropna().to_numpy()
mu, sd = r_all.mean(), r_all.std(ddof=1)
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [1.5, 1]})
ax = axes[0]
bins = np.linspace(-3, 3, 121)
ax.hist(r_all, bins=bins, density=True, color=C_UP, alpha=0.55, edgecolor="#fcfcfb", lw=0.4, label="емпірична густина")
xs = np.linspace(-3, 3, 600)
ax.plot(xs, stats.norm.pdf(xs, mu, sd), color=C_DOWN, label=f"N(μ={mu:.3f}, σ={sd:.3f})")
df_t, loc_t, sc_t = stats.t.fit(r_all)
ax.plot(xs, stats.t.pdf(xs, df_t, loc_t, sc_t), color=C_VIOLET, ls="--", lw=1.6, label=f"Student-t (ν={df_t:.1f})")
ax.set_title("Денні лог-дохідності GBP/USD, %"); ax.set_xlabel("r, %"); ax.legend()
jb = stats.jarque_bera(r_all)
ax.text(0.02, 0.95, f"асиметрія = {stats.skew(r_all):.2f}\nексцес = {stats.kurtosis(r_all):.2f}\nJarque–Bera p = {jb.pvalue:.1e}",
        transform=ax.transAxes, va="top", fontsize=9, color=C_INK2)
ax = axes[1]
(osm, osr), (slope, inter, _) = stats.probplot(r_all, dist="norm")
ax.scatter(osm, osr, s=8, color=C_UP, alpha=0.5, lw=0)
ax.plot(osm, slope * osm + inter, color=C_DOWN, lw=1.6)
ax.set_title("QQ-plot проти нормального розподілу"); ax.set_xlabel("теоретичні квантилі"); ax.set_ylabel("вибіркові квантилі")
plt.tight_layout(); show_fig()


# 3.3 Баланс класів у train і test
fig, ax = plt.subplots(figsize=(7, 3.2))
for i, (name, yy) in enumerate([("train", y_train), ("test", y_test)]):
    up = yy.mean()
    ax.barh(i, up, color=C_UP, height=0.55)
    ax.barh(i, 1 - up, left=up + 0.004, color=C_DOWN, height=0.55)
    ax.text(up / 2, i, f"↑ {up:.1%}  (n={yy.sum()})", ha="center", va="center", color="white", fontweight="bold")
    ax.text(up + (1 - up) / 2, i, f"↓ {1-up:.1%}  (n={len(yy)-yy.sum()})", ha="center", va="center", color="white", fontweight="bold")
ax.set_yticks([0, 1], ["train", "test"]); ax.set_xlim(0, 1); ax.axvline(0.5, color=C_INK, lw=0.8, ls=":")
ax.set_title("Баланс класів: класи майже рівноймовірні"); ax.grid(False); ax.set_xticks([])
plt.tight_layout(); show_fig()


# 3.4 Розподіл кожної ознаки за класами (train) + гауссівські наближення, які «бачить» модель
fig, axes = plt.subplots(2, 4, figsize=(14, 6.2))
for ax, feat in zip(axes.ravel(), FEATURES):
    v = train[feat].to_numpy()
    lo, hi = np.percentile(v, [0.5, 99.5])
    b = np.linspace(lo, hi, 50); grid = np.linspace(lo, hi, 300)
    for k in (0, 1):
        vk = v[y_train == k]
        ax.hist(vk, bins=b, density=True, alpha=0.35, color=CLASS_COLORS[k], edgecolor="none")
        ax.plot(grid, stats.norm.pdf(grid, vk.mean(), vk.std()), color=CLASS_COLORS[k], lw=1.8)
    t_p = stats.ttest_ind(v[y_train == 1], v[y_train == 0], equal_var=False).pvalue
    ax.set_title(feat, fontsize=10); ax.text(0.98, 0.95, f"Welch p={t_p:.2f}", transform=ax.transAxes, ha="right", va="top", fontsize=8, color=C_INK2)
    ax.set_yticks([])
fig.legend([Line2D([], [], color=C_UP), Line2D([], [], color=C_DOWN)], ["↑ наступного дня", "↓ наступного дня"],
           loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02))
fig.suptitle("Розподіли ознак за класами: гістограма + N(μ_k, σ_k)", y=1.06, fontweight="bold")
plt.tight_layout(); show_fig()


# 3.5 Кореляційна матриця ознак (train)
corr = train[FEATURES + ["next_ret"]].corr()
fig, ax = plt.subplots(figsize=(8, 6.4))
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
sns.heatmap(corr, mask=mask, cmap="RdBu_r", vmin=-1, vmax=1, center=0, annot=True, fmt=".2f", annot_kws={"size": 8},
            linewidths=1, linecolor="#fcfcfb", cbar_kws={"shrink": 0.7}, ax=ax)
ax.set_title("Кореляції Пірсона: ознаки та дохідність наступного дня"); ax.grid(False)
plt.tight_layout(); show_fig()


# 3.6 Попарні розподіли ключових ознак за класами (pairplot, train)
pp = train[["ret_0", "mom_5", "log_vol_10", "log_range", "y"]].sample(1500, random_state=SEED).copy()
pp["клас"] = pp.pop("y").map(CLASS_NAMES)
g = sns.pairplot(pp, hue="клас", palette={CLASS_NAMES[1]: C_UP, CLASS_NAMES[0]: C_DOWN}, corner=True, height=2.1,
                 plot_kws=dict(s=9, alpha=0.45, linewidth=0), diag_kws=dict(fill=True, alpha=0.3, linewidth=1.5))
g.figure.suptitle("Попарні розподіли ознак: класи сильно перекриваються", y=1.01, fontweight="bold")
show_fig()


# 3.7 Сезонність і стабільність: частка днів ↑ за днем тижня та в ковзному вікні
fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), gridspec_kw={"width_ratios": [1, 2]})
wd = data.groupby(data.target_date.dt.dayofweek)["y"].agg(["mean", "count"])
ci = 1.96 * np.sqrt(wd["mean"] * (1 - wd["mean"]) / wd["count"])
axes[0].bar(range(len(wd)), wd["mean"], yerr=ci, color=C_UP, width=0.6, error_kw=dict(ecolor=C_INK2, lw=1, capsize=3))
axes[0].axhline(0.5, color=C_INK, lw=0.8, ls=":"); axes[0].set_ylim(0.4, 0.6)
axes[0].set_xticks(range(len(wd)), ["Пн", "Вт", "Ср", "Чт", "Пт"][:len(wd)])
axes[0].set_title("P(↑) за днем тижня цілі (95% ДІ)")
roll = data.set_index("Date")["y"].rolling(250).mean()
axes[1].plot(roll.index, roll, color=C_UP, lw=1.4)
axes[1].axhline(0.5, color=C_INK, lw=0.8, ls=":"); axes[1].axvline(SPLIT, color=C_GREY, lw=1, ls="--")
axes[1].set_title("Частка днів ↑ у ковзному вікні 250 днів"); axes[1].set_ylim(0.38, 0.62)
plt.tight_layout(); show_fig()


# 3.8 Кластеризація волатильності: ACF дохідностей vs ACF |дохідностей|
def acf(x, nlags):
    x = x - x.mean(); den = (x ** 2).sum()
    return np.array([(x[:-k] * x[k:]).sum() / den for k in range(1, nlags + 1)])
lags = 30
fig, ax = plt.subplots(figsize=(12, 3.6))
ax.bar(np.arange(1, lags + 1) - 0.2, acf(r_all, lags), width=0.4, color=C_UP, label="ACF(r) — дохідності")
ax.bar(np.arange(1, lags + 1) + 0.2, acf(np.abs(r_all), lags), width=0.4, color=C_DOWN, label="ACF(|r|) — амплітуда")
band = 1.96 / np.sqrt(len(r_all))
ax.axhspan(-band, band, color=C_GREY, alpha=0.18, lw=0, label="95% межа шуму")
ax.set_xlabel("лаг, днів"); ax.set_title("Автокореляція дохідностей та їх модулів: кластеризація волатильності")
ax.legend(ncol=3, loc="upper right")
plt.tight_layout(); show_fig()


section('4. In-house класифікатор (40%) — Gaussian Discriminant Analysis з нуля')

class GaussianClassifier:
    """Власна реалізація гауссівського генеративного класифікатора (GDA).

    Parameters
    ----------
    covariance : {"full", "tied", "diag"}
        "full" — QDA, "tied" — LDA, "diag" — Gaussian Naive Bayes.
    reg : float
        Додається до діагоналі коваріації (full/tied) для додатної визначеності.
    var_smoothing : float
        Лише для "diag": до дисперсій додається var_smoothing * max(var(X)) (як у sklearn GaussianNB).
    ddof : int
        Поправка в знаменнику коваріації: 0 — ОМП-оцінка (за замовчуванням), 1 — поправка на оцінювання середніх.
        full/diag: знаменник n_k - ddof; tied: n - K * ddof.
        Конвенція sklearn залежить від моделі, solver і версії бібліотеки.
    priors : array-like або None
        Апріорні ймовірності класів; None — частоти в навчальній вибірці.
    """

    def __init__(self, covariance="full", reg=0.0, var_smoothing=1e-9, ddof=0, priors=None):
        if covariance not in ("full", "tied", "diag"):
            raise ValueError("covariance має бути 'full', 'tied' або 'diag'")
        self.covariance, self.reg, self.var_smoothing, self.ddof, self.priors = covariance, reg, var_smoothing, ddof, priors

    # ---------- навчання ----------
    def fit(self, X, y):
        X, y = np.asarray(X, dtype=float), np.asarray(y)
        self.classes_, y_idx = np.unique(y, return_inverse=True)
        K, d = len(self.classes_), X.shape[1]
        counts = np.bincount(y_idx, minlength=K)
        if np.any(counts <= self.ddof):
            raise ValueError("У кожному класі має бути більше спостережень, ніж ddof")
        self.priors_ = counts / counts.sum() if self.priors is None else np.asarray(self.priors, dtype=float)
        self.means_ = np.vstack([X[y_idx == k].mean(axis=0) for k in range(K)])
        scatter = [(X[y_idx == k] - self.means_[k]).T @ (X[y_idx == k] - self.means_[k]) for k in range(K)]

        if self.covariance == "full":
            covs = [S / (counts[k] - self.ddof) + self.reg * np.eye(d) for k, S in enumerate(scatter)]
        elif self.covariance == "tied":
            pooled = sum(scatter) / (counts.sum() - K * self.ddof) + self.reg * np.eye(d)
            covs = [pooled] * K
        else:  # diag
            eps = self.var_smoothing * X.var(axis=0).max()
            covs = [np.diag(np.diag(S) / (counts[k] - self.ddof) + eps) for k, S in enumerate(scatter)]

        self.covariances_ = np.array(covs)
        self._chol = np.array([cholesky(C, lower=True) for C in covs])
        self._logdet = np.array([2.0 * np.log(np.diag(L)).sum() for L in self._chol])
        self.n_features_in_ = d
        return self

    # ---------- інференс ----------
    def _log_likelihood(self, X):
        """ln N(x; μ_k, Σ_k) для кожного класу → масив (n, K)."""
        X = np.asarray(X, dtype=float)
        d = X.shape[1]
        out = np.empty((X.shape[0], len(self.classes_)))
        for k in range(len(self.classes_)):
            z = solve_triangular(self._chol[k], (X - self.means_[k]).T, lower=True)  # L^{-1}(x-μ)
            maha = np.einsum("ij,ij->j", z, z)                                         # ‖L^{-1}(x-μ)‖²
            out[:, k] = -0.5 * (d * np.log(2 * np.pi) + self._logdet[k] + maha)
        return out

    def decision_function_all(self, X):
        """Дискримінантні функції δ_k(x) = ln p(x|k) + ln π_k."""
        return self._log_likelihood(X) + np.log(self.priors_)

    def predict_log_proba(self, X):
        jll = self.decision_function_all(X)
        return jll - logsumexp(jll, axis=1, keepdims=True)

    def predict_proba(self, X):
        return np.exp(self.predict_log_proba(X))

    def predict(self, X):
        return self.classes_[np.argmax(self.decision_function_all(X), axis=1)]

    def decision_function(self, X):
        """Для бінарного випадку — логарифм відношення шансів ln P(1|x)/P(0|x)."""
        jll = self.decision_function_all(X)
        return jll[:, 1] - jll[:, 0] if jll.shape[1] == 2 else jll

    def score(self, X, y):
        return float(np.mean(self.predict(X) == np.asarray(y)))

    # сумісність із sklearn (clone, Pipeline, крос-валідація)
    def get_params(self, deep=True):
        return dict(covariance=self.covariance, reg=self.reg, var_smoothing=self.var_smoothing, ddof=self.ddof, priors=self.priors)

    def set_params(self, **p):
        for k, v in p.items():
            setattr(self, k, v)
        return self

    def __repr__(self):
        return f"GaussianClassifier(covariance='{self.covariance}', reg={self.reg}, ddof={self.ddof})"


print('— 4.1 Перевірка коректності на синтетичних даних з відомими параметрами')

rng = np.random.default_rng(SEED)
mu_true = [np.array([0.0, 0.0]), np.array([1.5, 1.0])]
cov_true = [np.array([[1.0, 0.6], [0.6, 1.0]]), np.array([[0.6, -0.3], [-0.3, 1.4]])]
n_syn = 20000
Xs = np.vstack([rng.multivariate_normal(mu_true[k], cov_true[k], n_syn) for k in (0, 1)])
ys = np.repeat([0, 1], n_syn)
syn = GaussianClassifier("full").fit(Xs, ys)
print("Оцінені середні:\n", syn.means_.round(3))
print("Оцінена Σ_1:\n", syn.covariances_[1].round(3), "\nСправжня Σ_1:\n", cov_true[1])
# Байєсівська (оптимальна) точність — класифікатор зі справжніми параметрами на нових даних
Xn = np.vstack([rng.multivariate_normal(mu_true[k], cov_true[k], n_syn) for k in (0, 1)])
true_ll = np.column_stack([stats.multivariate_normal(mu_true[k], cov_true[k]).logpdf(Xn) for k in (0, 1)])
bayes_acc = np.mean(true_ll.argmax(1) == ys)
print(f"\nТочність in-house QDA: {syn.score(Xn, ys):.4f}   |   байєсівська межа: {bayes_acc:.4f}")
# Логарифм густини — проти scipy
assert np.allclose(syn._log_likelihood(Xn[:500])[:, 1], stats.multivariate_normal(syn.means_[1], syn.covariances_[1]).logpdf(Xn[:500]))
print("ln N(x; μ, Σ) збігається зі scipy.stats.multivariate_normal ✓")


print('— 4.2 Навчання in-house моделей на GBP/USD')

inhouse = {
    "In-house QDA": GaussianClassifier("full"),
    "In-house LDA": GaussianClassifier("tied"),
    "In-house NB": GaussianClassifier("diag", var_smoothing=1e-9),
}
for name, m in inhouse.items():
    m.fit(X_train, y_train)
    print(f"{name:14s}  train acc = {m.score(X_train, y_train):.4f}   test acc = {m.score(X_test, y_test):.4f}")

qda_ih = inhouse["In-house QDA"]
show_result(pd.DataFrame(scaler.inverse_transform(qda_ih.means_), index=["μ(↓)", "μ(↑)"], columns=FEATURES))


# 4.3 Візуалізація навченої моделі: 2 ознаки, еліпси 1σ/2σ та апостеріорна ймовірність P(↑|x)
PAIR = ["ret_0", "log_range"]
idx = [FEATURES.index(f) for f in PAIR]
m2 = GaussianClassifier("full").fit(X_train[:, idx], y_train)

def draw_ellipses(ax, mean, cov, color):
    vals, vecs = np.linalg.eigh(cov)
    ang = np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1]))
    for nsig, ls in [(1, "-"), (2, "--")]:
        w, h = 2 * nsig * np.sqrt(vals[::-1])
        ax.add_patch(Ellipse(mean, w, h, angle=ang, fill=False, color=color, lw=2, ls=ls))

xx, yy = np.meshgrid(np.linspace(-3.5, 3.5, 300), np.linspace(-3.5, 3.5, 300))
P = m2.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1].reshape(xx.shape)
fig, ax = plt.subplots(figsize=(8.2, 6.3))
cf = ax.contourf(xx, yy, P, levels=np.linspace(0.44, 0.56, 13), cmap="RdBu", extend="both", alpha=0.6)
ax.contour(xx, yy, P, levels=[0.5], colors=C_INK, linewidths=1.5)
sub = np.random.default_rng(SEED).choice(len(y_train), 1200, replace=False)
for k in (0, 1):
    s = sub[y_train[sub] == k]
    ax.scatter(X_train[s, idx[0]], X_train[s, idx[1]], s=7, color=CLASS_COLORS[k], alpha=0.55, lw=0)
    draw_ellipses(ax, m2.means_[k], m2.covariances_[k], CLASS_COLORS[k])
plt.colorbar(cf, ax=ax, label="P(↑ | x)")
ax.set_xlabel("ret_0 (стандартизовано)"); ax.set_ylabel("log_range (стандартизовано)")
ax.set_xlim(-3.5, 3.5); ax.set_ylim(-3.5, 3.5)
ax.legend(handles=[Line2D([], [], color=C_UP, lw=2, label="↑: еліпси 1σ / 2σ"), Line2D([], [], color=C_DOWN, lw=2, label="↓: еліпси 1σ / 2σ"),
                   Line2D([], [], color=C_INK, lw=1.5, label="межа рішень P=0.5")], loc="upper left")
ax.set_title("In-house QDA: класові гауссіани та межа рішень"); ax.grid(False)
plt.tight_layout(); show_fig()


# 4.4 3D-поверхні густин p(x | клас), оцінених in-house моделлю
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
gx, gy = np.meshgrid(np.linspace(-3, 3, 120), np.linspace(-3, 3, 120))
dens = np.exp(m2._log_likelihood(np.c_[gx.ravel(), gy.ravel()])).T.reshape(2, *gx.shape)
fig = plt.figure(figsize=(13, 5))
for i, k in enumerate((1, 0)):
    ax = fig.add_subplot(1, 2, i + 1, projection="3d")
    ax.plot_surface(gx, gy, dens[k], cmap="viridis", linewidth=0, edgecolor="none", antialiased=False, rcount=80, ccount=80)
    ax.set_title(f"p(x | {CLASS_NAMES[k]})", pad=0)
    ax.set_xlabel("ret_0"); ax.set_ylabel("log_range"); ax.set_zlabel("густина")
    ax.view_init(elev=28, azim=-55)
fig.suptitle("Класові нормальні розподіли (in-house QDA, 2 ознаки)", fontweight="bold")
plt.tight_layout(); show_fig()


section('5. Out-of-box класифікатори (20%) — scikit-learn')

outbox = {
    "sklearn QDA": make_pipeline(StandardScaler(), QuadraticDiscriminantAnalysis()),
    "sklearn LDA": make_pipeline(StandardScaler(), LinearDiscriminantAnalysis()),
    "sklearn GaussianNB": make_pipeline(StandardScaler(), GaussianNB()),
}
for name, m in outbox.items():
    m.fit(X_train_raw, y_train)
    print(f"{name:20s} train acc = {m.score(X_train_raw, y_train):.4f}   test acc = {m.score(X_test_raw, y_test):.4f}")


# 5.1 Звірка in-house ↔ sklearn після узгодження оцінки коваріації.
# Конвенція оцінки коваріації залежить від моделі, solver і версії scikit-learn,
# тому для кожної пари визначаємо, яку конвенцію використовує встановлена версія, і порівнюємо з in-house з тим самим ddof.
import sklearn
pairs = [("In-house QDA", "sklearn QDA", "full"), ("In-house LDA", "sklearn LDA", "tied"), ("In-house NB", "sklearn GaussianNB", "diag")]
SK_DDOF, rows = {}, []
for a, b, cov in pairs:
    pb = outbox[b].predict_proba(X_test_raw)
    diffs = {dd: np.abs(GaussianClassifier(cov, ddof=dd).fit(X_train, y_train).predict_proba(X_test) - pb).max() for dd in (0, 1)}
    dd = min(diffs, key=diffs.get); SK_DDOF[cov] = dd
    twin = GaussianClassifier(cov, ddof=dd).fit(X_train, y_train)
    rows.append({"in-house": a, "sklearn": b, "знаменник Σ у sklearn " + sklearn.__version__: (("n" if dd == 0 else "n − K") if cov == "tied" else ("n_k" if dd == 0 else "n_k − 1")),
                 "max |ΔP| (той самий ddof)": diffs[dd], "max |ΔP| (ddof = 0)": diffs[0],
                 "збіг передбачень": np.mean(twin.predict(X_test) == outbox[b].predict(X_test_raw))})
cmp = pd.DataFrame(rows)
assert (cmp["max |ΔP| (той самий ddof)"] < 1e-8).all(), "розбіжність з sklearn!"
show_result(cmp.style.format({"max |ΔP| (той самий ddof)": "{:.2e}", "max |ΔP| (ddof = 0)": "{:.2e}", "збіг передбачень": "{:.2%}"}))


# 5.2 Швидкодія: in-house vs sklearn (на всьому train)
import time
def bench(fn, reps=20):
    t = time.perf_counter()
    for _ in range(reps): fn()
    return (time.perf_counter() - t) / reps * 1e3
speed = pd.DataFrame([
    {"модель": a, "fit + predict_proba, мс": bench(lambda: inhouse[a].fit(X_train, y_train).predict_proba(X_test))} for a, _, _ in pairs] + [
    {"модель": b, "fit + predict_proba, мс": bench(lambda: outbox[b].fit(X_train_raw, y_train).predict_proba(X_test_raw))} for _, b, _ in pairs])
show_result(speed)


section('6. Оцінка точності класифікації')

def wilson_ci(k, n, z=1.96):
    p = k / n; den = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / den; h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / den
    return c - h, c + h

def evaluate(name, y_true, y_pred, p_up):
    acc = metrics.accuracy_score(y_true, y_pred)
    lo, hi = wilson_ci((y_true == y_pred).sum(), len(y_true))
    return {"модель": name, "accuracy": acc, "95% ДІ": f"[{lo:.3f}; {hi:.3f}]",
            "balanced acc": metrics.balanced_accuracy_score(y_true, y_pred),
            "precision ↑": metrics.precision_score(y_true, y_pred, zero_division=0),
            "recall ↑": metrics.recall_score(y_true, y_pred, zero_division=0),
            "F1 ↑": metrics.f1_score(y_true, y_pred, zero_division=0),
            "ROC AUC": metrics.roc_auc_score(y_true, p_up),
            "log-loss": metrics.log_loss(y_true, np.clip(p_up, 1e-12, 1 - 1e-12)),
            "Brier": metrics.brier_score_loss(y_true, p_up)}

dummies = {"Dummy (most frequent)": DummyClassifier(strategy="most_frequent"),
           "Dummy (stratified)": DummyClassifier(strategy="stratified", random_state=SEED)}
for m in dummies.values():
    m.fit(X_train, y_train)

rows = []
for name, m in inhouse.items():
    rows.append(evaluate(name, y_test, m.predict(X_test), m.predict_proba(X_test)[:, 1]))
for name, m in outbox.items():
    rows.append(evaluate(name, y_test, m.predict(X_test_raw), m.predict_proba(X_test_raw)[:, 1]))
for name, m in dummies.items():
    # для «most frequent» беремо частоту ↑ з train як ймовірність (інакше log-loss нескінченний)
    p = np.full(len(y_test), y_train.mean()) if "most frequent" in name else m.predict_proba(X_test)[:, 1]
    rows.append(evaluate(name, y_test, m.predict(X_test), p))
results_bin = pd.DataFrame(rows).set_index("модель")
show_result(results_bin)


# 6.1 Статистична значущість: чи краща модель за вгадування?
# Біноміальний тест і ДІ Вільсона припускають незалежність днів. Для часового ряду перевіряємо це:
# (а) автокореляція ряду «вгадав / не вгадав»; (б) блочний bootstrap, який зберігає залежність усередині блоків.
BLOCK = 20   # ≈ 1 торговий місяць

def block_bootstrap_idx(n, r, block=BLOCK):
    """Індекси циклічного блочного bootstrap (moving block bootstrap): вибираються відрізки по `block` днів підряд."""
    starts = r.integers(0, n, int(np.ceil(n / block)))
    return ((starts[:, None] + np.arange(block)[None, :]).ravel()[:n]) % n

n_test = len(y_test)
base_rate = max(y_test.mean(), 1 - y_test.mean())   # точність «завжди найчастіший клас тесту» (оракул-базис)
sig = []
for name, m in inhouse.items():
    k = int((m.predict(X_test) == y_test).sum())
    hits = (m.predict(X_test) == y_test).astype(float)
    r_ = np.random.default_rng(SEED)
    boot_acc = np.array([hits[block_bootstrap_idx(n_test, r_)].mean() for _ in range(2000)])
    sig.append({"модель": name, "вірних": k, "з": n_test, "accuracy": k / n_test,
                "p (H0: acc = 0.5)": stats.binomtest(k, n_test, 0.5, alternative="greater").pvalue,
                f"p (H0: acc = {base_rate:.3f})": stats.binomtest(k, n_test, base_rate, alternative="greater").pvalue,
                "автокор. влучань, лаг 1": np.corrcoef(hits[:-1], hits[1:])[0, 1],
                "95% ДІ (блочний bootstrap)": f"[{np.percentile(boot_acc, 2.5):.3f}; {np.percentile(boot_acc, 97.5):.3f}]"})
show_result(pd.DataFrame(sig).set_index("модель"))


# 6.2 Матриці помилок (in-house моделі, тест)
fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
for ax, (name, m) in zip(axes, inhouse.items()):
    cm = metrics.confusion_matrix(y_test, m.predict(X_test), labels=[0, 1])
    cmn = cm / cm.sum(axis=1, keepdims=True)
    sns.heatmap(cmn, annot=np.array([[f"{cm[i,j]}\n{cmn[i,j]:.1%}" for j in range(2)] for i in range(2)]), fmt="",
                cmap="Blues", vmin=0, vmax=1, cbar=False, linewidths=2, linecolor="#fcfcfb", ax=ax,
                xticklabels=["прогноз ↓", "прогноз ↑"], yticklabels=["факт ↓", "факт ↑"])
    ax.set_title(f"{name}\nacc = {np.trace(cm)/cm.sum():.3f}"); ax.grid(False)
plt.tight_layout(); show_fig()


# 6.3 Walk-forward валідація лише на train 2010–2022 (тест 2023–2026 не залучено)
def walk_forward(model_factory, X, y, n_splits=8, use_proba=True):
    tss = TimeSeriesSplit(n_splits=n_splits, gap=1)
    accs, aucs = [], []
    for tr, te in tss.split(X):
        sc = StandardScaler().fit(X[tr])
        m = model_factory().fit(sc.transform(X[tr]), y[tr])
        p = m.predict_proba(sc.transform(X[te]))[:, 1]
        accs.append(np.mean((p > 0.5).astype(int) == y[te])); aucs.append(metrics.roc_auc_score(y[te], p))
    return np.array(accs), np.array(aucs)

wf_rows, wf_folds = [], {}
for name, fac in {"In-house QDA": lambda: GaussianClassifier("full"), "In-house LDA": lambda: GaussianClassifier("tied"),
                  "In-house NB": lambda: GaussianClassifier("diag")}.items():
    a, u = walk_forward(fac, X_train_raw, y_train)
    wf_folds[name] = a
    wf_rows.append({"модель": name, "accuracy (середнє ± std)": f"{a.mean():.4f} ± {a.std():.4f}",
                    "AUC (середнє ± std)": f"{u.mean():.4f} ± {u.std():.4f}", "фолдів з acc > 0.5": f"{(a > 0.5).sum()}/{len(a)}"})
show_result(pd.DataFrame(wf_rows).set_index("модель"))


# 6.4 Accuracy за роками тестового періоду (стабільність у часі)
yr = test.Date.dt.year.to_numpy()
fig, ax = plt.subplots(figsize=(10, 3.6))
years = np.unique(yr); w = 0.26
for i, (name, m) in enumerate(inhouse.items()):
    pred = m.predict(X_test)
    accs = [np.mean(pred[yr == Y] == y_test[yr == Y]) for Y in years]
    ax.bar(np.arange(len(years)) + (i - 1) * w, accs, width=w - 0.03, color=[C_UP, C_VIOLET, C_FLAT][i], label=name)
ax.axhline(0.5, color=C_INK, lw=0.8, ls=":")
ax.set_xticks(range(len(years)), [str(Y) + ("*" if Y == years[-1] else "") for Y in years]); ax.set_ylim(0.35, 0.65)
ax.set_title("Точність in-house моделей за роками тесту (*2026 — до жовтня)"); ax.legend(ncol=3, loc="upper left")
plt.tight_layout(); show_fig()


section('7. ROC-криві та AUC (20%)')

def roc_curve_inhouse(y_true, scores):
    """ROC-крива: (fpr, tpr, thresholds). Однакові оцінки обробляються як один поріг."""
    y_true, scores = np.asarray(y_true), np.asarray(scores, dtype=float)
    order = np.argsort(-scores, kind="mergesort")
    s, yt = scores[order], y_true[order]
    last_of_each = np.r_[np.where(np.diff(s) != 0)[0], len(s) - 1]   # індекси останніх елементів груп однакових оцінок
    tps = np.cumsum(yt)[last_of_each]
    fps = (last_of_each + 1) - tps
    P, N = yt.sum(), len(yt) - yt.sum()
    return np.r_[0.0, fps / N], np.r_[0.0, tps / P], np.r_[np.inf, s[last_of_each]]

def auc_trapezoid(fpr, tpr):
    return float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2))

def auc_mann_whitney(y_true, scores):
    y_true = np.asarray(y_true); ranks = stats.rankdata(scores)          # середні ранги для однакових значень
    n1 = y_true.sum(); n0 = len(y_true) - n1
    return float((ranks[y_true == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))

def bootstrap_auc(y_true, scores, B=2000, seed=SEED):
    """Блочний bootstrap AUC (блоки по BLOCK днів) — враховує залежність сусідніх днів."""
    r = np.random.default_rng(seed); n = len(y_true); out = []
    while len(out) < B:
        i = block_bootstrap_idx(n, r)
        if 0 < y_true[i].sum() < n:
            out.append(auc_mann_whitney(y_true[i], scores[i]))
    return np.array(out)

all_bin = {**{n: (m, X_test) for n, m in inhouse.items()}, **{n: (m, X_test_raw) for n, m in outbox.items()}}
scores_test = {n: m.predict_proba(X)[:, 1] for n, (m, X) in all_bin.items()}
auc_rows = []
for n, s in scores_test.items():
    fpr, tpr, _ = roc_curve_inhouse(y_test, s)
    boot = bootstrap_auc(y_test, s)
    auc_rows.append({"модель": n, "AUC трапеції (in-house)": auc_trapezoid(fpr, tpr), "AUC Манн–Уітні (in-house)": auc_mann_whitney(y_test, s),
                     "AUC sklearn": metrics.roc_auc_score(y_test, s), "95% ДІ (блочний bootstrap)": f"[{np.percentile(boot, 2.5):.3f}; {np.percentile(boot, 97.5):.3f}]",
                     "частка bootstrap AUC ≤ 0.5": float(np.mean(boot <= 0.5))})
auc_table = pd.DataFrame(auc_rows).set_index("модель")
assert np.allclose(auc_table.iloc[:, 0], auc_table["AUC sklearn"]) and np.allclose(auc_table.iloc[:, 1], auc_table["AUC sklearn"])
# Сама крива — теж проти sklearn
f1_, t1_, _ = roc_curve_inhouse(y_test, scores_test["In-house QDA"]); f2_, t2_, _ = metrics.roc_curve(y_test, scores_test["In-house QDA"], drop_intermediate=False)
assert np.allclose(f1_, f2_) and np.allclose(t1_, t2_)
print("In-house ROC/AUC ідентичні sklearn ✓")
show_result(auc_table)


# 7.1 ROC-криві всіх гауссівських моделей на тесті (in-house — кольорові, sklearn — чорний пунктир: криві майже збігаються)
style = {"QDA": C_UP, "LDA": C_VIOLET, "NB": C_FLAT, "GaussianNB": C_FLAT}
fig, ax = plt.subplots(figsize=(7.2, 6.6))
for n, s in scores_test.items():
    fpr, tpr, _ = roc_curve_inhouse(y_test, s)
    col = style[n.split()[-1]]
    is_ih = n.startswith("In-house")
    ax.plot(fpr, tpr, color=col if is_ih else C_INK, lw=2.6 if is_ih else 0.9, ls="-" if is_ih else (0, (2, 2)),
            label=f"{n}  (AUC = {auc_table.loc[n, 'AUC sklearn']:.3f})")
ax.plot([0, 1], [0, 1], color=C_GREY, lw=1, ls=":", label="випадкове вгадування (AUC = 0.5)")
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate"); ax.set_aspect("equal")
ax.set_title("ROC-криві: напрямок GBP/USD на наступний день (тест 2023–2026)")
ax.legend(loc="lower right", fontsize=8.5)
plt.tight_layout(); show_fig()


# 7.2 ROC in-house QDA: train vs test + 95% bootstrap-смуга (перевірка перенавчання)
def roc_band(y_true, s, B=500, grid=np.linspace(0, 1, 201), seed=SEED):
    r = np.random.default_rng(seed); n = len(y_true); curves = []
    for _ in range(B):
        i = block_bootstrap_idx(n, r)
        f, t, _ = roc_curve_inhouse(y_true[i], s[i]); curves.append(np.interp(grid, f, t))
    curves = np.array(curves)
    return grid, np.percentile(curves, 2.5, axis=0), np.percentile(curves, 97.5, axis=0)

fig, axes = plt.subplots(1, 2, figsize=(13, 6))
for ax, (name, m) in zip(axes, [("In-house QDA", inhouse["In-house QDA"]), ("In-house LDA", inhouse["In-house LDA"])]):
    s_tr, s_te = m.predict_proba(X_train)[:, 1], m.predict_proba(X_test)[:, 1]
    g_, lo_, hi_ = roc_band(y_test, s_te)
    ax.fill_between(g_, lo_, hi_, color=C_UP, alpha=0.15, lw=0, label="95% bootstrap-смуга (test)")
    for s, yy, lab, col, ls in [(s_tr, y_train, "train", C_DOWN, "--"), (s_te, y_test, "test", C_UP, "-")]:
        f, t, _ = roc_curve_inhouse(yy, s)
        ax.plot(f, t, color=col, ls=ls, label=f"{lab}: AUC = {auc_trapezoid(f, t):.3f}")
    ax.plot([0, 1], [0, 1], color=C_GREY, lw=1, ls=":")
    ax.set_aspect("equal"); ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title(name); ax.legend(loc="lower right")
fig.suptitle("ROC на train і test: різниця показує ступінь перенавчання", fontweight="bold")
plt.tight_layout(); show_fig()


# 7.3 Порівняння AUC з довірчими інтервалами + вибір порогу за індексом Юдена (на train!)
fig, ax = plt.subplots(figsize=(9, 3.8))
names = list(scores_test)
for i, n in enumerate(names):
    boot = bootstrap_auc(y_test, scores_test[n], B=1000)
    a = auc_table.loc[n, "AUC sklearn"]
    ax.errorbar(a, i, xerr=[[a - np.percentile(boot, 2.5)], [np.percentile(boot, 97.5) - a]], fmt="o", ms=8,
                color=style[n.split()[-1]], ecolor=C_INK2, capsize=4, lw=1.4)
ax.axvline(0.5, color=C_INK, lw=0.8, ls=":")
ax.set_yticks(range(len(names)), names); ax.invert_yaxis(); ax.set_xlabel("ROC AUC на тесті (95% bootstrap ДІ)")
ax.set_title("AUC усіх гауссівських моделей: інтервали перетинають 0.5?")
plt.tight_layout(); show_fig()

m = inhouse["In-house QDA"]
f, t, thr = roc_curve_inhouse(y_train, m.predict_proba(X_train)[:, 1])
j = np.argmax(t - f)
tau = thr[j]
pred_tau = (m.predict_proba(X_test)[:, 1] >= tau).astype(int)
print(f"Поріг Юдена (обраний на train) τ = {tau:.4f};  test accuracy при τ: {np.mean(pred_tau == y_test):.4f}  (при τ=0.5: {m.score(X_test, y_test):.4f})")


section('8. Extra: Multi-class championship (30%)')

q1, q2 = np.quantile(train["next_ret"], [1 / 3, 2 / 3])
def to3(r): return np.digitize(r, [q1, q2])          # 0: ↓, 1: ≈, 2: ↑
y3_train, y3_test = to3(train["next_ret"].to_numpy()), to3(test["next_ret"].to_numpy())
CLS3 = ["↓ сильне", "≈ слабке", "↑ сильне"]; COL3 = [C_DOWN, C_FLAT, C_UP]
print(f"Пороги (train): q1 = {q1:+.3f}%,  q2 = {q2:+.3f}%")
print(pd.DataFrame({"train": pd.Series(y3_train).value_counts(normalize=True).sort_index().values,
                    "test": pd.Series(y3_test).value_counts(normalize=True).sort_index().values}, index=CLS3))


# 8.1 Візуалізація: розподіл next_ret за класами та ключова ознака волатильності
fig, axes = plt.subplots(1, 2, figsize=(13, 3.9))
b = np.linspace(-2.5, 2.5, 101)
for k in range(3):
    axes[0].hist(train["next_ret"][y3_train == k], bins=b, color=COL3[k], alpha=0.8, label=CLS3[k])
for qv in (q1, q2): axes[0].axvline(qv, color=C_INK, lw=1, ls="--")
axes[0].set_title("Дохідність наступного дня та межі терцилів (train)"); axes[0].set_xlabel("r(t+1), %"); axes[0].legend()
for k in range(3):
    sns.kdeplot(train["log_vol_10"][y3_train == k], ax=axes[1], color=COL3[k], lw=2, label=CLS3[k], fill=True, alpha=0.08)
axes[1].set_title("log_vol_10 за класами: «≈» зміщений у бік низької волатильності"); axes[1].legend()
plt.tight_layout(); show_fig()


contestants = {
    "In-house QDA":        (GaussianClassifier("full"), "scaled"),
    "In-house LDA":        (GaussianClassifier("tied"), "scaled"),
    "In-house NB":         (GaussianClassifier("diag"), "scaled"),
    "Logistic Regression": (make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)), "raw"),
    "kNN (k=100)":         (make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=100)), "raw"),
    "Random Forest":       (RandomForestClassifier(n_estimators=400, min_samples_leaf=25, max_features="sqrt", n_jobs=-1, random_state=SEED), "raw"),
    "Gradient Boosting":   (HistGradientBoostingClassifier(learning_rate=0.03, max_iter=300, max_leaf_nodes=15, min_samples_leaf=40, l2_regularization=1.0, random_state=SEED), "raw"),
    "Dummy (prior)":       (DummyClassifier(strategy="prior"), "raw"),
}
XS = {"scaled": (X_train, X_test), "raw": (X_train_raw, X_test_raw)}

def macro_auc_inhouse(y, P):
    """Macro OvR AUC власною реалізацією (Манн–Уітні для кожного класу)."""
    return float(np.mean([auc_mann_whitney((y == k).astype(int), P[:, k]) for k in range(P.shape[1])]))


# 8.2 Walk-forward: ТІЛЬКИ навчальний період 2010–2022.
# У кожному фолді пороги класів і масштабування визначаються виключно з його train-частини.
# Тест 2023–2026 не входить ані до фолдів, ані до вибору чемпіона.
CV_METRICS = {"macro AUC": False, "accuracy": False, "balanced acc": False, "macro F1": False, "log-loss": True}
next_ret_cv = train["next_ret"].to_numpy()
tss = TimeSeriesSplit(n_splits=8, gap=1)
wf3 = {n: [] for n in contestants}
cv_fold_metrics = {n: [] for n in contestants}
wf3_folds = []
for tr, te in tss.split(X_train_raw):
    qa, qb = np.quantile(next_ret_cv[tr], [1 / 3, 2 / 3])
    y3_tr = np.digitize(next_ret_cv[tr], [qa, qb])
    y3_te = np.digitize(next_ret_cv[te], [qa, qb])
    wf3_folds.append({"train до": train.Date.iloc[tr[-1]].strftime("%m.%Y"),
                      "валідація": f"{train.Date.iloc[te[0]]:%m.%Y}–{train.Date.iloc[te[-1]]:%m.%Y}",
                      "q1, %": qa, "q2, %": qb})
    sc = StandardScaler().fit(X_train_raw[tr])
    for name, (model, kind) in contestants.items():
        Xtr = sc.transform(X_train_raw[tr]) if kind == "scaled" else X_train_raw[tr]
        Xte = sc.transform(X_train_raw[te]) if kind == "scaled" else X_train_raw[te]
        mdl = clone(model) if not isinstance(model, GaussianClassifier) else GaussianClassifier(**model.get_params())
        P = mdl.fit(Xtr, y3_tr).predict_proba(Xte)
        pred = P.argmax(axis=1)
        auc = metrics.roc_auc_score(y3_te, P, multi_class="ovr", average="macro")
        wf3[name].append(auc)
        cv_fold_metrics[name].append({
            "macro AUC": auc,
            "accuracy": metrics.accuracy_score(y3_te, pred),
            "balanced acc": metrics.balanced_accuracy_score(y3_te, pred),
            "macro F1": metrics.f1_score(y3_te, pred, average="macro", zero_division=0),
            "log-loss": metrics.log_loss(y3_te, P, labels=[0, 1, 2]),
        })
wf3 = pd.DataFrame(wf3)
validation_summary = pd.DataFrame({name: pd.DataFrame(rows).mean() for name, rows in cv_fold_metrics.items()}).T
assert set(validation_summary.columns) == set(CV_METRICS)
assert train.target_date.max() < SPLIT and (test.Date >= SPLIT).all()
order = wf3.mean().sort_values(ascending=False).index
fig, ax = plt.subplots(figsize=(11, 4.6))
is_gauss = lambda n: n.startswith("In-house")
import matplotlib
box_kw = {"orientation": "horizontal"} if tuple(map(int, matplotlib.__version__.split(".")[:2])) >= (3, 10) else {"vert": False}
bp = ax.boxplot([wf3[n] for n in order], **box_kw, patch_artist=True, widths=0.55, medianprops=dict(color=C_INK, lw=1.5),
                whiskerprops=dict(color=C_INK2), capprops=dict(color=C_INK2), flierprops=dict(marker="o", ms=4, mfc=C_GREY, mec="none"))
for patch, n in zip(bp["boxes"], order):
    patch.set_facecolor(C_UP if n.startswith("In-house") else C_YELLOW); patch.set_alpha(0.55); patch.set_edgecolor(C_INK2)
ax.scatter(wf3[order].mean(), np.arange(1, len(order) + 1), marker="D", color=C_INK, s=22, zorder=3, label="середнє")
ax.set_yticks(np.arange(1, len(order) + 1), order); ax.invert_yaxis(); ax.axvline(0.5, color=C_INK, lw=0.8, ls=":")
ax.legend(handles=[Patch(color=C_UP, alpha=0.55, label="in-house гауссівські"),
                   Patch(color=C_YELLOW, alpha=0.55, label="інші моделі"), Line2D([], [], marker="D", color=C_INK, ls="", label="середнє")], loc="lower right", fontsize=8.5)
ax.set_xlabel("macro OvR ROC AUC"); ax.set_title("Walk-forward (8 фолдів, лише 2010–2022): стабільність учасників")
plt.tight_layout(); show_fig()
print("Фолди walk-forward і пороги класів, обчислені з train кожного фолда:")
print(pd.DataFrame(wf3_folds).to_string(index=False, float_format=lambda v: f"{v:+.3f}"))
print("\nСередні метрики валідації (без тестового періоду):")
show_result(validation_summary.sort_values("macro AUC", ascending=False))
show_result(wf3.agg(["mean", "std", "min"]).T.assign(**{"фолдів з AUC > 0.5": (wf3 > 0.5).sum().astype(str) + "/8"}).sort_values("mean", ascending=False))


# 8.3 Підсумкова турнірна таблиця: 5 номінацій за walk-forward CV 2010–2022.
# Вибір чемпіона, сума балів і тай-брейк НЕ залежать від тесту 2023–2026.
pts = pd.DataFrame(index=validation_summary.index)
n_c = len(pts)
for col, lower_better in CV_METRICS.items():
    r = validation_summary[col].rank(ascending=lower_better, method="min")
    pts[f"{col} (CV)"] = (n_c + 1 - r).astype(int)
pts["СУМА БАЛІВ"] = pts.sum(axis=1)   # за кожну номінацію: 1-ше = 8 балів, ..., 8-ме = 1 бал
pts["тай-брейк: macro AUC (CV)"] = validation_summary["macro AUC"]
standings = pts.sort_values(["СУМА БАЛІВ", "тай-брейк: macro AUC (CV)"], ascending=False)
standings.insert(0, "місце", range(1, len(standings) + 1))
champion = standings.index[0]
best_gauss = next(n for n in standings.index if is_gauss(n))
assert standings["СУМА БАЛІВ"].max() <= n_c * len(CV_METRICS)
print(f"Чемпіон за валідацією 2010–2022: {champion}   |   найкраща гауссівська модель: {best_gauss}")
print("Тест 2023–2026 не використовувався для рейтингу, відбору або тай-брейку.")
show_result(standings)


# 8.3.1 Незалежна фінальна оцінка на тесті 2023–2026 (після фіксації чемпіона на CV).
# Тут переобучуємо моделі на всьому train 2010–2022. Тестові метрики не впливають на standings.
board, probs3 = [], {}
for name, (model, kind) in contestants.items():
    Xtr, Xte = XS[kind]
    model.fit(Xtr, y3_train)
    P = model.predict_proba(Xte); probs3[name] = P
    pred = P.argmax(1)
    auc_ih = macro_auc_inhouse(y3_test, P)
    assert np.isclose(auc_ih, metrics.roc_auc_score(y3_test, P, multi_class="ovr", average="macro"))
    board.append({"модель": name, "macro AUC": auc_ih, "accuracy": metrics.accuracy_score(y3_test, pred),
                  "balanced acc": metrics.balanced_accuracy_score(y3_test, pred), "macro F1": metrics.f1_score(y3_test, pred, average="macro"),
                  "log-loss": metrics.log_loss(y3_test, P, labels=[0, 1, 2]),
                  "AUC ↓": auc_mann_whitney((y3_test == 0).astype(int), P[:, 0]), "AUC ≈": auc_mann_whitney((y3_test == 1).astype(int), P[:, 1]),
                  "AUC ↑": auc_mann_whitney((y3_test == 2).astype(int), P[:, 2])})
# контроль: sklearn-двійники in-house моделей дають ті самі ймовірності і в мультикласі
for cov, sk in [("full", QuadraticDiscriminantAnalysis()), ("tied", LinearDiscriminantAnalysis()), ("diag", GaussianNB())]:
    twin = GaussianClassifier(cov, ddof=SK_DDOF[cov]).fit(X_train, y3_train)      # ddof — як у встановленій версії sklearn (§5.1)
    assert np.allclose(make_pipeline(StandardScaler(), sk).fit(X_train_raw, y3_train).predict_proba(X_test_raw), twin.predict_proba(X_test), atol=1e-8)
leaderboard = pd.DataFrame(board).sort_values("macro AUC", ascending=False).reset_index(drop=True)
leaderboard.index = leaderboard.index + 1; leaderboard.index.name = "місце"
print("Фінальна оцінка на незалежному тесті (не використана для вибору чемпіона):")
show_result(leaderboard)


# 8.3.2 Чи значуща перевага обраного за CV чемпіона? Парний блочний bootstrap різниці macro AUC (тест 2023–2026)
r_ = np.random.default_rng(SEED)
idx_list = [block_bootstrap_idx(len(y3_test), r_) for _ in range(1000)]
idx_list = [i for i in idx_list if len(np.unique(y3_test[i])) == 3]
champ_boot = np.array([macro_auc_inhouse(y3_test[i], probs3[champion][i]) for i in idx_list])
PRACTICAL = 0.005   # різниця macro AUC, меншу за яку вважаємо практично нехтовною
pair_rows = []
for other in standings.index[1:]:
    other_boot = np.array([macro_auc_inhouse(y3_test[i], probs3[other][i]) for i in idx_list])
    d = champ_boot - other_boot
    pair_rows.append({"чемпіон проти": other, "Δ macro AUC": macro_auc_inhouse(y3_test, probs3[champion]) - macro_auc_inhouse(y3_test, probs3[other]),
                      "95% ДІ Δ": f"[{np.percentile(d, 2.5):+.4f}; {np.percentile(d, 97.5):+.4f}]", "p (одностор.)": max(float(np.mean(d <= 0)), 1 / len(d))})
pairwise = pd.DataFrame(pair_rows).set_index("чемпіон проти")
# поправка Голма на множинні порівняння (7 порівнянь одночасно)
order_p = pairwise["p (одностор.)"].sort_values()
holm = np.minimum(1, np.maximum.accumulate(order_p.to_numpy() * (len(order_p) - np.arange(len(order_p)))))
pairwise["p (Голм)"] = pd.Series(holm, index=order_p.index)
def verdict(row):
    if row["p (Голм)"] >= 0.05: return "перевагу чемпіона не доведено"
    return "чемпіон кращий" if row["Δ macro AUC"] >= PRACTICAL else "різниця значуща, але практично нехтовна"
pairwise["висновок"] = pairwise.apply(verdict, axis=1)
show_result(pairwise)


# 8.4 Multiclass ROC (One-vs-Rest) — чемпіон і in-house QDA, з micro/macro усередненням (власна реалізація)
def multiclass_roc(y, P):
    curves = {k: roc_curve_inhouse((y == k).astype(int), P[:, k]) for k in range(P.shape[1])}
    Y = np.eye(P.shape[1])[y]
    curves["micro"] = roc_curve_inhouse(Y.ravel(), P.ravel())
    grid = np.linspace(0, 1, 501)
    curves["macro"] = (grid, np.mean([np.interp(grid, curves[k][0], curves[k][1]) for k in range(P.shape[1])], axis=0), None)
    return curves

show = [champion] + [n for n in ["In-house QDA"] if n != champion]
fig, axes = plt.subplots(1, len(show), figsize=(6.6 * len(show), 6.3), squeeze=False)
for ax, name in zip(axes[0], show):
    cv = multiclass_roc(y3_test, probs3[name])
    for k in range(3):
        f, t, _ = cv[k]; ax.plot(f, t, color=COL3[k], lw=2, label=f"{CLS3[k]} vs rest (AUC = {auc_trapezoid(f, t):.3f})")
    f, t, _ = cv["micro"]; ax.plot(f, t, color=C_INK, lw=1.4, ls="--", label=f"micro-average (AUC = {auc_trapezoid(f, t):.3f})")
    f, t, _ = cv["macro"]; ax.plot(f, t, color=C_VIOLET, lw=1.4, ls="-.", label=f"macro-average (AUC = {macro_auc_inhouse(y3_test, probs3[name]):.3f})")
    ax.plot([0, 1], [0, 1], color=C_GREY, lw=1, ls=":")
    ax.set_aspect("equal"); ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title(("Чемпіон — " if name == champion else "") + name); ax.legend(loc="lower right", fontsize=8.5)
fig.suptitle("Multiclass ROC (One-vs-Rest), тест 2023–2026", fontweight="bold")
plt.tight_layout(); show_fig()


# 8.5 Матриці помилок: чемпіон vs in-house QDA (3 класи)
fig, axes = plt.subplots(1, len(show), figsize=(6.2 * len(show), 4.6), squeeze=False)
for ax, name in zip(axes[0], show):
    cm = metrics.confusion_matrix(y3_test, probs3[name].argmax(1), labels=[0, 1, 2]); cmn = cm / cm.sum(1, keepdims=True)
    sns.heatmap(cmn, annot=np.array([[f"{cm[i,j]}\n{cmn[i,j]:.0%}" for j in range(3)] for i in range(3)]), fmt="", cmap="Blues", vmin=0, vmax=0.7,
                cbar=False, linewidths=2, linecolor="#fcfcfb", xticklabels=["прогноз " + c for c in CLS3], yticklabels=["факт " + c for c in CLS3], ax=ax)
    ax.set_title(f"{name}: acc = {np.trace(cm)/cm.sum():.3f}\n(«завжди найчастіший клас тесту» = {np.bincount(y3_test).max()/len(y3_test):.3f})"); ax.grid(False)
plt.tight_layout(); show_fig()


# 8.6 Значущість мультикласового сигналу: тест циклічного зсуву (circular shift test)
# Звичайна перестановка міток руйнує автокореляцію часового ряду й дає занадто оптимістичний p-value.
# Тут мітки тесту циклічно зсуваються відносно прогнозів на s ≥ BLOCK днів: структура ряду зберігається, а зв'язок «прогноз ↔ факт» руйнується.
shifts = np.arange(BLOCK, len(y3_test) - BLOCK)
fig, axes = plt.subplots(1, 2, figsize=(13, 3.4), sharey=True)
shift_rows = []
for ax, name in zip(axes, [champion, "In-house QDA"]):
    P_ = probs3[name]
    obs = macro_auc_inhouse(y3_test, P_)
    null = np.array([macro_auc_inhouse(np.roll(y3_test, k), P_) for k in shifts])
    pval = (np.sum(null >= obs) + 1) / (len(null) + 1)
    shift_rows.append({"модель": name, "macro AUC": obs, "нульовий розподіл": f"{null.mean():.3f} ± {null.std():.3f}",
                       "кількість зсувів": len(null), "p-value": pval})
    ax.hist(null, bins=40, color=C_GREY, alpha=0.7, label="AUC при зсунутих мітках")
    ax.axvline(obs, color=C_UP, lw=2.5, label=f"фактичний AUC = {obs:.3f}")
    ax.set_title(f"{name}: p = {pval:.4f}"); ax.legend(fontsize=8.5)
fig.suptitle("Тест циклічного зсуву: чи пояснюється мультикласовий AUC випадковістю?", fontweight="bold")
plt.tight_layout(); show_fig()
show_result(pd.DataFrame(shift_rows).set_index("модель"))


section('9. Демонстрація інференсу на останньому відомому дні')

x_last = scaler.transform(last_row[FEATURES].to_numpy())
p_bin = inhouse["In-house QDA"].predict_proba(x_last)[0]
p_3 = contestants["In-house QDA"][0].predict_proba(x_last)[0]
print(f"Дата ознак: {last_row.Date.iloc[0]:%d.%m.%Y}")
print(f"In-house QDA (2 класи):  P(↓) = {p_bin[0]:.3f},  P(↑) = {p_bin[1]:.3f}")
print("In-house QDA (3 класи):  " + ",  ".join(f"P({c}) = {p:.3f}" for c, p in zip(CLS3, p_3)))


section('10. Підсумки')


print("\nГотово. Графіки: results/figures/, текстовий звіт: results/output.txt")
