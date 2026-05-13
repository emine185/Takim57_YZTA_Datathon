"""
Bilişsel Performans Skoru Tahmini - NİHAİ ÇÖZÜM
Strateji:
  1) Genişletilmiş feature engineering
  2) K-fold target encoding (yüksek kardinaliteli kategorikler için)
  3) Log1p hedef dönüşümü (skew varsa)
  4) Optuna ile her model için hyperparameter tuning
  5) 10-fold CV + multi-seed averaging
  6) 4 model: LightGBM + XGBoost + CatBoost + HistGradientBoosting
  7) Ridge meta-model ile stacking
"""

import os
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

import optuna
from scipy.optimize import minimize

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
pd.set_option("display.max_columns", 100)

# ----------------------------------------------------------------------
# AYARLAR
# ----------------------------------------------------------------------
TRAIN_PATH = r"C:\Users\kilic\OneDrive\Masaüstü\train.csv"
TEST_PATH  = r"C:\Users\kilic\OneDrive\Masaüstü\test_x.csv"
SAMP_PATH  = r"C:\Users\kilic\OneDrive\Masaüstü\sample_submission.csv"

OUT_DIR    = "outputs"
SEED       = 42
N_SPLITS   = 10
N_TRIALS   = 40            # Optuna trial sayısı (model başına)
SEEDS      = [42, 123, 2024]   # multi-seed
TARGET     = "bilissel_performans_skoru"
ID_COL     = "id"
USE_LOG    = None          # None -> otomatik karar verilir (skew > 0.5 ise)

os.makedirs(OUT_DIR, exist_ok=True)


# ----------------------------------------------------------------------
# 1) VERİ YÜKLEME
# ----------------------------------------------------------------------
def load_data():
    for p in [TRAIN_PATH, TEST_PATH, SAMP_PATH]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Dosya bulunamadı: {p}")
    train = pd.read_csv(TRAIN_PATH)
    test  = pd.read_csv(TEST_PATH)
    samp  = pd.read_csv(SAMP_PATH)
    print(f"Veri yüklendi -> train: {train.shape} | test: {test.shape}")
    print(f"Hedef istatistikleri: skew={train[TARGET].skew():.3f}, "
          f"min={train[TARGET].min():.2f}, max={train[TARGET].max():.2f}")
    return train, test, samp


# ----------------------------------------------------------------------
# 2) FEATURE ENGINEERING
# ----------------------------------------------------------------------
CATEGORICAL_COLS_BASE = [
    "cinsiyet", "meslek", "ulke", "kronotip",
    "ruh_sagligi_durumu", "mevsim", "gun_tipi",
]

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Uyku kalitesi türevleri
    df["toplam_kaliteli_uyku_yuzdesi"] = df["rem_yuzdesi"] + df["derin_uyku_yuzdesi"]
    df["hafif_uyku_yuzdesi"] = 100 - df["toplam_kaliteli_uyku_yuzdesi"]
    df["rem_derin_orani"] = df["rem_yuzdesi"] / (df["derin_uyku_yuzdesi"] + 1e-3)

    # Yaşam tarzı etkileşimleri
    df["kafein_ekran_etkilesimi"] = df["uyku_oncesi_kafein_mg"] * df["uyku_oncesi_ekran_suresi_dk"]
    df["stres_calisma"] = df["stres_skoru"] * df["gunluk_calisma_saati"]
    df["adim_basi_calisma"] = df["gunluk_adim_sayisi"] / (df["gunluk_calisma_saati"] + 1e-3)
    df["kafein_dalma"] = df["uyku_oncesi_kafein_mg"] * df["uykuya_dalma_suresi_dk"]
    df["nabiz_stres"] = df["dinlenik_nabiz_bpm"] * df["stres_skoru"]

    # Uyku bozulması ve verimliliği
    df["uyku_bozulma"] = df["uykuya_dalma_suresi_dk"] + df["gecelik_uyanma_sayisi"] * 10
    df["uyku_verimi"] = df["toplam_kaliteli_uyku_yuzdesi"] / (df["uyku_bozulma"] + 1)

    # Polinomial / etkileşim
    df["yas_kare"] = df["yas"] ** 2
    df["bmi_yas"]  = df["vucut_kitle_indeksi"] * df["yas"]
    df["bmi_kare"] = df["vucut_kitle_indeksi"] ** 2

    # Kategorik gruplar
    df["yas_grubu"] = pd.cut(
        df["yas"], bins=[0, 25, 35, 45, 55, 120],
        labels=["genc", "yetiskin", "orta", "olgun", "ileri"]
    ).astype("object")

    df["bmi_grubu"] = pd.cut(
        df["vucut_kitle_indeksi"], bins=[0, 18.5, 25, 30, 100],
        labels=["zayif", "normal", "kilolu", "obez"]
    ).astype("object")

    df["aktivite_seviyesi"] = pd.cut(
        df["gunluk_adim_sayisi"], bins=[-1, 3000, 7000, 12000, 1e9],
        labels=["sedanter", "az", "orta", "aktif"]
    ).astype("object")

    df["stres_grubu"] = pd.cut(
        df["stres_skoru"], bins=[-1, 3, 6, 8, 100],
        labels=["dusuk", "orta", "yuksek", "cokyuksek"]
    ).astype("object")

    # Bayrak değişkenler
    df["sekerleme_var"] = (df["sekerleme_suresi_dk"] > 0).astype(int)
    df["asiri_calisma"] = (df["gunluk_calisma_saati"] > 9).astype(int)
    df["az_uyanma"]     = (df["gecelik_uyanma_sayisi"] <= 1).astype(int)
    df["yuksek_kafein"] = (df["uyku_oncesi_kafein_mg"] > 100).astype(int)

    return df


# ----------------------------------------------------------------------
# 3) ÖN İŞLEME (eksik değer + label encoding + target encoding)
# ----------------------------------------------------------------------
def kfold_target_encode(train, test, col, y_train, n_splits=5, smoothing=10):
    """K-fold target encoding (leakage'siz)."""
    train_te = np.zeros(len(train))
    global_mean = y_train.mean()
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    for tr_idx, va_idx in kf.split(train):
        tmp = train.iloc[tr_idx].copy()
        tmp["__y__"] = y_train.iloc[tr_idx].values
        means = tmp.groupby(col)["__y__"].mean()
        counts = tmp.groupby(col)["__y__"].count()
        smooth = (means * counts + global_mean * smoothing) / (counts + smoothing)
        train_te[va_idx] = train.iloc[va_idx][col].map(smooth).fillna(global_mean).values
    # Test için tüm train üzerinden
    tmp = train.copy()
    tmp["__y__"] = y_train.values
    means = tmp.groupby(col)["__y__"].mean()
    counts = tmp.groupby(col)["__y__"].count()
    smooth = (means * counts + global_mean * smoothing) / (counts + smoothing)
    test_te = test[col].map(smooth).fillna(global_mean).values
    return train_te, test_te


def preprocess(train, test):
    train = add_features(train)
    test  = add_features(test)

    cat_cols = CATEGORICAL_COLS_BASE + [
        "yas_grubu", "bmi_grubu", "aktivite_seviyesi", "stres_grubu"
    ]

    # Eksik değer doldurma
    for col in train.columns:
        if col in (TARGET, ID_COL):
            continue
        if col in cat_cols:
            train[col] = train[col].fillna("missing").astype(str)
            test[col]  = test[col].fillna("missing").astype(str)
        else:
            med = train[col].median()
            train[col] = train[col].fillna(med)
            test[col]  = test[col].fillna(med)

    # Label encoding (LGB/XGB/HGB için)
    for col in cat_cols:
        le = LabelEncoder()
        all_vals = pd.concat([train[col], test[col]], axis=0)
        le.fit(all_vals)
        train[col + "_le"] = le.transform(train[col])
        test[col + "_le"]  = le.transform(test[col])

    # Yüksek kardinaliteli kategorikler için target encoding
    high_card = [c for c in ["meslek", "ulke"] if c in cat_cols]
    y_raw = train[TARGET]
    for col in high_card:
        train[col + "_te"], test[col + "_te"] = kfold_target_encode(train, test, col, y_raw)

    return train, test, cat_cols


# ----------------------------------------------------------------------
# 4) METRİK
# ----------------------------------------------------------------------
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


# ----------------------------------------------------------------------
# 5) OPTUNA TUNING FONKSİYONLARI
# ----------------------------------------------------------------------
def tune_lightgbm(X, y, feature_cols, n_trials=40):
    def objective(trial):
        params = {
            "objective": "regression", "metric": "rmse",
            "verbose": -1, "seed": SEED,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 16, 255),
            "max_depth": trial.suggest_int("max_depth", 4, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }
        kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
        oof = np.zeros(len(X))
        for tr_idx, va_idx in kf.split(X):
            m = lgb.LGBMRegressor(**params, n_estimators=3000)
            m.fit(X.iloc[tr_idx][feature_cols], y.iloc[tr_idx],
                  eval_set=[(X.iloc[va_idx][feature_cols], y.iloc[va_idx])],
                  callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
            oof[va_idx] = m.predict(X.iloc[va_idx][feature_cols])
        return rmse(y, oof)

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    print(f"[LGB Tuning] En iyi RMSE: {study.best_value:.5f}")
    return study.best_params


def tune_xgboost(X, y, feature_cols, n_trials=40):
    def objective(trial):
        params = {
            "objective": "reg:squarederror", "eval_metric": "rmse",
            "tree_method": "hist", "random_state": SEED,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "max_depth": trial.suggest_int("max_depth", 4, 12),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        }
        kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
        oof = np.zeros(len(X))
        for tr_idx, va_idx in kf.split(X):
            m = xgb.XGBRegressor(**params, n_estimators=3000, early_stopping_rounds=100)
            m.fit(X.iloc[tr_idx][feature_cols], y.iloc[tr_idx],
                  eval_set=[(X.iloc[va_idx][feature_cols], y.iloc[va_idx])], verbose=False)
            oof[va_idx] = m.predict(X.iloc[va_idx][feature_cols])
        return rmse(y, oof)

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    print(f"[XGB Tuning] En iyi RMSE: {study.best_value:.5f}")
    return study.best_params


def tune_catboost(X, y, feature_cols, cat_idx, n_trials=30):
    def objective(trial):
        params = {
            "iterations": 3000,
            "loss_function": "RMSE", "eval_metric": "RMSE",
            "random_seed": SEED, "verbose": 0,
            "early_stopping_rounds": 100,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "depth": trial.suggest_int("depth", 4, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
            "random_strength": trial.suggest_float("random_strength", 0.0, 10.0),
        }
        kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
        oof = np.zeros(len(X))
        for tr_idx, va_idx in kf.split(X):
            m = CatBoostRegressor(**params, cat_features=cat_idx)
            m.fit(X.iloc[tr_idx][feature_cols], y.iloc[tr_idx],
                  eval_set=(X.iloc[va_idx][feature_cols], y.iloc[va_idx]))
            oof[va_idx] = m.predict(X.iloc[va_idx][feature_cols])
        return rmse(y, oof)

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    print(f"[CAT Tuning] En iyi RMSE: {study.best_value:.5f}")
    return study.best_params


# ----------------------------------------------------------------------
# 6) MODEL EĞİTİMİ (10-fold + multi-seed)
# ----------------------------------------------------------------------
def train_lightgbm(X, y, X_test, feature_cols, params, seeds=SEEDS):
    oof_avg  = np.zeros(len(X))
    pred_avg = np.zeros(len(X_test))
    for s in seeds:
        oof_s  = np.zeros(len(X))
        pred_s = np.zeros(len(X_test))
        kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=s)
        for fold, (tr_idx, va_idx) in enumerate(kf.split(X), 1):
            p = {**params, "seed": s, "verbose": -1, "objective": "regression", "metric": "rmse"}
            m = lgb.LGBMRegressor(**p, n_estimators=5000)
            m.fit(X.iloc[tr_idx][feature_cols], y.iloc[tr_idx],
                  eval_set=[(X.iloc[va_idx][feature_cols], y.iloc[va_idx])],
                  callbacks=[lgb.early_stopping(150), lgb.log_evaluation(0)])
            oof_s[va_idx] = m.predict(X.iloc[va_idx][feature_cols])
            pred_s += m.predict(X_test[feature_cols]) / N_SPLITS
        print(f"  [LGB seed={s}] OOF RMSE: {rmse(y, oof_s):.5f}")
        oof_avg  += oof_s  / len(seeds)
        pred_avg += pred_s / len(seeds)
    print(f"[LightGBM] Multi-seed OOF RMSE: {rmse(y, oof_avg):.5f}")
    return oof_avg, pred_avg


def train_xgboost(X, y, X_test, feature_cols, params, seeds=SEEDS):
    oof_avg  = np.zeros(len(X))
    pred_avg = np.zeros(len(X_test))
    for s in seeds:
        oof_s  = np.zeros(len(X))
        pred_s = np.zeros(len(X_test))
        kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=s)
        for fold, (tr_idx, va_idx) in enumerate(kf.split(X), 1):
            p = {**params, "random_state": s, "tree_method": "hist",
                 "objective": "reg:squarederror", "eval_metric": "rmse"}
            m = xgb.XGBRegressor(**p, n_estimators=5000, early_stopping_rounds=150)
            m.fit(X.iloc[tr_idx][feature_cols], y.iloc[tr_idx],
                  eval_set=[(X.iloc[va_idx][feature_cols], y.iloc[va_idx])], verbose=False)
            oof_s[va_idx] = m.predict(X.iloc[va_idx][feature_cols])
            pred_s += m.predict(X_test[feature_cols]) / N_SPLITS
        print(f"  [XGB seed={s}] OOF RMSE: {rmse(y, oof_s):.5f}")
        oof_avg  += oof_s  / len(seeds)
        pred_avg += pred_s / len(seeds)
    print(f"[XGBoost] Multi-seed OOF RMSE: {rmse(y, oof_avg):.5f}")
    return oof_avg, pred_avg


def train_catboost(X, y, X_test, feature_cols, cat_idx, params, seeds=SEEDS):
    oof_avg  = np.zeros(len(X))
    pred_avg = np.zeros(len(X_test))
    for s in seeds:
        oof_s  = np.zeros(len(X))
        pred_s = np.zeros(len(X_test))
        kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=s)
        for fold, (tr_idx, va_idx) in enumerate(kf.split(X), 1):
            m = CatBoostRegressor(
                **params, iterations=5000, loss_function="RMSE", eval_metric="RMSE",
                random_seed=s, cat_features=cat_idx,
                early_stopping_rounds=150, verbose=0,
            )
            m.fit(X.iloc[tr_idx][feature_cols], y.iloc[tr_idx],
                  eval_set=(X.iloc[va_idx][feature_cols], y.iloc[va_idx]))
            oof_s[va_idx] = m.predict(X.iloc[va_idx][feature_cols])
            pred_s += m.predict(X_test[feature_cols]) / N_SPLITS
        print(f"  [CAT seed={s}] OOF RMSE: {rmse(y, oof_s):.5f}")
        oof_avg  += oof_s  / len(seeds)
        pred_avg += pred_s / len(seeds)
    print(f"[CatBoost] Multi-seed OOF RMSE: {rmse(y, oof_avg):.5f}")
    return oof_avg, pred_avg


def train_hgb(X, y, X_test, feature_cols, seeds=SEEDS):
    oof_avg  = np.zeros(len(X))
    pred_avg = np.zeros(len(X_test))
    for s in seeds:
        oof_s  = np.zeros(len(X))
        pred_s = np.zeros(len(X_test))
        kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=s)
        for fold, (tr_idx, va_idx) in enumerate(kf.split(X), 1):
            m = HistGradientBoostingRegressor(
                max_iter=2000, learning_rate=0.03, max_depth=8,
                min_samples_leaf=20, l2_regularization=0.1,
                early_stopping=True, validation_fraction=0.1,
                random_state=s,
            )
            m.fit(X.iloc[tr_idx][feature_cols], y.iloc[tr_idx])
            oof_s[va_idx] = m.predict(X.iloc[va_idx][feature_cols])
            pred_s += m.predict(X_test[feature_cols]) / N_SPLITS
        oof_avg  += oof_s  / len(seeds)
        pred_avg += pred_s / len(seeds)
    print(f"[HGB] Multi-seed OOF RMSE: {rmse(y, oof_avg):.5f}")
    return oof_avg, pred_avg


# ----------------------------------------------------------------------
# 7) ANA AKIŞ
# ----------------------------------------------------------------------
def main():
    train, test, samp = load_data()
    train, test, cat_cols = preprocess(train, test)

    y_raw = train[TARGET].copy()

    # Hedef dönüşüm kararı
    skew = y_raw.skew()
    use_log = (abs(skew) > 0.5) if USE_LOG is None else USE_LOG
    if use_log and (y_raw.min() >= 0):
        print(f"\n>>> Hedef skew={skew:.3f}, log1p dönüşümü UYGULANACAK")
        y = pd.Series(np.log1p(y_raw.values), index=y_raw.index)
    else:
        print(f"\n>>> Hedef skew={skew:.3f}, dönüşüm uygulanmayacak")
        y = y_raw.copy()
        use_log = False

    # Feature setleri
    feat_lgb_xgb = [c for c in train.columns
                    if c not in (TARGET, ID_COL) and c not in cat_cols
                    and not pd.api.types.is_object_dtype(train[c])]

    feat_cat = [c for c in train.columns
                if c not in (TARGET, ID_COL) and not c.endswith("_le")]
    cat_features_idx = [feat_cat.index(c) for c in cat_cols if c in feat_cat]

    print(f"LGB/XGB/HGB feature sayısı: {len(feat_lgb_xgb)}")
    print(f"CatBoost feature sayısı: {len(feat_cat)} (kategorik: {len(cat_features_idx)})")

    # ------------------------------------------------------------------
    # OPTUNA TUNING
    # ------------------------------------------------------------------
    print("\n" + "="*60 + "\n  OPTUNA HYPERPARAMETER TUNING\n" + "="*60)
    print("\n--- LightGBM tuning ---")
    best_lgb = tune_lightgbm(train, y, feat_lgb_xgb, n_trials=N_TRIALS)
    print("\n--- XGBoost tuning ---")
    best_xgb = tune_xgboost(train, y, feat_lgb_xgb, n_trials=N_TRIALS)
    print("\n--- CatBoost tuning ---")
    best_cat = tune_catboost(train, y, feat_cat, cat_features_idx, n_trials=max(20, N_TRIALS // 2))

    # ------------------------------------------------------------------
    # NİHAİ EĞİTİM (10-fold + multi-seed)
    # ------------------------------------------------------------------
    print("\n" + "="*60 + "\n  NİHAİ EĞİTİM (10-fold + multi-seed)\n" + "="*60)
    print("\n--- LightGBM ---")
    oof_lgb, pred_lgb = train_lightgbm(train, y, test, feat_lgb_xgb, best_lgb)
    print("\n--- XGBoost ---")
    oof_xgb, pred_xgb = train_xgboost(train, y, test, feat_lgb_xgb, best_xgb)
    print("\n--- CatBoost ---")
    oof_cat, pred_cat = train_catboost(train, y, test, feat_cat, cat_features_idx, best_cat)
    print("\n--- HistGradientBoosting ---")
    oof_hgb, pred_hgb = train_hgb(train, y, test, feat_lgb_xgb)

    # ------------------------------------------------------------------
    # SKORLARI ORİJİNAL UZAYDA RAPORLA
    # ------------------------------------------------------------------
    def to_real(x): return np.expm1(x) if use_log else x

    print("\n" + "="*60 + "\n  ORİJİNAL UZAY OOF RMSE\n" + "="*60)
    oofs_real  = {
        "LGB": to_real(oof_lgb), "XGB": to_real(oof_xgb),
        "CAT": to_real(oof_cat), "HGB": to_real(oof_hgb),
    }
    preds_real = {
        "LGB": to_real(pred_lgb), "XGB": to_real(pred_xgb),
        "CAT": to_real(pred_cat), "HGB": to_real(pred_hgb),
    }
    for name, oof in oofs_real.items():
        print(f"  {name}: {rmse(y_raw, oof):.5f}")

    # ------------------------------------------------------------------
    # BLEND (Nelder-Mead ağırlık optimizasyonu)
    # ------------------------------------------------------------------
    oof_stack  = np.column_stack([oofs_real[k]  for k in ["LGB","XGB","CAT","HGB"]])
    pred_stack = np.column_stack([preds_real[k] for k in ["LGB","XGB","CAT","HGB"]])

    def neg_rmse(w):
        w = np.clip(w, 0, 1)
        if w.sum() == 0: return 1e9
        w = w / w.sum()
        return rmse(y_raw, oof_stack @ w)

    res = minimize(neg_rmse, x0=np.ones(4)/4, method="Nelder-Mead")
    w = np.clip(res.x, 0, 1); w = w / w.sum()
    oof_blend  = oof_stack @ w
    pred_blend = pred_stack @ w
    print(f"\n>>> BLEND ağırlıkları: LGB={w[0]:.3f} XGB={w[1]:.3f} "
          f"CAT={w[2]:.3f} HGB={w[3]:.3f}")
    print(f">>> BLEND OOF RMSE: {rmse(y_raw, oof_blend):.5f}")

    # ------------------------------------------------------------------
    # STACKING (Ridge meta-model, CV'li)
    # ------------------------------------------------------------------
    meta = Ridge(alpha=1.0)
    oof_stack_cv = cross_val_predict(meta, oof_stack, y_raw, cv=10)
    print(f">>> STACK (Ridge) OOF RMSE: {rmse(y_raw, oof_stack_cv):.5f}")
    meta.fit(oof_stack, y_raw)
    pred_stack_final = meta.predict(pred_stack)

    # En iyi stratejiyi seç
    rmse_blend = rmse(y_raw, oof_blend)
    rmse_stack = rmse(y_raw, oof_stack_cv)
    if rmse_stack < rmse_blend:
        print(f"\n>>> STACK seçildi (RMSE {rmse_stack:.5f} < {rmse_blend:.5f})")
        final_pred = pred_stack_final
    else:
        print(f"\n>>> BLEND seçildi (RMSE {rmse_blend:.5f} <= {rmse_stack:.5f})")
        final_pred = pred_blend

    # ------------------------------------------------------------------
    # GÖNDERİM DOSYASI
    # ------------------------------------------------------------------
    submission = pd.DataFrame({ID_COL: test[ID_COL], TARGET: final_pred})
    submission.columns = samp.columns
    out_path = os.path.join(OUT_DIR, "submission_final.csv")
    submission.to_csv(out_path, index=False)

    print(f"\nFormat: {list(submission.columns)} | satır: {len(submission)}")
    print(f"Tahmin istatistikleri: min={final_pred.min():.2f} "
          f"max={final_pred.max():.2f} mean={final_pred.mean():.2f}")
    print(f"\n✓ Gönderim dosyası kaydedildi: {out_path}")


if __name__ == "__main__":
    main()