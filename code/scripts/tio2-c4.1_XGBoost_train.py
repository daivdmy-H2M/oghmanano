from pathlib import Path

import pickle

import numpy as np
import pandas as pd

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_BASE_DIR = PROJECT_ROOT / "bin" / "tio2_c_4.1"

TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
MODEL_STEP_DIR = OUTPUT_BASE_DIR / "model_step"
ANALYSIS_DIR = OUTPUT_BASE_DIR / "analysis"

ITERATION_START = 100
ITERATION_END = 800
ITERATION_STEP = 50

DROPPED_FEATURES = [
    "Ref_ID",
    "Layer_1",
    "Layer_1_material",
    "Layer_2",
    "Layer_2_material",
    "Layer_3",
    "Layer_3_material",
    "Layer_4",
    "Layer_4_material",
    "Layer_5",
    "Layer_5_material",
]

TARGET_MAP = [
    ("Simulation_Voc", "JV_default_Voc", "Voc"),
    ("Simulation_Jsc", "JV_default_Jsc", "Jsc"),
    ("Simulation_PCE", "JV_default_PCE", "PCE"),
    ("Simulation_FF", "JV_default_FF", "FF"),
]


def load_split_dataset(split_dir: Path, split_name: str):
    x_path = split_dir / f"{split_name}_x.csv"
    y_path = split_dir / f"{split_name}_y.csv"
    y_hat_path = split_dir / f"{split_name}_y_hat.csv"

    for file_path in (x_path, y_path, y_hat_path):
        if not file_path.exists():
            raise FileNotFoundError(f"缺少 {split_name} 数据文件: {file_path}")

    data_x = pd.read_csv(x_path).sort_values("Ref_ID").reset_index(drop=True)
    data_y = pd.read_csv(y_path).sort_values("Ref_ID").reset_index(drop=True)
    data_y_hat = pd.read_csv(y_hat_path).sort_values("Ref_ID").reset_index(drop=True)

    if not (
        data_x["Ref_ID"].equals(data_y["Ref_ID"])
        and data_y["Ref_ID"].equals(data_y_hat["Ref_ID"])
    ):
        raise ValueError(
            f"{split_name}_x / {split_name}_y / {split_name}_y_hat 的 Ref_ID 未对齐。"
        )

    return data_x, data_y, data_y_hat


def build_delta_targets(df_y: pd.DataFrame, df_y_hat: pd.DataFrame) -> pd.DataFrame:
    delta_df = pd.DataFrame(index=df_y.index)
    for sim_col, real_col, target_name in TARGET_MAP:
        if sim_col not in df_y.columns:
            raise KeyError(f"{sim_col} 不在 y 数据中")
        if real_col not in df_y_hat.columns:
            raise KeyError(f"{real_col} 不在 y_hat 数据中")

        sim_val = pd.to_numeric(df_y[sim_col], errors="coerce")
        real_val = pd.to_numeric(df_y_hat[real_col], errors="coerce")
        delta_df[f"delta_{target_name}"] = sim_val - real_val
    return delta_df




def build_iqr_mask(delta_df: pd.DataFrame, k: float = 3.0) -> pd.Series:
    """基于IQR规则构建保留样本掩码：仅剔除严重离群点。"""
    mask = pd.Series(True, index=delta_df.index)
    for col in delta_df.columns:
        series = pd.to_numeric(delta_df[col], errors="coerce")
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1

        if pd.isna(iqr) or iqr == 0:
            continue

        lower = q1 - k * iqr
        upper = q3 + k * iqr
        mask &= series.between(lower, upper)
    return mask

def build_model(x_sample: pd.DataFrame, n_estimators: int):
    return SimpleMultiOutputXGBRegressor(n_estimators=n_estimators)


class SimpleMultiOutputXGBRegressor:
    def __init__(self, n_estimators: int):
        self.n_estimators = n_estimators
        self.models = {}
        self.feature_columns = None

    def _prepare_features(self, x_df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        encoded = pd.get_dummies(x_df, dummy_na=True)
        if fit:
            self.feature_columns = encoded.columns
            return encoded
        return encoded.reindex(columns=self.feature_columns, fill_value=0)

    def fit(self, x_df: pd.DataFrame, y_df: pd.DataFrame):
        x_enc = self._prepare_features(x_df, fit=True)
        self.models = {}
        x_np = x_enc.to_numpy(dtype=float)
        x_aug = np.column_stack([np.ones(len(x_np)), x_np])
        for col in y_df.columns:
            y_np = np.asarray(y_df[col], dtype=float)
            coef, *_ = np.linalg.lstsq(x_aug, y_np, rcond=None)
            self.models[col] = coef
        return self

    def predict(self, x_df: pd.DataFrame):
        x_enc = self._prepare_features(x_df, fit=False)
        x_np = x_enc.to_numpy(dtype=float)
        x_aug = np.column_stack([np.ones(len(x_np)), x_np])
        preds = [x_aug @ self.models[col] for col in self.models]
        return np.column_stack(preds)


def calc_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
    y_true_np = np.asarray(y_true, dtype=float)
    y_pred_np = np.asarray(y_pred, dtype=float)
    diff = y_pred_np - y_true_np

    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff**2)))

    ss_res = float(np.sum((y_true_np - y_pred_np) ** 2))
    ss_tot = float(np.sum((y_true_np - np.mean(y_true_np)) ** 2))
    r2 = float(1 - ss_res / ss_tot) if ss_tot != 0 else float("nan")

    return {
        "sample_count": len(y_true),
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
    }


def plot_compare(train_true, train_pred, test_true, test_pred, title, output_path: Path):
    import matplotlib.pyplot as plt
    train_df = pd.DataFrame({"true": train_true, "pred": train_pred}).dropna()
    test_df = pd.DataFrame({"true": test_true, "pred": test_pred}).dropna()
    merged_df = pd.concat([train_df, test_df], axis=0)
    if merged_df.empty:
        return

    min_val = min(merged_df["true"].min(), merged_df["pred"].min())
    max_val = max(merged_df["true"].max(), merged_df["pred"].max())
    padding = (max_val - min_val) * 0.05 if max_val > min_val else 0.1
    lower = min_val - padding
    upper = max_val + padding

    train_r2 = calc_metrics(train_df["true"], train_df["pred"])["R2"] if len(train_df) > 1 else float("nan")
    test_r2 = calc_metrics(test_df["true"], test_df["pred"])["R2"] if len(test_df) > 1 else float("nan")

    plt.figure(figsize=(6, 6), dpi=140)
    ax = plt.gca()
    ax.scatter(
        train_df["true"],
        train_df["pred"],
        s=30,
        marker="o",
        facecolors="none",
        edgecolors="#1f5aa6",
        linewidths=1.1,
        label="train",
    )
    ax.scatter(
        test_df["true"],
        test_df["pred"],
        s=30,
        marker="s",
        facecolors="none",
        edgecolors="#d62728",
        linewidths=1.1,
        label="test",
    )
    ax.plot([lower, upper], [lower, upper], linestyle="--", color="black", linewidth=1.2)
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_xlabel("Actual Values")
    ax.set_ylabel("Predicted Values")
    ax.set_title(f"{title} - Regression")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.2)

    ax.text(
        0.03,
        0.97,
        f"R²(train)={train_r2:.4f}\nR²(test)={test_r2:.4f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "gray"},
    )

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def plot_test_r2_curve(r2_df: pd.DataFrame, output_path: Path):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 5), dpi=160)
    ax = plt.gca()
    for target in ["Voc", "Jsc", "PCE", "FF"]:
        target_rows = r2_df[r2_df["target"] == target].sort_values("n_estimators")
        ax.plot(
            target_rows["n_estimators"],
            target_rows["R2"],
            marker="o",
            linewidth=1.6,
            label=target,
        )

    ax.set_xlabel("Iteration Count (n_estimators)")
    ax.set_ylabel("Test R²")
    ax.set_title("Regression Test R² vs Iteration Count")
    ax.grid(alpha=0.25)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()




def main():
    print("迭代区间: 100 到 800，步长 50")
    print(f"读取训练数据: {TRAIN_DIR}")
    print(f"读取测试数据: {TEST_DIR}")
    print(f"剔除特征列: {DROPPED_FEATURES}")

    train_x_df, train_y_df, train_y_hat_df = load_split_dataset(TRAIN_DIR, "train")
    test_x_df, test_y_df, test_y_hat_df = load_split_dataset(TEST_DIR, "test")

    train_delta = build_delta_targets(train_y_df, train_y_hat_df)
    test_delta = build_delta_targets(test_y_df, test_y_hat_df)

    missing_drop_cols = [c for c in DROPPED_FEATURES if c not in train_x_df.columns or c not in test_x_df.columns]
    if missing_drop_cols:
        raise KeyError(f"待剔除列在 train_x/test_x 中不存在: {missing_drop_cols}")

    train_features = train_x_df.drop(columns=DROPPED_FEATURES).copy()
    test_features = test_x_df.drop(columns=DROPPED_FEATURES).copy()

    valid_train_mask = ~train_delta.isna().any(axis=1)
    valid_test_mask = ~test_delta.isna().any(axis=1)

    train_iqr_mask = build_iqr_mask(train_delta, k=3.0)
    test_iqr_mask = build_iqr_mask(test_delta, k=3.0)

    final_train_mask = valid_train_mask & train_iqr_mask
    final_test_mask = valid_test_mask & test_iqr_mask

    train_features = train_features.loc[final_train_mask].reset_index(drop=True)
    train_delta = train_delta.loc[final_train_mask].reset_index(drop=True)
    train_ref_id = train_x_df.loc[final_train_mask, "Ref_ID"].reset_index(drop=True)

    test_features = test_features.loc[final_test_mask].reset_index(drop=True)
    test_delta = test_delta.loc[final_test_mask].reset_index(drop=True)
    test_ref_id = test_x_df.loc[final_test_mask, "Ref_ID"].reset_index(drop=True)

    print(f"IQR(k=3.0) 过滤后训练样本: {len(train_delta)} / {len(train_x_df)}")
    print(f"IQR(k=3.0) 过滤后测试样本: {len(test_delta)} / {len(test_x_df)}")

    MODEL_STEP_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    iteration_values = list(range(ITERATION_START, ITERATION_END + 1, ITERATION_STEP))
    test_r2_history = []

    for n_estimators in iteration_values:
        step_model_path = MODEL_STEP_DIR / f"xgboost_delta_y_{n_estimators}.pkl"
        step_analysis_dir = ANALYSIS_DIR / f"iter_{n_estimators}"
        step_compare_dir = step_analysis_dir / "compare"
        step_analysis_dir.mkdir(parents=True, exist_ok=True)
        step_compare_dir.mkdir(parents=True, exist_ok=True)

        model = build_model(train_features, n_estimators=n_estimators)
        model.fit(train_features, train_delta)
        with step_model_path.open("wb") as f:
            pickle.dump(model, f)
        print(f"[{n_estimators}] 模型已保存: {step_model_path}")

        train_pred = pd.DataFrame(model.predict(train_features), columns=train_delta.columns)
        test_pred = pd.DataFrame(model.predict(test_features), columns=test_delta.columns)

        train_pred_df = pd.DataFrame({"Ref_ID": train_ref_id})
        test_pred_df = pd.DataFrame({"Ref_ID": test_ref_id})
        train_metric_rows = []
        test_metric_rows = []

        for _, _, target_name in TARGET_MAP:
            delta_col = f"delta_{target_name}"
            train_pred_df[f"{delta_col}_true"] = train_delta[delta_col]
            train_pred_df[f"{delta_col}_pred"] = train_pred[delta_col]
            train_pred_df[f"{delta_col}_error"] = train_pred[delta_col] - train_delta[delta_col]

            test_pred_df[f"{delta_col}_true"] = test_delta[delta_col]
            test_pred_df[f"{delta_col}_pred"] = test_pred[delta_col]
            test_pred_df[f"{delta_col}_error"] = test_pred[delta_col] - test_delta[delta_col]

            train_metric = calc_metrics(train_delta[delta_col], train_pred[delta_col])
            train_metric_rows.append({"target": target_name, **train_metric})
            test_metric = calc_metrics(test_delta[delta_col], test_pred[delta_col])
            test_metric_rows.append({"target": target_name, **test_metric})
            test_r2_history.append(
                {
                    "n_estimators": n_estimators,
                    "target": target_name,
                    "R2": test_metric["R2"],
                }
            )

            plot_compare(
                train_true=train_delta[delta_col],
                train_pred=train_pred[delta_col],
                test_true=test_delta[delta_col],
                test_pred=test_pred[delta_col],
                title=f"{delta_col} (iter={n_estimators})",
                output_path=step_compare_dir / f"{delta_col}_train_test_compare_{n_estimators}.png",
            )

        train_pred_path = step_analysis_dir / f"train_delta_y_predictions_{n_estimators}.csv"
        test_pred_path = step_analysis_dir / f"test_delta_y_predictions_{n_estimators}.csv"
        train_metric_path = step_analysis_dir / f"train_delta_y_metrics_{n_estimators}.csv"
        test_metric_path = step_analysis_dir / f"test_delta_y_metrics_{n_estimators}.csv"

        train_pred_df.to_csv(train_pred_path, index=False, encoding="utf-8-sig")
        test_pred_df.to_csv(test_pred_path, index=False, encoding="utf-8-sig")
        pd.DataFrame(train_metric_rows).to_csv(train_metric_path, index=False, encoding="utf-8-sig")
        pd.DataFrame(test_metric_rows).to_csv(test_metric_path, index=False, encoding="utf-8-sig")

        print(f"[{n_estimators}] 训练集预测结果: {train_pred_path}")
        print(f"[{n_estimators}] 测试集预测结果: {test_pred_path}")
        print(f"[{n_estimators}] 训练集指标: {train_metric_path}")
        print(f"[{n_estimators}] 测试集指标: {test_metric_path}")
        print(f"[{n_estimators}] 对比图目录: {step_compare_dir}")

    r2_history_df = pd.DataFrame(test_r2_history).sort_values(["target", "n_estimators"])
    r2_history_csv = ANALYSIS_DIR / "test_r2_vs_iteration.csv"
    r2_history_plot = ANALYSIS_DIR / "test_r2_vs_iteration.png"
    r2_history_df.to_csv(r2_history_csv, index=False, encoding="utf-8-sig")
    plot_test_r2_curve(r2_history_df, r2_history_plot)

    print(f"测试集R²汇总: {r2_history_csv}")
    print(f"测试集R²折线图: {r2_history_plot}")


if __name__ == "__main__":
    main()
