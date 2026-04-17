from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parent.parent
DATA_DIR = PROJECT_ROOT / "bin" / "tio2-c_2.0"

TRAIN_DIR = DATA_DIR / "2.0train"
TEST_DIR = DATA_DIR / "2.0test"
OUTPUT_DIR = DATA_DIR / "XGBoost"
COMPARE_DIR = OUTPUT_DIR / "compare"

MODEL_PATH = OUTPUT_DIR / "delta_y_model.pkl"
TRAIN_PRED_PATH = OUTPUT_DIR / "train_delta_y_predictions.csv"
TEST_PRED_PATH = OUTPUT_DIR / "test_delta_y_predictions.csv"
TRAIN_METRIC_PATH = OUTPUT_DIR / "train_delta_y_metrics.csv"
TEST_METRIC_PATH = OUTPUT_DIR / "test_delta_y_metrics.csv"

TARGET_MAP = [
    ("Simulation_Voc", "JV_default_Voc", "delta_Voc"),
    ("Simulation_Jsc", "JV_default_Jsc", "delta_Jsc"),
    ("Simulation_PCE", "JV_default_PCE", "delta_PCE"),
    ("Simulation_FF", "JV_default_FF", "delta_FF"),
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
    for sim_col, real_col, delta_col in TARGET_MAP:
        if sim_col not in df_y.columns:
            raise KeyError(f"{sim_col} 不在 y 数据中")
        if real_col not in df_y_hat.columns:
            raise KeyError(f"{real_col} 不在 y_hat 数据中")

        sim_val = pd.to_numeric(df_y[sim_col], errors="coerce")
        real_val = pd.to_numeric(df_y_hat[real_col], errors="coerce")
        delta_df[delta_col] = sim_val - real_val
    return delta_df


def build_model(x_sample: pd.DataFrame) -> Pipeline:
    categorical_cols = x_sample.select_dtypes(
        include=["object", "category", "string"]
    ).columns
    numeric_cols = x_sample.select_dtypes(include=["number"]).columns

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                list(numeric_cols),
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                list(categorical_cols),
            ),
        ]
    )

    base_regressor = XGBRegressor(
        n_estimators=600,
        learning_rate=0.03,
        max_depth=6,
        min_child_weight=3,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.0,
        reg_lambda=1.0,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", MultiOutputRegressor(base_regressor)),
        ]
    )


def calc_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
    return {
        "sample_count": len(y_true),
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": mean_squared_error(y_true, y_pred) ** 0.5,
        "R2": r2_score(y_true, y_pred),
    }


def plot_compare(train_true, train_pred, test_true, test_pred, title, output_path: Path):
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

    train_r2 = r2_score(train_df["true"], train_df["pred"]) if len(train_df) > 1 else float("nan")
    test_r2 = r2_score(test_df["true"], test_df["pred"]) if len(test_df) > 1 else float("nan")

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
    ax.set_title(f"{title} - XGBoost")
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


def main():
    print(f"读取训练数据: {TRAIN_DIR}")
    print(f"读取测试数据: {TEST_DIR}")

    train_x_df, train_y_df, train_y_hat_df = load_split_dataset(TRAIN_DIR, "train")
    test_x_df, test_y_df, test_y_hat_df = load_split_dataset(TEST_DIR, "test")

    train_delta = build_delta_targets(train_y_df, train_y_hat_df)
    test_delta = build_delta_targets(test_y_df, test_y_hat_df)

    train_features = train_x_df.drop(columns=["Ref_ID"]).copy()
    test_features = test_x_df.drop(columns=["Ref_ID"]).copy()

    valid_train_mask = ~train_delta.isna().any(axis=1)
    valid_test_mask = ~test_delta.isna().any(axis=1)

    train_features = train_features.loc[valid_train_mask].reset_index(drop=True)
    train_delta = train_delta.loc[valid_train_mask].reset_index(drop=True)
    train_ref_id = train_x_df.loc[valid_train_mask, "Ref_ID"].reset_index(drop=True)

    test_features = test_features.loc[valid_test_mask].reset_index(drop=True)
    test_delta = test_delta.loc[valid_test_mask].reset_index(drop=True)
    test_ref_id = test_x_df.loc[valid_test_mask, "Ref_ID"].reset_index(drop=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    COMPARE_DIR.mkdir(parents=True, exist_ok=True)

    model = build_model(train_features)
    model.fit(train_features, train_delta)
    joblib.dump(model, MODEL_PATH)
    print(f"模型已保存: {MODEL_PATH}")

    train_pred = pd.DataFrame(model.predict(train_features), columns=train_delta.columns)
    test_pred = pd.DataFrame(model.predict(test_features), columns=test_delta.columns)

    train_pred_df = pd.DataFrame({"Ref_ID": train_ref_id})
    test_pred_df = pd.DataFrame({"Ref_ID": test_ref_id})
    train_metric_rows = []
    test_metric_rows = []

    for _, _, delta_col in TARGET_MAP:
        train_pred_df[f"{delta_col}_true"] = train_delta[delta_col]
        train_pred_df[f"{delta_col}_pred"] = train_pred[delta_col]
        train_pred_df[f"{delta_col}_error"] = train_pred[delta_col] - train_delta[delta_col]

        test_pred_df[f"{delta_col}_true"] = test_delta[delta_col]
        test_pred_df[f"{delta_col}_pred"] = test_pred[delta_col]
        test_pred_df[f"{delta_col}_error"] = test_pred[delta_col] - test_delta[delta_col]

        train_metric = calc_metrics(train_delta[delta_col], train_pred[delta_col])
        train_metric_rows.append({"target": delta_col, **train_metric})
        test_metric = calc_metrics(test_delta[delta_col], test_pred[delta_col])
        test_metric_rows.append({"target": delta_col, **test_metric})

        plot_compare(
            train_true=train_delta[delta_col],
            train_pred=train_pred[delta_col],
            test_true=test_delta[delta_col],
            test_pred=test_pred[delta_col],
            title=delta_col,
            output_path=COMPARE_DIR / f"{delta_col}_train_test_compare.png",
        )

    train_pred_df.to_csv(TRAIN_PRED_PATH, index=False, encoding="utf-8-sig")
    test_pred_df.to_csv(TEST_PRED_PATH, index=False, encoding="utf-8-sig")
    pd.DataFrame(train_metric_rows).to_csv(TRAIN_METRIC_PATH, index=False, encoding="utf-8-sig")
    pd.DataFrame(test_metric_rows).to_csv(TEST_METRIC_PATH, index=False, encoding="utf-8-sig")

    print(f"训练集预测结果: {TRAIN_PRED_PATH}")
    print(f"测试集预测结果: {TEST_PRED_PATH}")
    print(f"训练集指标: {TRAIN_METRIC_PATH}")
    print(f"测试集指标: {TEST_METRIC_PATH}")
    print(f"对比图目录: {COMPARE_DIR}")


if __name__ == "__main__":
    main()
