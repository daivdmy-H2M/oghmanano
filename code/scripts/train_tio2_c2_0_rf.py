from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parent.parent
DATA_DIR = PROJECT_ROOT / "bin" / "tio2-c_2.0"

TRAIN_DIR = DATA_DIR / "2.0train"
TEST_DIR = DATA_DIR / "2.0test"
ANALYSIS_TEST_DIR = DATA_DIR / "analysis" / "test"
ANALYSIS_COMPARE_DIR = DATA_DIR / "analysis" / "compare"

X_PATH = DATA_DIR / "tio2-c2.0_x.csv"
Y_PATH = DATA_DIR / "tio2-c2.0_y.csv"
Y_HAT_PATH = DATA_DIR / "tio2-c2.0_y_hat.csv"
MODEL_PATH = DATA_DIR / "delta_y_model.pkl"

TARGET_MAP = [
    ("Simulation_Voc", "JV_default_Voc", "delta_Voc"),
    ("Simulation_Jsc", "JV_default_Jsc", "delta_Jsc"),
    ("Simulation_PCE", "JV_default_PCE", "delta_PCE"),
    ("Simulation_FF", "JV_default_FF", "delta_FF"),
]


def load_aligned_data():
    for file_path in (X_PATH, Y_PATH, Y_HAT_PATH):
        if not file_path.exists():
            raise FileNotFoundError(f"缺少输入文件: {file_path}")

    df_x = pd.read_csv(X_PATH).sort_values("Ref_ID").reset_index(drop=True)
    df_y = pd.read_csv(Y_PATH).sort_values("Ref_ID").reset_index(drop=True)
    df_y_hat = pd.read_csv(Y_HAT_PATH).sort_values("Ref_ID").reset_index(drop=True)

    if not (
        df_x["Ref_ID"].equals(df_y["Ref_ID"])
        and df_y["Ref_ID"].equals(df_y_hat["Ref_ID"])
    ):
        raise ValueError("tio2-c2.0_x/y/y_hat 的 Ref_ID 未完全对齐。")

    return df_x, df_y, df_y_hat


def build_delta_targets(df_y, df_y_hat):
    delta_df = pd.DataFrame(index=df_y.index)
    for sim_col, real_col, delta_col in TARGET_MAP:
        if sim_col not in df_y.columns:
            raise KeyError(f"tio2-c2.0_y.csv 缺少列: {sim_col}")
        if real_col not in df_y_hat.columns:
            raise KeyError(f"tio2-c2.0_y_hat.csv 缺少列: {real_col}")

        sim_val = pd.to_numeric(df_y[sim_col], errors="coerce")
        real_val = pd.to_numeric(df_y_hat[real_col], errors="coerce")
        delta_df[delta_col] = sim_val - real_val
    return delta_df


def split_by_ref_id(df_x, df_y, df_y_hat):
    ref_ids = df_x["Ref_ID"].reset_index(drop=True)
    kfold = KFold(n_splits=4, shuffle=True, random_state=42)
    train_idx, test_idx = next(kfold.split(ref_ids))

    train_x = df_x.iloc[train_idx].sort_values("Ref_ID").reset_index(drop=True)
    train_y = df_y.iloc[train_idx].sort_values("Ref_ID").reset_index(drop=True)
    train_y_hat = df_y_hat.iloc[train_idx].sort_values("Ref_ID").reset_index(drop=True)

    test_x = df_x.iloc[test_idx].sort_values("Ref_ID").reset_index(drop=True)
    test_y = df_y.iloc[test_idx].sort_values("Ref_ID").reset_index(drop=True)
    test_y_hat = df_y_hat.iloc[test_idx].sort_values("Ref_ID").reset_index(drop=True)

    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    train_x.to_csv(TRAIN_DIR / "train_x.csv", index=False, encoding="utf-8-sig")
    train_y.to_csv(TRAIN_DIR / "train_y.csv", index=False, encoding="utf-8-sig")
    train_y_hat.to_csv(TRAIN_DIR / "train_y_hat.csv", index=False, encoding="utf-8-sig")
    test_x.to_csv(TEST_DIR / "test_x.csv", index=False, encoding="utf-8-sig")
    test_y.to_csv(TEST_DIR / "test_y.csv", index=False, encoding="utf-8-sig")
    test_y_hat.to_csv(TEST_DIR / "test_y_hat.csv", index=False, encoding="utf-8-sig")

    print(f"完成 4 组随机拆分：训练集 {len(train_x)} 条，测试集 {len(test_x)} 条。")
    print(f"训练集输出目录: {TRAIN_DIR}")
    print(f"测试集输出目录: {TEST_DIR}")

    return train_x, train_y, train_y_hat, test_x, test_y, test_y_hat


def build_model(x_sample):
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

    base_regressor = RandomForestRegressor(
        n_estimators=500,
        max_depth=16,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", MultiOutputRegressor(base_regressor)),
        ]
    )


def evaluate_and_save(
    model,
    train_x,
    train_delta,
    test_x,
    test_delta,
    test_y,
    test_y_hat,
):
    ANALYSIS_TEST_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_COMPARE_DIR.mkdir(parents=True, exist_ok=True)

    train_pred = pd.DataFrame(
        model.predict(train_x),
        columns=train_delta.columns,
        index=train_delta.index,
    )
    test_pred = pd.DataFrame(
        model.predict(test_x),
        columns=test_delta.columns,
        index=test_delta.index,
    )

    test_result = pd.DataFrame({"Ref_ID": test_y["Ref_ID"]})
    metric_rows = []

    for sim_col, real_col, delta_col in TARGET_MAP:
        sim_true_test = pd.to_numeric(test_y[sim_col], errors="coerce")
        real_test = pd.to_numeric(test_y_hat[real_col], errors="coerce")
        sim_pred_test = real_test + test_pred[delta_col]
        delta_true_test = test_delta[delta_col]
        delta_pred_test = test_pred[delta_col]

        valid_mask = ~(delta_true_test.isna() | delta_pred_test.isna())
        y_true = delta_true_test.loc[valid_mask]
        y_pred = delta_pred_test.loc[valid_mask]

        mae = mean_absolute_error(y_true, y_pred)
        rmse = mean_squared_error(y_true, y_pred) ** 0.5
        r2 = r2_score(y_true, y_pred)

        metric_rows.append(
            {
                "target": delta_col,
                "sample_count": int(valid_mask.sum()),
                "MAE": mae,
                "RMSE": rmse,
                "R2": r2,
            }
        )

        test_result[f"{delta_col}_true"] = delta_true_test
        test_result[f"{delta_col}_pred"] = delta_pred_test
        test_result[f"{delta_col}_error"] = delta_pred_test - delta_true_test
        test_result[f"{sim_col}_true"] = sim_true_test
        test_result[f"{sim_col}_pred"] = sim_pred_test

    metrics_df = pd.DataFrame(metric_rows)
    test_result.to_csv(
        ANALYSIS_TEST_DIR / "test_fit_predictions.csv", index=False, encoding="utf-8-sig"
    )
    metrics_df.to_csv(
        ANALYSIS_TEST_DIR / "test_fit_metrics.csv", index=False, encoding="utf-8-sig"
    )

    for _, _, delta_col in TARGET_MAP:
        plot_compare(
            train_true=train_delta[delta_col],
            train_pred=train_pred[delta_col],
            test_true=test_delta[delta_col],
            test_pred=test_pred[delta_col],
            title=f"{delta_col} - RF",
            output_path=ANALYSIS_COMPARE_DIR / f"{delta_col}_train_test_compare.png",
        )

    print(f"测试集拟合明细输出: {ANALYSIS_TEST_DIR / 'test_fit_predictions.csv'}")
    print(f"测试集拟合指标输出: {ANALYSIS_TEST_DIR / 'test_fit_metrics.csv'}")
    print(f"对比图输出目录: {ANALYSIS_COMPARE_DIR}")


def plot_compare(train_true, train_pred, test_true, test_pred, title, output_path):
    train_df = pd.DataFrame({"true": train_true, "pred": train_pred}).dropna()
    test_df = pd.DataFrame({"true": test_true, "pred": test_pred}).dropna()
    combined = pd.concat([train_df, test_df], axis=0)
    if combined.empty:
        return

    min_val = min(combined["true"].min(), combined["pred"].min())
    max_val = max(combined["true"].max(), combined["pred"].max())
    pad = (max_val - min_val) * 0.05 if max_val > min_val else 0.1
    lower = min_val - pad
    upper = max_val + pad

    r2_all = r2_score(combined["true"], combined["pred"]) if len(combined) >= 2 else float("nan")

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
    if pd.notna(r2_all):
        ax.set_title(f"{title} (R²={r2_all:.4f})")
    else:
        ax.set_title(title)
    ax.legend(loc="upper left")
    ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def main():
    print("读取 tio2-c2.0 数据...")
    df_x, df_y, df_y_hat = load_aligned_data()
    train_x_df, train_y_df, train_y_hat_df, test_x_df, test_y_df, test_y_hat_df = split_by_ref_id(
        df_x, df_y, df_y_hat
    )

    train_delta = build_delta_targets(train_y_df, train_y_hat_df)
    test_delta = build_delta_targets(test_y_df, test_y_hat_df)

    train_features = train_x_df.drop(columns=["Ref_ID"]).copy()
    test_features = test_x_df.drop(columns=["Ref_ID"]).copy()

    valid_train_mask = ~train_delta.isna().any(axis=1)
    valid_test_mask = ~test_delta.isna().any(axis=1)
    train_features = train_features.loc[valid_train_mask].reset_index(drop=True)
    train_delta = train_delta.loc[valid_train_mask].reset_index(drop=True)
    test_features = test_features.loc[valid_test_mask].reset_index(drop=True)
    test_delta = test_delta.loc[valid_test_mask].reset_index(drop=True)
    test_y_df = test_y_df.loc[valid_test_mask].reset_index(drop=True)
    test_y_hat_df = test_y_hat_df.loc[valid_test_mask].reset_index(drop=True)

    model = build_model(train_features)
    model.fit(train_features, train_delta)
    joblib.dump(model, MODEL_PATH)
    print(f"模型已保存: {MODEL_PATH}")

    evaluate_and_save(
        model=model,
        train_x=train_features,
        train_delta=train_delta,
        test_x=test_features,
        test_delta=test_delta,
        test_y=test_y_df,
        test_y_hat=test_y_hat_df,
    )


if __name__ == "__main__":
    main()
