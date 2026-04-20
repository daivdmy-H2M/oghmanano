from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "bin" / "analyses2"

SIM_FILE = DATA_DIR / "analyses_y.csv"
EXP_FILE = DATA_DIR / "analyses_y_hat.csv"

# (x_col, y_col, plot_name)
PLOT_PAIRS = [
    ("Jsc", "Voc", "Jsc-Voc"),
    ("Jsc", "PCE", "Jsc-PCE"),
    ("Jsc", "FF", "Jsc-FF"),
    ("Voc", "FF", "Voc-FF"),
    ("Voc", "PCE", "Voc-PCE"),
    ("PCE", "FF", "PCE-FF"),
]

SIM_COL_MAP = {
    "Voc": "Simulation_Voc",
    "Jsc": "Simulation_Jsc",
    "PCE": "Simulation_PCE",
    "FF": "Simulation_FF",
}

EXP_COL_MAP = {
    "Voc": "JV_default_Voc",
    "Jsc": "JV_default_Jsc",
    "PCE": "JV_default_PCE",
    "FF": "JV_default_FF",
}


def read_and_align_data() -> pd.DataFrame:
    if not SIM_FILE.exists():
        raise FileNotFoundError(f"未找到仿真数据文件: {SIM_FILE}")
    if not EXP_FILE.exists():
        raise FileNotFoundError(f"未找到实验数据文件: {EXP_FILE}")

    sim_df = pd.read_csv(SIM_FILE)
    exp_df = pd.read_csv(EXP_FILE)

    for required_col in ["Ref_ID", *SIM_COL_MAP.values()]:
        if required_col not in sim_df.columns:
            raise KeyError(f"analyses_y.csv 缺少列: {required_col}")

    for required_col in ["Ref_ID", *EXP_COL_MAP.values()]:
        if required_col not in exp_df.columns:
            raise KeyError(f"analyses_y_hat.csv 缺少列: {required_col}")

    merged_df = pd.merge(
        sim_df[["Ref_ID", *SIM_COL_MAP.values()]],
        exp_df[["Ref_ID", *EXP_COL_MAP.values()]],
        on="Ref_ID",
        how="inner",
    ).sort_values("Ref_ID").reset_index(drop=True)

    for col in [*SIM_COL_MAP.values(), *EXP_COL_MAP.values()]:
        merged_df[col] = pd.to_numeric(merged_df[col], errors="coerce")

    return merged_df


def plot_one_pair(merged_df: pd.DataFrame, x_key: str, y_key: str, title: str) -> None:
    sim_x_col = SIM_COL_MAP[x_key]
    sim_y_col = SIM_COL_MAP[y_key]
    exp_x_col = EXP_COL_MAP[x_key]
    exp_y_col = EXP_COL_MAP[y_key]

    valid_df = merged_df.dropna(subset=[sim_x_col, sim_y_col, exp_x_col, exp_y_col]).copy()
    if valid_df.empty:
        print(f"⚠️ {title} 无有效数据，跳过绘图。")
        return

    plt.figure(figsize=(8, 6), dpi=150)
    plt.scatter(
        valid_df[sim_x_col],
        valid_df[sim_y_col],
        color="#1f77b4",
        alpha=0.3,
        s=28,
        label="analyses_y (Simulation)",
        zorder=2,
    )
    plt.scatter(
        valid_df[exp_x_col],
        valid_df[exp_y_col],
        color="#ff7f0e",
        alpha=0.3,
        s=28,
        label="analyses_y_hat (Experiment)",
        zorder=2,
    )

    for _, row in valid_df.iterrows():
        plt.plot(
            [row[sim_x_col], row[exp_x_col]],
            [row[sim_y_col], row[exp_y_col]],
            color="black",
            linewidth=0.5,
            alpha=0.8,
            zorder=1,
        )

    plt.xlabel(x_key)
    plt.ylabel(y_key)
    plt.title(title)
    plt.grid(alpha=0.2)
    plt.legend(loc="best")
    plt.tight_layout()

    output_file = OUTPUT_DIR / f"{title}.png"
    plt.savefig(output_file, bbox_inches="tight")
    plt.close()
    print(f"✅ 已保存: {output_file}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    merged_df = read_and_align_data()

    for x_key, y_key, title in PLOT_PAIRS:
        plot_one_pair(merged_df, x_key, y_key, title)

    print("🎉 六张图已全部处理完成。")


if __name__ == "__main__":
    main()
