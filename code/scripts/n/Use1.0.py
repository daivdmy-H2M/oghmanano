import json
import math
import os
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[3]

BASE_SIM_JSON_CANDIDATES = [
    PROJECT_ROOT / "code" / "configs" / "sim.json",
    PROJECT_ROOT / "configs" / "sim.json",
]
BASE_SIM_JSON = next((p for p in BASE_SIM_JSON_CANDIDATES if p.exists()), BASE_SIM_JSON_CANDIDATES[0])
OUTPUT_DIR = PROJECT_ROOT / "code" / "bin" / "use_tio2_c_4.5_1.0"
RUN_OUTPUT_DIR = OUTPUT_DIR / "runs"

OGHMA_ENGINE_PATH = r'"C:\Program Files (x86)\OghmaNano\oghma_core.exe"'

# 固定参数
CELL_AREA = 0.06
SUBSTRATE_THICKNESS_UM = 50.0
ETL_THICKNESS_NM = 200.0
HTL_THICKNESS_NM = 200.0
BACKCONTACT_THICKNESS_NM = 100.0

PEROVSKITE_THICKNESS_LIST = list(range(200, 801, 100))

MODEL_DIR = PROJECT_ROOT / "code" / "bin" / "tio2_c_4.5" / "model_step"
MODEL_MAP = {
    "PCE": MODEL_DIR / "xgboost_delta_y_500.pkl",
    "FF": MODEL_DIR / "xgboost_delta_y_800.pkl",
    "Voc": MODEL_DIR / "xgboost_delta_y_700.pkl",
    "Jsc": MODEL_DIR / "xgboost_delta_y_400.pkl",
}


def build_x_row(perovskite_thickness_nm: float):
    return {
        "Cell_area": CELL_AREA,
        "Layer_1": "Substrate",
        "Layer_1_material": "FTO",
        "Substrate_thickness": SUBSTRATE_THICKNESS_UM,
        "Layer_2": "ETL",
        "Layer_2_material": "TiO2-c",
        "ETL_thickness": ETL_THICKNESS_NM,
        "Layer_3": "Perovskite",
        "Layer_3_material": "MAPbI3",
        "Perovskite_thickness": perovskite_thickness_nm,
        "Layer_4": "HTL",
        "Layer_4_material": "Spiro-MeOTAD",
        "HTL_thickness": HTL_THICKNESS_NM,
        "Layer_5": "Backcontact",
        "Layer_5_material": "Au",
        "Backcontact_thickness": BACKCONTACT_THICKNESS_NM,
    }


def update_sim_json(sim_data, perovskite_thickness_nm: float):
    if "sim" in sim_data:
        sim_data["sim"]["use_json_local_root"] = "false"

    sim_data["epitaxy"]["segment0"]["dy"] = SUBSTRATE_THICKNESS_UM * 1e-6
    sim_data["epitaxy"]["segment1"]["dy"] = ETL_THICKNESS_NM * 1e-9
    sim_data["epitaxy"]["segment2"]["dy"] = perovskite_thickness_nm * 1e-9
    sim_data["epitaxy"]["segment3"]["dy"] = HTL_THICKNESS_NM * 1e-9
    sim_data["epitaxy"]["segment4"]["dy"] = BACKCONTACT_THICKNESS_NM * 1e-9

    side_length_m = math.sqrt(CELL_AREA) / 100.0
    sim_data["world"]["world_data"]["dx"] = side_length_m
    sim_data["world"]["world_data"]["dz"] = side_length_m
    for i in range(5):
        seg = sim_data["epitaxy"].get(f"segment{i}")
        if seg:
            seg["dx"] = side_length_m
            seg["dz"] = side_length_m

    return sim_data


def run_simulation(perovskite_thickness_nm: float):
    run_dir = RUN_OUTPUT_DIR / f"pero_{int(perovskite_thickness_nm)}nm"
    run_dir.mkdir(parents=True, exist_ok=True)

    dst_sim = run_dir / "sim.json"
    shutil.copyfile(BASE_SIM_JSON, dst_sim)

    with open(dst_sim, "r", encoding="utf-8") as f:
        sim_data = json.load(f)

    sim_data = update_sim_json(sim_data, perovskite_thickness_nm)

    with open(dst_sim, "w", encoding="utf-8") as f:
        json.dump(sim_data, f, indent="\t", sort_keys=False)

    cwd_backup = Path.cwd()
    try:
        os.chdir(run_dir)
        os.system(OGHMA_ENGINE_PATH)
    finally:
        os.chdir(cwd_backup)

    sim_info_path = run_dir / "sim_info.dat"
    result = {
        "Voc": np.nan,
        "Jsc": np.nan,
        "PCE": np.nan,
        "FF": np.nan,
    }

    if sim_info_path.exists():
        with open(sim_info_path, "r", encoding="utf-8") as f:
            info = json.load(f)
        result["Voc"] = info.get("voc", np.nan)
        result["Jsc"] = info.get("jsc", np.nan)
        result["PCE"] = info.get("pce", info.get("efficiency", np.nan))
        result["FF"] = info.get("ff", info.get("fill_factor", np.nan))

    return result


def predict_with_model(model, x_row: dict, sim_value: float):
    x_df = pd.DataFrame([x_row])
    if hasattr(model, "feature_names_in_"):
        feature_cols = list(model.feature_names_in_)
        for col in feature_cols:
            if col not in x_df.columns:
                x_df[col] = 0
        x_df = x_df[feature_cols]
    delta_pred = float(np.asarray(model.predict(x_df)).reshape(-1)[0])
    return sim_value + delta_pred


def main():
    if not BASE_SIM_JSON.exists():
        candidate_text = "\n".join([f"- {p}" for p in BASE_SIM_JSON_CANDIDATES])
        raise FileNotFoundError(f"未找到基础配置文件，已尝试以下路径：\n{candidate_text}")
    for metric, model_path in MODEL_MAP.items():
        if not model_path.exists():
            raise FileNotFoundError(f"未找到 {metric} 模型: {model_path}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RUN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    models = {metric: joblib.load(path) for metric, path in MODEL_MAP.items()}

    sim_rows = []
    model_rows = []

    for t in PEROVSKITE_THICKNESS_LIST:
        x_row = build_x_row(t)
        sim_result = run_simulation(t)

        sim_rows.append(
            {
                **x_row,
                "Simulation_Voc": sim_result["Voc"],
                "Simulation_Jsc": sim_result["Jsc"],
                "Simulation_FF": sim_result["FF"],
                "Simulation_PCE": sim_result["PCE"],
            }
        )

        model_rows.append(
            {
                **x_row,
                "Model_Voc": predict_with_model(models["Voc"], x_row, sim_result["Voc"]),
                "Model_Jsc": predict_with_model(models["Jsc"], x_row, sim_result["Jsc"]),
                "Model_FF": predict_with_model(models["FF"], x_row, sim_result["FF"]),
                "Model_PCE": predict_with_model(models["PCE"], x_row, sim_result["PCE"]),
            }
        )
        print(f"Perovskite_thickness={t}nm 完成")

    sim_df = pd.DataFrame(sim_rows)
    model_df = pd.DataFrame(model_rows)

    sim_path = OUTPUT_DIR / "simulate_Perovskite_200_800.csv"
    model_path = OUTPUT_DIR / "module_simulate_Perovskite_200_800.csv"

    sim_df.to_csv(sim_path, index=False, encoding="utf-8-sig")
    model_df.to_csv(model_path, index=False, encoding="utf-8-sig")

    print("\n全部完成，输出文件：")
    print(sim_path)
    print(model_path)
    print(f"仿真目录：{RUN_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
