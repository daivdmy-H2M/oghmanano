# oghmanano
模拟太阳能电池效率

main 1 处理：
Substrate_stack_sequence：SLG | FTO
ETL_stack_sequence：TiO2-c
Perovskite_composition_long_form：MAPbI3
HTL_stack_sequence：Spiro-MeOTAD
Backcontact_stack_sequence：Au
并将结果进行保存，保存在bin文件夹中的simulation_results_summary.csv


main 2 处理：
Substrate_stack_sequence：SLG | FTO
ETL_stack_sequence：TiO2-c | TiO2-mp
Perovskite_composition_long_form：MAPbI3
HTL_stack_sequence：Spiro-MeOTAD
Backcontact_stack_sequence：Au
并将结果进行保存，目前尚未运行保存

main 3 处理：
1.在bin文件夹中新建一个名字叫做"analyses1"后续所有文件都放在这个文件夹中。
2.将simulation_results_summary.csv中的数据读取，然后筛选，如果“Simulation_Voc”这个内容的数据为空，则需要在表格中剔除这个数据，然后新建一个文档命名为"error_data"将Simulation_Voc所对应的Ref_ID记录下来，记录格式为"Ref_ID：xx ，数据错误，移除"所有的错误数据的ID记录在这个文档之中
3.将筛选后所保留下来的数据重新整合成一个文件"simulation_results_TiO2-c"，新的文件中所包含的依旧为原始文件的那几项。

main 4 处理：
处理原始数据，生成学习所需要的x数据，生成了一个analyses_x的表格在bin/analyses1文件夹中
Ref_ID
Cell_area:电池面积【默认0.06】
Layer_1：Substrate
Layer_1_material：FTO
Substrate_thickness：FTO厚度【默认50】
Layer_2：ETL
Layer_2_material：TiO2-c
ETL_thicknes：ETL厚度【默认200】
Layer_3：Perovskite
Layer_3_material：MAPbI3
Perovskite_thickness：Perovskite厚度【默认400】
Layer_4：HTL
Layer_4_material：Spiro-MeOTAD
HTL_thickness：HTL厚度【默认200】
Layer_5：Backcontact
Layer_5_material：Au
Backcontact_thickness：Backcontact厚度【默认100】


main 5 ：
将simulation_results_TiO2-c.csv中的数据进行拆分
1.analyses_y：
    Ref_ID
    Simulation_Voc
    Simulation_Jsc
    Simulation_PCE
    Simulation_FF
2.analyses_y_hat：
    Ref_ID
    JV_default_Voc
    JV_default_Jsc
    JV_default_PCE
    JV_default_FF

> 说明：`analyses_y.csv` 中不再保留 `JV_default_Voc`，避免该字段进入真实值侧造成数据污染。

## 训练/测试集重新规整与切分（随机四分法：3/4 train, 1/4 test）

在 `analyses_x.csv`、`analyses_y.csv`、`analyses_y_hat.csv` 生成后，执行：

```powershell
.\.venv\Scripts\python.exe .\scripts\split_train_test
```

脚本会：
- 自动剔除 `analyses_y.csv` 里的 `JV_default_Voc`（若存在历史残留）。
- 自动从 `train/test_delta_y_predictions.csv` 中识别 `delta_Jsc_true ≈ -4.2` 的异常点并记录其 `Ref_ID`，在切分前整体剔除该 ID 的全部数据。
- 若需手工指定剔除 ID，可在 `bin/excluded_ref_ids.txt` 中每行填写一个 `Ref_ID`（支持 `#` 注释）。
- 按 `Ref_ID` 对齐三份数据后，随机切分为 75% 训练集、25% 测试集（`random_state=42`）。
- 输出到：
  - `bin/train/train_x.csv`
  - `bin/train/train_y.csv`
  - `bin/train/train_y_hat.csv`
  - `bin/test/test_x.csv`
  - `bin/test/test_y.csv`
  - `bin/test/test_y_hat.csv`


main 6 ：
analyses_y：
Simulation_Voc和Simulation_Jsc所对应的点用蓝色的点标出，色值为#3491FA
横坐标写上Voc，纵坐标写上Jsc
图的标题写上"y"生成一个图，命名为"scatter_plot_y"
文字和坐标轴用黑色，底色为白色。

将analyses_y_hat
JV_default_Voc和JV_default_Jsc所对应的点用橙色的点标出，色值为#FFA500
横坐标写上Voc，纵坐标写上Jsc
图的标题写上"y"生成一个图，命名为"scatter_plot_y_hat"
文字和坐标轴用黑色，底色为白色。

原有的点的颜色不变但是将透明度降低为50%
将两组数据中的Ref_ID相同的点，在图中用细黑直线连接
生成一个全新的图，横坐标写上Voc，纵坐标写上Jsc
图的标题写上"y-y_hat"生成一个图
命名为"scatter_plot_y-y_hat"
文字和坐标轴用黑色，底色为白色。


## train 脚本本地运行（Windows）

如果你在 PowerShell 里执行：

`.\.venv\Scripts\Activate.ps1`

出现 `PSSecurityException / 因为在此系统上禁止运行脚本`，这是 **PowerShell 执行策略** 导致，和训练代码本身无关。可以用下面任意一种方式：

### 方式 1（推荐，当前终端临时放开）

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python .\scripts\train
```

> 只对当前 PowerShell 窗口生效，关闭后恢复默认策略。

### 方式 2（不激活 venv，直接调用 venv 的 python）

```powershell
.\.venv\Scripts\python.exe -m pip install -r .\requirements-train.txt
.\.venv\Scripts\python.exe .\scripts\train
```

### 方式 3（用 CMD 激活，不走 PowerShell 策略）

```bat
.\.venv\Scripts\activate.bat
python .\scripts\train
```

## test 集验证脚本（x -> 预测 delta_y，与真实 delta_y 对比）

当 `bin/train/delta_y_model.pkl` 已生成后，可运行：

```powershell
.\.venv\Scripts\python.exe .\scripts\validate
```

脚本会自动读取：
- `bin/test/test_x.csv`
- `bin/test/test_y.csv`
- `bin/test/test_y_hat.csv`

并输出：
- `bin/test/test_delta_y_predictions.csv`（每条样本的 true/pred/error）
- `bin/test/test_delta_y_metrics.csv`（MAE/MSE/RMSE/R2 汇总）

## train 集模拟与真实值对比（复用训练数据做一次验证）

如果希望在训练数据上也跑一次模型预测，并查看与真实值偏差，可运行：

```powershell
.\.venv\Scripts\python.exe .\scripts\validate --dataset train
```

输出文件：
- `bin/train/train_delta_y_predictions.csv`（每条样本的 true/pred/error）
- `bin/train/train_delta_y_metrics.csv`（MAE/MSE/RMSE/R2 汇总）

如果希望**一次性同时跑 train + test，并对比两边指标判断是否可能过拟合**，可运行：

```powershell
.\.venv\Scripts\python.exe .\scripts\validate --dataset both
```

会额外输出：
- `bin/overfit_comparison.csv`（train/test 指标并排 + gap 差值）

## 自动画图脚本（真实 vs 预测散点图 + 误差分布图）

当 `bin/test/test_delta_y_predictions.csv` 已生成后，可运行：

```powershell
.\.venv\Scripts\python.exe -m pip install -r .\requirements-train.txt
.\.venv\Scripts\python.exe .\scripts\plot_validate
```

脚本会按目标自动识别并生成图像到：
- `bin/test/figures/delta_Voc_analysis.png`
- `bin/test/figures/delta_Jsc_analysis.png`
- `bin/test/figures/delta_PCE_analysis.png`
- `bin/test/figures/delta_FF_analysis.png`

若要对 train 集输出回归图（含拟合线）与误差分布图：

```powershell
.\.venv\Scripts\python.exe .\scripts\plot_validate --dataset train
```

图像输出到：
- `bin/train/figures/delta_Voc_analysis.png`
- `bin/train/figures/delta_Jsc_analysis.png`
- `bin/train/figures/delta_PCE_analysis.png`
- `bin/train/figures/delta_FF_analysis.png`

如果希望画出 **train 与 test 在同一张 Actual vs Predicted 散点图上的分散对比**（蓝色圆圈 train，红色方框 test）：

```powershell
.\.venv\Scripts\python.exe .\scripts\plot_validate --dataset both
```

图像输出到：
- `bin/compare/figures/delta_Voc_train_test_compare.png`
- `bin/compare/figures/delta_Jsc_train_test_compare.png`
- `bin/compare/figures/delta_PCE_train_test_compare.png`
- `bin/compare/figures/delta_FF_train_test_compare.png`

> 若运行 `validate` 时提示 `PermissionError: ... denied`，通常是目标 CSV 正在被 Excel 占用。关闭该文件后重试即可；脚本也会自动回退为带时间戳的新文件名保存。
