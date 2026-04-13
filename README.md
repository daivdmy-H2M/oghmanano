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
    JV_default_Voc 
    Simulation_PCE
    Simulation_FF
2.analyses_y_hat：
    Ref_ID
    JV_default_Voc
    JV_default_Jsc
    JV_default_PCE
    JV_default_FF


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