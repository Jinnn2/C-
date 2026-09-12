# C题：微网与外部电网电力调控

- [建模方案](docs/建模方案.md)
- [时间与计量口径](docs/数据口径.md)
- [数据检查报告](reports/data_quality_report.md)
- [第一问实验报告](reports/q1_experiment.md)
- [第一问结果工作簿](results/result1.xlsx)
- [第二问v1设计](docs/问题二_v1设计.md)
- [第二问v1实验报告](reports/q2_experiment.md)
- [第二问结果工作簿](results/result2.xlsx)
- [第二问V2-A实验报告](reports/q2_v2a_experiment.md)
- [第二问V2-A设计](docs/问题二_v2a设计.md)
- [第二问V2-A结果工作簿](results/result2_v2a.xlsx)
- [第二问V2-B实验报告](reports/q2_v2b_experiment.md)
- [第二问V2-B设计](docs/问题二_v2b设计.md)
- [第二问V2-B结果工作簿](results/result2_v2b.xlsx)
- [第二问V2-C实验报告](reports/q2_v2c_experiment.md)
- [第二问V2-C设计](docs/问题二_v2c设计.md)
- [第二问V2-C结果工作簿](results/result2_v2c.xlsx)
- [第二问V3跨日设计](docs/问题二_v3跨日优化设计.md)
- [第二问V3实验报告](reports/q2_v3_experiment.md)
- [第二问V3主候选结果（R1，48小时）](results/result2_v3_r1.xlsx)
- [第二问V5实验报告（V2-B合同 + V4 delayed_1桥接）](reports/q2_v5_experiment.md)
- [第二问V5混合控制实验报告（主候选）](reports/q2_v5_hybrid_experiment.md)
- [第二问V5混合控制结果工作簿](results/result2_v5_delayed1_hybrid.xlsx)
- 数据库：`data/processed/microgrid.sqlite`
- CSV与源文件清单：`data/processed/`

在本目录运行：

```powershell
python -m pip install -r requirements.txt
python scripts/prepare_data.py
python scripts/solve_q1.py
python scripts/solve_q2.py
python scripts/solve_q2_v2a.py
python scripts/solve_q2_v2b.py
python scripts/solve_q2_v2c.py
python scripts/solve_q2_v3.py
python scripts/solve_q2_v5.py
python scripts/solve_q2_v5_hybrid.py
```

脚本只读取原题和附件，重建本项目的派生数据与检查报告。发生数据结构或数值错误时返回非零退出码；此时不要使用旧的派生数据。脚本校验原始文件执行前后的SHA256，检查通过后才发布新派生数据库。

已完成分析落盘、数据标准化、审计及第一问严格MILP、线性松弛和无储能对照实验。第一问最优购电费35126.948589元，相对无储能节费26.8981%；日初日末6000 kWh。详细状态、残差和输入哈希见 `results/q1/experiment.json`，独立校验代码为 `scripts/validate_q1.py`。结果模板时间标签在输出副本中规范化，原件保留。

第二问v1已完成：1月验证选参、固定点预测日前计划、跨日SOC、334天顺序结算、同预测无储能对照及完整工作簿。正式期总费用16157596.649166元，相对无储能节费16.2690%；334天均有紧急购电，说明此版仅作为后续改进基线。参数和独立审计见 `results/q2/experiment.json`，全年逐段执行见 `results/q2/execution.csv`。工作簿“全天购电量/费”指计划量/计划费；紧急费和实际总费用在报告与daily_summary.csv独立列出。

第二问V2-A已完成分组历史残差修正、17个1月候选比较及334天回测。按1月费用选择了零修正，正式期结果与v1一致，没有新增节费。费用最低的非零候选将1月净负载MAE从182.0874降至179.6665 kW，但费用增加3911.0710元；更低误差不保证更低购电费用。代码为 `scripts/q2_bias.py` 与 `scripts/solve_q2_v2a.py`，独立输出保存在 `results/q2_v2a/`，v1结果未覆盖。

第二问V2-B已完成：在固定预测与固定储能计划下，将历史误差场景的预期紧急费纳入日前目标。1月选择14日窗口，正式期总费用14587351.958055元，比v1/V2-A节省1570244.691110元（9.7183%），紧急费下降72.6405%。富余弃用增加1204304.029272 kWh，268天改善、66天恶化，2月整体略有恶化。代码为 `scripts/q2_risk.py` 与 `scripts/solve_q2_v2b.py`，独立输出在 `results/q2_v2b/`；旧版文件保留。共享场景计划并不等于已实现日内自适应控制。

第二问V2-C已完成固定V2-B全年购电表的逐小时储能重调度。C0不更新场景，8016次重算均保留原最优方案，动作和费用与V2-B完全一致；C1按1月选择γ=1、ρ=0.5，用已结束区间观测更新剩余场景。正式期总费用14558590.547767元，比V2-B再节省28761.410289元（0.1972%）；207天改善、121天恶化，12月整体略有恶化。购电合同逐段完全相同，LP恢复互斥与严格MILP对照、控制时序、物理、费用及导出检查通过。输出在 `results/q2_v2c/`，模型为 `scripts/q2_rolling.py`，实验入口 `scripts/solve_q2_v2c.py`。

第二问V3新增24/48/72小时无每日SOC目标的跨日优化，次日购电使用连续传递的SOC。三组共用无目标的一月预热和成熟72小时误差样本；旧V1/V2中的每日SOC惩罚及硬边界只作为受限模型对照。复现入口为 `scripts/solve_q2_v3.py`，各组全年执行、求解日志、预测来源、期末残值及视野敏感性在 `results/q2_v3/`。日内反馈与次日购电共同重算的R3及区间内即时保护尚未实现；第三四问尚未实现。后续求解必须遵守发布时间约束，不能直接将数据库中的未来实际值作为已知输入。

第二问V5新增两个隔离实验：`scripts/solve_q2_v5.py` 固定V2-B购电合同、替换V4 delayed_1储能控制；`scripts/solve_q2_v5_hybrid.py` 在V2-B低价充电/高价放电参考动作上加入一时段延迟DP安全纠偏，并只用1月选择阈值。主候选正式期总费用14596837.877306元，比V2-B高9485.919251元（0.0650%）；购电合同逐段完全一致，延迟、物理、费用和工作簿回读审计通过。延迟安全裕量的独立候选实验见 `scripts/solve_q2_v5_robust.py`，其1月选择退化为beta=0，因此不作为主结果。
