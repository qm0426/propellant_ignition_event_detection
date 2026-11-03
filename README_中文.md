# 代码提交包使用说明

这是独立的审稿代码快照，不会修改正在处理新压力数据的原目录。README.md给出英文说明；docs目录说明输入、输出、版本边界和限制。原始数据、标注、网络权重和全部大结果不随代码公开。科学核心源文件保持原样；历史脚本中的本机默认路径改为占位路径，差异和哈希记录在SOURCE_MANIFEST.json。

## 一般运行顺序

1. 在独立环境安装requirements-core.txt；VLM检测和FPN按各自环境说明安装。不要把两套不同torch版本强行装在同一个环境。
2. 复制revision_configs下相应template.json为自己的配置。填写真实输入位置、标注来源、参考队列和实验协议；模板中的draft、null和占位路径不会被当作已完成证据。
3. 从仓库根目录执行命令。原始方法的新录像流程使用run_pressure_pipeline.py，先dry-run，再正式运行。仅预测不需要GT；评价需要新录像自身的检测/身份/事件参考。
4. 对比和消融通过python -m ignition_revision选择detection、assignment、restoration、judges或ablations。HMM通过run_hmm_comparison.py单独运行，不会加入原始方法的新压力流水线。
5. 多组联合运行：修改suite.template.json为suite.my.json，每项config指向自己的已填写配置。enabled控制该项开关；--only选择要运行的项，--disable排除项。每组中的methods/variants可以只放一个方法，因此每个消融也可以独立运行。

```powershell
python run_submission_suite.py --suite revision_configs/suite.my.json --dry-run
python run_submission_suite.py --suite revision_configs/suite.my.json --only assignment restoration
python run_submission_suite.py --suite revision_configs/suite.my.json --disable detection --resume
python run_submission_suite.py --suite revision_configs/suite.my.json
```

训练和推理保持独立，不会被上面的全部运行误触发。先使用build_ignition_dataset.py构建FPN数据，明确训练/验证/测试划分、忽略未知区域及跨区间轨迹处理，再运行train_fpn_revision.py和infer_fpn_revision.py。build_comparison_dataset.py用于另一个对比数据接口。具体参数以各入口--help为准。

## 绘图与旧接口

```powershell
python plot_revision_results.py --runs outputs/submission_suite --output outputs/figures --formats png,pdf
```

新最小联合入口的逐家族结果可直接交给plot_revision_results.py。plot_review_v2_results.py读取旧review-v2注册表和协议文件，不能把新的suite_status.json冒充其旧输入。旧12个最终脚本保留在latest_experiment_scripts_20261018中，由兼容层加载；legacy_inputs.py支持旧数据适配，但数据移动后仍需更新CSV内部图像路径。

## 正式复现还需提供什么

- 原始全分辨率帧、正确的全局物理帧编号；新压力流程还需本地LocateAnything模型目录及其环境。
- 缓存检测池、原tracks.csv及ROI映射；固定ROI评价需rois.csv和完整正负labels.csv。
- MOT评价需人工一致身份的reference_boxes.csv；事件匹配需reference_events.csv。仅有点火帧不能代替逐帧身份真值。
- FPN数据划分/监督掩码和训练权重；未知框不能直接当背景。
- 正式运行对应的完整参数、输入来源和协议记录。没有额外数据时，随包小样例只能验证软件，不能复现论文全量指标。

SOURCE_MANIFEST.json记录来源与版本。NOTICE.md保留第三方许可；本包没有代作者选择新的开源许可，也没有发布到远程仓库。完成作者信息、许可和数据共享安排后再公开托管。

## 已整理的完整审稿方案

revision_configs/review_suite.template.json按现有注册表生成了判据、决策/特征、同成本/原生追迹、16组恢复因子、对应DoG判据、参数敏感性，以及逐项检测/清洗方案。先复制这些模板并填写自己的输入；仍需核实协议和参数，不能把模板当作已冻结的真实实验。下面的--only名称对应review_suite中的方案ID：

```powershell
python run_submission_suite.py --suite revision_configs/review_suite.my.json --only criteria decision_features
python run_submission_suite.py --suite revision_configs/review_suite.my.json --only assignment_native factorial factorial_dog
python run_submission_suite.py --suite revision_configs/review_suite.my.json
```

每个methods/variants项可拆成独立配置。generate_review_templates.py可重新生成模板，但会覆盖这些template文件，不应覆盖自己的my配置。完整方案不包含训练、推理、人工审核、新工况评价或需要模型重跑的提示词任务；这些保持独立入口。全量真实数据不会因代码整理而自动获得。
