# 多颗粒轨迹级光学事件检测：方法实现

本仓库仅整理本文方法本身：**LocateAnything 开放词汇检测 → 检测框清洗 → OT–Kalman 身份关联及 Rescue / Gap / Prebirth 恢复 → ROI 特征提取 → GMM–RTV 事件判断**。

运行不需要 GT、人工标签或原实验工程。仓库不包含其他方法对照、模块消融、FPN 训练或推理、人工审阅网站、论文绘图或指标评价程序。原始图像和模型权重需另行提供。

程序输出的是与颗粒轨迹关联的图像可见增亮事件及其起始帧；起始位置采用该帧轨迹 ROI 的图像平面坐标。

## 1. 安装环境

建议使用 Python 3.10。在仓库目录的 CMD 中运行：

```bat
python -m pip install -r requirements-core.txt
```

如果已有可以运行 LocateAnything 的 `x-anylabeling-sam2` 环境，可直接在该环境中运行全流程。检测额外依赖见 `requirements-detector.txt`，需要自行安装与本机 CUDA 匹配的 PyTorch，并准备 LocateAnything-3B 的本地模型快照及其所需文件。

只有已保存检测框的后续流程不需要 PyTorch 或 Transformers。两个阶段也可以使用不同环境：将配置中的 `detector.python` 设置为检测环境的 Python 完整路径，保持 `null` 则使用启动程序的当前 Python。

## 2. 必需输入

### 原始图像

将同一录像的连续、原分辨率图像放在一个目录。支持 8 位 RGB、RGBA 或灰度图像，所有帧尺寸必须一致。RGBA 读取时转换为 RGB，灰度图像复制为三个通道。

示例命名：`MCFF0000_00000000.png`、`MCFF0000_00000001.png`。默认正则 `_(?P<frame>\d+)$` 提取文件名末尾数字作为全局帧号；`frame_offset` 可统一调整。内部帧号从 0 开始，不能使用 ROI 文件夹中的局部排序代替原录像帧号。

同一帧有多个格式时，通过 `image_extensions` 只选择一种，例如 `[".bmp"]`，避免重复帧。默认要求所选区间连续；缺帧会报错。

### 检测输入二选一

- **全流程**：本地 LocateAnything-3B 模型目录；默认使用原有暗颗粒、亮颗粒两组共 12 条提示词，见 `method_core/locate_anything.py` 的 `DEFAULT_TWO_STATE_PROMPTS`。
- **已有检测框续跑**：每张图像对应一个同名 TXT，例如 `MCFF0000_00000000.txt`。每行格式为 `label x1 y1 x2 y2`，坐标为原图像素 XYXY，不是 YOLO 归一化坐标。空文件表示该帧没有检测框，缺少文件则报错。兼容第六列，但不会将其解释为原生检测置信度。

## 3. 配置与启动

所有相对路径均相对于仓库根目录解析。模板中的 `inputs`、`models` 是待填写路径，不含真实数据。建议复制模板为个人配置后修改。

### 从原始图像开始

复制 `configs/method.template.json`，修改 `image_dir`、`recording_id`、`output_dir`、`image_extensions` 和 `detector.model` 后运行：

```bat
python run_method.py --config configs/method.template.json
```

原启动名称仍可使用，功能相同：

```bat
python run_pressure_pipeline.py --config configs/method.template.json
```

### 使用已保存检测 TXT

修改 `configs/method.cached_txt.json` 中 `image_dir`、`detector.txt_dir` 和 `output_dir`，运行：

```bat
python run_method.py --config configs/method.cached_txt.json
```

这里仍从 `detect` 阶段启动：该阶段仅导入已有 TXT，不会调用检测模型；之后自动执行清洗、追迹及事件判断。

### 分段运行与继续

```bat
python run_method.py --config configs/method.template.json --dry-run
python run_method.py --config configs/method.template.json --stop-after detect
python run_method.py --config configs/method.template.json --start-from clean --resume
python run_method.py --config configs/method.template.json --resume
```

四个阶段名称依次是 `detect`、`clean`、`track`、`judge`。`--start-from` 要求前序阶段已经完成并有记录；对于另一处保存的历史 TXT，使用上面的 `cached_txt` 配置即可。

`--dry-run` 校验输入并计算指纹，不加载检测模型。每阶段保存状态和输出校验值；检测阶段还保存逐帧结果，意外中断后 `--resume` 可复用已完成帧。继续时必须保持输入、配置和代码一致；修改参数、图像或代码后应更换 `output_dir`。整理后的代码应使用新的输出目录，不能直接承接旧工程不同源码指纹的运行锁文件。

## 4. 参数与实际流程

模板保留工作版完整配置：OT 关联、三种恢复策略开启、GMM–RTV、首个候选终止策略、GMM 随机种子 2026。`cleaning`、`tracking`、`restoration.prebirth_settings` 和 `gmm_config` 的空项沿用原实现默认值。实际展开参数会写入输出文件，完整默认值也见 `docs/nominal_parameters.json`。

| 配置项 | 作用 |
|---|---|
| `image_dir`、`image_extensions` | 原图位置及选用格式 |
| `frame_regex`、`frame_offset`、`start_frame`、`end_frame` | 原录像帧号提取及处理区间，末帧包含在内 |
| `detector.mode` | `live` 模型检测或 `cached_txt` 导入已有框 |
| `detector.generation`、`detector.nms_iou` | 原检测生成设置及提示词合并后的去重 |
| `cleaning` | 背景、几何、嵌套框、重复框和 NMS 清洗参数；失焦过滤默认关闭 |
| `tracking` | OT 代价、门控、Kalman 和轨迹生命周期参数 |
| `tracker.rescue`、`tracker.restoration` | 三种恢复的工程配置；提供的模板均开启 |
| `restoration` | Gap 最大间隔和 Prebirth 图像回溯参数 |
| `crop_padding` | 轨迹框周边扩展像素，名义值 0 |
| `virtual_tracking_rois` | 默认只保存裁剪坐标，从原图读取 ROI；设为 `false` 则保存裁剪图像 |
| `gmm_config` | ROI 特征、GMM/BIC、几何/初始高亮筛查及 RTV 参数 |

追迹沿用工作版的 `incremental_state_frame_v1`：Kalman 从滤波状态所在帧向下一帧增量推进。跨缺失帧的差分采用 `contiguous_zero_boundary`。GMM 使用整条轨迹的中位数/MAD 归一化和未来观测，因此这是离线分析，`onset_frame` 与确认所需的 `confirmation_frame` 分别保存。

RTV 使用原始 ROI 特征的前后窗口增量，五个通道中任一达到阈值即可通过。这里没有加入 K-means 掩膜、HMM 或其他替代判据。

## 5. 输出文件

| 相对输出目录的路径 | 内容 |
|---|---|
| `frames.csv` | 实际参与运行的原图、全局帧号、尺寸及指纹 |
| `pipeline_config.json`、`pipeline_status.json` | 本次参数/源码指纹和四阶段完成状态 |
| `source_snapshot/` | 本次使用的四个核心源码快照 |
| `01_detection/per_frame/` | 在线检测逐帧 JSON 和 TXT；已有框模式不复制原始 TXT |
| `01_detection/raw_candidates.csv` | 未清洗候选框及提示词来源 |
| `02_cleaning/candidates.csv` | 清洗后保留框 |
| `02_cleaning/cleaning_audit.csv` | 所有框的清洗结果、质量量和剔除原因 |
| `02_cleaning/cleaning_resolved.json` | 实际清洗参数 |
| `03_tracking/track_observations.csv` | 每条轨迹逐帧框、中心、来源及恢复标志 |
| `03_tracking/association_trace.csv` | 已关联检测框、代价及 OT/Rescue 记录 |
| `03_tracking/PRESSURE_TRACKS/tracking_effective_config.json` | 实际关联参数、恢复开关及时间推进策略 |
| `03_tracking/roi_manifest.csv` | ROI 原图路径、裁剪边界及原图位置；是事件判断的输入 |
| `03_tracking/rois/` | 仅关闭虚拟 ROI 时生成的逐轨迹裁剪图像 |
| `tracks.csv` | 兼容旧接口的追迹表，帧号列为 `frame_id` |
| `03_tracking/motchallenge/` | 轨迹格式导出；帧号为内部全局帧号 +1，不计算 MOT 指标 |
| `04_judgment/events.csv` | **全部轨迹的最终判断结果，包括正例和负例** |
| `04_judgment/GMM_RTV/predictions.csv` | 同一方法的逐轨迹预测表 |
| `04_judgment/GMM_RTV/candidate_audit.json` | 候选起始帧、确认帧、各 RTV 通道增量和通过情况 |
| `04_judgment/GMM_RTV/diagnostics.json` | 几何、初始高亮、状态分配及未通过原因 |
| `04_judgment/gmm_effective_config.json`、`feature_policy.json` | 实际 GMM 参数及特征策略 |
| `runtime_summary.csv` | 阶段及子步骤计时；子步骤与总阶段计时重叠，不应直接相加 |

在 `events.csv` 中筛选 `has_ignition=True` 得到被判断为事件的序列。主要字段为：

- `recording_id`、`track_id`：本次录像和追迹身份。
- `onset_frame`：估计事件起始的原录像全局帧。
- `confirmation_frame`：连续高态窗口的确认帧。
- `onset_cx/cy`、`onset_x1/y1/x2/y2`：该帧轨迹 ROI 在原图的中心和框；恢复位置是轨迹支持位置，不是独立颗粒轮廓标注。
- `status`：如 `accepted`、`too_short`、`geometry_rejected`、`initial_hot_rejected`、`rtv_rejected` 或 `no_event`。

观测来源区分 `detector_primary`、`detector_rescue`、`gap_roi` 和 `prebirth_roi`。LocateAnything 该接口没有原生检测置信度，`native_score` 留空，清洗质量分单独保存。关联日志中的 OT 数值也不应解释为检测置信度。

## 6. 代码位置与验证

| 文件 | 职责 |
|---|---|
| `run_method.py` | 统一入口 |
| `ignition_revision/pressure_pipeline.py` | 四阶段串联、逐帧检测及断点继续 |
| `method_core/locate_anything.py` | LocateAnything 调用、提示词和结果解析 |
| `method_core/postprocess_detection_txt_modified.py` | 原检测框清洗规则 |
| `method_core/ot_kalman_tracker.py` | OT、Kalman、Rescue、Gap、Prebirth 核心计算 |
| `ignition_revision/tracking.py` | 帧序列推进、来源日志及 ROI 整理 |
| `method_core/ignition_detection_gmm_rtv_new.py` | 原 ROI 特征、GMM 配置及状态处理函数 |
| `ignition_revision/judges.py` | 特征差分、候选转移、预检查及原始特征 RTV |
| `ignition_revision/prediction.py` | 逐轨迹判决与结果保存 |

```bat
python -m unittest discover -s tests -v
```

测试使用自动生成的小型输入，仅验证程序实现，不是论文实验数据。已检查清洗、追迹/Gap、特征和 GMM–RTV 与工作版的一致性，以及完整已有框流程、空检测、重复帧、位置输出、断点继续和校验拒绝。GPU 检测模型未在本次整理时重新运行；原模型调用定义与提示词被保留。见 `VERIFICATION.json` 和 `SOURCE_MANIFEST.json`。

公开许可状态见 `NOTICE.md`。
