# HDF5 to LeRobot Converter

这个文档说明如何使用 [convert_hdf5_to_lerobot.py](/workspace/RLinf/denserwd/scripts/convert_hdf5_to_lerobot.py)，把
`/workspace/RLinf/denserwd/data/datasets` 下的 LIBERO HDF5 数据转换成一个统一的 LeRobot 格式数据集。

脚本特点：

- 只读取原始 HDF5，不会修改源数据。
- 默认会把 4 个 LIBERO 目录合并成 1 个 LeRobot 数据集。
- 支持只转换部分目录、部分 episode。
- 支持多进程并行，加速大规模转换。
- 输出标准 LeRobot 风格目录：`meta/` + `data/chunk-*/episode_*.parquet`。

默认会优先合并下面四个目录：

- `libero_spatial_collect_openpi`
- `libero_object_collect_openpi`
- `libero_goal_collect_openpi`
- `libero_10_collect_openpi`

## 1. 最常用用法

全量转换四个 LIBERO 目录到一个 LeRobot 数据集：

```bash
python scripts/convert_hdf5_to_lerobot.py \
  --output-dir /workspace/RLinf/denserwd/data/lerobot_libero \
  --overwrite
```

先只看计划，不真正写文件：

```bash
python scripts/convert_hdf5_to_lerobot.py --dry-run
```

用 4 个 worker 并行加速：

```bash
python scripts/convert_hdf5_to_lerobot.py \
  --output-dir /workspace/RLinf/denserwd/data/lerobot_libero \
  --num-workers 4 \
  --overwrite
```

## 2. 输出目录结构

转换完成后，输出目录通常长这样：

```text
<output-dir>/
├── meta/
│   ├── info.json
│   ├── tasks.jsonl
│   ├── episodes.jsonl
│   └── stats.json
└── data/
    ├── chunk-000/
    │   ├── episode_000000.parquet
    │   ├── episode_000001.parquet
    │   └── ...
    ├── chunk-001/
    └── ...
```

其中：

- `meta/info.json` 是整个数据集的总信息。
- `meta/tasks.jsonl` 是任务级别元数据。
- `meta/episodes.jsonl` 是每条 episode 的元数据。
- `meta/stats.json` 是数值特征统计量。
- `data/chunk-*/episode_*.parquet` 是真正的 LeRobot episode 数据。

## 3. 参数说明

### `--source-dir`

源数据根目录。脚本会在这个目录下面找 HDF5 子目录。

默认值：

```text
/workspace/RLinf/denserwd/data/datasets
```

什么时候改：

- 你的 HDF5 不在默认目录时再改。
- 如果你已经把数据软链到别的位置，也可以指向新的根目录。

示例：

```bash
python scripts/convert_hdf5_to_lerobot.py \
  --source-dir /mnt/my_data/datasets \
  --output-dir /mnt/my_data/lerobot_libero \
  --overwrite
```

### `--output-dir`

转换后 LeRobot 数据集的输出目录。

默认值：

```text
/workspace/RLinf/denserwd/data/lerobot_libero
```

什么时候改：

- 想把转换结果放到更大的磁盘。
- 想区分不同版本的数据集。
- 想先做一个小规模 smoke test。

推荐：

- 正式全量转换时，给一个明确的目录名。
- 调试时可写到 `/tmp/...`。

### `--dataset-name`

写入 `meta/info.json` 的数据集名字，不影响 parquet 内容，只影响元数据标识。

默认值：

```text
libero_combined_openpi
```

什么时候改：

- 你想区分不同转换版本。
- 你只转换某个子集，希望元数据里名字更清晰。

示例：

```bash
python scripts/convert_hdf5_to_lerobot.py \
  --dataset-name libero_10_only_v1 \
  --task-prefix libero_10 \
  --output-dir /workspace/RLinf/denserwd/data/lerobot_libero_10 \
  --overwrite
```

### `--robot-type`

写入 `meta/info.json` 的机器人类型字段。

默认值：

```text
panda
```

什么时候改：

- 只有当你的数据不是 Panda 或者你想严格记录来源时才改。
- 对当前 LIBERO 数据，通常保持默认即可。

### `--fps`

写入元数据和 `timestamp` 计算的帧率。

默认值：

```text
10
```

作用：

- `timestamp = frame_index / fps`
- 影响 `meta/info.json` 里的 `fps`

什么时候改：

- 只有当你确认原始数据不是 10 FPS 时才改。
- 对当前这批 LIBERO 数据，建议保持 `10`。

### `--chunk-size`

控制每个 chunk 目录下最多放多少个 episode parquet。

默认值：

```text
1000
```

作用：

- `episode_000000.parquet` 到 `episode_000999.parquet` 会放在 `chunk-000/`
- `episode_001000.parquet` 起会进入 `chunk-001/`

什么时候改：

- 一般不用改，`1000` 已经比较合理。
- 如果你更喜欢更小目录，可以改成 `200` 或 `500`。
- 如果你更关心目录数量少，也可以调大，比如 `2000`。

推荐：

- 正常训练和读取：`1000`
- 调试小集：无所谓，默认就够用

### `--num-workers`

并行 worker 数量。脚本会按 HDF5 文件粒度并行转换。

默认值：

- 自动取一个保守值，范围通常在 `1~4`

作用：

- `1` 表示单进程顺序转换
- `>1` 表示多进程并行
- 当任务太小、实际上只会处理一个 HDF5 文件时，脚本会自动退回 `1`

怎么设置：

- 小规模测试：`1`
- 正常全量转换：`2` 到 `4`
- 如果机器 CPU 核很多、磁盘吞吐也足够，可以试 `4`
- 如果磁盘较慢，worker 太多不一定更快

推荐：

- 先试 `--num-workers 2`
- 如果 CPU 和磁盘都比较充足，再试 `--num-workers 4`

示例：

```bash
python scripts/convert_hdf5_to_lerobot.py \
  --output-dir /workspace/RLinf/denserwd/data/lerobot_libero \
  --num-workers 2 \
  --overwrite
```

### `--source-subdirs`

显式指定要合并的源子目录。多个目录之间用空格分开。

默认行为：

- 如果不传，脚本会优先使用四个标准 LIBERO 目录。

什么时候用：

- 你只想转换某几个目录。
- 你想精确控制合并范围，而不是使用默认四目录。

示例：

```bash
python scripts/convert_hdf5_to_lerobot.py \
  --source-subdirs libero_spatial_collect_openpi libero_goal_collect_openpi \
  --output-dir /workspace/RLinf/denserwd/data/lerobot_libero_spatial_goal \
  --overwrite
```

### `--task-prefix`

按子目录名前缀过滤。

示例：

- `libero_10`
- `libero_goal`

什么时候用：

- 你只想转某一类目录，并且目录名有统一前缀。

示例：

```bash
python scripts/convert_hdf5_to_lerobot.py \
  --task-prefix libero_10 \
  --output-dir /workspace/RLinf/denserwd/data/lerobot_libero_10 \
  --overwrite
```

注意：

- `--source-subdirs` 的优先级更高。
- 如果你已经显式给了 `--source-subdirs`，通常就不需要再配 `--task-prefix`。

### `--max-episodes`

最多只转换多少条 episode。这个参数主要用于调试和 smoke test。

默认值：

- 不限制

什么时候用：

- 先做小样本验证
- 想快速检查输出格式
- 想测试并行配置是否正常

示例：

```bash
python scripts/convert_hdf5_to_lerobot.py \
  --max-episodes 10 \
  --output-dir /tmp/lerobot_debug \
  --overwrite
```

推荐：

- 快速检查：`2` 或 `10`
- 小规模试跑：`50` 或 `100`
- 正式转换：不要设置这个参数

### `--overwrite`

允许写入一个已经存在且非空的输出目录。

默认行为：

- 如果输出目录已存在且非空，脚本会报错，避免误覆盖。

什么时候用：

- 你确认要复用同一个输出目录重新生成数据时。

注意：

- 建议只在你明确知道输出目录内容可以被覆盖时使用。
- 更稳的方式是每次换一个新的输出目录名。

### `--dry-run`

只打印转换计划，不真正写任何文件。

作用：

- 查看会选中哪些子目录
- 查看任务数、episode 数、frame 数
- 查看 worker 配置是否符合预期

示例：

```bash
python scripts/convert_hdf5_to_lerobot.py --dry-run
```

## 4. 推荐配置

### 场景一：先做最小验证

推荐命令：

```bash
python scripts/convert_hdf5_to_lerobot.py \
  --max-episodes 2 \
  --output-dir /tmp/lerobot_smoketest \
  --num-workers 1 \
  --overwrite
```

适合：

- 第一次跑脚本
- 检查输出格式
- 确认磁盘路径没问题

### 场景二：全量正式转换

推荐命令：

```bash
python scripts/convert_hdf5_to_lerobot.py \
  --output-dir /workspace/RLinf/denserwd/data/lerobot_libero \
  --num-workers 2 \
  --overwrite
```

适合：

- 在普通工作站上稳定跑完整转换
- 想在速度和稳定性之间取平衡

### 场景三：机器较强，尽量提速

推荐命令：

```bash
python scripts/convert_hdf5_to_lerobot.py \
  --output-dir /workspace/RLinf/denserwd/data/lerobot_libero \
  --num-workers 4 \
  --overwrite
```

适合：

- CPU 核数较多
- 磁盘吞吐较好
- 想缩短整体转换时间

如果发现提速不明显，优先把 `--num-workers` 降回 `2`。

### 场景四：只转换某个 LIBERO 子集

只转 `libero_10`：

```bash
python scripts/convert_hdf5_to_lerobot.py \
  --task-prefix libero_10 \
  --dataset-name libero_10_openpi \
  --output-dir /workspace/RLinf/denserwd/data/lerobot_libero_10 \
  --overwrite
```

只转指定两个目录：

```bash
python scripts/convert_hdf5_to_lerobot.py \
  --source-subdirs libero_goal_collect_openpi libero_10_collect_openpi \
  --dataset-name libero_goal_and_10 \
  --output-dir /workspace/RLinf/denserwd/data/lerobot_goal_10 \
  --overwrite
```

## 5. 一个完整示例

建议第一次这样做：

1. 先 dry-run 看计划。

```bash
python scripts/convert_hdf5_to_lerobot.py --dry-run
```

2. 再做一个小样本 smoke test。

```bash
python scripts/convert_hdf5_to_lerobot.py \
  --max-episodes 5 \
  --output-dir /tmp/lerobot_smoketest \
  --num-workers 1 \
  --overwrite
```

3. 确认没问题后跑全量。

```bash
python scripts/convert_hdf5_to_lerobot.py \
  --output-dir /workspace/RLinf/denserwd/data/lerobot_libero \
  --num-workers 2 \
  --overwrite
```

## 6. 常见问题

### 输出目录已存在，脚本报错怎么办？

原因：

- 脚本默认防止覆盖已有结果。

处理：

- 换一个新的 `--output-dir`
- 或者确认可以覆盖后，加 `--overwrite`

### `--num-workers` 越大越快吗？

不一定。

影响因素：

- CPU 核数
- 磁盘读写速度
- parquet 编码开销
- 同时处理的 HDF5 文件数量

经验建议：

- 先试 `2`
- 再试 `4`
- 哪个更快就用哪个

### 为什么我设了 `--num-workers 4`，日志里却显示 `Worker processes : 1`？

这是正常的。

原因：

- 这次转换实际只涉及 1 个 HDF5 文件
- 或者 `--max-episodes` 太小，任务只落在 1 个文件里
- 脚本会自动退回单 worker，避免并行开销

### 源 HDF5 会被修改吗？

不会。

脚本只会：

- 读取源 HDF5
- 在 `--output-dir` 下写新的 LeRobot 数据集

## 7. 当前脚本对应的关键默认值

当前版本里，最关键的默认值是：

- `source_dir`: `/workspace/RLinf/denserwd/data/datasets`
- `output_dir`: `/workspace/RLinf/denserwd/data/lerobot_libero`
- `dataset_name`: `libero_combined_openpi`
- `robot_type`: `panda`
- `fps`: `10`
- `chunk_size`: `1000`
- `num_workers`: 自动选择，通常为 `1~4`
- `source_subdirs`: 默认四个 LIBERO 目录

如果你只是要把当前仓库里的四个 LIBERO 目录转成一个 LeRobot 数据集，最推荐直接用：

```bash
python scripts/convert_hdf5_to_lerobot.py \
  --output-dir /workspace/RLinf/denserwd/data/lerobot_libero \
  --num-workers 2 \
  --overwrite
```
