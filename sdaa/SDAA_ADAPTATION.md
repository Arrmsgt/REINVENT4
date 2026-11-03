# REINVENT4 SDAA 适配记录

## 一、环境信息

| 项目 | 内容 |
|---|---|
| 机器 | 10.71.13.47 (node175) |
| 架构 | 龙芯 loongarch64 (Loongson-3C6000/D, 64 核) |
| 加速卡 | 32 × SDAA |
| 容器 | tuyi_test |
| Python | 3.12.13 (`/home/py312`) |
| PyTorch | 2.12.0 + torch_sdaa 3.3.0 |
| SDAA 软件栈 | 3.3.0 (TecoDNN/TecoBLAS/TecoRAND/TCCL) |
| REINVENT4 | 4.8.24 (`/data01/tuyilist/REINVENT4`) |
| RDKit | 2026.09.1（**本机源码重编**，见第五节） |
| NumPy | 2.5.3 |

CUDA 对照环境：10.10.6.21，A100-PCIE-40GB，Python 3.12.13，
PyTorch 2.8.0+cu128，RDKit 2026.09.1（官方 wheel），NumPy 2.5.3。

## 二、分析结论

REINVENT4 是分子生成/强化学习框架，模型主体为 RNN（GRU）和
Transformer 两类，**不含任何自定义 CUDA 算子或 Triton kernel**，
全部由标准 `torch.nn` 模块构成。CUDA 相关调用仅 5 处，且都不是
推理必需路径：

| 位置 | 原代码 | 性质 | 修改 |
|---|---|---|---|
| `runmodes/TL/learning.py:194` | `if self.device == "cuda": torch.cuda.mem_get_info()` | 显存日志 | sdaa 分支 |
| `utils/helpers.py:37` | `torch.cuda.manual_seed(seed)` | 设种子（无 CUDA 时是 no-op） | 追加 sdaa |
| `Reinvent.py:108` | `torch.cuda.device_count()` | 启动日志 | 追加 sdaa 分支 |
| `runmodes/TL/run_transfer_learning.py:114` | `capturable=device.startswith("cuda")` | AdamW 参数（sdaa 走 False 安全） | 不改 |
| `utils/hw_report.py:65` | `if torch.cuda.is_available()` | 硬件报告 | 不改 |

程序本身通过 `torch.device(...)` / `--device` 参数选择设备，
设备无关性良好，`torch.device("sdaa:0")` 可直接使用。

## 三、功能适配改动

共 3 个代码文件、3 个配置文件：

### 3.1 `reinvent/runmodes/TL/learning.py`

显存报告改为设备无关，同时支持 CUDA 与 SDAA：

```python
# NOTE (SDAA adaptation): report device memory for both CUDA and SDAA
dev_type = getattr(self.device, "type", str(self.device)).split(":")[0]
mem_info = None
if dev_type == "cuda":
    mem_info = torch.cuda.mem_get_info()
elif dev_type == "sdaa":
    mem_info = torch.sdaa.mem_get_info()
if mem_info is not None:
    free_memory, total_memory = mem_info
    used_memory = total_memory - free_memory
    logger.info(f"{used_memory // 1024 ** 2} MiB device memory used in epoch {epoch_no}")
```

### 3.2 `reinvent/utils/helpers.py`

`set_seed()` 同时设置 SDAA 的随机种子：

```python
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
if getattr(torch, "sdaa", None) is not None and torch.sdaa.is_available():
    torch.sdaa.manual_seed(seed)
torch.backends.cudnn.deterministic = True
```

### 3.3 `reinvent/Reinvent.py`

启动日志打印 SDAA 设备数：

```python
if torch.cuda.is_available():
    logger.info(f"Number of PyTorch CUDA devices {torch.cuda.device_count()}")

if getattr(torch, "sdaa", None) is not None and torch.sdaa.is_available():
    logger.info(f"Number of PyTorch SDAA devices {torch.sdaa.device_count()}")
```

### 3.4 配置文件

`configs/sampling.toml`、`configs/staged_learning.toml`、
`configs/transfer_learning.toml` 中 `device = "cuda:0"` → `device = "sdaa:0"`。

三处改动均为「sdaa 优先 + cuda 兜底」的双兼容写法，
原始 CUDA 行为不受影响。

## 四、依赖

### 4.1 安装

```bash
cd /data01/tuyilist/REINVENT4
pip install -e . --no-deps      # torch 用环境自带
```

`pyproject.toml` 要求 `torch==2.12.0`，与本环境一致；
CUDA 对照机（torch 2.8.0）用 `--no-deps` 安装也可正常运行。

### 4.2 完整依赖清单

见同目录 `requirements_sdaa.txt`。要点：

- `numpy` 必须 2.x（REINVENT 4.8 要求 `>=2,<3`），本机 2.5.3
- `polars < 2`：2.x 与 REINVENT 的数据管线不兼容，需降级
- `xxhash >= 3,<4`：4.x 会触发 REINVENT 版本冲突
- `mmpdb >= 2.1,<3`：**3.x 删除了 `mmpdblib.do_fragment` 模块**，
  会导致 `models/transformer/mol2mol/dataset/preprocessing/mmp.py` 导入失败
- `rdkit >= 2025.09.1`：本机为源码重编版本（见第五节）
- `torchvision`：`runmodes/utils/image.py` 隐式依赖，需与 torch 版本匹配
- `scipy`：`runmodes/utils/plot.py` 隐式依赖（`scipy.stats.gaussian_kde`）
- `python-dotenv`：`Reinvent.py` 启动即导入

### 4.3 权重

```bash
# Zenodo (DOI 10.5281/zenodo.15641296)
# 注意：官方文件名是 reinvent_pubchem.prior，不是 reinvent.prior
cd priors
curl -L "https://zenodo.org/api/records/20701824/files/reinvent_pubchem.prior/content" \
     -o reinvent_pubchem.prior
```

可用文件：`reinvent_pubchem.prior` (23.5MB)、`libinvent.prior` (94.8MB)、
`linkinvent.prior` (94.8MB)、`pepinvent.prior` (79.9MB)，另有 Transformer
版本与 ECFP4 词表。

> 实测 Zenodo 对国内限速约 32 KB/s 且**不支持断点续传**
> （响应无 `accept-ranges`），23.5MB 约 12 分钟，建议一次下完。

## 五、RDKit 源码重编（本次适配的主要工作量）

REINVENT 要求 `rdkit >= 2025.09.1` 且 `numpy >= 2`，而本机原有 RDKit
是较早期的自包含 whl，与新 numpy 及 InChI 功能均不匹配，故从源码重编。

### 5.1 为什么不能用官方 wheel

loongarch64 无官方 wheel，必须源码编译。编译链路：

1. **boost.python312**：系统 boost 1.81 只带 python311 组件，
   Python 3.12 需源码编译 `libboost_python312.so` + `libboost_numpy312.so`
2. **boost.numpy 与 NumPy 2 不兼容**：`dtype.cpp` 用
   `PyArray_Descr::elsize`，NumPy 2 中该字段已移除，需改用
   `PyDataType_ELSIZE()` 宏：

   ```cpp
   // libs/python/src/numpy/dtype.cpp
   int dtype::get_itemsize() const {
       return PyDataType_ELSIZE(reinterpret_cast<PyArray_Descr*>(ptr()));
   }
   ```

3. **InChI IXA 层缺失**：`External/INCHI-API` 内置的 libinchi 源码
   （来自 openbabel 的裁剪副本）缺 IXA 层（`ixa_*.c`），
   而 RDKit 的 `inchi.cpp` 引用了 `MakeINCHIFromMolfileText`，
   链接期 `libRDKitRDInchiLib.so` 出现 undefined symbol。
   由于该入口只用于 **MolBlock → InChI**，SMILES ↔ InChI/InChIKey
   走 `GetINCHI` / `GetINCHIKeyFromINCHI` 不受影响，
   故补一个 stub 使链接通过、调用时安全失败：

   ```c
   /* External/INCHI-API/src/INCHI_BASE/src/ixa_stub.c */
   EXPIMP_TEMPLATE INCHI_API int INCHI_DECL
   MakeINCHIFromMolfileText(const char *moltext, char *options, inchi_Output *result)
   {
       (void)moltext; (void)options;
       if (result) memset(result, 0, sizeof(inchi_Output));
       return inchi_Ret_ERROR;
   }
   ```

### 5.2 打包自包含 whl

- C++ 库（219 个）拷入 `rdkit/.libs/`
- Python 扩展 patchelf 设置 RPATH
- **RPATH 层级要按「文件相对 rdkit 根的深度」算**：
  `rdkit/rdBase.so` → `$ORIGIN/.libs`；
  `rdkit/Chem/rdchem.so` → `$ORIGIN/../.libs`

  > 踩坑：最初按「斜杠数」算深度，顶层模块少算一层，
  > 装完报 `libRDKitRDBoost.so.1: cannot open shared object file`。
  > 另注意 `printf '/..%.0s' $(seq 1 0)` 在无参数时仍会输出一次 `/..`，
  > 深度为 0 必须单独处理。

### 5.3 验证结果

```
rdkit: 2026.09.1pre | numpy: 2.5.3
INCHI_AVAILABLE: True
SMILES->InChIKey: LFQSCWFLJHTTHZ-UHFFFAOYSA-N   (乙醇，正确)
3D embed: 0        (rdDistGeom，NumPy 2 兼容)
Cairo PNG: 3216 bytes
MolWt: 46.069
ldd 无缺失依赖
```

同时回归验证：`torch_scatter` on sdaa 正常、PSICHIC 相关 PyG 组件正常。

## 六、运行方式

### 6.1 采样

```bash
source /opt/tecoai/setvars.sh
source /home/py312/bin/activate
cd /data01/tuyilist/REINVENT4

cat > configs/sampling_sdaa.toml <<'EOF'
run_type = "sampling"
device = "sdaa:0"
seed = 42

[parameters]
model_file = "priors/reinvent_pubchem.prior"
output_file = "sampling_sdaa.csv"
num_smiles = 50
unique_molecules = true
randomize_smiles = true
EOF

reinvent -l run.log configs/sampling_sdaa.toml
```

实测日志：

```
Number of PyTorch SDAA devices 32
Using GPU device:0 /dev/tcaicard0
Writing sampled SMILES to CSV file sampling_sdaa.csv
```

### 6.2 模型打分（NLL）

```bash
python nll_compare.py sdaa:0 out.json priors/reinvent_pubchem.prior smiles.txt
```

`nll_compare.py` 通过 `reinvent.runmodes.create_adapter()` 加载 prior，
调用 `adapter.likelihood_smiles(List[str])` 得到每个分子的 NLL。

### 6.3 强化学习训练

```bash
cat > /tmp/rl.toml <<'EOF'
run_type = "staged_learning"
device = "sdaa:0"
tb_logdir = "/tmp/tb_rl"

[parameters]
summary_csv_prefix = "/tmp/rl"
prior_file = "priors/reinvent_pubchem.prior"
agent_file = "priors/reinvent_pubchem.prior"
batch_size = 32
randomize_smiles = true

[learning_strategy]
type = "dap"
sigma = 128
rate = 0.0001

[[stage]]
chkpt_file = '/tmp/rl.chkpt'
termination = "simple"
max_score = 0.99
min_steps = 10
max_steps = 10

[stage.scoring]
type = "geometric_mean"

[[stage.scoring.component]]
[stage.scoring.component.QED]
[[stage.scoring.component.QED.endpoint]]
name = "QED"
weight = 1.0
EOF

reinvent -l /tmp/rl.log /tmp/rl.toml
```

实测输出：

```
Score: 0.67 Agent NLL: 30.57 Valid:  97% Step: 10
```

> 注意：`parameters` 段用 pydantic 严格模式校验，
> 原示例 config 里的 `unique_sequences` / `tb_isim` 在当前版本会报
> `Extra inputs are not permitted`，需删掉。

## 七、精度对比

固定 100 个 SMILES（两边采样结果并集 + 常见药物分子），同一 prior、
同一 `set_seed(42)`，比较 `likelihood_smiles` 输出的 NLL。

三方（+1 对照）同时测：

- **CPU**：纯 fp32，作为浮点参考实现
- **SDAA**：TecoDNN/TecoBLAS，fp32
- **CUDA 默认**：torch 2.8 默认配置（`cudnn.allow_tf32 = True`）
- **CUDA 关 TF32**：显式 `cudnn.allow_tf32 = False`

### 7.1 以 CPU 为基准

| 对比 | 最大绝对差 | 平均绝对差 | 最大相对差 | RMSE | Pearson r |
|---|---|---|---|---|---|
| **SDAA vs CPU** | 0.0000229 | 0.0000060 | 0.000062% | 0.0000079 | **1.0000000000** |
| CUDA(默认 TF32) vs CPU | 0.0105934 | 0.0022890 | 0.019869% | 0.0030555 | 0.9999999792 |
| **CUDA(关 TF32) vs CPU** | 0.0000381 | 0.0000089 | 0.000115% | 0.0000120 | **1.0000000000** |

### 7.2 SDAA vs CUDA（同为 fp32）

| 对比 | 最大绝对差 | 平均绝对差 | 最大相对差 | RMSE | Pearson r |
|---|---|---|---|---|---|
| SDAA vs CUDA(fp32) | 0.0000305 | 0.0000088 | 0.000107% | 0.0000114 | **1.0000000000** |

### 7.3 结论

1. **SDAA 精度正确**。与 CPU fp32 参考的差异在 2e-5 量级（纯浮点舍入），
   Pearson r = 1.0000000000。
2. **CUDA 默认配置反而有偏差**：`torch.backends.cudnn.allow_tf32` 默认为
   `True`，REINVENT 的 GRU 在 CUDA 上走 cuDNN，被 TF32（10 位尾数）
   降精度，导致与 CPU 偏差达 0.011（NLL 量级 36）。
3. 关闭 TF32 后，CUDA 与 CPU、SDAA 三方一致，差异均在 1e-5 量级。

> 这意味着：若以「CUDA 默认配置」为基准去评判 SDAA，会误判 SDAA 有偏差；
> 实际情况正好相反 —— SDAA 的 fp32 结果更接近真实参考值。
> 做跨平台精度对标时，**对方平台也应关闭 TF32**，否则基准本身不准。

### 7.4 采样结果不可跨设备直接对比

`srun_type="sampling"` 在固定 seed 下跨设备**不可复现**：SDAA 用 TecoRAND、
CUDA 用 cuRAND，两者 RNG offset 推进方式不同（与 BoltzGen 适配中的发现一致），
同 seed 42 生成的 47/48 个分子交集为 0。这是框架层 RNG 兼容性问题，
不是算子精度问题。因此精度对比必须用**固定输入打分**，不能用采样结果对比。

同理，8.4 节 RL 训练中 0.67 vs 0.71 的分数差异也源于采样 RNG 不同，
两侧都正常完成了优化过程。

## 八、性能对比

环境：SDAA（10.71.13.47，32 卡）vs A100-PCIE-40GB（10.10.6.21）。
同一 prior、同一输入，含 warmup，取 3 次最优并显式 `synchronize()`。

### 8.1 模型加载

| 项目 | SDAA | A100 | 倍差 |
|---|---|---|---|
| 加载耗时（3 次均值） | 0.396 s | 0.191 s | 2.1× |

### 8.2 NLL 打分（`likelihood_smiles`）

| batch | SDAA | A100 | 倍差 | SDAA 吞吐 | A100 吞吐 |
|---|---|---|---|---|---|
| 50 | 0.532 s | 0.005 s | 97× | 94 mol/s | 9122 mol/s |
| 100 | 1.545 s | 0.009 s | 165× | 65 mol/s | 10655 mol/s |
| 300 | 4.042 s | 0.021 s | 192× | 74 mol/s | 14570 mol/s |
| 500 | 6.702 s | 0.027 s | 248× | 75 mol/s | 18288 mol/s |
| 1000 | 14.723 s | 0.052 s | 282× | 68 mol/s | 19141 mol/s |

关键观察：A100 吞吐随 batch 明显增长（9.1k → 19.1k mol/s），
SDAA 基本持平（65~94 mol/s）。说明 SDAA 的瓶颈不在 batch 维度，
加大 batch 无法弥补。

### 8.3 采样（端到端，含进程启动）

| num_smiles | SDAA（3 次均值） | A100（3 次均值） | 倍差 |
|---|---|---|---|
| 50 | 10.4 s | 4.08 s | 2.5× |
| 200 | 14.0 s | 4.07 s | 3.4× |

A100 上 50 与 200 分子耗时几乎相同（4.08 vs 4.07 s），说明
**进程启动 + 模型加载（约 4 s）占主导**；SDAA 侧采样本身耗时明显
（50 → 200 增加约 3.6 s）。端到端倍差看上去只有 2.5~3.4×，
是因为固定开销稀释了 GPU 侧的差距。

### 8.4 强化学习训练（staged_learning）

10 步、batch 32、QED reward、DAP 策略、IdenticalMurckoScaffold 多样性过滤：

| 项目 | SDAA | A100 | 倍差 |
|---|---|---|---|
| 10 步总耗时 | 71.7 s | 8.5 s | 8.4× |
| 最终总分 | 0.67 | 0.71 | — |
| Agent NLL | 30.57 | 30.12 | — |
| 有效分子比例 | 97% | 100% | — |

两边都完整跑完 10 步，分数从初始约 0.4 提升到 0.67~0.71，
**训练链路在 SDAA 上功能正常**。分数/NLL/有效率的小幅差异来自
采样 RNG 不同（TecoRAND vs cuRAND），不是算子精度问题（见 7.3 节，
SDAA 与 CPU fp32 参考的误差仅 1e-5 量级）。

RL 的倍差（8.4×）远小于纯打分（155~282×），因为每步负载由
「采样（串行，SDAA 慢）+ 打分（小批量）+ 反向传播」混合构成，
其中反向传播是批量大算子，SDAA 相对劣势小。

### 8.5 差距分析

Reinvent prior 是 RNN（GRU）模型，参数量仅约 5.9M，
打分过程按 token 逐步前向，**单个 token 的计算量极小**。
这类负载在 SDAA 上的瓶颈是：

1. 大量小算子（每 token 数次 GEMM + elementwise）→ kernel launch
   开销占比高，GPU 利用率低；
2. 批量维度虽可加大，但 RNN 的序列维无法并行，只能逐步展开，
   故 SDAA 吞吐不随 batch 提升；
3. A100 的 launch 开销与单核算力均占优，小模型下优势被放大。

对照前几个模型的实测（均为「SDAA 慢于 A100」）：

| 模型 | 倍差 | 计算特征 |
|---|---|---|
| RFdiffusion | 4.8× | SE(3) 等变大网络 |
| BoltzGen | 11.7× | 扩散 + 多模型串联 |
| PSICHIC | 15.2× | GNN + ESM2 特征 |
| EnzyGen2 | 29.5× | ESM2-650M + EGNN |
| **REINVENT4** | **155~282×** | **5.9M RNN，逐 token 串行** |

规律：模型越小、并行度越低，SDAA 相对劣势越大。
REINVENT 是这批里模型最小、串行最严重的，差距也最大。

### 8.6 对实际使用的影响

- 采样 50 分子约 10 s、200 分子约 14 s，绝对耗时可接受；
- 打分 100 分子约 1.5 s，但 10 万分子量级需约 25 分钟（A100 仅 5 s）；
- 大批量打分建议**拆输入 + 多进程各绑一卡**（框架本身不支持多卡 DDP）；
- RL 训练 10 步 72 s，按 100 步估算约 12 分钟，可接受。

## 九、踩坑记录

| # | 现象 | 原因 | 解决 |
|---|---|---|---|
| 1 | `rdkit` 导入报 NumPy 2 不兼容 | RDKit 用旧 NumPy C API | 源码重编 |
| 2 | `boost.numpy` 编译失败 `elsize` | NumPy 2 移除 `PyArray_Descr::elsize` | 改用 `PyDataType_ELSIZE()` |
| 3 | `libRDKitRDInchiLib.so` undefined symbol `MakeINCHIFromMolfileText` | 内置 libinchi 缺 IXA 层 | 补 stub（仅 MolBlock→InChI 不可用） |
| 4 | 装完 whl 报 `libRDKitRDBoost.so.1: cannot open` | RPATH 少一层 | 按相对深度重算 `$ORIGIN` |
| 5 | `ModuleNotFoundError: mmpdblib.do_fragment` | 装了 mmpdb 3.1.6，超出 `>=2.1,<3` | 降级 mmpdb 2.1 |
| 6 | `xxhash 4.0.1` 版本冲突 | 超出 `<4` 要求 | 降级 3.8.2 |
| 7 | CUDA 侧 `No module named 'dotenv'` | `--no-deps` 漏装隐式依赖 | 补 python-dotenv |
| 8 | `No module named 'torchvision'` | `runmodes/utils/image.py` 隐式依赖 | 装 torchvision 0.23.0（配 torch 2.8.0） |
| 9 | `No module named 'scipy'` | `runmodes/utils/plot.py` 隐式依赖 | 装 scipy |
| 10 | 同 seed 采样跨设备结果完全不同 | TecoRAND 与 cuRAND offset 不同 | 精度对比改用固定输入打分 |
| 11 | Zenodo 下载反复截断 | 限速 32KB/s 且不支持 Range | 一次下完，不支持续传 |