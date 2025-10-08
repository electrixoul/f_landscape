下面是一份可直接交给实验员执行的“**MNIST 线性分类器二维 loss 地形测量与可视化**”实验规划。方案严格遵循 Li 等人在 *Visualizing the Loss Landscape of Neural Nets* 提出的“随机方向 + 滤波器（filter‑wise）归一化”的方法学，并结合你给出的最简单双层线性分类脚本与已训练模型。文中给出完整的数学定义与可操作的伪代码（不含 Python 代码）。  
参考依据：论文中的二维切片公式与“filter normalization”定义（见第 3 节与第 4 节），以及作者开源实现中关于忽略 bias/BN 与网格分辨率的默认做法。fileciteturn0file0 citeturn0search2turn0search3turn0search0

---

## 0. 目标与交付

- **目标**：在训练终点参数 $\theta^\*$ 附近，按论文方法计算并绘制模型的二维损失地形 $f(\alpha,\beta)$ 的等高线（训练集为主，可选测试集对照）。  
- **交付**：  
  1) 训练集 loss 的二维等高线图（中心点 $(0,0)$ 标记为 ★），可选附加 3D 曲面图；  
  2) （可选）沿两坐标轴的一维剖面曲线；  
  3) 计算用的网格数据（例如 `Z_train.npy`、`Z_test.npy`）、方向向量与实验日志（含随机种子）；  
  4) 复现实验的简要说明（含依赖版本、命令行）。

---

## 1. 实验对象与环境

- **模型**：你提供的最小线性分类器（`LinearMNIST`），结构为单层全连接 $784\to 10$（含 bias），损失函数为交叉熵；训练脚本与保存检查点 `mnist_linear_model.pth` 已提供。fileciteturn0file1  
- **数据**：MNIST 训练/测试集；与训练时相同的标准化变换。fileciteturn0file1  
- **设备**：CPU/MPS/GPU 均可。线性模型计算量可控，但仍建议按第 6 节做向量化与缓存以降本增效。  
- **随机性**：固定随机种子（记录到日志），以保证可复现。

---

## 2. 方法学与数学定义（严格按照论文）

### 2.1 二维切片的基本定义
选择训练终点参数 $\theta^\*$ 为中心，采样两条随机方向 $\delta,\ \eta$（与参数形状一致），在以它们张成的二维平面上作损失切片：  
$$
f(\alpha,\beta)\ \triangleq\ L\big(\theta^\*+\alpha\,\delta+\beta\,\eta\big),
$$
其中 $L(\theta)=\frac{1}{m}\sum_{i=1}^{m}\ell(x_i,y_i;\theta)$，$\ell$ 为交叉熵。该二维切片形式即论文式（1）的直接应用。fileciteturn0file0

> **注意**：论文强调二维切片更能暴露非凸结构；我们采用与论文一致的**随机方向**而非优化轨迹方向。fileciteturn0file0

### 2.2 Filter‑wise 归一化（核心步骤）
为消除缩放不变性与尺度不一致带来的“假性尖/平”失真，随机方向需做**逐滤波器归一化**（filter normalization）：  
对网络第 $i$ 层的第 $j$ 个“滤波器”记为 $\theta_{i,j}$（对全连接层即**生成同一输出神经元的权重向量**），随机方向向量的对应块为 $d_{i,j}$。按论文定义进行归一化：  
$$
\tilde d_{i,j} \leftarrow \frac{d_{i,j}}{\|d_{i,j}\|}\,\|\theta_{i,j}\|,\quad
\tilde e_{i,j} \leftarrow \frac{e_{i,j}}{\|e_{i,j}\|}\,\|\theta_{i,j}\|.
$$
然后用 $\tilde\delta=\{\tilde d_{i,j}\}$、$\tilde\eta=\{\tilde e_{i,j}\}$ 作为最终方向进入 $f(\alpha,\beta)$。对全连接层，**一个滤波器对应输出神经元的权重行向量**（本模型是 $W\in\mathbb{R}^{10\times 784}$ 的每一行 $w_j$）。fileciteturn0file0

> **bias 处理**：作者开源实现常用设置是**忽略 bias 与 BN 参数**（命令行 `--xignore biasbn`），实践上可令 $\tilde \delta_b=\tilde \eta_b=\mathbf{0}$，从而仅扰动权重矩阵 $W$。本实验按此执行（更稳且与参考实现一致）。citeturn0search0

### 2.3 线性 softmax 分类器下的显式损失  
对样本 $(x_i,y_i)$，设 $z_i(\theta)=W x_i + b\in\mathbb{R}^{10}$ 为 logits，则
$$
\ell(x_i,y_i;\theta) = -\log\frac{\exp\big(z_i(\theta)_{y_i}\big)}{\sum_{k=1}^{10}\exp\big(z_i(\theta)_k\big)}.
$$
因此二维切片的点值为
$$
L(\alpha,\beta)=\frac1m\sum_{i=1}^m\Big(\mathrm{LSE}\big(z_i^\*+\alpha z_i^\delta+\beta z_i^\eta\big)-\big(z_i^\*+\alpha z_i^\delta+\beta z_i^\eta\big)_{y_i}\Big),
$$
其中  
$z_i^\*\coloneqq W^\*x_i+b^\*$, $\quad$ 
$z_i^\delta\coloneqq \tilde\delta_W x_i$（若忽略 bias 扰动）, $\quad$ 
$z_i^\eta\coloneqq \tilde\eta_W x_i$,  
$\mathrm{LSE}(u)\coloneqq \log\sum_k e^{u_k}$。  
> **要点**：由于模型关于参数是线性的，**只需一次性**预计算 $z^\*, z^\delta, z^\eta$，即可在任意 $(\alpha,\beta)$ 处快速组装 logits 并求 loss（见第 6 节的高效计算）。该思想与论文二维切片完全一致，只是针对线性模型做的工程加速。fileciteturn0file0

---

## 3. 网格与可视化设置（默认值遵循论文/代码）

- **二维坐标范围**：$\alpha,\beta\in[-1,1]$（如出现数值溢出或全图过平，可调整到 $[-0.5,0.5]$ 或 $[-2,2]$）。  
- **分辨率**：默认 $51\times 51$（论文附录 A.5 给出的 2D 轮廓默认分辨率；作者代码示例也使用 `-1:1:51`）。fileciteturn0file0 citeturn0search0  
- **可视化**：训练集 loss 的**等高线图**（更便于观察非凸结构），中心 $(0,0)$ 处以 ★ 标注；色条显示 loss 数值。可选：3D 曲面图、测试集 loss 的对照图、一维剖面（$\beta=0$、$\alpha=0$）。

---

## 4. 实验流程（面向实施）

### 4.1 数据与模型准备
1. 载入 MNIST 训练/测试集，使用与训练相同的标准化。fileciteturn0file1  
2. 构建 `LinearMNIST` 模型、加载 `mnist_linear_model.pth` 的 `model_state_dict`，置 `eval()`。fileciteturn0file1  
3. 提取最终参数 $\theta^\*=\{W^\*,b^\*\}$。

### 4.2 随机方向采样与 filter 归一化
1. 固定随机种子。  
2. 为每个需要扰动的参数张量采样高斯方向：  
   - 对本模型，仅对 $W^\* \in\mathbb{R}^{10\times 784}$ 采样 $\delta_W,\eta_W\sim\mathcal{N}(0,1)$；**忽略** bias（$\tilde\delta_b=\tilde\eta_b=\mathbf{0}$）。citeturn0search0  
3. **逐滤波器归一化**（全连接层视每一行 $w_j$ 为一个“滤波器”）：
   $$
   \tilde\delta_W[j,:]\leftarrow\frac{\delta_W[j,:]}{\|\delta_W[j,:]\|+\varepsilon}\,\|W^\*[j,:]\|,\quad
   \tilde\eta_W[j,:]\leftarrow\frac{\eta_W[j,:]}{\|\eta_W[j,:]\|+\varepsilon}\,\|W^\*[j,:]\|.
   $$
   其中 $\varepsilon$ 为极小正数以防零除。fileciteturn0file0  
4. （可选稳健性）对 $\tilde\delta_W,\tilde\eta_W$ 做一次 Gram–Schmidt 使其接近正交，以减少方向相关性（论文未强制要求，此处为工程增强）。

### 4.3 高效计算二维 loss（线性模型的向量化技巧）
**关键优化**：一次前向得到并缓存三组 logits，之后在每个网格点仅做“线性组合 + softmax 交叉熵”。  
1. 预计算并缓存（训练集）：  
   $$
   Z^\*(i,:)=W^\*x_i+b^\*,\quad Z^\delta(i,:)=\tilde\delta_W x_i,\quad Z^\eta(i,:)=\tilde\eta_W x_i.
   $$  
   存储为形状 $(m,10)$ 的三个矩阵；内存约 $m\times 10\times 3\times 4$ bytes（MNIST 约数 MB 量级）。  
2. 对每个网格点 $(\alpha,\beta)$：  
   $$
   Z_{\alpha,\beta}=Z^\*+\alpha Z^\delta+\beta Z^\eta,
   $$
   $$
   L_{\text{train}}(\alpha,\beta)=\frac1m\sum_{i=1}^m\Big(\mathrm{LSE}(Z_{\alpha,\beta}(i,:)) - Z_{\alpha,\beta}(i,y_i)\Big).
   $$  
   批处理计算并写入数组 $Z_{\text{loss}}\in\mathbb{R}^{n_\alpha\times n_\beta}$。  
3. （可选）对测试集重复上述步骤得到 $L_{\text{test}}(\alpha,\beta)$。

> 论文默认在训练集上绘制地形；我们按此为主，测试集曲面仅作参考。fileciteturn0file0

### 4.4 可视化与导出
- 使用等高线（contour/contourf）绘制 $L_{\text{train}}(\alpha,\beta)$，中心 $(0,0)$ 处以 ★ 标注；添加色条与轴标签（“α direction (filter‑normalized)”，“β direction (filter‑normalized)”）。  
- （可选）绘制 3D 曲面；绘制 $\beta=0$ 和 $\alpha=0$ 的 1D 剖面曲线。  
- 保存图像（PNG/PDF）与数值网格（NumPy `.npy/.npz`）。

---

## 5. 伪代码（不含任何 Python 语法依赖）

> 约定：`FMN` 表示“filter‑wise 归一化”，`LSE` 表示 `log-sum-exp`。

```pseudocode
PROC main():
    set_random_seed(SEED)
    # 1) 数据与模型
    train_set, test_set ← load_MNIST_with_same_normalization_as_training()
    model ← build_LinearMNIST()
    load_state_dict(model, "mnist_linear_model.pth")  # 仅加载 model_state_dict
    set_eval(model)

    # 2) 抽取参数
    W*, b* ← model.fc.weight, model.fc.bias

    # 3) 随机方向并做 filter-wise 归一化（忽略 bias）
    δW ∼ Normal(0,1) with shape = shape(W*)
    ηW ∼ Normal(0,1) with shape = shape(W*)
    FOR j in 0..9:  # 10 个输出神经元，每行是一个“滤波器”
        δW[j] ← δW[j] / (norm(δW[j]) + ε) * norm(W*[j])
        ηW[j] ← ηW[j] / (norm(ηW[j]) + ε) * norm(W*[j])
    END
    # 可选：Gram-Schmidt(δW, ηW)

    # 4) 预计算 logits（训练集/可选测试集）
    Zstar_train  ← logits_of_dataset(W*, b*, train_set)
    Zdelta_train ← logits_of_dataset(δW, 0,   train_set)  # bias 扰动为 0
    Zeta_train   ← logits_of_dataset(ηW, 0,   train_set)
    y_train      ← labels_vector(train_set)

    # 5) 建立二维网格并计算 loss
    α_grid ← linspace(-1, 1, 51)  # 论文/代码默认
    β_grid ← linspace(-1, 1, 51)
    Zloss_train ← zeros(len(α_grid), len(β_grid))
    FOR a_idx, α IN enumerate(α_grid):
        FOR b_idx, β IN enumerate(β_grid):
            Z ← Zstar_train + α * Zdelta_train + β * Zeta_train
            # 交叉熵的稳定实现：mean(LSE(Z, axis=1) - Z[range(m), y_train])
            L ← mean( LSE(Z, axis=1) - gather(Z, y_train) )
            Zloss_train[a_idx, b_idx] ← L
        END
    END

    # 6) 可视化
    plot_contour(α_grid, β_grid, Zloss_train, center_marker='★', title='MNIST linear - train loss landscape')
    save("landscape_train.png"); save("Zloss_train.npy")
    # 可选：测试集重复 4)-5)，保存 test 曲面与对比图；绘制 1D 剖面与 3D 曲面等
END

FUNC logits_of_dataset(W, b, dataset):
    # 逐批或一次性向量化：返回所有样本的 logits = X @ W^T + b（其中 b 可为 0）
    ACC ← []
    FOR batch IN dataset:
        X ← flatten_to_784(batch.images)  # 与训练脚本一致的展开
        Z ← X @ transpose(W) + b
        ACC.append(Z)
    END
    RETURN concatenate(ACC, axis=0)
```

---

## 6. 计算效率与稳定性建议（强烈推荐执行）

- **一次性缓存三份 logits**：$Z^\*,Z^\delta,Z^\eta$；线性模型下内存很小（MNIST 训练集约 $60\text{k}\times 10\times 3\times 4\text{B}\approx 7.2$MB），随后所有网格点仅做“线性组合 + LSE”，速度大幅提升。  
- **数值稳定**：使用 $\mathrm{LSE}$ 实现 softmax‑CE，避免上溢。  
- **自适应范围（可选）**：若 $[-1,1]$ 过大/过小，先在 $\beta=0$ 上对 $\alpha$ 做 1D 扫描选取合适半径 $r$，再设 $\alpha,\beta\in[-r,r]$。  
- **分辨率/抽样折衷**：  
  - 快速版：$31\times 31$ 网格 + 训练集子样本（如 10k）用于预览；  
  - 精确版：$51\times 51$ + 全训练集（与论文默认一致）。fileciteturn0file0  
- **重复性检查（建议）**：更换 2–3 个随机种子重复方向采样，确认地形形态稳定（见论文附录 A.4 的复现实验做法）。fileciteturn0file0

---

## 7. 质量校验与验收标准

1. **中心点一致性**：$(\alpha,\beta)=(0,0)$ 处的 $L_{\text{train}}$ 应与直接在 $\theta^\*$ 上评估的训练集 loss 相等（数值误差在 $10^{-6}\sim10^{-5}$ 量级）。  
2. **方向线性验证**：一维剖面上 $\alpha$ 很小时，loss 变化与二阶近似（Hessian 主曲率）一致性应合理（定性检查）。  
3. **可读性**：等高线层级与色带分布应能清楚区分等高线（可用分位数分层）；中心位置清晰可见。  
4. **稳定性**：不同随机种子得到的地形形态应保持定性一致（见论文的可重复性结果）。fileciteturn0file0

---

## 8. 与论文/代码实现的对齐点

- 二维切片形式 $f(\alpha,\beta)=L(\theta^\*+\alpha\delta+\beta\eta)$ 直接来源于论文第 3 节（式 (1)）。fileciteturn0file0  
- 方向的**filter‑wise 归一化**与“全连接层把每个输出神经元权重向量视作滤波器”的处理来自论文第 4 节；按此可跨架构比较曲率/尖平而不受尺度影响。fileciteturn0file0  
- 分辨率 $51\times 51$ 与范围 $[-1,1]$ 的默认选择与作者实现保持一致（示例命令 `--x=-1:1:51 --xnorm filter --xignore biasbn`）。citeturn0search0  
- 我们的模型为**无 BN、无 ReLU**的线性 softmax 分类器，虽然缺少论文中特别强调的 BN 缩放不变性，但 filter‑wise 归一化仍能提供**稳定且可比**的扰动尺度与更可靠的局部几何观察。fileciteturn0file0

---

## 9. 风险与备选方案

- **loss 全图“爆表”**：减小范围到 $[-0.5,0.5]$ 或启用对数色标；或先做 1D 探测自适应选择半径。  
- **地形几乎平坦**：增大范围到 $[-2,2]$ 或提高分辨率到 $101\times 101$（计算量线性增加）。  
- **bias 处理**：若确需观察 bias 扰动，可在“方向归一化”中为 bias 设置  
  $\tilde\delta_b[j]\leftarrow\mathrm{sign}(\delta_b[j])\cdot\max(|b^\*_j|,\epsilon)$；但与作者默认（忽略 bias）不同，会改变图形刻度，务必单列实验。citeturn0search0

---

## 10. 实验记录模板（建议）

- 代码版本、依赖（torch/torchvision 等）；  
- 随机种子；  
- 网格设置（范围、分辨率）、是否忽略 bias；  
- $L_{\text{train}}(0,0)$、$L_{\text{test}}(0,0)$；  
- 图像与 `.npy/.npz` 文件路径。

---

## 参考与依据（关键出处）

- 二维切片定义与 filter‑wise 归一化、全连接层“滤波器”解释、默认分辨率等均来自：*Visualizing the Loss Landscape of Neural Nets*（Li et al., NeurIPS 2018）。fileciteturn0file0  
- 作者开源代码 README 示例展示了 Filter 归一化与**忽略 bias/BN**的常用设置，以及 `-1:1:51` 的网格范围：`--xnorm filter --xignore biasbn --x=-1:1:51`。citeturn0search0  
- 本实验所用线性模型与数据管线源自你提供的脚本（`LinearMNIST`，784→10，交叉熵训练与保存）。fileciteturn0file1

> 备注：论文 arXiv 页面与会议版 PDF 同步可查（内容与本文所用关键方法一致），供实验员进一步核对。citeturn0search2turn0search3

---

### 附：本项目与论文图示的差异与一致性说明
- **一致性**：二维切片形式、随机方向采样、filter‑wise 归一化、默认网格分辨率、以训练集可视化为主。fileciteturn0file0 citeturn0search0  
- **差异**：我们的模型是**线性**而非深网，因此可用“logits 线性分解 + LSE”的方式显著加速计算（这是对论文方法的工程化实现，未改变其数学内核）。  
