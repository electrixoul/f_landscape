# 单样本二维 Loss Landscape 增量实验规划（基于 `visualize_loss_landscape.py`）

> 本规划在你们**已实现的系统**基础上，扩展得到**单个样本级**二维 loss landscape 的批量计算与拼贴展示。方法学延续 Li 等人在 *Visualizing the Loss Landscape of Neural Nets* 中的二维切片与 **filter‑wise 归一化**；实现层面复用现有脚本的方向生成、预计算 logits、网格求损失与绘图管线。关键方法出处：**二维切片定义（第 3 节，式 (1)）**与**filter‑wise 归一化（第 4 节）**；二维网格默认 **51×51（附录 A.5）**。fileciteturn0file1  
> 现有实现与接口参考：`visualize_loss_landscape.py`（你们当前版本，支持 filter‑wise + 可选 Gram–Schmidt；含 2D、3D、1D 可视化与数值导出）与 `train_mnist.py`（线性分类器 784→10、交叉熵、标准化）。fileciteturn0file0 fileciteturn0file2

---

## 0) 目标与交付

**目标**  
- 程序启动时**随机选取 10 个 MNIST 样本**（默认从训练集，可切换到测试集）。  
- **对每个样本**，按与现版本**完全相同**的方法学（同一对随机方向、相同的 filter‑wise 归一化、忽略 bias、相同网格）计算**该样本的二维 loss landscape**。  
- **将 10 张样本级等高线图拼接成一张大图**（共享色条与坐标范围，便于横向对比），保存至 `loss_landscape_results/`。

**交付**  
- 拼贴大图（PNG/PDF；默认 2×5 子图排布、共享色条）。  
- 记录所选样本索引、标签与中心点损失等元数据（`selected_samples.json`/`.npz`）。  
- （可选）每个样本单图与数值网格（`.npz`）。

---

## 1) 数学定义（单样本二维切片）

延续论文二维切片形式（p.3，式 (1)）：对训练终点参数 $\theta^\*$ 与两条随机方向 $\delta,\eta$（**做 filter‑wise 归一化**），**单个样本** $(x_i,y_i)$ 的二维 loss 切片为  
\[
f_i(\alpha,\beta)\ \triangleq\ \ell\!\Big(x_i,y_i;\ \theta^\*+\alpha\,\delta+\beta\,\eta\Big),
\]  
其中 $\ell$ 为交叉熵。**filter‑wise 归一化**：把每个“滤波器”块的方向范数归一到与对应权重块相同（对全连接层即**每一行权重**），例如对 $W\in\mathbb{R}^{10\times 784}$ 的第 $j$ 行 $w_j$：  
\[
\tilde d_j\leftarrow \frac{d_j}{\|d_j\|}\,\|w_j\|,\quad \tilde e_j\leftarrow \frac{e_j}{\|e_j\|}\,\|w_j\|,\quad
\delta_W=\{\tilde d_j\},\ \eta_W=\{\tilde e_j\},\ \delta_b=\eta_b=\mathbf{0}.
\]  
> 以上定义见论文第 4 节（p.4）；二维切片形式见第 3 节（p.3）。fileciteturn0file1

**线性 softmax 分类器下的显式形式**（沿用现版 LSE‑CE 实现）：  
令  
\[
z_i^\*\coloneqq W^\*x_i+b^\*,\quad 
z_i^\delta\coloneqq \delta_W x_i,\quad 
z_i^\eta\coloneqq \eta_W x_i,\quad
\mathrm{LSE}(u)\coloneqq \log\!\sum_k e^{u_k},
\]  
则单样本二维切片  
\[
f_i(\alpha,\beta)
= \mathrm{LSE}\!\Big(z_i^\*+\alpha z_i^\delta+\beta z_i^\eta\Big)
-\Big(z_i^\*+\alpha z_i^\delta+\beta z_i^\eta\Big)_{y_i}.
\]  
> **工程要点**：继续沿用**预计算 logits** 的思路：先在整集上得到 $Z^\*,Z^\delta,Z^\eta$，单样本仅取第 $i$ 行做线性组合，复杂度极低。fileciteturn0file0

---

## 2) 设计原则与对齐要求

- **方向共享**：10 个样本**共享同一对** $(\delta,\eta)$；坐标范围与分辨率统一（默认 $\alpha,\beta\in[-1,1]$、**51×51** 网格；见附录 A.5）。fileciteturn0file1  
- **filter‑wise 归一化**与**忽略 bias**保持不变（与现版实现及论文做法一致）。fileciteturn0file0 fileciteturn0file1  
- **Gram–Schmidt（若启用）后需再次做 filter‑wise 归一化**以恢复每行尺度（或直接关闭 GS）。fileciteturn0file1  
- **对比可视化**：10 张子图**共享色条与坐标轴范围**，避免因色标不一致造成错觉。

---

## 3) 对现有代码的最小改动清单（面向实施）

> 复用你们的 `create_random_directions`、`precompute_logits`、`compute_loss_grid`、`plot_2d_contour` 等函数。fileciteturn0file0

1. **新增 CLI / 配置项**  
   - `--num-samples 10`（默认 10）  
   - `--sample-set {train,test}`（默认 `train`）  
   - `--grid-layout ROWSxCOLS`（默认 `2x5`）  
   - `--share-colorbar {true,false}`（默认 `true`）  
   - `--color-scale {global,quantile}`（默认 `global`；`quantile` 使用 5–95 分位裁剪）

2. **样本选择**  
   - 加载数据后、创建方向前：**随机选 10 个索引**（固定随机种子，记录到日志；默认从 `train_dataset` 选取）。  
   - 保存选中样本的**图像缩略图、标签、中心点损失** $f_i(0,0)$ 与中心预测概率等元数据到 `selected_samples.json`。

3. **复用整集 logits 预计算**  
   - 调用 `precompute_logits(...)` 得到整集 $Z^\*,Z^\delta,Z^\eta$ 与 `labels`。  
   - 对每个选中索引 $i$，**抽取第 $i$ 行**：$(z_i^\*,z_i^\delta,z_i^\eta,y_i)$。

4. **新增单样本网格计算函数**  
   - `compute_single_sample_grid(z_star_row, z_delta_row, z_eta_row, y_i, alpha_vals, beta_vals)`  
   - 返回 `loss_grid_i ∈ ℝ^{res×res}`。建议**向量化**：先构造 $A∈ℝ^{res×1}$、$B∈ℝ^{1×res}$，广播得到 $Z_{a,b,:}=z_i^\*+A_{a,0}\,z_i^\delta+B_{0,b}\,z_i^\eta$，再对类别维做 `logsumexp` 并减去正确类 logit。

5. **新增拼贴绘图函数**  
   - `plot_samples_grid(alpha_vals, beta_vals, loss_grids, metas, layout, share_colorbar, color_scale)`  
   - `loss_grids`：长度为 10 的列表；`metas`：每个样本的 `idx / y / ŷ / p(y|θ*) / f_i(0,0)` 与**缩略图**。  
   - **统一色标**：  
     - `global`：在全部样本网格上取 $v_{\min},v_{\max}$；  
     - `quantile`：用 5%–95% 分位数；更稳健。  
   - **子图元素**：等高线（或 `contourf`）、中心点 ★、标题（`idx=#12345 | y=7, ŷ=9 | p=0.63 | L(0,0)=1.23`）、右侧共享色条；右上角可嵌入 28×28 灰度缩略图。

6. **输出与日志**  
   - `loss_landscape_results/samples_landscape_grid.png(.pdf)`（拼贴大图）。  
   - `selected_samples.json`（或 `.npz`）：索引、标签、预测、概率、中心损失。  
   - （可选）`sample_i_loss_grid.npz`（单样本网格）与 `sample_i.png`（单图）。

---

## 4) 伪代码（新增/改动逻辑）

```pseudocode
PROC main_incremental():
    cfg ← {num_samples=10, sample_set='train', resolution=51, ranges=[-1,1]×[-1,1], layout='2x5', color_scale='global'}
    set_random_seed(SEED)

    # 1) 模型与数据（与现版一致）
    model ← load_model('mnist_linear_model.pth'); set_eval(model)             # 784→10 线性分类器
    loaders ← load_MNIST_normalized()                                         # 与训练脚本相同的标准化
    dataset ← (cfg.sample_set=='train') ? loaders.train.dataset : loaders.test.dataset

    # 2) 随机方向 + filter-wise 归一化（忽略 bias）
    delta, eta ← create_random_directions(model, ignore_bias=True)
    IF use_gram_schmidt:
        eta ← gram_schmidt(delta, eta)
        eta ← filter_wise_renormalize(eta, model)  # ★ GS 后再次归一化

    # 3) 随机选 10 个样本索引
    S ← sample_indices_uniform(dataset, k=cfg.num_samples, seed=SEED)
    save_json('selected_samples.json', S)

    # 4) 预计算整集 logits（与现版一致）
    Z*, Zδ, Zη, labels ← precompute_logits(model, loaders[cfg.sample_set], delta, eta, device)

    # 5) 逐样本计算二维网格（驻留内存）
    α_vals ← linspace(cfg.ranges.α_min, cfg.ranges.α_max, cfg.resolution)
    β_vals ← linspace(cfg.ranges.β_min, cfg.ranges.β_max, cfg.resolution)
    loss_grids ← []; metas ← []

    FOR idx IN S:
        z* ← row(Z*, idx); zδ ← row(Zδ, idx); zη ← row(Zη, idx); y ← labels[idx]
        G ← compute_single_sample_grid(z*, zδ, zη, y, α_vals, β_vals)
        loss_grids.append(G)
        center_loss ← compute_single_sample_loss(z*, zδ, zη, y, α=0, β=0)
        y_hat, p_y ← predict_at_center(z*)
        metas.append({idx, y, y_hat, p_y, center_loss, image=dataset[idx].image})

    # 6) 拼贴绘图
    plot_samples_grid(α_vals, β_vals, loss_grids, metas, layout=cfg.layout, share_colorbar=True, color_scale=cfg.color_scale)
    savefig('loss_landscape_results/samples_landscape_grid.png')
```

**单样本网格（向量化）**
```pseudocode
FUNC compute_single_sample_grid(z*, zδ, zη, y, α_vals, β_vals):
    A ← α_vals.reshape(res, 1)                     # res×1
    B ← β_vals.reshape(1, res)                     # 1×res
    Z ← z* + A ⊗ zδ + B ⊗ zη                       # 广播得到 res×res×10
    LSE ← logsumexp(Z, axis=2)                     # res×res
    correct ← Z[:, :, y]                           # 取正确类 logit（res×res）
    RETURN LSE - correct                           # res×res
```

---

## 5) 可视化细节与对比策略

- **拼贴布局**：默认 2×5（更适合 16:9），所有子图共享 $(x,y)$ 轴范围与刻度（$[-1,1]$）。  
- **统一色标**：  
  - `global`：$v_{\min}=\min_{i,a,b} f_i(\alpha_a,\beta_b)$，$v_{\max}=\max_{i,a,b} f_i(\alpha_a,\beta_b)$；  
  - `quantile`：在所有样本值上取 5%–95% 分位，抗离群更稳。  
- **标注**：中心点 ★；标题含 `idx / y / ŷ / p(y|θ*) / L(0,0)`；右上角可嵌入该样本 28×28 缩略图。  
- **等高线层级**：建议 30 个 levels，与现版一致。

---

## 6) 质量校验（单样本版）

1. **中心点一致性**（每个样本）：验证  
   \[
   f_i(0,0)=\mathrm{LSE}(z_i^\*)-(z_i^\*)_{y_i}
   \]  
   与**直接前向 + CrossEntropyLoss(单样本)**一致（数值误差 $\approx 10^{-6}\!\sim\!10^{-5}$）。  
2. **索引与可视核对**：拼贴标题中的 `idx`、标签与缩略图一致。  
3. **统一色标**：确认 10 图共享色条（避免视觉误解）。  
4. **方向稳健性（可选）**：更换随机种子重复一次，定性形状应稳定（论文附录 A.4 有复现实验）。fileciteturn0file1

---

## 7) 性能与内存

- 仍采用**整集 logits 预计算**（$Z^\*,Z^\delta,Z^\eta$），MNIST 体量内存仅数 MB。  
- 单样本网格仅 $51^2=2601$ 点，10 个样本约 $2.6\times 10^4$ 点，计算量极小。  
- 建议在 `torch.no_grad()` 下构建 $Z$ 与做 LSE。fileciteturn0file0

---

## 8) 失败模式与兜底

- **色域被“拉爆”**：启用 `quantile` 色标或缩小坐标范围（如 $[-0.5,0.5]$）。  
- **GS 后尺度漂移**：若保留 Gram–Schmidt，务必**再次**对 $\eta$ 做 filter‑wise 归一化。  
- **数据下载**：为“一键复现”，`download=True` 或先检测本地再决定。fileciteturn0file0

---

## 9) 与现有实现/论文的对齐说明

- **二维切片** $f(\alpha,\beta)=L(\theta^\*+\alpha\delta+\beta\eta)$：继续使用现版 LSE‑CE 与切片实现（论文 p.3 式 (1)）。fileciteturn0file1  
- **filter‑wise 归一化**：FC 层按“行=滤波器”处理，忽略 BN/bias 扰动；与论文第 4 节（p.4）与现版一致。fileciteturn0file1 fileciteturn0file0  
- **分辨率/范围**：默认 **51×51** 与 $[-1,1]$ 对齐附录 A.5；现版已如此设置。fileciteturn0file1  
- **模型与数据管线**：延续 `train_mnist.py` 的线性分类器与标准化（ToTensor + Normalize）。fileciteturn0file2

---

## 10) 实验记录模板（新增字段）

- 随机种子、两方向向量（是否保存）、是否启用 GS（若启用，是否再归一化）。  
- 选样本索引 `idx` 列表、对应标签/预测/中心损失。  
- 网格范围与分辨率、色标策略（global / quantile）。  
- 输出文件路径（拼贴大图、可选的单样本网格与单图）。

---

### 附：实施者 Checklist

1. 在现有仓库复制 `visualize_loss_landscape.py` 为新文件，按第 3 节新增 CLI 与 3 个函数：  
   `sample_indices_uniform(...)`、`compute_single_sample_grid(...)`、`plot_samples_grid(...)`。fileciteturn0file0  
2. 在 `main()`：**选样本 → 预计算 logits → 循环 10 个样本算网格（驻留内存）→ 统一色标拼贴 → 保存**。  
3. 将所选索引/标签/预测/中心损失写入 `selected_samples.json`，并在 `experiment_log.txt` 追加记录。  
4. （可选）对比 `global` 与 `quantile` 色标，选择更稳定者为默认。

---

> 参考：*Visualizing the Loss Landscape of Neural Nets*（二维切片与 filter‑wise 归一化定义、网格分辨率等）；你们的 `visualize_loss_landscape.py` 与 `train_mnist.py` 提供了已验证的数据管线与模型接口，可直接复用上述改动落地。fileciteturn0file1 fileciteturn0file0 fileciteturn0file2
