# 生成式歌单推荐系统

本项目是一个完整的端到端（End-to-End）机器学习流程，用于根据用户输入的文本（如歌单标题、情绪描述）生成一个推荐的歌曲序列。系统核心采用两阶段模型架构：
1.  **RQ-VAE**: 首先，使用残差量化自编码器（RQ-VAE）将每首歌曲高维、连续的向量（例如100维）压缩为低维、离散的“语义ID”。这相当于为每首歌曲学习一个独特的、由几个数字组成的“代号”。
2.  **T5 模型**: 然后，使用一个强大的序列到序列模型（T5），学习从输入的文本描述到目标歌曲语义ID序列的映射关系。

所有与此歌单生成任务相关的代码都已整合在 `playlist_src/` 目录下。

## 📁 项目结构

```
.
├── playlist_src/           # 歌单生成任务的核心代码目录
│   ├── config.py           # 任务专属的配置文件
│   ├── train_rqvae.py      # 阶段1: 训练RQ-VAE，生成语义ID
│   ├── evaluate_rqvae.py   # 阶段1b: 评估语义ID质量
│   ├── prepare_corpus.py   # 阶段2: 生成T5模型训练语料
│   ├── train_tiger.py      # 阶段3: 训练T5生成模型
│   ├── evaluate_tiger.py   # 阶段4: 评估T5模型性能
│   └── generate_playlist.py# 阶段5: 交互式推理与演示
├── models/                 # 存储训练好的模型文件
├── outputs/                # 存储中间生成文件 (语义ID, 训练语料等)
└── logs/                   # 存储训练日志
```

## ⚙️ 环境配置

1.  **Python 版本**: 推荐使用 Python 3.9 或更高版本。
2.  **创建虚拟环境** (推荐):
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```
3.  **安装依赖**: 项目的核心依赖项已在 `requirements.txt` 中列出。
    ```bash
    pip install -r requirements.txt
    # 如果遇到问题，请确保以下核心库已安装
    # pip install torch transformers pandas numpy scikit-learn evaluate rouge_score
    ```

## 💿 数据准备

在运行任何脚本之前，您必须准备好以下四份数据文件，并**在 `playlist_src/config.py` 中配置好它们的路径**。

1.  **歌曲向量文件**: `song_vectors.csv` (示例名)
    *   格式: `mixsongid,v_0,v_1,...,v_99`
2.  **歌单信息文件**: `gen_playlist_info.csv` (示例名)
    *   格式: `glid,listname,tag_list`
3.  **歌单-歌曲关系文件**: `gen_playlist_song.csv` (示例名)
    *   格式: `special_gid,mixsongid`
4.  **歌曲元信息文件**: `gen_song_info.csv` (示例名)
    *   格式: `mixsongid,song_name,singer_name`

#### **关键配置步骤**

打开 `playlist_src/config.py` 文件，找到 `DataConfig` 类，并填入您数据文件的**绝对路径**。

```python
# 位于 playlist_src/config.py
@dataclass
class DataConfig:
    # --- 将下面四行修改为您的实际文件路径 ---
    song_info_file: str = "/path/to/your/gen_song_info.csv"
    playlist_info_file: str = "/path/to/your/gen_playlist_info.csv"
    playlist_songs_file: str = "/path/to/your/gen_playlist_song.csv"
    
    # 注意：song_vector_file 的配置在 SongRQVAEConfig 中
    ...

@dataclass
class SongRQVAEConfig:
    # --- 将下面这行修改为您的文件路径 ---
    song_vector_file: str = "/path/to/your/song_vectors.csv"
    ...
```

## 🚀 执行步骤

请严格按照以下顺序执行脚本。

### **阶段1: 生成歌曲语义ID**

此阶段的目标是将每首歌曲的100维向量转换为由2个数字组成的语义ID（例如 `[123, 456]`）。

*   **命令**:
    ```bash
    python playlist_src/train_rqvae.py
    ```
*   **输出**:
    *   训练好的RQ-VAE模型: `models/song_rqvae_best.pt`
    *   歌曲语义ID映射文件: `outputs/song_semantic_ids.jsonl`
*   **注意**:
    *   您可以在 `playlist_src/config.py` 的 `SongRQVAEConfig` 中调整模型参数（如 `epochs`, `batch_size`）和码本设置（`vocab_size`, `levels`）。当前配置为100万首歌曲提供了合理的默认值。

---

### **阶段1b (可选): 评估语义ID质量**

在进入下一步之前，强烈建议运行此脚本检查第一阶段的成果。

*   **命令**:
    ```bash
    python playlist_src/evaluate_rqvae.py
    ```
*   **输出**:
    一份关于模型重构质量（均方差、余弦相似度）和邻域保持度的报告。
*   **注意**:
    *   请重点关注报告中的 **`Average Cosine Similarity`** 指标。如果该值低于 `0.8`，说明语义ID可能丢失了过多信息，建议增加阶段1的训练轮数或调整模型参数。

---

### **阶段2: 生成训练语料**

此阶段将所有数据源整合，生成T5模型可以直接学习的文本文件。

*   **命令**:
    ```bash
    python playlist_src/prepare_corpus.py
    ```
*   **输出**:
    *   `outputs/train.tsv`
    *   `outputs/val.tsv`
    *   `outputs/test.tsv`
*   **注意**:
    *   脚本会自动对每个歌单内的歌曲按ID排序，这是一个重要的设计，旨在消除数据歧义，稳定模型学习。

---

### **阶段3: 训练T5生成模型**

这是最核心的模型训练步骤，脚本已为多GPU环境优化。

*   **命令 (多GPU, 推荐)**:
    ```bash
    # 将 [GPU数量] 替换为您的GPU卡数, 例如 4
    torchrun --nproc_per_node=[GPU数量] playlist_src/train_tiger.py
    ```
*   **命令 (单GPU)**:
    ```bash
    python playlist_src/train_tiger.py
    ```
*   **输出**:
    *   最终的生成模型: `models/tiger_final/`
*   **注意**:
    *   训练过程支持通过 TensorBoard 进行监控。运行 `tensorboard --logdir=logs/tiger_logs` 即可查看损失曲线。
    *   所有训练参数（如学习率、批量大小、训练轮数）都可以在 `playlist_src/config.py` 的 `PlaylistTIGERConfig` 中进行调整。

---

### **阶段4 (可选): 评估T5模型**

对训练好的T5模型进行量化评估。

*   **命令**:
    ```bash
    python playlist_src/evaluate_tiger.py
    ```
*   **输出**:
    一份包含ROUGE, BLEU, 和歌曲F1-Score等指标的评估报告。
*   **注意**:
    *   请重点关注 **`Avg. F1-Score`**，它直观地反映了模型生成歌单内容的准确性。

---

### **阶段5: 交互式推理演示**

与您最终训练好的模型进行实时互动。

*   **命令**:
    ```bash
    python playlist_src/generate_playlist.py
    ```
*   **输出**:
    一个交互式的命令行界面，您可以输入歌单标题，模型将返回推荐的歌曲列表（包含歌名和歌手）。
*   **注意**:
    *   请确保 `gen_song_info.csv` 的路径已在 `config.py` 中正确配置，否则将无法显示歌名和歌手信息。