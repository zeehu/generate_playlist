# generate_playlist: 生成式歌单推荐系统 (v2.0)

本项目是一个完整的端到端（End-to-End）机器学习流程，用于根据用户输入的文本（如歌单标题、情绪描述）生成一个推荐的歌曲序列。系统核心采用两阶段模型架构，并巧妙地利用了“向量量化碰撞”作为一种特性。

### 核心思想：簇推荐 (Cluster-based Recommendation)

1.  **语义ID即概念簇**: 我们首先使用残差量化方法（如 K-Means）将海量歌曲向量进行压缩。在这个过程中，多个内容相似的歌曲会被映射到同一个“语义ID”上。我们不再将此视为“碰撞”问题，而是将其看作一个**特性**：这个语义ID现在代表了一个由多首相似歌曲组成的“**概念簇**”（例如，“安静的纯音乐”这个概念）。

2.  **模型学习推荐“概念”**: 下游的T5模型在训练时，学习的是从输入文本到“概念簇”（语义ID）序列的映射。这大大简化了模型的学习任务。

3.  **推荐时采样扩展**: 在最终推荐时，对于模型生成的每一个“概念簇”ID，我们从该簇中**随机采样一首**具体的歌曲。这极大地提升了推荐结果的多样性和惊喜感。

## 🚀 项目特点

- **端到端流程**: 包含从数据处理、模型训练、效果评估到最终推理演示的完整步骤。
- **簇推荐思想**: 将“推荐物品”升级为“推荐概念”，并通过采样提升推荐多样性。
- **配置驱动**: 所有关键参数都在 `config.py` 中统一管理。
- **多GPU支持**: 训练和评估脚本均内置了对分布式计算的支持。

## 📁 项目结构

```
.
├── config.py           # 任务专属的配置文件
├── train_rqvae.py      # 阶段1: 训练RQ-VAE (方法一)
├── train_rq_kmeans.py  # 阶段1: 训练RQ-KMeans (方法二)
├── debug_collisions.py   # 调试工具: 分析碰撞情况，理解簇的大小
├── prepare_corpus.py   # 阶段2: 生成T5模型训练语料
├── train_tiger.py      # 阶段3: 训练T5生成模型
├── evaluate_tiger.py   # 阶段4: 进行簇扩展评估
├── generate_playlist.py# 阶段5: 进行簇采样交互式演示
├── README.md             # 项目说明文件
└── requirements.txt      # (需要您手动创建)
```

## ⚙️ 环境与安装

### 1. 克隆项目
```bash
git clone https://github.com/zeehu/generate_playlist.git
cd generate_playlist
```

### 2. 环境配置
- **Python 版本**: 推荐使用 Python 3.9 或更高版本。
- **创建虚拟环境** (推荐):
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```

### 3. 安装依赖
您需要创建一个 `requirements.txt` 文件，包含以下核心依赖，然后通过 `pip` 安装。

```
# requirements.txt
torch
torchvision
transformers
pandas
numpy
scikit-learn
rouge_score
nltk
tqdm
sentencepiece
faiss-cpu # 或 faiss-gpu
accelerate
```

```bash
# 强烈建议先单独安装 torch 和 torchvision 以确保版本兼容
pip install --upgrade torch torchvision

# 然后安装其余依赖
pip install -r requirements.txt
```

## 💿 数据准备

在运行任何脚本之前，您必须准备好所需的数据文件，并**在 `config.py` 中配置好它们的路径**。

#### **关键配置步骤**

打开 `config.py` 文件，找到 `DataConfig` 和 `SongRQVAEConfig` 类，并填入您数据文件的**绝对路径**。

## 🚀 执行步骤

请在项目根目录（`generate_playlist`）下，严格按照以下顺序执行脚本。

### **阶段1: 生成“歌曲-概念簇”映射**

此阶段的目标是为所有歌曲进行量化，得到它们的“概念ID”。推荐使用速度更快的 K-Means 方法。

*   **命令**:
    ```bash
    python train_rq_kmeans.py
    ```
*   **输出**: `outputs/song_semantic_ids.jsonl`
*   **(可选) 检查簇大小**: 运行 `python debug_collisions.py` 可以查看碰撞情况，即每个“概念簇”中包含了多少首歌曲。

---

### **阶段2: 生成训练语料**

*   **命令**: `python prepare_corpus.py`
*   **输出**: `outputs/train.tsv`, `outputs/val.tsv`, `outputs/test.tsv`。

---

### **阶段3: 训练T5生成模型**

*   **命令 (多GPU, 推荐)**:
    ```bash
    # 将 [GPU数量] 替换为您的GPU卡数, 例如 4
    torchrun --nproc_per_node=[GPU数量] train_tiger.py
    ```
*   **输出**: 最终模型 `models/tiger_final/`。

---

### **阶段4: 评估T5模型**

*   **命令 (多GPU, 推荐)**:
    ```bash
    torchrun --nproc_per_node=[GPU数量] evaluate_tiger.py
    ```
*   **输出**: 终端的总结报告，以及 `outputs/evaluation_results.txt` 中的详细对比文件。

---

### **阶段5: 交互式推理演示**

*   **命令**: `python generate_playlist.py`
*   **输出**: 一个交互式的命令行界面，输入标题即可生成歌单。

---

## 📊 评估指标解读

在您运行 `evaluate_tiger.py` 脚本后，会在终端看到模型的定量评估结果。以下是各项指标的含义及效果好坏的参考范围，帮助您更好地判断模型表现。

### 1. ROUGE-L (F1-Score on Semantic IDs)

- **指标含义**: 这是我们最关注的指标之一。它通过计算模型生成的“语义ID序列”与真实“语义ID序列”之间的最长公共子序列，来衡量两个序列的重合度。它更侧重于**召回率**，即“真实歌单里的概念，有多少被模型预测出来了”，并且对语序有一定的考虑。

- **效果范围参考**:
    - **< 0.2**: 模型基本没学到序列的结构信息。
    - **0.2 - 0.4**: 模型有了一定的序列生成能力，能预测出部分正确的“概念”，但还有很大提升空间。
    - **0.4 - 0.6**: **一个非常好的结果**。这表明模型准确地捕捉到了大部分核心“概念”以及它们的相对顺序。
    - **> 0.6**: 极好的结果，在学术界和工业界都很有竞争力。

### 2. BLEU-4 Score on Semantic IDs

- **指标含义**: 这是一个源于机器翻译的、非常严格的指标。它计算的是模型生成的序列中，连续4个语义ID组成的“词组”（4-gram）在真实序列中出现的精确度。它非常考验模型生成**局部连贯序列**的能力。

- **效果范围参考**:
    - **≈ 0**: 模型无法生成任何连续正确的“歌曲组合”。
    - **0.05 - 0.15**: **一个不错的起点**。对于我们这种词汇表巨大、且带有创造性的生成任务，获得一个显著非零的BLEU-4分数，已经证明了模型的有效性。
    - **0.15 - 0.30**: **非常好的结果**。说明模型能生成大量在真实歌单中出现过的、连贯的“歌曲组合”。
    - **> 0.30**: 极好的结果，通常很难达到。
