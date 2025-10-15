# generate_playlist: 生成式歌单推荐系统

本项目是一个完整的端到端（End-to-End）机器学习流程，用于根据用户输入的文本（如歌单标题、情绪描述）生成一个推荐的歌曲序列。系统核心采用两阶段模型架构：

1.  **RQ-VAE / RQ-Kmeans**: 首先，使用残差量化方法将每首歌曲高维、连续的向量（例如100维）压缩为低维、离散的“语义ID”。
2.  **T5 模型**: 然后，使用一个强大的序列到序列模型（T5），学习从输入的文本描述到目标歌曲语义ID序列的映射关系。

## 🚀 项目特点

- **端到端流程**: 包含从数据处理、模型训练、效果评估到最终推理演示的完整步骤。
- **两阶段模型**: 解耦了物品表示学习和序列生成，使流程更清晰，易于调试和扩展。
- **配置驱动**: 所有关键参数（如文件路径、模型超参）都在 `config.py` 中统一管理。
- **多GPU支持**: 训练和评估脚本均内置了对分布式计算的支持。

## 📁 项目结构

```
.
├── config.py           # 任务专属的配置文件
├── train_rqvae.py      # 阶段1: 训练RQ-VAE (方法一)
├── train_rq_kmeans.py  # 阶段1: 训练RQ-KMeans (方法二)
├── evaluate_rqvae.py   # 阶段1b: 评估RQ-VAE质量
├── evaluate_rq_kmeans.py # 阶段1b: 评估RQ-KMeans质量
├── prepare_corpus.py   # 阶段2: 生成T5模型训练语料
├── train_tiger.py      # 阶段3: 训练T5生成模型
├── evaluate_tiger.py   # 阶段4: 评估T5模型性能
├── generate_playlist.py# 阶段5: 交互式推理与演示
├── README.md             # 项目说明文件
├── requirements.txt      # (需要您手动创建)
├── models/                 # (自动生成) 存储训练好的模型文件
├── outputs/                # (自动生成) 存储中间生成文件
└── logs/                   # (自动生成) 存储训练日志
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
pip install -r requirements.txt
```

## 💿 数据准备

在运行任何脚本之前，您必须准备好以下四份数据文件，并**在 `config.py` 中配置好它们的路径**。

1.  **歌曲向量文件**: `song_vectors.csv` (示例名)
2.  **歌单信息文件**: `gen_playlist_info.csv` (示例名)
3.  **歌单-歌曲关系文件**: `gen_playlist_song.csv` (示例名)
4.  **歌曲元信息文件**: `gen_song_info.csv` (示例名)

#### **关键配置步骤**

打开 `config.py` 文件，找到 `DataConfig` 和 `SongRQVAEConfig` 类，并填入您数据文件的**绝对路径**。

## 🚀 执行步骤

请在项目根目录（`generate_playlist`）下，严格按照以下顺序执行脚本。

### **阶段1: 生成歌曲语义ID**
此阶段有两个方法可选，您只需执行其一。

*   **方法A: RQ-VAE (基于神经网络，效果可能更好但慢)**:
    ```bash
    python train_rqvae.py
    # 评估
    python evaluate_rqvae.py
    ```
*   **方法B: RQ-KMeans (基于聚类，速度快)**:
    ```bash
    python train_rq_kmeans.py
    # 评估
    python evaluate_rq_kmeans.py
    ```
*   **决策**: 比较两种方法 `evaluate` 脚本输出的 `Average Cosine Similarity`，选择一个效果更好的方案继续下一步。

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