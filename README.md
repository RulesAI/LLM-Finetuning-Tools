# LLM微调数据处理工具

基于大型语言模型(LLM)的文档处理系统，用于处理技术文档并生成高质量问答对，适用于模型微调训练。

## 项目结构

```
.
├── src/                      # 源代码目录
│   ├── core/                 # 核心功能模块
│   │   ├── document/        # 文档处理相关
│   │   ├── qa/              # 问答生成相关
│   │   ├── evaluation/      # 评估相关
│   │   └── utils/           # 工具函数
│   ├── models/              # 模型相关代码
│   └── config/              # 配置文件
│
├── data/                     # 数据目录
│   ├── raw/                 # 原始数据
│   ├── processed/           # 处理后的数据
│   └── output/              # 输出数据
│       ├── enhanced/        # 增强后的问答对
│       ├── split/           # 数据集拆分结果
│       └── evaluation/      # 评估结果
│
├── logs/                     # 日志目录
│   ├── document/            # 文档处理日志
│   ├── qa/                  # 问答生成日志
│   └── evaluation/          # 评估日志
│
├── tests/                    # 测试目录
│   ├── unit/                # 单元测试
│   └── integration/         # 集成测试
│
├── docs/                     # 文档目录
│   ├── api/                 # API文档
│   └── examples/            # 使用示例
│
├── scripts/                  # 脚本目录
│   ├── setup.sh             # 环境设置脚本
│   └── run.sh               # 运行脚本
│
├── requirements.txt          # 依赖文件
└── .gitignore               # Git忽略文件
```

## 主要特点

- 完全可配置的模型选择，支持不同环节使用不同的LLM
- 基于LLM的文档结构化处理和清洗
- 智能语义分段，确保内容完整性
- 基于理解的问答对生成，不局限于模板
- 智能问答对修复与增强
- LLM质量评估，筛选出适合训练的高质量问答对
- 自动化流程，支持断点续传和跳过步骤
- 防过拟合数据拆分和问答增强

## 安装依赖

```bash
pip install -r requirements.txt
```

## 配置模型

在使用前需要配置 `src/config/config.yaml` 文件，设置API密钥和选择使用的模型。

## 使用方法

### 运行全部流程

```bash
bash scripts/run.sh 你的文档.pdf
```

### 跳过特定步骤

```bash
bash scripts/run.sh 你的文档.pdf "--skip_steps clean,enhance"
```

可选的跳过步骤包括：`clean`, `segment`, `qa_gen`, `enhance`, `evaluate`

## 处理流程

1. **文档清洗**：使用LLM处理PDF提取的原始文本，去除噪声
2. **语义分段**：将清洗后的文本按语义分成段落
3. **问答生成**：基于分段内容生成问答对
4. **问答增强**：改写、扩展、缩写问答对，生成变体
5. **质量评估**：筛选高质量问答对
6. **覆盖率评估**：评估问答对对原文的覆盖度
7. **数据集拆分**：将问答对分为训练集、测试集和验证集

## 输出文件

处理完成后，将在data/output目录中生成以下文件：

- `enhanced/enhanced_qa_variants.json` - 增强后的问答对变体
- `enhanced/high_quality_qa.json` - 高质量问答对
- `split/train.jsonl` - 训练集（OpenAI格式）
- `split/test.jsonl` - 测试集
- `split/val.jsonl` - 验证集
- `evaluation/coverage_evaluation_report.txt` - 覆盖度评估报告

## 防过拟合措施

1. 多样化问答对：通过问答增强生成同一知识点的不同表述
2. 合理数据拆分：按主题分层随机拆分数据集
3. 数据增强：生成语义相似但表述不同的变体

## 注意事项

1. 确保已正确设置API密钥并安装相关依赖
2. 处理大型PDF时，可能需要较长时间，建议使用 `--skip_steps` 参数分步骤运行
3. 如遇模型API错误，系统会自动重试，但频繁失败可能是API限制或密钥问题
4. 所有日志会保存在logs目录的相应子目录中 