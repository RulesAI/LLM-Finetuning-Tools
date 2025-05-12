# LLM增强型供应链技术文档处理系统

这是基于大型语言模型(LLM)的供应链技术文档处理系统，用于处理PDF文档并生成高质量问答对，适用于模型微调训练。

## 主要特点

- 完全可配置的模型选择，支持不同环节使用不同的LLM
- 基于LLM的文档结构化处理和清洗
- 智能语义分段，确保内容完整性
- 基于理解的问答对生成，不局限于模板
- 智能问答对修复与增强
- LLM质量评估，筛选出适合训练的高质量问答对
- 自动化流程，支持断点续传和跳过步骤

## 系统架构

系统由以下主要组件构成：

1. **模型调用工具** (`model_utils.py`) - 统一的模型调用接口，支持OpenAI和Anthropic等多种模型
2. **文档清洗模块** (`enhanced_cleaner_llm.py`) - 使用LLM进行结构化处理和内容清洗
3. **语义分段模块** (`semantic_segmentation_llm.py`) - 智能分段保持语义完整性
4. **问答生成模块** (`qa_generation_llm.py`) - 提取主题并生成多样化问答对
5. **问答增强模块** (`qa_enhancement_llm.py`) - 提升问答质量，修复问题
6. **质量评估模块** (`qa_evaluation_llm.py`) - 评估问答对质量并筛选
7. **主流程脚本** (`llm_pipeline.py`) - 协调整个处理流程

## 安装依赖

```bash
pip install -r requirements.txt
```

## 配置模型

在使用前需要配置 `models_config.yaml` 文件，设置API密钥和选择使用的模型：

```yaml
# models_config.yaml
models:
  document_processing: "claude-3-opus"  # 文档清洗
  topic_extraction: "gpt-4o"           # 主题提取
  qa_generation: "gpt-4o"              # 问答生成
  qa_enhancement: "claude-3-sonnet"    # 问答修复
  relevance_evaluation: "gpt-3.5-turbo" # 相关性评估
  semantic_segmentation: "claude-3-opus" # 语义分段

api_keys:
  openai: "YOUR_OPENAI_API_KEY"
  anthropic: "YOUR_ANTHROPIC_API_KEY"

model_parameters:
  temperature: 0.3
  max_tokens: 1000
```

## 使用方法

### 运行全部流程

```bash
python llm_pipeline.py --pdf 你的文档.pdf --output_dir output
```

### 跳过特定步骤

```bash
python llm_pipeline.py --pdf 你的文档.pdf --output_dir output --skip_steps clean enhance
```

可选的跳过步骤包括：`clean`, `segment`, `qa_gen`, `enhance`, `evaluate`

### 单独运行各个模块

**1. 文档清洗：**
```bash
python enhanced_cleaner_llm.py --input 输入文件.txt --output 输出文件.txt
```

**2. 语义分段：**
```bash
python semantic_segmentation_llm.py --input 输入文件.txt --output 分段结果.json
```

**3. 问答生成：**
```bash
python qa_generation_llm.py --input 分段结果.json --output 问答对.json
```

**4. 问答增强：**
```bash
python qa_enhancement_llm.py --input 问答对.json --output 增强问答对.json
```

**5. 质量评估：**
```bash
python qa_evaluation_llm.py --input 增强问答对.json --output 高质量问答对.json
```

## 输出文件

全流程执行后，将在输出目录中生成以下文件：

- `processed_content.txt` - PDF提取的原始文本
- `enhanced_content.txt` - 增强清洗后的文本
- `segmented_content.json` - 分段后的内容
- `qa_pairs.json` - 生成的原始问答对
- `enhanced_qa_pairs.json` - 增强后的问答对
- `high_quality_qa.json` - 最终高质量问答对 (JSON格式)
- `high_quality_qa.jsonl` - 最终高质量问答对 (ChatGLM格式)
- `high_quality_qa_evaluation.json` - 质量评估详情

## 模型选择建议

- **文档清洗与分段**：推荐使用Claude 3 Opus，其对文档结构和长文本的理解能力强
- **问答生成**：推荐使用GPT-4o，创造力和多样性更强
- **问答增强**：推荐使用Claude 3 Sonnet，语言流畅自然
- **质量评估**：可使用更轻量级的GPT-3.5-Turbo，降低成本

## 注意事项

1. 确保已正确设置API密钥并安装相关依赖
2. 处理大型PDF时，可能需要较长时间，建议使用 `--skip_steps` 参数分步骤运行
3. 如遇模型API错误，系统会自动重试，但频繁失败可能是API限制或密钥问题
4. 所有日志会保存在各个步骤的日志文件中，方便排查问题 