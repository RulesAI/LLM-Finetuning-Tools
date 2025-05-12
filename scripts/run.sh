#!/bin/bash

# 设置环境变量
export PYTHONPATH=$PYTHONPATH:$(pwd)

# 创建必要的目录
mkdir -p data/{raw,processed,output/{enhanced,split,evaluation}}
mkdir -p logs/{document,qa,evaluation}

# 检查参数
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <pdf_file> [--skip_steps step1,step2,...]"
    exit 1
fi

PDF_FILE=$1
SKIP_STEPS=""
CONFIG_FILE="src/config/models_config.yaml"

# 解析可选参数
if [ "$#" -gt 1 ]; then
    SKIP_STEPS=$2
fi

echo "处理文件: $PDF_FILE"
echo "使用配置: $CONFIG_FILE"
echo "跳过步骤: $SKIP_STEPS"

# 处理PDF文件
if [[ ! $SKIP_STEPS =~ "pdf" ]]; then
    echo "步骤0: 提取PDF文本..."
    python -m src.core.document.pdf_processor --input "$PDF_FILE" --output data/processed/extracted_content.txt
    
    if [ $? -ne 0 ]; then
        echo "PDF处理失败，终止处理"
        exit 1
    fi
fi

# 运行文档处理
if [[ ! $SKIP_STEPS =~ "clean" ]]; then
    echo "步骤1: 运行文档清洗..."
    # 使用提取的原始文本而不是直接从PDF读取
    python -m src.core.document.enhanced_cleaner_llm --input data/processed/extracted_content.txt --output data/processed/cleaned_content.txt --config $CONFIG_FILE
fi

if [[ ! $SKIP_STEPS =~ "segment" ]]; then
    echo "步骤2: 运行语义分段..."
    python -m src.core.document.semantic_segmentation_llm --input data/processed/cleaned_content.txt --output data/processed/segmented_content.json --config $CONFIG_FILE
    
    # 检查JSON是否有效，如果无效则修复
    if [ $? -ne 0 ] || ! python -c "import json; json.load(open('data/processed/segmented_content.json', 'r'))" 2>/dev/null; then
        echo "修复JSON文件..."
        python -m src.core.utils.fix_json --input data/processed/segmented_content.json --output data/processed/segmented_content_fixed.json --config $CONFIG_FILE
        
        # 如果修复成功，使用修复后的文件
        if [ $? -eq 0 ]; then
            mv data/processed/segmented_content_fixed.json data/processed/segmented_content.json
            echo "JSON修复成功"
        else
            echo "警告: JSON修复失败"
        fi
    fi
fi

if [[ ! $SKIP_STEPS =~ "qa_gen" ]]; then
    echo "步骤3: 生成问答对..."
    python -m src.core.qa.qa_generation_llm --input data/processed/segmented_content.json --output data/output/enhanced/qa_pairs.json --config $CONFIG_FILE
fi

if [[ ! $SKIP_STEPS =~ "enhance" ]]; then
    echo "步骤4: 增强问答对..."
    python -m src.core.qa.qa_enhancement_llm --input data/output/enhanced/qa_pairs.json --output data/output/enhanced/enhanced_qa_variants.json --config $CONFIG_FILE
fi

if [[ ! $SKIP_STEPS =~ "evaluate" ]]; then
    echo "步骤5: 评估问答对质量..."
    python -m src.core.evaluation.simple_evaluate --input data/output/enhanced/enhanced_qa_variants.json --output data/output/enhanced/high_quality_qa.json --config $CONFIG_FILE
fi

echo "步骤6: 运行覆盖率评估..."
python -m src.core.evaluation.coverage_evaluation --doc data/processed/cleaned_content.txt --qa data/output/enhanced/high_quality_qa.json --output data/output/evaluation --config $CONFIG_FILE

echo "步骤7: 拆分数据集..."
python -m src.core.qa.split_dataset --input data/output/enhanced/high_quality_qa.json --output data/output/split

echo "处理完成！结果已保存到 data/output 目录" 