import argparse
import logging
import os
import subprocess
import json
import sys

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pipeline.log"),
        logging.StreamHandler()
    ]
)

def run_command(command, description):
    """运行命令并处理结果"""
    logging.info(f"正在{description}...")
    try:
        result = subprocess.run(command, check=True, text=True, capture_output=True)
        logging.info(f"{description}完成")
        logging.debug(f"命令输出: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"{description}失败: {e}")
        logging.error(f"错误输出: {e.stderr}")
        return False

def ensure_directory(path):
    """确保目录存在"""
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
        logging.info(f"创建目录: {directory}")

def main():
    # 命令行参数解析
    parser = argparse.ArgumentParser(description='LLM增强的供应链文档处理流水线')
    parser.add_argument('--pdf', required=True, help='PDF文件路径')
    parser.add_argument('--output_dir', default='output', help='输出目录')
    parser.add_argument('--config', default='models_config.yaml', help='模型配置文件')
    parser.add_argument('--skip_steps', nargs='+', choices=['clean', 'segment', 'qa_gen', 'enhance', 'evaluate'],
                      help='跳过指定步骤')
    args = parser.parse_args()
    
    # 检查PDF文件是否存在
    if not os.path.exists(args.pdf):
        logging.error(f"PDF文件不存在: {args.pdf}")
        return False
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 配置跳过步骤
    skip_steps = args.skip_steps or []
    
    # 定义文件路径
    processed_content_path = os.path.join(args.output_dir, "processed_content.txt")
    enhanced_content_path = os.path.join(args.output_dir, "enhanced_content.txt")
    segmented_content_path = os.path.join(args.output_dir, "segmented_content.json")
    qa_pairs_path = os.path.join(args.output_dir, "qa_pairs.json")
    enhanced_qa_path = os.path.join(args.output_dir, "enhanced_qa_pairs.json")
    high_quality_qa_path = os.path.join(args.output_dir, "high_quality_qa.json")
    
    # 步骤1: PDF文本提取 (使用原有的tunning.py)
    if not os.path.exists(processed_content_path):
        pdf_extraction_command = [
            sys.executable, "tunning.py",
            "--pdf", args.pdf,
            "--output", processed_content_path
        ]
        if not run_command(pdf_extraction_command, "提取PDF文本"):
            return False
    else:
        logging.info(f"使用已存在的处理文本: {processed_content_path}")
    
    # 步骤2: 增强型文档清洗
    if 'clean' not in skip_steps:
        if not os.path.exists(enhanced_content_path):
            doc_cleaning_command = [
                sys.executable, "enhanced_cleaner_llm.py",
                "--input", processed_content_path,
                "--output", enhanced_content_path,
                "--config", args.config
            ]
            if not run_command(doc_cleaning_command, "增强文档清洗"):
                # 如果失败，使用原始处理内容
                logging.warning("增强清洗失败，将使用原始处理内容继续")
                with open(processed_content_path, 'r') as src:
                    with open(enhanced_content_path, 'w') as dst:
                        dst.write(src.read())
        else:
            logging.info(f"使用已存在的增强内容: {enhanced_content_path}")
    else:
        # 如果跳过清洗，使用原始处理内容
        logging.info("跳过清洗步骤，使用原始处理内容")
        if not os.path.exists(enhanced_content_path):
            with open(processed_content_path, 'r') as src:
                with open(enhanced_content_path, 'w') as dst:
                    dst.write(src.read())
    
    # 步骤3: 语义分段
    if 'segment' not in skip_steps:
        if not os.path.exists(segmented_content_path):
            segmentation_command = [
                sys.executable, "semantic_segmentation_llm.py",
                "--input", enhanced_content_path,
                "--output", segmented_content_path,
                "--config", args.config
            ]
            if not run_command(segmentation_command, "语义分段"):
                return False
        else:
            logging.info(f"使用已存在的分段内容: {segmented_content_path}")
    else:
        logging.info("跳过分段步骤")
        if not os.path.exists(segmented_content_path):
            logging.error(f"缺少分段内容文件: {segmented_content_path}")
            return False
    
    # 步骤4: 问答对生成
    if 'qa_gen' not in skip_steps:
        if not os.path.exists(qa_pairs_path):
            qa_generation_command = [
                sys.executable, "qa_generation_llm.py",
                "--input", segmented_content_path,
                "--output", qa_pairs_path,
                "--config", args.config
            ]
            if not run_command(qa_generation_command, "生成问答对"):
                return False
        else:
            logging.info(f"使用已存在的问答对: {qa_pairs_path}")
    else:
        logging.info("跳过问答生成步骤")
        if not os.path.exists(qa_pairs_path):
            logging.error(f"缺少问答对文件: {qa_pairs_path}")
            return False
    
    # 步骤5: 问答对增强
    if 'enhance' not in skip_steps:
        if not os.path.exists(enhanced_qa_path):
            qa_enhancement_command = [
                sys.executable, "qa_enhancement_llm.py",
                "--input", qa_pairs_path,
                "--output", enhanced_qa_path,
                "--config", args.config
            ]
            if not run_command(qa_enhancement_command, "增强问答对"):
                # 如果增强失败，使用原始问答对
                logging.warning("问答对增强失败，将使用原始问答对继续")
                with open(qa_pairs_path, 'r') as src:
                    with open(enhanced_qa_path, 'w') as dst:
                        dst.write(src.read())
        else:
            logging.info(f"使用已存在的增强问答对: {enhanced_qa_path}")
    else:
        # 如果跳过增强，使用原始问答对
        logging.info("跳过问答增强步骤，使用原始问答对")
        if not os.path.exists(enhanced_qa_path):
            with open(qa_pairs_path, 'r') as src:
                with open(enhanced_qa_path, 'w') as dst:
                    dst.write(src.read())
    
    # 步骤6: 质量评估
    if 'evaluate' not in skip_steps:
        if not os.path.exists(high_quality_qa_path):
            qa_evaluation_command = [
                sys.executable, "qa_evaluation_llm.py",
                "--input", enhanced_qa_path,
                "--output", high_quality_qa_path,
                "--config", args.config
            ]
            if not run_command(qa_evaluation_command, "评估问答对质量"):
                return False
        else:
            logging.info(f"使用已存在的高质量问答对: {high_quality_qa_path}")
    else:
        logging.info("跳过质量评估步骤")
        # 如果跳过评估，直接使用增强后的问答对作为最终结果
        if not os.path.exists(high_quality_qa_path):
            try:
                with open(enhanced_qa_path, 'r') as src:
                    with open(high_quality_qa_path, 'w') as dst:
                        dst.write(src.read())
                
                # 转换为JSONL格式
                high_quality_jsonl_path = high_quality_qa_path.replace('.json', '.jsonl')
                with open(enhanced_qa_path, 'r') as f:
                    qa_data = json.load(f)
                
                with open(high_quality_jsonl_path, 'w') as f:
                    for group in qa_data:
                        for qa in group.get('qa_pairs', []):
                            chatglm_item = {
                                "instruction": f"结合材料，回答下面的问题: {qa.get('question', '')}",
                                "input": "",
                                "output": qa.get('answer', '')
                            }
                            f.write(json.dumps(chatglm_item, ensure_ascii=False) + '\n')
            except Exception as e:
                logging.error(f"创建最终结果文件失败: {str(e)}")
                return False
    
    # 处理成功
    logging.info(f"所有处理步骤已完成")
    logging.info(f"最终输出文件位于: {high_quality_qa_path}")
    logging.info(f"         以及JSONL格式: {high_quality_qa_path.replace('.json', '.jsonl')}")
    
    # 输出统计信息
    try:
        with open(high_quality_qa_path, 'r') as f:
            high_quality_qa = json.load(f)
        
        total_groups = len(high_quality_qa)
        total_qa_pairs = sum(len(group.get('qa_pairs', [])) for group in high_quality_qa)
        
        logging.info(f"生成的高质量问答对统计:")
        logging.info(f"  - 主题数量: {total_groups}")
        logging.info(f"  - 问答对总数: {total_qa_pairs}")
        
        # 输出按维度和难度的统计
        dimensions = {}
        difficulties = {}
        
        for group in high_quality_qa:
            for qa in group.get('qa_pairs', []):
                dim = qa.get('dimension', '未知')
                diff = qa.get('difficulty', '未知')
                
                dimensions[dim] = dimensions.get(dim, 0) + 1
                difficulties[diff] = difficulties.get(diff, 0) + 1
        
        logging.info(f"  - 按维度统计:")
        for dim, count in dimensions.items():
            logging.info(f"    * {dim}: {count}对 ({count/total_qa_pairs*100:.1f}%)")
        
        logging.info(f"  - 按难度统计:")
        for diff, count in difficulties.items():
            logging.info(f"    * {diff}: {count}对 ({count/total_qa_pairs*100:.1f}%)")
        
    except Exception as e:
        logging.warning(f"统计信息生成失败: {str(e)}")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        logging.error("流程执行失败，请查看日志获取详细信息")
        sys.exit(1) 