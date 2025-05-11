import os
import sys
import json
import logging
import argparse
import subprocess
from pathlib import Path
import shutil
import tempfile

# 检查必要依赖库
def check_dependencies():
    """检查必要的依赖库是否已安装"""
    required_libraries = [
        "langchain_community", 
        "langchain_text_splitters", 
        "tiktoken", 
        "unstructured"
    ]
    
    missing_libraries = []
    for lib in required_libraries:
        try:
            __import__(lib)
        except ImportError:
            missing_libraries.append(lib)
    
    if missing_libraries:
        print(f"错误: 缺少必要的依赖库: {', '.join(missing_libraries)}")
        print("请使用以下命令安装依赖:")
        print(f"pip install {' '.join(missing_libraries)}")
        return False
    
    return True

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pipeline.log"),
        logging.StreamHandler()
    ]
)

class Pipeline:
    def __init__(self, pdf_path=None, output_dir="output", skip_existing=False):
        self.pdf_path = pdf_path
        self.output_dir = Path(output_dir)
        self.skip_existing = skip_existing
        
        # 临时文件和链接列表，用于后续清理
        self.temp_files = []
        self.temp_links = []
        
        # 确保输出目录存在
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 各处理阶段的输出文件
        self.processed_content_file = self.output_dir / "processed_content.txt"
        self.enhanced_content_file = self.output_dir / "enhanced_content.txt"
        self.segmented_content_file = self.output_dir / "segmented_content.json"
        self.qa_pairs_file = self.output_dir / "qa_instructions_robust.json"
        self.qa_pairs_chatglm_file = self.output_dir / "qa_instructions_chatglm_robust.jsonl"
        self.fixed_qa_pairs_file = self.output_dir / "qa_instructions_fixed_improved.json"
        self.fixed_qa_pairs_chatglm_file = self.output_dir / "qa_instructions_chatglm_fixed_improved.jsonl"
        self.high_quality_qa_file = self.output_dir / "high_quality_qa.json"
        self.high_quality_qa_chatglm_file = self.output_dir / "high_quality_qa.jsonl"
        
        # 各阶段处理脚本
        self.scripts = {
            "extract": "tunning.py",
            "enhance": "enhanced_cleaner.py",
            "segment": "segment.py",
            "generate_qa": "generate_qa_pairs_improved.py",
            "fix_qa": "fix_qa_pairs_improved.py",
            "quality_check": "ensure_high_quality.py"
        }
        
        # 检查脚本是否存在
        for name, script in self.scripts.items():
            script_path = Path(script)
            if not script_path.exists():
                logging.error(f"缺少必要的脚本文件: {script}")
                raise FileNotFoundError(f"缺少必要的脚本文件: {script}")
    
    def __del__(self):
        """析构函数，清理临时文件和链接"""
        self.cleanup()
    
    def cleanup(self):
        """清理临时文件和链接"""
        # 移除临时创建的符号链接
        for link_path in self.temp_links:
            try:
                if Path(link_path).exists():
                    if Path(link_path).is_symlink():
                        Path(link_path).unlink()
                    else:
                        # 如果不是符号链接但是文件，也删除
                        Path(link_path).unlink()
                    logging.info(f"已移除临时链接: {link_path}")
            except Exception as e:
                logging.warning(f"移除临时链接时出错: {str(e)}")
        
        # 清空临时链接列表
        self.temp_links = []
        
        # 移除临时文件
        for temp_file in self.temp_files:
            try:
                if Path(temp_file).exists():
                    Path(temp_file).unlink()
                    logging.info(f"已移除临时文件: {temp_file}")
            except Exception as e:
                logging.warning(f"移除临时文件时出错: {str(e)}")
        
        # 清空临时文件列表
        self.temp_files = []
    
    def extract_pdf_content(self):
        """从PDF提取内容"""
        if self.skip_existing and self.processed_content_file.exists():
            logging.info(f"跳过PDF内容提取，使用现有文件: {self.processed_content_file}")
            return True
        
        if not self.pdf_path:
            logging.error("未提供PDF文件路径")
            return False
        
        try:
            cmd = [sys.executable, str(Path(self.scripts["extract"])), "--pdf", self.pdf_path, "--output", str(self.processed_content_file)]
            logging.info(f"执行命令: {' '.join(cmd)}")
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logging.error(f"PDF内容提取失败: {result.stderr}")
                return False
            
            logging.info(f"PDF内容提取成功，结果保存到: {self.processed_content_file}")
            return True
        
        except Exception as e:
            logging.error(f"执行PDF内容提取时出错: {str(e)}")
            return False
    
    def enhance_content(self):
        """增强清洗文本内容"""
        if self.skip_existing and self.enhanced_content_file.exists():
            logging.info(f"跳过增强清洗，使用现有文件: {self.enhanced_content_file}")
            return True
        
        if not self.processed_content_file.exists():
            logging.error(f"找不到PDF处理后的内容文件: {self.processed_content_file}")
            return False
        
        try:
            cmd = [
                sys.executable, 
                str(Path(self.scripts["enhance"])), 
                "--input", str(self.processed_content_file), 
                "--output", str(self.enhanced_content_file)
            ]
            logging.info(f"执行命令: {' '.join(cmd)}")
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logging.error(f"增强清洗失败: {result.stderr}")
                return False
            
            logging.info(f"增强清洗成功，结果保存到: {self.enhanced_content_file}")
            return True
        
        except Exception as e:
            logging.error(f"执行增强清洗时出错: {str(e)}")
            return False
    
    def segment_content(self):
        """对文本内容进行语义分段"""
        if self.skip_existing and self.segmented_content_file.exists():
            logging.info(f"跳过内容分段，使用现有文件: {self.segmented_content_file}")
            return True
        
        # 使用增强清洗后的文件进行分段，如果存在的话
        input_file = self.enhanced_content_file if self.enhanced_content_file.exists() else self.processed_content_file
        
        if not input_file.exists():
            logging.error(f"找不到处理后的内容文件: {input_file}")
            return False
        
        try:
            cmd = [
                sys.executable, 
                str(Path(self.scripts["segment"])), 
                "--input", str(input_file), 
                "--output", str(self.segmented_content_file)
            ]
            logging.info(f"执行命令: {' '.join(cmd)}")
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logging.error(f"内容分段失败: {result.stderr}")
                return False
            
            logging.info(f"内容分段成功，结果保存到: {self.segmented_content_file}")
            return True
        
        except Exception as e:
            logging.error(f"执行内容分段时出错: {str(e)}")
            return False
    
    def generate_qa_pairs(self):
        """生成问答对"""
        if self.skip_existing and self.qa_pairs_file.exists() and self.qa_pairs_chatglm_file.exists():
            logging.info(f"跳过问答对生成，使用现有文件")
            return True
        
        if not self.segmented_content_file.exists():
            logging.error(f"找不到分段内容文件: {self.segmented_content_file}")
            return False
        
        try:
            # 使用命令行参数直接指定输入文件和输出目录
            cmd = [
                sys.executable, 
                str(Path(self.scripts["generate_qa"])), 
                "--input", str(self.segmented_content_file),
                "--output_dir", str(self.output_dir)
            ]
            
            logging.info(f"执行命令: {' '.join(cmd)}")
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logging.error(f"问答对生成失败: {result.stderr}")
                return False
            
            # 验证输出文件是否存在
            if not self.qa_pairs_file.exists():
                logging.warning(f"未找到生成的JSON文件: {self.qa_pairs_file}")
                # 尝试在当前目录查找
                src_file = Path("qa_instructions_robust.json")
                if src_file.exists():
                    try:
                        shutil.copy2(src_file, self.qa_pairs_file)
                        logging.info(f"从当前目录复制JSON文件到输出目录: {self.qa_pairs_file}")
                        # 添加到临时文件列表以便后续清理
                        self.temp_files.append(src_file)
                    except Exception as copy_error:
                        logging.error(f"复制文件失败: {str(copy_error)}")
                        return False
            
            if not self.qa_pairs_chatglm_file.exists():
                logging.warning(f"未找到生成的JSONL文件: {self.qa_pairs_chatglm_file}")
                # 尝试在当前目录查找
                src_file = Path("qa_instructions_chatglm_robust.jsonl")
                if src_file.exists():
                    try:
                        shutil.copy2(src_file, self.qa_pairs_chatglm_file)
                        logging.info(f"从当前目录复制JSONL文件到输出目录: {self.qa_pairs_chatglm_file}")
                        # 添加到临时文件列表以便后续清理
                        self.temp_files.append(src_file)
                    except Exception as copy_error:
                        logging.error(f"复制文件失败: {str(copy_error)}")
                        return False
            
            # 最终验证文件是否存在
            if not self.qa_pairs_file.exists() or not self.qa_pairs_chatglm_file.exists():
                logging.error(f"无法找到生成的问答对文件")
                return False
            
            logging.info(f"问答对生成成功，结果保存到输出目录")
            return True
        
        except Exception as e:
            logging.error(f"执行问答对生成时出错: {str(e)}")
            return False
    
    def fix_qa_pairs(self):
        """修复和改进问答对"""
        if self.skip_existing and self.fixed_qa_pairs_file.exists() and self.fixed_qa_pairs_chatglm_file.exists():
            logging.info(f"跳过问答对修复，使用现有文件")
            return True
        
        if not self.qa_pairs_file.exists() or not self.qa_pairs_chatglm_file.exists():
            logging.error(f"找不到问答对文件")
            return False
        
        try:
            # 使用命令行参数指定输入文件和输出目录
            cmd = [
                sys.executable, 
                str(Path(self.scripts["fix_qa"])), 
                "--input_json", str(self.qa_pairs_file),
                "--input_jsonl", str(self.qa_pairs_chatglm_file),
                "--output_dir", str(self.output_dir)
            ]
            
            logging.info(f"执行命令: {' '.join(cmd)}")
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logging.error(f"问答对修复失败: {result.stderr}")
                return False
            
            # 确定期望的输出文件名
            expected_json = self.output_dir / "qa_instructions_robust_improved.json"
            expected_jsonl = self.output_dir / "qa_instructions_chatglm_robust_improved.jsonl"
            
            # 创建从实际文件到期望文件名的链接
            if expected_json.exists() and not self.fixed_qa_pairs_file.exists():
                try:
                    # 在output目录中创建符号链接
                    os.chdir(self.output_dir)
                    os.symlink(expected_json.name, self.fixed_qa_pairs_file.name)
                    os.chdir("..")  # 返回原目录
                    logging.info(f"创建符号链接: {self.fixed_qa_pairs_file}")
                except Exception as e:
                    logging.warning(f"创建符号链接失败: {str(e)}，尝试复制文件")
                    try:
                        shutil.copy2(expected_json, self.fixed_qa_pairs_file)
                        logging.info(f"复制文件: {expected_json} 到 {self.fixed_qa_pairs_file}")
                    except Exception as copy_error:
                        logging.error(f"复制文件失败: {str(copy_error)}")
                        return False
            
            if expected_jsonl.exists() and not self.fixed_qa_pairs_chatglm_file.exists():
                try:
                    # 在output目录中创建符号链接
                    os.chdir(self.output_dir)
                    os.symlink(expected_jsonl.name, self.fixed_qa_pairs_chatglm_file.name)
                    os.chdir("..")  # 返回原目录
                    logging.info(f"创建符号链接: {self.fixed_qa_pairs_chatglm_file}")
                except Exception as e:
                    logging.warning(f"创建符号链接失败: {str(e)}，尝试复制文件")
                    try:
                        shutil.copy2(expected_jsonl, self.fixed_qa_pairs_chatglm_file)
                        logging.info(f"复制文件: {expected_jsonl} 到 {self.fixed_qa_pairs_chatglm_file}")
                    except Exception as copy_error:
                        logging.error(f"复制文件失败: {str(copy_error)}")
                        return False
            
            # 最终验证文件是否存在
            if not self.fixed_qa_pairs_file.exists() or not self.fixed_qa_pairs_chatglm_file.exists():
                logging.error(f"无法找到修复后的问答对文件")
                return False
            
            logging.info(f"问答对修复成功，结果保存到输出目录")
            return True
        
        except Exception as e:
            logging.error(f"执行问答对修复时出错: {str(e)}")
            return False
    
    def quality_check(self):
        """质量检查和高质量问答对提取"""
        if self.skip_existing and self.high_quality_qa_file.exists() and self.high_quality_qa_chatglm_file.exists():
            logging.info(f"跳过质量检查，使用现有文件")
            return True
        
        if not self.fixed_qa_pairs_file.exists() or not self.fixed_qa_pairs_chatglm_file.exists():
            logging.error(f"找不到修复后的问答对文件")
            return False
        
        try:
            # 创建临时目录用于质量检查脚本需要的输入文件
            temp_dir = Path(tempfile.mkdtemp(prefix="qa_quality_"))
            self.temp_files.append(temp_dir)  # 添加到临时文件列表以便后续清理
            
            # 准备质量检查所需的输入文件
            temp_json = temp_dir / "qa_instructions_fixed_improved.json"
            temp_jsonl = temp_dir / "qa_instructions_chatglm_fixed_improved.jsonl"
            
            # 复制文件到临时目录
            shutil.copy2(self.fixed_qa_pairs_file, temp_json)
            shutil.copy2(self.fixed_qa_pairs_chatglm_file, temp_jsonl)
            
            # 创建从临时目录到当前目录的符号链接或复制
            target_json = Path("qa_instructions_fixed_improved.json")
            target_jsonl = Path("qa_instructions_chatglm_fixed_improved.jsonl")
            
            try:
                # 尝试创建符号链接（macOS支持）
                if target_json.exists():
                    target_json.unlink()
                if target_jsonl.exists():
                    target_jsonl.unlink()
                
                target_json.symlink_to(temp_json)
                target_jsonl.symlink_to(temp_jsonl)
                
                # 记录创建的符号链接
                self.temp_links.append(target_json)
                self.temp_links.append(target_jsonl)
                
                logging.info(f"创建输入文件符号链接成功")
            except Exception as link_error:
                logging.warning(f"创建符号链接失败: {str(link_error)}，使用复制方式")
                # 如果符号链接失败，使用复制
                shutil.copy2(temp_json, target_json)
                shutil.copy2(temp_jsonl, target_jsonl)
                # 记录创建的临时文件
                self.temp_files.append(target_json)
                self.temp_files.append(target_jsonl)
            
            # 执行质量检查脚本
            cmd = [
                sys.executable, 
                str(Path(self.scripts["quality_check"])), 
                "--output_dir", str(self.output_dir)  # 假设quality_check脚本支持此参数
            ]
            
            logging.info(f"执行命令: {' '.join(cmd)}")
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logging.error(f"质量检查失败: {result.stderr}")
                return False
            
            # 移动生成的高质量问答对文件到输出目录
            for src_path, dst_path in [
                (Path("high_quality_qa.json"), self.high_quality_qa_file),
                (Path("high_quality_qa.jsonl"), self.high_quality_qa_chatglm_file)
            ]:
                if src_path.exists():
                    try:
                        if dst_path.exists():
                            dst_path.unlink()
                        shutil.copy2(src_path, dst_path)
                        logging.info(f"复制高质量问答对文件到: {dst_path}")
                        # 记录创建的临时文件
                        self.temp_files.append(src_path)
                    except Exception as copy_error:
                        logging.error(f"复制文件失败: {str(copy_error)}")
            
            # 移动质量报告到输出目录
            for report_file in Path('.').glob('quality_report_*.json'):
                dst_report = self.output_dir / report_file.name
                try:
                    if dst_report.exists():
                        dst_report.unlink()
                    shutil.copy2(report_file, dst_report)
                    logging.info(f"复制质量报告到: {dst_report}")
                    # 记录创建的临时文件
                    self.temp_files.append(report_file)
                except Exception as copy_error:
                    logging.error(f"复制质量报告失败: {str(copy_error)}")
            
            logging.info(f"质量检查成功，结果保存到输出目录")
            return True
        
        except Exception as e:
            logging.error(f"执行质量检查时出错: {str(e)}")
            return False
        finally:
            # 清理临时文件和链接
            self.cleanup()
    
    def run(self):
        """运行完整的处理流程"""
        logging.info("开始执行完整处理流程...")
        
        # 步骤1: 提取PDF内容
        if self.pdf_path:
            if not self.extract_pdf_content():
                logging.error("PDF内容提取失败，终止流程")
                return False
        else:
            logging.warning("未提供PDF路径，跳过提取步骤")
        
        # 步骤2: 增强清洗
        if not self.enhance_content():
            logging.error("增强清洗失败，终止流程")
            return False
        
        # 步骤3: 语义分段
        if not self.segment_content():
            logging.error("语义分段失败，终止流程")
            return False
        
        # 步骤4: 生成问答对
        if not self.generate_qa_pairs():
            logging.error("问答对生成失败，终止流程")
            return False
        
        # 步骤5: 修复问答对
        if not self.fix_qa_pairs():
            logging.error("问答对修复失败，终止流程")
            return False
        
        # 步骤6: 质量检查和高质量数据提取
        if not self.quality_check():
            logging.error("质量检查失败，终止流程")
            return False
        
        logging.info("完整处理流程执行成功!")
        
        # 显示处理结果
        self.report_stats()
        
        return True
    
    def report_stats(self):
        """报告处理结果统计"""
        try:
            stats = {
                "原始PDF内容": self.processed_content_file.exists(),
                "增强清洗文本": self.enhanced_content_file.exists(),
                "语义分段数": 0,
                "生成问答对数": 0,
                "修复后问答对数": 0,
                "高质量问答对数": 0
            }
            
            # 统计分段数
            if self.segmented_content_file.exists():
                with open(self.segmented_content_file, 'r', encoding='utf-8') as f:
                    segments = json.load(f)
                    stats["语义分段数"] = len(segments)
            
            # 统计问答对数
            if self.qa_pairs_file.exists():
                with open(self.qa_pairs_file, 'r', encoding='utf-8') as f:
                    qa_pairs = json.load(f)
                    stats["生成问答对数"] = len(qa_pairs)
            
            # 统计修复后问答对数
            if self.fixed_qa_pairs_file.exists():
                with open(self.fixed_qa_pairs_file, 'r', encoding='utf-8') as f:
                    fixed_qa_pairs = json.load(f)
                    stats["修复后问答对数"] = len(fixed_qa_pairs)
            
            # 统计高质量问答对数
            if self.high_quality_qa_file.exists():
                with open(self.high_quality_qa_file, 'r', encoding='utf-8') as f:
                    high_quality_qa = json.load(f)
                    stats["高质量问答对数"] = len(high_quality_qa)
            
            # 打印统计结果
            logging.info("处理结果统计:")
            for key, value in stats.items():
                logging.info(f"  - {key}: {value}")
            
            # 如果有质量报告，显示摘要
            quality_reports = list(self.output_dir.glob('quality_report_*.json'))
            if quality_reports:
                logging.info("质量报告摘要:")
                for report_file in quality_reports:
                    try:
                        with open(report_file, 'r', encoding='utf-8') as f:
                            report = json.load(f)
                            filename = report.get("filename", "未知文件")
                            avg_score = report.get("average_score", 0)
                            total = report.get("total_pairs", 0)
                            excellent = report.get("score_distribution", {}).get("excellent", 0)
                            # 显示常见问题统计
                            issues = report.get("common_issues", {})
                            mismatch = issues.get("question_answer_mismatch", 0)
                            topic_mismatch = issues.get("topic_mismatch", 0)
                            logging.info(f"  - {Path(filename).name}: 平均分数 {avg_score}, 共 {total} 对, 优秀 {excellent} 对")
                            logging.info(f"    问题与答案不匹配: {mismatch} 对, 主题不匹配: {topic_mismatch} 对")
                    except:
                        continue
            
            # 如果有清洗报告，显示摘要
            cleaning_report = self.output_dir / "cleaning_report.json"
            if cleaning_report.exists():
                try:
                    with open(cleaning_report, 'r', encoding='utf-8') as f:
                        report = json.load(f)
                        stats = report.get("statistics", {})
                        logging.info("清洗报告摘要:")
                        logging.info(f"  - 识别标题数: {stats.get('identified_titles', 0)}")
                        logging.info(f"  - 识别图表数: {stats.get('identified_figures', 0)}")
                        logging.info(f"  - 列表项数: {stats.get('list_sections', 0)}")
                        logging.info(f"  - 标准化术语数: {len(stats.get('standardized_terms', {}))}")
                        logging.info(f"  - 移除引用网址数: {stats.get('removed_references', 0)}")
                except:
                    pass
                    
            # 添加生成后的问题与答案质量评估分析
            if self.high_quality_qa_file.exists():
                logging.info("问答对主题分布:")
                try:
                    with open(self.high_quality_qa_file, 'r', encoding='utf-8') as f:
                        high_quality_qa = json.load(f)
                        
                        # 分析主题分布
                        topic_counter = {}
                        for qa in high_quality_qa:
                            topic = qa.get("topic", "未知主题")
                            topic_counter[topic] = topic_counter.get(topic, 0) + 1
                        
                        # 排序并显示主题分布
                        sorted_topics = sorted(topic_counter.items(), key=lambda x: x[1], reverse=True)
                        for topic, count in sorted_topics[:10]:  # 显示前10个最常见主题
                            logging.info(f"  - {topic}: {count} 对")
                            
                        # 分析维度分布
                        dimension_counter = {}
                        for qa in high_quality_qa:
                            dim = qa.get("dimension", "未知维度")
                            dimension_counter[dim] = dimension_counter.get(dim, 0) + 1
                        
                        logging.info("问答对维度分布:")
                        for dim, count in dimension_counter.items():
                            logging.info(f"  - {dim}: {count} 对")
                except:
                    pass
        
        except Exception as e:
            logging.error(f"生成统计报告时出错: {str(e)}")

def main():
    # 检查依赖库是否已安装
    if not check_dependencies():
        return 1
    
    parser = argparse.ArgumentParser(description="供应链技术PDF处理自动化流程")
    parser.add_argument("--pdf", help="输入PDF文件路径")
    parser.add_argument("--output", default="output", help="输出目录，默认为'output'")
    parser.add_argument("--skip-existing", action="store_true", help="如果输出文件已存在，跳过相应步骤")
    
    args = parser.parse_args()
    
    if not args.pdf and not Path('processed_content.txt').exists():
        logging.error("必须提供PDF文件路径，或者当前目录已存在processed_content.txt文件")
        return 1
    
    pipeline = Pipeline(
        pdf_path=args.pdf,
        output_dir=args.output,
        skip_existing=args.skip_existing
    )
    
    try:
        if pipeline.run():
            logging.info(f"处理结果已保存到目录: {args.output}")
            return 0
        else:
            logging.error("处理流程未完成")
            return 1
    finally:
        # 确保清理临时文件和链接
        pipeline.cleanup()

if __name__ == "__main__":
    sys.exit(main()) 