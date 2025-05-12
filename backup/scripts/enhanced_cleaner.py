import re
import os
import json
import logging
import argparse
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("enhanced_cleaning.log"),
        logging.StreamHandler()
    ]
)

# 供应链技术相关术语及其缩写映射
TERMINOLOGY_MAP = {
    # 系统类
    "WMS": ["仓库管理系统", "仓储管理系统", "Warehouse Management System"],
    "TMS": ["运输管理系统", "Transportation Management System"],
    "WES": ["仓库执行系统", "Warehouse Execution System"],
    "WCS": ["仓库控制系统", "Warehouse Control System"],
    "ERP": ["企业资源规划", "Enterprise Resource Planning"],
    "MES": ["制造执行系统", "Manufacturing Execution System"],
    "SCM": ["供应链管理", "Supply Chain Management"],
    "RTTVP": ["实时运输可视化平台", "Real-time Transportation Visibility Platform"],
    
    # 技术类
    "AI": ["人工智能", "Artificial Intelligence", "AI技术", "人工智能技术"],
    "IoT": ["物联网", "Internet of Things", "IoT技术", "物联网技术"],
    "ML": ["机器学习", "Machine Learning"],
    "RFID": ["射频识别", "Radio Frequency Identification"],
    "GAN": ["生成对抗网络", "Generative Adversarial Network"],
    "数字孪生": ["Digital Twin", "digital twin技术"],
    "区块链": ["Blockchain", "blockchain技术"],
    "大数据": ["Big Data"],
    "自动驾驶": ["Autonomous Driving", "无人驾驶"],
    "机器人编排": ["Robot Orchestration", "RaaS"],
    
    # 业务概念
    "供应链融合": ["Supply Chain Convergence", "供应链整合"],
    "最后一公里配送": ["Last Mile Delivery"],
    "库存优化": ["Inventory Optimization"]
}

# 标题模式匹配
TITLE_PATTERNS = [
    r'^[\d\.]+\s+[\u4e00-\u9fa5a-zA-Z\s]{3,30}$',  # 数字+点+标题文本
    r'^[一二三四五六七八九十]+[、.\s][\u4e00-\u9fa5a-zA-Z\s]{3,30}$',  # 中文数字+顿号/点+标题文本
    r'^第[一二三四五六七八九十]+[章节]\s*[\u4e00-\u9fa5a-zA-Z\s]{3,30}$',  # 第X章/节+标题文本
    r'^[\u4e00-\u9fa5a-zA-Z\s]{2,20}[:：]$'  # 短文本+冒号
]

# 图表模式匹配
FIGURE_PATTERNS = [
    r'(图\s*\d+[\.:：][\u4e00-\u9fa5a-zA-Z\s]+)',
    r'(表\s*\d+[\.:：][\u4e00-\u9fa5a-zA-Z\s]+)',
    r'(Figure\s*\d+[\.:：][\u4e00-\u9fa5a-zA-Z\s]+)',
    r'(Table\s*\d+[\.:：][\u4e00-\u9fa5a-zA-Z\s]+)'
]

# 列表项模式匹配
LIST_PATTERNS = [
    r'^\d+[\.)]\s+',  # 数字+点/括号+空格
    r'^[a-zA-Z][\.)]\s+',  # 字母+点/括号+空格
    r'^[•\-*]\s+',  # 项目符号+空格
    r'^[\(（]\d+[\)）]\s+'  # 括号中数字+空格
]

# 引用网址模式匹配
REFERENCE_URL_PATTERNS = [
    r'\s+https?://[a-zA-Z0-9\.-]+\.[a-zA-Z]{2,}[^\s]*',  # 常规URL
    r'\s+www\.[a-zA-Z0-9\.-]+\.[a-zA-Z]{2,}[^\s]*',      # www开头
    r'\s+[a-zA-Z0-9\.-]+\.(?:com|org|net|cn|io|ai)[^\s]*',  # 常见顶级域名
    r'\s+\([^)]*?(?:https?://|www\.)[^)]*?\)',           # 括号中包含的URL
    r'\s+（[^）]*?(?:https?://|www\.)[^）]*?）',           # 中文括号中包含的URL
    r'\s+\([^)]*?[a-zA-Z0-9\.-]+\.(?:com|org|net|cn|io|ai)[^)]*?\)',  # 括号中的域名
    r'\s+（[^）]*?[a-zA-Z0-9\.-]+\.(?:com|org|net|cn|io|ai)[^）]*?）',   # 中文括号中的域名
    r'[^\s\.](?:onerail|greyorange|vimaan|softeon|rebus|thescxchange|loginextsolutions|optioryx|go|blog|jascicloud|businesswire|dcvelocity)\.(?:com|io|ai|org)(?:[\s\.,?!;:]|$)',  # 特定引用站点
    r'[^\s\.][a-zA-Z0-9\.-]+\.[a-zA-Z]{2,}[\s\.,?!;:]',  # 域名后跟空格或标点
    r'[^\s\.][a-zA-Z0-9\.-]+\.[a-zA-Z]{2,}$',             # 行尾的域名
    r'(?<=[^\.])\b(?:onerail|greyorange|vimaan|softeon|rebus|thescxchange|loginextsolutions|optioryx|go|blog|jascicloud|businesswire|dcvelocity)\b'  # 独立出现的站点名称
]

class EnhancedCleaner:
    def __init__(self, input_file=None, output_file=None):
        self.input_file = input_file
        self.output_file = output_file
        
        # 文本处理状态
        self.titles = []
        self.figures = []
        self.terminology_instances = {}
        self.list_sections = []
        self.removed_references = 0
        
    def clean(self):
        """执行全面清洗流程"""
        if not self.input_file or not os.path.exists(self.input_file):
            logging.error(f"输入文件不存在: {self.input_file}")
            return False
        
        try:
            # 读取原始文本
            with open(self.input_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 执行清洗步骤
            logging.info(f"开始增强清洗: {self.input_file}")
            
            # 1. 结构化处理
            content = self.identify_structure(content)
            
            # 2. 术语统一处理
            content = self.standardize_terminology(content)
            
            # 3. 重复内容处理
            content = self.remove_redundant_content(content)
            
            # 4. 图表处理
            content = self.process_figures_and_tables(content)
            
            # 5. 列表结构保留
            content = self.preserve_list_structure(content)
            
            # 6. 技术概念关联强化
            content = self.enhance_concept_associations(content)
            
            # 7. 清理引用网址
            content = self.remove_reference_urls(content)
            
            # 8. 最终格式标准化
            content = self.final_formatting(content)
            
            # 保存清洗后的文件
            with open(self.output_file, 'w', encoding='utf-8') as f:
                f.write(content)
            
            # 生成清洗报告
            self.generate_cleaning_report()
            
            logging.info(f"增强清洗完成，结果保存到: {self.output_file}")
            return True
            
        except Exception as e:
            logging.error(f"增强清洗过程出错: {str(e)}")
            return False
    
    def identify_structure(self, content):
        """识别文档结构元素，包括标题、段落、列表等"""
        logging.info("识别文档结构...")
        
        lines = content.split('\n')
        structured_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                structured_lines.append(line)
                continue
                
            # 识别标题
            is_title = False
            for pattern in TITLE_PATTERNS:
                if re.match(pattern, line):
                    structured_lines.append(f"\n<标题>{line}</标题>\n")
                    self.titles.append(line)
                    is_title = True
                    break
            
            if is_title:
                continue
                
            # 识别图表标题
            is_figure = False
            for pattern in FIGURE_PATTERNS:
                match = re.search(pattern, line)
                if match:
                    figure_text = match.group(1)
                    structured_lines.append(f"\n<图表>{figure_text}</图表>\n")
                    self.figures.append(figure_text)
                    is_figure = True
                    break
            
            if is_figure:
                continue
                
            # 识别列表项
            is_list_item = False
            for pattern in LIST_PATTERNS:
                if re.match(pattern, line):
                    # 保持列表项的缩进和结构
                    structured_lines.append(f"<列表项>{line}</列表项>")
                    self.list_sections.append(line)
                    is_list_item = True
                    break
            
            if is_list_item:
                continue
                
            # 普通段落
            structured_lines.append(line)
        
        return '\n'.join(structured_lines)
    
    def standardize_terminology(self, content):
        """统一术语表达，规范化专业术语"""
        logging.info("标准化术语表达...")
        
        standardized_content = content
        
        # 遍历术语映射表
        for standard_term, variations in TERMINOLOGY_MAP.items():
            # 检测各变体在文本中的出现
            term_count = 0
            for term in variations:
                term_count += len(re.findall(r'\b' + re.escape(term) + r'\b', standardized_content))
            
            if term_count > 0:
                self.terminology_instances[standard_term] = term_count
            
            # 如果文中已经对术语进行了定义，保留第一次出现的完整表达
            # 例如: "仓库管理系统(WMS)"出现后，后续可以统一使用WMS
            definition_pattern = fr'([^(（]*?)[（(]({standard_term}|{"|".join(variations)})[)）]'
            match = re.search(definition_pattern, standardized_content)
            
            if match:
                defined_term = match.group(1).strip()
                abbreviation = match.group(2).strip()
                
                # 记录定义的术语
                if standard_term not in self.terminology_instances:
                    self.terminology_instances[standard_term] = 1
                
                # 只替换未定义的术语变体
                for term in variations:
                    if term != defined_term and term != abbreviation:
                        pattern = r'\b' + re.escape(term) + r'\b(?![^(（]*[)）])'
                        if len(abbreviation) <= 5:  # 如果缩写较短，优先使用缩写
                            standardized_content = re.sub(pattern, abbreviation, standardized_content)
                        else:
                            standardized_content = re.sub(pattern, defined_term, standardized_content)
        
        return standardized_content
    
    def remove_redundant_content(self, content):
        """检测并删除重复内容"""
        logging.info("删除重复内容...")
        
        paragraphs = re.split(r'\n\s*\n', content)
        unique_paragraphs = []
        
        # 检测句子级别的重复
        processed_sentences = set()
        redundant_count = 0
        
        for paragraph in paragraphs:
            if not paragraph.strip():
                continue
                
            # 分解段落为句子
            sentences = re.split(r'[。！？\.!?]', paragraph)
            filtered_sentences = []
            
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                    
                # 使用句子的小写形式检查重复
                sentence_key = sentence.lower()
                if len(sentence) > 10 and sentence_key in processed_sentences:
                    redundant_count += 1
                    continue
                
                processed_sentences.add(sentence_key)
                filtered_sentences.append(sentence)
            
            # 重建段落
            if filtered_sentences:
                rebuilt_paragraph = '。'.join(filtered_sentences)
                if not rebuilt_paragraph.endswith(('。', '！', '？', '.', '!', '?')):
                    rebuilt_paragraph += '。'
                unique_paragraphs.append(rebuilt_paragraph)
        
        logging.info(f"已移除 {redundant_count} 个重复句子")
        return '\n\n'.join(unique_paragraphs)
    
    def process_figures_and_tables(self, content):
        """处理图表内容，增强图表说明文本的关联性"""
        logging.info("处理图表内容...")
        
        # 识别图表区域
        figure_pattern = r'<图表>(.*?)</图表>\s*([\s\S]*?)(?=<|$)'
        figure_matches = re.finditer(figure_pattern, content)
        
        for match in figure_matches:
            figure_title = match.group(1)
            following_text = match.group(2).strip()
            
            if not following_text:
                continue
            
            # 为图表说明添加关联标记
            enhanced_text = f'<图表>{figure_title}</图表>\n<图表说明>{following_text}</图表说明>'
            
            # 替换原文本
            content = content.replace(match.group(0), enhanced_text)
        
        return content
    
    def preserve_list_structure(self, content):
        """保留列表结构"""
        logging.info("保留列表结构...")
        
        # 识别连续的列表项
        list_pattern = r'(<列表项>.*?</列表项>\s*){2,}'
        list_matches = re.finditer(list_pattern, content)
        
        for match in list_matches:
            list_section = match.group(0)
            
            # 为整个列表区域添加标记
            enhanced_list = f'\n<列表区域>\n{list_section}\n</列表区域>\n'
            
            # 替换原文本
            content = content.replace(match.group(0), enhanced_list)
        
        return content
    
    def enhance_concept_associations(self, content):
        """增强技术概念之间的关联性"""
        logging.info("增强技术概念关联...")
        
        # 识别技术定义段落
        definition_pattern = r'([^。\n]{0,50}(是指|指的是|被定义为|定义为|technology)[^。\n]{10,200})'
        definitions = re.finditer(definition_pattern, content)
        
        for match in definitions:
            definition_text = match.group(1)
            
            # 检查是否包含已识别的术语
            contains_term = False
            term_name = ""
            
            for term in self.terminology_instances.keys():
                if term in definition_text or any(var in definition_text for var in TERMINOLOGY_MAP.get(term, [])):
                    contains_term = True
                    term_name = term
                    break
            
            if contains_term:
                # 为技术定义添加标记
                enhanced_text = f'<技术定义 term="{term_name}">{definition_text}</技术定义>'
                
                # 替换原文本
                content = content.replace(definition_text, enhanced_text)
        
        return content
    
    def remove_reference_urls(self, content):
        """移除文本中的引用网址"""
        logging.info("清理引用网址...")
        
        cleaned_content = content
        url_count = 0
        
        # 第一轮：移除明显的URL引用
        for pattern in REFERENCE_URL_PATTERNS:
            # 记录原始长度
            original_length = len(cleaned_content)
            
            # 移除URL引用
            cleaned_content = re.sub(pattern, '', cleaned_content)
            
            # 统计移除的URL数量
            if len(cleaned_content) < original_length:
                url_count += 1
        
        # 第二轮：移除常见的域名后缀省略格式，如".com"、".io"等
        domain_suffix_patterns = [
            r'\s+\.\s*(?:com|org|net|cn|io|ai)',  # 单独的.com等
            r'\s+(?:com|org|net|cn|io|ai)\s*\.',  # com.等
            r'(?<=\s)[a-z]{2,}\.(?=[,.\s])',      # 如go.、ai.等
            r'(?<=\s)(?:onerail|greyorange|vimaan|softeon|rebus|thescxchange|loginext|optioryx|go|blog|jasci|businesswire|dcvelocity)',  # 单独的站点名称
            r'onerail(?=[,.\s]|$)',               # 特别处理onerail
            r'\s+[a-zA-Z0-9]{3,}(?=。)',          # 句号前的短英文词组
        ]
        
        for pattern in domain_suffix_patterns:
            original_length = len(cleaned_content)
            cleaned_content = re.sub(pattern, '', cleaned_content)
            if len(cleaned_content) < original_length:
                url_count += 1
        
        # 第三轮：移除残留的引用符号和不完整的域名引用
        cleanup_patterns = [
            r'\s+\([^)]{0,10}\)',          # 移除只包含少量字符的括号
            r'\s+（[^）]{0,10}）',           # 移除只包含少量字符的中文括号
            r'(?<=\s)(?:onerail|greyorange|vimaan|softeon|rebus|thescxchange|loginextsolutions|optioryx|go)(?=[\s.,?!;:])',  # 常见站点名称
            r'(?<=。)[a-z]{2,}(?=。)',      # 两个句号之间的短字母组合
            r'(?<=\s)[a-z]{2,}(?=[,.])',   # 空格后跟短字母组合再跟逗号或句号
        ]
        
        for pattern in cleanup_patterns:
            original_length = len(cleaned_content)
            cleaned_content = re.sub(pattern, '', cleaned_content)
            if len(cleaned_content) < original_length:
                url_count += 1
        
        # 特殊清理：直接移除已知的网站名称（不管出现在什么位置）
        site_names = ['onerail', 'greyorange', 'vimaan', 'softeon', 'rebus', 'thescxchange', 'loginext', 'optioryx']
        for site in site_names:
            if site in cleaned_content:
                cleaned_content = cleaned_content.replace(site, '')
                url_count += 1
        
        # 第四轮：修复清理后可能产生的不规则空格和标点
        cleaned_content = re.sub(r'\s+([,.?!;:])', r'\1', cleaned_content)  # 修复标点前的空格
        cleaned_content = re.sub(r'([,.?!;:])([,.?!;:])', r'\1', cleaned_content)  # 修复重复标点
        cleaned_content = re.sub(r'\s{2,}', ' ', cleaned_content)  # 修复多余空格
        
        self.removed_references = url_count
        logging.info(f"已移除 {url_count} 处引用网址")
        
        return cleaned_content
    
    def final_formatting(self, content):
        """最终格式清理"""
        logging.info("执行最终格式清理...")
        
        # 1. 移除所有XML标记
        clean_content = re.sub(r'<[^>]+>', '', content)
        
        # 2. 规范化空行
        clean_content = re.sub(r'\n{3,}', '\n\n', clean_content)
        
        # 3. 确保段落之间有空行
        clean_content = re.sub(r'([。！？\.!?])\s*\n([^\s])', r'\1\n\n\2', clean_content)
        
        # 4. 规范化标点符号
        clean_content = re.sub(r'([。，；：！？])([^\s])', r'\1 \2', clean_content)
        
        # 5. 规范化空格
        clean_content = re.sub(r'\s+', ' ', clean_content)
        clean_content = re.sub(r' +(?=\n)', '', clean_content)
        
        return clean_content
    
    def generate_cleaning_report(self):
        """生成清洗报告"""
        report_path = Path(self.output_file).parent / 'cleaning_report.json'
        
        report = {
            "input_file": self.input_file,
            "output_file": self.output_file,
            "statistics": {
                "identified_titles": len(self.titles),
                "identified_figures": len(self.figures),
                "list_sections": len(self.list_sections),
                "removed_references": self.removed_references,
                "standardized_terms": self.terminology_instances
            }
        }
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        logging.info(f"清洗报告已保存到: {report_path}")

def main():
    # 添加命令行参数解析
    parser = argparse.ArgumentParser(description='增强型文档清洗工具')
    parser.add_argument('--input', required=True, help='输入文件路径')
    parser.add_argument('--output', help='输出文件路径')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 如果未指定输出文件，则使用默认名称
    if not args.output:
        input_path = Path(args.input)
        output_path = input_path.with_name(f"{input_path.stem}_enhanced{input_path.suffix}")
        args.output = str(output_path)
    
    # 确保输出目录存在
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 创建清洗器并执行清洗
    cleaner = EnhancedCleaner(args.input, args.output)
    if cleaner.clean():
        print(f"增强清洗成功完成，结果已保存到: {args.output}")
        return 0
    else:
        print("增强清洗过程中出现错误，请查看日志获取详情。")
        return 1

if __name__ == "__main__":
    main() 