#!/usr/bin/env python3
import json
import sys

def fix_json_file(input_file, output_file):
    try:
        # 读取文件内容
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 尝试解析JSON
        try:
            data = json.loads(content)
            print(f"文件 {input_file} 已经是有效的JSON格式")
        except json.JSONDecodeError as e:
            print(f"发现JSON格式错误: {str(e)}")
            
            # 根据错误信息，尝试修复常见问题
            # 1. 检查是否有额外的空格或换行符
            # 2. 检查结尾是否有多余的逗号
            # 3. 检查引号是否配对
            
            # 将内容按行拆分
            lines = content.split("\n")
            
            # 删除所有行尾空格
            lines = [line.rstrip() for line in lines]
            
            # 尝试通过正则表达式识别和提取JSON数组
            import re
            array_pattern = r'\[\s*\{.*\}\s*\]'
            match = re.search(array_pattern, content, re.DOTALL)
            
            if match:
                # 提取匹配到的JSON数组
                array_content = match.group(0)
                try:
                    # 尝试解析提取的数组
                    data = json.loads(array_content)
                    print("成功从内容中提取JSON数组")
                    content = array_content
                except json.JSONDecodeError:
                    print("提取的JSON数组仍然无效")
                    
                    # 尝试更激进的修复方式：逐个解析对象
                    print("尝试手动解析段落...")
                    # 从文件内容中识别段落结构
                    segments = []
                    current_segment = None
                    segment_id = 0
                    
                    # 手动解析
                    in_segment = False
                    segment_content = ""
                    for line in lines:
                        if line.strip() == "{":
                            in_segment = True
                            segment_content = "{\n"
                        elif line.strip() == "}," or line.strip() == "}" and in_segment:
                            segment_content += line + "\n"
                            try:
                                obj = json.loads(segment_content.replace("},", "}"))
                                segments.append(obj)
                                print(f"成功解析段落 {len(segments)}")
                            except:
                                print(f"段落解析失败，跳过: {segment_content[:50]}...")
                            in_segment = False
                            segment_content = ""
                        elif in_segment:
                            segment_content += line + "\n"
                    
                    if segments:
                        data = segments
                        print(f"手动解析成功，共 {len(segments)} 个段落")
                    else:
                        # 最后的方案：手动构建有效的JSON结构
                        print("所有自动修复方法失败，创建新的JSON结构...")
                        data = []
                        for i, line in enumerate(lines):
                            if '"segment_id":' in line:
                                # 提取segment_id
                                id_match = re.search(r'"segment_id":\s*(\d+)', line)
                                if id_match:
                                    segment_id = int(id_match.group(1))
                                    
                                    # 查找内容部分
                                    content_start = None
                                    for j in range(i, len(lines)):
                                        if '"content":' in lines[j]:
                                            content_start = j
                                            break
                                    
                                    if content_start:
                                        # 查找内容结束
                                        content_text = ""
                                        j = content_start
                                        while j < len(lines) and '"segment_id":' not in lines[j]:
                                            if '"content":' in lines[j]:
                                                # 提取行中的内容部分
                                                content_part = lines[j].split(':', 1)[1].strip()
                                                if content_part.startswith('"'):
                                                    content_part = content_part[1:]
                                                if content_part.endswith('",'):
                                                    content_part = content_part[:-2]
                                                elif content_part.endswith('"'):
                                                    content_part = content_part[:-1]
                                                content_text += content_part
                                            else:
                                                # 处理多行内容
                                                line_text = lines[j].strip()
                                                if line_text.startswith('"'):
                                                    line_text = line_text[1:]
                                                if line_text.endswith('",'):
                                                    line_text = line_text[:-2]
                                                elif line_text.endswith('"'):
                                                    line_text = line_text[:-1]
                                                
                                                if line_text and not line_text.startswith('}'):
                                                    content_text += line_text
                                            j += 1
                                            
                                            # 如果遇到下一个段落或文件结束，跳出
                                            if j < len(lines) and ('"segment_id":' in lines[j] or '}' in lines[j]):
                                                break
                                        
                                        # 添加解析出的段落
                                        data.append({
                                            "segment_id": segment_id,
                                            "content": content_text
                                        })
                                        print(f"手动构建段落 {segment_id}")
                        
                        if not data:
                            print("无法修复JSON文件")
                            return False
            else:
                print("无法在文件中识别JSON数组模式")
                # 尝试最基本的替换
                content = content.replace('\n', '\\n')
                try:
                    data = json.loads(content)
                    print("替换换行符后成功解析")
                except:
                    print("替换换行符后仍然失败")
                    return False
                    
        # 将修复后的数据写回文件
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"已将修复后的JSON保存到 {output_file}")
        return True
                
    except Exception as e:
        print(f"处理文件时出错: {str(e)}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("用法: python fix_json.py 输入文件 输出文件")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    success = fix_json_file(input_file, output_file)
    sys.exit(0 if success else 1) 