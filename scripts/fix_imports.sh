#!/bin/bash

# 修复所有Python文件中的导入路径
find src/core -name "*.py" -type f -exec sed -i '' 's/from model_utils import/from src.core.utils.model_utils import/g' {} \;

# 修复特定文件中的导入路径
files=(
  "src/core/document/enhanced_cleaner_llm.py"
  "src/core/document/semantic_segmentation_llm.py"
  "src/core/qa/qa_generation_llm.py"
  "src/core/qa/qa_enhancement_llm.py"
  "src/core/qa/qa_evaluation_llm.py"
  "src/core/evaluation/coverage_evaluation.py"
)

for file in "${files[@]}"; do
  echo "修复文件: $file"
  sed -i '' 's/from model_utils import/from src.core.utils.model_utils import/g' "$file"
done

# 添加自定义头部（例如导入路径）
for file in $(find src/core -name "*.py" -type f); do
  if ! grep -q "import sys" "$file"; then
    temp_file=$(mktemp)
    cat > "$temp_file" << EOF
import sys
import os
# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

EOF
    cat "$file" >> "$temp_file"
    mv "$temp_file" "$file"
    echo "添加系统路径导入到: $file"
  fi
done

echo "导入路径修复完成！" 