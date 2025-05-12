#!/bin/bash

# 检查Python版本
python_version=$(python3 --version 2>&1 | awk '{print $2}')
required_version="3.8.0"

echo "检查Python版本..."
if [[ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]]; then
    echo "错误: 需要Python $required_version或更高版本"
    exit 1
else
    echo "Python版本检查通过: $python_version"
fi

# 创建项目目录结构
echo "创建项目目录结构..."
mkdir -p data/{raw,processed,output/{enhanced,split,evaluation}}
mkdir -p logs/{document,qa,evaluation}
mkdir -p tests/{unit,integration}
mkdir -p docs/{api,examples}

# 安装依赖
echo "安装依赖..."
pip install -r requirements.txt

# 检查依赖安装是否成功
if [ $? -ne 0 ]; then
    echo "依赖安装失败，请检查requirements.txt文件"
    exit 1
else
    echo "依赖安装成功"
fi

# 设置API密钥环境变量
echo "配置API密钥..."
if [ ! -f "src/config/config_local.yaml" ]; then
    cp src/config/config.yaml src/config/config_local.yaml
    echo "已创建本地配置文件: src/config/config_local.yaml"
    echo "请编辑该文件，设置你的API密钥"
else
    echo "本地配置文件已存在: src/config/config_local.yaml"
fi

# 设置执行权限
chmod +x scripts/run.sh

echo "环境设置完成！"
echo "接下来可以运行: bash scripts/run.sh 你的文档.pdf" 