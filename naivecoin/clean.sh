#!/bin/bash

# 清理编译后的文件
echo "正在清理编译后的文件..."

# 删除所有的.js和.js.map文件，但保留node_modules目录中的文件
find ./src -name "*.js" -type f -delete
find ./src -name "*.js.map" -type f -delete

echo "清理完成！"