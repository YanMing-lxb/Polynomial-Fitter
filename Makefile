.PHONY: all clean html rst whl pack inswhl upload

# 变量定义
UV_RUN = @uv run python
TOOLS_DIR = ./tools

# 默认目标
all:
	$(UV_RUN) $(TOOLS_DIR)/make.py all

# 清理
clean:
	$(UV_RUN) $(TOOLS_DIR)/utils.py clean

# 文档生成
html:
	$(UV_RUN) $(TOOLS_DIR)/make.py html

rst:
	$(UV_RUN) $(TOOLS_DIR)/make.py rst

# 构建wheel包
whl: clean
	@uv build

# 打包可执行文件
pack:
	$(UV_RUN) $(TOOLS_DIR)/pack.py pack

# 安装wheel包测试
inswhl:
	$(UV_RUN) $(TOOLS_DIR)/make.py inswhl

# 上传标签
upload:
	$(UV_RUN) $(TOOLS_DIR)/make.py upload

