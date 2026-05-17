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

rst:
	$(UV_RUN) $(TOOLS_DIR)/make.py rst

# 打包可执行文件
pack: clean
	$(UV_RUN) $(TOOLS_DIR)/pack.py pack

# 上传标签
upload:
	$(UV_RUN) $(TOOLS_DIR)/make.py upload

