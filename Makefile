.PHONY: all clean html rst whl pack inswhl upload pot mo poup excel-addin app_builder release-build

# 变量定义
UV_RUN = @uv run python ./tools/

# 默认目标
all:
	$(UV_RUN)make.py all

# 清理
clean:
	$(UV_RUN)utils.py clean

# 文档生成
html:
	$(UV_RUN)make.py html

rst:
	$(UV_RUN)make.py rst

# 构建wheel包
whl: clean
	@uv build

# 打包可执行文件
pack:
	$(UV_RUN)pack.py pack

# 安装wheel包测试
inswhl:
	$(UV_RUN)make.py inswhl

# 上传标签
upload:
	$(UV_RUN)make.py upload

# 国际化
pot mo poup:
	$(UV_RUN)lang_tool.py $@

# 构建Excel/WPS插件
excel-addin:
	@echo "Building Excel/WPS integration add-in..."
	$(UV_RUN) build_addin.py

# 构建完整应用
app_builder:
	@echo "Building full application package..."
	$(UV_RUN) app_builder.py

# 完整发布构建（包括插件和完整应用）
release-build: whl pack excel-addin app_builder
	@echo "Release excel-addin complete! Ready for publishing."

	
