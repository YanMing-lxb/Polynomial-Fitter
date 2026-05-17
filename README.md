# Polynomial-Fitter - 多项式拟合工具

## 简介

Polynomial-Fitter 是一个基于 Python 的多项式曲线拟合工具，可以从 Excel 文件读取数据，进行多项式拟合，并生成拟合曲线图表。

## 使用方法

### 运行程序

直接运行 `Polynomial-Fitter.exe` 进入交互模式，程序会提示输入：
- Excel 数据文件路径
- 多项式阶数
- 输出文件路径

## 开发构建

### 环境准备

确保已安装 uv：

```bash
pip install uv
```

安装依赖：

```bash
uv sync --dev
```

### 打包为 exe

```bash
uv run tools/pack.py pack
```

或使用 Make：

```bash
make pack
```

### 清理临时文件

```bash
uv run tools/pack.py --clean
```

或使用 Make：

```bash
make clean
```

## 版本历史

详见 [CHANGELOG.md](./CHANGELOG.md)

## 许可证

GPL-3.0-or-later