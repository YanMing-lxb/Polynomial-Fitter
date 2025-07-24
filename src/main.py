import argparse
import os
import sys

import numpy as np
import rich_argparse
from openpyxl import load_workbook
from rich.console import Console
from rich.panel import Panel
from rich.prompt import IntPrompt, Prompt
from rich.table import Table
from scipy.optimize import curve_fit

from plot_show import plot_fits_pygal_combined
from utils import polynomial_fit
from version import script_name

console = Console()


def read_excel_data(file_path: str) -> dict[str, np.ndarray]:
    """从Excel文件中读取数据，支持双层表头结构

    该函数读取指定路径的Excel文件，将前两行作为双层表头处理，
    从第三行开始作为数据行读取，并将数据组织成字典结构，
    其中键为双层表头的元组，值为对应的NumPy数组。

    Parameters
    ----------
    file_path : str
        Excel文件的路径，支持.xlsx格式

    Returns
    -------
    dict[tuple[str, str], np.ndarray]
        以双层表头元组为键，对应列数据的NumPy数组为值的字典
    """
    if not isinstance(file_path, str) or not file_path.endswith(".xlsx"):
        raise ValueError("file_path 必须是非空的 .xlsx 文件路径")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件未找到: {file_path}")

    wb = None
    try:
        wb = load_workbook(filename=file_path, read_only=True, data_only=True)
        ws = wb.active

        rows = list(ws.iter_rows(values_only=True))

        if len(rows) < 2:
            raise ValueError("Excel 文件至少需要两行作为双层表头")

        # 读取前两行作为双层 header
        header_row1 = rows[0]
        header_row2 = rows[1]

        combined_headers = [tuple(pair) for pair in zip(header_row1, header_row2)]

        # 初始化每列的数据列表
        data_dict = {header: [] for header in combined_headers}

        # 从第三行开始读取数据
        for row in rows[2:]:
            for i, value in enumerate(row):
                if i < len(combined_headers):
                    data_dict[combined_headers[i]].append(value)

        # 转换为 NumPy 数组并返回
        return {k: np.array(v) for k, v in data_dict.items()}

    except Exception as e:
        raise RuntimeError(f"读取 Excel 数据时出错: {e}") from e
    finally:
        if wb is not None:
            wb.close()


def fit_polynomial(
    data: dict[str, np.ndarray], max_degree: int = 9, threshold: float = 0.9999
) -> dict[str, dict[str, int | float | np.ndarray]]:
    """
    对给定数据进行多项式拟合，寻找最优拟合度满足阈值要求的最低阶多项式模型。

    参数:
    data : dict[str, np.ndarray]
        输入数据字典，其中第一个键对应的值为自变量 x 的数据，
        其余键对应因变量 y 的数据。所有值应为 numpy 数组。
    max_degree : int, optional
        拟合多项式的最高阶数，默认为 9。
    threshold : float, optional
        判定拟合优度是否合格的 R² 和调整 R² 阈值，默认为 0.9999。

    返回:
    dict[str, dict[str, int | float | np.ndarray]]
        每个因变量的拟合结果字典：
        - 若成功拟合，则包含 "degree"（阶数）、"coefficients"（系数）、
          "r_squared"（R²）和 "adj_r_squared"（调整 R²）；
        - 若拟合失败或无法收敛，则返回错误信息字符串。
    """
    results = {}

    # 校验输入数据
    if not isinstance(data, dict) or len(data) < 2:
        raise ValueError("输入数据必须为至少包含两个键的字典")

    keys = list(data.keys())
    xdata = data[keys[0]]

    if not isinstance(xdata, np.ndarray):
        raise TypeError("xdata 必须为 numpy 数组")

    # 遍历每一个因变量进行多项式拟合
    for dependent_var in keys[1:]:
        ydata = data[dependent_var]

        if not isinstance(ydata, np.ndarray):
            results[dependent_var] = {"error": "ydata 必须为 numpy 数组"}
            continue

        if len(xdata) != len(ydata):
            results[dependent_var] = {"error": "xdata 与 ydata 长度不一致"}
            continue

        results[dependent_var] = {}

        # 尝试从1到max_degree阶的多项式拟合
        for degree in range(1, max_degree + 1):
            # 初始猜测参数
            p0 = np.zeros(degree)  # 更合理的初始值

            try:
                coefficients, _ = curve_fit(
                    polynomial_fit, xdata, ydata, maxfev=10000, p0=p0
                )
            except Exception:
                results[dependent_var] = {"error": "无法拟合"}
                break

            # 计算拟合优度指标 R² 和调整 R²
            y_fit = polynomial_fit(xdata, *coefficients)
            residuals = ydata - y_fit
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((ydata - np.mean(ydata)) ** 2)

            if ss_tot == 0:
                r_squared = 1.0  # 所有值相同，完全拟合
            else:
                r_squared = 1 - (ss_res / ss_tot)

            n = len(ydata)
            if n - degree - 1 == 0:
                adj_r_squared = np.nan  # 无法计算调整 R²
            else:
                adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - degree - 1)

            # 如果当前阶数满足拟合精度要求，则记录结果并跳出循环
            if r_squared >= threshold and adj_r_squared >= threshold:
                results[dependent_var] = {
                    "degree": degree,
                    "coefficients": coefficients,
                    "r_squared": r_squared,
                    "adj_r_squared": adj_r_squared,
                }
                break
            # 如果达到最大阶数仍未满足条件，则标记为超限
            elif degree == max_degree:
                results[dependent_var] = {"error": "多项式系数超限"}
                break

    return results


def generate_polynomial_expression(
    xname: str, yname: str, coefficients: np.ndarray
) -> str:
    """生成多项式表达式的字符串表示形式

    根据给定的系数数组生成形如 "y = a0 + a1*x + a2*x^2 + ..." 的多项式表达式字符串，
    会自动处理系数为0、1、-1等特殊情况，并格式化系数显示。

    Parameters
    ----------
    xname : str
        自变量的名称，用于构建多项式中的x项
    yname : str
        因变量的名称，用于等式左侧的变量名
    coefficients : np.ndarray
        多项式的系数数组，索引对应幂次，coefficients[i]为x^i项的系数

    Returns
    -------
    str
        格式化后的多项式表达式字符串
    """
    if len(coefficients) == 0:
        return f"{yname} = 0"

    expression = yname + " = "
    terms = []

    for i, coef in enumerate(coefficients):
        if coef == 0:
            continue

        term = ""
        sign = "+" if coef > 0 and i > 0 else ""  # 首项不加正号

        if i == 0:
            term = f"{sign}{coef:.6g}"
        elif i == 1:
            if coef == 1:
                term = f"{sign}{xname}"
            elif coef == -1:
                term = f"-{xname}"
            else:
                term = f"{sign}{coef:.6g}\\times {xname}"
        else:
            if coef == 1:
                term = f"{sign}{xname}^{i}"
            elif coef == -1:
                term = f"-{xname}^{i}"
            else:
                term = f"{sign}{coef:.6g}\\times {xname}^{i}"

        terms.append(term)

    full_expr = "".join(terms)
    # 移除首项多余的正号
    result = full_expr.lstrip("+")
    return expression + result if result else expression + "0"


def run(
    file_path: str,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, int | float | np.ndarray]]]:
    """
    执行数据读取和多项式拟合流程，并将结果以表格形式输出到控制台。

    该函数首先从指定路径读取Excel数据，然后对数据进行多项式拟合。拟合结果会根据变量名分组显示，
    包括拟合状态（如系数超限或无法拟合）以及详细的拟合信息，例如LaTeX表达式、StarCCM+系数、
    Fluent系数以及其他统计指标。

    Parameters
    ----------
    file_path : str
        Excel文件的路径，用于读取待拟合的数据。

    Returns
    -------
    tuple[dict[str, np.ndarray], dict[str, dict[str, int | float | np.ndarray]]]
        一个元组，包含两个元素：
        - 第一个元素是原始数据字典，键为变量名，值为对应的数值数组；
        - 第二个元素是拟合结果字典，外层键为变量名元组，内层键为拟合属性（如系数、R²等），值为相应的拟合结果数据；
    """

    try:
        # 读取Excel中的数据
        data = read_excel_data(file_path)
    except Exception as e:
        console.print(f"[red]读取Excel数据失败: {e}[/red]")
        raise

    try:
        # 对读取的数据执行多项式拟合操作
        results = fit_polynomial(data)
    except Exception as e:
        console.print(f"[red]多项式拟合失败: {e}[/red]")
        raise

    # 遍历拟合结果并格式化输出至控制台
    for key1, value1 in results.items():
        var_name = " ".join(key1)
        console.print(Panel(f"[bold blue]{var_name}[/bold blue]", expand=False))

        # 处理拟合失败的情况
        if value1 == "多项式系数超限":
            console.print("[red]多项式系数超限[/red]\n")
            continue
        elif value1 == "无法拟合":
            console.print("[red]无法拟合[/red]\n")
            continue

        # 创建表格用于展示当前变量的拟合结果
        table = Table(show_header=False, box=None)
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="magenta")

        # 填充表格内容
        for key2, value2 in value1.items():
            if key2 == "coefficients":
                # 生成LaTeX格式的多项式表达式
                latex_polynomial_expression = generate_polynomial_expression(
                    "T", "y", value2
                )
                table.add_row("LaTeX 公式", latex_polynomial_expression)

                # 缓存格式化后的系数字符串
                formatted_coeffs = [f"{x:.6g}" for x in value2]

                # 添加StarCCM+格式的系数表示
                table.add_row("StarCCM+ 系数", f"[{', '.join(formatted_coeffs)}]")

                # 添加Fluent格式的系数表示
                degree = value1.get("degree", len(value2) - 1)
                table.add_row(
                    "Fluent 系数",
                    f"{degree} {' '.join(formatted_coeffs)}",
                )
            else:
                # 显示其他拟合属性（如R²、均方误差等）
                table.add_row(
                    key2, f"{value2:.6f}" if isinstance(value2, float) else str(value2)
                )

        # 输出当前变量的结果表格
        console.print(table)
        console.print()

    # 返回原始数据和拟合结果
    return data, results


def cli_main():
    """_summary_"""
    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 构建默认文件路径
    default_file_path = os.path.join(current_dir, "data.xlsx")

    # 导入 rich_argparse 并设置为默认格式化类
    rich_argparse.RichHelpFormatter.styles["argparse.prog"] = "bold cyan"
    rich_argparse.RichHelpFormatter.styles["argparse.args"] = "yellow"
    rich_argparse.RichHelpFormatter.styles["argparse.metavar"] = "yellow"

    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(
        description="[bold green]多项式拟合工具[/bold green]",
        formatter_class=rich_argparse.RichHelpFormatter,
    )
    parser.add_argument(
        "file_path",
        type=str,
        nargs="?",
        default=default_file_path,
        help="Excel文件路径 (默认: data.xlsx)",
    )
    parser.add_argument("--no-plot", "-n", action="store_true", help="不显示图表")
    parser.add_argument(
        "--columns", "-c", type=int, default=2, help="图表显示列数 (默认: 2)"
    )

    # 解析命令行参数
    args = parser.parse_args()

    # 显示程序标题和输入信息
    console.print(Panel("[bold blue]多项式拟合工具[/bold blue]", expand=False))
    console.print(f"输入文件: [cyan]{args.file_path}[/cyan]")

    # 运行主程序
    try:
        data, results = run(args.file_path)
    except FileNotFoundError:
        console.print(f"[red]错误：找不到文件 '{args.file_path}'[/red]")
        exit(1)
    except Exception as e:
        console.print(f"[red]错误：{str(e)}[/red]")
        exit(1)

    # 如果没有指定--no-plot，则显示图表
    if not args.no_plot:
        plot_fits_pygal_combined(data, results, filename=None, columns=args.columns)
    else:
        console.print("[yellow]已跳过图表显示[/yellow]")


def input_main():
    """_summary_"""
    try:
        # 显示程序标题
        console = Console(width=60)
        console.print(f"\n[bold green]{script_name}[/bold green]\n", justify="center")

        # 交互式输入参数
        while True:
            try:
                console.print("[bold cyan]参数配置[/bold cyan]", justify="center")
                console.print("=" * 60)
                # 获取当前脚本所在目录
                current_dir = os.getcwd()
                # 构建默认文件路径
                default_file_path = os.path.join(current_dir, "data.xlsx")
                # 输入文件路径
                input_file_path = Prompt.ask(
                    "[bold blue]Excel文件路径[/bold blue]",
                    default=default_file_path,
                    console=console,
                )
                console.print(f"文件路径: [cyan]{input_file_path}[/cyan]\n")

                # 是否显示图像
                plot_show_input = Prompt.ask(
                    "[bold blue]是否显示图像?[/bold blue] [dim](y/n)[/dim]",
                    choices=["y", "n"],
                    default="y",
                    console=console,
                )
                plot_show = plot_show_input.lower() in ["y", "yes"]
                status = "[green]启用[/green]" if plot_show else "[red]禁用[/red]"
                console.print(f"图像显示: {status}\n")

                # 图表列数
                columns = IntPrompt.ask(
                    "[bold blue]图表显示列数[/bold blue]", default=2, console=console
                )
                console.print(f"显示列数: [magenta]{columns}[/magenta]\n")

                # 显示分隔线
                console.print("=" * 60)
                console.print("[bold green]开始处理数据...[/bold green]\n")

            except ValueError as e:
                console.print(f"[red]输入格式错误: {str(e)}，请重新输入[/red]\n")
                continue
            except KeyboardInterrupt:
                console.print("\n[yellow]用户取消操作[/yellow]")
                return

            # 运行主程序
            try:
                data, results = run(input_file_path)

                # 显示图表
                if plot_show:
                    plot_fits_pygal_combined(
                        data, results, filename=None, columns=int(columns)
                    )
                    console.print("[green]图表已生成并显示[/green]\n")
                else:
                    console.print("[yellow]已跳过图表显示[/yellow]\n")

            except Exception as e:
                console.print(f"[red]处理过程中发生错误: {str(e)}[/red]\n")
                continue

            # 显示完成信息
            console.print("=" * 60)
            console.print(
                "[bold green]所有任务已完成![/bold green]",
                justify="center",
            )

            # 等待用户退出
            console.input("\n[bold cyan]按任意键退出...[/bold cyan]")
            break

    except Exception as e:
        console.print(f"[red]程序发生异常: {str(e)}[/red]")
        console.input("\n[bold red]程序运行出错，按任意键退出...[/bold red]")


def main():
    if len(sys.argv) > 1:
        cli_main()
    else:
        input_main()


if __name__ == "__main__":
    main()

# 🧪 pyinstaller --onefile --add-data "src/assets;src/assets" --console -i "src/assets/icon.ico" --clean ./src/main.py
# 🧪 pyinstaller --onefile --console --add-data "src/assets;assets" --collect-data pygal -i "src/assets/icon.ico" --clean --name "Polynomial-Fitter" ./src/main.py
