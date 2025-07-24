import os
import tempfile
import webbrowser

import numpy as np
import pygal
from pygal.style import LightStyle

from utils import folder_path, polynomial_fit


def generate_chart(
    x_column: tuple,
    xdata: np.ndarray,
    y_column: tuple,
    ydata: np.ndarray,
    coefficients: np.ndarray,
    degree: int,
) -> pygal.XY:
    """
    生成一个包含原始数据点和多项式拟合曲线的XY图表。

    该函数使用给定的系数对输入数据进行多项式拟合，并将原始数据与拟合结果一同绘制在图表中，
    图表采用pygal库的XY类型，支持自定义样式、图例、坐标轴标题等。

    Parameters
    ----------
    x_column : tuple
        表示X轴数据列的元组，通常用于构造X轴标题文本
    xdata : np.ndarray
        X轴的原始数据数组
    y_column : tuple
        表示Y轴数据列的元组，通常用于构造Y轴标题文本
    ydata : np.ndarray
        Y轴的原始数据数组
    coefficients : np.ndarray
        多项式拟合得到的系数数组，用于计算拟合值
    degree : int
        拟合多项式的次数，用于图例显示

    Returns
    -------
    pygal.XY
        配置完成并添加了原始数据和拟合曲线的XY图表对象
    """

    # 计算拟合后的y值
    y_fit = polynomial_fit(xdata, *coefficients)

    # 构造原始数据和拟合数据的坐标点列表
    original_data = [(x, y) for x, y in zip(xdata, ydata)]
    fitted_data = [(x, y) for x, y in zip(xdata, y_fit)]

    # 自定义图表样式
    custom_style = LightStyle(
        colors=("#E853A0", "#29B6F6"),
        font_family="googlefont:Raleway",
        background="transparent",
        label_font_size=12,
    )

    # 初始化XY图表对象并设置基本属性
    xy_chart = pygal.XY(
        style=custom_style,
        show_legend=True,
        legend_at_bottom=True,  # 将图例放在底部
        tooltip_border_radius=10,
        human_readable=True,
    )

    # 设置坐标轴标题和图表尺寸
    xy_chart.x_title = " ".join(x_column)
    xy_chart.y_title = " ".join(y_column)
    xy_chart.width = 600
    xy_chart.height = 400

    # 设置坐标轴数值格式化方式
    xy_chart.x_value_formatter = lambda x: f"{x:.2f}"
    xy_chart.y_value_formatter = lambda y: f"{y:.2f}"

    # 显示网格线和辅助线
    xy_chart.show_x_guides = True
    xy_chart.show_y_guides = True
    xy_chart.grid = True

    # 添加原始数据系列到图表中
    xy_chart.add(
        "原始数据",
        original_data,
        stroke=False,
        show_dots=True,
        dot_size=5,
        fill=False,
    )

    # 添加拟合曲线系列到图表中
    xy_chart.add(
        f"{degree} 次多项式",
        fitted_data,
        stroke=True,
        fill=False,
        show_dots=False,
        stroke_width=5,
        dots_size=0,
    )

    return xy_chart


def render_html_charts(
    charts: list[pygal.XY], results: dict, container_width: float, template_path: str
) -> str:
    """将图表数据渲染成HTML页面

    该函数读取HTML模板文件，将图表和拟合信息插入到模板中，生成完整的HTML页面。
    对于拟合结果为"多项式系数超限"的数据将被跳过不显示。

    Parameters
    ----------
    charts : list[pygal.XY]
        图表对象列表，每个元素是一个pygal.XY图表实例
    results : dict
        拟合结果字典，键为数据标识，值为包含degree、r_squared、adj_r_squared的字典
    container_width : float
        图表容器的宽度值，用于替换CSS变量
    template_path : str
        HTML模板文件的路径

    Returns
    -------
    str
        渲染完成的完整HTML页面字符串
    """
    # 读取HTML模板文件
    with open(template_path, "r", encoding="utf-8") as f:
        html_template = f.read()

    # 构建图表HTML片段
    chart_divs = ""
    for chart, (key, value) in zip(charts, results.items()):
        if value == "多项式系数超限":
            continue
        degree = value["degree"]
        r_squared = value["r_squared"]
        adj_r_squared = value["adj_r_squared"]

        chart_divs += f"""
    <div class="chart-container">
        <div class="chart-wrapper">
            {chart.render(disable_xml_declaration=True)}
            <div class="fit-info">
                <strong>拟合信息:</strong> {degree}次多项式 | R² = {r_squared:.6f} | 调整 R² = {adj_r_squared:.6f}
            </div>
        </div>
    </div>"""

    # 将图表片段和容器宽度替换到模板中
    full_html = html_template.replace(
        "<!-- CHART_DIVS_PLACEHOLDER -->", chart_divs
    ).replace("var(--container-width)", str(container_width))

    return full_html


def plot_fits_pygal_combined(
    data: dict, results: dict, filename: str = None, columns: int = 2
):
    """使用 pygal 绘制拟合结果图表并保存为 HTML 文件或在浏览器中打开。

    该函数根据输入的数据和拟合结果，为每组数据生成拟合图表，并将所有图表整合到一个 HTML 页面中。
    用户可以选择保存为文件或直接在浏览器中查看。

    Parameters
    ----------
    data : dict
        包含原始数据的字典，其中键为列名，值为对应的数据列表。第一个键将被用作 x 轴数据。
    results : dict
        包含拟合结果的字典，键对应 data 中的列名，值为包含 "coefficients" 和 "degree" 的字典，
        或字符串 "多项式系数超限" 表示该组数据无法拟合。
    filename : str, optional
        输出的 HTML 文件路径。如果未提供，则使用临时文件并在浏览器中打开，默认为 None。
    columns : int, optional
        图表在页面中的列数，用于控制图表布局，默认为 2。

    Returns
    -------
    None
        无返回值，但会生成 HTML 文件或打开浏览器窗口。
    """
    # 获取第一个键作为 x 列
    x_column = list(data.keys())[0]
    xdata = data[x_column]

    # 创建图表列表
    charts = []

    # 遍历拟合结果
    for key, value in results.items():
        if value == "多项式系数超限":
            continue

        ydata = data[key]
        coefficients = value["coefficients"]
        degree = value["degree"]

        chart = generate_chart(x_column, xdata, key, ydata, coefficients, degree)
        charts.append(chart)

    container_width = 100 / columns

    # 读取模板文件
    template_path = folder_path("assets", "template.html")

    # 渲染 HTML
    full_html = render_html_charts(charts, results, container_width, template_path)

    if filename:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(full_html)
        print(f"所有图表已保存至 {filename}")
    else:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".html", delete=False, encoding="utf-8"
        ) as f:
            f.write(full_html)
            temp_filename = f.name

        webbrowser.open("file://" + os.path.abspath(temp_filename))
        print("图表已在浏览器中打开")
