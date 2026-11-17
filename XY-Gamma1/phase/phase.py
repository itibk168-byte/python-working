from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# 报告中的固定参数与坐标范围
J, gamma, Gamma = 1.0, 0.6, 0.6
alpha_min, alpha_max = -1.0, 1.0
h_min, h_max = 0.0, 2.0

# 三条解析相界相交于 (alpha_c, J)
alpha_c = -(J * gamma) ** 2 / (4 * Gamma**2)
alpha_left = np.linspace(alpha_min, alpha_c, 500)
alpha_right = np.array([alpha_c, alpha_max])
h_spiral_pm = np.sqrt(J**2 - (J * gamma) ** 2 - 4 * Gamma**2 * alpha_left)

mpl.rcParams.update({
    "font.size": 10,
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,
})

fig, ax = plt.subplots(figsize=(5.4, 4.2), layout="constrained")

# 填充顺磁相、无隙螺旋相和反铁磁相
ax.set_facecolor("#F1F2F4")
ax.fill_between(alpha_left, h_min, h_spiral_pm, color="#B6D9E9")
ax.fill_between(alpha_right, h_min, J, color="#F3D5A6")

# 绘制三条相界
line_style = {"color": "#23303B", "lw": 1.25}
ax.plot(alpha_left, h_spiral_pm, **line_style)        # 螺旋–顺磁
ax.plot([alpha_c, alpha_c], [h_min, J], **line_style)  # 螺旋–反铁磁
ax.plot(alpha_right, [J, J], **line_style)            # 反铁磁–顺磁

ax.text(-0.625, 0.56, "Gapless spiral", ha="center", va="center")
ax.text(0.375, 0.56, "Antiferromagnetic", ha="center", va="center")
ax.text(0.0, 1.62, "Paramagnetic", ha="center", va="center")

ax.set(
    xlim=(alpha_min, alpha_max),
    ylim=(h_min, h_max),
    xlabel=r"$\alpha$",
    ylabel=r"$h$",
    xticks=np.linspace(alpha_min, alpha_max, 5),
    yticks=np.linspace(h_min, h_max, 5),
    title=rf"$XY$–$\Gamma$ chain ($J={J:g},\ \gamma={gamma:g},\ \Gamma={Gamma:g}$)",
)
ax.tick_params(direction="out", length=3.5)

# 创建目标目录并保存矢量 PDF
output_dir = Path(r"D:\python-code\python-working\XY-Gamma1\phase")
output_dir.mkdir(parents=True, exist_ok=True)
output_file = output_dir / "xy_gamma_phase_diagram.png"

fig.savefig(output_file, bbox_inches="tight")
print(f"相图已保存至 {output_file}")
plt.show()