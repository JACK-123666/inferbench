"""内联 SVG 图表（零依赖）。

为什么不用 matplotlib：报告要能直接贴进 Markdown / 浏览器看，
而且 `git clone` 后不装任何东西就能出图。这些 SVG 是纯字符串拼接，
两个函数加起来 100 行，比引入一个 40MB 的依赖划算。
"""
from __future__ import annotations

PALETTE = ["#2E5BFF", "#00A37A", "#F2994A", "#EB5757", "#9B51E0", "#2D9CDB"]
FONT = "Segoe UI, Microsoft YaHei, sans-serif"


def esc(text) -> str:
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _svg_open(width: int, height: int, title: str, y_label: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="{FONT}">',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text x="56" y="22" font-size="14" font-weight="600" fill="#111">{esc(title)}</text>',
        f'<text x="{width - 16}" y="22" font-size="11" fill="#666" text-anchor="end">{esc(y_label)}</text>',
    ]


def _y_grid(pad_l: int, width: int, pad_t: int, plot_h: int, vmin: float, vmax: float,
            percent: bool) -> list[str]:
    out = []
    for i in range(5):
        v = vmin + (vmax - vmin) * i / 4
        y = pad_t + plot_h - plot_h * i / 4
        out.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{width - 16}" y2="{y:.1f}" stroke="#E6E8EE"/>')
        label = f"{v:.0f}%" if percent else f"{v:.0f}"
        out.append(f'<text x="{pad_l - 8}" y="{y + 4:.1f}" font-size="10" fill="#888" '
                   f'text-anchor="end">{label}</text>')
    return out


def _legend(series: dict, x: int, y: int, step: int = 22) -> list[str]:
    out = []
    for i, name in enumerate(series):
        color = PALETTE[i % len(PALETTE)]
        ly = y + i * 18
        out.append(f'<rect x="{x}" y="{ly - 9}" width="9" height="9" fill="{color}" rx="2"/>')
        out.append(f'<text x="{x + 14}" y="{ly - 1}" font-size="10.5" fill="#444">{esc(name)}</text>')
        _ = step
    return out


def bar_chart(labels: list[str], series: dict[str, list[float]], *, title: str,
              y_label: str = "", unit: str = "", width: int = 720, height: int = 300) -> str:
    """分组柱状图（用于档位/配置对比）。"""
    n_groups, n_series = max(1, len(labels)), max(1, len(series))
    pad_l, pad_r, pad_t, pad_b = 56, 16, 40, 56
    plot_w, plot_h = width - pad_l - pad_r, height - pad_t - pad_b
    vmax = max([v for vals in series.values() for v in vals] or [0.0]) * 1.15 or 1.0
    group_w = plot_w / n_groups
    bar_w = max(6.0, group_w * 0.72 / n_series)

    out = _svg_open(width, height, title, y_label)
    out += _y_grid(pad_l, width, pad_t, plot_h, 0.0, vmax, percent=False)
    for gi, label in enumerate(labels):
        gx = pad_l + group_w * gi
        for si, vals in enumerate(series.values()):
            val = vals[gi] if gi < len(vals) else 0.0
            bh = plot_h * (val / vmax)
            x = gx + group_w * 0.14 + si * bar_w
            y = pad_t + plot_h - bh
            out.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w - 2:.1f}" '
                       f'height="{max(0.5, bh):.1f}" fill="{PALETTE[si % len(PALETTE)]}" rx="2"/>')
            out.append(f'<text x="{x + (bar_w - 2) / 2:.1f}" y="{y - 4:.1f}" font-size="9" '
                       f'fill="#444" text-anchor="middle">{val:.1f}{unit}</text>')
        out.append(f'<text x="{gx + group_w / 2:.1f}" y="{height - pad_b + 18}" font-size="10.5" '
                   f'fill="#333" text-anchor="middle">{esc(label)}</text>')
    lx = pad_l
    for i, name in enumerate(series):
        color = PALETTE[i % len(PALETTE)]
        out.append(f'<rect x="{lx}" y="{height - 20}" width="9" height="9" fill="{color}" rx="2"/>')
        out.append(f'<text x="{lx + 14}" y="{height - 12}" font-size="10.5" fill="#444">{esc(name)}</text>')
        lx += 22 + 7 * len(name)
    out.append("</svg>")
    return "\n".join(out)


def line_chart(x_values: list[float], series: dict[str, list[float]], *, title: str,
               x_label: str = "", y_label: str = "", width: int = 720, height: int = 300,
               percent: bool = True) -> str:
    """折线图（用于阈值扫描 / 并发梯度）。"""
    pad_l, pad_r, pad_t, pad_b = 56, 130, 40, 56
    plot_w, plot_h = width - pad_l - pad_r, height - pad_t - pad_b
    all_vals = [v for vals in series.values() for v in vals] or [0.0]
    vmax = max(all_vals) * 1.12 or 1.0
    xmin, xmax = min(x_values), max(x_values)
    span = (xmax - xmin) or 1.0

    def px(x: float) -> float:
        return pad_l + plot_w * (x - xmin) / span

    def py(v: float) -> float:
        return pad_t + plot_h - plot_h * v / vmax

    out = _svg_open(width, height, title, y_label)
    out += _y_grid(pad_l, width, pad_t, plot_h, 0.0, vmax, percent)
    for x in x_values:
        out.append(f'<text x="{px(x):.1f}" y="{height - pad_b + 18}" font-size="10" '
                   f'fill="#333" text-anchor="middle">{x:g}</text>')
    out.append(f'<text x="{pad_l + plot_w / 2:.1f}" y="{height - 8}" font-size="10.5" '
               f'fill="#333" text-anchor="middle">{esc(x_label)}</text>')
    for si, (_, vals) in enumerate(series.items()):
        color = PALETTE[si % len(PALETTE)]
        pts = " ".join(f"{px(x):.1f},{py(v):.1f}" for x, v in zip(x_values, vals))
        out.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2"/>')
        for x, v in zip(x_values, vals):
            out.append(f'<circle cx="{px(x):.1f}" cy="{py(v):.1f}" r="3" fill="{color}"/>')
    out += _legend(series, width - pad_r + 10, pad_t)
    out.append("</svg>")
    return "\n".join(out)
