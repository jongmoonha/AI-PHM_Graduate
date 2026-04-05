import matplotlib.pyplot as plt
import matplotlib
import os

COLORS = {
    'primary': '#1B4F72',
    'warning': '#C0392B',
    'code_bg': '#F2F3F4',
    'concept_bg': '#EBF5FB',
    'table_header': '#2C3E50',
    'line1': '#2980B9',
    'line2': '#E74C3C',
    'line3': '#27AE60',
    'line4': '#F39C12',
    'line5': '#8E44AD',
}

LINE_COLORS = [COLORS['line1'], COLORS['line2'], COLORS['line3'],
               COLORS['line4'], COLORS['line5']]


def setup_notebook():
    plt.rcParams.update({
        'figure.figsize': (10, 6),
        'font.size': 12,
        'font.family': 'sans-serif',
        'axes.grid': True,
        'grid.alpha': 0.3,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'lines.linewidth': 1.5,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'axes.prop_cycle': plt.cycler(color=LINE_COLORS),
        'legend.loc': 'upper right',
        'legend.fontsize': 10,
        'legend.framealpha': 0.9,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'axes.titleweight': 'bold',
        'figure.constrained_layout.use': True,
    })
    # 한글 폰트 설정 (Windows)
    if os.name == 'nt':
        plt.rcParams['font.family'] = 'Malgun Gothic'
        plt.rcParams['axes.unicode_minus'] = False
