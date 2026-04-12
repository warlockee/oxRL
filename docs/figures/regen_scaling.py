import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

scales = [0.5, 1.5, 3.0, 7.0]
scale_labels = ['0.5', '1.5', '3.0', '7.0']

# Mean values — 3B uses optimized LR for SP-RFT and DPO; None for others
data = {
    'base':   [19.56, 31.16, 10.69, 75.82],
    'SP-RFT': [33.97, 54.36, 55.70, 77.38],
    'DPO':    [33.97, 49.08, 34.55, 83.38],
    'SimPO':  [26.08, 38.67, None,  83.32],
    'IPO':    [34.50, 52.24, None,  80.39],
    'KTO':    [33.81, 51.15, None,  80.24],
    'SGRPO':  [32.45, 58.00, None,  80.59],
}

stds = {
    'base':   [0, 0, 0, 0],
    'SP-RFT': [0, 0.59, 1.09, 1.11],
    'DPO':    [0, 0.61, 4.52, 0.56],
    'SimPO':  [0, 1.78, 0, 1.79],
    'IPO':    [0.3, 0.22, 0, 0.92],
    'KTO':    [0, 1.77, 0, 0.16],
    'SGRPO':  [0, 0.57, 0, 0],
}

colors = {
    'base':   '#888888',
    'SP-RFT': '#555555',
    'DPO':    '#1f77b4',
    'SimPO':  '#ff7f0e',
    'IPO':    '#2ca02c',
    'KTO':    '#d62728',
    'SGRPO':  '#17becf',
}
markers = {
    'base': 'o', 'SP-RFT': 's', 'DPO': 'D', 'SimPO': '^',
    'IPO': 'v', 'KTO': 'P', 'SGRPO': 'X',
}

fig, ax = plt.subplots(1, 1, figsize=(8, 5))

draw_order = ['base', 'SP-RFT', 'DPO', 'SimPO', 'IPO', 'KTO', 'SGRPO']

for name in draw_order:
    vals = data[name]
    errs = stds[name]
    lw = 2.5 if name in ['SP-RFT', 'DPO'] else 1.5
    zorder = 10 if name in ['SP-RFT', 'DPO'] else 5
    ls = '--' if name == 'base' else '-'
    ms = 8 if name in ['SP-RFT', 'DPO'] else 6

    if None in vals:
        idx = vals.index(None)
        # Before gap
        xs1 = [scales[j] for j in range(idx)]
        ys1 = [vals[j] for j in range(idx)]
        es1 = [errs[j] for j in range(idx)]
        # After gap
        xs2 = [scales[j] for j in range(idx+1, len(scales))]
        ys2 = [vals[j] for j in range(idx+1, len(scales))]
        es2 = [errs[j] for j in range(idx+1, len(scales))]

        ax.errorbar(xs1, ys1, yerr=es1, color=colors[name], marker=markers[name],
                    linewidth=lw, markersize=ms, linestyle=ls, zorder=zorder, capsize=3)
        ax.errorbar(xs2, ys2, yerr=es2, color=colors[name], marker=markers[name],
                    linewidth=lw, markersize=ms, linestyle=ls, zorder=zorder, capsize=3,
                    label=name)
        # Dotted connector across gap
        ax.plot([xs1[-1], xs2[0]], [ys1[-1], ys2[0]], color=colors[name],
                linewidth=lw*0.6, linestyle=':', alpha=0.5, zorder=zorder-1)
    else:
        ax.errorbar(scales, vals, yerr=errs, color=colors[name], marker=markers[name],
                    linewidth=lw, markersize=ms, linestyle=ls, zorder=zorder, capsize=3,
                    label=name)

# Annotations
ax.annotate('SP-RFT > DPO', xy=(1.5, 55.5), xytext=(0.9, 63),
            fontsize=8.5, fontstyle='italic', color='#333333',
            arrowprops=dict(arrowstyle='->', color='#666666', lw=0.8))
ax.annotate('DPO > SP-RFT', xy=(6.85, 80.5), xytext=(4.8, 88),
            fontsize=8.5, fontstyle='italic', color='#333333',
            arrowprops=dict(arrowstyle='->', color='#666666', lw=0.8))

ax.set_xlabel('Model Scale (Billion Parameters)', fontsize=12)
ax.set_ylabel('GSM8K Accuracy (%)', fontsize=12)
ax.set_title('Post-Training Algorithm Scaling on GSM8K', fontsize=13, fontweight='bold')
ax.set_xticks(scales)
ax.set_xticklabels(scale_labels)
ax.set_ylim(0, 92)
ax.legend(loc='upper left', fontsize=8.5, ncol=2, framealpha=0.9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/ec2-user/fsx/oxRL/docs/figures/scaling_curves_gsm8k.pdf',
            dpi=300, bbox_inches='tight')
plt.savefig('/home/ec2-user/fsx/oxRL/docs/figures/scaling_curves_gsm8k.png',
            dpi=300, bbox_inches='tight')
print("Figure saved successfully")
