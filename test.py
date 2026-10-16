import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ==============================================================================
# 1. GENERATE TARGET DISTRIBUTION (target_distribution.png)
# ==============================================================================
fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)

categories = ['No failure (0)', 'Failure (1)']
counts = [9661, 339]
percentages = [96.61, 3.39]
colors = ['#1f77b4', '#1f77b4']

bars = ax.bar(categories, counts, color=colors, width=0.55, edgecolor='black', linewidth=0.8)

# Add exact value labels on top of bars
for bar, count, pct in zip(bars, counts, percentages):
    yval = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2.0,
        yval + 180,
        f'{count:,} ({pct:.2f}%)',
        ha='center',
        va='bottom',
        fontsize=11,
        fontweight='medium'
    )

ax.set_title('AI4I 2020 target class distribution', fontsize=14, pad=12)
ax.set_ylabel('Number of records', fontsize=12)
ax.set_ylim(0, 11000)
ax.grid(axis='y', linestyle='-', alpha=0.3)
ax.set_axisbelow(True)

# Clean spine styling
for spine in ['top', 'right', 'left', 'bottom']:
    ax.spines[spine].set_color('#333333')
    ax.spines[spine].set_linewidth(1.0)

plt.tight_layout()
plt.savefig('target_distribution.png', bbox_inches='tight')
plt.close()
print("✓ Successfully generated 'target_distribution.png'")

# ==============================================================================
# 2. GENERATE SYSTEM ARCHITECTURE (system_architecture.png)
# ==============================================================================
fig, ax = plt.subplots(figsize=(13, 6.8), dpi=300)
ax.set_xlim(0, 13)
ax.set_ylim(0, 7)
ax.axis('off')

# Diagram Title
ax.text(6.5, 6.6, 'System architecture of the explainable predictive-maintenance prototype',
        ha='center', va='center', fontsize=14, fontweight='bold', color='#111111')

# Define boxes: (x, y, width, height, title, subtitle)
nodes = {
    'data': (0.5, 4.6, 3.0, 1.2, 'AI4I 2020 CSV', '10,000 × 14'),
    'val': (4.0, 4.6, 3.2, 1.2, 'Data validation & EDA', 'missing / duplicates / balance'),
    'feat': (7.6, 4.6, 3.2, 1.2, 'Leakage-safe feature set', '6 predictors + target'),
    'train': (7.6, 2.6, 3.2, 1.2, 'Train 3 classifiers', 'LR / Random Forest / SVM'),
    'prep': (4.0, 2.6, 3.2, 1.2, 'Preprocessing pipelines', 'scale numeric + encode Type'),
    'split': (0.5, 2.6, 3.0, 1.2, 'Stratified split', '80% train / 20% test'),
    'eval': (0.5, 0.6, 3.0, 1.2, 'Held-out evaluation', 'metrics + confusion matrices'),
    'xai': (4.0, 0.6, 3.2, 1.2, 'Explainability', 'SHAP / LR contributions /\nSVM permutation importance'),
    'app': (7.6, 0.6, 3.2, 1.2, 'Streamlit prototype', 'risk scores + explanations')
}

# Draw Boxes
for key, (x, y, w, h, title, sub) in nodes.items():
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="square,pad=0",
        ec="black",
        fc="white",
        lw=1.5
    )
    ax.add_patch(box)
    
    # Text inside box
    ax.text(x + w / 2.0, y + h * 0.62, title, ha='center', va='center', fontsize=10.5, fontweight='bold')
    ax.text(x + w / 2.0, y + h * 0.30, sub, ha='center', va='center', fontsize=9.5, color='#333333')

# Draw Flow Arrows
arrows = [
    ((3.5, 5.2), (4.0, 5.2)),       # data -> val
    ((7.2, 5.2), (7.6, 5.2)),       # val -> feat
    ((9.2, 4.6), (9.2, 3.8)),       # feat -> train (down)
    ((7.6, 3.2), (7.2, 3.2)),       # train -> prep
    ((4.0, 3.2), (3.5, 3.2)),       # prep -> split
    ((2.0, 2.6), (2.0, 1.8)),       # split -> eval (down)
    ((3.5, 1.2), (4.0, 1.2)),       # eval -> xai
    ((7.2, 1.2), (7.6, 1.2))        # xai -> app
]

for start, end in arrows:
    arrow = FancyArrowPatch(
        start, end,
        arrowstyle='-|>',
        mutation_scale=18,
        color='black',
        lw=1.5
    )
    ax.add_patch(arrow)

plt.tight_layout()
plt.savefig('system_architecture.png', bbox_inches='tight')
plt.close()
print("✓ Successfully generated 'system_architecture.png'")