import matplotlib.pyplot as plt
from matplotlib import gridspec

# Data for MSR-VTT and MSVD
percentages = [0.05, 0.25, 0.50, 0.75, 1.00]

# MSR-VTT scores
msr_vtt_b1 = [73.8, 78.3, 79.1, 78.9, 80.2]
msr_vtt_b4 = [27.2, 30.7, 32.3, 31.8, 33.3]
msr_vtt_m = [23.7, 25.5, 26.2, 26.1, 26.6]
msr_vtt_rl = [52.1, 53.9, 55.1, 54.7, 55.3]
msr_vtt_cider = [32.9, 39.9, 41.7, 41.6, 43.4]
msr_vtt_spice = [6.1, 6.8, 7.1, 7.1, 7.3]

# MSVD scores
msvd_b1 = [68.0, 77.1, 78.3, 79.7, 80.9]
msvd_b4 = [30.2, 42.1, 44.2, 47.6, 50.1]
msvd_m = [28.9, 34.6, 35.7, 37.0, 38.0]
msvd_rl = [61.5, 67.9, 69.2, 70.9, 71.9]
msvd_cider = [34.1, 71.3, 78.2, 83.0, 91.5]
msvd_spice = [5.0, 6.9, 7.4, 8.0, 8.3]

# Marker styles for each metric
markers = {
    'B@1': 'o',
    'B@4': 's',
    'M': '^',
    'R-L': 'D',
    'CIDEr': 'v',
    'SPICE': 'P'
}

# Create a figure with subfigures
fig = plt.figure(figsize=(14, 6))
gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])

# First subfigure (MSR-VTT)
ax0 = plt.subplot(gs[0])
ax0.plot(percentages, msr_vtt_b1, label="B@1", marker=markers['B@1'])
ax0.plot(percentages, msr_vtt_b4, label="B@4", marker=markers['B@4'])
ax0.plot(percentages, msr_vtt_m, label="M", marker=markers['M'])
ax0.plot(percentages, msr_vtt_rl, label="R-L", marker=markers['R-L'])
ax0.plot(percentages, msr_vtt_cider, label="CIDEr", marker=markers['CIDEr'])
ax0.plot(percentages, msr_vtt_spice, label="SPICE", marker=markers['SPICE'])

ax0.grid(True, linestyle='--', linewidth=1)
ax0.set_title("MSR-VTT Dataset", fontsize=20)
ax0.tick_params(axis='both', labelsize=20)  # Adjust font size for MSR-VTT

fig.legend(loc='center', ncol=6, bbox_to_anchor=(0.5, -0.05), frameon=False, fontsize=20)

# Second subfigure (MSVD)
ax1 = plt.subplot(gs[1])
ax1.plot(percentages, msvd_b1, label="B@1", marker=markers['B@1'])
ax1.plot(percentages, msvd_b4, label="B@4", marker=markers['B@4'])
ax1.plot(percentages, msvd_m, label="M", marker=markers['M'])
ax1.plot(percentages, msvd_rl, label="R-L", marker=markers['R-L'])
ax1.plot(percentages, msvd_cider, label="CIDEr", marker=markers['CIDEr'])
ax1.plot(percentages, msvd_spice, label="SPICE", marker=markers['SPICE'])

ax1.grid(True, linestyle='--', linewidth=1)
ax1.set_title("MSVD Dataset", fontsize=20)
ax1.tick_params(axis='both', labelsize=20)  # Adjust font size for ticks
# Add a global legen

# Adjust layout
plt.tight_layout()

# Save the plot as a PDF
plt.savefig('ablation_study_plots_subfigure.pdf', bbox_inches='tight')
