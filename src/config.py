import matplotlib.pyplot as plt
import seaborn as sns

path_szb_commons = 'C:/Users/szb37/My Drive/Work efforts/szb_commons/'

### Graphics settings
save_PNG = True
save_SVG = True
plt.rcParams.update({'font.family': 'arial'})
title_fontdict = {'fontsize': 18, 'fontweight': 'bold'}
axislabel_fontdict = {'fontsize': 16, 'fontweight': 'bold'}
ticklabel_fontsize = 14
sns.set_style("whitegrid")

errorbar='se'
err_kws={'capsize': 4, 'elinewidth': 0.75,'capthick': 0.75}

### Miscs
tps = ['bsl','A1', 'B1', 'B30']
parkers_csvs = ['infindex_exploratory.csv', 'infindex_inflammatory.csv', 'infindex_mitokines.csv']

# timepoint to number of days conversion
tp_to_ndays={ # Based form the PDP1 paper, average number of days between tps
    'bsl': 0,
    'A1': 18,
    'B1': 18+14,
    'B30': 18+14+30,}        

### Marker groupings / order
#   Markers excluded due to poor data quality: eotaxin3, MIP-1alpha, IL-17, IL-31, IL-23, IL17-alpha, BDNF

markers_mitokines = ['FGF-21', 'GDF-15']

markers_infind_comps = ['IL-6', 'IL-8', 'TNF-alpha', 'CRP', 'IFN-gamma',]
markers_inflamation_miscs = ['IL-10', 'sIL-6R',]
markers_inflamation = ['infindex'] + markers_infind_comps + markers_inflamation_miscs

markers_explorat1 = ['IL-7', 'IL-12', 'IL-15', 'IL-16','IL-27', 'TSLP', 'IL-1RA',]
markers_explorat2 = ['MCP-1', 'MIP-1beta', 'eotaxin', 'MCP-4', 'TARC', 'MDC', 'IP-10',]
markers_explorat3 = ['VEGF', 'VEGF-C', 'VEGF-D', 'PlGF', 'bFGF','TIE-2', 'FLT-1']

markers_explorat = markers_explorat1 + markers_explorat2 + markers_explorat3
markers_main = markers_inflamation + markers_mitokines 
markers = markers_main + markers_explorat

heatmap_groups = {
    'infindex': ['infindex'],
    'inflam_comps': markers_infind_comps,
    'inflam_mics': markers_inflamation_miscs,
    'mitokines': markers_mitokines,
    'explore1': markers_explorat1,
    'explore2': markers_explorat2,
    'explore3': markers_explorat3,}