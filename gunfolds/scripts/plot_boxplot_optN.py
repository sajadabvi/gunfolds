from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from gunfolds.utils import zickle as zkl
from os import listdir
from matplotlib.ticker import MultipleLocator


folder3 = '/Users/sajad/Code_local/mygit/gunfolds/gunfolds/scripts/results/VAR_simulation_results/optN/res_simulation' \
         '/8nodes/n8sfmf14/'

folder = '/Users/sajad/Code_local/mygit/gunfolds/gunfolds/scripts/results/VAR_simulation_results/optN/res_simulation' \
         '/8nodes/n8stmt14/'
folder2 = '/Users/sajad/Code_local/mygit/gunfolds/gunfolds/scripts/results/VAR_simulation_results/optN/res_simulation' \
         '/8nodes/n8stmf14/'


file_list3 = listdir(folder3)
file_list3.sort()
if file_list3[0].startswith('.'):
    file_list3.pop(0)

file_list = listdir(folder)
file_list.sort()
if file_list[0].startswith('.'):
    file_list.pop(0)

file_list2 = listdir(folder2)
file_list2.sort()
if file_list2[0].startswith('.'):
    file_list2.pop(0)

res3 = [zkl.load(folder3 + file) for file in file_list3]
res = [zkl.load(folder + file) for file in file_list]
res2 = [zkl.load(folder2 + file) for file in file_list2]


G1_opt_error_GT_om = []
G1_opt_error_GT_com = []
Gu_opt_errors_network_GT_U_om = []
Gu_opt_errors_network_GT_U_com = []
Gu_opt_errors_g_estimated_om = []
Gu_opt_errors_g_estimated_com = []


# G1_opt_error_GT_om_WRT_GuOptVsGTu = []
# G1_opt_error_GT_com_WRT_GuOptVsGTu = []
# Gu_opt_errors_network_GT_U_om_WRT_GuOptVsGTu = []
# Gu_opt_errors_network_GT_U_com_WRT_GuOptVsGTu = []
# Gu_opt_errors_g_estimated_om_WRT_GuOptVsGTu = []
# Gu_opt_errors_g_estimated_com_WRT_GuOptVsGTu = []
#
#
# G1_opt_error_GT_om_WRT_G1OptVsGT = []
# G1_opt_error_GT_com_WRT_G1OptVsGT = []
# Gu_opt_errors_network_GT_U_om_WRT_G1OptVsGT = []
# Gu_opt_errors_network_GT_U_com_WRT_G1OptVsGT = []
# Gu_opt_errors_g_estimated_om_WRT_G1OptVsGT = []
# Gu_opt_errors_g_estimated_com_WRT_G1OptVsGT = []


if __name__ == '__main__':
    df = pd.DataFrame()
    Err = []
    method = []
    ErrType = []
    u = []
    deg = []
    node = []
    WRT = []
    ErrVs = []

    for item in res3+res+res2:
        ErrVs.extend([ 'GuVsGest', 'GuVsGTu','G1VsGT'])
        WRT.extend(['GuOptVsGest','GuOptVsGest','GuOptVsGest'])
        Err.extend([item['GuOptVsGest']['Gu_opt_errors_network_GT_U_WRT_GuOptVsGest'][0],
                    item['GuOptVsGest']['Gu_opt_errors_g_estimated_WRT_GuOptVsGest'][0],
                    item['GuOptVsGest']['G1_opt_error_GT_WRT_GuOptVsGest'][0]])
        ErrType.extend(['omm', 'omm','omm'])
        ErrVs.extend(['GuVsGest', 'GuVsGTu', 'G1VsGT'])
        WRT.extend(['GuOptVsGest', 'GuOptVsGest', 'GuOptVsGest'])
        Err.extend([item['GuOptVsGest']['Gu_opt_errors_network_GT_U_WRT_GuOptVsGest'][1],
                    item['GuOptVsGest']['Gu_opt_errors_g_estimated_WRT_GuOptVsGest'][1],
                    item['GuOptVsGest']['G1_opt_error_GT_WRT_GuOptVsGest'][1]])
        ErrType.extend(['comm', 'comm', 'comm'])
#################################################################################################
        ErrVs.extend(['GuVsGest', 'GuVsGTu', 'G1VsGT'])
        WRT.extend(['GuOptVsGTu', 'GuOptVsGTu', 'GuOptVsGTu'])
        Err.extend([item['GuOptVsGTu']['Gu_opt_errors_network_GT_U_WRT_GuOptVsGTu'][0],
                    item['GuOptVsGTu']['Gu_opt_errors_g_estimated_WRT_GuOptVsGTu'][0],
                    item['GuOptVsGTu']['G1_opt_error_GT_WRT_GuOptVsGTu'][0]])
        ErrType.extend(['omm', 'omm', 'omm'])
        ErrVs.extend(['GuVsGest', 'GuVsGTu', 'G1VsGT'])
        WRT.extend(['GuOptVsGTu', 'GuOptVsGTu', 'GuOptVsGTu'])
        Err.extend([item['GuOptVsGTu']['Gu_opt_errors_network_GT_U_WRT_GuOptVsGTu'][1],
                    item['GuOptVsGTu']['Gu_opt_errors_g_estimated_WRT_GuOptVsGTu'][1],
                    item['GuOptVsGTu']['G1_opt_error_GT_WRT_GuOptVsGTu'][1]])
        ErrType.extend(['comm', 'comm', 'comm'])
#################################################################################################
        ErrVs.extend(['GuVsGest', 'GuVsGTu', 'G1VsGT'])
        WRT.extend(['G1OptVsGT', 'G1OptVsGT', 'G1OptVsGT'])
        Err.extend([item['G1OptVsGT']['Gu_opt_errors_network_GT_U_WRT_G1OptVsGT'][0],
                    item['G1OptVsGT']['Gu_opt_errors_g_estimated_WRT_G1OptVsGT'][0],
                    item['G1OptVsGT']['G1_opt_error_GT_WRT_G1OptVsGT'][0]])
        ErrType.extend(['omm', 'omm', 'omm'])
        ErrVs.extend(['GuVsGest', 'GuVsGTu', 'G1VsGT'])
        WRT.extend(['G1OptVsGT', 'G1OptVsGT', 'G1OptVsGT'])
        Err.extend([item['G1OptVsGT']['Gu_opt_errors_network_GT_U_WRT_G1OptVsGT'][1],
                    item['G1OptVsGT']['Gu_opt_errors_g_estimated_WRT_G1OptVsGT'][1],
                    item['G1OptVsGT']['G1_opt_error_GT_WRT_G1OptVsGT'][1]])
        ErrType.extend(['comm', 'comm', 'comm'])



    df['Err'] = Err
    df['ErrVs'] = ErrVs
    df['ErrType'] = ErrType
    df['WRT'] = WRT

    sns.set({"xtick.minor.size": 0.2})
    pal = dict(old_opt="gold", Capped_optim_the_sRASL="maroon",
               new_optN="blue", optim_then_sRASL="green")
    g = sns.FacetGrid(df, col="WRT", row="ErrType", height=4, aspect=1, margin_titles=True)


    def custom_boxplot(*args, **kwargs):
        sns.boxplot(*args, **kwargs)


    g.map_dataframe(custom_boxplot, x='ErrVs', y='Err')
    g.add_legend()
    g.set_axis_labels("error type", "normalized error")

    for i in range(g.axes.shape[0]):
        for j in range(g.axes.shape[1]):
            ax = g.facet_axis(i, j)
            # ax.xaxis.grid(True, "minor", linewidth=.75)
            # ax.xaxis.grid(True, "major", linewidth=3)
            # ax.xaxis.set_major_locator(MultipleLocator(1))
            ax.set_ylim(0, 1)

    plt.show()
    # plt.savefig("figs/weighted_experiment_optN_all_four_ways_undersampling2.svg")
