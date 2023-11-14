from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from gunfolds.utils import zickle as zkl
from os import listdir
from matplotlib.ticker import MultipleLocator
from gunfolds.viz import gtool as gt


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
############################################################

folder6 = '/Users/sajad/Code_local/mygit/gunfolds/gunfolds/scripts/results/VAR_simulation_results/optN/soft_weights' \
          '/res_simulation/8nodes/n8sfmf14/'
folder4 = '/Users/sajad/Code_local/mygit/gunfolds/gunfolds/scripts/results/VAR_simulation_results/optN/soft_weights' \
          '/res_simulation/8nodes/n8stmt14/'
folder5 = '/Users/sajad/Code_local/mygit/gunfolds/gunfolds/scripts/results/VAR_simulation_results/optN/soft_weights' \
          '/res_simulation/8nodes/n8stmf14/'

file_list6 = listdir(folder6)
file_list6.sort()
if file_list6[0].startswith('.'):
    file_list6.pop(0)

file_list4 = listdir(folder4)
file_list4.sort()
if file_list4[0].startswith('.'):
    file_list4.pop(0)

file_list5 = listdir(folder5)
file_list5.sort()
if file_list5[0].startswith('.'):
    file_list5.pop(0)

res6 = [zkl.load(folder6 + file) for file in file_list6]
res4 = [zkl.load(folder4 + file) for file in file_list4]
res5 = [zkl.load(folder5 + file) for file in file_list5]


############################################################

folder9 = '/Users/sajad/Code_local/mygit/gunfolds/gunfolds/scripts/results/VAR_simulation_results/optN/soft_weight_no_priority' \
          '/res_simulation/8nodes/n8sfmf14/'
folder7 = '/Users/sajad/Code_local/mygit/gunfolds/gunfolds/scripts/results/VAR_simulation_results/optN/soft_weight_no_priority' \
          '/res_simulation/8nodes/n8stmt14/'
folder8 = '/Users/sajad/Code_local/mygit/gunfolds/gunfolds/scripts/results/VAR_simulation_results/optN/soft_weight_no_priority' \
          '/res_simulation/8nodes/n8stmf14/'

file_list9 = listdir(folder9)
file_list9.sort()
if file_list9[0].startswith('.'):
    file_list9.pop(0)

file_list7 = listdir(folder7)
file_list7.sort()
if file_list7[0].startswith('.'):
    file_list7.pop(0)

file_list8 = listdir(folder8)
file_list8.sort()
if file_list8[0].startswith('.'):
    file_list8.pop(0)

res9 = [zkl.load(folder9 + file) for file in file_list9]
res7 = [zkl.load(folder7 + file) for file in file_list7]
res8 = [zkl.load(folder8 + file) for file in file_list8]

############################################################

folder12 = '/Users/sajad/Code_local/mygit/gunfolds/gunfolds/scripts/results/VAR_simulation_results/optN/ringmore' \
          '/res_simulation/7nodes/'
# folder10 = '/Users/sajad/Code_local/mygit/gunfolds/gunfolds/scripts/results/VAR_simulation_results/optN/soft_weight_no_priority' \
#           '/res_simulation/8nodes/n8stmt14/'
# folder11 = '/Users/sajad/Code_local/mygit/gunfolds/gunfolds/scripts/results/VAR_simulation_results/optN/soft_weight_no_priority' \
#           '/res_simulation/8nodes/n8stmf14/'

file_list12 = listdir(folder12)
file_list12.sort()
if file_list12[0].startswith('.'):
    file_list12.pop(0)

# file_list10 = listdir(folder10)
# file_list10.sort()
# if file_list10[0].startswith('.'):
#     file_list10.pop(0)

# file_list11 = listdir(folder11)
# file_list11.sort()
# if file_list11[0].startswith('.'):
#     file_list11.pop(0)

res12 = [zkl.load(folder12 + file) for file in file_list12]
# res10 = [zkl.load(folder10 + file) for file in file_list10]
# res11 = [zkl.load(folder11 + file) for file in file_list11]

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
    weights_scheme = []

    for item in res3+res+res2:
        ErrVs.extend([ 'GuVsGTu', 'GuVsGest','G1VsGT'])
        WRT.extend(['GuOptVsGest','GuOptVsGest','GuOptVsGest'])
        Err.extend([item['GuOptVsGest']['Gu_opt_errors_network_GT_U_WRT_GuOptVsGest'][0],
                    item['GuOptVsGest']['Gu_opt_errors_g_estimated_WRT_GuOptVsGest'][0],
                    item['GuOptVsGest']['G1_opt_error_GT_WRT_GuOptVsGest'][0]])
        ErrType.extend(['omm', 'omm','omm'])
        weights_scheme.extend(['hard','hard','hard'])
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['GuOptVsGest', 'GuOptVsGest', 'GuOptVsGest'])
        Err.extend([item['GuOptVsGest']['Gu_opt_errors_network_GT_U_WRT_GuOptVsGest'][1],
                    item['GuOptVsGest']['Gu_opt_errors_g_estimated_WRT_GuOptVsGest'][1],
                    item['GuOptVsGest']['G1_opt_error_GT_WRT_GuOptVsGest'][1]])
        ErrType.extend(['comm', 'comm', 'comm'])
        weights_scheme.extend(['hard', 'hard', 'hard'])
#################################################################################################
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['GuOptVsGTu', 'GuOptVsGTu', 'GuOptVsGTu'])
        Err.extend([item['GuOptVsGTu']['Gu_opt_errors_network_GT_U_WRT_GuOptVsGTu'][0],
                    item['GuOptVsGTu']['Gu_opt_errors_g_estimated_WRT_GuOptVsGTu'][0],
                    item['GuOptVsGTu']['G1_opt_error_GT_WRT_GuOptVsGTu'][0]])
        ErrType.extend(['omm', 'omm', 'omm'])
        weights_scheme.extend(['hard', 'hard', 'hard'])
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['GuOptVsGTu', 'GuOptVsGTu', 'GuOptVsGTu'])
        Err.extend([item['GuOptVsGTu']['Gu_opt_errors_network_GT_U_WRT_GuOptVsGTu'][1],
                    item['GuOptVsGTu']['Gu_opt_errors_g_estimated_WRT_GuOptVsGTu'][1],
                    item['GuOptVsGTu']['G1_opt_error_GT_WRT_GuOptVsGTu'][1]])
        ErrType.extend(['comm', 'comm', 'comm'])
        weights_scheme.extend(['hard', 'hard', 'hard'])
#################################################################################################
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['G1OptVsGT', 'G1OptVsGT', 'G1OptVsGT'])
        Err.extend([item['G1OptVsGT']['Gu_opt_errors_network_GT_U_WRT_G1OptVsGT'][0],
                    item['G1OptVsGT']['Gu_opt_errors_g_estimated_WRT_G1OptVsGT'][0],
                    item['G1OptVsGT']['G1_opt_error_GT_WRT_G1OptVsGT'][0]])
        ErrType.extend(['omm', 'omm', 'omm'])
        weights_scheme.extend(['hard', 'hard', 'hard'])
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['G1OptVsGT', 'G1OptVsGT', 'G1OptVsGT'])
        Err.extend([item['G1OptVsGT']['Gu_opt_errors_network_GT_U_WRT_G1OptVsGT'][1],
                    item['G1OptVsGT']['Gu_opt_errors_g_estimated_WRT_G1OptVsGT'][1],
                    item['G1OptVsGT']['G1_opt_error_GT_WRT_G1OptVsGT'][1]])
        ErrType.extend(['comm', 'comm', 'comm'])
        weights_scheme.extend(['hard', 'hard', 'hard'])

    for item in res6 + res4 + res5:
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['GuOptVsGest', 'GuOptVsGest', 'GuOptVsGest'])
        Err.extend([item['GuOptVsGest']['Gu_opt_errors_network_GT_U_WRT_GuOptVsGest'][0],
                    item['GuOptVsGest']['Gu_opt_errors_g_estimated_WRT_GuOptVsGest'][0],
                    item['GuOptVsGest']['G1_opt_error_GT_WRT_GuOptVsGest'][0]])
        ErrType.extend(['omm', 'omm', 'omm'])
        weights_scheme.extend(['soft', 'soft', 'soft'])
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['GuOptVsGest', 'GuOptVsGest', 'GuOptVsGest'])
        Err.extend([item['GuOptVsGest']['Gu_opt_errors_network_GT_U_WRT_GuOptVsGest'][1],
                    item['GuOptVsGest']['Gu_opt_errors_g_estimated_WRT_GuOptVsGest'][1],
                    item['GuOptVsGest']['G1_opt_error_GT_WRT_GuOptVsGest'][1]])
        ErrType.extend(['comm', 'comm', 'comm'])
        weights_scheme.extend(['soft', 'soft', 'soft'])
        #################################################################################################
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['GuOptVsGTu', 'GuOptVsGTu', 'GuOptVsGTu'])
        Err.extend([item['GuOptVsGTu']['Gu_opt_errors_network_GT_U_WRT_GuOptVsGTu'][0],
                    item['GuOptVsGTu']['Gu_opt_errors_g_estimated_WRT_GuOptVsGTu'][0],
                    item['GuOptVsGTu']['G1_opt_error_GT_WRT_GuOptVsGTu'][0]])
        ErrType.extend(['omm', 'omm', 'omm'])
        weights_scheme.extend(['soft', 'soft', 'soft'])
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['GuOptVsGTu', 'GuOptVsGTu', 'GuOptVsGTu'])
        Err.extend([item['GuOptVsGTu']['Gu_opt_errors_network_GT_U_WRT_GuOptVsGTu'][1],
                    item['GuOptVsGTu']['Gu_opt_errors_g_estimated_WRT_GuOptVsGTu'][1],
                    item['GuOptVsGTu']['G1_opt_error_GT_WRT_GuOptVsGTu'][1]])
        ErrType.extend(['comm', 'comm', 'comm'])
        weights_scheme.extend(['soft', 'soft', 'soft'])
        #################################################################################################
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['G1OptVsGT', 'G1OptVsGT', 'G1OptVsGT'])
        Err.extend([item['G1OptVsGT']['Gu_opt_errors_network_GT_U_WRT_G1OptVsGT'][0],
                    item['G1OptVsGT']['Gu_opt_errors_g_estimated_WRT_G1OptVsGT'][0],
                    item['G1OptVsGT']['G1_opt_error_GT_WRT_G1OptVsGT'][0]])
        ErrType.extend(['omm', 'omm', 'omm'])
        weights_scheme.extend(['soft', 'soft', 'soft'])
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['G1OptVsGT', 'G1OptVsGT', 'G1OptVsGT'])
        Err.extend([item['G1OptVsGT']['Gu_opt_errors_network_GT_U_WRT_G1OptVsGT'][1],
                    item['G1OptVsGT']['Gu_opt_errors_g_estimated_WRT_G1OptVsGT'][1],
                    item['G1OptVsGT']['G1_opt_error_GT_WRT_G1OptVsGT'][1]])
        ErrType.extend(['comm', 'comm', 'comm'])
        weights_scheme.extend(['soft', 'soft', 'soft'])


    for item in res9 + res7 + res8:
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['GuOptVsGest', 'GuOptVsGest', 'GuOptVsGest'])
        Err.extend([item['GuOptVsGest']['Gu_opt_errors_network_GT_U_WRT_GuOptVsGest'][0],
                    item['GuOptVsGest']['Gu_opt_errors_g_estimated_WRT_GuOptVsGest'][0],
                    item['GuOptVsGest']['G1_opt_error_GT_WRT_GuOptVsGest'][0]])
        ErrType.extend(['omm', 'omm', 'omm'])
        weights_scheme.extend(['same_priority', 'same_priority', 'same_priority'])
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['GuOptVsGest', 'GuOptVsGest', 'GuOptVsGest'])
        Err.extend([item['GuOptVsGest']['Gu_opt_errors_network_GT_U_WRT_GuOptVsGest'][1],
                    item['GuOptVsGest']['Gu_opt_errors_g_estimated_WRT_GuOptVsGest'][1],
                    item['GuOptVsGest']['G1_opt_error_GT_WRT_GuOptVsGest'][1]])
        ErrType.extend(['comm', 'comm', 'comm'])
        weights_scheme.extend(['same_priority', 'same_priority', 'same_priority'])
        #################################################################################################
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['GuOptVsGTu', 'GuOptVsGTu', 'GuOptVsGTu'])
        Err.extend([item['GuOptVsGTu']['Gu_opt_errors_network_GT_U_WRT_GuOptVsGTu'][0],
                    item['GuOptVsGTu']['Gu_opt_errors_g_estimated_WRT_GuOptVsGTu'][0],
                    item['GuOptVsGTu']['G1_opt_error_GT_WRT_GuOptVsGTu'][0]])
        ErrType.extend(['omm', 'omm', 'omm'])
        weights_scheme.extend(['same_priority', 'same_priority', 'same_priority'])
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['GuOptVsGTu', 'GuOptVsGTu', 'GuOptVsGTu'])
        Err.extend([item['GuOptVsGTu']['Gu_opt_errors_network_GT_U_WRT_GuOptVsGTu'][1],
                    item['GuOptVsGTu']['Gu_opt_errors_g_estimated_WRT_GuOptVsGTu'][1],
                    item['GuOptVsGTu']['G1_opt_error_GT_WRT_GuOptVsGTu'][1]])
        ErrType.extend(['comm', 'comm', 'comm'])
        weights_scheme.extend(['same_priority', 'same_priority', 'same_priority'])
        #################################################################################################
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['G1OptVsGT', 'G1OptVsGT', 'G1OptVsGT'])
        Err.extend([item['G1OptVsGT']['Gu_opt_errors_network_GT_U_WRT_G1OptVsGT'][0],
                    item['G1OptVsGT']['Gu_opt_errors_g_estimated_WRT_G1OptVsGT'][0],
                    item['G1OptVsGT']['G1_opt_error_GT_WRT_G1OptVsGT'][0]])
        ErrType.extend(['omm', 'omm', 'omm'])
        weights_scheme.extend(['same_priority', 'same_priority', 'same_priority'])
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['G1OptVsGT', 'G1OptVsGT', 'G1OptVsGT'])
        Err.extend([item['G1OptVsGT']['Gu_opt_errors_network_GT_U_WRT_G1OptVsGT'][1],
                    item['G1OptVsGT']['Gu_opt_errors_g_estimated_WRT_G1OptVsGT'][1],
                    item['G1OptVsGT']['G1_opt_error_GT_WRT_G1OptVsGT'][1]])
        ErrType.extend(['comm', 'comm', 'comm'])
        weights_scheme.extend(['same_priority', 'same_priority', 'same_priority'])


    for item in res12:
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['GuOptVsGest', 'GuOptVsGest', 'GuOptVsGest'])
        Err.extend([item['GuOptVsGest']['Gu_opt_errors_network_GT_U_WRT_GuOptVsGest'][0],
                    item['GuOptVsGest']['Gu_opt_errors_g_estimated_WRT_GuOptVsGest'][0],
                    item['GuOptVsGest']['G1_opt_error_GT_WRT_GuOptVsGest'][0]])
        ErrType.extend(['omm', 'omm', 'omm'])
        weights_scheme.extend(['ringmore', 'ringmore', 'ringmore'])
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['GuOptVsGest', 'GuOptVsGest', 'GuOptVsGest'])
        Err.extend([item['GuOptVsGest']['Gu_opt_errors_network_GT_U_WRT_GuOptVsGest'][1],
                    item['GuOptVsGest']['Gu_opt_errors_g_estimated_WRT_GuOptVsGest'][1],
                    item['GuOptVsGest']['G1_opt_error_GT_WRT_GuOptVsGest'][1]])
        ErrType.extend(['comm', 'comm', 'comm'])
        weights_scheme.extend(['ringmore', 'ringmore', 'ringmore'])
        #################################################################################################
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['GuOptVsGTu', 'GuOptVsGTu', 'GuOptVsGTu'])
        Err.extend([item['GuOptVsGTu']['Gu_opt_errors_network_GT_U_WRT_GuOptVsGTu'][0],
                    item['GuOptVsGTu']['Gu_opt_errors_g_estimated_WRT_GuOptVsGTu'][0],
                    item['GuOptVsGTu']['G1_opt_error_GT_WRT_GuOptVsGTu'][0]])
        ErrType.extend(['omm', 'omm', 'omm'])
        weights_scheme.extend(['ringmore', 'ringmore', 'ringmore'])
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['GuOptVsGTu', 'GuOptVsGTu', 'GuOptVsGTu'])
        Err.extend([item['GuOptVsGTu']['Gu_opt_errors_network_GT_U_WRT_GuOptVsGTu'][1],
                    item['GuOptVsGTu']['Gu_opt_errors_g_estimated_WRT_GuOptVsGTu'][1],
                    item['GuOptVsGTu']['G1_opt_error_GT_WRT_GuOptVsGTu'][1]])
        ErrType.extend(['comm', 'comm', 'comm'])
        weights_scheme.extend(['ringmore', 'ringmore', 'ringmore'])
        #################################################################################################
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['G1OptVsGT', 'G1OptVsGT', 'G1OptVsGT'])
        Err.extend([item['G1OptVsGT']['Gu_opt_errors_network_GT_U_WRT_G1OptVsGT'][0],
                    item['G1OptVsGT']['Gu_opt_errors_g_estimated_WRT_G1OptVsGT'][0],
                    item['G1OptVsGT']['G1_opt_error_GT_WRT_G1OptVsGT'][0]])
        ErrType.extend(['omm', 'omm', 'omm'])
        weights_scheme.extend(['ringmore', 'ringmore', 'ringmore'])
        ErrVs.extend(['GuVsGTu', 'GuVsGest', 'G1VsGT'])
        WRT.extend(['G1OptVsGT', 'G1OptVsGT', 'G1OptVsGT'])
        Err.extend([item['G1OptVsGT']['Gu_opt_errors_network_GT_U_WRT_G1OptVsGT'][1],
                    item['G1OptVsGT']['Gu_opt_errors_g_estimated_WRT_G1OptVsGT'][1],
                    item['G1OptVsGT']['G1_opt_error_GT_WRT_G1OptVsGT'][1]])
        ErrType.extend(['comm', 'comm', 'comm'])
        weights_scheme.extend(['ringmore', 'ringmore', 'ringmore'])


    df['Err'] = Err
    df['ErrVs'] = ErrVs
    df['ErrType'] = ErrType
    df['WRT'] = WRT
    df['weights_scheme'] = weights_scheme

    sns.set({"xtick.minor.size": 0.2})
    pal = dict(soft="gold", hard="blue",
               same_priority="maroon", ringmore="green")
    g = sns.FacetGrid(df, col="WRT", row="ErrType", height=4, aspect=1, margin_titles=True)


    def custom_boxplot(*args, **kwargs):
        sns.boxplot(*args, **kwargs, palette=pal)


    g.map_dataframe(custom_boxplot, x='ErrVs', y='Err', hue='weights_scheme')
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
