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
#
# for directory in node_directories2:
#     file_list = listdir(f'./res_drasl_after_optim/{directory}')
#     file_list.sort()
#     if file_list[0].startswith('.'):
#         file_list.pop(0)
#     item_list2 = [zkl.load(f'./res_drasl_after_optim/{directory}/{name}') for name in file_list]
#     list_of_lists2.append(item_list2)

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

    for item in res3 + res + res2:
        ErrVs.extend(['GuVsGest', 'GuVsGTu', 'G1VsGT'])
        WRT.extend(['GuOptVsGest', 'GuOptVsGest', 'GuOptVsGest'])
        Err.extend([item['GuOptVsGest']['Gu_opt_errors_network_GT_U_WRT_GuOptVsGest'][0],
                    item['GuOptVsGest']['Gu_opt_errors_g_estimated_WRT_GuOptVsGest'][0],
                    item['GuOptVsGest']['G1_opt_error_GT_WRT_GuOptVsGest'][0]])
        ErrType.extend(['omm', 'omm', 'omm'])
        ErrVs.extend(['GuVsGest', 'GuVsGTu', 'G1VsGT'])
        WRT.extend(['GuOptVsGest', 'GuOptVsGest', 'GuOptVsGest'])
        Err.extend([item['GuOptVsGest']['Gu_opt_errors_network_GT_U_WRT_GuOptVsGest'][1],
                    item['GuOptVsGest']['Gu_opt_errors_g_estimated_WRT_GuOptVsGest'][1],
                    item['GuOptVsGest']['G1_opt_error_GT_WRT_GuOptVsGest'][1]])
        ErrType.extend(['comm', 'comm', 'comm'])

    for index, item_list in enumerate(list_of_lists4, start=6):
        for item in item_list:
            if item['u'] == undersampling and item['density'] in deg_list:
                Err.extend([item['normed_errors']['total'][0], item['normed_errors']['total'][1]])
                ErrType.extend(['omm', 'comm'])
                method.extend(['old_opt'] * 2)
                u.extend([item['u'], item['u']])
                deg.extend([item['density'], item['density']])
                node.extend([str(index)] * 2)

    for index, item_list in enumerate(list_of_lists, start=6):
        for item in item_list:
            if item['u'] == undersampling and item['density'] in deg_list:
                Err.extend([item['normed_errors']['total'][0], item['normed_errors']['total'][1]])
                ErrType.extend(['omm', 'comm'])
                method.extend(['new_optN'] * 2)
                u.extend([item['u'], item['u']])
                deg.extend([item['density'], item['density']])
                node.extend([str(index)] * 2)

    # for index, item_list2 in enumerate(list_of_lists2, start=6):
    #     for item in item_list2:
    #         if item[0]['u'] == undersampling:
    #             Err.extend([item[1]['min_norm_err'][0], item[1]['min_norm_err'][1]])
    #             ErrType.extend(['omm', 'comm'])
    #             method.extend(['optim_then_sRASL'] * 2)
    #             u.extend([item[0]['u'], item[0]['u']])
    #             deg.extend([item[0]['density'], item[0]['density']])
    #             node.extend([str(index)] * 2)

    df['Err'] = Err
    df['method'] = method
    df['ErrType'] = ErrType
    df['deg'] = deg
    df['node'] = node
    df['u'] = u

    sns.set({"xtick.minor.size": 0.2})
    pal = dict(old_opt="gold", Capped_optim_the_sRASL="maroon",
               new_optN="blue", optim_then_sRASL="green")
    g = sns.FacetGrid(df, col="deg", row="ErrType", height=4, aspect=0.5, margin_titles=True)


    def custom_boxplot(*args, **kwargs):
        sns.boxplot(*args, **kwargs, palette=pal)


    g.map_dataframe(custom_boxplot, x='node', y='Err', hue="method")
    g.add_legend()
    g.set_axis_labels("Number of node", "normalized error")

    for i in range(g.axes.shape[0]):
        for j in range(g.axes.shape[1]):
            ax = g.facet_axis(i, j)
            # ax.xaxis.grid(True, "minor", linewidth=.75)
            # ax.xaxis.grid(True, "major", linewidth=3)
            # ax.xaxis.set_major_locator(MultipleLocator(1))
            ax.set_ylim(0, 1)

    plt.show()
    # plt.savefig("figs/weighted_experiment_optN_all_four_ways_undersampling2.svg")
