from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from gunfolds.utils import zickle as zkl
from os import listdir
from matplotlib.patheffects import withStroke
from matplotlib.ticker import MultipleLocator


#7 CPUs
# memories = {'handy' : [940024000, 1990172000, 4836284000, 4836360000, 940120000, 5295232000, 1923124000],
# 'crafty' : [1464312000, 2381996000, 4901820000, 4837104000, 5360852000, 2709560000, 4836316000],
# 'auto' : [80684000, 6174236000, 3225068000, 4812628000, 1193540000, 4601372000, 1259036000],
# 'tweety' : [931304000, 5649948000, 5533520000, 4812628000, 1473360000, 6385504000, 1390108000],
# 'many' : [80684000, 13059128000, 6123348000, 4878160000, 2835560000, 5336924000, 1262692000],
# 'frumpy' : [45033200000, 36250636000, 90095000000, 80684000, 4811552000, 5074584000, 2891196000],
# 'jumpy' : [940020000, 1005696000, 939932000, 4836644000, 940216000, 4967428000, 940084000],
# 'trendy' : [940016000, 1398932000, 4836284000, 4836360000, 4836376000, 5164044000, 940084000]}

#15 CPUs
memories = {'handy' : [80684000, 1529980000, 4836532000, 80948000, 80948000, 2250824000, 80948000],
'crafty' : [80684000, 2644132000, 1750000000, 80948000, 80948000, 5196776000, 80948000],
'auto' : [80684000, 7585568000, 5140308000, 80684000, 2023952000, 6012700000, 1621796000],
'tweety' : [80684000, 7388960000, 5271380000, 80948000, 80948000, 5357340000, 80948000],
'many' : [80684000, 20247988000, 20444516000, 4353656000, 3397116000, 20708996000, 5074796000],
'frumpy' : [80684000, 78823388000, 55682608000, 4812632000, 3778248000, 54699704000, 2860736000],
'jumpy' : [80684000, 1530044000, 1529880000, 4830320000, 1530032000, 1529924000, 1529968000],
'trendy' : [80684000, 1529980000, 1529832000, 80948000, 80948000, 5131244000, 1529920000]}


list_of_lists1 = []
list_of_lists2 = []
folder = '/Users/sajad/Code_local/mygit/gunfolds/gunfolds/scripts/results/test_configuration_options/zkl/crafty/zkl'
directories = listdir(folder)
directories.sort()
if directories[0].startswith('.'):
    directories.pop(0)
directories = directories[-1:] + directories[:-1]
for directory in directories:
    file_list1 = listdir(f'{folder}/{directory}/')
    file_list1.sort()
    if file_list1[0].startswith('.'):
        file_list1.pop(0)
    item_list = ([zkl.load(f'{folder}/{directory}/{name}') for name in file_list1],directory) 
    list_of_lists1.append(item_list)


# for directory in directories:
#     file_list1 = listdir(f'{folder}/{directory}/15')
#     file_list1.sort()
#     if file_list1[0].startswith('.'):
#         file_list1.pop(0)
#     item_list = [zkl.load(f'{folder}/{directory}/15/{name}') for name in file_list1]
#     list_of_lists2.append(item_list)

if __name__ == '__main__':
    df = pd.DataFrame()
    Err = []
    config = []
    time = []
    pnum = []
    # for item_list in list_of_lists2:
    #     for index, item in enumerate(item_list):
    #         time.append(memories[item['config']][index])
    #         config.append(item['config'])
    #         Err.append((item['Gu_opt_errors_network_GT_U'][0] + item['Gu_opt_errors_network_GT_U'][1])/2)
    #         pnum.append(15)

    for index, item_list in enumerate(list_of_lists1):
        for item in item_list[0]:
            time.append(item['total_time'])
            config.append(item_list[1])


min_range = 1e7
max_range = 1e11
min_value = min(time)
max_value = max(time)

# normalized_times = [((x - min_value) / (max_value - min_value))* (max_range - min_range) + min_range for x in time]


# Create a new DataFrame
data = pd.DataFrame({'config': config, 'time':time})

# Create a box plot using Seaborn
plt.figure(figsize=(10, 6))  # Adjust figure size if needed
sns.boxplot(x='config', y='time', data=data, palette='Set1')

# Set labels and title
plt.xlabel('Configuration')
plt.ylabel('time')
plt.title('Box Plot of time by Configuration')
plt.yscale('log')  # Set the y-axis scale to log


# Add legend
plt.legend(title='Config', loc='upper right')

# Show the
# plt.savefig("only15_memory.svg")
plt.show()
