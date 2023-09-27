from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from gunfolds.utils import zickle as zkl
from os import listdir
from matplotlib.patheffects import withStroke
from matplotlib.ticker import MultipleLocator

list_of_lists1 = []
list_of_lists2 = []
folder = '/Users/sajad/Code_local/mygit/gunfolds/gunfolds/scripts/results/test_configuration_options/zkl'
directories = listdir(folder)
directories.sort()
if directories[0].startswith('.'):
    directories.pop(0)

for directory in directories:
    file_list1 = listdir(f'{folder}/{directory}/7')
    file_list1.sort()
    if file_list1[0].startswith('.'):
        file_list1.pop(0)
    item_list = [zkl.load(f'{folder}/{directory}/7/{name}') for name in file_list1]
    list_of_lists1.append(item_list)


for directory in directories:
    file_list1 = listdir(f'{folder}/{directory}/15')
    file_list1.sort()
    if file_list1[0].startswith('.'):
        file_list1.pop(0)
    item_list = [zkl.load(f'{folder}/{directory}/15/{name}') for name in file_list1]
    list_of_lists2.append(item_list)

if __name__ == '__main__':
    df = pd.DataFrame()
    Err = []
    config = []
    time = []
    pnum = []
    for index, item_list in enumerate(list_of_lists1):
        for item in item_list:
            time.append(item['total_time'])
            config.append(item['config'])
            Err.append((item['Gu_opt_errors_network_GT_U'][0] + item['Gu_opt_errors_network_GT_U'][1])/2)
            pnum.append(7)

    # for index, item_list in enumerate(list_of_lists2):
    #     for item in item_list:
    #         time.append(item['total_time'])
    #         config.append(item['config'])
    #         Err.append((item['Gu_opt_errors_network_GT_U'][0] + item['Gu_opt_errors_network_GT_U'][1])/2)
    #         pnum.append(15)


# Create a new DataFrame
data = pd.DataFrame({'config': config, 'Err': Err, 'pnum': pnum, 'time':time})

# Create a box plot using Seaborn
plt.figure(figsize=(10, 6))  # Adjust figure size if needed
sns.boxplot(x='config', y='time', hue='config', data=data, palette='Set1')

# Set labels and title
plt.xlabel('Configuration')
plt.ylabel('time')
plt.title('Box Plot of time by Configuration')
plt.ylim(0.05, 5000)
plt.yscale('log')  # Set the y-axis scale to log


# Add legend
plt.legend(title='Config', loc='upper right')

# Show the
# plt.savefig("only7_cor.svg")
plt.show()
