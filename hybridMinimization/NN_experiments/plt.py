import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Initialize predefined lists
method_list = ['hybridM','hybridD','CoCaBO','EXP3BO','roundrobinMAB','randomMAB','skoptDummy','skoptForest','skoptGP','SMAC','TPE'] 
color_map = dict(zip(method_list, sns.color_palette("husl", len(method_list))))

# Initialize actual lists
method_list = ['hybridM','hybridMD','TPE','skoptDummy','skoptForest','skoptGP','SMAC'] 
task_list = ['redshift','boston','concreteslump','ca','airfoil']

# To store results for plotting
plot_data = {}

for task_name in task_list:
    plot_data[task_name] = {
        'methods': [],
        'mean_min_values': [],
        'min_errors': [],
        'max_errors': [],
        'mean_steps': [],
        'min_values': []
    }

    for method_name in method_list:
        all_matrices = []

        for root, dirs, files in os.walk('NN_'+task_name):
            for k in range(10):
                x_file = 'NN_' + task_name + '_' + method_name + '_' + str(k) + '_X.csv'
                y_file = 'NN_' + task_name + '_' + method_name + '_' + str(k) + '_Y.csv'
                try:
                    if x_file in files and y_file in files:
                        X = np.loadtxt(os.path.join(root, x_file), delimiter=',')
                        Y = np.loadtxt(os.path.join(root, y_file), delimiter=',')
                        Y = Y.reshape(-1, 1)
                        combined_matrix = np.hstack((X, Y))
                        new_col = int(k) * np.ones((combined_matrix.shape[0], 1))
                        final_matrix = np.hstack((new_col, combined_matrix))
                        all_matrices.append(final_matrix)
                except:
                    continue

        big_matrix = np.vstack(all_matrices)
        outfile = 'NN_' + task_name + '_' + method_name + '_all.csv'
        np.savetxt(outfile, big_matrix, delimiter=',')

        data = np.loadtxt(outfile, delimiter=',')
        unique_ks = np.unique(data[:, 0])
        k_values, range_values, min_so_far_values, max_so_far_values = [], [], [], []

        for k in unique_ks:
            batch_data = data[data[:, 0] == k]
            min_so_far = np.minimum.accumulate(batch_data[:, -1])
            max_so_far = np.maximum.accumulate(batch_data[:, -1])
            k_values.extend([k] * len(min_so_far))
            range_values.extend(range(1, len(min_so_far) + 1))
            min_so_far_values.extend(min_so_far)
            max_so_far_values.extend(max_so_far)

        final_matrix = np.column_stack((k_values, range_values, min_so_far_values, max_so_far_values))
        outfile = 'NN_' + task_name + '_' + method_name + '_history.csv'
        np.savetxt(outfile, final_matrix, delimiter=',')

        data = np.loadtxt(outfile, delimiter=',')
        unique_batches = np.unique(data[:, 0])
        max_steps, min_steps, max_values, min_values = [], [], [], []

        for batch in unique_batches:
            batch_data = data[data[:, 0] == batch]
            if len(batch_data) >= 100:
                subset = batch_data[:100]
                max_step = np.argmax(subset[:, -2]) + 1
                min_step = np.argmax(subset[:, -1]) + 1
                max_steps.append(max_step)
                min_steps.append(min_step)

            max_values.append(np.max(np.abs(batch_data[:, -2])))
            min_values.append(np.min(np.abs(batch_data[:, -1])))

        # Remove nan values
        min_values = [x for x in min_values if not np.isnan(x)]
        max_values = [x for x in max_values if not np.isnan(x)]
        print(task_name,method_name,min_values)
        if method_name == 'hybridMD':
            plot_data[task_name]['methods'].append('hybridD')
        else:
            plot_data[task_name]['methods'].append(method_name)
        plot_data[task_name]['mean_min_values'].append(np.mean(min_values))
        plot_data[task_name]['min_errors'].append(np.min(min_values))
        plot_data[task_name]['max_errors'].append(np.max(min_values))
        plot_data[task_name]['mean_steps'].append(np.mean(min_steps))
        plot_data[task_name]['min_values'].append(min_values)

from matplotlib.ticker import FuncFormatter 
def sci_notation(x, _):
    """Function to format tick values using scientific notation"""
    return f"{x:.1e}"

# ... [rest of your code]

# Plotting
fig, axes = plt.subplots(nrows=1, ncols=len(task_list), figsize=(12, 4))

for ax, task_name in zip(axes, task_list):
    data = plot_data[task_name]
    x = np.arange(1, len(data['methods']) + 1)  # Using 1-based indexing for x-values of box plots
    colors = [color_map[method] for method in data['methods']]

    box_data = []
    for min_vals in data['min_values']:
        box_data.append(min_vals)

    # Create the boxplots
    boxes = ax.boxplot(box_data, patch_artist=True, widths=0.9, positions=x)

    # Change the colors of boxes
    for patch, color in zip(boxes['boxes'], colors):
        patch.set_facecolor(color)
    
    # Change the colors of medians to black
    for median in boxes['medians']:
        median.set_color('black')

    # Adding text to the boxes
    for i, box in enumerate(boxes['boxes']):
        y_position = np.median(box_data[i])
        y_position = 0
        x_position = x[i]  # Using the x-values for the position of the box plot
        if task_name=='redshift':
            y_position = 3.5
        if task_name=='boston':
            y_position = 2
        if task_name=='concreteslump':
            y_position = 2.5e-3
        if task_name=='ca':
            y_position = 5
        if task_name=='airfoil':
            y_position = 2.5e-2
            
        ax.text(x_position, y_position, f'{data["mean_steps"][i]:.0f}', ha='center', va='center', color='black', fontsize=12, rotation = 0)
    
    ax.set_xticks(x)
    ax.set_xticklabels(data['methods'], rotation=90)
    ax.set_yticklabels(x, fontsize=8)
    ax.set_title(task_name+'\n')
    if task_name == 'redshift':
        ax.set_ylabel('Minimum MSE')
    else:
        ax.set_ylabel(None)
    if task_name == 'ca':
        ax.set_ylim(-2*1e-1, 5.)
    if task_name == 'boston':
        ax.set_ylim(-2*1e-1, 2.)
    if task_name == 'concreteslump':
        ax.set_ylim(-0.1*1e-3, 2.5e-3)
    if task_name == 'airfoil':
        ax.set_ylim(-0.1*1e-2, 2.5e-2)
    if task_name == 'redshift':
        ax.set_ylim(-0.1, 3.5)
    ax.yaxis.set_major_formatter(FuncFormatter(sci_notation))

plt.tight_layout()
plt.savefig("NN_compare.pdf", format="pdf", bbox_inches="tight")
plt.show()

