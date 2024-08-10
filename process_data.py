import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import torch

combined_results = np.genfromtxt('combined_results.csv', delimiter=',')

Record_Output = combined_results[:, 12:]
Record_Input = combined_results[:, :12]

# Delete nans rows
nan_mask = np.isnan(Record_Output).any(axis=1)
value_mask = Record_Output[:, 1] > 500
combined_mask = ~(nan_mask | value_mask)
filtered_Output = Record_Output[combined_mask]
filtered_Input = Record_Input[combined_mask]

current_dir = os.getcwd()
dataAnalysis_dir = os.path.join(current_dir, 'dataAnalysis')
if not os.path.exists(dataAnalysis_dir):
    os.makedirs(dataAnalysis_dir)

# Process data
# filtered_Input = np.log(filtered_Input)
# filtered_Input[:, 23:25] = filtered_Input[:, 23:25] / filtered_Output[:, 29].reshape(-1, 1)
# filtered_Input[:, 27:29] = filtered_Input[:, 27:29] / filtered_Output[:, 29].reshape(-1, 1)
# filtered_Output[:, 30:32] = filtered_Output[:, 30:32] / filtered_Output[:, 29].reshape(-1, 1)

data_input_tensor = torch.tensor(filtered_Input, dtype=torch.float32)
data_output_tensor = torch.tensor(filtered_Output, dtype=torch.float32)

torch.save(data_input_tensor, 'filtered_input.pt')
torch.save(data_output_tensor, 'filtered_output.pt')

# Labels
Input_labels = [
    'E_LV_act', 'E_LV_pas', 'E_RV_act', 'E_RV_pas',
    'E_LA_act', 'E_LA_pas', 'E_RA_act', 'E_RA_pas',
    'V0_LV', 'V0_RV', 'V0_LA', 'V0_RA',
]

Output_labels = [
    'P_sys', 'P_dias', 'P_sys_pul', 'P_dias_pul', 'MAP', 'mPAP', 'CVP', 'PAWP',
    'LVEDV', 'LVESV', 'RVEDV', 'RVESV', 'LAEDV', 'LAESV', 'RAEDV', 'RAESV',
    'LVSV', 'RVSV', 'LASV', 'RASV', 'LVEF', 'RVEF', 'CO', 'SVR', 'PVR',
    'LVSW', 'RVSW'
]

# sns.histplot(np.log10(filtered_Output[:, -1]), color='r', kde=True, bins=50)
# plt.title(f'Distribution Plot of {Output_labels[-1]}')
# plt.xlabel(Output_labels[-1])
# plt.ylabel('Frequency')
# plt.show()

# Step 3: Plot the distribution
for i in range(len(Input_labels)):
    sns.histplot(filtered_Input[:, i], color='b', kde=True, bins=50)
    if i < 0:
        plt.title(f'Distribution Plot of log({Input_labels[i]})')
        plt.xlabel(f'log({Input_labels[i]})')
        plt.ylabel('Frequency')
        plt.savefig(f'{dataAnalysis_dir}/Input_log_{Input_labels[i]}.png')
        plt.close()
        print(f'{dataAnalysis_dir}/Input_log_{Input_labels[i]}.png is generated\n')
    else:
        plt.title(f'Distribution Plot of {Input_labels[i]}')
        plt.xlabel(Input_labels[i])
        plt.ylabel('Frequency')
        plt.savefig(f'{dataAnalysis_dir}/Input_{Input_labels[i]}.png')
        plt.close()
        print(f'{dataAnalysis_dir}/Input_{Input_labels[i]}.png is generated\n')

for i in range(len(Output_labels)):
    sns.histplot(filtered_Output[:, i], color='r', kde=True, bins=50)
    plt.title(f'Distribution Plot of {Output_labels[i]}')
    plt.xlabel(Output_labels[i])
    plt.ylabel('Frequency')
    plt.savefig(f'{dataAnalysis_dir}/Output_{Output_labels[i]}.png')
    plt.close()
    print(f'{dataAnalysis_dir}/Output_{Output_labels[i]}.png is generated\n')
