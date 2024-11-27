import sys
import pandas as pd
import numpy as np
from numpy.random import *
import time
import matplotlib.pyplot as plt
import seaborn as sns

##################
# read IR tables #
##################

def read_tables_OpenROAD(data_root, design=None):
    cell_cell_path = data_root + "cell_cell_edge.csv"
    cell_pin_path = data_root  + "cell_pin_edge.csv"
    cell_path = data_root  + "cell_properties.csv"
    net_pin_path = data_root  +  "net_pin_edge.csv"
    net_path = data_root  + "net_properties.csv"
    pin_pin_path = data_root  + "pin_pin_edge.csv"
    pin_path = data_root  + "pin_properties.csv"
    net_cell_path = data_root  + "cell_net_edge.csv"
    all_fo4_delay_path = data_root + "libcell_properties.csv"
    ### load tables
    fo4_df = pd.read_csv(all_fo4_delay_path)
    pin_df = pd.read_csv(pin_path)
    cell_df = pd.read_csv(cell_path)
    net_df = pd.read_csv(net_path)
    cell_cell_df = pd.read_csv(cell_cell_path)
    pin_pin_df = pd.read_csv(pin_pin_path)
    cell_pin_df = pd.read_csv(cell_pin_path)
    net_pin_df = pd.read_csv(net_pin_path)
    net_cell_df = pd.read_csv(net_cell_path)
    return pin_df, cell_df, net_df, pin_pin_df, cell_pin_df, net_pin_df, net_cell_df, cell_cell_df, fo4_df

pin_df, cell_df, net_df, pin_pin_df, cell_pin_df, net_pin_df, net_cell_df, cell_cell_df, fo4_df = read_tables_OpenROAD("./IRs/nangate45/gcd/")

#df = pd.DataFrame({
#    'x': [10, 20, 30, 45, 55, 90, 100, 150],
#    'y': [15, 35, 60, 75, 85, 20, 55, 100],
#    'static_power': [0.2, 0.5, 0.4, 0.7, 0.3, 0.9, 0.6, 0.8]
#})

db_to_um = 2000

filtered_cell_df = cell_df[['x0','y0','x1','y1','cell_static_power']]
filtered_cell_df['mid_x'] = (filtered_cell_df['x0'] + filtered_cell_df['x1'])/(2*db_to_um)
filtered_cell_df['mid_y'] = (filtered_cell_df['y0'] + filtered_cell_df['y1'])/(2*db_to_um)

print(filtered_cell_df)
#Set the grid size, e.g., 10x10
size_x = 35
size_y = 35

grid_size = 0.5

# Step 1: Determine grid coordinates for each cell
filtered_cell_df['x_grid'] = (filtered_cell_df['mid_x'] // grid_size).astype(int)
filtered_cell_df['y_grid'] = (filtered_cell_df['mid_y'] // grid_size).astype(int)
# Step 2: Group by grid and sum the static power for each grid
static_power_map = filtered_cell_df.groupby(['x_grid', 'y_grid'])['cell_static_power'].sum().unstack(fill_value=0)

# Step 3: Plot the heatmap
max_x_grid = static_power_map.index.max()  
max_y_grid = static_power_map.columns.max()  

plt.figure(figsize=(max_x_grid, max_y_grid))
heatmap = sns.heatmap(static_power_map, cmap="jet", linewidths=0.5, annot=False, cbar_kws={'label': 'Total Power'})
plt.title('Power Heatmap',fontsize=80)
plt.xlabel('X Grid',fontsize=80)
plt.ylabel('Y Grid',fontsize=80)
colorbar = heatmap.collections[0].colorbar
colorbar.ax.tick_params(labelsize=80)
plt.savefig("heat_map.png")
plt.show()


#static_power_map = torch.Tensor(static_power_map).unsqueeze(0).unsqueeze(0)

#static_IR_map = torch.Tensor(static_IR_map)

#output_tensor = model([static_power_map, dynamic_power_map, m1_congestion_map, m2_congestion_map, m3_congestion_map]).squeeze()

#output_array = output_tensor.squeeze().detach().numpy()

#l1loss = nn.L1Loss()
#loss = l1loss(static_IR_map, output_tensor)
