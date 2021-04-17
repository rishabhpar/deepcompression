import pandas as pd
import glob
from pathlib import Path

LOG_DIR = 'logs'
assert Path(LOG_DIR).exists()

table1 = pd.DataFrame(columns=('params', 'mem', 'latency', 'power', 'energy', 'num_images'))

for inf_log in glob.glob(f"{LOG_DIR}/*_inf_log.csv"):
    # model_name,test_acc,runtime,avg_latency,num_images
    inf_log_df = pd.read_csv(inf_log)
    new_row = {'params': inf_log_df['model_name'], 'mem': None, 'latency': inf_log_df['avg_latency'], 'power': None, 'energy': None, 'num_images': inf_log_df['num_images']}
    table1 = table1.append(new_row, ignore_index=True)


for supl_log in glob.glob(f"{LOG_DIR}/*_supl_log.csv"):
    # "run_name time W energy_usage max_mem"
    supl_log_df = pd.read_csv(supl_log)
    table1.loc[table1['params'] == supl_log_df['run_name'], ['mem', 'power', 'energy']] = (supl_log_df['max_mem'].iloc[-1], supl_log_df['W'].max(), supl_log_df['energy_usage'].max()/table1.loc[table1['params'] == supl_log_df['run_name'], 'num_images'])

