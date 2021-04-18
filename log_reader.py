import pandas as pd
import glob
from pathlib import Path
from onnx_opcounter import calculate_macs, calculate_params
import onnx

LOG_DIR = 'logs'
ONNX_MODELS_DIR = 'onnx_models'
assert Path(LOG_DIR).exists()
assert Path(ONNX_MODELS_DIR).exists()

table1 = pd.DataFrame(columns=('params', 'mem', 'latency', 'power', 'energy', 'num_images', 'num_params'))

for inf_log in glob.glob(f"{LOG_DIR}/*_inf_log.*"):
    # model_name,test_acc,runtime,avg_latency,num_images
    inf_log_df = pd.read_csv(inf_log)
    new_row = {'params': inf_log_df['model_name'].iloc[0].replace(f'{ONNX_MODELS_DIR}/', '').replace('.onnx', ''), 'mem': None, 'latency': inf_log_df['avg_latency'].iloc[0], 'power': None, 'energy': None, 'num_images': inf_log_df['num_images'].iloc[0], 'num_params': None}
    table1 = table1.append(new_row, ignore_index=True)


for supl_log in glob.glob(f"{LOG_DIR}/*_supl_log.*"):
    # "run_name time W energy_usage max_mem"
    supl_log_df = pd.read_csv(supl_log, index_col=False)
    table1.loc[table1['params'] == supl_log_df['run_name'].iloc[0], ['mem', 'power', 'energy']] = (supl_log_df['max_mem'].iloc[-1], supl_log_df['W'].max(), supl_log_df['energy_usage'].max()/table1.loc[table1['params'] == supl_log_df['run_name'].iloc[0], 'num_images'])

for model_name in table1['params']:
    model_path = f'{ONNX_MODELS_DIR}/{model_name}.onnx'
    print(f"Model: {model_path}")
    model = onnx.load_model(model_path)
    num_params = calculate_params(model)
    table1.loc[table1['params'] == model_name, ['num_params']] = (num_params,)

print(table1)
table1.to_csv('table1.csv')
table1.to_html('table1.html')