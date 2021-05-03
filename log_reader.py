import pandas as pd
import glob
from pathlib import Path
from onnx_opcounter import calculate_macs, calculate_params
import onnx

LOG_DIR = 'logs_run2'
ONNX_MODELS_DIR = 'onnx_models'
assert Path(LOG_DIR).exists()
# assert Path(ONNX_MODELS_DIR).exists()

table1 = pd.DataFrame(columns=('params', 'frac', 'epochs', 'mem', 'quantType', 'latency', 'power', 'energy', 'num_images','acc'))

for inf_log in glob.glob(f"{LOG_DIR}/*_inf_log.*"):
    # model_name,test_acc,runtime,avg_latency,num_images
    inf_log_df = pd.read_csv(inf_log)
    model_name = inf_log_df['model_name'].iloc[0].replace(f'{ONNX_MODELS_DIR}/', '').replace('.onnx', '')
    epochs = 100
    frac = model_name.split('_')[2]
    quant = model_name.split('_')[3]
    new_row = {'params': model_name,
               'epochs': epochs,
               'frac': frac,
               'mem': None,
		  'quantType': quant,
               'latency': inf_log_df['avg_latency'].iloc[0],
               'power': None,
               'energy': None,
               'num_images': inf_log_df['num_images'].iloc[0],
               'acc': inf_log_df['test_acc'].iloc[0]}
    table1 = table1.append(new_row, ignore_index=True)


for supl_log in glob.glob(f"{LOG_DIR}/*_supl_log.*"):
    # "run_name time W energy_usage max_mem"
    supl_log_df = pd.read_csv(supl_log, index_col=False, error_bad_lines=False)
    table1.loc[table1['params'] == supl_log_df['run_name'].iloc[0], ['mem', 'power', 'energy']] = (supl_log_df['max_mem'].iloc[-1]/1024/1024, supl_log_df['W'].max(), supl_log_df['energy_usage'].max()/table1.loc[table1['params'] == supl_log_df['run_name'].iloc[0], 'num_images'])


table1 = table1.sort_values(by=['frac'])
table1 = table1.rename(columns={'params': 'Parameters',
                                'epochs': 'Retraining epochs',
                                'frac': 'Pruning Fraction',
                                'mem': 'Memory [MB]',
                                'quantType': 'Quantization Type',
                                'latency': 'Latency [s]',
                                'power': 'Max Power [W]',
                                'energy': 'Energy/image [J]',
                                'acc': 'Accuracy'})
rounded_table1 = table1.round(decimals=3)
table1.to_csv('table3_best.csv')
rounded_table1.to_csv('rounded_table3_best.csv')