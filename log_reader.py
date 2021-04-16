import pandas as pd
import glob
from pathlib import Path

LOG_DIR = 'logs'
assert Path(LOG_DIR).exists()

table1 = pd.DataFrame(columns=('params', 'mem', 'latency', 'power', 'energy'))

for inf_log in glob.glob(f"{LOG_DIR}/*_inf_log.csv"):
    pass

for supl_log in glob.glob(f"{LOG_DIR}/*_supl_log.csv"):
    pass

