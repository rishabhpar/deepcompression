import argparse

import psutil
import telnetlib as tel
import time
from pathlib import Path
import threading
# import gpiozero
from timeit import default_timer as timer
import subprocess
import numpy as np


class Logger:

    def __init__(self, LOG_DIR='logs'):
        super(Logger, self).__init__()
        self.SP2_tel = None
        self.LOG_DIR = LOG_DIR
        Path(self.LOG_DIR).mkdir(exist_ok=True, parents=True)
        self.logger_thread = None
        self.is_running = False


    def wait_for_startup(self):
        SP2_tel = tel.Telnet("192.168.4.1")

    def start_logger_thread(self, run_name):
        print(f"LOGGER: Starting logger thread for {run_name}")
        self.is_running = True
        self.logger_thread = threading.Thread(target=lambda: self._log_run(run_name))
        self.logger_thread.start()

    def stop_logger_thread(self):
        print(f"LOGGER: Stopping logger thread")
        self.is_running = False
        self.logger_thread.join()
        print("LOGGER: Logger thread stopped")

    def _log_run(self, run_name):
        out_fname = f'{run_name}_info_supl_log.txt'
        header = "time W energy_usage max_mem"
        header = ",".join(header.split(' '))
        out_file = open(out_fname, 'a')
        out_file.write(header)
        out_file.write("\n")

        print(f"LOGGER: Started with log name: {out_fname}")

        total_power = 0.0
        cum_energy = 0.0
        prev_start_time = time.time()
        run_start_time = time.time()
        max_mem = 0

        assert self.is_running

        while self.is_running:
            cur_start_time = time.time()

            max_mem = max(max_mem, psutil.virtual_memory()[3])
            total_power = self.getBoardPower(total_power)
            cum_energy += total_power * (cur_start_time - prev_start_time)  # J = W * s

            # temps = getTemps()
            # avg_temp = np.mean(temps)

            time_stamp = cur_start_time

            fmt_str = "{}," * 5
            out_ln = fmt_str.format(time_stamp - run_start_time, total_power, cum_energy, max_mem)

            out_file.write(out_ln)
            out_file.write("\n")
            elapsed = time.time() - cur_start_time
            DELAY = 0.01
            prev_start_time = cur_start_time
            time.sleep(max(0, DELAY - elapsed))

        out_file.close()

    def getBoardPower(self, prev):
        # use telnet to read power usage
        tel_dat = str(self.SP2_tel.read_very_eager())

        findex = tel_dat.rfind('\n')
        findex2 = tel_dat[:findex].rfind('\n')
        findex2 = findex2 if findex2 != -1 else 0
        ln = tel_dat[findex2:findex].strip().split(',')
        if len(ln) < 2:
            total_power = prev
        else:
            total_power = float(ln[-2])
        return total_power


