# config.py
import os
BASE_PATH = r"C:\Users\Neermita\Desktop\memory_and_task\ds004395"

# Which subjects to process (all available)
# If some fail, we'll add them later
SUBJECTS = ['LTP065', 'LTP066', 'LTP067', 'LTP068', 'LTP069']

# EEG settings
FILTER_LOW = 4.0   # Hz (theta band start)
FILTER_HIGH = 12.0 # Hz (alpha band end)
TMIN = -0.2        # seconds before event
TMAX = 1.0         # seconds after event
REJECT_CRITERION = 200e-6  # 200 microvolts