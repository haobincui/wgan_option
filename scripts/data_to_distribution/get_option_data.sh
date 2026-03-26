#!/bin/bash

CURRENT_DATE=$(date +%Y%m%d)
echo "Current Date:${CURRENT_DATE}"

source activate option_data_processor

LOG_FILE="./log/get_option_data_${CURRENT_DATE}.log"

nohup python -u get_option_data.py > "${LOG_FILE}" 2>&1 &
echo "Runing [get_option_data.py]"
echo "Saved to log file: ${LOG_FILE}"

conda deactivate
