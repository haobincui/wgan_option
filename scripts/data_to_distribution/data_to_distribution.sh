#!/bin/bash

CURRENT_DATE=$(date +%Y%m%d)
echo "Current Date:${CURRENT_DATE}"

source activate option_data_processor

LOG_FILE="./log/data_to_distribution_${CURRENT_DATE}.log"

nohup python -u data_to_distribution_gpus.py  > "${LOG_FILE}" 2>&1 &
echo "Runing [data_to_distribution_gpus]"
echo "Saved to log file: ${LOG_FILE}"

conda deactivate
