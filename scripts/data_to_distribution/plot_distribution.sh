#!/bin/bash

CURRENT_DATE=$(date +%Y%m%d)
echo "Current Date:${CURRENT_DATE}"

source activate option_data_processor

LOG_FILE="./log/plot_distribution_${CURRENT_DATE}.log"

nohup python -u plot_distribution.py  > "${LOG_FILE}" 2>&1 &
echo "Runing [plot_distribution]"
echo "Saved to log file: ${LOG_FILE}"

conda deactivate
