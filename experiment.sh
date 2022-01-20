#!/bin/bash

echo "start experiment run!"
echo "this should take about 60 seconds"

echo "data directory: $USER_EXPERIMENT_DATA_DIRECTORY"
echo "artifacts directory: $USER_EXPERIMENT_ARTIFACTS_PATH"

ls $USER_EXPERIMENT_DATA_DIRECTORY

python $USER_EXPERIMENT_DIRECTORY/train.py
cat $USER_EXPERIMENT_ARTIFACTS_PATH/output.txt

sleep 60

echo "end experiment run!"
