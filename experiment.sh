#!/bin/bash

echo "start experiment run!"

echo "data directory: $USER_EXPERIMENT_DATA_DIRECTORY"
echo "artifacts directory: $USER_EXPERIMENT_ARTIFACTS_PATH"

ls $USER_EXPERIMENT_DATA_DIRECTORY

python $USER_EXPERIMENT_DIRECTORY/train.py
cat $USER_EXPERIMENT_ARTIFACTS_PATH/output.txt

echo "end experiment run!"
