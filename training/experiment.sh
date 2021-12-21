#!/bin/bash

echo "start experiment run!"
echo "this should take about 60 seconds"

ls /data

python $USER_EXPERIMENT_DIRECTORY/training/train.py

sleep 60

echo "end experiment run!"
