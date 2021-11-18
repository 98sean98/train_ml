#!/bin/bash

echo "start training, hahaha!"
echo "this should take about 60 seconds"

sleep 30

mkdir $USER_TRAINING_PATH/artifacts
echo "I'm an artifact" >> $USER_TRAINING_PATH/artifacts/artifact-1.txt

sleep 30

echo "end training!"
