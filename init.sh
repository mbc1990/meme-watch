#!/bin/bash 
source venv/bin/activate
CAFFE_ROOT="/home/malcolm/Projects/caffe"
export PYTHONPATH="$CAFFE_ROOT/distribute/python:$PYTHONPATH"
export LD_LIBRARY_PATH="$CAFFE_ROOT/distribute/lib/"
python classify.py
