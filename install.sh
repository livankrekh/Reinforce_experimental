#!/bin/sh

virtualenv venv
source venv/bin/activate
pip3 install -r requirements.txt
rm -rf venv/lib/python3.7/site-packages/gym/envs/
cp -r ./envs venv/lib/python3.7/site-packages/gym/envs/
cp ./utils/rendering.py ./venv/lib/python3.7/site-packages/gym/envs/classic_control/
