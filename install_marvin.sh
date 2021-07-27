#!/bin/sh

sudo apt install python3.7 python3-venv python3.7-venv
virtualenv --python=/usr/bin/python3.7 --no-site-packages venv
. venv/bin/activate
pip3 install -r requirements.txt
rm -rf venv/lib/python3.7/site-packages/gym/envs/
cp -r ./envs venv/lib/python3.7/site-packages/gym/envs/
cp ./utils/rendering.py ./venv/lib/python3.7/site-packages/gym/envs/classic_control/
