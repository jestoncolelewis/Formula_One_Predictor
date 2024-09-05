#!/usr/bin/bash
python3 -m pip install virtualenv
virtualenv .venv --python=python3.11
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt