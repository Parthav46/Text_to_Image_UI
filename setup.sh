#!/bin/bash

# Basic environment installs
sudo apt update
sudo apt upgrade -y
sudo apt install -y libsm6 libxext6 libxrender-dev

# Pip installs
pip install -r requirements.txt
