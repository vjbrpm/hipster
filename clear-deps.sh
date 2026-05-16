#!/bin/bash

source ./.venv/bin/activate
pip freeze | xargs pip uninstall -y
