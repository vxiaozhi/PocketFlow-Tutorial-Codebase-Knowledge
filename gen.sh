#!/bin/bash

export $(cat .env | xargs)
python3.10 main.py --dir $1 --include "*.py" --exclude "*test*" --language "Chinese"
