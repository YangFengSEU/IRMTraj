#!/bin/bash


# python train.py --train_data1 "EP" --train_data2 "FT" --test_data "SR"
# python train.py --train_data1 "EP" --train_data2 "FT" --test_data "LN"
# python train.py --train_data1 "EP" --train_data2 "FT" --test_data "GL"
# python train.py --train_data1 "EP" --train_data2 "FT" --test_data "MA"
# python train.py --train_data1 "EP" --train_data2 "FT" --test_data "EP0"

# python train.py --train_data1 "EP0" --train_data2 "MA" --test_data "GL"
# python train.py --train_data1 "EP0" --train_data2 "GL" --test_data "MA"
# python train.py --train_data1 "MA" --train_data2 "GL" --test_data "OF"
python train.py --train_data1 "EP0" --train_data2 "MA" --test_data "SR"
python train.py --train_data1 "GL" --train_data2 "EP0" --test_data "LN"

