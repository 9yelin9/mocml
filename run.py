#!/usr/bin/env python3

import os
import argparse

from mocml import data, model

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
# data
parser.add_argument('-d', '--dos',       type=str, nargs='+', help='<hf/dmft> <Ne> <eta> [cstrs:m,gap] [opts:f,l,r] : GenDOS')
parser.add_argument('-cd', '--checkdos', type=str, nargs='+', help='<path_dos> [verbose=t/f] : CheckDOS')
parser.add_argument('-sd', '--showdos',  type=str, nargs='+', help='<path_dos> [idx=10] : ShowDOS')
# model
parser.add_argument('-p', '--predict',   type=str, nargs='+', help='<path_train> <path_test> <mc> [ratio=0.3] [verbose=t/f] : Predict')
args = parser.parse_args()                                                                     

d, m = data.Data(), model.Model()
# data
if args.dos: d.GenDOS(*args.dos)
elif args.checkdos: d.CheckDOS(*args.checkdos)
elif args.showdos: d.ShowDOS(*args.showdos)
# model
elif args.predict: m.Predict(*args.predict)
else: parser.print_help()
