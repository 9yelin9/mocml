#!/usr/bin/env python3

from mocml import config, data, model

import os
num_thread = config.num_thread
os.environ['OMP_NUM_THREADS'] = str(num_thread)
os.environ['OPENBLAS_NUM_THREADS'] = str(num_thread)

import argparse
parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
# data
parser.add_argument('-d', '--dos',       type=str, nargs='+', help='<hf/dmft> <Ne> <eta> [opts:f,l,r] [cstrs:m,gap] : GenDOS')
parser.add_argument('-s', '--spec',      type=str, nargs='+', help='* Only HF\n<Ne> <eta> [opts:f,l,r] : GenSpec')
parser.add_argument('-p', '--peak',      type=str, nargs='+', help='* Only DOS with gap\n<path_dos> : GenPeak')
parser.add_argument('-cd', '--checkdos', type=str, nargs='+', help='<path_dos> [verbose=t/f] : CheckDOS')
parser.add_argument('-sd', '--showdos',  type=str, nargs='+', help='<path_dos> [idx=10] : ShowDOS')
# model
parser.add_argument('-pr', '--predict',   type=str, nargs='+', help='<path_train> <path_test> <mc> [ratio=0.3] [verbose=t/f] : Predict')
args = parser.parse_args()                                                                     

d, m = data.Data(), model.Model()
# data
if args.dos: d.GenDOS(*args.dos)
elif args.spec: d.GenSpec(*args.spec)
elif args.peak: d.GenPeak(*args.peak)
elif args.checkdos: d.CheckDOS(*args.checkdos)
elif args.showdos: d.ShowDOS(*args.showdos)
# model
elif args.predict: m.Predict(*args.predict)
else: parser.print_help()
