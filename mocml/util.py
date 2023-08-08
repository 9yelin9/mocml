from . import config

import re
import numpy as np

def GetPm(pm, string):
	return float(re.sub(pm, '', re.search('%s[-]?\d+[.]\d+' % pm, string).group()))

def PmDictH(string):
	t = re.sub('_', '', re.search('_[A-Z]\d_', string).group())

	pm_dict = {
		'type': config.type_dict[t[0]]*10 + int(t[1]),
		'JU':   GetPm('JU',   string),
		'N':    GetPm('N',    string),
		'U':    GetPm('_U',   string),
		'm':    GetPm('_m',   string),
		'gap':  GetPm('_gap', string),
		'Hz':   0,
	}

	return pm_dict

def PmDictD(string):
	pm_dict = {
		'type': int(re.sub('_AF', '', re.search('_AF\d', string).group()))*10,
		'JU':   GetPm('_J',  string),
		'N':    0,
		'U':    GetPm('_UF', string),
		'm':    0,
		'gap':  0,
		'Hz':   GetPm('_Hz', string),
	}

	with open(re.sub('lattice\S+', 'result/u%.3f/filling.dat' % pm_dict['U'], string), 'r') as f:
		pm_dict['N'] = np.genfromtxt(f)[-1, 1] / 4
	with open(re.sub('lattice\S+', 'result/u%.3f/mag.dat'     % pm_dict['U'], string), 'r') as f:
		pm_dict['m'] = np.genfromtxt(f)[-1, 2] * 2

	return pm_dict
