from . import config, util

import os
import re
import sys
import ctypes
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from scipy import interpolate
from timeit import default_timer as timer
from sklearn.model_selection import train_test_split

class Data:
	def GenEnergy(self, Ne):
		Ne = int(Ne)

		if not Ne < config.Ne_max:
			print('Ne should be less than %d' % config.Ne_max)
			sys.exit()

		path_save   = '/'.join(['data', 'energy_Ne%d.dat' % Ne])
		path_energy = '/'.join(['data', 'energy_Ne%d.dat' % config.Ne_max])

		with open(path_energy, 'r') as f: energy = np.genfromtxt(f)
		np.savetxt(path_save, energy[::config.Ne_max//Ne], fmt='%.10f')

		print('GenEnergy(%s)' % (path_save))

	def ReadConfig(self, dir_data):
		path = ''
		Nb = 0
		Nk = 0
		is_k = 0
		k_label = []
		k_point = []

		path_config = '/'.join(['data', dir_data, 'config.txt'])

		if not os.path.isfile(path_config):
			print('"%s" not found' % path_config)
			sys.exit()

		with open(path_config, 'r') as f:
			for line in f:
				if re.match('path', line): path = line.split()[1]
				if re.match('Nb', line): Nb = int(line.split()[1])

				if re.match('end kpoints', line): break
				if is_k:
					lines = line.split()
					k_label.append(lines[0])
					k_point.append(int(lines[1]))
				if re.match('begin kpoints', line): is_k = 1

		return path, Nb, k_label, k_point

	def GetGrounds(self, f_list):
		grds_pm = ['type', 'JU', 'N', 'U', 'e']

		data = np.zeros(len(grds_pm))
		for f in f_list: data = np.vstack((data, [util.PmDictH(f)[pm] for pm in grds_pm[:-1]] + [util.GetPm('e', f)]))
		data = np.delete(data, 0, 0)

		df = pd.DataFrame(data, columns=grds_pm)
		df = df.sort_values(by=['JU', 'N', 'U', 'e', 'type'])
		df['type'] = df['type'] // 10
		df = df.drop_duplicates(subset=['JU', 'N', 'U', 'type'], keep='first')
		grds = df.index.to_list()

		return [f_list[i] for i in grds]

	def GetEta(self, eta, energy, options):
		energy_max = np.max(np.abs(energy))

		opts_dict = {
			'n': np.ones(len(energy)),
			'f': np.array([0 if e > 0 else 1 for e in energy]),
			'l': np.array([np.abs(e/energy_max) for e in energy]),
			'r': np.random.rand(len(energy)),
		}
		opts_list = [opts_dict[opt] for opt in options]
		eta = eta * np.multiply.reduce(opts_list)

		return eta

	def DOSH(self, dir_data, Ne, energy, eta, is_spec=False):
		path, Nb, k_label, k_point = self.ReadConfig(dir_data)

		if is_spec:
			k_label = ['K%d' % i for i in range(k_point[-1]+1)]
			k_point = range(k_point[-1]+1)
			dos_list = ['%s_%d' % (l, i) for l in k_label for i in range(Ne)]
		else:
			dos_list = ['%s%d' % (l, i) for l in k_label for i in range(Ne)]

		energy_c = np.ctypeslib.as_ctypes(energy)
		eta_c    = np.ctypeslib.as_ctypes(eta)

		b2d = ctypes.cdll.LoadLibrary('mocml/b2d.so')

		i, pm, band = 0, np.zeros(len(config.pm_list)), np.zeros(len(k_point) * 2*Nb)
		for f in self.GetGrounds([f for f in glob(path) if not re.search('nost_F', f)]):
			pm_dict = util.PmDictH(f)

			if pm_dict['gap'] > config.gap_tol:
				fermi_list = np.arange(-pm_dict['gap']/2, pm_dict['gap']/2, config.fermi_itv)		
			else: fermi_list = [0]

			for fermi in fermi_list:
				pm   = np.vstack((pm, [i] + list(pm_dict.values())))
				band = np.vstack((band+fermi, np.ravel(np.genfromtxt(f, skip_header=1)[k_point, :])))
				i += 1
		pm, band = np.delete(pm, 0, 0), np.delete(band, 0, 0)

		band_c = np.ctypeslib.as_ctypes(np.ravel(band))
		dos_c  = np.ctypeslib.as_ctypes(np.zeros(len(band)*len(k_point)*Ne))

		b2d.Band2DOS(len(k_point), Nb, Ne, len(band), band_c, energy_c, eta_c, dos_c)
		dos = np.reshape(np.ctypeslib.as_array(dos_c), (len(band), len(k_point)*Ne))
		
		return pm, dos, dos_list

	def DOSD_old(self, dir_data, Ne, energy, eta):
		path, Nb, k_label, k_point = self.ReadConfig(dir_data)
		dos_list = ['%s%d' % (l, i) for l in k_label for i in range(Ne)]
		fermi_idx = np.min(np.where(energy > 0))

		i, pm, dos = 0, np.zeros(len(config.pm_list)), np.zeros(len(k_point) * config.Ne_max)
		for f in [f for f in glob(path) if re.search('ep%.2f' % eta[-1], f)]:
			dosi = [np.genfromtxt(re.sub('_kG_', '_k%s_' % l, f))[:, 1] * 6 for l in k_label]
			ddosi = np.diff(dosi) > 0 # boolean

			gap = []
			for ddosij in ddosi:
				peak_all = [j+1 for j in range(len(ddosij)-1) if ddosij[j] and not ddosij[j+1]]

				peak_pos = [i for i in peak_all if i > fermi_idx]
				peak_pos = np.min(peak_pos) if len(peak_pos) else 999

				peak_neg = [i for i in peak_all if i < fermi_idx]
				peak_neg = np.max(peak_neg) if len(peak_neg) else -999

				gap += [energy[peak_pos] - energy[peak_neg]]

			pm_dict = util.PmDictD(f)
			pm_dict['gap'] = np.min(gap)

			pm  = np.vstack((pm, [i] + list(pm_dict.values())))
			dos = np.vstack((dos, np.ravel(dosi)))
			i += 1
		pm, dos = np.delete(pm, 0, 0), np.delete(dos, 0, 0)

		if Ne < config.Ne_max:
			de = energy[config.Ne_max//Ne] - energy[0]
			dos = np.add.reduceat(dos*de, range(0, dos.shape[1], config.Ne_max//Ne), 1)

		return pm, dos, dos_list

	def DOSD(self, dir_data, Ne, energy, eta):
		path, Nb, k_label, k_point = self.ReadConfig(dir_data)
		dos_list = ['%s%d' % (l, i) for l in k_label for i in range(Ne)]
		fermi_idx = np.min(np.where(energy > 0))

		i, pm, dos = 0, np.zeros(len(config.pm_list)), np.zeros(len(k_point) * Ne)
		for f in [f for f in glob(path) if re.search('ep%.2f' % eta[-1], f)]:
			itp_list = []
			for fl in [re.sub('_kG_', '_k%s_' % l, f) for l in k_label]:
				fl_eng = np.genfromtxt(fl)[:, 0]
				fl_dos = np.genfromtxt(fl)[:, 1] * 6
				itp_list.append(interpolate.interp1d(fl_eng, fl_dos, fill_value='extrapolate'))

			dosi = np.array([itp(energy) for itp in itp_list])
			ddosi = np.diff(dosi) > 0 # boolean

			gap = []
			for ddosij in ddosi:
				peak_all = [j+1 for j in range(len(ddosij)-1) if ddosij[j] and not ddosij[j+1]]

				peak_pos = [i for i in peak_all if i > fermi_idx]
				peak_pos = np.min(peak_pos) if len(peak_pos) else 999

				peak_neg = [i for i in peak_all if i < fermi_idx]
				peak_neg = np.max(peak_neg) if len(peak_neg) else -999

				gap += [energy[peak_pos] - energy[peak_neg]]

			pm_dict = util.PmDictD(f)
			pm_dict['gap'] = np.min(gap)

			pm  = np.vstack((pm, [i] + list(pm_dict.values())))
			dos = np.vstack((dos, np.ravel(dosi)))
			i += 1
		pm, dos = np.delete(pm, 0, 0), np.delete(dos, 0, 0)

		return pm, dos, dos_list

	def GenDOS(self, dir_data, Ne, eta, options='n', cstrs='n'):
		Ne, eta = int(Ne), float(eta)
		Ne_energy = config.Ne_max if re.search('dmft.old', dir_data) else Ne

		if re.search('hf', dir_data): DOS = self.DOSH
		elif re.search('dmft_old', dir_data): DOS = self.DOSD_old
		else: DOS = self.DOSD

		cstrs_dict = {
			'n': -1,
			'm': 0.1,
			'gap': config.gap_tol,
		}
		cstrs_str = '_%s%.2f' % (cstrs, cstrs_dict[cstrs]) if cstrs_dict[cstrs] > 0 else ''

		path_save   = '/'.join(['data', dir_data, 'dos_%s_Ne%d_eta%.2f%s.csv' % (''.join(options), Ne, eta, cstrs_str)])
		path_energy = '/'.join(['data', 'energy_Ne%d.dat' % Ne_energy])
		
		if not os.path.isfile(path_energy): self.GenEnergy(Ne_energy)
		with open(path_energy, 'r') as f: energy = np.genfromtxt(f)
		eta = self.GetEta(eta, energy, options)

		t0 = timer()

		pm, dos, dos_list = DOS(dir_data, Ne, energy, eta)
		print('DOS shape :', dos.shape)

		print('%d' % len(pm), end=' -> ')
		idx = np.array(range(len(pm)))
		if cstrs_dict[cstrs] > 0:
			idx = np.where(pm[:, config.pm_list.index(cstrs)] > cstrs_dict[cstrs])[0]
			idx = np.intersect1d(idx, idx)
		pm, dos = pm[idx, :], dos[idx, :]
		print('(%s)%d' % (cstrs, len(pm)))

		np.savetxt(path_save, np.hstack((pm, dos)), fmt='%.10f', delimiter=',', header=','.join(config.pm_list + dos_list)) 

		t1 = timer()
		print('GenDOS(%s) : %fs' % (path_save, t1-t0))

	def GenSpec(self, Ne, eta, options='n'):
		dir_data, Ne, eta = 'hf', int(Ne), float(eta)

		path_save   = '/'.join(['data', dir_data, 'spec_%s_Ne%d_eta%.2f.csv' % (''.join(options), Ne, eta)])
		path_energy = '/'.join(['data', 'energy_Ne%d.dat' % Ne])
		
		if not os.path.isfile(path_energy): self.GenEnergy(Ne)
		with open(path_energy, 'r') as f: energy = np.genfromtxt(f)
		eta = self.GetEta(eta, energy, options)

		t0 = timer()

		pm, dos, dos_list = self.DOSH(dir_data, Ne, energy, eta, is_spec=True)
		print('DOS shape :', dos.shape)

		np.savetxt(path_save, np.hstack((pm, dos)), fmt='%.10f', delimiter=',', header=','.join(config.pm_list + dos_list)) 

		t1 = timer()
		print('GenSpec(%s) : %fs' % (path_save, t1-t0))

	def GenPeak(self, path_dos):
		_, _, k_label, k_point = self.ReadConfig(path_dos.split('/')[1])
		peak_list = ['%s%s' % (l, s) for l in k_label for s in ['p', 'n']]
		Ne = int(re.sub('_Ne', '', re.search('_Ne\d+', path_dos).group()))

		path_save   = re.sub('dos', 'peak', path_dos)
		path_energy = '/'.join(['data', 'energy_Ne%d.dat' % Ne])

		with open(path_energy, 'r') as f: energy = np.genfromtxt(f)
		with open(path_dos, 'r') as f: dos = np.genfromtxt(f, skip_header=1, delimiter=',')
		fermi_idx = np.min(np.where(energy > 0))
		pm = dos[:, :len(config.pm_list)]
		dos = dos[:, len(config.pm_list):]

		t0 = timer()
		
		peak = np.zeros(len(peak_list))
		for dosi in dos:
			dosi = np.reshape(dosi, (len(k_point), Ne))
			ddosi = np.diff(dosi) > 0 # boolean

			peaki = []
			for ddosij in ddosi:
				peak_all = [j+1 for j in range(len(ddosij)-1) if ddosij[j] and not ddosij[j+1]]

				peak_pos = [i for i in peak_all if i > fermi_idx]
				peak_pos = np.min(peak_pos) if len(peak_pos) else 999

				peak_neg = [i for i in peak_all if i < fermi_idx]
				peak_neg = np.max(peak_neg) if len(peak_neg) else -999

				peaki += [peak_pos, peak_neg]
			peak = np.vstack((peak, peaki))
		peak = np.delete(peak, 0, 0)

		np.savetxt(path_save, np.hstack((pm, peak)), fmt='%.10f', delimiter=',', header=','.join(config.pm_list + peak_list))

		t1 = timer()
		print('GenPeak(%s) : %fs' % (path_save, t1-t0))
	
	def CheckDOS(self, path_dos, verbose='t'):
		_, _, k_label, k_point = self.ReadConfig(path_dos.split('/')[1])
		Ne = int(re.sub('Ne', '', re.search('Ne\d+', path_dos).group()))

		path_energy = '/'.join(['data', 'energy_Ne%s.dat' % Ne])

		with open(path_energy, 'r') as f: energy = np.genfromtxt(f)
		de = energy[1] - energy[0]

		with open(path_dos, 'r') as f: dos = np.genfromtxt(f, skip_header=1, delimiter=',')
		pm = dos[:, :len(config.pm_list)]

		dos = dos[:, len(config.pm_list):] * de
		dos_sum = np.add.reduceat(dos, range(0, len(k_point)*Ne, Ne), 1)
		if verbose == 't': print(dos_sum)
		print('DOS sum shape :', dos_sum.shape, '\n')

		dft_tol_min, dft_tol_max = 5, 7
		dft_idx = np.unique(np.where((dos_sum < dft_tol_min) | (dos_sum > dft_tol_max))[0])
		df = pd.DataFrame(np.hstack((pm[dft_idx, :], dos_sum[dft_idx, :])), columns=config.pm_list+k_label)
		df = df.astype({'idx':'int', 'type':'int'})
		df = df.set_index('idx')
		print('Defacted DOS list (tol < %.3f | tol > %.3f):' % (dft_tol_min, dft_tol_max))
		print(df)

	def ShowDOS(self, path_dos, idx=-1, axes=[]):
		idx = int(idx)
		data_type = re.sub('data/', '', re.search('data/[a-z_]+/', path_dos).group())

		_, _, k_label, k_point = self.ReadConfig(path_dos.split('/')[1])
		Ne = int(re.sub('Ne', '', re.search('Ne\d+', path_dos).group()))

		path_energy = '/'.join(['data', 'energy_Ne%s.dat' % Ne])
		path_peak = re.sub('dos', 'peak', path_dos)

		with open(path_energy, 'r') as f: energy = np.genfromtxt(f)
		with open(path_dos, 'r') as f: dos = np.genfromtxt(f, skip_header=1, delimiter=',')

		if idx < 0:
			print('Random Idx')
			idx = np.random.randint(len(dos))
		else: idx = np.where(np.abs(dos[:, 0] - idx) < 1e-6)[0][0]

		if os.path.isfile(path_peak):
			with open(path_peak, 'r') as f: peak = np.genfromtxt(f, skip_header=1, delimiter=',')
			peak = peak[idx, len(config.pm_list):].astype(int)
			peaks = np.split(peak, len(k_point))
		else:
			print('No Peak')
			peaks = np.zeros((len(k_point), 2)).astype(int)

		pm = dos[idx, :len(config.pm_list)]
		pmA = ' '.join(['%s %.1f' % (config.pm_list[i], pm[i]) for i in range(5)]) + '\n'
		pmB = ' '.join(['%s %.3f' % (config.pm_list[i], pm[i]) for i in range(5, len(pm))])

		dos = dos[idx, len(config.pm_list):]
		doss = np.split(dos, len(k_point))

		show = False
		if not len(axes):
			show = True
			fig, axes = plt.subplots(1, len(k_point))

		for i, ax in enumerate(axes):
			ax.plot(doss[i], energy, label=data_type+k_label[i])
			ax.scatter(np.array(doss[i])[peaks[i]], energy[peaks[i]], s=35, alpha=0.6)
			ax.axhline(y=0, ls='--', color='gray')
			ax.legend(handlelength=0.5, handletextpad=0.1)
			if i: ax.get_yaxis().set_visible(False)

		if show:
			fig.suptitle(pmA + pmB)	
			fig.supxlabel('DOS')
			fig.supylabel('Energy')
			fig.savefig('figs/%s.png' % ('_'.join([re.sub('.csv', '', re.sub('/', '_', path_dos)), 'idx%d' % idx])))
			plt.show()

		return axes
