#define USE_MATH_DEFINES

#define DOS_IDX  (Nkp*Ne*i + Ne*j + m)
#define BAND_IDX (Nkp*(2*Nb)*i + (2*Nb)*j + n)

#include <omp.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void Band2DOS(int Nkp, int Nb, int Ne, int Nband, double *band, double *energy, double *eta, double *dos) {
	int i, j, m, n;

	for(i=0; i<Nband; i++) {
		for(j=0; j<Nkp; j++) {
			for(m=0; m<Ne; m++) {
				for(n=0; n<Nb; n++) {
					dos[DOS_IDX] += (eta[m] / (pow(eta[m], 2) + pow(energy[m] - band[BAND_IDX], 2))) * band[BAND_IDX + Nb];
				}
				dos[DOS_IDX] /= M_PI;
			}
		}
	}	
}
