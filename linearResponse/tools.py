import numpy as np 
import scipy.linalg as scla
import sys
import argparse
import os
import re
#from tools import *

from PyQuante.LA2 import geigh,mkdens,trace2,mkdens_spinavg,simx,SymOrthCutoff
from PyQuante import SCF, Molecule
from PyQuante.NumWrap import arange
from PyQuante.cints import coulomb_repulsion
from PyQuante import configure_output
from PyQuante.Constants import hartree2ev,bohr2ang
from PyQuante.Ints import getbasis
#from PyQuante.Ints import getT,getS,getV,get1ints,getints,get2JmK
from Ints_MPI import get2JmK_mpi
from scipy.linalg import expm
from PyQuante.Convergence import DIIS



###########################################
## SELF CONSISTANT HF
###########################################
def rhf(mol,bfs,S,Hcore,Ints,MaxIter=100,eps_SCF=1E-4,_diis_=True):

	##########################################################
	##					Get the system information
	##########################################################
	
	# size
	nbfs = len(bfs)

	# get the nuclear energy
	enuke = mol.get_enuke()
		
	# determine the number of electrons
	# and occupation numbers
	nelec = mol.get_nel()
	nclosed,nopen = mol.get_closedopen()
	nocc = nclosed


	if nopen != 0:
		print '\t\t ================================================================='
		print '\t\t Warning : using restricted HF with open shell is not recommended'
		print "\t\t Use only if you know what you're doing"
		print '\t\t ================================================================='

	# get a first DM
	#D = np.zeros((nbfs,nbfs))
	L,C = scla.eigh(Hcore,b=S)
	D = mkdens(C,0,nocc)

	# initialize the old energy
	eold = 0.

	# initialize the DIIS
	if _diis_:
		avg = DIIS(S)
	
	#print '\t SCF Calculations'
	for iiter in range(MaxIter):
		
		# form the G matrix from the 
		# density matrix and  the 2electron integrals
		G = get2JmK_mpi(Ints,D)

		# form the Fock matrix
		F = Hcore + G 

		# if DIIS
		if _diis_:
			F = avg.getF(F,D)

		# diagonalize the Fock matrix
		L,C = scla.eigh(F,b=S)

		# new density mtrix
		D = mkdens(C,0,nocc)
				
		# compute the total energy
		e = np.sum(D*(Hcore+F)) + enuke
		
		print "\t\t Iteration: %d    Energy: %f    EnergyVar: %f"%(iiter,e.real,np.abs((e-eold).real))

		# stop if done
		if (np.abs(e-eold) < eps_SCF) :
			break
		else:
			eold = e

	if iiter < MaxIter-1:
		print("\t\t SCF for HF has converged in %d iterations, Final energy %1.3f Ha\n" % (iiter,e.real))
		
	else:
		print("\t\t SCF for HF has failed to converge after %d iterations")
		sys.exit()

	# compute the density matrix in the
	# eigenabsis of the Fock matrix

	# orthogonalization matrix
	X = SymOrthCutoff(S)
	Xm1 = np.linalg.inv(X)

	# density matrix in ortho basis
	Dp = np.dot(Xm1,np.dot(D,Xm1.T))

	# eigenvector in ortho basis
	Cp = np.dot(Xm1,C)

	# density matrix
	P = np.dot(Cp.T,np.dot(Dp,Cp))

	# done
	return L,C,P


#####################################################
#####################################################

#####################################################
##	Compute non SC Fock Matrix
#####################################################
def compute_F(P,Hcore,X,Ints,mu=0):

	# get the matrix of 2J-K for a given fixed P
	# P is here given in the non- orthonormal basis
	G = get2JmK_mpi(Ints,P)

	# pass the G matrix in the orthonormal basis
	Gp = simx(G,X,'T')
		
	# form the Fock matrix in the orthonomral basis
	F = Hcore + _ee_inter_*Gp + mu

	return F


#####################################################
##	Compute doverlap in atomic basis
#####################################################
def compute_idrj(bfs):
	nbf = len(bfs)
	mu_at = np.zeros((nbf,nbf))
	# form the matrix of the dm
	# between all the combinations
	# of possible ATOMIC orbitals 
	for i in range(nbf):
		bfi = bfs[i]
		for j in range(nbf):
			bfj = bfs[j]
			mu_at[i,j] = bfi.doverlap(bfj,0) + bfi.doverlap(bfj,1) + bfi.doverlap(bfj,2)

	return mu_at

#####################################################
##	Compute dipole moments in the atomic basis
#####################################################
def compute_dipole_atoms(bfs):
	nbf = len(bfs)
	mu_x = mu_y = mu_z = np.zeros((nbf,nbf))
	# form the matrix of the dm
	# between all the combinations
	# of possible ATOMIC orbitals 
	for i in range(nbf):
		bfi = bfs[i]
		for j in range(nbf):
			bfj = bfs[j]			
			mu_x[i,j] = bfi.multipole(bfj,1,0,0)
			mu_y[i,j] = bfi.multipole(bfj,0,1,0)
			mu_z[i,j] = bfi.multipole(bfj,0,0,1)	
	return mu_x,mu_y,mu_z


#####################################################
##	create a pdb file from the xyz
#####################################################
def create_pdb(pdb_file,xyz_file,units):

	# create the pdb file if it does not exists
	if not os.path.isfile(pdb_file):

		# if it was provided in bohr 
		# change to angstom
		if units == 'bohr':

			# read the file
			f = open(xyz_file,'r')
			data = f.readlines()
			f.close()

			#write a xyz file in angstrom
			name_mol = re.split(r'\.|/',xyz_file)[-2]
			fname = name_mol+'_angs.xyz'
			f = open(fname,'w')
			f.write('%s\n' %data[0])
			for i in range(2,len(data)):
				l = data[i].split()
				if len(l)>0:
					x,y,z = bohr2ang*float(l[1]),bohr2ang*float(l[2]),bohr2ang*float(l[3])
					f.write('%s %f %f %f\n' %(l[0],x,y,z))
			f.close()

			# convert to pdb
			os.system('obabel -ixyz %s -opdb -O %s' %(fname,pdb_file))






