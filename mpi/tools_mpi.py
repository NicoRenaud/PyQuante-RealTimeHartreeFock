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

_ee_inter_ = 1

###########################################
## SELF CONSISTANT HF
###########################################
def rhf(mol,bfs,S,Hcore,Ints,mu=0,MaxIter=100,eps_SCF=1E-4,_diis_=True):

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

	# orthogonalization matrix
	X = SymOrthCutoff(S)

	# we desactivate ee interaction when there is only 1 electron
	if nelec == 1:
		_ee_inter_ = 0
	else:
		_ee_inter_ = 1

	if _ee_inter_ == 0:
		print '\t\t ==================================================='
		print '\t\t == Electrons-Electrons interactions desactivated =='
		print '\t\t ==================================================='

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
		F = Hcore + _ee_inter_*G + mu

		# if DIIS
		if _diis_:
			F = avg.getF(F,D)

		# orthogonalize the Fock matrix
		Fp = np.dot(X.T,np.dot(F,X))
		
		# diagonalize the Fock matrix
		Lp,Cp = scla.eigh(Fp)
		
		# form the density matrix in the OB
		if nopen == 0:
			Dp = mkdens(Cp,0,nocc)
		else:
			Dp = mkdens_spinavg(Cp,nclosed,nopen)

		# pass the eigenvector back to the AO
		C = np.dot(X,Cp)

		# form the density matrix in the AO
		if nopen == 0:
			D = mkdens(C,0,nocc)
		else:
			D = mkdens_spinavg(C,nclosed,nopen)

		# compute the total energy
		e = np.sum(D*(Hcore+F)) + enuke
		
		print "\t\t Iteration: %d    Energy: %f    EnergyVar: %f"%(iiter,e.real,np.abs((e-eold).real))

		# stop if done
		if (np.abs(e-eold) < eps_SCF) :
			break
		else:
			eold = e

	if iiter < MaxIter:
		print("\t\t SCF for HF has converged in %d iterations, Final energy %1.3f Ha\n" % (iiter,e.real))
		
	else:
		print("\t\t SCF for HF has failed to converge after %d iterations")



	# compute the density matrix in the
	# eigenbasis of F
	P = np.dot(Cp.T,np.dot(Dp,Cp))
	#print D

	return Lp,C,Cp,F,Fp,D,Dp,P,X


#####################################################
#####################################################

#####################################################
##	Field
#####################################################
def compute_field(t,**kwargs):

	# get the argumetns
	field_form = kwargs.get('fform','sin')
	intensity = kwargs.get('fint')
	frequency = kwargs.get('ffreq')

	# continuous sinusoidal i.e. no enveloppe
	if field_form == 'sin':
		E = intensity*np.sin(frequency*t)

	# gaussian enveloppe
	elif field_form == 'gsin':
		t0 = kwargs.get('t0')
		s = kwargs.get('s')
		g = np.exp(-(t-t0)**2/s**2)
		E = g*intensity*np.sin(frequency*t)

	# triangular up and down
	elif field_form == 'linear':
		t0 = kwargs.get('t0')
		if t<t0:
			g = (1-(t0-t)/t0)
		elif t<2*t0:
			g = (1-(t-t0)/t0)
		else:
			g = 0
		E = g*intensity*np.sin(frequency*t)

	# triangular up-flat-dow
	elif field_form == 'linear_flat':
		tstep = 2*np.pi/frequency
		if t<tstep:
			g = t/tstep
		elif t<2*tstep:
			g=1.0
		elif t<3*tstep:
			g = (3.-t/tstep)
		else:
			g = 0
		E = g*intensity*np.sin(frequency*t)

	# if not recognized
	else:
		print 'Field form %s not recognized' %field_form
		sys.exit()
	return E

#####################################################
##	Propagate the field
#####################################################
def propagate_dm(D,F,dt,**kwargs):

	_prop_ = 'padme'
	method = kwargs.get('method')

	if method == 'relax':

		# direct diagonalization
		if _prop_ == 'direct':

			'''
			Warning : That doesn't work so well
			But I don't know why
			'''
			lambda_F,U_F = scla.eig(F)
			
			v = np.exp(1j*dt*lambda_F)
			df = np.diag(simx(D,U_F,'N'))
			prod = np.diag(v*df*np.conj(v))
			Dq = simx(prod,U_F,'T')

		# padde approximation			
		if _prop_ == 'padme':

			U = expm(1j*F*dt)
			D = np.dot(U,np.dot(D,np.conj(U.T)))
			#print D
		
	else:
		print 'Propagation method %s not regognized' %method
		sys.exit()

	return D


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
def compute_dipole_atoms(bfs,fdir):
	nbf = len(bfs)
	mu_at = np.zeros((nbf,nbf))
	# form the matrix of the dm
	# between all the combinations
	# of possible ATOMIC orbitals 
	for i in range(nbf):
		bfi = bfs[i]
		for j in range(nbf):
			bfj = bfs[j]
			if fdir == 'x':
				mu_at[i,j] = bfi.multipole(bfj,1,0,0)
			elif fdir == 'y':
				mu_at[i,j] = bfi.multipole(bfj,0,1,0)
			elif fdir == 'z':
				mu_at[i,j] = bfi.multipole(bfj,0,0,1)
			elif fidr == 'sum':
				mu_at[i,j] = bfi.multipole(bfj,1,0,0)+bfi.multipole(bfj,0,1,0)+bfi.multipole(bfj,0,0,1)
			else:
				print '\t Error : direction %s for the field not recognized\n\t Options are x,y,z,sum\n' %fdir
				sys.exit()
	return mu_at

#####################################################
##	Compute molecular dipole moments
#####################################################
def compute_dipole_orbs(bfs,E,C,mu_at):	
	nbf = len(bfs)
	mu_orb = np.zeros((nbf,nbf),dtype='complex64')
	# form the matrix of the dm
	# between all the combinations
	# of possible ORBITALS orbitals 
	for a in range(nbf-1):
		for b in range(a+1,nbf):
			cmat = np.outer(C[:,a],C[:,b])
			mu_orb[a,b] = 1j/(E[a]-E[b])*np.sum(cmat*mu_at)
			mu_orb[b,a] = -mu_orb[a,b]

	# return
	return mu_orb

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



