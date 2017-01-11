#!/usr/bin/env python

import numpy as np 
import scipy.linalg as scla
import scipy.constants
import sys
import argparse
import re
import itertools

from Cube import *
from tools import *

from PyQuante.LA2 import geigh,mkdens,trace2,CholOrth,CanOrth,SymOrthCutoff,simx,sym
from PyQuante import SCF, Molecule
from PyQuante.NumWrap import arange
from PyQuante.cints import coulomb_repulsion
from PyQuante import configure_output
from PyQuante.Constants import hartree2ev
from PyQuante.cints import ijkl2intindex as intindex
#from PyQuante.Ints import getT,getS,getV,get1ints,getints,get2JmK

from mpi4py import MPI
import Ints_MPI
from Ints_MPI import getbasis

# print option for numpy
np.set_printoptions(precision=3)

# debug flags
_debug_mpi_ = 0
_print_basis_ = 0
_print_detail_exc_ = 0


####################################################
##	MAIN FUNCTION
####################################################
def main(argv):


	############################
	# Initialize MPI
	############################
	comm = MPI.COMM_WORLD
	rank = comm.rank
	############################

	parser = argparse.ArgumentParser(description='Hartree Fock Calculation from scratch')

	# molecule information
	parser.add_argument('mol',help='xyz file of the molecule',type=str)
	parser.add_argument('-basis',default = 'sto-3g', help='basis set to be used in the calculation',type=str)
	parser.add_argument('-charge',default = 0, help='Charge of the system',type=float)
	parser.add_argument('-units',default = 'angs', help='Units in the xyz file',type=str)

	# HF calculations
	parser.add_argument('-MaxIter',default = 100, help='Maximum number of SCF iterations',type=int)
	parser.add_argument('-eps_SCF',default = 1E-4, help='Criterion for SCF termination',type=float)

	# LR-TDHF arguments
	parser.add_argument('-nb_exc',default = 10, help='Number of excitations to compute in the Davidson diagonalization',type=int)
	parser.add_argument('-tda',default = 0, help='Perform the Tamm-Dancoff approximation',type=int)
	parser.add_argument('-hermitian',default = 1, help='Reduce the TDHD equation to their hermitian form',type=int)

	# export
	parser.add_argument('-nb_print_mo',default = 10, help='Number of orbitals to be written',type=int)
	parser.add_argument('-export_mo',default = 1, help='Export the MO in Gaussian Cube format',type=int)
	parser.add_argument('-export_blender',default = 0, help='Export the MO in bvox format',type=int)

	'''
	Possible basis
	'3-21g' 'sto3g' 'sto-3g' 'sto-6g'
	'6-31g' '6-31g**' '6-31g(d,p)' '6-31g**++' '6-31g++**' '6-311g**' '6-311g++(2d,2p)'
    '6-311g++(3d,3p)' '6-311g++(3df,3pd)'
    'lacvp'
    
    'ccpvdz' 'cc-pvdz' 'ccpvtz' 'cc-pvtz' 'ccpvqz' 'cc-pvqz' 'ccpv5z' 'cc-pv5z' 'ccpv6z' 'cc-pv6z'

    'augccpvdz' 'aug-cc-pvdz' 'augccpvtz' 'aug-cc-pvtz' 'augccpvqz'
    'aug-cc-pvqz' 'augccpv5z' 'aug-cc-pv5z' 'augccpv6z' 'aug-cc-pv6z'    
    'dzvp':'dzvp',

	'''
	
	# done
	args=parser.parse_args()

	if rank == 0:
		print'\n\n=================================================='
		print '== PyQuante - Linear response TDHF calculation  =='
		print '== MPI version on %02d procs                      ==' %(comm.size)
		print '==================================================\n'

	#-------------------------------------------------------------------------------------------
	#
	#									PREPARE SIMULATIONS
	#
	#-------------------------------------------------------------------------------------------

	##########################################################
	##					Read Molecule
	##########################################################

	# read the xyz file of the molecule
	if rank == 0:
		print '\t Read the position of the molecule\n\t',
		print '-'*50

	f = open(args.mol,'r')
	data = f.readlines()
	f.close

	# get the molecule name
	name_mol = re.split(r'\.|/',args.mol)[-2]

	# create the molecule object
	xyz = []
	for i in range(2,len(data)):
		d = data[i].split()
		xyz.append((d[0],(float(d[1]),float(d[2]),float(d[3]))))

	natom = len(xyz)
	mol = Molecule(name=name_mol,units=args.units)
	mol.add_atuples(xyz)
	mol.set_charge(args.charge)
	nelec = mol.get_nel()
	
	if np.abs(args.charge) == 1:
		mol.set_multiplicity(2)
	if args.charge>1:
		print 'charge superior to one are not implemented'

	# get the basis function
	bfs = getbasis(mol,args.basis)
	nbfs = len(bfs)
	nclosed,nopen = mol.get_closedopen()
	nocc = nclosed

	# get the dipole moment in the AO basis
	mu_at_x,mu_at_y,mu_at_z = compute_dipole_atoms(bfs)
	mu_tot = mu_at_x + mu_at_y + mu_at_z

	if rank == 0:
		print '\t\t Molecule %s' %args.mol
		print '\t\t Basis %s' %args.basis
		print '\t\t %d electrons' %nelec 
		print '\t\t %d basis functions' %(nbfs)
	
		if _print_basis_:
			for i in range(nbfs):
				print bfs[i]
	
	# compute all the integrals
	if rank == 0:
		print '\n\t Compute the integrals and form the matrices'
	S,Hcore,Ints = Ints_MPI.getints_mpi(bfs,mol,rank,comm,_debug_=_debug_mpi_)


	
	##########################################################
	##
	##				HF GROUND STATE
	##
	##########################################################
	comm.Barrier()

	######################################################
	# only the master proc computes the HF ground state
	# That should change somehwow
	######################################################
	if rank == 0:

		################################################
		# compute the HF ground state of the system
		################################################

		print '\n\t Compute the ground state HF Ground State\n\t',
		print '-'*50
		L,C,P = rhf(mol,bfs,S,Hcore,Ints,MaxIter=args.MaxIter,eps_SCF=args.eps_SCF)
	
		print '\t Energy of the HF orbitals\n\t',
		print '-'*50
		index_homo = nocc-1
		nb_print = int(min(nbfs,args.nb_print_mo)/2)
		for ibfs in range(index_homo-nb_print+1,index_homo+nb_print+1):
			print '\t\t orb %02d \t occ %1.1f \t\t Energy %1.3f eV' %(ibfs,np.abs(2*P[ibfs,ibfs].real),L[ibfs].real/hartree2ev)


		################################################	
		##	Export the MO in VMD Format
		################################################
		if args.export_mo:

			print '\t Export MO Gaussian Cube format'
			
			index_homo = nocc-1
			nb_print = int(min(nbfs,args.nb_print_mo)/2)
			fmo_names = []
			for ibfs in range(index_homo-nb_print+1,index_homo+nb_print+1):
				if ibfs <= index_homo:
					motyp = 'occ'
				else:
					motyp = 'virt'
				file_name =mol.name+'_mo'+'_'+motyp+'_%01d.cube' %(ibfs)
				xyz_min,nb_pts,spacing = mesh_orb(file_name,mol,bfs,C,ibfs)
				fmo_names.append(file_name)
			

			##########################################################
			##		Export the MO in bvox Blender format
			##########################################################

			if args.export_blender:
				print '\t Export MO volumetric data for Blender'
				bvox_files_mo = []
				for fname in fmo_names:
					fn = cube2blender(fname)
					bvox_files_mo.append(fn)
			
				##	Create the  Blender script to visualize the orbitals
				path_to_files = os.getcwd()
				pdb_file = name_mol+'.pdb'
				create_pdb(pdb_file,args.mol,args.units)

				# create the blender file
				blname = name_mol+'_mo_volumetric.py'
				create_blender_script_mo(blname,xyz_min,nb_pts,spacing,pdb_file,bvox_files_mo,path_to_files)

		
	##########################################################
	##
	##				LR-TDHF Calculations
	##
	##########################################################

	comm.Barrier()

	if rank == 0:
		print '\n\t Compute the LR-TDHF response of the system\n\t',
		print '-'*50

	if rank == 0:
		print '\t\t Transform the 2e integrals in the MO basis'

	# Rearrange the integrals with MPI
	# so far it does not work

	# broadcast the C matrix from 0 to all the procs
	#if rank != 0:
	#	C = np.zeros((nbfs,nbfs))
	#comm.Bcast(C,root=0)

	# rearrange the INTs
	#MOInts_mpi = Ints_MPI.TransformInts_mpi(Ints,C,rank,comm,_debug_mpi_)


	# master proc determine which orbitals to account for
	if rank == 0:

		# rearrange the integrals with only 1 proc.
		MOInts = Ints_MPI.TransformInts(Ints,C)

		# energies of the occupied/virtual orbitals
		eocc,evirt = L[:nocc],L[nocc:]
		nb_hole, nb_elec = len(eocc),len(evirt)

		# index of the excitations
		nb_exc = nb_hole*nb_elec
		ind_hole = range(nocc-1,-1,-1)
		ind_elec = range(nocc,nbfs)
		index_exc = [x for x in itertools.product(ind_hole,ind_elec)]

		# init the matrices
		A = np.zeros((nb_exc,nb_exc))
		if not args.tda:
			B = np.zeros((nb_exc,nb_exc))

		# form the A and B matrices
		for e1 in range(nb_exc):
			for e2 in range(nb_exc):

				i,j = index_exc[e1][0],index_exc[e2][0]
				a,b = index_exc[e1][1],index_exc[e2][1]

				# coulomb/exchange integrals for A
				j_int = MOInts[intindex(i,a,j,b)]
				k_int = MOInts[intindex(i,j,a,b)]

				# diagonal/offdiagonal element of A
				if e1==e2:
					eif = L[a]-L[i]
					A[e1,e2] =  eif + j_int - k_int 
				else: 
					A[e1,e2] = j_int - k_int

				# B matrix
				if not args.tda:
					# coulomb/exchange integrals
					j_int = MOInts[intindex(i,a,b,j)]
					k_int = MOInts[intindex(i,b,a,j)]
					B[e1,e2] = j_int-k_int
					

		# for the total  matrix
		# and diagonalize it
		# in the non-Hermitian case
		if not args.hermitian:

			if args.tda:
				Q = A
				B = np.zeros((nb_exc,nb_exc))
				I = np.eye(nb_exc)
			else:
				Q = np.zeros((2*nb_exc,2*nb_exc))
				Q[:nb_exc,:nb_exc] = A
				Q[nb_exc:,nb_exc:] = np.conj(A)
				Q[:nb_exc,nb_exc:] = B
				Q[nb_exc:,:nb_exc] = np.conj(B)
				I = np.eye(2*nb_exc)
				I[nb_exc:,nb_exc:] *= -1

			# diagonalize the matrix
			if args.nb_exc<nbfs:
				w,Cex = scla.eig(Q,b=I,eigvals=[0,args.nb_exc])
			else:
				w,Cex = scla.eig(Q,b=I)

		# for the total  matrix
		# and diagonalize it
		# in the Hermitian case
		else :

			if args.tda:
				qm = scla.sqrtm(A)
				qp = A
			else:
				qm = scla.sqrtm(A-B)
				qp = A+B

			Q = np.dot(qm,np.dot(qp,qm))
			print '\t\t Diagonalize the %02dx%02d response matrix' %(nb_exc,nb_exc)
			if args.nb_exc<nbfs:
				w2,Cex = scla.eigh(Q,eigvals=[0,args.nb_exc])
			else:
				w2,Cex = scla.eigh(Q)
			w = np.sqrt(w2)


			print '\n\t Energy of HF excitation\n\t',
			print '-'*50
			for iexc in range(len(w)):

				# frequency
				freq = w[iexc].real

				# print the excitaiton in details
				if _print_detail_exc_:
					print_first = 1
					for kxc in range(nb_exc):
						trans = Cex[kxc,iexc]**2
						init,final = index_exc[kxc][0],index_exc[kxc][1]
						osc = 2./3*freq*(np.inner(C[:,init],np.inner(mu_tot,C[:,final]).T))**2
						if trans > 0.001:
							print '\t\t \t \t \t \t \t %02d->%02d (%1.3f %%)\t osc %1.3f' %(init,final,trans,osc)

				# print only the max contribution
				else:
					# maximum contrbution
					index_max = np.argmax(Cex[:,iexc]**2)
					max_trans = Cex[index_max,iexc]**2
					init_max,final_max = index_exc[index_max][0],index_exc[index_max][1]
					osc = 2./3*freq*(np.inner(C[:,init_max],np.inner(mu_tot,C[:,final_max]).T))**2
					print '\t\t exc %02d \t Energy %1.3f eV \t %02d->%02d (%1.3f %%)\t osc %1.3f' \
									%(iexc,freq/hartree2ev,init_max,final_max,max_trans,osc)

				
			
	if rank == 0:

		print '\n\n=================================================='
		print '==                       Calculation done       =='
		print '==================================================\n'


if __name__=='__main__':
	main(sys.argv[1:])
	