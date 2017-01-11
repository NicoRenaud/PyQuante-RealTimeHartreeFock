
from PyQuante.NumWrap import zeros,dot,reshape
from PyQuante.cints import ijkl2intindex as intindex
from PyQuante.Basis.Tools import get_basis_data
import numpy as np

import sys
import time

import logging
from PyQuante.Ints import getbasis, sortints,fetch_jints,fetch_kints
from PyQuante.Ints import getT,getS,getV,get1ints,getints,get2JmK
from PyQuante.cints import ijkl2intindex

logger = logging.getLogger("pyquante")
sym2powerlist = {
    'S' : [(0,0,0)],
    'P' : [(1,0,0),(0,1,0),(0,0,1)],
    'D' : [(2,0,0),(1,1,0),(1,0,1),(0,2,0),(0,1,1),(0,0,2)],
    'F' : [(3,0,0),(2,1,0),(2,0,1),(1,2,0),(1,1,1),(1,0,2),
           (0,3,0),(0,2,1),(0,1,2), (0,0,3)]
    }

sorted = True
jints = {}
kints = {}

###################################################
##  get the basis info of the molecule
##  as this is not time consuming no MPI there 
###################################################
def getbasis(atoms,basis_data=None,**kwargs):
    """\
    bfs = getbasis(atoms,basis_data=None)
    
    Given a Molecule object and a basis library, form a basis set
    constructed as a list of CGBF basis functions objects.
    """
    from PyQuante.Basis.basis import BasisSet
    if not basis_data:
        basis_data = kwargs.get('basis_data')
    if kwargs.get('bfs'):
        return kwargs.get('bfs')
    elif not basis_data:
        basis = kwargs.get('basis')
        if basis:
            basis_data = get_basis_data(basis)
    return BasisSet(atoms, basis_data, **kwargs)

###################################################
##  wrapper to get all the integrals at once
###################################################
def getints_mpi(bfs,atoms,rank,comm,**kwargs):

	# that I have no idea ..
    if kwargs.get('integrals'):
        return kwargs.get('integrals')
    logger.info("Calculating Integrals...")

    # debug flag
    _debug_ = kwargs.pop('_debug_',False)

    # get the 1elec integral
    S,h = get1ints_mpi(bfs,atoms,rank,comm,_debug_=_debug_)

    # get the two electron integrals
    Ints = get2ints_mpi(bfs,rank,comm,x=0.5,EPS=6,_debug_=_debug_)

    logger.info("Integrals Calculated.")
    return S,h,Ints

###################################################
##  Compute the 1electron integrals
###################################################
def get1ints_mpi(bfs,atoms,rank,comm,_debug_=False):

    "Form the overlap S and h=t+vN one-electron Hamiltonian matrices"
    nbf = len(bfs)

    # determine which orbitals we will compute
    npp = nbf/comm.size
    if rank < comm.size-1:
    	terms = range(rank*npp,(rank+1)*npp)
    else:
    	terms = range(rank*npp,nbf)

    nterm = len(terms)
    S = zeros((nterm,nbf))
    h = zeros((nterm,nbf))

    # print for debugging
    if _debug_:
    	comm.Barrier()
    	if rank == 0:
    		print '\t Computing the 1electron integrals on %d procs' %comm.size
    		print '\t There is %d basis functions in the system' %(len(bfs))
    		print '\t',
    		print '-'*50
    	print '\t [%03d] computes %d terms ' %(rank,nterm)

    # Compute the terms
    ilocal = 0
    for i in terms:
    	bfi = bfs[i]
    	for j in xrange(nbf):
    		bfj = bfs[j]
    		S[ilocal,j] = bfi.overlap(bfj)
    		h[ilocal,j] = bfi.kinetic(bfj)
    		for atom in atoms:
    			h[ilocal,j] = h[ilocal,j] + atom.Z*bfi.nuclear(bfj,atom.pos())
    	ilocal += 1         

	##########################################
	# Start communication between nodes
	# to gather the total matrix on 0
	##########################################

	# wait for everyone to be here
    comm.barrier()
    

	# Create the receive buffers
	# only rank = 0 will receive
    if rank == 0:	
    	Stot = np.zeros((nbf,nbf))
    	htot = np.zeros((nbf,nbf))
    	Stot[0:npp,:] = S
    	htot[0:npp] = h
    else:
    	htot = None
    	Stot = None

    if rank == 0 and _debug_:
    	print '\n\t Procs 0 Gather all 1electron matrix'
    # exchange the 1 electron matrix
    if rank == 0:
    	for iC in range(1,comm.size):
    		if iC < comm.size-1:
    			recvbuff = np.zeros((npp,nbf))
    			comm.Recv(recvbuff,source=iC)
    			htot[npp*iC:npp*(iC+1),:] = recvbuff
    			if _debug_:
    				print '\t\t\t Proc [%03d] : data received from [%03d]' %(0,iC)
    		else:
    			recvbuff = np.zeros((nbf-iC*npp,nbf))
    			comm.Recv(recvbuff,source=iC)
    			htot[npp*iC:,:] = recvbuff
    			if _debug_:
    				print '\t\t\t Proc [%03d] : data received from [%03d]' %(0,iC)
    else:
    	comm.Send(h,dest=0)

    comm.Barrier()
    if rank == 0 and _debug_:
    	print '\n\t Procs 0 Gather all overlap matrix'
    # exchange the overlap matrix
    if rank == 0:
    	for iC in range(1,comm.size):
    		if iC < comm.size-1:
    			recvbuff = np.zeros((npp,nbf))
    			comm.Recv(recvbuff,source=iC)
    			Stot[npp*iC:npp*(iC+1),:] = recvbuff
    			if _debug_:
    				print '\t\t\t Proc [%03d] : data received from [%03d]' %(0,iC)
    		else:
    			recvbuff = np.zeros((nbf-iC*npp,nbf))
    			comm.Recv(recvbuff,source=iC)
    			Stot[npp*iC:,:] = recvbuff
    			if _debug_:
    				print '\t\t\t Proc [%03d] : data received from [%03d]' %(0,iC)
    else:
    	comm.Send(S,dest=0)	

    # done
    return Stot,htot

###################################################
##  Get the 2 electron integrals with MPI
###################################################
def get2ints_mpi(basis,rank,comm,x=0.5,EPS=6,_debug_=False):

	import PyQuante.clibint

	# total size
	lenbasis = len(basis.bfs)
	NTOT = lenbasis**4
	Ints = np.zeros(NTOT,dtype=np.float64)

	# number of shells
	lenshells = len(basis.shells)

	# compute the number of terms
	# that this proc will compute
	nterm = np.zeros(comm.size)
	for i in range(1,comm.size):
		nterm[i] = x*(1-x)**(i-1)+nterm[i-1]
	nterm[0] = 0

	# fix the last two terms
	nterm[-1] += x*(1.-nterm[-1])
	nterm = np.append(nterm,1)

	# get the indexes
	nterm *= lenshells
	nterm = nterm.astype(int)
	terms = range(nterm[rank],nterm[rank+1])
	

	# particular case if we have onlye 1 proc
	if comm.size == 1:
		terms = range(len(basis.shells))

	if _debug_:
		comm.Barrier()
		if rank == 0:
			print '\n\t Computing the 2electron integrals on %d procs' %(comm.size)
			print '\t There is %d shells  in the system' %(len(basis.shells))
			print '\t',
			print '-'*50
		print '\t\t Proc [%03d] computes %d terms' %(rank,len(terms))

	#each proc runs over all the terms
	# that were assigned to it
	nb_term = 0
	t0 = time.time()
	for i in terms:
		
		# get the the basis sheel
		a = basis.shells[i]

		# second loop
		for j,b in enumerate(basis.shells[:i+1]):

			# third loop
			for k,c in enumerate(basis.shells):

				# fourth loop
				for l,d in enumerate(basis.shells[:k+1]):

					if (i+j)>=(k+l):
						nb_term += 1
						PyQuante.clibint.shell_compute_eri(a,b,c,d,Ints)

	if _debug_:
		print '\t\t Calculation [%03d] of %d terms done in %f sec ' %(rank,nb_term,time.time()-t0)
		
	# wait for everyone to be here	
	comm.barrier()

	##########################################
	# Start communication between nodes
	# to aggregate the values of the Integrals
	##########################################
	if rank == 0 and _debug_:
	   	print '\n\t\t Procs 0 Gather all the values'


	# PROC 0 receive the data
	if rank == 0:	
		rec_data = np.zeros(NTOT)
		for iC in range(1,comm.size):

				comm.Recv(rec_data,source=iC)

				# unfortunately clibint compute some 
				# elements several time and we need to check that
				# find the elements that are already present
				rec_data[Ints!=0] = 0 
				Ints += rec_data

				if _debug_:
					print '\t\t\t Proc [%03d] : data received from [%03d]' %(0,iC)

	# PROC n (n>0) send their data
	else:
		comm.Send(Ints,dest=0)

	##########################################
	# Broadcast the values of Ints for sorting
	##########################################
	if rank == 0 and _debug_:
		print '\n\t\t Broadcast the data from 0 to all procs'
	comm.Bcast( Ints, root=0 )

	# sort the integrals
	if sorted :
		if rank ==0 and _debug_:
			print '\n\t\t Sort Integrals'
		sortints_mpi(rank,comm,len(basis),Ints,_debug_=_debug_)

	if _debug_:
		print '\t\t Proc [%03d] contains %d elements in Jints ' %(rank,len(jints))
	
	# done
	return Ints
###################################################
##  sort the 2 electron integrals
###################################################
def sortints_mpi(rank,comm,nbf,Ints,x=0.5,_debug_=False):

	#########################################
	# determine the number of terms
	# that this proc will compute
	#########################################
	nterm = np.zeros(comm.size)
	for i in range(1,comm.size):
		nterm[i] = x*(1-x)**(i-1)+nterm[i-1]
	nterm[0] = 0

	# fix the last two terms
	nterm[-1] += x*(1.-nterm[-1])
	nterm = np.append(nterm,1)
	
	# get the indexes
	nterm *= nbf
	nterm = nterm.astype(int)
	terms = range(nterm[rank],nterm[rank+1])

	#particular case if we have only 1 proc.
	if comm.size == 1:
		terms = range(nbf)

	if _debug_:
		print '\t\t Proc [%03d] sort %d ints' %(rank,len(terms))

	#########################################
	# fill up the the dictionaires that are
	# defined as global variable 
	# each proc fills different elements
	#########################################
	nt = 0
	t0 = time.time()
	for i in terms:
		for j in xrange(i+1):
			nt += 1
			jints[i,j] = fetch_jints(Ints,i,j,nbf)
			kints[i,j] = fetch_kints(Ints,i,j,nbf)

	if _debug_:
		print '\t\t Sorting [%03d] of %d terms done in %f sec ' %(rank,nt,time.time()-t0)

	#########################################
	# send all the dictionaries to rank = 0
	# and update its jints and kints dictionaries
	# after that only proc 0 contains complete dictionaries
	#########################################
	comm.Barrier()
	if _debug_ and rank == 0:
		print '\n\t\t Gather all the dictionaries on 0'

	# gather all the dictionaries fo jints together
	if rank == 0:
		for iC in range(1,comm.size):
			new_data = comm.recv(source=iC,tag=11)
			jints.update(new_data)
	else:	
		comm.send(jints,dest=0,tag=11)
	comm.Barrier()

	#gather all the kints dictionaries on 0	
	if rank == 0:
		for iC in range(1,comm.size):
			new_data = comm.recv(source=iC,tag=11)
			kints.update(new_data)
	else:	
		comm.send(kints,dest=0,tag=11)
	comm.Barrier()

	return

###################################################
## Fetch the 2JmK matrix
###################################################
def get2JmK_mpi(Ints,D):
    "Form the 2J-K integrals corresponding to a density matrix D"
    nbf = D.shape[0]
    D1d = reshape(D,(nbf*nbf,)) #1D version of Dens
    G = zeros((nbf,nbf),'complex64')
    for i in xrange(nbf):
        for j in xrange(i+1):
            if sorted:
                temp = 2*jints[i,j]-kints[i,j]
            else:
                temp = 2*fetch_jints(Ints,i,j,nbf)-fetch_kints(Ints,i,j,nbf)
            G[i,j] = dot(temp,D1d)
            G[j,i] = np.conj(G[i,j])
    G = G.real
    return G

#####################################################
##	FROM PYQUANTE
##  Transform the 2e integral from the AO basis
##  To the MO basis 
#####################################################
def TransformInts(Ints,orbs):

    """O(N^5) 4-index transformation of the two-electron integrals. Not as
    efficient as it could be, since it inflates to the full rectangular
    matrices rather than keeping them compressed. But at least it gets the
    correct result."""

    nbf,nmo = orbs.shape
    totlen = nmo*(nmo+1)*(nmo*nmo+nmo+2)/8

    temp = np.zeros((nbf,nbf,nbf,nmo),'d')
    tempvec = np.zeros(nbf,'d')
    temp2 = np.zeros((nbf,nbf,nmo,nmo),'d')

    mos = range(nmo) # preform so we don't form inside loops
    bfs = range(nbf)

    # Start with (mu,nu|sigma,eta)
    # Unpack aoints and transform eta -> l
    for mu in bfs:
        for nu in bfs:
            for sigma in bfs:
                for l in mos:
                    for eta in bfs:
                        tempvec[eta] = Ints[ijkl2intindex(mu,nu,sigma,eta)]
                    temp[mu,nu,sigma,l] = np.dot(orbs[:,l],tempvec)

    # Transform sigma -> k
    for mu in bfs:
        for nu in bfs:
            for l in mos:
                for k in mos:
                    temp2[mu,nu,k,l] = np.dot(orbs[:,k],temp[mu,nu,:,l])

    # Transform nu -> j
    for mu in bfs:
        for k in mos:
            for l in mos:
                for j in mos:
                    temp[mu,j,k,l] = np.dot(orbs[:,j],temp2[mu,:,k,l])

    # Transform mu -> i and repack integrals:
    MOInts = np.zeros(totlen,'d')
    for i in mos:
        for j in xrange(i+1):
            ij = i*(i+1)/2+j
            for k in mos:
                for l in xrange(k+1):
                    kl = k*(k+1)/2+l
                    if ij >= kl:
                        ijkl = ijkl2intindex(i,j,k,l)
                        MOInts[ijkl] = np.dot(orbs[:,i],temp[:,j,k,l])

    del temp,temp2,tempvec #force garbage collection now
    return MOInts


#####################################################
##	MPI VERSION OF THE PYQUANTE ROUTINE
##  Transform the 2e integral from the AO basis
##  To the MO basis 
#####################################################
def TransformInts_mpi(Ints,orbs,rank,comm,_debug_=True):

    """
    O(N^5) 4-index transformation of the two-electron integrals. Not as
    efficient as it could be, since it inflates to the full rectangular
    matrices rather than keeping them compressed. But at least it gets the
    correct result.
    """

    '''
    THIS ROUTINE DOES NOT WORK THE PARALLELISATION
    IS HERE COMPLIATED AS THE THE FINAL RESULTS
    NEEDS ALL THE VALUES OF THE OUTER LOOP ALL THE time
    WE SHOULD FINE A WAY OF MAKING IT PARALLEL THOUGH ... 
    PLUS THE MOINTS LOOP IS UNBALANCED WITH THE IF IJ<=kl
    IT NEEDS A SPECIAL TREATMENT TO DIVIDE THE LOAD
    '''
 
    nbf_tot,nmo_tot = orbs.shape
    totlen = nmo_tot*(nmo_tot+1)*(nmo_tot*nmo_tot+nmo_tot+2)/8

    # number of terms per process
    nmo_pp = int(nmo_tot/comm.size)
    nbf_pp = int(nbf_tot/comm.size)

    # precompute the index each process takes care of
    mos_term = range(rank*nmo_pp,(rank+1)*nmo_pp)
    bfs_term = range(rank*nbf_pp,(rank+1)*nbf_pp)


    # for the inner loops we still have the full range
    mos = range(nmo_tot) 
    bfs = range(nbf_tot)

    # special case for the last one
    if rank == comm.size-1:
    	mos_term = range(rank*nmo_pp,nmo_tot)
    	bfs_term = range(rank*nbf_pp,nbf_tot)

    # actual number of terms calculated
    nbf = len(bfs_term)
    nmo = len(mos_term)

    # initialize the temp vectors
    temp = np.zeros((nbf_tot,nbf_tot,nbf_tot,nmo_tot),'d')
    tempvec = np.zeros(nbf_tot,'d')
    temp2 = np.zeros((nbf_tot,nbf_tot,nmo_tot,nmo_tot),'d')

    if rank == 0:
    	print '\n\t Rearrange the 2electron integrals on %d procs' %(comm.size)
    	print '\t There is %d BFS in the system' %(nbf_tot)
    	print '\t',
    	print '-'*50
    print '\t\t Proc [%03d] computes %d terms' %(rank,len(bfs_term))

    t0 = time.time()

	##########################################
	# Each proc compute a part of the temp
	# vector that will be assembled afterwards
	##########################################

    # Start with (mu,nu|sigma,eta)
    # Unpack aoints and transform eta -> l
    print bfs_term
    for mu in bfs:
        for nu in bfs:
            for sigma in bfs:
                for l in mos:
                    for eta in bfs:
                        tempvec[eta] = Ints[ijkl2intindex(mu,nu,sigma,eta)]
                    temp[mu,nu,sigma,l] = np.dot(orbs[:,l],tempvec)

    print temp
    # Transform sigma -> k
    for mu in bfs_term:
        for nu in bfs:
            for l in mos:
                for k in mos:
                    temp2[mu,nu,k,l] = np.dot(orbs[:,k],temp[mu,nu,:,l])

    # Transform nu -> j
    for mu in bfs_term:
        for k in mos:
            for l in mos:
                for j in mos:
                    temp[mu,j,k,l] = np.dot(orbs[:,j],temp2[mu,:,k,l])


    if _debug_:
		print '\t\t Calculation [%03d] of %d terms done in %f sec ' %(rank,len(bfs_term),time.time()-t0)

	##########################################
	# Start communication between nodes
	# to aggregate the values of the temp
	##########################################

    comm.Barrier()

    # PROC 0 receive the data
    if rank == 0:	

    	if _debug_:
    		print '\t\t Gather all the data on 0'

    	rec_data = np.zeros((nbf_tot,nbf_tot,nbf_tot,nmo_tot),'d')
    	for iC in range(1,comm.size):

    		# receive the data
    		comm.Recv(rec_data,source=iC)

    		rec_data[temp!=0] = 0 
    		temp += rec_data

    		if _debug_:
    			print '\t\t\t Proc [%03d] : data received from [%03d]' %(0,iC)

    # PROC n send data
    else:
    	comm.Send(temp,dest=0)

	# force garbage collection now
    if rank == 0:
    	del rec_data 

	##########################################
	# Broadcast the values of temp to all
	# to finalize the transofrmation
	##########################################

    if rank == 0 and _debug_:
    	print '\n\t\t Broadcast the data from 0 to all procs'
    comm.Bcast( temp, root=0 )

	##########################################
	# Each procs process a part of MOInts
	# that are then aggregated 
	##########################################

    if rank == 0:
    	print '\n\t Compute MO 2electron integrals on %d procs' %(comm.size)
    	print '\t There is %d MO in the system' %(nmo_tot)
    	print '\t',
    	print '-'*50
    print '\t\t Proc [%03d] computes %d terms' %(rank,len(mos_term))

    # Transform mu -> i and repack integrals:
    MOInts = np.zeros(totlen,'d')
    for i in mos:
        for j in xrange(i+1):
            ij = i*(i+1)/2+j
            for k in mos:
                for l in xrange(k+1):
                    kl = k*(k+1)/2+l
                    if ij >= kl:
                        ijkl = ijkl2intindex(i,j,k,l)
                        MOInts[ijkl] = np.dot(orbs[:,i],temp[:,j,k,l])

    #force garbage collection now
    del temp,temp2,tempvec 

	##########################################
	# Start communication between nodes
	# to aggregate the values of the MOint
	##########################################

    comm.Barrier()
    #print MOInts
    # PROC 0 receive the data
    if rank == 0:	

    	if _debug_:
    		print '\t\t Gather all the data on 0'

    	rec_data = np.zeros(totlen,'d')
    	for iC in range(1,comm.size):

    		# receive the data
    		comm.Recv(rec_data,source=iC)

    		rec_data[MOInts!=0] = 0 
    		MOInts += rec_data

    		if _debug_:
    			print '\t\t\t Proc [%03d] : data received from [%03d]' %(0,iC)

    # PROC n send data
    else:
    	comm.Send(MOInts,dest=0)

    # remove the unwanted
    if rank == 0:
    	del rec_data

	# done
    return MOInts






###################################################
## test function
###################################################
def test():


	from PyQuante import Molecule
	from mpi4py import MPI

	############################
	# Initialize MPI
	############################
	comm = MPI.COMM_WORLD
	rank = comm.rank
	############################

	if rank == 0:
		print '\n'
		print '#'*60
		print '# Parallel PyQuante'
		print '# Integral calculation with MPI'
		print '#'*60
		tinit = time.time()
		print '\n'

	#####################################
	#	Creates the molecule
	#####################################
	mol = 'benzene.xyz'
	units = 'angs'
	basis_set = '6-31g'

	# read the xyz file of the molecule
	if rank == 0:
		print '\t Read molecule position'

	f = open(mol,'r')
	data = f.readlines()
	f.close


	# create the molecule object
	xyz = []
	for i in range(2,len(data)):
		d = data[i].split()
		xyz.append((d[0],(float(d[1]),float(d[2]),float(d[3]))))

	natom = len(xyz)
	mol = Molecule(name='molecule',units=units)
	mol.add_atuples(xyz)
	nelec = mol.get_nel()

	#####################################
	# get the basis function
	#####################################
	basis = getbasis(mol,basis_set)
	nbfs = len(basis)
	nclosed,nopen = mol.get_closedopen()
	nocc = nclosed

	#####################################
	# Print for debug
	#####################################
	if rank == 0:	
		print '\t Basis %s' %basis_set
		print '\t %d basis functions' %(nbfs)
		print '\t %d shells\n ' %(len(basis.shells))

		if 0:
			for i in range(nbfs):
				print basis[i]

	comm.barrier()

	#####################################
	# compute  all the integrals
	#####################################	
	S,h,Ints = getints_mpi(basis,mol,rank,comm,_debug_=True)


	comm.barrier()
	if rank == 0:
		tfinal = time.time()
		print '\n\t Total computation time %f ' %(tfinal-tinit)
	
	

###################################################
if __name__ == '__main__':
	test()


