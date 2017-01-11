import numpy as np 
import sys

##########################################################
##			PLOT THE DATA WITH MATPLOTLIB
##########################################################
def plot(T,FIELD,N,Q,MU,cutlow=0.015,cuthigh=0.95):


	print '\n\t Plotting the orbitals populations'
	print '\t and the Lowdin atomic charges'
	import matplotlib.pyplot as plt

	f,ax = plt.subplots(5,sharex=True)
	nbfs = len(N)
	natom = len(Q)

	# plot the field
	FIELD -= np.min(FIELD)
	if np.max(FIELD) != 0:
		FIELD /= np.max(FIELD)
	ax[0].plot(T,FIELD,linewidth=2)

	# plot tie dipole
	ax[1].plot(T,MU,linewidth=2)

	# plot the population of the orbitals
	for i in range(nbfs):
		ax[2].plot(T,N[i,:],linewidth=2)
		ax[3].plot(T,N[i,:],linewidth=2)
	ax[2].set_ylim(cuthigh, 1.) 
	ax[3].set_ylim(0, cutlow) 
	ax[2].spines['bottom'].set_visible(False)
	ax[3].spines['top'].set_visible(False)
	ax[2].xaxis.tick_top()
	ax[2].tick_params(labeltop='off') 
	ax[3].xaxis.tick_bottom()

	d = .015  # how big to make the diagonal lines in axes coordinates
	# arguments to pass plot, just so we don't keep repeating them
	kwargs = dict(transform=ax[2].transAxes, color='k', clip_on=False)
	ax[2].plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
	ax[2].plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

	kwargs.update(transform=ax[3].transAxes)  # switch to the bottom axes
	ax[3].plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
	ax[3].plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

	# plot the Lowdin charge
	for i in range(natom):
		ax[4].plot(T,Q[i,:],linewidth=2)
	plt.savefig('population.png')


##########################################################
##			PLOT THE DATA WITH MATPLOTLIB
##########################################################
def plot1winpop(T,FIELD,N,Q,MU,cutlow=0.015,cuthigh=0.95):


	print '\n\t Plotting the orbitals populations'
	print '\t and the Lowdin atomic charges'
	import matplotlib.pyplot as plt

	f,ax = plt.subplots(4,sharex=True)
	nbfs = len(N)
	natom = len(Q)

	# plot the field
	FIELD -= np.min(FIELD)
	if np.max(FIELD) != 0:
		FIELD /= np.max(FIELD)
	ax[0].plot(T,FIELD,linewidth=2)

	# plot tie dipole
	ax[1].plot(T,MU,linewidth=2)

	# plot the population of the orbitals
	for i in range(nbfs):
		ax[2].plot(T,N[i,:],linewidth=2)
		
	ax[2].set_ylim(0., 1.) 
	
	# plot the Lowdin charge
	for i in range(natom):
		ax[3].plot(T,Q[i,:],linewidth=2)
	plt.savefig('population.png')



##########################################################
##			RE-PLOT THE DATA WITH MATPLOTLIB
##########################################################
def replot():
	# load the data	
	T = np.loadtxt('time.dat')

	N = np.loadtxt('orb_pops.dat')
	Q = np.loadtxt('charges.dat')
	MU = np.loadtxt('dipole.dat')
	FIELD = np.loadtxt('field.dat')

	# plot
	plot1winpop(T,FIELD,N,Q,MU,cutlow=1,cuthigh=0.)

def extract_for_gnuplot():

	# load the data	
	T = np.loadtxt('time.dat')
	N = np.loadtxt('orb_pops.dat')
	
	Q = np.loadtxt('charges.dat')
	MU = np.loadtxt('dipole.dat')
	FIELD = np.loadtxt('field.dat')
	N = N.T
	Q = Q.T

	nT=len(T)
	nbs = len(N[0])
	nat = len(Q[0])

	nC = nat+nbs+3
	data = np.zeros((nT,nC))
	
	for i in range(nT):
		data[i,0] = T[i]
		data[i,1] = MU[i]
		data[i,2:nbs+2] = N[i,:]
		data[i,nbs+2:-1] = Q[i,:]
		data[i,-1] = FIELD[i]
		
	print nC
	np.savetxt('data_gnuplot.dat',data)

if __name__=='__main__':
	#replot()
	extract_for_gnuplot()

