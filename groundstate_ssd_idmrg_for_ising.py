# 2017/08/03
#ssd_iDMRG for transverse Ising

from scipy import integrate
from scipy.linalg import expm
from pylab import *
import pylab as pl
import numpy as np
from scipy.linalg import svd
import scipy.sparse.linalg.eigen.arpack as arp
import scipy.sparse as sparse 
np.set_printoptions(precision=3)

""" Conventions:
B[i,a,b] has axes (physical, left virtual, right virtual),
W[a,b,i,j] has axes (virtual left, virtual right, physical out, physical in)
S[i] are schmidt values between sites (i, i+1),
H_bond[i] is the bond hamiltonian between (i,i+1) with (only physical)
axes (out left, out right, in left, in right)"""





def init_fm_mps(L):
    """ Return FM Ising MPS"""
    d = 2
    B = []
    s = []
    for i in range(L):
        B.append(np.zeros([2,1,1])); B[-1][0,0,0]=1
        s.append(np.ones([1]))
    s.append(np.ones([1]))
    return B,s

def full_hamiltonian(g,J,x,L): 
    """" Generates the Hamiltonian """    
    sx = sparse.csr_matrix(np.array([[0.,1.],[1.,0.]]))
    sz = sparse.csr_matrix(np.array([[1.,0.],[0.,-1.]]))
       
    d = 2
    sx_list = []
    sz_list = []
       
    for i_bond in range(L): 
        if i_bond==0: 
            X=sx
            Z=sz 
        else: 
            X= sparse.csr_matrix(np.eye(d)) 
            Z= sparse.csr_matrix(np.eye(d))
        for j_site in range(1,L): 
            if j_site==i_bond: 
                X=sparse.kron(X,sx, 'csr')
                Z=sparse.kron(Z,sz, 'csr') 
            else: 
                X=sparse.kron(X,np.eye(d),'csr') 
                Z=sparse.kron(Z,np.eye(d),'csr') 
        sx_list.append(X)
        sz_list.append(Z)
        
    H = sparse.csr_matrix((2**L,2**L))
    for i in range(L-1):
    	H = H + (J*sz_list[i]*sz_list[np.mod(i+1,L)] + x[i]*g/2.*sx_list[i] + x[i+1]*g/2.*sx_list[i+1])
    H = H + x[0]*g/2.*sx_list[0]
    H = H + x[L-1]*g/2.*sx_list[L-1]
    
    return H

def init_ising_H_mpo(g,J,L):
    """ Returns hamiltonian in MPO form"""
    s0 = np.eye(2)
    sx = np.array([[0.,1.],[1.,0.]])
    sy = np.array([[0.,-1j],[1j,0.]])
    sz = np.array([[1.,0.],[0.,-1.]])
    d = 2

    w = np.zeros((3,3,d,d),dtype=np.float)
    w[0,:2] = [s0,sz]
    w[0:,2] = [g*sx, -J*sz, s0]
    return w


def ssd_init_ising_H_mpo(g,J,L,N,t):
    """ Returns hamiltonian in MPO form"""
    s0 = np.eye(2)
    sx = np.array([[0.,1.],[1.,0.]])
    sy = np.array([[0.,-1j],[1j,0.]])
    sz = np.array([[1.,0.],[0.,-1.]])
    d = 2

    w = np.zeros((3,3,d,d),dtype=np.float)
    w[0,:2] = [s0,sz]
    w[0:,2] = [g*sx, -J*sz, s0]
    #print "t",t
    #print "sin",sin(pi*t/N)
    #print "sin^2",sin(pi*(t+1)*1.0/(2*N))**2
    
    return w*sin(pi*(t+1)*1.0/(2*N))**2
    #return w
def init_ising_H_bond(g,J,L):
    """ Returns bond hamiltonian"""
    sx = np.array([[0.,1.],[1.,0.]])
    sy = np.array([[0.,-1j],[1j,0.]])
    sz = np.array([[1.,0.],[0.,-1.]])
    d = 2

    H = -J*np.kron(sz,sz) + g*np.kron(sx,np.eye(2))
    H_bond = np.reshape(H,(d,d,d,d))
    return H_bond
    
def bond_expectation(B,s,O_list):
    " Expectation value for a bond operator "
    E=[]
    L = len(B)
    for i_bond in range(L):
        BB = np.tensordot(B[i_bond],B[np.mod(i_bond+1,L)],axes=(2,1))
        sBB = np.tensordot(np.diag(s[np.mod(i_bond-1,L)]),BB,axes=(1,1))
        C = np.tensordot(sBB,O_list[i_bond],axes=([1,2],[2,3]))        
        sBB=np.conj(sBB)
        E.append(np.squeeze(np.tensordot(sBB,C,axes=([0,3,1,2],[0,1,2,3]))).item()/np.linalg.norm(sBB)**2)
    return E

def site_expectation(B,s,O_list):
    " Expectation value for a site operator "
    E=[]
    L = len(B)
    for isite in range(0,L):
        sB = np.tensordot(np.diag(s[np.mod(isite-1,L)]),B[isite],axes=(1,1))
        C = np.tensordot(sB,O_list[isite],axes=(1,0))
        sB=sB.conj()
        E.append(np.squeeze(np.tensordot(sB,C,axes=([0,1,2],[0,2,1]))).item())
    return(E)

class H_mixed(object):
    def __init__(self,Lp,Rp,M1,M2,dtype=float):
        self.Lp = Lp
        self.Rp = Rp
        self.M1 = M1
        self.M2 = M2
        self.d = M1.shape[3]
        self.chi1 = Lp.shape[0]
        self.chi2 = Rp.shape[0]
        self.shape = np.array([self.d**2*self.chi1*self.chi2,self.d**2*self.chi1*self.chi2])
        self.dtype = dtype
        
    def matvec(self,x):
        x=np.reshape(x,(self.d,self.chi1,self.d,self.chi2)) # i a j b
        x=np.tensordot(self.Lp,x,axes=(0,1))                # ap m i j b
        x=np.tensordot(x,self.M1,axes=([1,2],[0,2]))        # ap j b mp ip
        x=np.tensordot(x,self.M2,axes=([3,1],[0,2]))        # ap b ip m jp
        x=np.tensordot(x,self.Rp,axes=([1,3],[0,2]))        # ap ip jp bp
        x=np.transpose(x,(1,0,2,3))
        x=np.reshape(x,((self.d*self.d)*(self.chi1*self.chi2)))
        if(self.dtype==float):
            return np.real(x)
        else:
            return(x)

def diag(B,s,H,ia,ib,ic,chia,chic):
    """ Diagonalizes the mixed hamiltonian """
    # Get a guess for the ground state based on the old MPS
    d = B[0].shape[0]
    theta0 = np.tensordot(np.diag(s[ia]),np.tensordot(B[ib],B[ic],axes=(2,1)),axes=(1,1))
    theta0 = np.reshape(np.transpose(theta0,(1,0,2,3)),((chia*chic)*(d**2)))

    # Diagonalize Hamiltonian
    e0,v0 = arp.eigsh(H,k=1,which='SA',return_eigenvectors=True,v0=theta0,ncv=20)
    
    return np.reshape(v0.squeeze(),(d*chia,d*chic)),e0

def sweep(B,s,chi,H_mpo,Lp,Rp):
    """ One iDMRG sweep through unit cell"""
    
    d = B[0].shape[0]
    for i_bond in [0,1]:
        ia = np.mod(i_bond-1,2); ib = np.mod(i_bond,2); ic = np.mod(i_bond+1,2)
        chia = B[ib].shape[1]; chic = B[ic].shape[2]

        # Construct theta matrix #
        H = H_mixed(Lp,Rp,H_mpo[i_bond],H_mpo[i_bond])
        theta,e0 = diag(B,s,H,ia,ib,ic,chia,chic)
        
        # Schmidt deomposition #
        X, Y, Z = np.linalg.svd(theta); Z = Z.T

        chib = np.min([np.sum(Y>10.**(-8)), chi])
        X=np.reshape(X[:d*chia,:chib],(d,chia,chib))
        Z=np.transpose(np.reshape(Z[:d*chic,:chib],(d,chic,chib)),(0,2,1))
        
        # Update Environment #
        Lp = np.tensordot(Lp, H_mpo[i_bond], axes=(2,0))
        Lp = np.tensordot(Lp, X, axes=([0,3],[1,0]))
        Lp = np.tensordot(Lp, np.conj(X), axes=([0,2],[1,0]))
        Lp = np.transpose(Lp,(1,2,0))

        Rp = np.tensordot(H_mpo[i_bond], Rp, axes=(1,2))
        Rp = np.tensordot(np.conj(Z),Rp, axes=([0,2],[2,4]))
        Rp = np.tensordot(Z,Rp, axes=([0,2],[2,3]))

        # Obtain the new values for B and s #
        s[ib] = Y[:chib]/np.sqrt(sum(Y[:chib]**2))
        B[ib] = np.transpose(np.tensordot(np.diag(s[ia]**(-1)),X,axes=(1,1)),(1,0,2))
        B[ib] = np.tensordot(B[ib], np.diag(s[ib]),axes=(2,1))

        B[ic] = Z
    return Lp,Rp,e0
	










print("iDMRG")


N = 30
#N_list = [5,10,15,20]
N_list = np.arange(1,N+1)

chi_list = [10]

print("iDMRG")
for chi in chi_list:
    e0_list = []
    e0_devi_list = []
    
        
    ######## Define the simulation parameter ##########################################
    #chi = 500
    #N = 100
    
    L = 4*N
    x = np.array(range(L/2) + range(L/2)[::-1])/(L/2-1.0)
    print("xx",x)
    x = np.ones(L)
   
    
    ######## Define the Spin operators  ################################################
    J = 1
    g = 1.
    
    
    ########## Get the energy for the full model : Only for small N!!! ##################
    #H_full = full_hamiltonian(g,J,x,L)
    #E0_ED,v = arp.eigsh(H_full,k=1,which='SA',return_eigenvectors=True)
    #print L,"E0_ED=",E0_ED[0]
    
    B,s = init_fm_mps(2)
    H_bond = init_ising_H_bond(g,J,2)
    
    sz = np.array([[1.,0.],[0.,-1.]])
    Lp = np.zeros([1,1,3]); Lp[0,0,0] = 1.
    Rp = np.zeros([1,1,3]); Rp[0,0,2] = 1.
    
    
    
    f = lambda k,g : -2*np.sqrt(1+g**2-2*g*np.cos(k))/np.pi/2.
    e0_exact = integrate.quad(f, 0, np.pi, args=(g,))[0]
    for step in range(N):
        H_mpo = [init_ising_H_mpo(x[2*step]*g,J,2),init_ising_H_mpo(x[2*step+1]*g,J,2)]
         
        Lp,Rp,E0_DMRG = sweep(B,s,chi,H_mpo,Lp,Rp)
        
        e0 = np.mean(bond_expectation(B,s,[H_bond,H_bond]))
        print(L,"E0_DMRG=",E0_DMRG[0], "e0=",e0)
        #print "deviation",np.abs(e0-e0_exact)
        e0_devi_list.append(log10(np.abs(e0-e0_exact)))
        e0_list.append(e0)
        #print e0
        #print "log",np.log10(np.abs(e0-e0_exact))
    pl.plot(N_list,e0_devi_list,'-+',label = "idmrg")
    pl.plot()
    #print "N",N,"chi",chi
        
    f = lambda k,g : -2*np.sqrt(1+g**2-2*g*np.cos(k))/np.pi/2.
    e0_exact = integrate.quad(f, 0, np.pi, args=(g,))[0]
    #print "N=",N
    #print "e0_exact=",e0_exact, 'de0=',np.abs(e0-e0_exact)


    
    fileout = "ising_idmrg_chi="+str(chi)+"_N="+str(len(N_list)) +".dat"
    fileout = str(fileout)
    f = open(fileout,'w')
    for i in range(len(N_list)):
        f.write(str(N_list[i])+"   "+str(e0_exact)+"  "+str(e0_list[i])+" "+str(e0_devi_list[i]) +       "\n")  
        f.close
    

#transvers field




"""

print "linear-deform-iDMRG"

for chi in chi_list:
    e0_ssd_list = []
    e0_devi_ssd_list = []
    for N in N_list:
        
        ######## Define the simulation parameter ##########################################
        #chi = 500
        #N = 100

        L = 4*N
        x = np.array(range(L/2) + range(L/2)[::-1])/(L/2-1.0)
        x = np.ones(L)
        x = np.array(range(L/2) + range(L/2)[::-1])/(L/2-1.0)

        

        
        ######## Define the Spin operators  ################################################
        J = 1
        g = 1.


        ########## Get the energy for the full model : Only for small N!!! ##################
        #H_full = full_hamiltonian(g,J,x,L)
        #E0_ED,v = arp.eigsh(H_full,k=1,which='SA',return_eigenvectors=True)
        #print L,"E0_ED=",E0_ED[0]

        B,s = init_fm_mps(2)
        H_bond = init_ising_H_bond(g,J,2)

        sz = np.array([[1.,0.],[0.,-1.]])
        Lp = np.zeros([1,1,3]); Lp[0,0,0] = 1.
        Rp = np.zeros([1,1,3]); Rp[0,0,2] = 1.
    
    
    
        f = lambda k,g : -2*np.sqrt(1+g**2-2*g*np.cos(k))/np.pi/2.
        e0_exact = integrate.quad(f, 0, np.pi, args=(g,))[0]
        for step in range(N):
            
            H_mpo = [init_ising_H_mpo(g*x[2*step],J,2),init_ising_H_mpo(g*x[2*step+1],J,2)]
            
            Lp,Rp,E0_DMRG = sweep(B,s,chi,H_mpo,Lp,Rp)
    
            e0 = np.mean(bond_expectation(B,s,[H_bond,H_bond]))
            print L,"E0_DMRG=",E0_DMRG[0], "e0=",e0
            #print "deviation",np.abs(e0-e0_exact)
    
        f = lambda k,g : -2*np.sqrt(1+g**2-2*g*np.cos(k))/np.pi/2.
        e0_exact = integrate.quad(f, 0, np.pi, args=(g,))[0]
        print "N=",N
        print "e0_exact=",e0_exact, 'de0=',np.abs(e0-e0_exact)
        e0_devi_ssd_list.append(log10(np.abs(e0-e0_exact)))
        e0_ssd_list.append(e0)
        
        print "log",np.log10(np.abs(e0-e0_exact))
    pl.plot(N_list,e0_devi_ssd_list,'-o',label = "linear-deform")
    #pl.plot(N_list,e0_ssd_list,'-o')
    print "e0_ssd",e0_ssd_list[N-1]
    print "e0    ",e0_list[N-1]
    print "N_list",N_list
    
"""
"""
    fileout = "ising_idmrg_field_linear-deform_chi="+str(chi)+"_N="+str(len(N_list)) +".dat"
    fileout = str(fileout)
    f = open(fileout,'w')
    for i in range(len(N_list)):
        f.write(str(N_list[i])+"   "+str(e0_exact)+"  "+str(e0_ssd_list[i])+" "+str(e0_devi_ssd_list[i]) +       "\n")  
        f.close
"""












print "sine-square-deformation-iDMRG"
print "field"


for chi in chi_list:
    e0_ssd_list = []
    e0_devi_ssd_list = []
    for N in N_list:
        
        ######## Define the simulation parameter ##########################################
        #chi = 500
        #N = 100

        L = 4*N
        x = np.array(range(L/2) + range(L/2)[::-1])/(L/2-1.0)
        x = np.ones(L)
        x = np.array(range(L/2) + range(L/2)[::-1])/(L/2-1.0)

        x = []
        for i in range(L):
            x.append(np.sin(pi*(i)/(L-1))**2)
        

        
        ######## Define the Spin operators  ################################################
        J = 1
        g = 1.


        ########## Get the energy for the full model : Only for small N!!! ##################
        #H_full = full_hamiltonian(g,J,x,L)
        #E0_ED,v = arp.eigsh(H_full,k=1,which='SA',return_eigenvectors=True)
        #print L,"E0_ED=",E0_ED[0]

        B,s = init_fm_mps(2)
        H_bond = init_ising_H_bond(g,J,2)

        sz = np.array([[1.,0.],[0.,-1.]])
        Lp = np.zeros([1,1,3]); Lp[0,0,0] = 1.
        Rp = np.zeros([1,1,3]); Rp[0,0,2] = 1.
    
    
    
        f = lambda k,g : -2*np.sqrt(1+g**2-2*g*np.cos(k))/np.pi/2.
        e0_exact = integrate.quad(f, 0, np.pi, args=(g,))[0]
        for step in range(N):
           

            H_mpo = [init_ising_H_mpo(g*x[2*step],J,2),init_ising_H_mpo(g*x[2*step+1],J,2)]
        
    
            Lp,Rp,E0_DMRG = sweep(B,s,chi,H_mpo,Lp,Rp)
    
            e0 = np.mean(bond_expectation(B,s,[H_bond,H_bond]))
            #print L,"E0_DMRG=",E0_DMRG[0], "e0=",e0
            #print "deviation",np.abs(e0-e0_exact)
            
        f = lambda k,g : -2*np.sqrt(1+g**2-2*g*np.cos(k))/np.pi/2.
        e0_exact = integrate.quad(f, 0, np.pi, args=(g,))[0]
        print "N=",N
        print "e0_exact=",e0_exact, 'de0=',np.abs(e0-e0_exact)
        e0_devi_ssd_list.append(log10(np.abs(e0-e0_exact)))
        e0_ssd_list.append(e0)
        
        print "log",np.log10(np.abs(e0-e0_exact))
    pl.plot(N_list,e0_devi_ssd_list,'-*',label = "field-sin^2")
    #pl.plot(N_list,e0_ssd_list,'-o')

    
"""
    fileout = "ising_idmrg_field_ssd_chi="+str(chi)+"_N="+str(len(N_list)) +".dat"
    fileout = str(fileout)
    f = open(fileout,'w')
    for i in range(len(N_list)):
        f.write(str(N_list[i])+"   "+str(e0_exact)+"  "+str(e0_ssd_list[i])+" "+str(e0_devi_ssd_list[i]) +       "\n")  
        f.close
"""





#coupling constant



"""
print "linear-deform-iDMRG"

for chi in chi_list:
    e0_ssd_list = []
    e0_devi_ssd_list = []
    for N in N_list:
        
        ######## Define the simulation parameter ##########################################
        #chi = 500
        #N = 100

        L = 4*N
        x = np.array(range(L/2) + range(L/2)[::-1])/(L/2-1.0)
        x = np.ones(L)
        x = np.array(range(L/2) + range(L/2)[::-1])/(L/2-1.0)



        
        ######## Define the Spin operators  ################################################
        J = 1
        g = 1.


        ########## Get the energy for the full model : Only for small N!!! ##################
        #H_full = full_hamiltonian(g,J,x,L)
        #E0_ED,v = arp.eigsh(H_full,k=1,which='SA',return_eigenvectors=True)
        #print L,"E0_ED=",E0_ED[0]

        B,s = init_fm_mps(2)
        H_bond = init_ising_H_bond(g,J,2)

        sz = np.array([[1.,0.],[0.,-1.]])
        Lp = np.zeros([1,1,3]); Lp[0,0,0] = 1.
        Rp = np.zeros([1,1,3]); Rp[0,0,2] = 1.
    
    
    
        f = lambda k,g : -2*np.sqrt(1+g**2-2*g*np.cos(k))/np.pi/2.
        e0_exact = integrate.quad(f, 0, np.pi, args=(g,))[0]
        for step in range(N):
          
            H_mpo = [init_ising_H_mpo(x[2*step]*g,J,2),init_ising_H_mpo(x[2*step+1]*g,J,2)]
            Lp,Rp,E0_DMRG = sweep(B,s,chi,H_mpo,Lp,Rp)
    
            e0 = np.mean(bond_expectation(B,s,[H_bond,H_bond]))
            print L,"E0_DMRG=",E0_DMRG[0], "e0=",e0
            #print "deviation",np.abs(e0-e0_exact)
    
        f = lambda k,g : -2*np.sqrt(1+g**2-2*g*np.cos(k))/np.pi/2.
        e0_exact = integrate.quad(f, 0, np.pi, args=(g,))[0]
        print "N=",N
        print "e0_exact=",e0_exact, 'de0=',np.abs(e0-e0_exact)
        e0_devi_ssd_list.append(log10(np.abs(e0-e0_exact)))
        e0_ssd_list.append(e0)
        
        print "log",np.log10(np.abs(e0-e0_exact))
    pl.plot(N_list,e0_devi_ssd_list,'-o',label = "linear-deform")
    #pl.plot(N_list,e0_ssd_list,'-o')
    print "e0_ssd",e0_ssd_list[N-1]
    print "e0    ",e0_list[N-1]
    print "N_list",N_list
"""

"""
    fileout = "ising_idmrg_coupling_linear-deform_chi="+str(chi)+"_N="+str(len(N_list)) +".dat"
    fileout = str(fileout)
    f = open(fileout,'w')
    for i in range(len(N_list)):
        f.write(str(N_list[i])+"   "+str(e0_exact)+"  "+str(e0_ssd_list[i])+" "+str(e0_devi_ssd_list[i]) +       "\n")  
    f.close
"""











print "sine-square-deformation-iDMRG"
print "coupling constant"


for chi in chi_list:
    e0_ssd_list = []
    e0_devi_ssd_list = []
    for N in N_list:
        
        ######## Define the simulation parameter ##########################################
        #chi = 500
        #N = 100

        L = 4*N
        #x = np.array(range(L/2) + range(L/2)[::-1])/(L/2-1.0)
        #x = np.ones(L)
        #x = np.array(range(L/2) + range(L/2)[::-1])/(L/2-1.0)

        x = []
        for i in range(L):
            x.append(np.sin(pi*(i)/(L-1))**2)
        

        
        ######## Define the Spin operators  ################################################
        J = 1
        g = 1.


        ########## Get the energy for the full model : Only for small N!!! ##################
        #H_full = full_hamiltonian(g,J,x,L)
        #E0_ED,v = arp.eigsh(H_full,k=1,which='SA',return_eigenvectors=True)
        #print L,"E0_ED=",E0_ED[0]

        B,s = init_fm_mps(2)
        H_bond = init_ising_H_bond(g,J,2)

        sz = np.array([[1.,0.],[0.,-1.]])
        Lp = np.zeros([1,1,3]); Lp[0,0,0] = 1.
        Rp = np.zeros([1,1,3]); Rp[0,0,2] = 1.
    
    
    
        f = lambda k,g : -2*np.sqrt(1+g**2-2*g*np.cos(k))/np.pi/2.
        e0_exact = integrate.quad(f, 0, np.pi, args=(g,))[0]
        for step in range(N):
           
            H_mpo = [init_ising_H_mpo(g,x[2*step]*J,2),init_ising_H_mpo(g,x[2*step+1]*J,2)]
    
            Lp,Rp,E0_DMRG = sweep(B,s,chi,H_mpo,Lp,Rp)
    
            e0 = np.mean(bond_expectation(B,s,[H_bond,H_bond]))
            #print L,"E0_DMRG=",E0_DMRG[0], "e0=",e0
            #print "deviation",np.abs(e0-e0_exact)
            print "step",x[2*step],x[2*step]
        f = lambda k,g : -2*np.sqrt(1+g**2-2*g*np.cos(k))/np.pi/2.
        e0_exact = integrate.quad(f, 0, np.pi, args=(g,))[0]
        print "N=",N
        print "e0_exact=",e0_exact, 'de0=',np.abs(e0-e0_exact)
        e0_devi_ssd_list.append(log10(np.abs(e0-e0_exact)))
        e0_ssd_list.append(e0)
        
        print "log",np.log10(np.abs(e0-e0_exact))
    pl.plot(N_list,e0_devi_ssd_list,'-*',label = "coupling-sin^2")
    #pl.plot(N_list,e0_ssd_list,'-o')
    
    
"""
    fileout = "ising_idmrg_coupling_ssd_chi="+str(chi)+"_N="+str(len(N_list)) +".dat"
    fileout = str(fileout)
    f = open(fileout,'w')
    for i in range(len(N_list)):
        f.write(str(N_list[i])+"   "+str(e0_exact)+"  "+str(e0_ssd_list[i])+" "+str(e0_devi_ssd_list[i]) +       "\n")  
        f.close
"""

"""
print "linear deformation"

for chi in chi_list:
    e0_ssd_list = []
    e0_devi_ssd_list = []
    for N in N_list:
        
        ######## Define the simulation parameter ##########################################
        #chi = 500
        #N = 100

        L = 4*N
        #x = np.array(range(L/2) + range(L/2)[::-1])/(L/2-1.0)
        #x = np.ones(L)
        x = np.array(range(L/2) + range(L/2)[::-1])/(L/2-1.0)
        
        #x = []
        #for i in range(L):
        #    x.append(np.sin(pi*(i)/(L-1))**2)
        

        
        ######## Define the Spin operators  ################################################
        J = 1
        g = 1.


        ########## Get the energy for the full model : Only for small N!!! ##################
        #H_full = full_hamiltonian(g,J,x,L)
        #E0_ED,v = arp.eigsh(H_full,k=1,which='SA',return_eigenvectors=True)
        #print L,"E0_ED=",E0_ED[0]

        B,s = init_fm_mps(2)
        H_bond = init_ising_H_bond(g,J,2)

        sz = np.array([[1.,0.],[0.,-1.]])
        Lp = np.zeros([1,1,3]); Lp[0,0,0] = 1.
        Rp = np.zeros([1,1,3]); Rp[0,0,2] = 1.
    
    
    
        f = lambda k,g : -2*np.sqrt(1+g**2-2*g*np.cos(k))/np.pi/2.
        e0_exact = integrate.quad(f, 0, np.pi, args=(g,))[0]
        for step in range(N):
           
            H_mpo = [init_ising_H_mpo(x[2*step]*g,x[2*step]*J,2),init_ising_H_mpo(x[2*step+1]*g,x[2*step+1]*J,2)]
    
            Lp,Rp,E0_DMRG = sweep(B,s,chi,H_mpo,Lp,Rp)
    
            e0 = np.mean(bond_expectation(B,s,[H_bond,H_bond]))
            #print L,"E0_DMRG=",E0_DMRG[0], "e0=",e0
            #print "deviation",np.abs(e0-e0_exact)
            print "step",x[2*step],x[2*step]
        f = lambda k,g : -2*np.sqrt(1+g**2-2*g*np.cos(k))/np.pi/2.
        e0_exact = integrate.quad(f, 0, np.pi, args=(g,))[0]
        print "N=",N
        print "e0_exact=",e0_exact, 'de0=',np.abs(e0-e0_exact)
        e0_devi_ssd_list.append(log10(np.abs(e0-e0_exact)))
        e0_ssd_list.append(e0)
        
        print "log",np.log10(np.abs(e0-e0_exact))
    pl.plot(N_list,e0_devi_ssd_list,'-*',label = "ssd")
    #pl.plot(N_list,e0_ssd_list,'-o')
    
   
    
    fileout = "ising_idmrg_coupling_field_linear_chi="+str(chi)+"_N="+str(len(N_list)) +".dat"
    fileout = str(fileout)
    f = open(fileout,'w')
    for i in range(len(N_list)):
        f.write(str(N_list[i])+"   "+str(e0_exact)+"  "+str(e0_ssd_list[i])+" "+str(e0_devi_ssd_list[i]) +       "\n")  
        f.close

"""



"""

    
print "sine-square-deformation-iDMRG"



for chi in chi_list:
    e0_ssd_list = []
    e0_devi_ssd_list = []
    for N in N_list:
        
        ######## Define the simulation parameter ##########################################
        #chi = 500
        #N = 100

        L = 4*N
        #x = np.array(range(L/2) + range(L/2)[::-1])/(L/2-1.0)
        #x = np.ones(L)
        #x = np.array(range(L/2) + range(L/2)[::-1])/(L/2-1.0)

        x = []
        for i in range(L):
            x.append(np.sin(pi*(i)/(L-1))**2)
        

        
        ######## Define the Spin operators  ################################################
        J = 1
        g = 1.


        ########## Get the energy for the full model : Only for small N!!! ##################
        #H_full = full_hamiltonian(g,J,x,L)
        #E0_ED,v = arp.eigsh(H_full,k=1,which='SA',return_eigenvectors=True)
        #print L,"E0_ED=",E0_ED[0]

        B,s = init_fm_mps(2)
        H_bond = init_ising_H_bond(g,J,2)

        sz = np.array([[1.,0.],[0.,-1.]])
        Lp = np.zeros([1,1,3]); Lp[0,0,0] = 1.
        Rp = np.zeros([1,1,3]); Rp[0,0,2] = 1.
    
    
    
        f = lambda k,g : -2*np.sqrt(1+g**2-2*g*np.cos(k))/np.pi/2.
        e0_exact = integrate.quad(f, 0, np.pi, args=(g,))[0]
        for step in range(N):
           
            H_mpo = [init_ising_H_mpo(x[2*step]*g,x[2*step]*J,2),init_ising_H_mpo(x[2*step+1]*g,x[2*step+1]*J,2)]
    
            Lp,Rp,E0_DMRG = sweep(B,s,chi,H_mpo,Lp,Rp)
    
            e0 = np.mean(bond_expectation(B,s,[H_bond,H_bond]))
            #print L,"E0_DMRG=",E0_DMRG[0], "e0=",e0
            #print "deviation",np.abs(e0-e0_exact)
            print "step",x[2*step],x[2*step]
        f = lambda k,g : -2*np.sqrt(1+g**2-2*g*np.cos(k))/np.pi/2.
        e0_exact = integrate.quad(f, 0, np.pi, args=(g,))[0]
        print "N=",N
        print "e0_exact=",e0_exact, 'de0=',np.abs(e0-e0_exact)
        e0_devi_ssd_list.append(log10(np.abs(e0-e0_exact)))
        e0_ssd_list.append(e0)
        
        print "log",np.log10(np.abs(e0-e0_exact))
    pl.plot(N_list,e0_devi_ssd_list,'-*',label = "ssd")
    #pl.plot(N_list,e0_ssd_list,'-o')
    
   
    
    fileout = "ising_idmrg_coupling_field_ssd_chi="+str(chi)+"_N="+str(len(N_list)) +".dat"
    fileout = str(fileout)
    f = open(fileout,'w')
    for i in range(len(N_list)):
        f.write(str(N_list[i])+"   "+str(e0_exact)+"  "+str(e0_ssd_list[i])+" "+str(e0_devi_ssd_list[i]) +       "\n")  
        f.close
    

"""

    
# A1B2A1B
pl.legend()
pl.show()
