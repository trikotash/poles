import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from numba import jit
import math
from scipy.optimize import minimize,Bounds,minimize_scalar
from techip import tech
from time import time


glob_tech1 = np.array([1,0],dtype = 'double')
glob_tech2 = np.array([0,-1],dtype = 'double')

@jit(nopython=True)
def integrand_re_tau(l1,l2,u,t):
    x = 1000
    y1 = l1 -math.sqrt(3)*l2
    y2 = l1 +l2/math.sqrt(3)
    p11 = (x*(1-u)**2)**(2/3)
    p21 = u**2 +(4-y1)*((1-u)**2)
    p31 = (u**2+4*((1-u)**2))**(1/3)
    z1 = p21/(p11*p31)
    p12 = (x*(1-u)**2)**(2/3)
    p22 = u**2 +(4-y2)*((1-u)**2)
    p32 = (u**2+4*((1-u)**2))**(1/3)
    z2 = p22/(p12*p32)
    a = math.cos(math.pi/3+math.sqrt(3)*z2*t/2.)
    f =t*a*math.exp(-z1*t/2.-t*t*t/3.)
    return f

@jit(nopython=True)
def integrand_im_tau(l1,l2,u,t):
        x = 1000
        y1 = l1 -math.sqrt(3)*l2
        y2 = l1 +l2/math.sqrt(3)
        p11 = (x*(1-u)**2)**(2/3)
        p21 = u**2 +(4-y1)*((1-u)**2)
        p31 = (u**2+4*((1-u)**2))**(1/3)
        z1 = p21/(p11*p31)
        p12 = (x*(1-u)**2)**(2/3)
        p22 = u**2 +(4-y2)*((1-u)**2)
        p32 = (u**2+4*((1-u)**2))**(1/3)
        z2 = p22/(p12*p32)
        a = -math.sin(math.pi/3+math.sqrt(3)*z2*t/2.)
        f =t*a*math.exp(-z1*t/2.-t*t*t/3.)
        return f

@jit(nopython=True)
def pi_1(u):
    a =((1-u)**(1/3))*(2*u*u+6*(1-u)*(1-u))/((u*u+4*(1-u)*(1-u))**(13/6))
    return a

@jit(nopython=True)
def pi_2(u):
    a =((1-u)**(1./3.))*(2*u**2+12*(1-u)**2)/((u**2+4*(1-u)**2)**(13./6.))
    return a

def pi_1_re(l1,l2):
    x = 1000
    kapa = lambda t,u:integrand_re_tau(l1,l2,u,t)
    p = lambda u:pi_1(u)*(integrate.quad(kapa,0,np.inf,args=(u))[0])
    a = (4*(x**(2/3))/(137*3*math.pi))*(integrate.quad(p,0,1)[0])
    return a

def pi_1_im(l1,l2):
    x = 1000
    kapa = lambda t,u:integrand_im_tau(l1,l2,u,t)
    p = lambda u:pi_1(u)*(integrate.quad(kapa,0,np.inf,args=(u))[0])
    a = (4*(x**(2/3))/(3*137*math.pi))*(integrate.quad(p,0,1)[0])
    return a

def f_tech(x):
    return (x[0]-pi_1_re(x[0],x[1]))**2 + (x[1]-pi_1_im(x[0],x[1]))**2



def one_dim_tech_to_min(l):
    return f_tech(glob_tech1[0]+l*glob_tech2[0],glob_tech1[1]+l*glob_tech2[1])

def tech_to_min(l):
    x = 1000
    return (tech(l[0],l[1],x))
    

#im_g = np.loadtxt('first_im.txt',delimiter='\n')
#re_g = np.loadtxt('first_re.txt',delimiter='\n')
#im = np.linspace(-2000.,2000.,20)
#re = np.linspace(-2000.,2000.,20)
#sol = root(f_tech,[pi_1_re(0,0),pi_1_im(0,0)],jac=False,method='hybr')
#f = np.vectorize(f_tech2)
#x,y = np.meshgrid(re,im)
#z = f(x,y)

#for i in range(r.shape[0]):
#    im = r[i]*math.sin(-2.*math.pi/3.)
#    re = r[i]*math.cos(-2.*math.pi/3.)
#sol = root(f_tech,[2911,-360],jac=False)
#print(sol.success)
#print(sol.x)
    #print(i)
#    a1[i] = pi_1_re(re,im)
#    a2[i] = pi_1_im(re,im)

#a = np.loadtxt('roots_right.txt',delimiter='\n')
#b = np.linspace(6,6,17)
#for i in range(l.shape[0]):
#     a1[i] = pi_1_re(l[i],0)
#     print(i)
#    kapa = lambda t,u:integrand_re_tau(a[i],a[i],u,t)
#    p = lambda u:pi_1(u)*(integrate.quad(kapa,0,np.inf,args=(u))[0])
#    a1[i] = (integrate.quad(p,0,1)[0])
#    print(i)
#np.savetxt('l2.txt',l,delimiter='\n')
#np.savetxt('pol_first_right',a1,delimiter='\n')
#fig = plt.figure(figsize=[12,8])
#ax = plt.axes(projection='3d')
#ax.plot_surface(x,y,z)
#plt.contour(x,y,z)
#plt.show()
#fig, ax = plt.subplots()
#plt.plot(r,a1)
#plt.plot(r,a2)
#plt.grid()
#plt.show()430.3974169  -736.79620355
def minimize1(chi,bound, n):
    u1 = np.array([1,0],dtype = 'double')
    u2 = np.array([0,-1],dtype = 'double')
    init =  np.array([0,-500],dtype = 'double')
    l_tech = 0
    x = chi
    tech1 = np.zeros(2)
    tech2 = np.zeros(2)
    tech3 = np.zeros(2)
    for i in range(n):
        global glob_tech1
        global glob_tech2
        glob_tech1 = np.copy(init)
        glob_tech2 = np.copy(u1)
        l_tech = minimize_scalar(one_dim_tech_to_min,bounds = (10,bound),method = 'bounded').x       
        tech1 =np.copy(init + l_tech*u1)
        glob_tech1 = np.copy(tech1)
        glob_tech2 = np.copy(u2)
        l_tech = minimize_scalar(one_dim_tech_to_min,bounds = (10,bound),method = 'bounded').x
        tech2 = np.copy(tech1 + l_tech*u2)
        u1 = np.copy(u2)
        u2 = np.copy(tech2 - init)
        glob_tech1 = np.copy(init)
        glob_tech2 = np.copy(u2)
        l_tech = minimize_scalar(one_dim_tech_to_min,bounds = (10,bound),method = 'bounded').x
        init = init + l_tech*u2
        print(i)
    return init
start = time()
b = Bounds(np.array([0,-800]),np.array([500,-700]))
print(minimize(tech_to_min,np.array([0,-600]),method = 'Powell',bounds = b))
print(time()-start)

#b = Bounds(np.array([0,-800]),np.array([500,-700]))
#print(minimize(f_tech,np.array([0,-600]),method = 'Powell',bounds = b
              # ,options = {'direc': np.array([[1.,0.],[0.,-1.]])}))
#print(time()-start)