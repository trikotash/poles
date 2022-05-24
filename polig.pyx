import cython
cimport cython
import numpy as np
cimport numpy as np
from libc.math cimport sin,cos,sqrt,pi,exp
from scipy.integrate import dblquad
from scipy.optimize import minimize_scalar


cdef: 
    double x = 1000
    double [:] glob_tech1,glob_tech2
    
glob_tech1 = np.array([1,0],dtype = 'double')
glob_tech2 = np.array([1,0],dtype = 'double')


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef double [:] mult(double [:] x, double l):
    cdef: 
        int n
        double [:] tech_1
        
    n = x.shape[0]
    tech_1 = np.zeros(n)
    for i in range(n):
        tech_1[i] = l*x[i]
    
    return tech_1
        
cdef  double [:]  suma(double [:] x1, double [:] x2):
    cdef: 
        np.ndarray tech1
        int n
    n = x1.shape[0]
    tech1 = np.zeros(n)
    for i in range(n):
        tech1[i]  = x1[i] + x2[i]
    
    return tech1

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef void eq(double [:] x1, double [:] x2):
    cdef int n
    n = x1.shape[0]
    for i in range(n):
        x1[i] = x2[i]

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef  double integrand_re_tau(double u, double t,double l1,double l2):
    cdef:
        double y1,y2,z1,z2,a
        
    y1 = l1 -sqrt(3)*l2
    y2 = l1 +l2/sqrt(3) 
    z1 = (u**2 +(4-y1)*((1-u)**2))/(((x*(1-u)**2)**(2/3))*((u**2+4*((1-u)**2))**(1/3)))
    z2 = (u**2 +(4-y2)*((1-u)**2))/(((x*(1-u)**2)**(2/3))*((u**2+4*((1-u)**2))**(1/3)))
    a = cos(pi/3+sqrt(3)*z2*t/2.)
    return t*a*exp(-z1*t/2.-t*t*t/3.)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef  double integrand_im_tau(double u,double t,double l1,double l2):
        cdef:
            double y1,y2,z1,z2,a
      
        y1 = l1 -sqrt(3)*l2
        y2 = l1 +l2/sqrt(3)
        z1 = (u**2 +(4-y1)*((1-u)**2))/(((x*(1-u)**2)**(2/3))*((u**2+4*((1-u)**2))**(1/3)))
        z2 = (u**2 +(4-y2)*((1-u)**2))/(((x*(1-u)**2)**(2/3))*((u**2+4*((1-u)**2))**(1/3)))
        a = -sin(pi/3+sqrt(3)*z2*t/2.)
        return t*a*exp(-z1*t/2.-t*t*t/3.)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef double pi_1(double u):
    return ((1-u)**(1/3))*(2*u*u+6*(1-u)*(1-u))/((u*u+4*(1-u)*(1-u))**(13/6))


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef double pi_2(double u):
    return ((1-u)**(1./3.))*(2*u**2+12*(1-u)**2)/((u**2+4*(1-u)**2)**(13./6.))

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef double integrand_pi_1_re(double t, double u,double l1,double l2):
    return pi_1(u)*integrand_re_tau(u,t,l1,l2)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef double integrand_pi_1_im(double t, double u, double l1, double l2):
    return pi_1(u)*integrand_im_tau(u,t,l1,l2)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef double pi_1_re(double l1,double l2):
   return (4*(x**(2/3))/(137*3*pi))*(dblquad(integrand_pi_1_re,0,1,0,np.inf,args=(l1,l2))[0])

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef double pi_1_im(double l1, double l2):
    return (4*(x**(2/3))/(137*3*pi))*(dblquad(integrand_pi_1_im,0,1,0,np.inf,args=(l1,l2))[0])
    
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef double tech_to_min(double l1,double l2):
 return (l1-pi_1_re(l1,l2))*(l1-pi_1_re(l1,l2))+(l2-pi_1_im(l1,l2))*(l2-pi_1_im(l1,l2))

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef double one_dim_tech_to_min(double l):
    return tech_to_min(glob_tech1[0]+l*glob_tech2[0],glob_tech1[1]+l*glob_tech2[1])



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cpdef minimize(chi,bound, n):
    cdef:
        
        double [:] init, tech1, tech2, tech3, u1, u2
        double l_tech
    global x
    u1 = np.array([1,0],dtype = 'double')
    u2 = np.array([0,-1],dtype = 'double')
    l_tech = 0
    x = chi
    init = np.array([0,-500],dtype = 'double')
    tech1 = np.zeros(2)
    tech2 = np.zeros(2)
    tech3 = np.zeros(2)
    for i in range(n):
        global glob_tech1
        global glob_tech2
        eq(glob_tech1,init)
        eq(glob_tech2,u1)
        l_tech = minimize_scalar(one_dim_tech_to_min,bounds = (0,bound),method = 'bounded').x       
        eq(tech1,suma(init, mult(u1,l_tech)))
        eq(glob_tech1, tech1)
        eq(glob_tech2, u2)
        l_tech = minimize_scalar(one_dim_tech_to_min,bounds = (0,bound),method = 'bounded').x
        eq(tech2, suma(tech1, mult(u2,l_tech)))
        eq(u1, u2)
        eq(u2, suma(tech2,mult(init,-1)))
        eq(glob_tech1,init)
        eq(glob_tech2, u2)
        l_tech = minimize_scalar(one_dim_tech_to_min,bounds = (0,bound),method = 'bounded').x
        eq(init,suma(init,mult(u2,l_tech)))
        print([init[0],init[1]])
    return [init[0],init[1]]

print(minimize(1000,500,10))

