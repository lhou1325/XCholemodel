import os
import re
import sys
import glob
import h5py
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import InterpolatedUnivariateSpline
import numpy as np
import math
import time
from scipy.special import erf
import mpmath
from scipy import integrate
from tqdm import trange


def print_progress(iteration, total, offset=0):
    progress = (iteration - offset) / total
    bar_length = 30
    filled_length = int(bar_length * progress)
    bar = '#' * filled_length + '-' * (bar_length - filled_length)
    percent = round(progress * 100, 1)

    if percent % 1 == 0:  # Check if percent is divisible by 10
        print(f'[{bar}] {percent}% ({iteration}/{total})', end='\r')

###################################################
def DFThxcmodel(path):
#   hxcmax = 2.00 * (numpy.amax(hxcss) + numpy.amax(hxcos)) + 0.04
#   hxcmin = 1.10 * (numpy.amin(hxcss) + numpy.amin(hxcos))
#   
#   hxmax = 2.00 * (numpy.amax(hxss) + numpy.amax(hxos)) + 0.01
#   hxmin = 1.10 * (numpy.amin(hxss) + numpy.amin(hxos))
#   
#   hcmax = 2.00 * (numpy.amax(hcss) + numpy.amax(hcos)) + 0.01
#   hcmin = 1.10 * (numpy.amin(hcss) + numpy.amin(hcos))
    print("start")
    start = time.time()
    
    print("Start Reading dens from " + path)
#     print("Start Reading CCSD hole from" + path2)
    
    f = h5py.File(path, 'r')
    
    na = f['rho'][:,0]
    nb = f['rho'][:,1]
    
    ga = f['grd'][:,0:3]
    gb = f['grd'][:,4:7]
    
    w = f['xyz'][:,3]
    f.close()
    Ntot = np.sum(np.dot(w, (na+nb)))
    Na   = np.sum(np.dot(w, (na)))
    Nb   = np.sum(np.dot(w, (nb)))
    print("N_up: ", np.sum(np.dot(w, (na))))
    print("N_down: ", np.sum(np.dot(w, (nb))))   
    print("N_tot: ", Ntot)      
    ###### Exact density
    '''
    rr = np.linspace(0,10,num=1000)
    rr[0] = 1e-5
    na = (((np.sqrt(np.pi/2.)*rr)/(2.*np.exp(rr**2/2.)) + (np.sqrt(np.pi/2.)*rr*(3 + rr**2))/(8.*np.exp(rr**2/2.)) + \
     -    (2*rr + np.exp(rr**2/2.)*np.sqrt(2*np.pi)*(1 + rr**2)*erf(rr/np.sqrt(2)))/(4.*np.exp(rr**2)))/(2.*(np.pi**1.5 + (5*np.pi**2)/8.)*rr))/2
    nb = na
    w = 4*np.pi*rr**2*0.01
    print(na)
    '''
          
#     data = h5py.File(path2,'r')
#     axis = data['xyz'][:,2]
#     hxcss = data['h_xc_bar'][0,:]/2
#     hxcos = data['h_xc_bar'][1,:]/2

#     hxss = data['h_xc_lam'][0,:,0]/2
#     hxos = data['h_xc_lam'][1,:,0]/2

#     hcss = (data['h_xc_bar'][0,:] - data['h_xc_lam'][0,:,0])/2
#     hcos = (data['h_xc_bar'][1,:] - data['h_xc_lam'][1,:,0])/2
    
#     data.close()
    end = time.time()
          
    print("Reading dens time", end - start) 
          
#     delta_u = axis[-1] - axis[-2]
          
#     print("sumrule hxc ", np.sum(4*np.pi*axis**2*(hxcss + hxcos)) * delta_u)
#     print("sumrule hx  ", np.sum(4*np.pi*axis**2*(hxss  + hxos)) * delta_u)
#     print("sumrule hc  ", np.sum(4*np.pi*axis**2*(hcss  + hcos)) * delta_u)


    vabs = lambda a,b: np.sum(np.multiply(a,b),axis=0)
    
    gaa = np.array(list(map(vabs, ga, ga))) # |grad|^2 spin 0
    gbb = np.array(list(map(vabs, gb, gb))) # |grad|^2 spin 1
    gab = np.array(list(map(vabs, ga, gb)))
    gtt = gaa + gbb + 2*gab
    
    #print(gtt) 
          
    npts = 4001
    delta_u = 0.0125
    #u = np.linspace(0,axis[-1],npts)           
    #u[0]=1e-6
    u=1
    print("npts points: ", npts)
    print("u range: 0~", (npts-1) * delta_u)

    kf = (3*math.pi**2*(na+nb))**(1/3)
    #r = np.linspace(0,5,101)
    s = np.sqrt(gtt)/(2*kf*(na+nb))
    s_2a = 2*np.sqrt(gaa)/(2*(3*math.pi**2*(na*2))**(1/3)*(na*2))
    s_2b = 2*np.sqrt(gbb)/(2*(3*math.pi**2*(nb*2))**(1/3)*(nb*2))
    
    #constant for GGA
    a1 = 0.00979681
    a2 = 0.041083
    a3 = 0.187440
    a4 = 0.00120824
    a5 = 0.0347188
    
    A_gga = 1.0161144
    B_gga = -0.37170836
    C_gga = -0.077215461
    D_gga = 0.57786348
    E_gga = -0.051955731
    
    hx_lda = [0]*npts
    hx_pbe = [0]*npts

    hc_lda = [0]*npts
    hc_pbe = [0]*npts
    
    rs   = (3/(4*np.pi*(na+nb)))**(1/3)
    zeta = (na-nb)/(na+nb)
    ks   = (4*kf/np.pi)**(0.5)

    def reduced_density_gradient(gradn, kf, n):
        numerator = abs(gradn)
        denominator = 2*kf*n
        out = numerator/denominator
        return out
    #s1 =  reduced_density_gradient(grad_n, kf, n)
    
    #s = np.linspace(1e-6,6,500)
    def H_function(a,b,c,d,e,s):
        numerator = a*s**2+b*s**4
        denominator = 1+c*s**4+d*s**5+e*s**6
        h_out = numerator/denominator
        return h_out
    
    def F_function(h):
        f_out = 6.475*H_gga+0.4797
        return f_out
    
    def constant_a(a,b,c,d,e,h,f,s):
        first_term = 15*e+(6*c*(1+f*s**2)*(d+h*s**2))
        second_term = 4*b*((d+h*s**2)**2)+8*a*((d+h*s**2)**3)
        first_value = np.sqrt(math.pi)*(first_term+second_term)
        third_term = 1/(16*(d+h*s**2)**(7/2))
        fourth_term = (3*math.pi*np.sqrt(a)/4)*(np.exp(9*h*s**2/(4*a)))*(1-erf((3*s/2)*np.sqrt(h/a)))
    
        a_out = (first_value*third_term)-fourth_term
        return a_out
    
    def constant_b(s,d,h):
        numerator = 15*np.sqrt(math.pi)*s**2
        denominator = 16*(d+h*s**2)**(7/2)
        b_out  = numerator/denominator
        return b_out
    
    def G_function(v1,v2,e):
        numerator = 0.75*math.pi+v1
        denominator = v2*e
        out = -(numerator/denominator)
        return out
    
    def J_gga(s,x):
        a1 = 0.00979681
        a2 = 0.041083
        a3 = 0.187440
        a4 = 0.00120824
        a5 = 0.0347188
    
        A_gga = 1.0161144
        B_gga = -0.37170836
        C_gga = -0.077215461
        D_gga = 0.57786348
        E_gga = -0.051955731
        
        H_gga = H_function(a1,a2,a3,a4,a5,s)
    
        F_gga = 6.475*H_gga+0.4797
    #thiction is to obtain the value of G
        a = constant_a(A_gga,B_gga,C_gga,D_gga,E_gga,H_gga,F_gga,s)
    
        b = constant_b(s,D_gga,H_gga)
    
        G_gga = G_function(a,b,E_gga)
        #out = (-A_gga/x**2)*(1/(1+(4/9)*A_gga*x**2))*np.exp(-s**2*H_gga*x**2)\
        #    +((A_gga/x**2)+B_gga+C_gga*(1+s**2*F_gga)*x**2+E_gga*(1+s**2*G_gga)*x**4)*(np.exp(-D_gga*x**2))*np.exp(-(s**2)*H_gga*x**2)
        out = np.where(
            s == 0,
            J_lda(x),
            (-A_gga / x**2) * (1 / (1 + (4 / 9) * A_gga * x**2)) * np.exp(-s**2 * H_gga * x**2) +
            ((A_gga / x**2) + B_gga + C_gga * (1 + s**2 * F_gga) * x**2 + E_gga * (1 + s**2 * G_gga) * x**4) *
            np.exp(-D_gga * x**2) * np.exp(-s**2 * H_gga * x**2)
        )
        return out
    
    def J_lda(x):
        out = (-A_gga/x**2)*(1/(1+(4/9)*A_gga*x**2))+(A_gga/x**2+B_gga+C_gga*x**2+E_gga*x**4)*np.exp(-D_gga*x**2)
        return out
    
    #progress_bar = tqdm(total=npts)
    #for i in range(npts):
    for i in range(npts):
#         delta_u = axis[-1] - axis[-2]
        if i == 0:
            u = 1e-6
        else:
            u = i * delta_u
    
        #y = u*kf

        # apply spin scaling
        hx_lda[i] = np.sum(np.dot(w,(2*na)**2*J_lda((1+zeta)**(1/3)*kf*u) + (2*nb)**2*J_lda((1-zeta)**(1/3)*kf*u))) /  (2*Ntot)
        hx_pbe[i] = np.sum(np.dot(w,(2*na)**2*J_gga(s_2a, (1+zeta)**(1/3)*kf*u) + (2*nb)**2*J_gga(s_2b, (1-zeta)**(1/3)*kf*u))) /  (2*Ntot)
        #progress_bar.update(1)
        print_progress(i+1, npts-1)

        #progress_bar.set_description(f"Iteration: {i}")

    #progress_bar.close()
    print(" ") 
    u_x = np.linspace(0,(npts-1)*delta_u,npts)
    #hx_lda = np.einsum('i,ai->a', w, (2*na)**2*J_lda(np.outer(u, (1+zeta)**(1/3)*kf)) + (2*nb)**2*J_lda(np.outer(u, (1-zeta)**(1/3)*kf))) /  (2*Ntot)
    print("hx_lda finish")
    #hx_pbe = np.einsum('i,ai->a', w, (2*na)**2*J_gga(s_2a, np.outer(u, (1+zeta)**(1/3)*kf)) + (2*nb)**2*J_gga(s_2b, np.outer(u,(1-zeta)**(1/3)*kf))) /  (2*Ntot)
    print("hx pbe finish")
    #print("size",len(hx_lda),len(hc_lda),len(hx_pbe),len(hc_pbe))
    print("X hole finish")
    print("SUMrule ldax:", np.trapz(4 * np.pi * u_x ** 2 * hx_lda, x=u_x))
    print("SUMrule ggax:", np.trapz(4 * np.pi * u_x ** 2 * hx_pbe, x=u_x))
    print("Ex lda:", np.trapz(2 * np.pi * u_x * hx_lda, x=u_x))
    print("Ex pbe:", np.trapz(2 * np.pi * u_x * hx_pbe, x=u_x))
#########################
    #u = np.linspace(0,axis[-1],npts)
    #print("size",len(hx_lda),len(hx_pbe))

    rs   = (3/(4*np.pi*(na+nb)))**(1/3)
    zeta = (na-nb)/(na+nb)
    ks   = (4*kf/np.pi)**(0.5)
    
    phi  = 0.5*((1+zeta)**(2/3)+(1-zeta)**(2/3))
    t    = np.sqrt(gtt)/(2*ks*phi*(na+nb))
    ####
    a1 = -0.1244
    a2 = 0.027032
    a3 = 0.0024317
    b1 = 0.2199
    b2 = 0.086664
    b3 = 0.012858
    b4 = 0.0020
    
    alpha = 0.193
    beta  = 0.525
    gamma = 0.3393
    delta = 0.9
    epsilon = 0.10161
    #########
    d  = 0.305-0.136*zeta*zeta
    p  = np.pi*kf*d/(4*phi**4)
    
    def f(z):
        f = ((1+z)**(4/3)+(1-z)**(4/3)-2)/(2**(4/3)-2)
        return f
    
    def G(rs_tmp,A,alpha1,beta1,beta2,beta3,beta4,P):
        G = -2*A*(1+alpha1*rs_tmp)*np.log(1+1/(2*A*(beta1*rs**(1/2)+beta2*rs+beta3*rs**(3/2)+beta4*rs**(P+1))))
        return G
    
    ec_0 = G(rs, 0.031091, 0.21370, 7.5957, 3.5876, 1.6382, 0.49294, 1)
    ec_1 = G(rs, 0.015545, 0.20548, 14.1189, 6.1977, 3.3662, 0.62517, 1)
    alpha_c = -G(rs, 0.016887, 0.11125, 10.357, 3.6231, 0.88026, 0.49671, 1)
    
    ec = ec_0 + alpha_c*f(zeta)/1.709921*(1-zeta**4) + (ec_1 - ec_0)*f(zeta)*zeta**4
    
    c1 = -0.0012529\
            + 0.1244*p\
            + 0.61386*(1-zeta**2)/(phi**5*rs**2)*((1+alpha*rs)/(1+beta*rs+alpha*beta*rs**2)-1)
    c2 = 0.0033894-0.054388*p\
            + 0.39270*(1-zeta**2)/(phi**6*rs**1.5)*((1+gamma*rs)/(2+delta*rs+epsilon*rs**2))*((1+alpha*rs)/(1+beta*rs+alpha*beta*rs**2))
    c3 = 0.10847*p**2.5\
            + 1.4604*p**2\
            + 0.51749*p**1.5\
            - 3.5297*c1*p\
            - 1.9030*c2*p**0.5\
            + 1.0685*p**2*np.log(p)\
            + 34.356*phi**(-3)*ec*p**2
    c4 = -0.081596*p**3\
            -1.0810*p**2.5\
            -0.31677*p**2\
            +1.9030*c1*p**1.5\
            +0.76485*c2*p\
            -0.71019*p**2.5*np.log(p)\
            -22.836*phi**(-3)*ec*p**2.5

    print("LDA C hole starts")
    
    #progress_bar = tqdm(total=npts)

    for i in range(npts):
#         delta_u = axis[-1] - axis[-2]
        if i == 0:
            u = 1e-10
        else:
            u = i * delta_u

    ########### GGA  C hole    ############
        v    = phi*ks*u
        #print(i)
        R    = u
    #print((v**2).size)
    
        f1 = (a1 + a2*v + a3*v**2) / (1 + b1*v + b2*v**2 + b3*v**3 + b4*v**4)
        f2 = (-a1 - (a2-a1*b1)*v + c1*v**2 + c2*v**3 + c3*v**4 + c4*v**5) * np.exp(- d*(kf*R/phi)**2)
        #print(f1.size)
        #f1 = 2*f1bar+0.5*v*((a2+2*a3*v)(1+b1*v+b2*v**2+b3*v**3+b4*v**4)-(a1+a2*v+a3*v**2)(b1+2*b2*v+3*b3*v**2+4*b4*v*3))/(1+b1*v+b2*v**2+b3*v**3+b4*v**4)**2
    
        Ac = 1/(4*np.pi*v**2)*(f1+f2)
    
        #print(Ac)
    
       #def E1(x):
       #    return x*np.exp(x)*mpmath.gammainc(0, x)
    
       ##np.array(map(E1, p))
    
       #beta = 2*p**2/(3*np.pi**2)*(1-np.array([E1(12*v) for v in p])) 
       #BcLM = (18*np.pi**3*(1+v**2/12)**2)**(-1)
       #Bc = BcLM*(1-np.exp(-p*v**2)) + beta*v**2*np.exp(-p*v**2)
    
        nc_lda = phi**5*ks**2*Ac
    
       # F = 4*np.pi*v**2*(Ac + t**2*Bc)
    
        hc_lda[i] = np.sum(np.dot(w,nc_lda*(na+nb)))
        
        print_progress(i + 1, npts-1)
        
        #progress_bar.update(1)
        #progress_bar.set_description(f"Iteration: {i}")
    
    #print("size",len(hx_lda),len(hc_lda),len(hx_pbe),len(hc_pbe))
    #progress_bar.close()
    print(" ")
    print("size",len(hx_lda),len(hc_lda),len(hx_pbe),len(hc_pbe)) 
 
    u = np.zeros(npts)
    for i in range(npts):
        if i == 0:
            u[i] = 1e-10
        else:
            u[i] = i * delta_u

    u = np.transpose(u)
    
    ########### GGA  C hole  ############
    rs   = (3/(4*np.pi*(na+nb)))**(1/3)
    zeta = (na-nb)/(na+nb)
    ks   = (4*kf/np.pi)**(0.5)
    
    phi  = 0.5*((1+zeta)**(2/3)+(1-zeta)**(2/3))
    t    = np.sqrt(gtt)/(2*ks*phi*(na+nb))
    
    v    = phi*np.outer(u,ks)
    
    #print(v[:,j].max)
    R    = u
    #print((v[:,j]**2).size)
    
    ########
    a1 = -0.1244
    a2 = 0.027032
    a3 = 0.0024317
    b1 = 0.2199
    b2 = 0.086664
    b3 = 0.012858
    b4 = 0.0020
    
    alpha = 0.193
    beta  = 0.525
    gamma = 0.3393
    delta = 0.9
    epsilon = 0.10161
    kapa =(4/(3*np.pi))*(9*np.pi/4)**(1/3)
    
    #########
    d  = 0.305-0.136*zeta*zeta
    p  = np.pi*kf*d/(4*phi**4)
    
    
    
    def f(z):
        f = ((1+z)**(4/3)+(1-z)**(4/3)-2)/(2**(4/3)-2)
        return f
    
    def G(rs_tmp,A,alpha1,beta1,beta2,beta3,beta4,P):
        G = -2*A*(1+alpha1*rs_tmp)*np.log(1+1/(2*A*(beta1*rs**(1/2)+beta2*rs+beta3*rs**(3/2)+beta4*rs**(P+1))))
        return G
    
    def ldaJ(y):
        A = 0.59
        B = -0.54354
        C = 0.027678
        D = 0.18843
        
        return -A/y**2*1/(1+4/9*A*y**2)+((A/y**2)+B+C*y**2)*np.exp(-D*y**2)
    
    def findvc(x):
        vc = 0
        Fint = integrate.cumtrapz(v[:,j]**2*x, v[:,j], initial=0)
        #print("Fintsize",Fint.size)
        for i in range(v[:,j].size-3,-1,-1):
            #print(Fint[i]*Fint[i-1])
            if (Fint[i]*Fint[i-1]<0):
                vc = v[i,j]
                #vc = v[i+2,j]
                #print("vc=",vc)
                break
        #print(vc)
        return vc
    ''' 
    def findvc(x):
        vc = v[-1,j]
        Fint = integrate.cumtrapz(v[:,j]**2*x, v[:,j], initial=0)
        #print(Fint.size)
        for i in range(v[:,j].size-1,-1,-1):
            #print(Fint[i]*Fint[i-1])
            if (Fint[i]*Fint[i-1]<=0) & (Fint[i]>=0):
                
                vc = v[i,j]
                #print(vc)
                break
            elif Fint[i]<=0:
                
                vc = v[i,j]
                #print(vc)
                break
        return vc   
    '''
    ec_0 = G(rs, 0.031091, 0.21370, 7.5957, 3.5876, 1.6382, 0.49294, 1)
    ec_1 = G(rs, 0.015545, 0.20548, 14.1189, 6.1977, 3.3662, 0.62517, 1)
    alpha_c = -G(rs, 0.016887, 0.11125, 10.357, 3.6231, 0.88026, 0.49671, 1)
    
    ec = ec_0 + alpha_c*f(zeta)/1.709921*(1-zeta**4) + (ec_1 - ec_0)*f(zeta)*zeta**4
    
    c1 = -0.0012529\
            + 0.1244*p\
            + 0.61386*(1-zeta**2)/(phi**5*rs**2)*((1+alpha*rs)/(1+beta*rs+alpha*beta*rs**2)-1)
    c2 = 0.0033894-0.054388*p\
            + 0.39270*(1-zeta**2)/(phi**6*rs**1.5)*((1+gamma*rs)/(2+delta*rs+epsilon*rs**2))*((1+alpha*rs)/(1+beta*rs+alpha*beta*rs**2))
    c3 = 0.10847*p**2.5\
            + 1.4604*p**2\
            + 0.51749*p**1.5\
            - 3.5297*c1*p\
            - 1.9030*c2*p**0.5\
            + 1.0685*p**2*np.log(p)\
            + 34.356*phi**(-3)*ec*p**2
    c4 = -0.081596*p**3\
            -1.0810*p**2.5\
            -0.31677*p**2\
            +1.9030*c1*p**1.5\
            +0.76485*c2*p\
            -0.71019*p**2.5*np.log(p)\
            -22.836*phi**(-3)*ec*p**2.5
    
    nc_gga = np.zeros((na.size,npts))
    nc_lsd = np.zeros((na.size,npts))
    print(" ")
    print("size",len(hx_lda),len(hc_lda),len(hx_pbe),len(hc_pbe))
  
    #progress_bar = tqdm(total=na.size) 
    #range(na.size)[4000, 10000, 30000 ,54577]
    for j in range(na.size):
        f1 = (a1 + a2*v[:,j] + a3*v[:,j]**2) / (1 + b1*v[:,j] + b2*v[:,j]**2 + b3*v[:,j]**3 + b4*v[:,j]**4)
        f2 = (-a1 - (a2-a1*b1)*v[:,j] + c1[j]*v[:,j]**2 + c2[j]*v[:,j]**3 + c3[j]*v[:,j]**4 + c4[j]*v[:,j]**5)\
        * np.exp(- d[j]*(kf[j]*R/phi[j])**2)
        #print(f1.size)
    #f1 = 2*f1bar+0.5*v[:,j]*((a2+2*a3*v[:,j])(1+b1*v[:,j]+b2*v[:,j]**2+b3*v[:,j]**3+b4*v[:,j]**4)-(a1+a2*v[:,j]+a3*v[:,j]**2)(b1+2*b2*v[:,j]+3*b3*v[:,j]**2+4*b4*v[:,j]*3))/(1+b1*v[:,j]+b2*v[:,j]**2+b3*v[:,j]**3+b4*v[:,j]**4)**2
    
    
        Ac = 1/(4*np.pi*v[:,j]**2)*(f1+f2)
    
        #print("rs = ", rs[j], "zeta = ", zeta[j], "t = ", t[j])
    
        def E1(x):
            return x*np.exp(x)*mpmath.gammainc(0, x)
    
        #np.array(map(E1, p)) np.array([E1(12*p[j]) for v[:,j] in p]
    
        beta_rs = 2*p[j]**2/(3*np.pi**3)*(1-E1(12*p[j]))
        BcLM = (18*np.pi**3*(1+v[:,j]**2/12)**2)**(-1)
        Bc = BcLM*(1-np.exp(-p[j]*v[:,j]**2))\
        + beta_rs*v[:,j]**2*np.exp(-p[j]*v[:,j]**2)
    
        nc_lsd[j] = phi[j]**5*ks[j]**2*Ac

        #nc_lsd2 = kapa**(-1)*phi[j]**3*rs[j]*(f1+f2)/(kf[j]*R)**2
        nc_gea = phi[j]**5*ks[j]**2*(Ac + t[j]**2*Bc)
        
        #print('Ac', Ac, 'nc_gea', nc_gea.size, 'nc_lsd', nc_lsd, 'v', v.shape)
        
        #F_gea_int = integrate.cumtrapz(4*np.pi*v[:,j]**2*nc_gea/(phi[j]*ks[j])**2, v[:,j], initial=0)
        vc = findvc(Ac + t[j]**2*Bc)
        #print(vc)
        nc_gga[j] = nc_gea*np.heaviside(vc-v[:,j], 0)
        #print(j,w.size)
        print_progress(j + 1, (na.size // 10)*10)
        #progress_bar.update(1)
        #print(v.shape, nc_gga.shape, nc_gea.size, na.shape, w.shape)
    
        #F = 4*np.pi*v[:,j]**2*(Ac + t[j]**2*Bc)
    
        
        #vc[j] = find_max_int1(F_int)
    
    #nn = nc_gga.T*(na+nb)
    #print(nn.shape)
    #progress_bar.close()
    print(" ")
    hc_lda = np.dot(nc_lsd.T*(na+nb), w)/Ntot
    hc_pbe = np.dot(nc_gga.T*(na+nb), w)/Ntot
    
    #u = np.linspace(0,5,npts)
    u = u.T
    print("size",len(hx_lda),len(hc_lda),len(hx_pbe),len(hc_pbe))
    
    hx_lda = np.array(hx_lda)
    hc_lda = np.array(hc_lda)
    hxc_lda= hx_lda + hc_lda


    hx_pbe = np.array(hx_pbe)
    hc_pbe = np.array(hc_pbe)
    hxc_pbe= hx_pbe + hc_pbe


    cx_lda  = (hx_lda[1]  - hx_lda[0])  / (u[1]-u[0])
    cc_lda  = (hc_lda[1]  - hc_lda[0])  / (u[1]-u[0])
    cxc_lda = (hxc_lda[1] - hxc_lda[0]) / (u[1]-u[0])
    
    cx_pbe  = (hx_pbe[1]  - hx_pbe[0])  / (u[1]-u[0])
    cc_pbe  = (hc_pbe[1]  - hc_pbe[0])  / (u[1]-u[0])
    cxc_pbe = (hxc_pbe[1] - hxc_pbe[0]) / (u[1]-u[0])
    

    print("hc_pbe",hc_pbe)
    print("dens_a",np.dot(na,w))
    print("dens_b",np.dot(nb,w))
    print("dens_+",np.dot(na+nb,w))

    print("sumrule ldax", np.sum(4*np.pi*u**2*hx_lda) * delta_u)
    print("sumrule ggax", np.sum(4*np.pi*u**2*hx_pbe) * delta_u)
    print("sumrule ldac", np.sum(4*np.pi*u**2*hc_lda) * delta_u)
    print("sumrule ggac", np.sum(4*np.pi*u**2*hc_pbe) * delta_u)
    
    
        # Get the filename from the path
    filename = os.path.basename(path)

    # Split the filename into name and extension
    name, extension = os.path.splitext(filename)

    # Split the name by '/'
    name_parts = name.split('/')

    # Get the part after the last '/'
    after = name_parts[-1]

    ofile = open(f'XChole_energy_{after}.txt','w')


    ex_lda  = integrate.cumtrapz(4 * np.pi * hx_lda * u, u, initial=0)
    ec_lda  = integrate.cumtrapz(4 * np.pi * hc_lda * u, u, initial=0)
    exc_lda = integrate.cumtrapz(4 * np.pi * (hx_lda + hc_lda) * u, u, initial=0)

    ex_pbe  = integrate.cumtrapz(4 * np.pi * hx_pbe * u, u, initial=0)
    ec_pbe  = integrate.cumtrapz(4 * np.pi * hc_pbe * u, u, initial=0)
    exc_pbe = integrate.cumtrapz(4 * np.pi * (hx_pbe + hc_pbe) * u, u, initial=0)
    
    print("E_LDA", ex_lda[-1], ec_lda[-1], exc_lda[-1])
    print("E_PBE", ex_pbe[-1], ec_pbe[-1], exc_pbe[-1])


    ofile.write('\n')
    ofile.write('LDA:Ex  = {0: 16.12f}  \n'.format(ex_lda[-1]))
    ofile.write('LDA:Ec  = {0: 16.12f}  \n'.format(ec_lda[-1]))
    ofile.write('LDA:Exc = {0: 16.12f}  \n'.format(exc_lda[-1]))
    ofile.write('\n')
    ofile.write('PBE:Ex  = {0: 16.12f}  \n'.format(ex_pbe[-1]))
    ofile.write('PBE:Ec  = {0: 16.12f}  \n'.format(ec_pbe[-1]))
    ofile.write('PBE:Exc = {0: 16.12f}  \n'.format(exc_pbe[-1]))
    ofile.write('\n')
    ofile.write('LDA:Sumx  = {0: 16.12f}  \n'.format(np.sum(4*np.pi*u**2*hx_lda) * delta_u))
    ofile.write('LDA:Sumc  = {0: 16.12f}  \n'.format(np.sum(4*np.pi*u**2*hc_lda) * delta_u))
    ofile.write('\n')
    ofile.write('PBE:Sumx  = {0: 16.12f}  \n'.format(np.sum(4*np.pi*u**2*hx_pbe) * delta_u))
    ofile.write('PBE:Sumc  = {0: 16.12f}  \n'.format(np.sum(4*np.pi*u**2*hc_pbe) * delta_u))
    ofile.write('\n')      
    ofile.write('ontop x  = {0: 16.12f}  \n'.format(hx_lda[0]))
    ofile.write('ontop c  = {0: 16.12f}  \n'.format(hc_lda[0]))
    ofile.write('ontop xc = {0: 16.12f}  \n'.format(hxc_lda[0]))
    ofile.write('\n')
    ofile.write('cusp x  = {0: 16.12f}  \n'.format(cx_pbe))
    ofile.write('cusp c  = {0: 16.12f}  \n'.format(cc_pbe))
    ofile.write('cusp xc = {0: 16.12f}  \n'.format(cxc_pbe))
    ofile.close()

    odat = h5py.File(f"XCholemodel_{after}.plot","w")
    
    odat.create_dataset('u_axis'   ,(npts,)   ,dtype='f8',compression='gzip')
    odat.create_dataset('LDA_X'    ,(npts,)   ,dtype='f8',compression='gzip')
    odat.create_dataset('LDA_C'    ,(npts,)   ,dtype='f8',compression='gzip')
    odat.create_dataset('LDA_XC'   ,(npts,)   ,dtype='f8',compression='gzip')
    odat.create_dataset('PBE_X'    ,(npts,)   ,dtype='f8',compression='gzip')
    odat.create_dataset('PBE_C'    ,(npts,)   ,dtype='f8',compression='gzip')
    odat.create_dataset('PBE_XC'   ,(npts,)   ,dtype='f8',compression='gzip')
    
    odat.create_dataset('LDA_EX'    ,(npts,)   ,dtype='f8',compression='gzip')
    odat.create_dataset('LDA_EC'    ,(npts,)   ,dtype='f8',compression='gzip')
    odat.create_dataset('LDA_EXC'   ,(npts,)   ,dtype='f8',compression='gzip')
    odat.create_dataset('PBE_EX'    ,(npts,)   ,dtype='f8',compression='gzip')
    odat.create_dataset('PBE_EC'    ,(npts,)   ,dtype='f8',compression='gzip')
    odat.create_dataset('PBE_EXC'   ,(npts,)   ,dtype='f8',compression='gzip')

    odat['u_axis'][:]  = u[:]
    odat['LDA_X'][:]   = hx_lda[:]
    odat['LDA_C'][:]   = hc_lda[:]
    odat['LDA_XC'][:]  = hx_lda[:] + hc_lda[:]
    odat['PBE_X'][:]  = hx_pbe[:]           
    odat['PBE_C'][:]   = hc_pbe[:]
    odat['PBE_XC'][:]  = hx_pbe[:] + hc_pbe[:]
    
    odat['LDA_EX'][:]  = ex_lda[:]   
    odat['LDA_EC'][:]  = ec_lda[:]   
    odat['LDA_EXC'][:] = exc_lda[:]   
    odat['PBE_EX'][:]  = ex_pbe[:]
    odat['PBE_EC'][:]  = ec_pbe[:]   
    odat['PBE_EXC'][:] = exc_pbe[:]

    odat.close()
                   
#     ax1 = plt.subplot(321)
#     #plt.plot(axis,hxcss,label=r"$\sf \langle \bar{h}^{\sf \sigma\sigma }_{\sf xc} (u) \rangle$",lw=5.0,color='#0000FF')
#     #plt.plot(axis,hxcos,label=r"$\sf \langle \bar{h}^{\sf \sigma\sigma'}_{\sf xc} (u) \rangle$",lw=5.0,color='#FF0000')
#     plt.plot(u,hx_lda+hc_lda,label=r"$\sf \langle LDA_{\sf xc} (u) \rangle$",lw=5.0,color='#FFD700')
#     plt.plot(u,hx_pbe+hc_pbe,label=r"$\sf \langle PBE_{\sf xc} (u) \rangle$",lw=5.0,color='#00FF00')
#     plt.plot(axis,hxcss+hxcos,label=r"$\sf \langle \bar{h}_{\sf xc} (u) \rangle$",lw=5.0,ls='--',color='k')
#     #plt.plot(axis,hxcss+hxcos,label=r"$\sf \langle \bar{h}_{\sf xc} (u) \rangle$",lw=5.0,ls='--',color='k')
#     ax1.spines['right'].set_visible(False)
#     ax1.spines['bottom'].set_visible(False)
#     ax1.spines['top'].set_position(('data',0))
#     ax1.xaxis.set_ticks_position('top')
#     ax1.yaxis.set_ticks_position('left')
#     ax1.tick_params(direction='out',labelsize=16)
#     ax1.set_xlim(0,5)
#     #ax1.set_ylim(hxcmin,hxcmax)
#     ax1.set_xlabel('u / a.u.',fontsize=18)
#     ax1.set_ylabel(r"$\sf \langle \bar{h}_{\sf xc} (u) \rangle$",fontsize=24)
#     ax1.legend(loc='lower right',ncol=1,fontsize=24,frameon=False)
#     ax1.set_title("Li uccsd dens", fontsize=40)

#     ax2 = plt.subplot(323)
#     #plt.plot(axis,hxss,label=r"$\sf \langle \bar{h}^{\sf \sigma\sigma }_{\sf x} (u) \rangle$",lw=5.0,color='#0000FF')
#     #plt.plot(axis,hxos,label=r"$\sf \langle \bar{h}^{\sf \sigma\sigma'}_{\sf x} (u) \rangle$",lw=5.0,color='#FF0000')
#     plt.plot(u,hx_lda,label=r"$\sf \langle LDA_{\sf x} (u) \rangle$",lw=5.0,color='#FFD700')
#     plt.plot(u,hx_pbe,label=r"$\sf \langle PBE_{\sf x} (u) \rangle$",lw=5.0,color='#00FF00')
#     plt.plot(axis,hxss+hxos,label=r"$\sf \langle \bar{h}_{\sf x} (u) \rangle$",lw=5.0,ls='--',color='k')
#     #plt.plot(axis,hxss+hxos,label=r"$\sf \langle \bar{h}_{\sf x} (u) \rangle$",lw=5.0,ls='--',color='k')
#     ax2.spines['right'].set_visible(False)
#     ax2.spines['bottom'].set_visible(False)
#     ax2.spines['top'].set_position(('data',0))
#     ax2.xaxis.set_ticks_position('top')
#     ax2.yaxis.set_ticks_position('left')
#     ax2.tick_params(direction='out',labelsize=16)
#     ax2.set_xlim(0,5)
#     #ax2.set_ylim(hxmin,hxmax)
#     ax2.set_xlabel('u / a.u.',fontsize=18)
#     ax2.set_ylabel(r"$\sf \langle \bar{h}_{\sf x} (u) \rangle$",fontsize=24)
#     ax2.legend(loc='lower right',ncol=1,fontsize=24,frameon=False)

#     ax3 = plt.subplot(325)
#     #plt.plot(axis,hcss,label=r"$\sf \langle \bar{h}^{\sf \sigma\sigma }_{\sf c} (u) \rangle$",lw=5.0,color='#0000FF')
#     #plt.plot(axis,hcos,label=r"$\sf \langle \bar{h}^{\sf \sigma\sigma'}_{\sf c} (u) \rangle$",lw=5.0,color='#FF0000')
#     plt.plot(u,hc_lda,label=r"$\sf \langle LDA_{\sf c} (u) \rangle$",lw=5.0,color='#FFD700')
#     plt.plot(u,hc_pbe,label=r"$\sf \langle PBE_{\sf c} (u) \rangle$",lw=5.0,color='#00FF00')
#     plt.plot(axis,hcss+hcos,label=r"$\sf \langle \bar{h}_{\sf c} (u) \rangle$",lw=5.0,ls='--',color='k')
#     #plt.plot(axis,hcss+hcos,label=r"$\sf \langle \bar{h}_{\sf c} (u) \rangle$",lw=5.0,ls='--',color='k')
#     ax3.spines['right'].set_visible(False)
#     ax3.spines['bottom'].set_visible(False)
#     ax3.spines['top'].set_position(('data',0))
#     ax3.xaxis.set_ticks_position('top')
#     ax3.yaxis.set_ticks_position('left')
#     ax3.tick_params(direction='out',labelsize=16)
#     ax3.set_xlim(0,5)
#     #ax3.set_ylim(hcmin,hcmax)
#     ax3.set_xlabel('u / a.u.',fontsize=18)
#     ax3.set_ylabel(r"$\sf \langle \bar{h}_{\sf c} (u) \rangle$",fontsize=24)
#     ax3.legend(loc='lower right',ncol=1,fontsize=24,frameon=False)

#     ax1 = plt.subplot(322)
#     #plt.plot(axis,hxcss,label=r"$\sf \langle \bar{h}^{\sf \sigma\sigma }_{\sf xc} (u) \rangle$",lw=5.0,color='#0000FF')
#     #plt.plot(axis,hxcos,label=r"$\sf \langle \bar{h}^{\sf \sigma\sigma'}_{\sf xc} (u) \rangle$",lw=5.0,color='#FF0000')
#     plt.plot(u,u*(hx_lda+hc_lda),label=r"$\sf \langle LDA_{\sf xc} (u) \rangle$",lw=5.0,color='#FFD700')
#     plt.plot(u,u*(hx_pbe+hc_pbe),label=r"$\sf \langle PBE_{\sf xc} (u) \rangle$",lw=5.0,color='#00FF00')
#     plt.plot(axis,axis*(hxcss+hxcos),label=r"$\sf \langle \bar{h}_{\sf xc} (u) \rangle$",lw=5.0,ls='--',color='k')
#     ax1.spines['right'].set_visible(False)
#     ax1.spines['bottom'].set_visible(False)
#     ax1.spines['top'].set_position(('data',0))
#     ax1.xaxis.set_ticks_position('top')
#     ax1.yaxis.set_ticks_position('left')
#     ax1.tick_params(direction='out',labelsize=16)
#     ax1.set_xlim(0,5)
#     #ax1.set_ylim(hxcmin,hxcmax)
#     ax1.set_xlabel('u / a.u.',fontsize=18)
#     ax1.set_ylabel(r"$\sf u\langle \bar{h}_{\sf xc} (u) \rangle$",fontsize=24)
#     ax1.legend(loc='lower right',ncol=1,fontsize=24,frameon=False)
#     #ax1.set_title("R = 1.3, Q = 0.0", fontsize=40)

#     ax2 = plt.subplot(324)
#     #plt.plot(axis,hxss,label=r"$\sf \langle \bar{h}^{\sf \sigma\sigma }_{\sf x} (u) \rangle$",lw=5.0,color='#0000FF')
#     #plt.plot(axis,hxos,label=r"$\sf \langle \bar{h}^{\sf \sigma\sigma'}_{\sf x} (u) \rangle$",lw=5.0,color='#FF0000')
#     plt.plot(u,u*hx_lda,label=r"$\sf \langle LDA_{\sf x} (u) \rangle$",lw=5.0,color='#FFD700')
#     plt.plot(u,u*hx_pbe,label=r"$\sf \langle PBE_{\sf x} (u) \rangle$",lw=5.0,color='#00FF00')
#     plt.plot(axis,axis*(hxss+hxos),label=r"$\sf \langle \bar{h}_{\sf x} (u) \rangle$",lw=5.0,ls='--',color='k')
#     ax2.spines['right'].set_visible(False)
#     ax2.spines['bottom'].set_visible(False)
#     ax2.spines['top'].set_position(('data',0))
#     ax2.xaxis.set_ticks_position('top')
#     ax2.yaxis.set_ticks_position('left')
#     ax2.tick_params(direction='out',labelsize=16)
#     ax2.set_xlim(0,5)
#     #ax2.set_ylim(hxmin,hxmax)
#     ax2.set_xlabel('u / a.u.',fontsize=18)
#     ax2.set_ylabel(r"$\sf u\langle \bar{h}_{\sf x} (u) \rangle$",fontsize=24)
#     ax2.legend(loc='lower right',ncol=1,fontsize=24,frameon=False)

#     ax3 = plt.subplot(326)
#     #plt.plot(axis,hcss,label=r"$\sf \langle \bar{h}^{\sf \sigma\sigma }_{\sf c} (u) \rangle$",lw=5.0,color='#0000FF')
#     #plt.plot(axis,hcos,label=r"$\sf \langle \bar{h}^{\sf \sigma\sigma'}_{\sf c} (u) \rangle$",lw=5.0,color='#FF0000')
#     plt.plot(u,u*hc_lda,label=r"$\sf \langle LDA_{\sf c} (u) \rangle$",lw=5.0,color='#FFD700')
#     plt.plot(u,u*hc_pbe,label=r"$\sf \langle PBE_{\sf c} (u) \rangle$",lw=5.0,color='#00FF00')
#     plt.plot(axis,axis*(hcss+hcos),label=r"$\sf \langle \bar{h}_{\sf c} (u) \rangle$",lw=5.0,ls='--',color='k')
#     ax3.spines['right'].set_visible(False)
#     ax3.spines['bottom'].set_visible(False)
#     ax3.spines['top'].set_position(('data',0))
#     ax3.xaxis.set_ticks_position('top')
#     ax3.yaxis.set_ticks_position('left')
#     ax3.tick_params(direction='out',labelsize=16)
#     ax3.set_xlim(0,5)
#     #ax3.set_ylim(hcmin,hcmax)
#     ax3.set_xlabel('u / a.u.',fontsize=18)
#     ax3.set_ylabel(r"$\sf u\langle \bar{h}_{\sf c} (u) \rangle$",fontsize=24)
#     ax3.legend(loc='lower right',ncol=1,fontsize=24,frameon=False)

#     plt.gcf().set_size_inches(20,30)
#     plt.tight_layout()
#     plt.savefig('DFT_xchole_ccsd.pdf',transparent=False)
#     plt.close()

#    return 

def main():
    #Path = '/home/lhou/test/xchole_example/H2/200_800/work_ave/quest_uhf/1.000_7.279.plot' 
    #Path = '/home/lhou/test/xchole_example/H2/200_800/work_ave/dens/7.3_1.0/dens_ccsd_H2.plot'
    #Path = '/home/lhou/test/xchole_example/H2/Hookes/He/dens_ccsd_He.plot'
    if len(sys.argv) >= 2:
        Path = sys.argv[1]
#         Path2 = sys.argv[2]
        print("Variable 1:", Path)
#         print("Variable 2:", Path2)
    else:
        print("Please provide one variables as command-line arguments.")
    #Path = '/home/lhou/test/xchole_example/H2/200_800/work_ave/work_dft/7.3_1.0/ccsd_H2.plot'
    #Path = '/home/lhou/test/xchole_example/H2/200_800/work_ave/dens/1.3_0.0/dens_ccsd_H2.plot'
    #Path2= '/home/lhou/test/xchole_example/H2/200_800/work_ave/work_35/7.3_1.0/liebplot_ccsd_H2_hxc_int.plot'
    DFThxcmodel(Path)

if __name__ == "__main__":
    main()
