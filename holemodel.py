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
from scipy.special import erf, exp1, erfcx
from scipy import integrate
from tqdm import trange

CUMTRAPZ = (
    integrate.cumulative_trapezoid
    if hasattr(integrate, "cumulative_trapezoid")
    else integrate.cumtrapz
)

RHO_FLOOR = 1e-18
PBE_S_REGULARIZATION_START = 8.0
PBE_S_REGULARIZATION_LIMIT = 11.0


def safe_clip_density(rho):
    return np.clip(rho, 0.0, None)


def regularize_reduced_gradient(s, s1=PBE_S_REGULARIZATION_START, s2=PBE_S_REGULARIZATION_LIMIT):
    """Regularize large reduced gradients to avoid the known PBE hole pathologies."""
    s = np.asarray(s, dtype=float)
    out = np.clip(s, 0.0, None).copy()
    mask = out > s1
    if np.any(mask):
        delta = out[mask] - s1
        out[mask] = out[mask] - delta * np.exp(-(s2 - s1) / delta)
        out[mask] = np.minimum(out[mask], np.nextafter(s2, 0.0))
    return out


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
    
    na = safe_clip_density(f['rho'][:,0])
    nb = safe_clip_density(f['rho'][:,1])
    
    ga = f['grd'][:,0:3]
    gb = f['grd'][:,4:7]
    
    w = f['xyz'][:,3]
    f.close()
    density_total = na + nb
    Ntot = np.dot(w, density_total)
    Na = np.dot(w, na)
    Nb = np.dot(w, nb)
    print("N_up: ", Na)
    print("N_down: ", Nb)
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


    gaa = np.einsum('ij,ij->i', ga, ga) # |grad|^2 spin 0
    gbb = np.einsum('ij,ij->i', gb, gb) # |grad|^2 spin 1
    gab = np.einsum('ij,ij->i', ga, gb)
    gtt = gaa + gbb + 2*gab
    
    #print(gtt) 
          
    npts = 4001
    delta_u = 0.0125
    #u = np.linspace(0,axis[-1],npts)           
    #u[0]=1e-6
    print("npts points: ", npts)
    print("u range: 0~", (npts-1) * delta_u)

    u_x = np.linspace(0, (npts - 1) * delta_u, npts)
    u_exchange = u_x.copy()
    u_exchange[0] = 1e-6
    u = u_x.copy()
    u[0] = 1e-10

    active_total = density_total > RHO_FLOOR
    active_up = na > RHO_FLOOR
    active_down = nb > RHO_FLOOR

    density_safe = np.where(active_total, density_total, RHO_FLOOR)
    na_safe = np.where(active_up, na, RHO_FLOOR)
    nb_safe = np.where(active_down, nb, RHO_FLOOR)

    sqrt_gtt = np.sqrt(np.clip(gtt, 0.0, None))
    sqrt_gaa = np.sqrt(np.clip(gaa, 0.0, None))
    sqrt_gbb = np.sqrt(np.clip(gbb, 0.0, None))

    kf = (3 * math.pi**2 * density_safe)**(1/3)
    #r = np.linspace(0,5,101)
    s = np.zeros_like(density_total)
    s[active_total] = sqrt_gtt[active_total] / (2 * kf[active_total] * density_total[active_total])
    s = regularize_reduced_gradient(s)

    spin_kf_up = (3 * math.pi**2 * (2 * na_safe))**(1/3)
    spin_kf_down = (3 * math.pi**2 * (2 * nb_safe))**(1/3)
    s_2a = np.zeros_like(na)
    s_2b = np.zeros_like(nb)
    s_2a[active_up] = sqrt_gaa[active_up] / (spin_kf_up[active_up] * (2 * na[active_up]))
    s_2b[active_down] = sqrt_gbb[active_down] / (spin_kf_down[active_down] * (2 * nb[active_down]))
    s_2a = regularize_reduced_gradient(s_2a)
    s_2b = regularize_reduced_gradient(s_2b)
    
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
    
    hx_lda = np.zeros(npts)
    hx_pbe = np.zeros(npts)

    hc_lda = np.zeros(npts)
    hc_pbe = np.zeros(npts)
    
    rs = (3 / (4 * np.pi * density_safe))**(1/3)
    zeta = np.zeros_like(density_total)
    zeta[active_total] = (na[active_total] - nb[active_total]) / density_total[active_total]
    zeta = np.clip(zeta, -1.0, 1.0)
    ks = np.sqrt(4 * kf / np.pi)

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
        f_out = 6.475 * h + 0.4797
        return f_out
    
    def constant_a(a,b,c,d,e,h,f,s):
        h = np.clip(h, 0.0, None)
        denom_term = d + h * s**2
        first_term = 15*e+(6*c*(1+f*s**2)*denom_term)
        second_term = 4*b*(denom_term**2)+8*a*(denom_term**3)
        first_value = np.sqrt(math.pi)*(first_term+second_term)
        third_term = 1/(16*denom_term**(7/2))
        erfcx_arg = (3 * s / 2) * np.sqrt(h / a)
        fourth_term = (3 * math.pi * np.sqrt(a) / 4) * erfcx(erfcx_arg)

        a_out = (first_value * third_term) - fourth_term
        return a_out
    
    def constant_b(s,d,h):
        numerator = 15*np.sqrt(math.pi)*s**2
        denominator = 16*(d+h*s**2)**(7/2)
        b_out  = numerator/denominator
        return b_out
    
    def G_function(v1,v2,e):
        numerator = 0.75 * math.pi + v1
        denominator = v2 * e
        out = np.zeros_like(numerator, dtype=float)
        np.divide(-numerator, denominator, out=out, where=np.abs(denominator) > 0)
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

    spin_scale_up = (1 + zeta)**(1/3) * kf
    spin_scale_down = (1 - zeta)**(1/3) * kf
    x_up = np.outer(u_exchange, spin_scale_up)
    x_down = np.outer(u_exchange, spin_scale_down)
    density_up = (2 * na)**2
    density_down = (2 * nb)**2

    hx_lda = (
        (J_lda(x_up) * density_up + J_lda(x_down) * density_down) @ w
    ) / (2 * Ntot)
    hx_pbe = (
        (J_gga(s_2a[np.newaxis, :], x_up) * density_up +
         J_gga(s_2b[np.newaxis, :], x_down) * density_down) @ w
    ) / (2 * Ntot)

    print(" ") 
    print("hx_lda finish")
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

    phi = 0.5 * ((1 + zeta)**(2/3) + (1 - zeta)**(2/3))
    t = np.zeros_like(density_total)
    t[active_total] = sqrt_gtt[active_total] / (2 * ks[active_total] * phi[active_total] * density_total[active_total])
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
        G = -2*A*(1+alpha1*rs_tmp)*np.log(1+1/(2*A*(beta1*rs_tmp**(1/2)+beta2*rs_tmp+beta3*rs_tmp**(3/2)+beta4*rs_tmp**(P+1))))
        return G

    def scaled_exp1(x):
        x = np.asarray(x, dtype=float)
        out = np.empty_like(x)
        zero = x <= 0
        large = x > 50
        small = ~(zero | large)

        if np.any(zero):
            out[zero] = 0.0

        if np.any(small):
            x_small = x[small]
            out[small] = x_small * np.exp(x_small) * exp1(x_small)

        if np.any(large):
            inv_x = 1 / x[large]
            out[large] = 1 - inv_x + 2 * inv_x**2 - 6 * inv_x**3

        return out

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

    v = np.outer(u, phi * ks)
    v_sq = v**2
    radial_scale = np.outer(u, kf / phi)
    exp_factor = np.exp(-d[np.newaxis, :] * radial_scale**2)

    f1 = (a1 + a2*v + a3*v_sq) / (1 + b1*v + b2*v_sq + b3*v**3 + b4*v**4)
    f2 = (
        -a1 - (a2-a1*b1)*v + c1*v_sq + c2*v**3 + c3*v**4 + c4*v**5
    ) * exp_factor
    Ac = (f1 + f2) / (4 * np.pi * v_sq)

    phi_ks_factor = (phi**5 * ks**2)[np.newaxis, :]
    nc_lsd = phi_ks_factor * Ac
    hc_lda = ((nc_lsd * density_total[np.newaxis, :]) @ w) / Ntot

    beta_rs = 2 * p**2 / (3 * np.pi**3) * (1 - scaled_exp1(12 * p))
    BcLM = (18*np.pi**3*(1+v_sq/12)**2)**(-1)
    decay = np.exp(-p[np.newaxis, :] * v_sq)
    Bc = BcLM * (1 - decay) + beta_rs[np.newaxis, :] * v_sq * decay

    gga_hole = Ac + (t**2)[np.newaxis, :] * Bc
    nc_gea = phi_ks_factor * gga_hole

    # Keep the last sign change in the cumulative integral for each grid column.
    Fint = CUMTRAPZ(v_sq * gga_hole, v, axis=0, initial=0)
    sign_change = Fint[1:] * Fint[:-1] < 0
    candidate_rows = np.arange(1, npts)[:, np.newaxis]
    last_crossing = np.max(np.where(sign_change, candidate_rows, 0), axis=0)
    vc = np.zeros_like(phi)
    crossing_cols = np.nonzero(last_crossing)[0]
    if crossing_cols.size:
        idx_right = last_crossing[crossing_cols]
        idx_left = idx_right - 1
        v0 = v[idx_left, crossing_cols]
        v1 = v[idx_right, crossing_cols]
        f0 = Fint[idx_left, crossing_cols]
        f1 = Fint[idx_right, crossing_cols]
        denom = f1 - f0
        vc[crossing_cols] = np.where(np.abs(denom) > 0, v0 - f0 * (v1 - v0) / denom, v1)

    nc_gga = np.where(v <= vc[np.newaxis, :], nc_gea, 0.0)
    nc_gga[:, ~active_total] = 0.0

    print(" ")
    print("size",len(hx_lda),len(hc_lda),len(hx_pbe),len(hc_pbe))
    print(" ")
    hc_pbe = ((nc_gga * density_total[np.newaxis, :]) @ w) / Ntot

    print("size",len(hx_lda),len(hc_lda),len(hx_pbe),len(hc_pbe))
    hxc_lda= hx_lda + hc_lda

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


    ex_lda  = CUMTRAPZ(4 * np.pi * hx_lda * u, u, initial=0)
    ec_lda  = CUMTRAPZ(4 * np.pi * hc_lda * u, u, initial=0)
    exc_lda = CUMTRAPZ(4 * np.pi * (hx_lda + hc_lda) * u, u, initial=0)

    ex_pbe  = CUMTRAPZ(4 * np.pi * hx_pbe * u, u, initial=0)
    ec_pbe  = CUMTRAPZ(4 * np.pi * hc_pbe * u, u, initial=0)
    exc_pbe = CUMTRAPZ(4 * np.pi * (hx_pbe + hc_pbe) * u, u, initial=0)
    
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
