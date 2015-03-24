#!/opt/local/bin/python
#-*- Encoding: UTF-8 -*-


"""
Compare analytic solution of 2d diffusion equation to
numerical result
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
from twodads.twodads import input2d
from scipy.integrate import quad

# See ~/source/source/diffusion/2ddff.py
#
# Solution to the 2d diffusion equation:
#
# u(x,y,t) = \frac{1}{4 \pi D T}
#                \int_{-\infty}^{\infty} dx'
#                \int_{-\infty}^{\infty} dy'
#                    exp(-\frac{(x-x') - (y-y')^2}{4Dt}) phi(x', y')
# where phi(x,y) is the initial profile
# D is the diffusion coefficient
#
def diff_analytic_inf(xx, yy, u0, t, D):
    """
    Analytic solution of the diffusion equation on infinite plane
    Input:
    xx, yy: Grid
    u0: Initial condition
    t: Solution at time t
    D: Diffusion coefficient
    """
    res = np.zeros_like(u0)
    fdt = 4. * D * t
    print 'Diffusion coefficient %f\tTime %f\n' % (D, t)
    delta_xy = (xx[0,1] - xx[0,0]) * (yy[1,0] - yy[0,0])
    for i in np.arange(0, np.shape(u0)[0]):
        x = xx[0,i]
        #print '%d/%d' % (i, np.shape(u0)[0])
        for j in np.arange(0, np.shape(u0)[1]):
            y = yy[j,0]
            #print 'x=%f\ty=%f\tint=%f' % (x, y, np.sum( np.sum( np.exp( (-(yy)**2-(xx)**2)/fdt) , axis=1 ) ) * delta_xy / (np.pi*fdt))
            res[i,j] = np.sum(np.sum(np.exp(-((yy - y) ** 2 +
                                              (xx - x) ** 2) / fdt) * u0 , axis=1))
            res[i, j] *= delta_xy / (np.pi * fdt)
    return res



#    Stuff for the analytic solution of the 2d diffusion equation with Dirichlet BC:
#    u_t = k (u_xx + u_yy)
#    u(x,0) = u(x,Lx) = 0
#    u(0,y) = u(y,Ly) = 0
phi_x = lambda x, Lx: np.sin(2.0 * np.pi * x / Lx)
phi_y = lambda y, Ly: np.sin(2.0 * np.pi * y / Ly)
phi_intx = lambda x, n, Lx: phi_x(x, Lx) * np.sin(float(n) * np.pi * x / Lx)
phi_inty = lambda y, m, Ly: phi_y(y, Ly) * np.sin(float(m) * np.pi * y / Ly)
#u = lambda x, y, t, Anm, k, Lx, Ly: np.sum([ a_nm * np.sin( nm[0] * np.pi * x / Lx)\
#    * np.sin( nm[1] * np.pi * y / Ly)\
#    * np.exp( -( (nm[0]/Lx)**2+(nm[1]/Ly)**2)*np.pi**2 * k * t)\
#    for a_nm, nm in zip( Anm, [(n,m) for n in np.arange(1,Nx) for m in np.arange(1,Ny)]) ], axis = 0 )


def u(x, y, t, Anm, k, Lx, Ly):
    foo = [a_nm * np.sin(nm[0] * np.pi * x / Lx)\
        * np.sin(nm[1] * np.pi * y / Ly)\
        * np.exp(-((nm[0] / Lx) ** 2 + (nm[1] / Ly) ** 2) * np.pi ** 2 * k * t) \
        for a_nm, nm in zip(Anm, [(n,m) for n in np.arange(1,Nx)
                                  for m in np.arange(1,Ny)])]
    #print zip(Anm, [(n,m) for n in np.arange(1,Nx) for m in np.arange(1,Ny)])
    foo2 = np.sum(foo, axis=0)
    return foo2


sim_dir = '/home/rku000/cuda-workspace/cuda_array2/run/run_diff'
inp = input2d(sim_dir)

# Domain Setup
Nx, Ny = inp['Nx'], inp['My']
Lx = inp['Lx']        # Length in x-direction
Ly = inp['Ly']        # Length in y-direction
kappa = inp['model_params'][0]     # Diffusion coefficient
dx, dy = inp['deltax'], inp['deltay']
T = inp['tend']
dt = inp['deltat']
dtout = inp['tout']
n_time = int(T/dt)  # Number of timesteps
t_range = np.arange(0, T, dtout)

x = np.arange(-Lx / 2, Lx / 2, dx)
y = np.arange(-Ly / 2, Ly / 2, dy)
xx, yy = np.meshgrid(x,y)

print 'Loading simulation data from: %s' % sim_dir
print 'Simulation domain: [%f:%f]x[%f:%f]' % (-Lx / 2., Lx / 2.,
                                              -Ly / 2., Ly / 2.)
print 'Lx = %f, Ly = %f' % (Lx, Ly)
print 'Tend = %f, %d output' % (T, int(T / dtout))
print 'Diffusion coefficient: %f' % kappa

df = h5py.File('%s/output.h5' % sim_dir)
num_init = df['/T/0'].value
sol_num = df['/T/%d' % (int(T/dtout))].value

A_nm = [(4.0 / (Lx*Ly)) * quad(phi_intx , 0., Lx, args=(n, Lx))[0] *\
    quad(phi_inty, 0., Ly, args=(m, Ly))[0]\
    for n in np.arange(1, Nx)\
    for m in np.arange(1, Ny)]

# Analytic solution
#sa = u(xx, yy, T, A_nm, kappa, Lx, Ly)

sol_an = diff_analytic_inf(xx, yy, num_init, T, kappa)

plt.figure()
plt.title('T=%3.1f, analytic solution' % T)
plt.contourf(xx, yy, sol_an)
plt.colorbar()

plt.figure()
plt.title('T=%3.1f, numeric solution' % T )
plt.contour(xx, yy, sol_num)
plt.colorbar()

plt.figure()
plt.title('T=%3.1f, difference numerical - analytic' % T)
plt.contourf(xx, yy,  sol_num - sol_an)
plt.colorbar()
plt.show()


#######
res2ndf = np.zeros_like(t_range)
for t in t_range[::10]:
    t_idx = int(t / dtout)
    print 'time %3.1f / %3.1f, idx %d' % (t, T, t_idx)

    # sa = u(xx, yy, t, A_nm, kappa, Lx, Ly)
    sol_an = diff_analytic_inf(xx, yy, num_init, t, kappa)
    sol_num = df['/T/%d' % t_idx].value

    res2ndf[t_idx] = ((sol_an - sol_num)**2).sum()

    plt.figure()
    plt.subplot(131)
    plt.title('t=%3.1f, Analytical solution' % t)
    plt.contourf(xx, yy, sol_an)
    plt.colorbar()

    plt.subplot(132)
    plt.title('t=%3.1f, Numerical solution' % t)
    plt.contourf(xx, yy, sol_num)
    plt.colorbar()

    plt.subplot(133)
    plt.title('t=%3.1f, error' % t)
    plt.contourf(xx, yy, sol_an - sol_num)
    plt.colorbar()


res2ndf = res2ndf / float(Nx*Ny)

plt.figure()
plt.title('Error')
plt.plot(t_range, res2ndf, 'kx')


#np.savez( 'res2ndf_2d_Nx64_T5.0.npz', res2ndf = res2ndf, t_range = t_range)

plt.show()

