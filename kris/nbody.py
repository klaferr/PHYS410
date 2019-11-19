# This code uses initial conditions from a data file and other parameters
# given by the user and integrates the system using second-order
# Leapfrog (LF2) until something "interesting" happens

# By Ben Flaggs, Vivian Carvajal, Kris Laferriere
# 11/18/18

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors


epsil2 = 2     # the softening parameter
soft = 1           # the eccentricity of the orbit
frequency = 2
h = 0.05        # the size of steps
n = 50     # the number of steps
a = n - 2
tn = n - 1

# Read in the data from the file name given
mass, r, phi, z, r_dot, w, z_dot = np.loadtxt("C:\\Users/Kris/PycharmProjects/senior_fall/phys410/data_Nparticles.txt", unpack=True)
x = r*np.cos(phi)
y = r*np.sin(phi)
x_dot = r_dot*np.cos(phi) - r*w*np.sin(phi)
y_dot = r_dot*np.sin(phi) + r*w*np.cos(phi)

#Define the mass, position, velocity arrays
mass_array = mass
pos_array = np.array([x, y, z])
velo_array = np.array([x_dot, y_dot, z_dot])

pos_array = np.transpose(pos_array)
velo_array = np.transpose(velo_array)

"""
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
p = ax.scatter(x, y, z, '.', c=mass)
fig.colorbar(p, ax=ax)

ax.set_zlim(-15, 15)
ax.set_xlim(-15, 15)
ax.set_ylim(-15, 15)

plt.title('N-Body Simulation of 500 Particles in a Disk')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()
"""

y = np.array([x, y, z, x_dot, y_dot, z_dot])
y = np.transpose(y)
N = len(mass)


# Derivs function that outputs derivative information
def derivs(N, pos, velo, eps2, m):
    vdot, m = gravity2(N, pos[:, 0:3], eps2, m) # need to do if m == 0 remove row.
    out = np.array([m, velo[:, 0], velo[:, 1], velo[:, 2], vdot[:, 0], vdot[:, 1], vdot[:, 2]])
    return np.transpose(out)


# Gravity function to calculate the accelerations
def gravity2(N, pos, eps2, m):
    acc = np.zeros(pos.shape)
    for i in range(N):
        rs = pos[i] - pos  # array of relative position vectors
        # find where rs <= eps2, print('collision?'), set the mass equal to the sum,
        test = np.array([j for j in range(N) if (np.sqrt(rs[j, 0] **2 +rs[j, 1]**2 + rs[j, 2]**2) <= eps2) & (j != i)])
        if np.size(test) >= 2:
            rs[test[0]] = 0
            if m[test[0]] > m[test[1]]:
                m[test[0]] += m[test[1]]
                m[test[1]] = 0
            else:
                m[test[1]] += m[test[0]]
                m[test[0]] = 0

        r2 = (rs ** 2).sum(axis=1) + (soft) ** 2  # the epsilon factor keeps them from being zero.
        # scalar part of accel. for all pairs
        ir3 = -1* np.divide(m * np.ones_like(r2), np.sqrt(r2) * r2, out=np.zeros_like(r2), where=r2 != 0)
        acc[i] = (ir3[:, np.newaxis] * rs).sum(axis=0)  # add accel.
        # acc[i] = 0
        # need a 'if two or more particles are within the same x,y,z, combine their mass and velocities.
    return acc, m


# Leapfrog integrator
def leapfrog(derivs, y, dt, n, N, m):
    froggy = np.zeros(shape=(N, 7, n))
    froggy[:, 1:7, 0] = y[:, 0:6]
    froggy[:, 0, 0] = m[:]
    count = 1
    for i in range(1, n):
        froggy[:, 0, i] = m[:]
        half = froggy[:, 1:4, i - 1] + 0.5 * dt * froggy[:, 4:7, i - 1]
        froggy[:, 4:7, i] = froggy[:, 4:7, i - 1] + dt * derivs(N, half, froggy[:, 4:7, i - 1], epsil2, m)[:, 4:7]
        froggy[:, 1:4, i] = half + 0.5 * dt * froggy[:, 4:7, i]
        if i % frequency == 0:
            count = count + 1
            # data = np.concatenate((ms[:, 0, i], froggy[:, :, i]), axis=None)
            # np.savetxt('n_body_%05d.txt' % i, data)
            np.savetxt('n_body_%05d.txt' % i, froggy[:, :, i])

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            # p = ax.scatter(froggy[:, 0, i], froggy[:, 1, i], froggy[:, 2, i], '.', c=mass, norm=matplotlib.colors.LogNorm())
            p = ax.scatter(froggy[:, 1, i], froggy[:, 2, i], froggy[:, 3, i], '.', c=mass, norm=matplotlib.colors.LogNorm())

            fig.colorbar(p, ax=ax)
            ax.set_zlim(-10, 10)
            ax.set_xlim(-10, 10)
            ax.set_ylim(-10, 10)
            plt.title('N-Body Simulation of 500 Particles in a Disk')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
            plt.savefig("n_body_%05d.png" % count)
            plt.close(fig)
    return froggy       # in shape particle, position(x,y,z), velocity(x,y,z), time


trial = leapfrog(derivs, y, h, n, N, mass_array) # is this right?
print(np.shape(trial))
print(trial[:, :, 49])


def vel_pos(data, t, rcm, M, vcm):
    vel_t = np.sqrt( (data[:, 4, t])**2 + (data[:, 5, t])**2 + (data[:, 6, t])**2 )
    x_cm = (data[:, 1, t] - rcm[0])**2 + (data[:, 2, t]-rcm[1])**2 + (data[:, 3, t]-rcm[2])**2
    pos_t = np.sqrt( (data[:, 1, t])**2 + (data[:, 2, t])**2 + (data[:, 3, t])**2 )
    # ke_0 = 0.5 * data[0, 0, :] * vel_t ** 2
    # pe_0 = 1
    # energy_0 = 0.5 * data[:, 0, t] * (data[:, 4:7, t]) ** 2  # + potential energy in gravity field?
    # ang_mom_0 = rcm*M*vcm + np.sum(data[:, 0, t] * vel_t * pos_t)
    ang_mom_0 = np.sum(data[:, 0, t] * vel_t * pos_t)
    energy = np.sum(0.5*data[:, 0, t]*vel_t**2) + np.sum(M*data[:, 0, t]/x_cm)
    return ang_mom_0, energy


def com(data_array, time):
    M = np.sum(data_array[:, 0, time])
    com_pos = np.zeros(3)
    com_vel = np.zeros(3)
    for i in range(3):
        com_pos[i] = np.sum(data_array[:, 0, time]*data_array[:, i+1, time]/M)
        com_vel[i] = np.sum(data_array[:, i+1, time]*data_array[:, 0, time]/M)
    # pos_array = np.sqrt( (data_array[:, 1, time])**2 + (data_array[:, 2, time])**2  + (data_array[:, 3, time])**2 )
    # vel_array = np.sqrt( (data_array[:, 4, time])**2 + (data_array[:, 5, time])**2  + (data_array[:, 6, time])**2 )
    # com_pos = np.sum(pos_array*data_array[:, 0, time]/M)
    # com_vel = np.sum(data_array[:, 0, time]*vel_array/M)
    return com_pos, com_vel, M


r_com, v_com, Mass = com(trial, 0)

# trial = timestep, value, particle
print('the angular momentum at time 0 is')
ang_1, en_1 = vel_pos(trial, 0, r_com, Mass, v_com)
print(ang_1)

r_com, v_com, Mass = com(trial, tn)
print('the angular momentum at time end is')
ang_2, eng_2 = vel_pos(trial, tn, r_com, Mass, v_com)
print(ang_2)
print(en_1)
print(eng_2)

print('the max vy is ')
print(np.max(trial[:, 4, a]))
print('the max distance at end is')
print(np.max(np.sqrt((trial[:, 0, a])**2 + (trial[:, 1, a])**2 + (trial[:, 2, a])**2)))
print('the average distance at end is')
print(np.average(np.sqrt((trial[:, 0, a])**2 + (trial[:, 1, a])**2 + (trial[:, 2, a])**2)))



# Plot the y vs. x data for each particle for the given number of steps,
# given frequency of output and given eccentricity
# plt.figure()
# for i in range(0, N):
#     plt.plot(trial[i, 0, :], trial[i, 1, :], '.', label='Particle %f' % i)
# plt.title('y vs. x for LF2 with e={0} and h={1}'.format(e, h))
# plt.xlabel('x')
# plt.ylabel('y')
# plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i in range(0, N):
    # p = ax.plot(trial[i, 1, :], trial[i, 2, :], trial[i, 3, :], '-', label='Particle %f' % i)
    if trial[i, 0, 49] != 0:
        ax.plot(trial[i, 1, :], trial[i, 2, :], trial[i, 3, :], '-', label='Particle %f' %i)

# fig.colorbar(p, ax=ax)

ax.set_zlim(-10, 10)
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)

# plt.title('y vs. x for LF2 with e={0} and h={1}'.format(e, h))
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')


# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
rs=np.zeros((tn, 3))
vs= np.zeros((tn, 3))
ms= np.zeros(tn)
for i in range(0, tn):
    rs[i, 0:3], vs[i], ms[i] = com(trial, i)

# plt.plot(rs[:, 0], rs[:, 1], rs[:, 2], 'g.')
print(rs[tn-1, 0], rs[tn-1, 1], rs[tn-1, 2])
plt.show()