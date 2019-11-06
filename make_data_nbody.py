#By Ben Flaggs, Vivian Carvajal, Kris Laferriere
#11/20/18
# Edited by Kris 10/10/19

#This code will generate a set of random initial conditions for particles in a disk of thickness 4, and particles with
# mass between 1 and 5.  The radius and number of particles can be adjusted as needed. The code then saves the
# initial conditions to a text file to be used in an N-body code.

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the number of particles
N = 150
radius_out = 10
radius_in = 5
radius_ratio = radius_out/radius_in

# Define an array of the positions of the particles as random numbers in the range of [0,50). These two arrays will
# include the x, y and z positions meaning that all of the particles will be confined to a torus with outer radius 10.
# pos = radius_out*np.random.uniform(-1, 1, size=(N, 2))    # defines positions from the min radius to the outer
# for i in range(0, N):
    # value = np.sqrt(pos[i, 0]**2 + pos[i, 1]**2)
    # while radius_out < value < radius_in:
    #     pos[i, :] = radius_in * np.random.uniform(-1, 1, size=(1, 2))
    #     value = np.sqrt(pos[i, 0]**2 + pos[i, 1]**2)
# pos_z = np.random.uniform(-2, 2, size=(N, 1))

# to build a torus
theta = np.random.uniform(0, 2*np.pi, size=(N, 1))
phi = np.random.uniform(0, 2*np.pi, size=(N, 1))
# phi = np.linspace(0, 2*np.pi, N)
# theta, phi = np.meshgrid(theta, phi)
pos = np.zeros((N, 3))
for i in range(0, N):
    pos[i, 0] = (radius_out + radius_in*np.cos(theta[i]))*np.cos(phi[i])
    pos[i, 1] = (radius_out + radius_in*np.cos(theta[i]))*np.sin(phi[i])
    pos[i, 2] = radius_in*np.sin(theta[i])
# x = radius_out + radius_in*np.cos(theta)*np.cos(phi)
# y = radius_out + radius_in*np.cos(theta)*np.sin(phi)
# z = radius_in*np.sin(theta)

# Define an array of velocities for the particles as being random (for the velocities in the x, y and z directions)
# values all in the same direction, ranging from zero to 20 m/s, but with no velocity in the z direction initially.
i_vel = 1
# vel = i_vel * np.random.uniform(0, 2, size=(N, 2))
vel_x = i_vel * np.sin(theta)*np.cos(phi)
vel_y = i_vel * np.sin(theta)*np.sin(phi)
vel_z = np.zeros((N, 1))
vel = np.hstack((vel_x, vel_y, vel_z))


# Define an array of random masses for each particle in the range [1,5)
mass = np.random.uniform(1, 5, size=(N, 1))

# Saving the data
d = np.hstack((mass, pos, vel))
np.savetxt("phys410\data_Nparticles.txt", d) #, fmt='%2.2f', delimiter='\t',
           # header='mass \t posx \t posy \t posz \t velx \t vely \t velz')

#Generate plot of initial conditions
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
p = ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], '.', c=mass[:, 0])
fig.colorbar(p, ax=ax)

ax.set_zlim(-20, 20)
ax.set_xlim(-20, 20)
ax.set_ylim(-20, 20)

plt.title('N-Body Simulation of 500 Particles in a Disk')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

plt.savefig("n_body_%05d.png" % 1)
plt.show()