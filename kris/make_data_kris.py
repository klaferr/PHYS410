#By Ben Flaggs, Vivian Carvajal, Kris Laferriere
#11/20/18

#This code will generate a set of random initial conditions for particles
#in a disk of thickness 2. The radius and number of particles can be
#adjusted as needed.
#The code then saves the initial conditions to a text file to be used in an 
#N-body code.

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the number of particles

N = 500
radius = 5

# Define an array of the positions of the particles as random numbers in the
# range of [0,50). These two arrays will include the x, y and z positions 
# meaning that all of the particles will be confined to a disk of radius 50.
pos = radius*np.random.uniform(0, 1, size=(N, 1))
phi = np.random.uniform(0, 2*np.pi, size=(N, 1))
# for i in range(0, N):
#     value = np.sqrt(pos[i,0]**2 + pos[i,1]**2)
#     while value > radius:
#         pos[i,:] = radius*np.random.uniform(-1,1,size=(1,2))
#         value = np.sqrt(pos[i,0]**2 + pos[i,1]**2)
z = np.random.uniform(-1, 1, size=(N, 1))

# Define an array of velocities for the particles as being random (for the
# velocities in the x, y and z directions) between values of [-1,1)
# vel = np.abs(pos)*np.random.uniform(0, 1, size=(N, 2))
velr = 1*np.random.uniform(0, 1, size=(N, 1))
velphi = np.pi*2/(pos)**(1/2)
vel_z = np.zeros((N, 1))

# Define an array of random masses for each particle in the range [1,5)
mass = np.random.uniform(1, 5, size=(N, 1))

d = np.hstack((mass, pos, phi, z, velr, velphi, vel_z))

np.savetxt("phys410\data_Nparticles.txt", d, fmt='%f')

#Generate plot of initial conditions
x = pos*np.cos(phi)
y = pos*np.sin(phi)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
p = ax.scatter(x, y, z, '.', c=mass[:, 0])
fig.colorbar(p,ax=ax)

ax.set_zlim(-5, 5)
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)

plt.title('N-Body Simulation of 500 Particles in a Disk')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

plt.savefig("n_body_%05d.png" %1)
plt.show()

# need to transform the velocity and then plot them as arrow markers.