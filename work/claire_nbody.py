#importing modules
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

# Inputting All Desired Values from user

name = 1
while name == 1:
	try:
		file1 = str(input("Input desired data txt file: "))
	except ValueError:
		print("Try again")
		continue
	else:
		m, x, y, z, x_dot, y_dot, z_dot = np.loadtxt('C:\\Users/Kris/PycharmProjects/senior_fall/phys410/data_q2.txt', unpack = True)
	try:
		eps = float(input("Input desired epsilon value: "))
	except ValueError:
		print("Try again")
		continue
	try:
		h = float(input("Input desired time step: "))
	except ValueError:
		print('Try again')
		continue
	try:
		ns = int(input("Input desired number of steps: "))
	except ValueError:
		print("Try again")
		continue
	try:
		f = int(input("Input desired output frequency: "))
	except ValueError:
		print("Try again")
		continue
	try:
		e = float(input("Input desired eccentricity: "))
	except ValueError:
		print("Try again")
		continue
	integrate = input("Input desired integrator- RK4 or LF2: ")
	if integrate not in ('LF2', 'RK4'):
		print("Try again")
	elif integrate == "LF2":
		integrate = 2
	else:
		integrate = 4

	break

# Defining Gravity Problem

# Calculate the velocity given inputs
v = np.sqrt(2 * (1-e))

# Calculate step size given inputs
P = np.pi * np.sqrt(2 / (1 + e)**3)
ns = int(100 * P / h)

# Square the softening
eps2 = eps*eps

# Define number of particles
N = len(m)

pos = np.transpose(np.array([x, y, z]))
vel = np.transpose(np.array([v * x_dot, v * y_dot, v * z_dot]))

#Calculating the acceleration due to gravity

def gravity(N, pos, eps2, m):
	acc = np.zeros(pos.shape)
	for i in range(N):
		rs = pos[i] - pos
		r2 = (rs**2).sum(axis=1) + eps2
		ir3 = -np.divide(m * np.ones_like(r2), np.sqrt(r2) * r2, \
		 out = np.zeros_like(r2), where=r2 !=0)
		acc[i] = (ir3[:,np.newaxis] * rs).sum(axis=0)
	return acc


def dergravity(N, pos, vel, eps2, m):
	xdot = vel[:]
	vdot = gravity(N, pos, eps2, m)
	return np.append(xdot,vdot).reshape(2*N, 3)


# Define Integrators

def lf2(pos,vel,h,N):
	half_step = pos + (h/2) * vel
	vel_next = vel + h * dergravity(N, half_step, vel, eps2, m)[N:]
	next_step = half_step + (h/2) * vel_next
	return next_step, vel_next

def rk4(pos,vel,h,N):
	k1 = h * dergravity(N, pos, vel, eps2, m)
	k2 = h * dergravity(N, pos + k1[:N]/2, vel + k1[N:] / 2, eps2, m)
	k3 = h * dergravity(N, pos + k2[:N]/2, vel + k2[N:] / 2, eps2, m)
	k4 = h * dergravity(N, pos + k3[:N], vel + k3[N:], eps2, m)
	next = np.append(pos, vel).reshape(2*N, 3) + k1/6 + k2/3 + k3/3 + k4/6
	return next[:N], next[N:]

# Define Plotting

n_out = int(ns / f + 0.5)
x = np.empty(n_out)
y = np.empty(n_out)
z = np.empty(n_out)

#Time how much it takes to integrate

start_time = time.time()

#Iterating over each time step

for i in range(ns):

	if integrate == 2:
		pos, vel = lf2(pos, vel, h, N) #leapfrog integrator
	else:
		pos, vel = rk4(pos, vel, h, N) #runge kutta integrator

	if i % f == 0:
		x = pos[:, 0]
		y = pos[:, 1]
		z = pos[:, 2]
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		im = ax.scatter(x, y, z, c=m, cmap='viridis')
		fig.colorbar(im)
		ax.set_xlabel('x')
		ax.set_ylabel('y')
		ax.set_zlabel('z')
		ax.set_xlim([-15, 15])
		ax.set_ylim([-15, 15])
		ax.set_zlim([-15, 15])
		plt.title('N Body Simulation (N = 300)')
		if i < 10:
			plt.savefig("step00" + str(i) + ".png")
		if i >= 10 and i <= 100:
			plt.savefig("step0" + str(i) + ".png")
		if i >= 100:
			plt.savefig("step" + str(i) + ".png")
		plt.close()

end_time = time.time()
total = end_time - start_time
print(total) #printing the total time it takes to run
