# By Kris Laferriere, Claire Hinrichs, Matt Wilkin
# 11/20/19
# Based off of a program written by Kris Laferriere, Vivian Carvajal, and Ben Flaggs for ASTR415
# This code uses initial conditions from a data file and integrates the system using second-order Leapfrog (LF2), and
# ensures that energy is conserved.

# ----------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors

# Functions
# ----------------------------------------------------


def collisions(N, p, epsilon_0, dist, mass_arr, vnew):
    coll = np.array([part for part in range(N) if (np.sqrt(dist[part, 0] ** 2 + dist[part, 1] ** 2 + dist[part, 2] ** 2) <= epsilon_0) & (part != p)])
    if np.size(coll) >= 2:
        # need to adjust the momentum
        if mass_arr[coll[0]] > mass_arr[coll[1]] and mass_arr[coll[0]] > 0:
            vnew[coll[0]] = (mass_arr[coll[0]]*vnew[coll[0]] + mass_arr[coll[1]]*vnew[coll[1]])/(mass_arr[coll[0]]
                                                                                                 + mass_arr[coll[1]])
            mass_arr[coll[0]] += mass_arr[coll[1]]
            mass_arr[coll[1]] = 0
        # elif mass_arr[coll[0]] == 0 and mass_arr[coll[1]] == 0:
        #     vnew[coll[0]] = 0
        #     vnew[coll[1]] = 0
        else:
            vnew[coll[1]] = (mass_arr[coll[0]]*vnew[coll[0]] + mass_arr[coll[1]]*vnew[coll[1]])/(mass_arr[coll[0]]
                                                                                                 + mass_arr[coll[1]])
            mass_arr[coll[1]] += mass_arr[coll[0]]
            mass_arr[coll[0]] = 0
    return vnew, mass_arr


def derivatives(n_particle, position, velocity, epsilon, mass_a, soften):
    """ This function outputs derivative information using the function gravity2"""
    v_dot, velocity, m = gravity2(n_particle, position[:, 0:3], epsilon, mass_a, soften, velocity)
    out = np.array([mass_a, velocity[:, 0], velocity[:, 1], velocity[:, 2], v_dot[:, 0], v_dot[:, 1], v_dot[:, 2]])
    return np.transpose(out)


def gravity2(Np, position, epsilon_0, mass_arr, softening, velocity):
    """ Gravity function to calculate the accelerations. Checks the distance between the particles, and if it is less
    than the epsilon value, make it a collision. Output the acceleration for each particle. """

    acc = np.zeros(position.shape)
    for p in range(Np):
        rs = position[p] - position         # array of relative position vectors
        velocity, mass_arr = collisions(N, p, epsilon_0, rs, mass_arr, velocity) # insert collision function here using velocity vector
        r2 = (rs ** 2).sum(axis=1) + softening**2
        ir3 = -1 * np.divide(mass_arr * np.ones_like(r2), np.sqrt(r2) * r2, out=np.zeros_like(r2), where=r2 != 0)
        acc[p] = (ir3[:, np.newaxis] * rs).sum(axis=0)
    return acc, velocity, mass_arr


def leapfrog(func, values, dt, time, Np, particle_mass, epsilon, soften):
    """ Leapfrog integrator. """
    froggy = np.zeros(shape=(Np, 7, time))
    froggy[:, 1:7, 0] = values[:, 0:6]
    froggy[:, 0, 0] = particle_mass[:]
    count = 1
    for t in range(1, time):
        froggy[:, 0, t] = particle_mass[:]
        half = froggy[:, 1:4, t - 1] + 0.5 * dt * froggy[:, 4:7, t - 1]
        froggy[:, 4:7, t] = froggy[:, 4:7, t - 1] + dt * func(Np, half, froggy[:, 4:7, t - 1], epsilon, particle_mass,
                                                              soften)[:, 4:7]
        froggy[:, 1:4, t] = half + 0.5 * dt * froggy[:, 4:7, t]

        if t % frequency == 0:
            count = count + 1
            np.savetxt('n_body_%05d.txt' % t, froggy[:, :, t])

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            p = ax.scatter(froggy[:, 1, t], froggy[:, 2, t], froggy[:, 3, t], '.', c=particle_mass)
                           # norm=matplotlib.colors.LogNorm())
            fig.colorbar(p, ax=ax)
            ax.set_zlim(-15, 15)
            ax.set_xlim(-100, 100)
            ax.set_ylim(-100, 100)
            plt.title('N-Body Simulation of 500 Particles in a Disk')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
            plt.savefig("n_body_%05d.png" % count)
            plt.close(fig)
    return froggy


def vel_pos(data, t, rcm, M, vcm):
    vel_t = np.sqrt((data[:, 4, t])**2 + (data[:, 5, t])**2 + (data[:, 6, t])**2)
    x_cm = (data[:, 1, t] - rcm[0])**2 + (data[:, 2, t]-rcm[1])**2 + (data[:, 3, t]-rcm[2])**2
    pos_t = np.sqrt((data[:, 1, t])**2 + (data[:, 2, t])**2 + (data[:, 3, t])**2)
    ang_mom_0 = np.sum(data[:, 0, t] * vel_t * pos_t)
    energy = np.sum(0.5*data[:, 0, t]*vel_t**2) + np.sum(M*data[:, 0, t]/x_cm)
    return ang_mom_0, energy


def com(data_array, time):
    com_m = np.sum(data_array[:, 0, time])
    com_pos = np.zeros(3)
    com_vel = np.zeros(3)
    for i in range(3):
        com_pos[i] = np.sum(data_array[:, 0, time]*data_array[:, i+1, time]/com_m)
        com_vel[i] = np.sum(data_array[:, i+1, time]*data_array[:, 0, time]/com_m)
    return com_pos, com_vel, com_m


if __name__ == "__main__":
    # Parameters
    epsilon = 0.00          # distance at which particles are considered to be colliding
    soft = 1  # the softening parameter
    frequency = 10      # how many steps occur between plot output
    h = 0.05            # the size of steps
    n = 100            # the number of steps
    ta = n - 2
    tn = n - 1

    # Read in the data
    filename = "C:\\Users/Kris/PycharmProjects/senior_fall/phys410/data_Nparticles.txt"
    mass, r, phi, z, r_dot, w, z_dot = np.loadtxt(filename, unpack=True)

    # Transform to cartesian coordinates.
    x = r*np.cos(phi)
    y = r*np.sin(phi)
    x_dot = r_dot*np.cos(phi) - r*w*np.sin(phi)
    y_dot = r_dot*np.sin(phi) + r*w*np.cos(phi)

    # Define the mass, position, velocity arrays
    mass_array = mass
    pos_array = np.transpose(np.array([x, y, z]))
    velocity_array = np.transpose(np.array([x_dot, y_dot, z_dot]))
    a = np.transpose(np.array([x, y, z, x_dot, y_dot, z_dot]))
    N = len(mass)

    # Integrate over time
    trial = leapfrog(derivatives, a, h, n, N, mass_array, epsilon, soft)

    # Find the center of mass at time zero to calculate the angular momentum at time 0
    r_com, v_com, Mass = com(trial, 0)
    print('the angular momentum at time 0 is')
    ang_1, en_1 = vel_pos(trial, 0, r_com, Mass, v_com)
    print(ang_1)

    # Find the center of mass at time end-2 and calculate the angular momentum at that time
    r_com, v_com, Mass = com(trial, tn)
    print('the angular momentum at time end is')
    ang_2, eng_2 = vel_pos(trial, tn, r_com, Mass, v_com)
    print(ang_2)

    # Print the energy values
    print('The energy values are')
    print(en_1)
    print(eng_2)

    # Print the maximum distance and velocities.
    print('the max vy is ')
    print(np.max(trial[:, 4, ta]))
    print('the max distance at end is')
    print(np.max(np.sqrt((trial[:, 0, ta])**2 + (trial[:, 1, ta])**2 + (trial[:, 2, ta])**2)))
    print('the average distance at end is')
    print(np.average(np.sqrt((trial[:, 0, tn])**2 + (trial[:, 1, tn])**2 + (trial[:, 2, tn])**2)))

    # Plot the y vs. x data for each particle for the given number of steps,
    # given frequency of output and given eccentricity
    # plt.figure()
    # for i in range(0, N):
    #     plt.plot(trial[i, 0, :], trial[i, 1, :], '.', label='Particle %f' % i)
    # plt.title('y vs. x for LF2 with e={0} and h={1}'.format(e, h))
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.show()

    # Plot and save the final product
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for j in range(0, N):
        # p = ax.plot(trial[i, 1, :], trial[i, 2, :], trial[i, 3, :], '-', label='Particle %f' % i)
        if trial[j, 0, 49] != 0:
            ax.plot(trial[j, 1, :], trial[j, 2, :], trial[j, 3, :], '-', label='Particle %f' % j)

    ax.set_zlim(-5, 5)
    ax.set_xlim(-30, 30)
    ax.set_ylim(-30, 30)

    # plt.title('y vs. x for LF2 with e={0} and h={1}'.format(e, h))
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    r_s=np.zeros((tn, 3))
    vs= np.zeros((tn, 3))
    ms= np.zeros(tn)
    for i in range(0, tn):
        r_s[i, 0:3], vs[i], ms[i] = com(trial, i)

    # plt.plot(rs[:, 0], rs[:, 1], rs[:, 2], 'g.')
    print(r_s[tn-1, 0], r_s[tn-1, 1], r_s[tn-1, 2])
    plt.show()


    # from below is copy pasted
    # Finding the maginute of v from particle one and two, which will be used later in the E(t) equation
    v_mag_p1 = np.sqrt(np.power(trial[0, 4, :], 2) + np.power(trial[0, 5, :], 2) + np.power(trial[0, 6, :], 2))
    v_mag_p2 = np.sqrt(np.power(trial[1, 4, :], 2) + np.power(trial[1, 5, :], 2) + np.power(trial[1, 6, :], 2))
    r_mag_p2 = np.sqrt(np.power(trial[1, 1, :]-trial[0, 1, :], 2) + np.power(trial[1, 2, :]-trial[0, 2, :], 2)
                       + np.power(trial[1, 3, :]-trial[0, 3, :], 2))

    # Find E(0) and E(t) where equations for both these values are given
    # in the homework handout       # need to find this equation
    E_0 = -1 * 0.5      # removed the eccentricity value since that is not included in this version
    E_t = 0.5 * np.power(v_mag_p1[:], 2) + 0.5 * np.power(v_mag_p2[:], 2) - np.divide(1, r_mag_p2[:])

    # Find the fractional change in total energy of the system, the equation
    # for this value was also given in the homework handout
    frac_E = np.divide((E_t - E_0), np.absolute(E_0))

    # Define the time that the system is integrated over
    time = np.linspace(0, h * n, np.size(frac_E, 0))
    print(np.shape(time))

    # Plot the fractional change in total energy vs. time for this system
    plt.figure()
    plt.plot(time, frac_E, 'k-', label=r'$\frac{E(t) - E(0)}{\mid E(0) \mid}$')
    # plt.title('Fractional Energy vs. Time for LF2 with e={0} and h={1}'.format(e, h))
    plt.xlabel('Time')
    plt.ylabel('Fractional Change in Total Energy')
    plt.legend()
    plt.show()