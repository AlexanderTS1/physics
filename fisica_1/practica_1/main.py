import logging
import sys

import matplotlib.pyplot as plt
import numpy as np

log = logging.getLogger(__name__)

G = 9.8

def update_acceleration_stokes(mass, velocity, b, ve):
    return ((mass * G - b * pow(velocity, ve)) / mass)

def update_acceleration_stokes2(mass, velocity, b, ve):
  return (G - (b * velocity**ve / mass))

def update_velocity(velocity, acceleration, dt):
    return velocity + acceleration * dt

def update_position(position, velocity, dt):
    return position - velocity * dt

def simulate(dt, y0, v0, a0, b, mass, ve):

    time_samples_ = np.empty(0)
    position_samples_ = np.empty(0)
    velocity_samples_ = np.empty(0)
    acceleration_samples_ = np.empty(0)

    t_ = 0.0
    a_ = a0
    v_ = v0
    y_ = y0

    log.info("Starting simulation...")
    log.info("Drop mass is {0}".format(mass))
    log.info("Position at t=0 [s] is {0} [m]".format(y_))
    log.info("Velocity at t=0 [s] is {0} [m/s]".format(v_))
    log.info("Acceleration at t=0 [s] is {0} [m/s2]".format(a_))

    while y_ > 0:

        time_samples_ = np.append(time_samples_, t_)
        position_samples_ = np.append(position_samples_, y_)
        velocity_samples_ = np.append(velocity_samples_, v_)
        acceleration_samples_ = np.append(acceleration_samples_, a_)

        #log.info("Time: {0} [s]".format(t_))
        #log.info("Position: {0} [m]".format(y_))
        #log.info("Velocity: {0} [m/s]".format(v_))
        #log.info("Acceleration: {0} [m/s2]".format(a_))

        t_ += dt
        y_ = update_position(y_, v_, dt)
        v_ = update_velocity(v_, a_, dt)
        a_ = update_acceleration_stokes2(mass, np.sqrt(v_ * v_), b, ve)

    log.info("Elapsed time until hitting the ground: {0} [s]".format(t_))
    log.info("Impact velocity: {0} [m/s]".format(v_))
    log.info("Impact acceleration: {0} [m/s2]".format(a_))

    return time_samples_, position_samples_, velocity_samples_, acceleration_samples_

def plot(times, positions, velocities, accelerations, labels):

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rc('font', size='22')

    plt.subplot(3, 1, 1)
    for i in range(len(positions)):
      plt.plot(times[i], positions[i], "-", linewidth=2)

    plt.xticks(np.arange(0, 90, 5.0))
    plt.xlabel(r"\textbf{Tiempo $[s]$}")
    plt.ylabel(r"\textbf{Posición $[m]$}")
    plt.legend(labels)
    plt.grid()

    plt.subplot(3, 1, 2)
    for i in range(len(velocities)):
      plt.plot(times[i], velocities[i], "-", linewidth=2)
    plt.xticks(np.arange(0, 90, 5.0))
    plt.xlabel(r"\textbf{Tiempo $[s]$}")
    plt.ylabel(r"\textbf{Velocidad $[m/s]$}")
    plt.legend(labels)
    plt.grid()

    plt.subplot(3, 1, 3)
    for i in range(len(accelerations)):
      plt.plot(times[i], accelerations[i], "-", linewidth=2)
    plt.xticks(np.arange(0, 90, 5.0))
    plt.xlabel(r"\textbf{Tiempo $[s]$}")
    plt.ylabel(r"\textbf{Aceleración $[m/s^2]$}")
    plt.legend(labels)
    plt.grid()

    plt.show()

if __name__ == "__main__":

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    # Simulation parameters
    dt_ = 0.001 # [s]
    height_ = 100 # [m]

    times_list_ = []
    positions_list_ = []
    velocities_list_ = []
    accelerations_list_ = []
    label_list_ = []

    for r in [0.25]:
        for ve in [1, 2, 3, 4]:
            drop_radius_ = r # [mm]
            drop_radius_ = drop_radius_ / 1000.0 # [m]
            log.info("Drop radius {0} [m]".format(drop_radius_))

            drop_volume_ = 4.0 / 3.0 * np.pi * drop_radius_ * drop_radius_ * drop_radius_ # [m3]
            drop_density_ = 1000 # [kg / m3]
            drop_mass_ = drop_volume_ * drop_density_ # [kg]

            temperature_ = 20 # [Cº]
            temperature_ = temperature_ + 273.15 # [K]
            air_viscosity_ = 18e-6


            b_ = 6 * np.pi * drop_radius_ * air_viscosity_

            label_ = "r = " + str(drop_radius_ * 1000.0) + " [mm], $v^" + str(ve) + "$"

            times_, positions_, velocities_, accelerations_ = simulate(dt_, height_, 0.0, G, b_, drop_mass_, ve)
            times_list_.append(times_)
            positions_list_.append(positions_)
            velocities_list_.append(velocities_)
            accelerations_list_.append(accelerations_)
            label_list_.append(label_)

        plot(times_list_, positions_list_, velocities_list_, accelerations_list_, label_list_)
