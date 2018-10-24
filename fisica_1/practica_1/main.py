import logging
import sys

import matplotlib.pyplot as plt
import numpy as np

log = logging.getLogger(__name__)

G = 9.8

def update_velocity(velocity, acceleration, dt):
    return velocity + acceleration * dt

def update_position(position, velocity, dt):
    return position + velocity * dt

def update_acceleration_stokes(acceleration, mass, velocity, b):
    return ((mass * G - b * velocity) / mass)

def update_velocity_stokes(velocity, mass, acceleration, b):
    return ((mass * acceleration - mass * G) / (-1.0 * b))

def update_position_stokes(position, velocity, dt):
    return position - velocity * dt

def simulate(dt, y0, v0, a0, b, mass):
    
    time_samples_ = np.empty(0)
    position_samples_ = np.empty(0)
    velocity_samples_ = np.empty(0)
    acceleration_samples_ = np.empty(0)
    
    t_ = 0.0
    a_ = a0
    v_ = v0
    y_ = y0
    
    log.info("Starting simulation...")
    log.info("Position at t=0 [s] is {0} [m]".format(y_))
    log.info("Velocity at t=0 [s] is {0} [m/s]".format(v_))
    log.info("Acceleration at t=0 [s] is {0} [m/s2]".format(a_))
    
    while y_ > 0:
        
        time_samples_ = np.append(time_samples_, t_)
        position_samples_ = np.append(position_samples_, y_)
        velocity_samples_ = np.append(velocity_samples_, v_)
        acceleration_samples_ = np.append(acceleration_samples_, a_)
        
        log.info("Time: {0} [s]".format(t_))
        log.info("Position: {0} [m]".format(y_))
        log.info("Velocity: {0} [m/s]".format(v_))
        log.info("Acceleration: {0} [m/s2]".format(a_))
        
        v_t_1_ = v_
        
        t_ += dt
        y_ = update_position_stokes(y_, v_, dt)
        #v_ = update_velocity_stokes(v_, mass, a_, b)
        v_ = update_velocity(v_, a_, dt)
        a_ = update_acceleration_stokes(a_, mass, v_, b)
        
    log.info("Elapsed time until hitting the ground: {0} [s]".format(t_))
    log.info("Impact velocity: {0} [m/s]".format(v_))
    log.info("Impact acceleration: {0} [m/s2]".format(a_))
        
    return time_samples_, position_samples_, velocity_samples_, acceleration_samples_

def plot(times, positions, velocities, accelerations):
    
    plt.subplot(3, 1, 1)
    plt.plot(times, positions, "-")
    plt.xlabel("Time [s]")
    plt.ylabel("Position [m]")
    plt.grid()
    
    plt.subplot(3, 1, 2)
    plt.plot(times, velocities, "-")
    plt.xlabel("Time [s]")
    plt.ylabel("Velocity [m/s]")
    plt.grid()
    
    plt.subplot(3, 1, 3)
    plt.plot(times, accelerations, "-")
    plt.xlabel("Time [s]")
    plt.ylabel("Acceleration [m/s2]")
    plt.grid()
    
    plt.show()
    
if __name__ == "__main__":

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    
    # Simulation parameters
    dt_ = 0.01 # [s]
    height_ = 10000 # [m]
    
    drop_radius_ = 1 # [mm]
    drop_radius_ = drop_radius_ / 1000 # [m]
    
    drop_volume_ = 4.0/3.0 * np.pi * pow(drop_radius_, 3) # [m3]
    drop_density_ = 997 # [kg / m3]
    drop_mass_ = drop_volume_ * drop_density_ # [kg]
    
    temperature_ = 20 # [Cº]
    temperature_ = temperature_ + 273.15 # [K]
    air_viscosity_ = 1.81 * pow(10, -5) # [Pa s]
    
    b_ = 6 * np.pi * drop_radius_ * air_viscosity_
    
    times_, positions_, velocities_, accelerations_ = simulate(dt_, height_, 0.0, G, b_, drop_mass_)
    plot(times_, positions_, velocities_, accelerations_)