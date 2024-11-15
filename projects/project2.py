import numpy as np
import matplotlib.pyplot as plt
from ase import Atoms

# Simulation parameters
dt = 0.01  # Time step
total_steps = 10000  # Number of steps
box_size = 100.0  # Size of the cubic box (periodic boundary conditions)
k = 1.0  # Spring constant
mass = 1.0  # Particle mass
r0 = 1.0  # Equilibrium separation between adjacent particles
target_temperature = 0.1  # Target temperature for velocity rescaling
rescale_interval = 100  # Number of steps between velocity rescaling
n_particles = 20  # Number of particles
epsilon_repulsive = 1.0  # Depth of the repulsive Lennard-Jones potential
epsilon_attractive = 0.5  # Depth of the attractive Lennard-Jones potential
sigma = 1.0  # Distance at which the Lennard-Jones potential is zero

# Function to generate a random chain configuration on a 3D cubic lattice
def generate_random_chain(n_particles, box_size, r0):
    positions = np.zeros((n_particles, 3))
    directions = [np.array([1, 0, 0]), np.array([-1, 0, 0]),
                  np.array([0, 1, 0]), np.array([0, -1, 0]),
                  np.array([0, 0, 1]), np.array([0, 0, -1])]
    used_positions = set()
    current_position = np.array([box_size / 2, box_size / 2, box_size / 2])
    used_positions.add(tuple(current_position))
    positions[0] = current_position

    for i in range(1, n_particles):
        np.random.shuffle(directions)
        for direction in directions:
            next_position = current_position + r0 * direction
            next_position = np.mod(next_position, box_size)  # Apply periodic boundary conditions
            if tuple(next_position) not in used_positions:
                positions[i] = next_position
                used_positions.add(tuple(next_position))
                current_position = next_position
                break
        else:
            raise ValueError("Unable to place all particles without overlap. Try reducing the number of particles or increasing the box size.")
    return positions

# Particle initial positions using random chain configuration
positions = generate_random_chain(n_particles, box_size, r0)

# Particle initial velocities sampled from the Maxwell-Boltzmann distribution
velocities = np.random.normal(0, np.sqrt(target_temperature / mass), (n_particles, 3))

# Helper functions to calculate forces and apply periodic boundary conditions
def apply_pbc(position, box_size):
    return np.mod(position, box_size)

def harmonic_force(positions, k, r0, box_size):
    forces = np.zeros_like(positions)
    # Calculate forces only between adjacent particles
    for i in range(n_particles - 1):
        displacement = positions[i + 1] - positions[i]
        displacement = displacement - box_size * np.round(displacement / box_size)  # Apply minimum image convention
        distance = np.linalg.norm(displacement)
        direction = displacement / distance
        force_magnitude = -k * (distance - r0)
        force = force_magnitude * direction
        forces[i] -= force
        forces[i + 1] += force
    return forces

def repulsive_lennard_jones_force(positions, epsilon, sigma, box_size):
    forces = np.zeros_like(positions)
    # Calculate repulsive Lennard-Jones forces between particles separated by one spacer particle
    for i in range(n_particles - 2):
        j = i + 2
        displacement = positions[j] - positions[i]
        displacement = displacement - box_size * np.round(displacement / box_size)  # Apply minimum image convention
        distance = np.linalg.norm(displacement)
        if distance < 3.0 * sigma:  # Apply a cutoff for efficiency
            direction = displacement / distance
            force_magnitude = 24 * epsilon * ((sigma / distance) ** 13 - 0.5 * (sigma / distance) ** 7) / distance
            force = force_magnitude * direction
            forces[i] -= force
            forces[j] += force
    return forces

def attractive_lennard_jones_force(positions, epsilon, sigma, box_size):
    forces = np.zeros_like(positions)
    # Calculate attractive Lennard-Jones forces between particles separated by more than one spacer particle
    for i in range(n_particles):
        for j in range(i + 3, n_particles):  # Only consider particles separated by more than one spacer
            displacement = positions[j] - positions[i]
            displacement = displacement - box_size * np.round(displacement / box_size)  # Apply minimum image convention
            distance = np.linalg.norm(displacement)
            if distance < 3.0 * sigma:  # Apply a cutoff for efficiency
                direction = displacement / distance
                force_magnitude = 24 * epsilon * ((sigma / distance) ** 13 - 0.5 * (sigma / distance) ** 7) / distance
                force = force_magnitude * direction
                forces[i] -= force
                forces[j] += force
    return forces

# Lists to store particle trajectories, energies, and bond lengths
trajectories = [[] for _ in range(n_particles)]
kinetic_energies = []
potential_energies = []
temperatures = []
bond_lengths = [[] for _ in range(n_particles - 1)]

# Simulation loop
for step in range(total_steps):
    # Calculate the harmonic forces between adjacent particles
    harmonic_forces = harmonic_force(positions, k, r0, box_size)

    # Calculate the repulsive Lennard-Jones forces between particles separated by one spacer
    repulsive_forces = repulsive_lennard_jones_force(positions, epsilon_repulsive, sigma, box_size)

    # Calculate the attractive Lennard-Jones forces between particles separated by more than one spacer
    attractive_forces = attractive_lennard_jones_force(positions, epsilon_attractive, sigma, box_size)

    # Total forces acting on each particle
    forces = harmonic_forces + repulsive_forces + attractive_forces

    # Update velocities using velocity Verlet algorithm
    velocities += 0.5 * forces / mass * dt

    # Update positions
    positions += velocities * dt

    # Apply periodic boundary conditions
    positions = apply_pbc(positions, box_size)

    # Update velocities again (full time step)
    velocities += 0.5 * forces / mass * dt

    # Store positions for visualization
    for i in range(n_particles):
        trajectories[i].append(positions[i].copy())

    # Calculate kinetic energy
    kinetic_energy = 0.5 * mass * np.sum(np.linalg.norm(velocities, axis=1) ** 2)
    kinetic_energies.append(kinetic_energy)

    # Calculate potential energy (harmonic + Lennard-Jones)
    potential_energy = 0.0
    # Harmonic potential energy
    for i in range(n_particles - 1):
        displacement = positions[i + 1] - positions[i]
        displacement = displacement - box_size * np.round(displacement / box_size)  # Apply minimum image convention
        distance = np.linalg.norm(displacement)
        potential_energy += 0.5 * k * (distance - r0) ** 2
        bond_lengths[i].append(distance)
    # Repulsive Lennard-Jones potential energy
    for i in range(n_particles - 2):
        j = i + 2
        displacement = positions[j] - positions[i]
        displacement = displacement - box_size * np.round(displacement / box_size)  # Apply minimum image convention
        distance = np.linalg.norm(displacement)
        if distance < 3.0 * sigma:  # Apply a cutoff for efficiency
            potential_energy += 4 * epsilon_repulsive * ((sigma / distance) ** 12 - (sigma / distance) ** 6)
    # Attractive Lennard-Jones potential energy
    for i in range(n_particles):
        for j in range(i + 3, n_particles):  # Only consider particles separated by more than one spacer
            displacement = positions[j] - positions[i]
            displacement = displacement - box_size * np.round(displacement / box_size)  # Apply minimum image convention
            distance = np.linalg.norm(displacement)
            if distance < 3.0 * sigma:  # Apply a cutoff for efficiency
                potential_energy += 4 * epsilon_attractive * ((sigma / distance) ** 12 - (sigma / distance) ** 6)
    potential_energies.append(potential_energy)

    # Calculate temperature (using kinetic energy)
    temperature = (2.0 / 3.0) * (kinetic_energy / (n_particles * mass))
    temperatures.append(temperature)

    # Velocity rescaling to thermostat the system at specified intervals
    if step % rescale_interval == 0:
        scaling_factor = np.sqrt(target_temperature / temperature)
        velocities *= scaling_factor

# Convert trajectories to numpy arrays for easier plotting
trajectories = [np.array(trajectory) for trajectory in trajectories]

# Function to convert trajectories to ASE Atoms objects
def trajectories_to_ase_atoms(trajectories, box_size):
    atoms_list = []
    for step in range(len(trajectories[0])):
        positions = np.array([trajectories[i][step] for i in range(n_particles)])
        atoms = Atoms(positions=positions, symbols="H" * n_particles, cell=[box_size, box_size, box_size], pbc=True)
        atoms_list.append(atoms)
    return atoms_list

# Convert the trajectories to ASE Atoms objects
atoms_trajectory = trajectories_to_ase_atoms(trajectories, box_size)

# Plot the trajectories
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i in range(n_particles):
    ax.plot(trajectories[i][:, 0], trajectories[i][:, 1], trajectories[i][:, 2], label=f'Particle {i+1}')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Trajectories of Particles in a Harmonic Chain with LJ Potential')
ax.legend()
plt.savefig('trajectories_chain_lj.png')

# Plot kinetic, potential energies, and temperature
plt.figure()
plt.plot(kinetic_energies, label='Kinetic Energy')
plt.plot(potential_energies, label='Potential Energy')
plt.xlabel('Time Step')
plt.ylabel('Energy')
plt.title('Kinetic and Potential Energy vs Time')
plt.legend()
plt.savefig('energy_vs_time_chain_lj.png')

plt.figure()
plt.plot(temperatures, label='Temperature')
plt.xlabel('Time Step')
plt.ylabel('Temperature')
plt.title('Temperature vs Time')
plt.legend()
plt.savefig('temperature_vs_time_chain_lj.png')

# Plot bond lengths vs time
plt.figure()
for i in range(n_particles - 1):
    plt.plot(bond_lengths[i], label=f'Bond Length {i+1}-{i+2}')
plt.xlabel('Time Step')
plt.ylabel('Bond Length')
plt.title('Bond Lengths vs Time')
plt.legend()
plt.savefig('bond_length_vs_time_chain_lj.png')

# # Visualize the trajectory using ASE's view function
# from ase.visualize import view
# view(atoms_trajectory)

# Save the trajectory to a file for visualization in VMD
from ase.io import write
write('trajectory_chain_lj.xyz', atoms_trajectory)
