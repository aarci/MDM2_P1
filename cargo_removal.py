import numpy as np
import matplotlib.pyplot as plt


def calculate_ballonet_volume(container_mass, air_density=1.225, helium_density=0.1786):
    """
    Calculate total ballonet volume needed to compensate for container removal

    Returns:
    - total_volume: Total volume needed for complete compensation
    - half_volume: Volume needed for initial descent
    """
    # Total volume needed to compensate for container mass
    total_volume = container_mass / (air_density - helium_density)
    # Half volume for initial descent
    half_volume = total_volume / 2
    return total_volume, half_volume


def generate_cycle_data(start_time, container_mass, total_system_mass, flow_rate=230):
    """
    Generate displacement data for one cargo removal cycle with controlled descent and ascent
    """
    total_volume, half_volume = calculate_ballonet_volume(container_mass)

    # Calculate phase timings
    initial_fill_time = half_volume / flow_rate  # Time to fill first half
    container_removal_time = 2  # Time to remove container
    ascent_time = 3  # Time for ascent
    final_fill_time = half_volume / flow_rate  # Time to fill second half

    dt = 0.1

    # Generate time points for each phase
    t1 = np.arange(0, initial_fill_time, dt)  # Initial descent
    t2 = np.arange(0, container_removal_time, dt)  # Container removal
    t3 = np.arange(0, ascent_time, dt)  # Ascent
    t4 = np.arange(0, final_fill_time, dt)  # Return to equilibrium

    # Calculate mass factor for scaling displacement
    mass_factor = container_mass / total_system_mass
    max_displacement = 15 * mass_factor  # Maximum displacement amplitude

    # Calculate displacements for each phase
    # Phase 1: Controlled descent as first half of air enters
    d1 = -max_displacement * (t1 / initial_fill_time)

    # Phase 2: Hold position during container removal
    d2 = np.full_like(t2, -max_displacement)

    # Phase 3: Quick ascent after container removal
    d3 = max_displacement * np.sin(t3 * np.pi / (2 * ascent_time))

    # Phase 4: Controlled return to equilibrium as second half of air enters
    d4 = max_displacement * (1 - t4 / final_fill_time)

    # Combine all phases
    time = np.concatenate([
        t1 + start_time,
        t2 + start_time + initial_fill_time,
        t3 + start_time + initial_fill_time + container_removal_time,
        t4 + start_time + initial_fill_time + container_removal_time + ascent_time
    ])

    displacement = np.concatenate([d1, d2, d3, d4])

    return time, displacement


def plot_cycles(num_cycles=10, initial_container_mass=3707, initial_system_mass=400000):
    """
    Plot multiple cycles of the airship operation
    """
    time_arrays = []
    displacement_arrays = []
    current_system_mass = initial_system_mass

    for i in range(num_cycles):
        # Calculate cycle timing
        total_volume, _ = calculate_ballonet_volume(initial_container_mass)
        cycle_time = (total_volume / 230) + 5  # Total fill time + handling time

        # Generate cycle data
        t, d = generate_cycle_data(
            start_time=i * cycle_time,
            container_mass=initial_container_mass,
            total_system_mass=current_system_mass
        )

        time_arrays.append(t)
        displacement_arrays.append(d * 100)

        # Update system mass for next cycle
        current_system_mass -= initial_container_mass

    # Combine all cycles
    time = np.concatenate(time_arrays)
    displacement = np.concatenate(displacement_arrays)

    # Plot displacement
    plt.figure(figsize=(15, 8))
    plt.plot(time, displacement, 'b-', linewidth=2)
    plt.title('Airship Displacement During Cargo Removal Cycles', fontsize=14)
    plt.xlabel('Time (seconds)', fontsize=12)
    plt.ylabel('Displacement (meters)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

    # Print volume calculations for first cycle
    total_vol, half_vol = calculate_ballonet_volume(initial_container_mass)
    print(f"\nVolume calculations for first container removal:")
    print(f"Container mass: {initial_container_mass:.2f} kg")
    print(f"Total ballonet volume needed: {total_vol:.2f} m³")
    print(f"Volume for initial descent (half): {half_vol:.2f} m³")
    print(f"Time to fill half volume at {230} m³/s: {half_vol / 230:.2f} seconds")


# Run simulation
plot_cycles(10)
plt.savefig('cargo_removal.png')