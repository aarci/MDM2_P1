import numpy as np
import matplotlib.pyplot as plt
import math

L = 300
W = 100
rho_air_o = 1.225
rho_helium_o = 0.1785
V_max = 4/3 * np.pi * L/2 * W/2 * W/2
H_mass = V_max * rho_helium_o

# Constants
alpha = 0.00012   # height coefficient (1/m)
p_0 = 101325      # sea level pressure (Pa) - note: changed from kPa
T_0 = 288.15      # sea level temperature (K)
R = 287.05        # specific gas constant for air (J/kg·K)
l = 0.0065        # temperature lapse rate (K/m)
g = 9.81          # acceleration due to gravity (m/s²)

def pressure(h):
    """
    Calculate atmospheric pressure at height h
    Returns pressure in Pa
    """
    return p_0 * math.exp(-alpha * h)

def rho_air(h):
    """
    Calculate air density accounting for both pressure and temperature changes with height
    Returns density in kg/m³
    """
    T_h = T_0 - l * h
    return pressure(h) / (R * T_h)

def rho_helium(h):
    """
    Calculate helium density at height h using pressure ratio
    Returns density in kg/m³
    """
    p_h = pressure(h)
    return rho_helium_o * (p_h / p_0)


def calculate_volumes(h):
    """
    Calculate helium and ballonet volumes at height h using Boyle's law
    Returns tuple of (V_helium, V_ballonet)
    """
    # Initial conditions (at sea level)
    V_helium_0 = 0.8 * V_max  # Start with 80% helium

    # Calculate expanded helium volume using Boyle's law
    p_h = pressure(h)
    V_helium = V_helium_0 * (p_0 / p_h)

    # Limit helium volume to maximum volume
    V_helium = min(V_helium, V_max)

    # Calculate ballonet volume
    V_ballonet = max(0, V_max - V_helium)

    return V_helium, V_ballonet


# plot helium and air desnity with alitude
h = np.linspace(0, 3000, 100)
rho_a = [rho_air(hi) for hi in h]
rho_h = [rho_helium(hi) for hi in h]
#plot these against h
plt.figure(figsize=(12, 8))
plt.plot(h, rho_a, label='Air Density')
plt.plot(h, rho_h, label='Helium Density')
plt.xlabel('Altitude (m)')
plt.ylabel('Density (kg/m³)')
plt.title('Density vs Altitude')
plt.legend()
plt.grid(True)
plt.show()
#add labels and titles






# V_lift = np.linspace(0, V_max, 100)
# V_ballonet = [V_max - V for V in V_lift]
#
#
#
# # Test the function
# h = np.linspace(0, 3000, 100)
# rho_a = [rho_air(hi) for hi in h]
#
# # Calculate the density ratio σ
# sigma = [rho / rho_air_o for rho in rho_a]
#
# # Calculate the density of Helium at altitude
# rho_h = [s * rho_helium_o for s in sigma]
# rho_h_mass = [H_mass / i for i in V_lift[:-1]]
# print(rho_h_mass)
#
#
#
#
#
# F_buoyancy_altitude = [rho_a[i] * V_ballonet[i] * 9.81 - rho_h[i] * V_lift[i] * 9.81 for i in range(len(h))]
# cargo = [5000] * len(h)
#
# F_down = [m*g + h for m, h in zip(cargo, rho_h_mass)]
# print(F_down)
#
# pres_alt = ((rho_a[99]-rho_h[99])*V_max)
# # print(pres_alt)
# # print(F_buoyancy_altitude)



# Calculate values across altitude range
h = np.linspace(0, 3000, 100)
rho_a = [rho_air(hi) for hi in h]
rho_h = [rho_helium(hi) for hi in h]

# Calculate volumes at each height
volumes = [calculate_volumes(hi) for hi in h]
V_helium = [v[0] for v in volumes]
V_ballonet = [v[1] for v in volumes]

# Calculate forces
F_buoyancy = [rho_a[i] * V_max * g for i in range(len(h))]
F_helium = [rho_h[i] * V_helium[i] * g for i in range(len(h))]
F_ballonet = [rho_a[i] * V_ballonet[i] * g for i in range(len(h))]
F_cargo = [5000 * g] * len(h)

# Calculate net force
F_net = [F_buoyancy[i] - (F_helium[i] + F_ballonet[i] + F_cargo[i]) for i in range(len(h))]



# Plotting
plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.plot(h, F_buoyancy, label='Buoyancy Force')
plt.plot(h, [-f for f in F_helium], label='Helium Weight')
plt.plot(h, [-f for f in F_ballonet], label='Ballonet Weight')
plt.plot(h, [-f for f in F_cargo], label='Cargo Weight')
plt.plot(h, F_net, '--', label='Net Force')
plt.xlabel('Altitude (m)')
plt.ylabel('Force (N)')
plt.title('Forces vs Altitude')
plt.grid(True)
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(h, V_helium, label='Helium Volume')
plt.plot(h, V_ballonet, label='Ballonet Volume')
plt.plot(h, [V_max] * len(h), '--', label='Total Volume')
plt.xlabel('Altitude (m)')
plt.ylabel('Volume (m³)')
plt.title('Volumes vs Altitude')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

# Find equilibrium points
equilibrium_heights = []
for i in range(len(h)-1):
    if (F_net[i] * F_net[i+1] <= 0):  # Sign change indicates crossing zero
        h1, h2 = h[i], h[i+1]
        f1, f2 = F_net[i], F_net[i+1]
        h_eq = h1 + (h2 - h1) * (-f1)/(f2 - f1)  # Linear interpolation
        equilibrium_heights.append(h_eq)

print("\nEquilibrium heights:", [f"{h:.1f}m" for h in equilibrium_heights])



















