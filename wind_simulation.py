import numpy as np
import matplotlib.pyplot as plt
from wind_field import generate_wind_field, plot_wind_field, generate_wind_field_uniform
import matplotlib.animation as animation


T_MAX = 250
C_d = 0.5

rho = 0.5
dt = 1

#define ellipsoid shape
a = 5
b = 5
A_yz = np.pi * (b ** 2)
A_zx = 3 * np.pi * a * b

A = 2

m = 1000
v0 = np.array([2,2])
vmax = 10
r0 = np.array([200, 200])

def perpendicular( a ) :
    '''
    find perpendicular vector to unit normal
    '''

    b = np.empty_like(a)
    b[0] = -a[1]
    b[1] = a[0]
    return b

def normalize(a):
    '''
    compute unit vector
    '''
    a = np.array(a)
    return a/np.linalg.norm(a)

class vector:

    '''
    useful for converting between different forms of a vector
    '''
    def __init__(self, x, y):
        self.magnitude = np.sqrt(x ** 2 + y **2)
        self.direction = np.arctan(y / x)

    def to_column(self):
        return np.array([self.magnitude * np.cos(self.direction), self.magnitude * np.sin(self.direction)])


class Point:
    '''
    used to define a ship's motion
    '''
    def __init__(self, coords, mass=1.0,  speed=None, **properties):
        self.coords = coords
        
        self.speed = speed
        self.acc = np.zeros(2)
        self.mass = mass
        self.__params__ = ["coords", "speed", "acc", "q"] + list(properties.keys())
        
        for prop in properties:
            setattr(self, prop, properties[prop])

    def move(self, dt):
        self.coords = self.coords + self.speed * dt

    def accelerate(self, dt):
        self.speed = self.speed + self.acc * dt

    def accinc(self, force):  
        self.acc = self.acc + force / self.mass

    def clean_acc(self):
        self.acc = self.acc * 0

def calculate_drag_geometric(wind_v, ship_v, T):

    T_hat = normalize(T)
    U_hat = normalize(perpendicular(T))
    v_diff = (np.linalg.norm(wind_v) * wind_v) - (np.linalg.norm(ship_v) * ship_v)

    D_yz = 0.5 * rho * A_yz * (np.dot(T_hat, v_diff)) * T_hat
    D_zx = 0.5 * rho * A_zx * (np.dot(U_hat, v_diff)) * U_hat
    
    D_t = D_yz + D_zx

    return D_t


def calculate_drag(v):
    '''
    calculates drag force due to fluid opposing motion
    '''
    D = 0.5 * rho * v * np.linalg.norm(v) * C_d * A
    return D

def find_wind_v_constant(X, Y, ship, method):
    '''
    Finds the velocity of a wind field at a set of coordinates. Splits up the vector field into squares of constant velocity.
    Returns empty array if the requested point is outside the vector field'''
    coords = ship.coords
    
    if method == "reactive": 
        i, j = int(np.round(coords[0])), int(np.round(coords[1]))
    if method == "proactive":
        ship.move(dt)
        coords2 = ship.coords
        i, j = int(np.round(coords2[0])), int(np.round(coords2[1]))
        ship.coords = coords
    
    if i >= 0 and j >= 0:
        try:
            v = np.array([X[i, j], Y[i, j]])

            return v
        except IndexError:
            return np.array([])
    else:
        return np.array([])

def find_wind_v_average(X, Y, x, y):

    # come back to this 

    return None

def trajectory(loc, target):
    '''
    Calculates the angle between the x axis and the ray between a position and its target position
    '''
    trajectory = np.arctan2((target[1] - loc[1]) , (target[0] - loc[0]))

    return trajectory

def init_sim(fig):
    '''
    Initialises the simulation
    '''
    ship = Point(r0, m, v0)
    goal = np.array([900, 900])
    X, Y = generate_wind_field()
    #plot_wind_field(X, Y)
    if fig:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10), tight_layout=True)
    else:
        ax = None

    return ship, goal, X, Y, ax

def simulation_main(ship, goal, X, Y, wind_method):
    '''
    runs the simulation
    '''
    positions = [[ship.coords[0]], [ship.coords[1]]]
    while np.linalg.norm(goal - ship.coords) > 20:
    #for i in range(1000):        
        wind_v = find_wind_v_constant(X, Y, ship, wind_method)
        
        if wind_v.any(): 
            
            target_traj = trajectory(ship.coords, goal)
            
            # drag_v = calculate_drag(ship.speed) * -1
            # drag_wind = calculate_drag(wind_v)
            # drag_total = np.add(drag_v, drag_wind)

            

            if np.linalg.norm(ship.speed) < vmax:

                T = np.array([T_MAX * np.cos(target_traj), T_MAX * np.sin(target_traj)])  
                
            else: 

                dir_v = vector(ship.speed[0], ship.speed[1]).direction
                T = np.sign(target_traj - dir_v) * perpendicular(ship.speed) * (T_MAX/10)
                plt.scatter(*ship.coords, color = "r")
            
            drag_total = calculate_drag_geometric(wind_v, ship.speed, T)
            
            F_total = np.add(drag_total, T)
            
            ship.accinc(F_total)
            ship.accelerate(dt)
            
            ship.move(dt)
            ship.clean_acc()

            

            for y in range(2):

                positions[y].append(ship.coords[y])

        else:
            break
    return positions


def simulation_plot(ax, X, Y, positions, goal):
    '''
    plots simulation results
    '''
    ax.quiver(X, Y, units = 'xy', minlength = 0)       
    ax.scatter(*positions)
    ax.plot(*goal, "go")

def simulation(method, fig):
    '''
    uses a proactive wind finding strategy to navigate to target
    '''
    ship, goal, X, Y, ax = init_sim(fig)
    positions = simulation_main(ship, goal, X, Y, method)
    
    return ax, X, Y, positions, goal
    



def evaluate_sim(method, iter = 100):
    dist_list = []
    time_list = []
    for i in range(100):

        ax, X, Y, pos, goal = simulation(method, fig = False)
        origin = pos[0]
        m = (goal[1] - origin[1]) / (goal[0] - origin[0])
        c = origin[1] - (m * origin[0])
        squares = 0

        for coords in pos:

            squares += (coords[1] - (coords[0] * m + c)) ** 2

        dist_list.append(squares)
        time_list.append(len(pos))
        plt.close()
    # test line plot
    # simulation_plot(ax, X, Y, pos, goal)
    # xs = np.linspace(0, 1000, 1000)
    # ys = [m * x + c for x in xs]
    # ax.plot(xs, ys)

    return dist_list, time_list

    

    
        

    


def main():
    
    plot_args = simulation("reactive")
    simulation_plot(*plot_args[0:-1])
    #reactive = evaluate_sim("reactive")
    #proactive = evaluate_sim("reactive")
    
    #plt.scatter(reactive[0], reactive[1], c = "r")
    #plt.scatter(proactive[0], proactive[1], c = "g")
    plt.show()

if __name__ == "__main__":
    main()