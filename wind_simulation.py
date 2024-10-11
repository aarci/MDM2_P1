import numpy as np
import matplotlib.pyplot as plt
from wind_field import generate_wind_field, plot_wind_field
import matplotlib.animation as animation


T_MAX = 15
C_d = 0.5
A = 4
rho = 0.5
dt = 1

drag_params = [rho, C_d, A]

m = 10
v0 = np.array([2,2])
r0 = np.array([200, 200])
class vector:
    def __init__(self, x, y):
        self.magnitude = np.sqrt(x ** 2 + y **2)
        self.direction = np.arctan(y / x)

    
    
    def to_column(self):
        return np.array([self.magnitude * np.cos(self.direction), self.magnitude * np.sin(self.direction)])


class Point:
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


def calculate_drag(v, rho, drag_coeff, area):
    
    D = 0.5 * rho * v * drag_coeff * area

    return D

def find_wind_v_constant(X, Y, x, y):

    i, j = int(np.round(x)), int(np.round(y))
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

    trajectory = np.arctan2((target[1] - loc[1]) , (target[0] - loc[0]))

    return trajectory

#def 

def simulation_reactive():
    plt.close()
    ship = Point(r0, m, v0)
    goal = np.array([900, 900])
    X, Y = generate_wind_field()
    #plot_wind_field(X, Y)
    fig, ax = plt.subplots(1, 1, figsize=(10, 10), tight_layout=True)
    ax.plot(*goal, "go")
    
    
    

    positions = [[ship.coords[0]], [ship.coords[1]]] 

    while np.linalg.norm(goal - ship.coords) > 20:
               
        wind_v = find_wind_v_constant(X, Y, *ship.coords)
        
        if wind_v.any(): 
            target_traj = trajectory(ship.coords, goal)
            T = np.array([T_MAX * np.cos(target_traj), T_MAX * np.sin(target_traj)])  
            
            drag_v = calculate_drag(ship.speed, *drag_params) * -1
            drag_wind = calculate_drag(wind_v, *drag_params)
            drag_total = np.add(drag_v, drag_wind)

            F_total = np.add(drag_total, T)
            
            ship.accinc(F_total)
            ship.accelerate(dt)
            
            ship.move(dt)
            ship.clean_acc()

            

            for y in range(2):

                positions[y].append(ship.coords[y])

        else:
            break
    ax.quiver(X, Y, units = 'xy', minlength = 0)       
    ax.scatter(*positions)
    
    

    
        

    


def main():
    
    simulation_reactive()

    plt.show()

if __name__ == "__main__":
    main()