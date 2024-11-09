import numpy as np
import matplotlib.pyplot as plt
from wind_field import generate_wind_field, plot_wind_field, generate_wind_field_uniform
import matplotlib.animation as animation


T_MAX = 30000
C_d = 0.3

rho = 0.5
dt = 1

#define ellipsoid shape
a = 100
b = 65
A_yz = np.pi * (b ** 2)
A_zx = 3 * np.pi * a * b

A = 2

m = 613000
v0 = np.array([5,5])
vmax = 30
r0 = np.array([200, 200])
goal = np.array([800, 800])

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
        self.direction = np.arctan2(y, x)

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
    u_dot_wind = np.dot(U_hat, wind_v)

    D_yz = 0.5 * rho * C_d * A_yz * (np.dot(T_hat, wind_v)**2 - np.dot(T_hat, ship_v)**2) * T_hat
    if u_dot_wind>=0:
        D_zx = 0.5 * rho * C_d * A_zx * (u_dot_wind**2) * U_hat
    else:
        D_zx = 0.5 * rho * C_d * A_zx * (u_dot_wind**2) * (-1 * U_hat)
    D_t = D_yz + D_zx

    return D_t


def calculate_drag(v):
    '''
    calculates drag force due to fluid opposing motion
    '''
    D = 0.5 * rho * v * np.linalg.norm(v) * C_d * A
    return D

def find_wind_v_constant(X, Y, ship, phi):
    '''
    Finds the velocity of a wind field at a set of coordinates. Splits up the vector field into squares of constant velocity.
    Returns empty array if the requested point is outside the vector field'''
    
    
    
    
    
    i, j = int(np.round(ship.coords[0])), int(np.round(ship.coords[1]))
    valid_coord = False
    if 0<=i<=(goal[0]+500) and 0<=j<=(goal[1]+500):
        ship2 = Point(ship.coords, m, ship.speed)
        ship2.move(dt*phi)        
        coords2 = ship2.coords
        i2, j2 = int(np.round(coords2[0])), int(np.round(coords2[1]))
        try:
            v = np.array([X[i2%1000, j2%1000], Y[i2%1000, j2%1000]])
            return v
            
            
        except IndexError:
            
            return np.array([])
    else:
        
        return np.array([])
    
    
            
    
    # if method == "reactive": 
    #     i, j = int(np.round(coords[0])), int(np.round(coords[1]))
    # if method == "proactive":
    #     ship.move(dt)
    #     coords2 = ship.coords
    #     i, j = int(np.round(coords2[0])), int(np.round(coords2[1]))
    #     ship.coords = coords
    
    # if i >= 0 and j >= 0:
    #     try:
    #         v = np.array([X[i, j], Y[i, j]])

    #         return v
    #     except IndexError:
    #         goal_reached = False
    #         return np.array([])
    # else:
    #     goal_reached = False
    #     return np.array([])

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
    
    X, Y = generate_wind_field()
    #plot_wind_field(X, Y)
    if fig:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10), tight_layout=True)
    else:
        ax = None

    return ship, goal, X, Y, ax

def simulation_main(ship, X, Y, phi):
    '''
    runs the simulation
    '''
    positions = [[ship.coords[0]], [ship.coords[1]]]
    t_ns = []
    t_trajs = []
    d_ns = []
    d_trajs = []
    T = normalize(goal - ship.coords)
    goal_reached=True
    while np.linalg.norm(goal - ship.coords) > 100:

    #for i in range(1000):        
        wind_v = find_wind_v_constant(X, Y, ship, 0)
        wind_v_pred = find_wind_v_constant(X, Y, ship, phi)
        if wind_v.any(): 
            
            
            
            # drag_v = calculate_drag(ship.speed) * -1
            # drag_wind = calculate_drag(wind_v)
            # drag_total = np.add(drag_v, drag_wind)

            
            # target_traj = trajectory(ship.coords, goal)
            # if np.linalg.norm(ship.speed) < vmax:

            #     T = T_MAX * normalize(np.subtract(goal,ship.coords))  
                
            # else: 

            #     dir_v = vector(ship.speed[0], ship.speed[1]).direction
            #     T = np.sign(target_traj - dir_v) * normalize(perpendicular(ship.speed)) * (T_MAX/100)
            #     plt.scatter(*ship.coords, color = "r")
            
            traj = normalize(goal - ship.coords)
            n = perpendicular(traj)
            

            drag_total = calculate_drag_geometric(wind_v, ship.speed, T)
            drag_total_pred = calculate_drag_geometric(wind_v_pred, ship.speed, T)
            if np.dot(drag_total_pred, n)<0:
                n= -1*n
            drag_traj = np.dot(drag_total, traj) * traj
            drag_n = np.dot(drag_total_pred, n) * n
            
            mag_drag_n = np.linalg.norm(drag_n)
            
            if mag_drag_n<=T_MAX:
                T_n = mag_drag_n * (-1) * n
            else:
                T_n = T_MAX * (-1) * n
            
            if np.linalg.norm(ship.speed)<vmax:
                T_traj = T_MAX - np.linalg.norm(T_n) * traj
            else:
                T_traj = np.zeros(2)
            T = T_n + T_traj
            t_ns.append(np.linalg.norm(T_n)*np.sign(np.dot(T_n, n)))
            t_trajs.append(np.linalg.norm(T_traj)*np.sign(np.dot(T_traj, traj)))
            d_ns.append(np.linalg.norm(drag_n)*np.sign(np.dot(drag_n, n)))
            d_trajs.append(np.linalg.norm(drag_traj)*np.sign(np.dot(drag_traj, traj)))

            F_total = np.add(drag_total, T)
            
            ship.accinc(F_total)
            ship.accelerate(dt)
            
            ship.move(dt)
            ship.clean_acc()

            

            for y in range(2):

                positions[y].append(ship.coords[y])

        else:
            goal_reached=False
            break
    fig2, (ax1,ax2) = plt.subplots(1,2, figsize=(10, 10), tight_layout=True)
    ns = np.linspace(0, len(t_ns)-1, len(t_ns))
    ax1.plot(ns, t_ns, label = 't_n')
    ax1.plot(ns, d_ns, label = 'd_n')
    ax1.legend()

    ax2.plot(ns, t_trajs, label = 't_traj')
    ax2.plot(ns, d_trajs, label = 'd_traj')
    ax2.legend()
    return positions, goal_reached


def simulation_plot(ax, X, Y, positions):
    '''
    plots simulation results
    '''
    #ax.quiver(X, Y, units = 'xy', minlength = 0)       
    ax.plot(*positions)
    ax.set_axis_off()
    ax.plot(*goal, "go")

def simulation(phi, fig=None):
    '''
    uses a proactive wind finding strategy to navigate to target
    '''
    ship, goal, X, Y, ax = init_sim(fig)
    positions, goal_reached = simulation_main(ship, X, Y, phi)
    
    return ax, X, Y, np.array(positions), goal_reached
    



def evaluate_sim(phi, iter = 5):
    dist_list = []
    time_list = []
    goal_count = 0
    while goal_count<=iter:
        
        ax, X, Y, pos, goal_reached = simulation(phi, fig = False)
        
        pos = pos.T
        
        dist = 0
        pos_prev = pos[0]
        for position in pos[1:]:
            dist += np.linalg.norm(np.array(position) - np.array(pos_prev))
            pos_prev = position
        if goal_reached:
            dist_list.append(dist)
            time_list.append(len(pos))
        goal_count += int(goal_reached)
        print(goal_reached)
        plt.close()
    print('gcount', goal_count, phi)
    # test line plot
    # simulation_plot(ax, X, Y, pos, goal)
    # xs = np.linspace(0, 1000, 1000)
    # ys = [m * x + c for x in xs]
    # ax.plot(xs, ys)

    return dist_list, time_list

def validate_geometry():

    
    
    
    #plot_wind_field(X, Y)
    
    fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(15, 10), tight_layout=True)
    vs = []
    r0 = [10,500]    
    v0 = [0,0]
    
    ship = Point(r0, m, v0)
    positions = [[ship.coords[0]], [ship.coords[1]]]

    T = [0,1]

    X, Y = generate_wind_field_uniform(0)
    for i in range(1000):   
        vs.append(np.linalg.norm(ship.speed))   
        wind_v = find_wind_v_constant(X, Y, ship, 0)
        
        if wind_v.any(): 
            
            
            
            # drag_v = calculate_drag(ship.speed) * -1
            # drag_wind = calculate_drag(wind_v)
            # drag_total = np.add(drag_v, drag_wind)

            

            
            drag_total = calculate_drag_geometric(wind_v, ship.speed, T)
            
            F_total = drag_total
            
            ship.accinc(F_total)
            ship.accelerate(dt)
            
            ship.move(dt)
            ship.clean_acc()

            

            for y in range(2):

                positions[y].append(ship.coords[y])

        else:
            break
    xx = np.linspace(0, len(vs)-1, len(vs))
    ax1.plot(xx, vs)
    # cs = lambda v: v/np.max(vs)
    
    # ax1.quiver(X, Y, units = 'xy', minlength = 0) 
    # ax1.set_axis_off()      
    # im = ax1.scatter(*positions, c = [cs(v) for v in vs])
    # print(vs[-1])
    


    
    r0 = [500,10]    
    v0 = [0,0]
    
    ship = Point(r0, m, v0)
    positions = [[ship.coords[0]], [ship.coords[1]]]

    vs2 = []
    X, Y = generate_wind_field_uniform(np.pi/2)
    for i in range(1000):   
        vs2.append(np.linalg.norm(ship.speed))   
        wind_v = find_wind_v_constant(X, Y, ship, 'reactive')
        
        if wind_v.any(): 
            
            
            
            # drag_v = calculate_drag(ship.speed) * -1
            # drag_wind = calculate_drag(wind_v)
            # drag_total = np.add(drag_v, drag_wind)

            

            
            drag_total = calculate_drag_geometric(wind_v, ship.speed, T)
            
            F_total = drag_total
            
            ship.accinc(F_total)
            ship.accelerate(dt)
            
            ship.move(dt)
            ship.clean_acc()

            

            for y in range(2):

                positions[y].append(ship.coords[y])

        else:
            break
    ax1.plot(xx, vs2[:len(xx)], label = r'$\phi=\frac{\pi}{2}')
    ax1.legend()
    # ax2.quiver(X, Y, units = 'xy', minlength = 0) 
    # ax2.set_axis_off()      
    # im = ax2.scatter(*positions, c = [cs(v) for v in vs2])
    # cax = fig.add_axes([0.94, 0.1, 0.05, 0.75])
    # fig.colorbar(im, cax = cax)
     
    # FIX COLORBAR

def figures(dists, times, phis, ax1, ax2):

    ax1.plot(phis, dists)
    ax1.set_ylabel(r'$d$ (m)')
    ax1.set_xlabel(r'$\phi$')
    ax1.set_title(r'$d$ against $\phi$')
    

    ax2.plot(phis, times)
    ax2.set_ylabel(r'$t_{total}$ (m)')
    ax2.set_xlabel(r'$\phi$')
    ax2.set_title(r'$t$ against $\phi$')
    

def remove_outliers(array, alpha = 2):

    med = np.median(array)
    std = np.std(array)

    array2 = np.array(array)[np.where((array<=(med+alpha*std)) & (array>=(med-alpha*std)))]
    return (array2)


n_phi = 900
def main():
    #validate_geometry()
    # plot_args = simulation(100, fig = True)
    # simulation_plot(*plot_args[:-1])

    #reactive = evaluate_sim(1)
    #proactive = evaluate_sim("reactive")

    phis = np.linspace(-150, 175,3)
    d_opt = np.linalg.norm(goal - r0) - 10
    k = len(phis)
    
    dists = np.zeros(k)
    #times = np.zeros(k)
    for i in range(k):
        dist_list, time_list = evaluate_sim(int(phis[i]), iter = 10)
        
        #time = np.mean(remove_outliers(time_list))
        dist = np.mean(remove_outliers(dist_list))
        dists[i] = dist - d_opt
        #times[i] = time
    
    fig, ax1 = plt.subplots(1, 1, figsize=(30, 10), tight_layout=True)
    #figures(dists, times, phis, ax1, ax2)
    deg = 6
    y_model = np.polyfit(phis, dists, deg)
    y_val = np.polyval(y_model, phis)
    ax1.plot(phis, dists, label = f'Data')
    ax1.plot(phis, y_val, label = f'Fitted polynomial of degree {deg}')
    ax1.legend()
    ax1.set_ylabel(r'$d-d_s$ (m)')
    ax1.set_xlabel(r'$\phi$')
    ax1.set_title(r'$d$ against $\phi$')
    plt.savefig('phi.svg')
    #plt.scatter(proactive[0], proactive[1], c = "g")
    

if __name__ == "__main__":
    main()