import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mp
import matplotlib.animation as animation
import IPython

class PlanarQuad:
    def __init__(self, dt=0.01, mass=0.6, length=0.2, inertia=0.15, gravity=9.81):
        self.m = mass
        self.r = length
        self.I = inertia
        self.g = gravity
        self.dt = dt
       
    def linear_dyn(self, z, u):
        A = np.array([[1., self.dt, 0., 0., 0., 0.],
                      [0., 1., 0., 0., (-(u[0] + u[1]) * self.dt * np.cos(z[4])) / self.m, 0.],
                      [0., 0., 1., self.dt, 0., 0.],
                      [0., 0., 0., 1., (-(u[0] + u[1]) * self.dt * np.sin(z[4])) / self.m, 0.],
                      [0., 0., 0., 0., 1., self.dt],
                      [0., 0., 0., 0., 0., 1.]])
        B = np.array([[0., 0.],
                      [-(self.dt * np.sin(z[4])) / self.m, -(self.dt * np.sin(z[4])) / self.m],
                      [0., 0.],
                      [(self.dt * np.cos(z[4])) / self.m, (self.dt * np.cos(z[4])) / self.m],
                      [0., 0.],
                      [self.r * self.dt / self.I, -self.r * self.dt / self.I]])
        return A, B
    
    def get_next_state(self,z,u):
        """
        Inputs:
        z: state of the quadrotor as a numpy array (x, vx, y, vy, theta, omega)
        u: control as a numpy array (u1, u2)

        Output:
        the new state of the quadrotor as a numpy array
        """
        x = z[0]
        vx = z[1]
        y = z[2]
        vy = z[3]
        theta = z[4]
        omega = z[5]
        dydt = np.zeros([6,])
        dydt[0] = vx
        dydt[1] = (-(u[0] + u[1]) * np.sin(theta)) / self.m
        dydt[2] = vy
        dydt[3] = ((u[0] + u[1]) * np.cos(theta) - self.m * self.g) / self.m
        dydt[4] = omega
        dydt[5] = (self.r * (u[0] - u[1])) / self.I
        z_next = z + dydt * self.dt

        return z_next
    
    
    def animate(self, x, u, goal, dt = 0.01):
        """
        This function makes an animation showing the behavior of the quadrotor
        takes as input the result of a simulation (with dt=0.01s)
        """

        min_dt = 0.1
        if(dt < min_dt):
            steps = int(min_dt/dt)
            use_dt = int(np.round(min_dt * 1000))
        else:
            steps = 1
            use_dt = int(np.round(dt * 1000))

        #what we need to plot
        plotx = x[:,::steps]
        plotx = plotx[:,:-1]
        plotu = u[:,::steps]

        fig = mp.figure.Figure(figsize=[8.5,8.5])
        mp.backends.backend_agg.FigureCanvasAgg(fig)
        ax = fig.add_subplot(111, autoscale_on=False, xlim=[-5,5], ylim=[-5,5])
        
        ax.grid()

        list_of_lines = []

        #create the robot
        # the main frame
        line, = ax.plot([], [], 'k', lw=6)
        list_of_lines.append(line)
        # the left propeller
        line, = ax.plot([], [], 'b', lw=4)
        list_of_lines.append(line)
        # the right propeller
        line, = ax.plot([], [], 'b', lw=4)
        list_of_lines.append(line)
        # the left thrust
        line, = ax.plot([], [], 'r', lw=1)
        list_of_lines.append(line)
        # the right thrust
        line, = ax.plot([], [], 'r', lw=1)
        list_of_lines.append(line)
        # Goal line
        line, = ax.plot([], [], '.r', marker='x', markersize=10)
        list_of_lines.append(line)
        def _animate(i):
            for l in list_of_lines: #reset all lines
                l.set_data([],[])

            theta = plotx[4,i]
            x = plotx[0,i]
            y = plotx[2,i]
            trans = np.array([[x,x],[y,y]])
            rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

            main_frame = np.array([[-self.r, self.r], [0,0]])
            main_frame = rot @ main_frame + trans 

            left_propeller = np.array([[-1.3 * self.r, -0.7*self.r], [0.1,0.1]])
            left_propeller = rot @ left_propeller + trans

            right_propeller = np.array([[1.3 * self.r, 0.7*self.r], [0.1,0.1]])
            right_propeller = rot @ right_propeller + trans


            left_thrust = np.array([[self.r, self.r], [0.1, 0.1+plotu[0,i]*0.04]])
            left_thrust = rot @ left_thrust + trans

            right_thrust = np.array([[-self.r, -self.r], [0.1, 0.1+plotu[0,i]*0.04]])
            right_thrust = rot @ right_thrust + trans

            list_of_lines[0].set_data(main_frame[0,:], main_frame[1,:])
            list_of_lines[1].set_data(left_propeller[0,:], left_propeller[1,:])
            list_of_lines[2].set_data(right_propeller[0,:], right_propeller[1,:])
            list_of_lines[3].set_data(left_thrust[0,:], left_thrust[1,:])
            list_of_lines[4].set_data(right_thrust[0,:], right_thrust[1,:])
            list_of_lines[5].set_data(goal[0], goal[1])
            return list_of_lines

        def _init():
            return _animate(0)

        ani = animation.FuncAnimation(fig, _animate, np.arange(0, len(plotx[0,:])),
            interval=use_dt, blit=True, init_func=_init)
        plt.close(fig)
        plt.close(ani._fig)
        IPython.display.display_html(IPython.core.display.HTML(ani.to_html5_video()))