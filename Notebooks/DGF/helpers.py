import numpy as np                  #type: ignore
import matplotlib.pyplot as plt     #type: ignore
from matplotlib import patches      #type: ignore

class DGFsimulation:
    def __init__(
        self, 
        amplitude_range = np.array([0.0, 0.5]), 
        frequency_range = np.array([0.5, 2*np.pi]), 
        nx = 50, 
        ny = 25, 
        Lx = 2.0, 
        Ly = 1.0, 
        dt = 0.1, 
        T  = 10.0, 
        ntrajectories = 100
    ):
        # amplitude and frequency ranges for the double gyre flow
        self.amplitude_range = amplitude_range
        self.frequency_range = frequency_range

        # spatial discretization
        self.nx = nx
        self.ny = ny
        self.Lx = Lx
        self.Ly = Ly
        self.x = np.linspace(0, self.Lx, self.nx)
        self.y = np.linspace(0, self.Ly, self.ny)
        self.nstate = len(self.x) * len(self.y)

        # temporal discretization
        self.dt = dt
        self.T = T
        self.t = np.arange(0, self.T + self.dt, self.dt)
        self.ntimes = len(self.t)

        # training data generation
        self.ntrajectories = ntrajectories
        self.U = np.zeros((self.ntrajectories, self.ntimes, self.nx, self.ny))
        self.V = np.zeros((self.ntrajectories, self.ntimes, self.nx, self.ny))

        # others
        self.sensor_indices = None
        self.Ureshaped = None
        self.Vreshaped = None


    def generate_data(self):
        # Generate the double gyre flow data for each trajectory with random amplitude and frequency
        for i in range(self.ntrajectories):
            amplitude = np.random.uniform(self.amplitude_range[0], self.amplitude_range[1])
            frequency = np.random.uniform(self.frequency_range[0], self.frequency_range[1])
            self.U[i], self.V[i] = self.double_gyre_flow(amplitude, frequency, self.x, self.y, self.t)

        self.Ureshaped = self.U.reshape(self.ntrajectories, self.ntimes, self.nstate)
        self.Vreshaped = self.V.reshape(self.ntrajectories, self.ntimes, self.nstate)
        
        return self.U, self.V
    
    def generate_measurements_U(self, nsensors = 3):
        
        U = self.U.reshape(self.ntrajectories, self.ntimes, self.nstate)
        self.sensor_indices = np.random.choice(self.nstate, nsensors, replace = False)
        U_measurements = U[:, :, self.sensor_indices]

        return U_measurements
    
    def get_reshaped_data(self):
        # Reshape the velocity fields into the desired format for training
        if self.Ureshaped is None or self.Vreshaped is None:
            self.Ureshaped = self.U.reshape(self.ntrajectories, self.ntimes, self.nstate)
            self.Vreshaped = self.V.reshape(self.ntrajectories, self.ntimes, self.nstate)
        return self.Ureshaped, self.Vreshaped

    def double_gyre_flow(self, amplitude, frequency, x, y, t, intensity = 0.1):
        # Create spatial and temporal grids 
    
        xgrid, ygrid = np.meshgrid(x, y)         # spatial grid
        u = np.zeros((len(t), len(x), len(y)))   # horizontal velocity
        v = np.zeros((len(t), len(x), len(y)))   # vertical velocity
        f = lambda x,t: amplitude * np.sin(frequency * t) * x**2 + x - 2 * amplitude * np.sin(frequency * t) * x

        # compute solution
        for i in range(len(t)):
            u[i] = (-np.pi * intensity * np.sin(np.pi * f(xgrid, t[i])) * np.cos(np.pi * ygrid)).T
            v[i] = (np.pi * intensity * np.cos(np.pi * f(xgrid, t[i])) * np.sin(np.pi * ygrid) * (2 * amplitude * np.sin(frequency * t[i]) * xgrid + 1.0 - 2 * amplitude * np.sin(frequency * t[i]))).T

        return u, v
    
    def vorticity(self, u, v):
        # Compute the vorticity of the flow field given the horizontal and vertical velocity components
        dx = self.Lx / self.nx
        dy = self.Ly / self.ny
        du_dy = np.gradient(u, dy, axis = 1)
        dv_dx = np.gradient(v, dx, axis = 0)
        return dv_dx - du_dy

    def plot_data(self, which_trajectory, which_time=-1):
        offset = 0.1
        plt.figure(figsize = (10,5))
        plt.contourf(self.x, self.y, self.vorticity(self.U[which_trajectory, which_time], self.V[which_trajectory, which_time]).T, cmap = 'seismic', levels = 100)
        plt.streamplot(self.x, self.y, self.U[which_trajectory, which_time].T, self.V[which_trajectory, which_time].T, color='black', linewidth = 1, density = 1)
        plt.axis('off')
        plt.axis([0 - offset, self.Lx + offset, 0 - offset, self.Ly + offset])
        plt.title(f'Trajectory {which_trajectory} at time t = {round(self.t[which_time], 3)}')
        plt.grid(True)
        plt.gca().add_patch(patches.Rectangle((0, 0), self.Lx, self.Ly, linewidth = 5, edgecolor = 'black', facecolor = 'none'))

        # Plot measurement sites if available
        if self.sensor_indices is not None:
            idx_sensors_x, idx_sensors_y = np.unravel_index(self.sensor_indices, (self.nx, self.ny))
            sensors_x = self.x[idx_sensors_x]
            sensors_y = self.y[idx_sensors_y]
            plt.plot(sensors_x, sensors_y, 'o', mfc='magenta', mec='black', ms=8, mew=1.5, label='Measurement Sites')
            plt.legend(loc='upper right')

        plt.show()

   