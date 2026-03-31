
import numpy as np  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
from matplotlib import patches  # type: ignore


class DGFsimulation:

    def __init__(
        self,
        amplitude_range=np.array([0.0, 0.5]),
        frequency_range=np.array([0.5, 2 * np.pi]),
        nx=50,
        ny=25,
        Lx=2.0,
        Ly=1.0,
        dt=0.1,
        T=10.0,
        ntrajectories=100,
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

        permutation = np.random.permutation(self.ntrajectories)
        self.U = self.U[permutation]
        self.V = self.V[permutation]

        self.Ureshaped = self.U.reshape(self.ntrajectories, self.ntimes, self.nstate)
        self.Vreshaped = self.V.reshape(self.ntrajectories, self.ntimes, self.nstate)

        return self.U, self.V
    

    def generate_measurements_U(self, nsensors=3, idx_short_broken=None, idx_long_broken=None, p_short_broken=0.2, verbose=False):

        # Reshape the velocity fields and randomly select sensor locations, then simulate broken sensors by introducing NaN values
        U = self.U.reshape(self.ntrajectories, self.ntimes, self.nstate)
        self.sensor_indices = np.random.choice(self.nstate, nsensors, replace=False)
        U_measurements = U[:, :, self.sensor_indices].copy()

        # Get dimensions
        ntimes = U_measurements.shape[1]
        ntrajectories = U_measurements.shape[0]
        nsensors = U_measurements.shape[2]

        if idx_short_broken is None:
            idx_short_broken = []
        if idx_long_broken is None:
            idx_long_broken = []

        # Short broken sensors: random nans with probability p_short_broken
        for idx in idx_short_broken:
            mask = np.random.rand(ntrajectories, ntimes) < p_short_broken
            U_measurements[:, :, idx][mask] = np.nan

        # Long broken sensors: one contiguous nan interval per sensor
        for idx in idx_long_broken:
            for traj in range(ntrajectories):
                t1, t2 = np.random.randint(0, ntimes, size=2)
                t_start, t_end = min(t1, t2), max(t1, t2)
                if t_end - t_start < 30:
                    t_end = min(t_start + 38, ntimes-1)
                U_measurements[traj, t_start:t_end+1, idx] = np.nan

        if verbose:
            self.plot_data_availability(U_measurements, title_prefix="Sensor Data Availability")

        return U_measurements

    def plot_data_availability(self, sensor_data, title_prefix, trajectory=0, sensor_labels=None):
        traj_idx = min(trajectory, sensor_data.shape[0] - 1)
        availability = (~np.isnan(sensor_data[traj_idx])).astype(float)
        time = np.arange(availability.shape[0])
        nsensors_plot = availability.shape[1]

        if sensor_labels is None:
            sensor_labels = np.arange(nsensors_plot)
        sensor_labels = np.asarray(sensor_labels)

        if sensor_labels.shape[0] != nsensors_plot:
            raise ValueError("sensor_labels length must match number of sensor columns")

        fig_height = max(2.0, 0.6* nsensors_plot)
        _, ax = plt.subplots(1, 1, figsize=(14, fig_height))

        for sensor_idx in range(nsensors_plot):
            line = np.where(availability[:, sensor_idx] > 0.5, sensor_idx + 1.0, np.nan)
            ax.plot(time, line, linewidth=4, solid_capstyle="butt", label=f"S{sensor_labels[sensor_idx]}")

        ax.set_xlabel("Time step")
        ax.set_ylabel("Sensor")
        ax.set_yticks(np.arange(1, nsensors_plot + 1))
        ax.set_yticklabels([f"S{label}" for label in sensor_labels])
        ax.set_ylim(0.5, nsensors_plot + 0.5)
        ax.grid(True, axis="x", alpha=0.25)
        ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), title="Sensors")
        ax.set_title(f"{title_prefix} (Trajectory {traj_idx})")
        plt.show()
    
    def get_reshaped_data(self):
        # Reshape the velocity fields into the desired format for training
        if self.Ureshaped is None or self.Vreshaped is None:
            self.Ureshaped = self.U.reshape(self.ntrajectories, self.ntimes, self.nstate)
            self.Vreshaped = self.V.reshape(self.ntrajectories, self.ntimes, self.nstate)
        return self.Ureshaped, self.Vreshaped

    def double_gyre_flow(self, amplitude, frequency, x, y, t, intensity=0.1):
        # Create spatial and temporal grids
        xgrid, ygrid = np.meshgrid(x, y)  # spatial grid
        u = np.zeros((len(t), len(x), len(y)))  # horizontal velocity
        v = np.zeros((len(t), len(x), len(y)))  # vertical velocity
        f = lambda x, t: amplitude * np.sin(frequency * t) * x ** 2 + x - 2 * amplitude * np.sin(frequency * t) * x

        # compute solution
        for i in range(len(t)):
            u[i] = (
                -np.pi * intensity * np.sin(np.pi * f(xgrid, t[i])) * np.cos(np.pi * ygrid)
            ).T
            v[i] = (
                np.pi * intensity * np.cos(np.pi * f(xgrid, t[i])) * np.sin(np.pi * ygrid)
                * (2 * amplitude * np.sin(frequency * t[i]) * xgrid + 1.0 - 2 * amplitude * np.sin(frequency * t[i]))
            ).T
        return u, v
    
    def vorticity(self, u, v):
        # Compute the vorticity of the flow field given the horizontal and vertical velocity components
        dx = self.Lx / self.nx
        dy = self.Ly / self.ny
        du_dy = np.gradient(u, dy, axis=1)
        dv_dx = np.gradient(v, dx, axis=0)
        return dv_dx - du_dy

    def plot_data(self, which_trajectory, which_time=-1):
        offset = 0.1
        plt.figure(figsize=(10, 5))
        plt.contourf(
            self.x, self.y,
            self.vorticity(self.U[which_trajectory, which_time], self.V[which_trajectory, which_time]).T,
            cmap='seismic', levels=100
        )
        plt.streamplot(
            self.x, self.y,
            self.U[which_trajectory, which_time].T,
            self.V[which_trajectory, which_time].T,
            color='black', linewidth=1, density=1
        )
        plt.axis('off')
        plt.axis([0 - offset, self.Lx + offset, 0 - offset, self.Ly + offset])
        plt.title(f'Trajectory {which_trajectory} at time t = {round(self.t[which_time], 3)}')
        plt.grid(True)
        plt.gca().add_patch(
            patches.Rectangle((0, 0), self.Lx, self.Ly, linewidth=5, edgecolor='black', facecolor='none')
        )
        # Plot measurement sites if available
        if self.sensor_indices is not None:
            idx_sensors_x, idx_sensors_y = np.unravel_index(self.sensor_indices, (self.nx, self.ny))
            sensors_x = self.x[idx_sensors_x]
            sensors_y = self.y[idx_sensors_y]
            plt.plot(sensors_x, sensors_y, 'o', mfc='magenta', mec='black', ms=8, mew=1.5, label='Measurement Sites')
            plt.legend(loc='upper right')
        plt.show()

    def plot_comparison(self, test_reconstruction, test_indices, which_trajectory, which_time=-1):
        # Extract the reconstructed U and V fields for the specified trajectory and time
        U_recon, V_recon = test_reconstruction["U"], test_reconstruction["V"]
        Urecon = U_recon.reshape(-1, self.ntimes, self.nx, self.ny)
        Vrecon = V_recon.reshape(-1, self.ntimes, self.nx, self.ny)

        # Bounds check for which_trajectory
        n_trajectories = Urecon.shape[0]
        if which_trajectory < 0 or which_trajectory >= n_trajectories:
            print(f"Error: which_trajectory={which_trajectory} is out of bounds. Available: 0 to {n_trajectories-1}")
            return

        # Map reconstructed index to original index
        if test_indices is not None:
            orig_idx = test_indices[which_trajectory]
        else:
            orig_idx = which_trajectory

        # Compute mean relative error (MRE) for U and V using the provided formula
        U_true = self.U[orig_idx, which_time]
        V_true = self.V[orig_idx, which_time]
        U_pred = Urecon[which_trajectory, which_time]
        V_pred = Vrecon[which_trajectory, which_time]

        offset = 0.1
        fig, axes = plt.subplots(1, 2, figsize=(18, 6))

        # Original
        ax = axes[0]
        ax.contourf(
            self.x, self.y,
            self.vorticity(U_true, V_true).T,
            cmap='seismic', levels=100
        )
        ax.streamplot(
            self.x, self.y,
            U_true.T,
            V_true.T,
            color='black', linewidth=1, density=1
        )
        ax.set_title(f'Original Trajectory {orig_idx} at t={round(self.t[which_time], 3)}')
        ax.axis('off')
        ax.axis([0 - offset, self.Lx + offset, 0 - offset, self.Ly + offset])
        ax.grid(True)
        ax.add_patch(
            patches.Rectangle((0, 0), self.Lx, self.Ly, linewidth=5, edgecolor='black', facecolor='none')
        )
        if self.sensor_indices is not None:
            idx_sensors_x, idx_sensors_y = np.unravel_index(self.sensor_indices, (self.nx, self.ny))
            sensors_x = self.x[idx_sensors_x]
            sensors_y = self.y[idx_sensors_y]
            ax.plot(sensors_x, sensors_y, 'o', mfc='magenta', mec='black', ms=8, mew=1.5, label='Measurement Sites')
            ax.legend(loc='upper right')

        # Reconstructed
        ax = axes[1]
        ax.contourf(
            self.x, self.y,
            self.vorticity(U_pred, V_pred).T,
            cmap='seismic', levels=100
        )
        ax.streamplot(
            self.x, self.y,
            U_pred.T,
            V_pred.T,
            color='black', linewidth=1, density=1
        )
        ax.set_title(f'Reconstructed Trajectory {which_trajectory} at t={round(self.t[which_time], 3)} ')
        ax.axis('off')
        ax.axis([0 - offset, self.Lx + offset, 0 - offset, self.Ly + offset])
        ax.grid(True)
        ax.add_patch(
            patches.Rectangle((0, 0), self.Lx, self.Ly, linewidth=5, edgecolor='black', facecolor='none')
        )
        if self.sensor_indices is not None:
            idx_sensors_x, idx_sensors_y = np.unravel_index(self.sensor_indices, (self.nx, self.ny))
            sensors_x = self.x[idx_sensors_x]
            sensors_y = self.y[idx_sensors_y]
            ax.plot(sensors_x, sensors_y, 'o', mfc='magenta', mec='black', ms=8, mew=1.5, label='Measurement Sites')
            ax.legend(loc='upper right')

        plt.tight_layout()
        plt.show()