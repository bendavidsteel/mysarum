from collections import namedtuple
import multiprocessing
import threading
import time
import traceback

from jax import random
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider

from physarums import Physarum, ParticleLife, ParticleSystem, Lenia

def audio_timer(get_audio, fs=44100, blocksize=16000, interval=1):
    def callback(outdata, frames, time, status):
        audio_data = get_audio(fs, outdata.shape[1])
        outdata[:] = audio_data.reshape(outdata.shape)

    stream = sd.OutputStream(callback=callback, samplerate=fs, channels=2, blocksize=blocksize)
    with stream:
        while True:
            time.sleep(interval)

class VisualizeSimulation:
    def __init__(self, simulation):
        self.simulation = simulation
        grid_size = simulation.grid_size

        phase = 0  # To keep track of phase between updates

        self.fig, self.ax = plt.subplots(figsize=(10, 6) if simulation.num_dims == 1 else (6, 6))
        self.fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.5)

        if simulation.num_dims == 1:
            self.lines = {}
            x = np.linspace(0, grid_size, grid_size)

            self.lines['chemical'], = self.ax.plot(x, self.simulation.get_intensity()[0,:], label='chemical')

            # plot agent positions
            agent_pos = self.simulation.get_agent_positions()
            self.lines['agents'] = self.ax.scatter(agent_pos, np.zeros_like(agent_pos), label='agents', c='r')

            if self.simulation.compute_fields:
                self.lines['U'], = self.ax.plot(x, np.zeros_like(x), label='U')
                self.lines['G'], = self.ax.plot(x, np.zeros_like(x), label='G')
                self.lines['R'], = self.ax.plot(x, np.zeros_like(x), label='R')
                self.lines['E'], = self.ax.plot(x, np.zeros_like(x), label='E')

            self.ax.set_ylim(0, 1)
            self.ax.legend()

        elif simulation.num_dims == 2:
            self.image = self.ax.imshow(self.simulation.get_display(), cmap='viridis', interpolation='nearest')
            self.fig.colorbar(self.image, ax=self.ax)

        self.ani = FuncAnimation(self.fig, self.update_image, interval=100, blit=False)

        axcolor = 'lightgoldenrodyellow'
        ax_force_factor = plt.axes([0.25, 0.06, 0.65, 0.03], facecolor=axcolor)
        ax_decay_rate = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
        ax_deposit_amount = plt.axes([0.25, 0.14, 0.65, 0.03], facecolor=axcolor)
        ax_mu_k = plt.axes([0.25, 0.18, 0.65, 0.03], facecolor=axcolor)
        ax_sigma_k = plt.axes([0.25, 0.22, 0.65, 0.03], facecolor=axcolor)
        ax_mu_g = plt.axes([0.25, 0.26, 0.65, 0.03], facecolor=axcolor)
        ax_sigma_g = plt.axes([0.25, 0.3, 0.65, 0.03], facecolor=axcolor)
        ax_c_rep = plt.axes([0.25, 0.34, 0.65, 0.03], facecolor=axcolor)

        self.s_force_factor = Slider(ax_force_factor, 'Force Factor', 0.1, 10.0, valinit=simulation.force_factor)
        self.s_decay_rate = Slider(ax_decay_rate, 'Decay Rate', 0.0, 1.0, valinit=simulation.decay_rate)
        self.s_deposit_amount = Slider(ax_deposit_amount, 'Deposit Amount', 0.1, 5.0, valinit=simulation.deposit_amount)
        self.s_mu_k = Slider(ax_mu_k, 'mu_k', 0.0, 1.0, valinit=simulation.mu_k)
        self.s_sigma_k = Slider(ax_sigma_k, 'sigma_k', 0.0, 1.0, valinit=simulation.sigma_k)
        self.s_mu_g = Slider(ax_mu_g, 'mu_g', 0.0, 1.0, valinit=simulation.mu_g)
        self.s_sigma_g = Slider(ax_sigma_g, 'sigma_g', 0.0, 1.0, valinit=simulation.sigma_g)
        self.s_c_rep = Slider(ax_c_rep, 'c_rep', 0.0, 1.0, valinit=simulation.c_rep)

        self.s_force_factor.on_changed(self.update_simulation_params)
        self.s_decay_rate.on_changed(self.update_simulation_params)
        self.s_deposit_amount.on_changed(self.update_simulation_params)
        self.s_mu_k.on_changed(self.update_simulation_params)
        self.s_sigma_k.on_changed(self.update_simulation_params)
        self.s_mu_g.on_changed(self.update_simulation_params)
        self.s_sigma_g.on_changed(self.update_simulation_params)
        self.s_c_rep.on_changed(self.update_simulation_params)

        self.idx = 0
        self.do_audio = False

        if self.do_audio:
            use_process = False
            kwargs = {'target': audio_timer, 'args': (self.generate_audio_data,), 'kwargs': {'interval': 0.1}}
            if use_process:
                self.audio_timer = multiprocessing.Process(**kwargs)
            else:
                self.audio_timer = threading.Thread(**kwargs)
            self.audio_timer.start()

    def run(self):
        plt.show()

    def update_simulation_params(self, val):
        self.simulation.force_factor = self.s_force_factor.val
        self.simulation.decay_rate = self.s_decay_rate.val
        self.simulation.deposit_amount = self.s_deposit_amount.val
        self.simulation.mu_k = self.s_mu_k.val
        self.simulation.sigma_k = self.s_sigma_k.val
        self.simulation.mu_g = self.s_mu_g.val
        self.simulation.sigma_g = self.s_sigma_g.val
        self.simulation.c_rep = self.s_c_rep.val

    def update_image(self, frame):
        self.idx += 1
        self.simulation.update()

        if self.simulation.num_dims == 2:
            self.image.set_data(self.simulation.get_display())

        elif self.simulation.num_dims == 1:
            self.lines['chemical'].set_ydata(self.simulation.get_intensity())
            agent_pos = self.simulation.get_agent_positions()
            self.lines['agents'].set_offsets(np.column_stack((agent_pos, np.zeros_like(agent_pos))))
            if self.simulation.compute_fields:
                fields = self.simulation.get_grid_fields()
                self.lines['U'].set_ydata(getattr(fields, 'U'))
                self.lines['G'].set_ydata(getattr(fields, 'G'))
                self.lines['R'].set_ydata(getattr(fields, 'R'))
                self.lines['E'].set_ydata(getattr(fields, 'E'))

        self.fig.canvas.draw()

    def generate_audio_data(self, fs, num_channels):
        length = fs // 10
        t = np.arange(length) + self.phase
        audio_data = self.simulation.get_audio(t / fs)
        self.phase += length

        if num_channels > 1:
            audio_data = np.tile(audio_data.reshape(-1, 1), (1, num_channels))
        return audio_data

    def close(self):
        if hasattr(self, 'audio_timer'):
            self.audio_timer.close()

def main():
    random_key = random.PRNGKey(3)

    record = True

    jit = False
    num_dims = 1
    num_agents = 2
    num_species = 1
    num_chemicals = 1
    grid_size = 100
    speed = 1.0
    decay_rate = 10e-3
    deposit_amount = 1.0
    sensor_angle = jnp.pi
    sensor_distance = 3
    kwargs = {}

    physarum_type = 'lenia'
    if physarum_type == 'particle_life':
        num_species = 1
        kwargs['beta'] = 0.4
        simulation_cls = ParticleLife
    elif physarum_type == 'particle_system':
        simulation_cls = ParticleSystem
    elif physarum_type == 'lenia':
        simulation_cls = Lenia
        kwargs['compute_fields'] = True

    simulation = simulation_cls(
        random_key,
        jit=jit,
        num_dims=num_dims,
        num_agents=num_agents, 
        grid_size=grid_size, 
        num_species=num_species, 
        num_chemicals=num_chemicals, 
        speed=speed, 
        decay_rate=decay_rate, 
        deposit_amount=deposit_amount, 
        sensor_angle=sensor_angle, 
        sensor_distance=sensor_distance,
        **kwargs
    )

    visual = None
    try:
        visual = VisualizeSimulation(simulation)
        visual.run()
    except Exception as e:
        print(e)
        print(traceback.format_exc())
    finally:
        print('Closing')
        if hasattr(visual, 'close'):
            visual.close()

if __name__ == '__main__':
    main()
