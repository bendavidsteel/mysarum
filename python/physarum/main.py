from collections import namedtuple
import multiprocessing
import threading
import time

import einops
import jax
from jax import random
import jax.numpy as jnp
import jax.scipy as jsp
from jax import lax
import numpy as np
# import soundcard as sc
from vispy import app, scene, visuals
from vispy.scene.cameras.panzoom import PanZoomCamera

from physarums import Physarum, ParticleLife, ParticleSystem, Lenia


def audio_timer(get_audio, fs=44100, blocksize=16000, interval=1):
    import soundcard as sc
    default_speaker = sc.default_speaker()
    num_channels = default_speaker.channels
    with default_speaker.player(fs, channels=num_channels, blocksize=blocksize) as player:
        next_call = time.time()
        while True:
            audio_data = get_audio(fs, num_channels)
            player.play(audio_data)
            next_call = next_call + interval
            time.sleep(max(next_call - time.time(), 0))

class VisualizeSimulation:
    def __init__(self, simulation):
        self.simulation = simulation
        grid_size = simulation.grid_size

        # Audio setup
        # self.default_speaker = sc.default_speaker()
        # self.fs = 44100  # Sample rate
        # self.blocksize = 16000
        # self.num_channels = self.default_speaker.channels
        phase = 0  # To keep track of phase between updates

        if simulation.num_dims == 1:
            window_size = (1000, 600)
        else:
            window_size = (600, 600)

        self.canvas = scene.SceneCanvas(keys='interactive', size=window_size)
        self.canvas.show()

        if simulation.num_dims == 2:
            self.view = self.canvas.central_widget.add_view()
            self.view.camera = PanZoomCamera(aspect=1.0)

            im = self.simulation.get_display()
            self.image = scene.visuals.Image(im, parent=view.scene, clim=(0, 255))
        
        elif simulation.num_dims == 1:
            # vertex positions of data to draw
            self.pos = np.zeros((grid_size, 2), dtype=np.float32)
            x_lim = [0., float(grid_size)]
            self.pos[:, 0] = np.linspace(x_lim[0], x_lim[1], grid_size)
            line = self.simulation.get_intensity()
            self.pos[:, 1] = line

            # color array
            color = np.ones((grid_size, 4), dtype=np.float32)
            color[:, 0] = np.linspace(0, 1, grid_size)
            color[:, 1] = color[::-1, 0]

            grid = self.canvas.central_widget.add_grid(spacing=0)

            self.view = grid.add_view(row=0, col=1, camera='panzoom')

            # add some axes
            x_axis = scene.AxisWidget(orientation='bottom')
            x_axis.stretch = (1, 0.1)
            grid.add_widget(x_axis, row=1, col=1)
            x_axis.link_view(self.view)
            y_axis = scene.AxisWidget(orientation='left')
            y_axis.stretch = (0.1, 1)
            grid.add_widget(y_axis, row=0, col=0)
            y_axis.link_view(self.view)

            # add a line plot inside the viewbox
            self.line = scene.Line(self.pos, color, parent=self.view.scene)
            self.view.camera.set_range(y=tuple([0., 1.]), x=tuple(x_lim))

        # start audio system
        
        # if self.record:
        #     self.writer = imageio.get_writer('physarum.gif', duration=0.1)
        self.idx = 0

        # Update the image data periodically
        self.display_timer = app.Timer()
        self.display_timer.connect(self.update_image)
        self.display_timer.start()

        do_audio = False
        if do_audio:
            use_process = False
            kwargs = {'target': audio_timer, 'args': (self.generate_audio_data,), 'kwargs': {'interval':self.display_timer.interval}}
            if use_process:
                self.audio_timer = multiprocessing.Process(**kwargs)
            else:
                self.audio_timer = threading.Thread(**kwargs)
            self.audio_timer.start()

    def update_image(self, event):
        self.idx += 1
        self.simulation.update()
        if self.simulation.num_dims == 2:
            im = self.simulation.get_display()
            self.image.set_data(im)
        elif self.simulation.num_dims == 1:
            lines = self.simulation.get_intensity()
            self.pos[:, 1] = lines
            self.line.set_data(pos=self.pos)
            self.view.camera.set_range(y=tuple([0., 1.]), x=tuple([0., self.simulation.grid_size]))
        self.canvas.update()
        # audio_data = self.generate_audio_data()
        # self.player.play(audio_data)
        # if self.record:
        #     self.writer.append_data(im)

    def generate_audio_data(self, fs, num_channels):
        # Example method to generate audio data
        length = fs // 10  # Generate 0.1 seconds of audio at a time, for example
        t = np.arange(length) + self.phase
        audio_data = self.simulation.get_audio(t / fs)
        self.phase += length  # Update phase

        if num_channels > 1:
            audio_data = np.tile(audio_data.reshape(-1, 1), (1, num_channels))
        return audio_data


    def close(self):
        # if self.record:
        #     self.writer.close()
        self.display_timer.stop()
        if hasattr(self, 'audio_timer'):
            self.audio_timer.close()
        self.canvas.close()

def main():
    # Initial setup
    random_key = random.PRNGKey(3)

    record = True

    # Initialize simulation
    jit = True
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
        app.run()
    except:
        raise
    finally:
        print('Closing')
        if hasattr(visual, 'close'):
            visual.close()

if __name__ == '__main__':
    main()