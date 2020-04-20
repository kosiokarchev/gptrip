from kivy import app, config, clock
from kivy.graphics.texture import Texture

import numpy as np
from matplotlib import pyplot as plt

from ..abc import AbstractGPTrip, AbstractSong
from .songs import SDSong, EndlessPlayingSong


class GPLiveApp(app.App, AbstractGPTrip):
    def __init__(self, song: AbstractSong, width, height, fftscale=2,
                 minfreq=60, maxfreq=1e4, minwav=0., maxwav=1.,
                 cmap=plt.get_cmap('hsv', 6), norm=plt.Normalize(-3, 3, clip=True),
                 random_seeds=True, coherent=True,
                 **kwargs):
        self.song = song
        self.songiter = None
        self.cmap = cmap
        self.norm = norm

        config.Config.set('kivy', 'show_fps', True)
        config.Config.set('graphics', 'width', width)
        config.Config.set('graphics', 'height', height)
        config.Config.set('graphics', 'resizable', False)

        super().__init__(
            width=width//fftscale, height=height//fftscale, fspec=self.song.fspec,
            minfreq=minfreq, maxfreq=maxfreq, minwav=minwav, maxwav=maxwav,
            random_directions=False, random_seeds=random_seeds,
            **kwargs
        )

        if coherent:
            self.coherent()

    def draw(self, dt):
        self.title = f'{1/dt:.2f} fps'

        spec, a = next(self.songiter)
        if spec is None:
            return

        img, amps = self.generate_image(spec / self.song.fspec, self.get_seeds(None))
        sigma = self.math.sqrt((img ** 2).mean())
        img_final = self.cmap(self.norm(np.fft.fftshift(img / sigma)))[..., :3]

        self.root.texture.blit_buffer(
            img_final.astype(np.float32).tobytes(), colorfmt='rgb', bufferfmt='float'
        )
        self.root.canvas.ask_update()

    def on_start(self):
        self.root.texture = Texture.create(size=(self.width, self.height), colorfmt='rgb', bufferfmt='float')
        self.songiter = iter(self.song)
        clock.Clock.schedule_interval(self.draw, 1/self.song.specrate)
