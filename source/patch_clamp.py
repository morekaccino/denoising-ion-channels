from typing import List

import numpy as np

from source.ion_channel import IonChannel


class PatchClamp:
    """
    Patch Clamp can contain multiple ion channels, and sums up the current from each channel to get the total current.
    """

    def __init__(self, channels: List[IonChannel] = []):
        self.channels = channels

    def get_current(self):
        return np.sum([channel.X for channel in self.channels], axis=0)

    def get_state(self):
        return np.sum([channel.y for channel in self.channels], axis=0)

    @property
    def X(self):
        return self.get_current()

    @property
    def y(self):
        return self.get_state()
