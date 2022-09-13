import numpy as np
import imageio
import os
import yaml
import logging

from glob import glob
from tensorboardX import SummaryWriter


logger = logging.getLogger(__name__)


class Logger(object):
    def __init__(self, log_dir):
        """
        General logger.

        Args:
            log_dir (str): log directory
        """
        self.writer = SummaryWriter(log_dir)
        self.info = logger.info
        self.debug = logger.debug
        self.warning = logger.warning

    def scalar_summary(self, tag, value, step):
        """
        Log scalar value to the disk.
        Args:
            tag (str): name of the value
            value (float): value
            step (int): update step
        """
        self.writer.add_scalar(tag, value, step)

    def close(self):
        self.writer.close()


class OUNoise(object):
    def __init__(
        self,
        dim,
        low,
        high,
        mu=0.0,
        theta=0.15,
        max_sigma=0.3,
        min_sigma=0.3,
        decay_period=10_000,
    ):
        self.mu = mu
        self.theta = theta
        self.sigma = max_sigma
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.action_dim = dim
        self.low = low
        self.high = high

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def evolve_state(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state

    def get_action(self, action, t=0):
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(
            1.0, t / self.decay_period
        )
        action = action.cpu().detach().numpy()
        return np.clip(action + ou_state, self.low, self.high)


def make_gif(source_dir, output):
    """
    Make gif file from set of .jpeg images.
    Args:
        source_dir (str): path with .jpeg images
        output (str): path to the output .gif file
    Returns: None
    """
    batch_sort = lambda s: int(s[s.rfind("/") + 1 : s.rfind(".")])
    image_paths = sorted(glob(os.path.join(source_dir, "*.png")), key=batch_sort)

    images = []
    for filename in image_paths:
        images.append(imageio.imread(filename))
    imageio.mimsave(output, images)


def read_config(path):
    """
    Return python dict from .yml file.

    Args:
        path (str): path to the .yml config.

    Returns (dict): configuration object.
    """
    with open(path, "r") as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    return cfg


def empty_torch_queue(q):
    while True:
        try:
            o = q.get_nowait()
            del o
        except:
            break
    q.close()
