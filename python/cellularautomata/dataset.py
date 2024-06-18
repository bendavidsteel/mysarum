import jax.numpy as jnp

def add_sequence(sequence: jnp.array, coord, channel_idx, interval=2):
    for idx in range(0, sequence.shape[1], interval):
        sequence = sequence.at[:, idx, coord[0], coord[1], channel_idx].set(1.0)
    return sequence

class RhythmDataset:
    def __init__(self, img_size: int = 64):
        self.img_size = img_size
        self.img_shape = (img_size, img_size, 1)

    def get_batch(self, batch_size: int = 1, sequence_length: int = 16):
        channels = [0, 1]
        sequence = jnp.zeros((batch_size, sequence_length, self.img_size, self.img_size, len(channels)))
        sequence = add_sequence(sequence, (self.img_size // 2, self.img_size // 2), channels[0], interval=2)
        sequence = add_sequence(sequence, (self.img_size // 4, self.img_size // 4), channels[1], interval=5)
        
        return sequence, jnp.stack(channels)