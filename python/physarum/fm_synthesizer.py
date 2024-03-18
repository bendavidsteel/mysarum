import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
import soundcard as sc

def main():
    # Synthesizer parameters
    fs = 44100  # Sampling rate, 44100 samples per second
    f_carrier = 440.0  # Carrier frequency in Hz
    f_modulator = 1000.0  # Modulator frequency in Hz
    modulation_index = 1.0  # Modulation index
    duration = 2.0  # Duration of the sound in seconds

    default_speaker = sc.default_speaker()

    num_channels = default_speaker.channels

    with default_speaker.player(fs, blocksize=16000) as player:
        audio_data = np.sin(2 * np.pi * np.arange(fs * 2) * 440.0 / fs).astype(np.float32).reshape(-1, num_channels)
        player.play(audio_data)

    # Time array
    t = jnp.linspace(0, duration, int(fs * duration), endpoint=False)

    # Define the carrier and modulator signals
    carrier = jnp.sin(2 * jnp.pi * f_carrier * t)
    modulator = modulation_index * jnp.sin(2 * jnp.pi * f_modulator * t)

    # Apply frequency modulation
    fm_signal = jnp.sin(2 * jnp.pi * f_carrier * t + modulator)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 4))
    ax.plot(t[:1000], fm_signal[:1000])  # Plot the first 1000 samples
    ax.set_title('FM Signal Waveform')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    fig.savefig('./audio/fm_signal_waveform.png')

    # Convert the JAX array to a numpy array for saving to WAV file
    fm_signal_np = np.array(fm_signal)

    # Normalize the signal to 16-bit range for WAV file
    fm_signal_normalized = np.int16((fm_signal_np / fm_signal_np.max()) * 32767)

    # Save the FM sound to a WAV file
    wav_file_path = './audio/simple_fm_synthesizer.wav'
    wavfile.write(wav_file_path, fs, fm_signal_normalized)

    # wav_file_path

if __name__ == '__main__':
    main()