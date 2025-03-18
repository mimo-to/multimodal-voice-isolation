import numpy as np
import librosa
import soundfile as sf
import torch


# STFT settings — must match dataset.py exactly
SR = 16000
N_FFT = 512
HOP_LENGTH = 160


def load_magnitude(audio_path, num_frames=200):
    """
    Load a wav file and return its STFT magnitude and phase.
    Pads or truncates to num_frames time steps.
    """
    y, _ = librosa.load(audio_path, sr=SR)

    # fix length to 2 seconds so the spectrogram always has the same shape
    target_samples = SR * 2
    if len(y) < target_samples:
        y = np.pad(y, (0, target_samples - len(y)))
    else:
        y = y[:target_samples]

    stft = librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH)
    mag, phase = librosa.magphase(stft)  # mag: [257, T], phase: [257, T]

    # align to exactly num_frames columns
    if mag.shape[1] < num_frames:
        mag = np.pad(mag, ((0, 0), (0, num_frames - mag.shape[1])))
        phase = np.pad(phase, ((0, 0), (0, num_frames - phase.shape[1])), constant_values=1)
    else:
        mag = mag[:, :num_frames]
        phase = phase[:, :num_frames]

    return mag, phase  # both [257, num_frames]


def reconstruct_audio(noisy_audio_path, predicted_mask, output_path, num_frames=200):
    """
    Apply the predicted soft mask to the noisy audio and save the result.

    predicted_mask: torch.Tensor of shape [257, T] or numpy array, values in [0, 1]
    """
    # load noisy audio and get full STFT
    mag_noisy, phase_noisy = load_magnitude(noisy_audio_path, num_frames=num_frames)

    # convert mask to numpy if needed
    if isinstance(predicted_mask, torch.Tensor):
        mask_np = predicted_mask.detach().cpu().numpy()
    else:
        mask_np = np.array(predicted_mask)

    # apply mask: clean magnitude = noisy magnitude * mask
    clean_mag = mag_noisy * mask_np

    # reconstruct: combine clean magnitude with original noisy phase
    clean_stft = clean_mag * phase_noisy

    # inverse STFT -> waveform
    clean_audio = librosa.istft(clean_stft, hop_length=HOP_LENGTH)

    sf.write(output_path, clean_audio, SR)
    return output_path
