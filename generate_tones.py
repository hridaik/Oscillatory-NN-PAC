# Generate tone sequences for the auditory oddball task (250 tones with probabilities 0.8 for ST and 0.2 for TT)
import numpy as np

def generate_oddball_sequence(total_tones=250, st_freq=250, tt_freq=4000):
    """
    Generate an auditory oddball sequence of tones (ST and TT).
    
    Args:
        total_tones (int): Total number of tones in the sequence.
        st_freq (float): Frequency of the standard tone (ST) in Hz.
        tt_freq (float): Frequency of the target tone (TT).
    
    Returns:
        sequence (list): List of frequencies (either ST or TT) of length total_tones.
    """
    sequence = []
    tone_count = 0
    
    while tone_count < total_tones - 1:  # Ensure we stay within bounds
        # Generate standard tones (ST)
        st_series_length = np.random.randint(2, 8)
        if tone_count + st_series_length >= total_tones:
            st_series_length = total_tones - tone_count - 1  # Ensure total tones <= 250
        
        sequence.extend([st_freq] * st_series_length)
        tone_count += st_series_length
        
        # Generate target tone (TT)
        if tone_count < total_tones:
            sequence.append(tt_freq)
            tone_count += 1
    
    # Pad with ST if needed to ensure exactly 250 tones
    if len(sequence) < total_tones:
        sequence.extend([st_freq] * (total_tones - len(sequence)))

    return sequence

# Generate waveform for each tone (e.g., 250 samples per tone)
def generate_waveform(frequency, sampling_rate=250, duration=1.0):
    """
    Generate a waveform for a given frequency tone.
    
    Args:
        frequency (float): Frequency of the tone in Hz.
        sampling_rate (int): Sampling rate (samples per second).
        duration (float): Duration of the tone in seconds.
    
    Returns:
        waveform (np.array): Generated waveform for the tone.
    """
    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
    waveform = np.sin(2 * np.pi * frequency * t)
    return waveform

# Create the input waveforms for the network
def create_oddball_input_waveforms(sequence, sampling_rate=250, duration=1.0):
    """
    Create waveform input data from the oddball sequence.
    
    Args:
        sequence (list): Sequence of tone frequencies (ST and TT).
        sampling_rate (int): Sampling rate for each waveform.
        duration (float): Duration of each tone.
    
    Returns:
        input_waveforms (np.array): Array of waveforms for each tone in the sequence.
    """
    waveforms = [generate_waveform(freq, sampling_rate, duration) for freq in sequence]
    return np.array(waveforms)

# Generate the oddball task input
oddball_sequence = generate_oddball_sequence()
oddball_waveforms = create_oddball_input_waveforms(oddball_sequence)

# The generated oddball_waveforms is now ready to be used as input to the neural network
print(oddball_waveforms.shape)  # Should be (250, 250) -> 250 tones with 250 samples each
