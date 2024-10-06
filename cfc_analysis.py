import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from pactools import Comodulogram
import seaborn as sns

# Constants
TOTAL_TONES = 250
ST_FREQ = 250  # Standard tone frequency (Hz)
TT_FREQ = 4000  # Target tone frequency (Hz)
SAMPLING_RATE = 44100  # Hz
TONE_DURATION = 0.1  # seconds
SAMPLES_PER_TONE = int(SAMPLING_RATE * TONE_DURATION)

def generate_oddball_sequence(total_tones=TOTAL_TONES, tt_probability=0.2):
    """Generate an auditory oddball sequence of tones."""
    return np.where(np.random.random(total_tones) < tt_probability, TT_FREQ, ST_FREQ)

def generate_waveform(frequency, duration=TONE_DURATION, sampling_rate=SAMPLING_RATE):
    """Generate a waveform for a given frequency tone."""
    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
    return np.sin(2 * np.pi * frequency * t)

def create_oddball_input_waveforms(sequence):
    """Create waveform input data from the oddball sequence."""
    return np.array([generate_waveform(freq) for freq in sequence])

def create_target_tones(sequence):
    """Create target labels for the oddball sequence."""
    return np.array([1 if freq == TT_FREQ else 0 for freq in sequence])

# Generate dataset
def generate_dataset(num_sequences):
    sequences = [generate_oddball_sequence() for _ in range(num_sequences)]
    X = np.array([create_oddball_input_waveforms(seq) for seq in sequences])
    y = np.array([create_target_tones(seq) for seq in sequences])
    return X, y

def extract_oscillations(model, input_data):
    """
    Extract oscillation data from the Hopf layers of the model.
    
    Args:
    model (tf.keras.Model): The trained model
    input_data (np.array): Input data to feed into the model
    
    Returns:
    dict: Dictionary containing oscillation data from both Hopf layers
    """
    # Create a new model that outputs the states of the Hopf layers
    hopf1_output = model.osc1.output
    hopf2_output = model.osc2.output
    oscillation_model = tf.keras.Model(inputs=model.input, outputs=[hopf1_output, hopf2_output])
    
    # Get the oscillation data
    hopf1_data, hopf2_data = oscillation_model.predict(input_data)
    
    return {
        'hopf1': {'real': hopf1_data[0], 'imag': hopf1_data[1]},
        'hopf2': {'real': hopf2_data[0], 'imag': hopf2_data[1]}
    }

def analyze_pac(oscillation_data, fs=1000, low_fq_range=(1, 20), high_fq_range=(30, 200)):
    """
    Analyze phase-amplitude coupling in the oscillation data.
    
    Args:
    oscillation_data (dict): Dictionary containing oscillation data
    fs (int): Sampling frequency
    low_fq_range (tuple): Range of low frequencies to consider
    high_fq_range (tuple): Range of high frequencies to consider
    
    Returns:
    dict: Dictionary containing PAC analysis results
    """
    results = {}
    
    for layer in ['hopf1', 'hopf2']:
        # Combine real and imaginary parts
        signal = oscillation_data[layer]['real'] + 1j * oscillation_data[layer]['imag']
        
        # Analyze a subset of oscillators (e.g., first 10)
        for i in range(10):
            estimator = Comodulogram(fs=fs, low_fq_range=low_fq_range, high_fq_range=high_fq_range,
                                     method='tort', progress_bar=False)
            estimator.fit(signal[:, i])
            results[f'{layer}_oscillator_{i}'] = estimator
    
    return results

def plot_pac_results(pac_results):
    """
    Plot the phase-amplitude coupling results.
    
    Args:
    pac_results (dict): Dictionary containing PAC analysis results
    """
    n_plots = len(pac_results)
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    
    for i, (key, estimator) in enumerate(pac_results.items()):
        ax = axes[i]
        mesh = estimator.plot(ax=ax, vmin=0, vmax=estimator.comod_.max(), cmap='viridis')
        ax.set_title(key)
    
    plt.tight_layout()
    plt.show()

# Main execution
if __name__ == "__main__":
    # Load the trained model
    model = tf.keras.models.load_model('Oddball_001.keras')
    
    # Generate some test data
    # This should match the input shape expected by your model
    num_sequences = 100
    X, _ = generate_dataset(num_sequences)
    X = (X - np.mean(X)) / np.std(X)
    X = X.reshape((num_sequences, TOTAL_TONES, SAMPLES_PER_TONE))
    
    # Extract oscillations
    oscillation_data = extract_oscillations(model, X)
    
    # Analyze PAC
    pac_results = analyze_pac(oscillation_data)
    
    # Plot results
    plot_pac_results(pac_results)
    
    # Additional analysis: compare PAC strength between layers or oscillators
    pac_strengths = {key: estimator.comod_.max() for key, estimator in pac_results.items()}
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(pac_strengths.keys()), y=list(pac_strengths.values()))
    plt.title('Maximum PAC Strength Across Oscillators')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()