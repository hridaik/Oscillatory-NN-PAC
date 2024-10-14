from pactools import Comodulogram
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

print("Hi")

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

@tf.function
def oscillator_loop(X_r, X_i, omega_param, num_steps, dt, input_scaler, beta1):
    """
    Loop to run Euler method on Hopf equations
    Args:
        X_r (Tensor): Input to real
        X_i (Tensor): Input to imaginary
        omega_param (Tensor): frequency values
        num_steps (int): time steps
        dt (float): step size
        input_scaler (float)
        beta1 (float)
    Returns:
        Tensor, Tensor: r and phi
    """
    r_arr = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    phi_arr = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    omega_param = omega_param * (2 * np.pi)
    r_t = tf.ones((tf.shape(X_r)[0], X_r.shape[-1]))
    phis = tf.zeros((tf.shape(X_r)[0], X_r.shape[-1]))
    for t in tf.range(num_steps):
        input_r = input_scaler * X_r[:, t, :] * tf.math.cos(phis)
        input_phi = input_scaler * X_i[:, t, :] * tf.math.sin(phis)
        r_t = r_t + ((1 - beta1 * tf.square(r_t)) * r_t + input_r) * dt
        phis = phis + (omega_param - input_phi) * dt
        r_arr = r_arr.write(r_arr.size(), r_t)
        phi_arr = phi_arr.write(phi_arr.size(), phis)
    r_arr = tf.transpose(r_arr.stack(), [1, 0, 2])
    phi_arr = tf.transpose(phi_arr.stack(), [1, 0, 2])
    return r_arr, phi_arr

class Hopf(tf.keras.layers.Layer):
    def __init__(self, dim, num_steps, min_omega=0.1, max_omega=10.1, train_omegas=False,
                 dt=0.001, input_scaler=5.0, beta1=1.0, **kwargs):
        """
        Hopf oscillator layer
        Args:
            dim (int): number of hopf units
            num_steps (int): no. of time steps in sequence
            min_omega (float, optional): minimum frequency (in Hz). Defaults to 0.1.
            max_omega (float, optional): maximum frequency (in Hz). Defaults to 10.1.
            train_omegas (bool, optional): whether to train omegas also. Defaults to False.
            dt (float, optional): step size. Defaults to 0.001.
            input_scaler (float, optional): multiplier of input to hopf. Defaults to 5.
            beta1 (float, optional): width of basin of attraction for resonance. Defaults to 1.
        """
        super(Hopf, self).__init__(**kwargs)
        self.dim = dim
        self.num_steps = num_steps
        self.min_omega = min_omega
        self.max_omega = max_omega
        self.train_omegas = train_omegas
        self.dt = dt
        self.input_scaler = input_scaler
        self.beta1 = beta1
        omega_init = tf.random.uniform((1, self.dim), -1, 1)
        self.omegas = tf.Variable(omega_init, trainable=train_omegas)

    def call(self, X_r, X_i):
        """
        Forward function
        Args:
            X_r (Tensor): Input to real
            X_i (Tensor): Input to imaginary
        Returns:
            Tensor, Tensor: real and complex output
        """
        omega_inp = self.omegas * (self.max_omega - self.min_omega) + self.min_omega
        r, phi = oscillator_loop(X_r, X_i, omega_inp, self.num_steps,
                                 self.dt, self.input_scaler, self.beta1)
        z_real = r * tf.math.cos(phi)
        z_imag = r * tf.math.sin(phi)
        return z_real, z_imag

@tf.keras.saving.register_keras_serializable()
class Model(tf.keras.Model):
    def __init__(self, units, **kwargs):
        super(Model, self).__init__(**kwargs)
        self.d1_r = tf.keras.layers.Dense(units, activation='relu')
        self.d1_i = tf.keras.layers.Dense(units, activation='relu')
        self.osc1 = Hopf(units, num_steps=TOTAL_TONES, min_omega=0.1,
                         max_omega=12, train_omegas=True, dt=0.001)
        self.d2_r = tf.keras.layers.Dense(units, activation='relu')
        self.d2_i = tf.keras.layers.Dense(units, activation='relu')
        self.osc2 = Hopf(units, num_steps=TOTAL_TONES, min_omega=0.1,
                         max_omega=12, train_omegas=True, dt=0.001)
        self.d = tf.keras.layers.Dense(units, activation='tanh')
        self.out_dense = tf.keras.layers.Dense(1)

    def call(self, X):
        out1_r = tf.keras.layers.TimeDistributed(self.d1_r)(X)
        out1_i = tf.keras.layers.TimeDistributed(self.d1_i)(X)
        z1_r, z1_i = self.osc1(out1_r, out1_i)
        z1_r = tf.keras.layers.TimeDistributed(self.d2_r)(z1_r)
        z1_i = tf.keras.layers.TimeDistributed(self.d2_i)(z1_i)
        z2_r, z2_i = self.osc2(z1_r, z1_i)
        concat_inp = tf.concat([z2_r, z2_i], 2)
        out2 = tf.keras.layers.TimeDistributed(self.d)(concat_inp)
        out_final = tf.keras.layers.TimeDistributed(self.out_dense)(out2)
        return out_final

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
    model = tf.keras.models.load_model('C:/Users/hridai/Desktop/DONN/Oddball_002.keras', custom_objects={'Hopf': Hopf})
    
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