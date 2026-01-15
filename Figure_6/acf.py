import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from matplotlib.patches import Patch

# Set plotting style for publication quality
plt.style.use('default')

plt.rcParams.update({
    'font.size': 12,  # Increase base font size
    'font.family': 'Times New Roman',  # Changed to Times New Roman
    'axes.labelsize': 14,  # Increase axis label size
    'axes.titlesize': 16,  # Increase title size
    'xtick.labelsize': 12,  # Increase x-axis tick label size
    'ytick.labelsize': 12,  # Increase y-axis tick label size
    'legend.fontsize': 13,  # Increase legend font size (increased from 11)
    'lines.linewidth': 2.5,  # Increase line width
    'axes.grid': True,
    'grid.alpha': 0.2,
    'grid.linestyle': '--',
    'figure.figsize': (16, 6),  # Increase figure width for right legend
    'mathtext.fontset': 'dejavuserif',  # Changed to match Times New Roman
    'mathtext.default': 'regular',
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'axes.linewidth': 1.2,  # Increase axis line width
    'axes.edgecolor': 'black',
    'xtick.major.width': 1.2,
    'ytick.major.width': 1.2,
})

def digitize_value(val):
    """Digitize value: E-6 magnitude (close to 0) becomes -1, E-2 magnitude becomes 1"""
    if abs(val) < 0.5:  # Values with E-6 magnitude are < 0.5
        return -1
    else:
        return 1

def load_and_digitize_csv(filename, required_points=1050000):
    """Load CSV file and digitize signal values"""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Error: File '{filename}' does not exist!")
    
    try:
        # Read CSV
        df = pd.read_csv(filename, header=None, skiprows=1)
        
        # Choose column based on filename
        if "5G" in filename.upper() or "5g" in filename:
            # 5G case
            if df.shape[1] >= 2:
                last_col_index = df.shape[1] - 1
                second_last_col_index = df.shape[1] - 2
                df = df.iloc[:, [second_last_col_index, last_col_index]]
        else:
            # For 1GHz, 2GHz, 3GHz case - use first two columns
            df = df.iloc[:, [0, 1]]
        
        # Rename columns
        df.columns = ['time', 'sample']
        
        total_points = len(df)
        
        if total_points < required_points:
            required_points = total_points
        
        # Digitize signal
        df['digitized'] = df['sample'].apply(digitize_value)
        
        # Read only required_points dots
        signal_data = df['digitized'].values[:required_points]
        
        # Extract time information for sampling rate
        time_data = df['time'].values[:required_points]
        
        # Calculate sampling rate (assuming uniform sampling)
        if len(time_data) > 1:
            dt = np.mean(np.diff(time_data))
            fs = 1.0 / dt if dt > 0 else 1.0
        else:
            fs = 1.0  # Default fallback
            
        return signal_data, fs, total_points
        
    except Exception as e:
        print(f"Error in load_and_digitize_csv for {filename}: {e}")
        raise e

def calculate_acf_simple(signal_data, max_lag=100):
    """Calculate autocorrelation function"""
    n = len(signal_data)
    
    # Calculate mean and center the signal
    mean_val = np.mean(signal_data)
    data_centered = signal_data - mean_val
    
    # Initialize ACF array (from lag 0 to max_lag)
    acf = np.zeros(max_lag + 1)
    
    # Calculate ACF for lags 0 to max_lag
    actual_max_lag = min(max_lag, n - 1)
    
    for lag in range(0, actual_max_lag + 1):
        numerator = np.sum(data_centered[:n-lag] * data_centered[lag:])
        
        denominator_left = np.sum(data_centered[:n-lag]**2)
        denominator_right = np.sum(data_centered[lag:]**2)
        denominator = np.sqrt(denominator_left * denominator_right)
        
        if denominator != 0:
            acf[lag] = numerator / denominator
        else:
            acf[lag] = 0.0
    
    # Ensure lag 0 = 1.0
    if max_lag >= 0 and len(data_centered) > 0:
        norm = np.sqrt(np.sum(data_centered**2) * np.sum(data_centered**2))
        if norm != 0:
            acf[0] = np.sum(data_centered * data_centered) / norm
        else:
            acf[0] = 1.0
    
    return acf

def calculate_noise_contributions():
    """Calculate noise contributions from the provided data files"""
    
    # Data from 1GHz 1.3V (original data)
    data_1G_1_3V = {
        'id': [23.79, 23.79, 14.88, 14.88],  # Thermal noise (id)
        'fn': [8.71, 8.71, 2.25, 2.25],      # Flicker noise (fn)
        'rs': [0.20, 0.20, 0.17, 0.17],      # Other noise
        'rd': [0.00, 0.00, 0.00, 0.00],      # Other noise
        'igd': [0.00, 0.00, 0.00],           # Other noise
        'igs': [0.00]                        # Other noise
    }
    
    # Data from 1GHz 0.9V (original data)
    data_1G_0_9V = {
        'id': [27.20, 27.19, 17.63, 17.62],  # Thermal noise (id)
        'fn': [4.02, 4.02, 0.87, 0.87],      # Flicker noise (fn)
        'rs': [0.20, 0.20, 0.09, 0.09],      # Other noise
        'rd': [0.00, 0.00, 0.00, 0.00],      # Other noise
        'igd': [0.00, 0.00],                 # Other noise
        'igs': [0.00, 0.00]                  # Other noise
    }
    
    # Data from 5GHz 1.3V (first image data)
    data_5G_1_3V = {
        'id': [23.79, 23.79, 14.88, 14.88],  # Thermal noise (id) - total = 77.34%
        'fn': [8.71, 8.71, 2.25, 2.25],      # Flicker noise (fn) - total = 21.92%
        'rs': [0.20, 0.20, 0.17, 0.17],      # Other noise - total = 0.74%
        'rd': [0.00, 0.00, 0.00, 0.00],      # Other noise - total = 0.00%
        'igd': [0.00, 0.00, 0.00],           # Other noise - total = 0.00%
        'igs': [0.00]                        # Other noise - total = 0.00%
    }
    
    # Data from 3GHz 0.9V (second image data)
    data_3G_0_9V = {
        'id': [27.20, 27.19, 17.63, 17.62],  # Thermal noise (id) - total = 89.64%
        'fn': [4.02, 4.02, 0.87, 0.87],      # Flicker noise (fn) - total = 9.78%
        'rs': [0.20, 0.20, 0.09, 0.09],      # Other noise - total = 0.58%
        'rd': [0.00, 0.00, 0.00, 0.00],      # Other noise - total = 0.00%
        'igd': [0.00, 0.00],                 # Other noise - total = 0.00%
        'igs': [0.00, 0.00]                  # Other noise - total = 0.00%
    }
    
    # Calculate total percentages for each noise type for each frequency
    thermal_1G_1_3V = sum(data_1G_1_3V['id'])
    flicker_1G_1_3V = sum(data_1G_1_3V['fn'])
    else_1G_1_3V = (sum(data_1G_1_3V['rs']) + sum(data_1G_1_3V['rd']) + 
                    sum(data_1G_1_3V['igd']) + sum(data_1G_1_3V['igs']))
    
    thermal_1G_0_9V = sum(data_1G_0_9V['id'])
    flicker_1G_0_9V = sum(data_1G_0_9V['fn'])
    else_1G_0_9V = (sum(data_1G_0_9V['rs']) + sum(data_1G_0_9V['rd']) + 
                    sum(data_1G_0_9V['igd']) + sum(data_1G_0_9V['igs']))
    
    thermal_5G_1_3V = sum(data_5G_1_3V['id'])
    flicker_5G_1_3V = sum(data_5G_1_3V['fn'])
    else_5G_1_3V = (sum(data_5G_1_3V['rs']) + sum(data_5G_1_3V['rd']) + 
                    sum(data_5G_1_3V['igd']) + sum(data_5G_1_3V['igs']))
    
    thermal_3G_0_9V = sum(data_3G_0_9V['id'])
    flicker_3G_0_9V = sum(data_3G_0_9V['fn'])
    else_3G_0_9V = (sum(data_3G_0_9V['rs']) + sum(data_3G_0_9V['rd']) + 
                    sum(data_3G_0_9V['igd']) + sum(data_3G_0_9V['igs']))
    
    # Create arrays for stacked bar chart - 4 bars total
    frequency_labels = ['1 GHz (0.9 V)', '1 GHz (1.3 V)', '3 GHz (0.9 V)', '5 GHz (1.3 V)']
    thermal_data = [thermal_1G_0_9V, thermal_1G_1_3V, thermal_3G_0_9V, thermal_5G_1_3V]
    flicker_data = [flicker_1G_0_9V, flicker_1G_1_3V, flicker_3G_0_9V, flicker_5G_1_3V]
    else_data = [else_1G_0_9V, else_1G_1_3V, else_3G_0_9V, else_5G_1_3V]
    
    return frequency_labels, thermal_data, flicker_data, else_data

def plot_combined_analysis():
    """Plot combined autocorrelation and noise contribution analysis"""
    
    # CSV files to process - excluding 2G
    csv_files = ["1G_30nm_0.9V_count0.csv", "5G_30nm_1.3V_count0.csv", 
                 "1G_30nm_1.3V_count0.csv", "3G_30nm_0.9V_count0.csv"]
    
    # Check which files actually exist
    existing_files = []
    for csv_file in csv_files:
        if os.path.exists(csv_file):
            existing_files.append(csv_file)
        else:
            print(f"Warning: File '{csv_file}' does not exist, skipping.")
    
    if not existing_files:
        print("Error: No CSV files found!")
        return
    
    required_points = 1050000
    
    print("="*70)
    print(f"COMBINED ANALYSIS: Autocorrelation and Noise Contributions")
    print(f"Using {required_points:,} data points for each signal")
    print(f"Found {len(existing_files)} CSV files: {existing_files}")
    print("="*70)
    
    # Load and process signals
    signals = {}
    acf_results = {}
    voltage_info = {}
    sampling_rates = {}
    actual_points_used = {}
    
    try:
        for csv_file in existing_files:
            print(f"\nProcessing {csv_file}...")
            
            # Load signal and sampling rate
            signal_data, fs, total_points = load_and_digitize_csv(csv_file, required_points)
            
            # Create label based on filename including voltage
            if "5G" in csv_file or "5g" in csv_file:
                freq_label = "5 GHz (1.3 V)"
                voltage = "1.3 V"
                plot_label = "5 GHz (1.3 V)"
            elif "1G_30nm_1.3V" in csv_file:
                freq_label = "1 GHz (1.3 V)"
                voltage = "1.3 V"
                plot_label = "1 GHz (1.3 V)"
            elif "3G" in csv_file or "3g" in csv_file:
                freq_label = "3 GHz (0.9 V)"
                voltage = "0.9 V"
                plot_label = "3 GHz (0.9 V)"
            elif "1G_30nm_0.9V" in csv_file:
                freq_label = "1 GHz (0.9 V)"
                voltage = "0.9 V"
                plot_label = "1 GHz (0.9 V)"
            else:
                # Default naming
                freq_label = csv_file.split('_')[0]
                voltage = "N/A"
                plot_label = freq_label
                
            # Use a unique key for each file to avoid overwriting
            # We'll use the full plot_label as the key
            key = plot_label
            signals[key] = signal_data
            voltage_info[key] = voltage
            sampling_rates[key] = fs
            
            # Calculate ACF using ALL data points
            acf = calculate_acf_simple(signal_data, 100)
            acf_results[key] = acf
            
            actual_points_used[key] = len(signal_data)
            
            print(f"  Data points loaded: {total_points:,}")
            print(f"  Data points used for analysis: {len(signal_data):,}")
            print(f"  Sampling rate: {fs/1e9:.2f} GHz")
            print(f"  Voltage: {voltage}")
            print(f"  ACF(1): {acf[1]:.6f}")
            print(f"  ACF(10): {acf[10]:.6f}")
            print(f"  ACF(100): {acf[100]:.6f}")
        
    except (FileNotFoundError, ValueError) as e:
        print(f"\nERROR: {e}")
        print(f"Please ensure that CSV files exist: {existing_files}")
        return
    
    # Calculate noise contributions from provided data
    frequency_labels, thermal_data, flicker_data, else_data = calculate_noise_contributions()
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Use appropriate color scheme for scientific plots
    # Using tab10 color scheme, which is commonly used in scientific plots
    colors = {
        '1 GHz (0.9 V)': '#2ca02c',  # tab10 green
        '1 GHz (1.3 V)': '#ff7f0e',  # tab10 orange
        '5 GHz (1.3 V)': '#d62728',  # tab10 red
        '3 GHz (0.9 V)': '#9467bd'   # tab10 purple
    }
    
    # Line styles
    line_styles = {
        '1 GHz (0.9 V)': '--',
        '1 GHz (1.3 V)': '-.',
        '5 GHz (1.3 V)': ':',
        '3 GHz (0.9 V)': (0, (3, 1, 1, 1))
    }
    
    # Subplot 1: Autocorrelation (Linear scale) - LEFT SIDE
    ax1 = axes[0]
    max_lag = 100
    lags = np.arange(0, max_lag + 1)
    
    # Get all keys sorted in a specific order
    all_keys = list(acf_results.keys())
    
    # Sort keys to ensure consistent ordering
    def sort_key_func(key):
        # Extract frequency and voltage for sorting
        if '1 GHz' in key:
            if '0.9 V' in key:
                return (1, 0.9)  # 1 GHz 0.9V first
            else:
                return (1, 1.3)  # 1 GHz 1.3V second
        elif '3 GHz' in key:
            return (3, 0.9)
        elif '5 GHz' in key:
            return (5, 1.3)
        else:
            return (999, 0)  # Default
    
    all_keys.sort(key=sort_key_func)
    
    for key in all_keys:
        if key in acf_results:
            acf = acf_results[key]
            
            # Get color and linestyle
            line_color = colors.get(key, '#000000')
            line_style = line_styles.get(key, '-')
            
            ax1.plot(lags[1:max_lag+1], acf[1:max_lag+1], 
                   linewidth=2.5,  # Increase line width
                   color=line_color,
                   linestyle=line_style,
                   label=key,
                   alpha=0.85)  # Slightly reduce transparency
    
    ax1.set_xlabel(r'Lag $\tau$', fontsize=14, fontweight='bold')
    ax1.set_ylabel(r'Autocorrelation $p(\tau)$', fontsize=14, fontweight='bold')
    ax1.set_title(f'(a) Autocorrelation Function', 
                 fontsize=16, fontweight='bold', pad=12)
    
    ax1.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
    
    # Adjust left plot legend position and size
    ax1.legend(loc='upper right', fontsize=13, frameon=True, 
               edgecolor='gray', facecolor='white', framealpha=0.9)  # Increased fontsize
    
    # MODIFICATION 1: Expand x-axis range
    ax1.set_xlim([0, 100])  
    ax1.set_xticks(np.arange(0, 101, 10))  # Updated to match new range
    
    # MODIFICATION 2: Set y-axis limits from 0.00 to 0.06 only
    ax1.set_ylim([0.00, 0.06])
    ax1.axhline(y=0, color='black', alpha=0.5, linestyle='-', linewidth=1.2)
    
    # Increase text size inside plot
    for label in ax1.get_xticklabels() + ax1.get_yticklabels():
        label.set_fontsize(12)
    
    # Subplot 2: Noise Contribution Bar Chart - RIGHT SIDE
    ax2 = axes[1]
    
  
    bar_width = 0.71  
    
    x_pos = np.arange(len(frequency_labels)) * 1.2
    
    # Use specified color scheme
    # Light fill colors
    fill_colors = {
        'thermal': '#FFB366',  # Light orange fill - sigma_w (Thermal)
        'flicker': '#66B2FF',  # Light blue fill - sigma_f (Flicker)
        'else': '#66FFB2'      # Light green fill - else
    }
    
    # Dark border colors
    edge_colors = {
        'thermal': '#E69500',  # Dark orange border - sigma_w (Thermal)
        'flicker': '#0066CC',  # Dark blue border - sigma_f (Flicker)
        'else': '#00CC66'      # Dark green border - else
    }
    
    # Create stacked bar chart with dark borders + light fill for 4 bars
    bottom = np.zeros(len(frequency_labels))
    
    # Changed labels to Greek letters sigma_w and sigma_f
    # MODIFICATION 3: Increased legend fontsize to make sigma bigger
    # Plot thermal noise - orange (now sigma_w)
    thermal_bars = ax2.bar(x_pos, thermal_data, bar_width, label=r'$\sigma_w$', 
            color=fill_colors['thermal'], edgecolor=edge_colors['thermal'], 
            linewidth=2.0, bottom=bottom, zorder=3)  # Increased linewidth
    bottom += thermal_data
    
    # Plot flicker noise - blue (now sigma_f)
    flicker_bars = ax2.bar(x_pos, flicker_data, bar_width, label=r'$\sigma_f$', 
            color=fill_colors['flicker'], edgecolor=edge_colors['flicker'], 
            linewidth=2.0, bottom=bottom, zorder=3)  # Increased linewidth
    bottom += flicker_data
    
    # Plot else noise - green
    else_bars = ax2.bar(x_pos, else_data, bar_width, label='else', 
            color=fill_colors['else'], edgecolor=edge_colors['else'], 
            linewidth=2.0, bottom=bottom, zorder=3)  # Increased linewidth
    
    # Increase fontsize for percentage numbers
    for i, (thermal, flicker, else_val) in enumerate(zip(thermal_data, flicker_data, else_data)):
        # Thermal noise label
        thermal_y_pos = thermal/2
        if thermal > 10:  # Only show label if value is significant
            ax2.text(x_pos[i], thermal_y_pos, f'{thermal:.1f}%', ha='center', va='center', 
                    fontsize=14, fontweight='bold', color='black', zorder=4)  # Increased fontsize from 12 to 14
        
        # Flicker noise label
        flicker_mid = thermal + flicker/2
        if flicker > 5:  # Only show label if value is significant
            ax2.text(x_pos[i], flicker_mid, f'{flicker:.1f}%', ha='center', va='center', 
                    fontsize=14, fontweight='bold', color='black', zorder=4)  # Increased fontsize from 12 to 14
        
        # Else noise label
        if else_val > 0.5:
            else_mid = thermal + flicker + else_val/2
            ax2.text(x_pos[i], else_mid, f'{else_val:.1f}%', ha='center', va='center', 
                    fontsize=14, fontweight='bold', color='black', zorder=4)  # Increased fontsize from 12 to 14
    
    # MODIFICATION 4
    ax2.set_xlim([x_pos[0] - 1, x_pos[-1] + 1])
    
    # Customize the plot
    ax2.set_xlabel('Frequency and Voltage', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Noise Contribution (%)', fontsize=14, fontweight='bold')
    ax2.set_title('(b) Noise Contribution Analysis', 
                 fontsize=16, fontweight='bold', pad=12)
    
    # Set x-axis ticks for 4 bars
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(frequency_labels, fontsize=12, rotation=15)  # Increased fontsize
    
    ax2.set_ylim([0, 105])
    ax2.grid(True, alpha=0.2, linestyle='--', linewidth=0.5, axis='y', zorder=1)
    
    # MODIFICATION 6: 
    ax2.legend(loc='lower right', fontsize=14, frameon=True,  # 改为 lower right
               edgecolor='gray', facecolor='white', framealpha=0.9,
               bbox_to_anchor=(1, 0))
    
    # Increase right plot tick label size
    for label in ax2.get_xticklabels() + ax2.get_yticklabels():
        label.set_fontsize(12)
    
    # Removed the main suptitle completely
    
    # Adjust subplot spacing
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.25)
    
    # Save high-quality figure for publication
    output_file = 'combined_autocorrelation_noise_analysis.png'
    plt.savefig(output_file, dpi=600, bbox_inches='tight', facecolor='white', 
                transparent=False, pad_inches=0.1)
    print(f"\nFigure saved as: {output_file} (600 DPI for publication)")
    
    # Also save as PDF for publication (vector graphics)
    pdf_output_file = 'combined_autocorrelation_noise_analysis.pdf'
    plt.savefig(pdf_output_file, format='pdf', bbox_inches='tight', facecolor='white')
    print(f"Figure saved as: {pdf_output_file} (PDF for publication)")
    
    # Show figure
    plt.show()
    
    # Print detailed statistics
    print("\n" + "="*70)
    print("DETAILED NOISE STATISTICS")
    print("="*70)
    
    print(f"\nNoise Contributions from Provided Data:")
    print("-" * 50)
    
    for i, freq in enumerate(frequency_labels):
        print(f"\n{freq}:")
        print(f"  sigma_w (Thermal): {thermal_data[i]:.2f}%")
        print(f"  sigma_f (Flicker): {flicker_data[i]:.2f}%")
        print(f"  else: {else_data[i]:.2f}%")
        print(f"  Total: {thermal_data[i] + flicker_data[i] + else_data[i]:.2f}%")
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)

# Run main program
if __name__ == "__main__":
    print("Starting combined autocorrelation and noise contribution analysis")
    print("Will use 1,050,000 data points for each signal")
    print("="*70)
    
    plot_combined_analysis()