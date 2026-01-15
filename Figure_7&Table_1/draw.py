import numpy as np
import os
import struct
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
from scipy.stats import norm
import warnings
import matplotlib as mpl
import re  # Regular expression module
warnings.filterwarnings('ignore')

# ====================================================
# Set professional plotting style
# ====================================================

# Set global style
plt.style.use('seaborn-v0_8-whitegrid')
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
# Set math font to match Times New Roman
mpl.rcParams['mathtext.fontset'] = 'stix'  # STIX font compatible with Times New Roman
mpl.rcParams['mathtext.default'] = 'regular'  # Use regular font by default
mpl.rcParams['axes.labelsize'] = 14
mpl.rcParams['axes.titlesize'] = 16
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12
mpl.rcParams['legend.fontsize'] = 12
mpl.rcParams['figure.titlesize'] = 20  # Increase overall title font size
mpl.rcParams['figure.titleweight'] = 'bold'

# ====================================================
# Helper function: Extract frequency from folder name and convert to GHz format
# ====================================================

def extract_frequency_from_folder(folder_name):
    """
    Extract frequency information from folder name and convert to GHz format
    
    Parameters:
    -----------
    folder_name : str
        Folder name, e.g., 'V1.3_F1_T27'
        
    Returns:
    --------
    freq_str : str
        Formatted frequency string, e.g., '1GHz'
    """
    # Use regular expression to extract numeric part
    match = re.search(r'F(\d+)', folder_name)
    if match:
        freq_num = int(match.group(1))
        return f"{freq_num}GHz"
    else:
        # If extraction fails, return original folder name
        return folder_name

# ====================================================
# Data reading and analysis functions (updated according to Algorithm 1)
# ====================================================

def read_binary_file(file_path):
    """Read binary file and extract LSB"""
    with open(file_path, 'rb') as f:
        file_data = f.read()
    
    # Convert bytes to integers in range 0-255
    data_bytes = np.frombuffer(file_data, dtype=np.uint8)
    
    # Extract LSB of each byte
    bits = data_bytes & 1  # Get least significant bit
    
    return bits

def compute_acf_corrected(b, max_lag):
    """Calculate ACF according to Step 10 of Algorithm 1"""
    N = len(b)
    p_hat = np.mean(b)
    
    # Calculate autocorrelation function
    acf = np.zeros(max_lag + 1)
    
    for tau in range(max_lag + 1):
        if tau == 0:
            acf[tau] = 1.0
        else:
            # Calculate covariance
            numerator = np.sum((b[:N-tau] - p_hat) * (b[tau:] - p_hat))
            denominator = np.sqrt(np.sum((b[:N-tau] - p_hat)**2) * np.sum((b[tau:] - p_hat)**2))
            acf[tau] = numerator / denominator if denominator != 0 else 0
    
    return acf

def extract_platform_parameters(bits, max_lag=100):
    """
    Extract platform parameters according to Algorithm 1
    
    Parameters:
    -----------
    bits : array
        Original bit stream sequence
    max_lag : int
        Maximum lag
        
    Returns:
    --------
    results : dict
        Dictionary containing all calculation results
    """
    N = len(bits)
    b = bits.astype(float)
    
    # ================== Step 1: Statistical Estimation & Bias Correction ==================
    # Step 2: Calculate bit stream mean
    p_hat = np.mean(b)
    
    # Step 3: Estimate normalized bias
    if p_hat == 0:
        p_hat_adj = np.finfo(float).eps
    elif p_hat == 1:
        p_hat_adj = 1 - np.finfo(float).eps
    else:
        p_hat_adj = p_hat
    
    r_u = norm.ppf(p_hat_adj)  # r_u = Φ^{-1}(p_hat)
    
    # Step 4: Calculate central moments (mapped to ±1 domain)
    mu_B = 1 - 2 * p_hat
    sigma2_B = 1 - mu_B**2  # By definition, when b∈{0,1}, σ_B² = 1 - μ_B²
    
    # ================== Step 2: Correlation Inversion ==================
    # Step 6: Calculate linearization gain
    K_ru = (1 / sigma2_B) * (2 / np.pi) * np.exp(-r_u**2)
    
    # Steps 7-10: Calculate digital ACF vector ρ_B and analog ACF vector ρ_V
    acf = compute_acf_corrected(b, max_lag)
    rho_B = acf[1:max_lag+1]  # τ from 1 to K
    
    # Step 11: Calculate ρ_V
    rho_V = rho_B / K_ru
    
    # ================== Step 3: Spectral Regression (Log-Log Domain) ==================
    # Steps 12-13: Construct regression vectors
    tau = np.arange(1, max_lag + 1)
    valid_idx = (np.abs(rho_V) > 1e-10) & (tau > 0)
    
    if np.sum(valid_idx) < 2:
        print("Warning: Not enough valid points for regression")
        C = 0
        beta = 0
        R2 = 0
    else:
        Y = np.log(np.abs(rho_V[valid_idx]))
        X = np.log(tau[valid_idx])
        
        # Step 14: Solve linear model Y = C - βX
        X_matrix = np.column_stack([np.ones_like(X), X])
        beta_vec = np.linalg.lstsq(X_matrix, Y, rcond=None)[0]
        C = beta_vec[0]
        beta = -beta_vec[1]  # Because model is Y = C - βX
        
        # Calculate R²
        Y_pred = C - beta * X
        SS_res = np.sum((Y - Y_pred)**2)
        SS_tot = np.sum((Y - np.mean(Y))**2)
        R2 = 1 - SS_res / SS_tot if SS_tot > 0 else 0
    
    # ================== Step 4: Parameter Recovery ==================
    # Step 15: Calculate A
    A = np.exp(C)
    
    # Step 16: Calculate r_f
    if A >= 1:
        r_f = np.inf
    elif A <= 0:
        r_f = 0
    else:
        r_f = np.sqrt(A / (1 - A))
    
    # Calculate gamma = 1 - beta
    gamma = 1 - beta
    
    # Calculate fitted ρ_V curve
    rho_V_fit = A * np.power(tau, -beta)
    
    # Print results
    print(f"  p̂ = {p_hat:.4f}, r_u = {r_u:.4f}, r_f = {r_f:.4f}")
    print(f"  β = {beta:.4f}, γ = 1-β = {gamma:.4f}")
    print(f"  A = {A:.6f}, K(r_u) = {K_ru:.6f}, R² = {R2:.4f}")
    print(f"  Signal length: {N} samples\n")
    
    # Return results
    results = {
        'N': N,
        'p_hat': p_hat,
        'r_u': r_u,
        'mu_B': mu_B,
        'sigma2_B': sigma2_B,
        'K_ru': K_ru,
        'tau': tau,
        'rho_B': rho_B,
        'rho_V': rho_V,
        'A': A,
        'C': C,
        'beta': beta,
        'gamma': gamma,
        'r_f': r_f,
        'R2': R2,
        'rho_V_fit': rho_V_fit
    }
    
    return results

def analyze_single_file(folder_path, count_idx, max_lag=100):
    """
    Analyze 1.bin file in a single folder
    
    Parameters:
    -----------
    folder_path : str
        Main folder path (e.g., V1.3_F1_T27)
    count_idx : int
        Selected subfolder index (e.g., 0 for count_0)
    max_lag : int
        Maximum lag
        
    Returns:
    --------
    results : dict
        Dictionary containing all calculation results
    """
    
    # Build file path
    subfolder_name = f'count_{count_idx}'
    file_path = os.path.join(folder_path, subfolder_name, '1.bin')
    
    if not os.path.exists(file_path):
        print(f"ERROR: File {file_path} does not exist!")
        return None
    
    print(f"Analyzing: {folder_path}/{subfolder_name}/1.bin")
    
    # Read binary file
    bits = read_binary_file(file_path)
    
    # Extract platform parameters according to Algorithm 1
    results = extract_platform_parameters(bits, max_lag)
    
    # Add folder information
    results['folder_name'] = os.path.basename(folder_path)
    results['subfolder'] = subfolder_name
    results['freq_str'] = extract_frequency_from_folder(results['folder_name'])
    
    return results

def analyze_multiple_files(folder_list, count_idx=0, max_lag=100):
    """
    Analyze data from multiple folders
    
    Parameters:
    -----------
    folder_list : list
        List of folder names
    count_idx : int
        Selected subfolder index
    max_lag : int
        Maximum lag
        
    Returns:
    --------
    all_results : dict
        Dictionary containing all analysis results
    """
    
    all_results = {}
    
    for folder_name in folder_list:
        if not os.path.exists(folder_name):
            print(f"WARNING: Folder {folder_name} does not exist, skipping")
            continue
        
        results = analyze_single_file(folder_name, count_idx, max_lag)
        if results is not None:
            all_results[folder_name] = results
    
    return all_results

# ====================================================
# Create professional dual plot (right plot shows β)
# ====================================================

def create_beautiful_dual_plot(all_results):
    """
    Create professional dual plot: left plot shows ρ_B-τ in log scale (starting from τ=2, 
    without absolute value), right plot shows ρ_V data points and fitted curve in linear scale.
    Right plot displays β instead of γ.
    """
    
    # Create output directory
    output_dir = 'Beautiful_Dual_Plot'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Define color scheme
    colors = [
        '#1f77b4',  # Blue
        '#ff7f0e',  # Orange
        '#2ca02c',  # Green
        '#d62728',  # Red
        '#9467bd'   # Purple
    ]
    
    # Define marker styles
    markers = ['o', 's', '^', 'D', 'v']
    
    # Define line styles
    line_styles = ['-', '--', '-.', ':', '-']
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # ================== Left plot: ρ_B vs τ in log-log scale (starting from τ=2) ==================
    ax_left = axes[0]
    
    for i, (folder_name, results) in enumerate(all_results.items()):
        # Extract data
        tau = results['tau']
        rho_B = results['rho_B']
        
        # Get frequency string (e.g., '1GHz')
        freq_str = results['freq_str']
        
        # Plot curve - start from τ=2, show ρ_B (without absolute value)
        # Note: tau[0] corresponds to τ=1, so start from index 1 corresponding to τ=2
        ax_left.loglog(tau[1:], rho_B[1:], 
                      color=colors[i % len(colors)],
                      linestyle=line_styles[i % len(line_styles)],
                      linewidth=2.5,
                      alpha=0.85,
                      label=freq_str)
        
        # Add data point markers - start from τ=2, mark every 5 points
        ax_left.loglog(tau[1:][::5], rho_B[1:][::5],
                      color=colors[i % len(colors)],
                      marker=markers[i % len(markers)],
                      markersize=8,
                      markeredgecolor='white',
                      markeredgewidth=1.5,
                      linestyle='None')
    
    # Style left plot
    ax_left.set_xlabel('Lag τ', fontsize=14, fontweight='bold', fontfamily='Times New Roman')
    ax_left.set_ylabel(r'$\rho_B(\tau)$', fontsize=14, fontweight='bold', fontfamily='Times New Roman')
    ax_left.set_title(r'Autocorrelation $\rho_B(\tau)$ in Log-Log Scale ', 
                     fontsize=16, fontweight='bold', pad=25, fontfamily='Times New Roman')
    
    # Add grid
    ax_left.grid(True, which='both', alpha=0.3, linestyle='--', linewidth=0.5)
    ax_left.grid(True, which='major', alpha=0.5, linestyle='-', linewidth=0.8)
    
    # Set axis range - start from 2
    ax_left.set_xlim(1.8, 110)  # Start from τ=2, so set left boundary to 1.8
    
    # Add legend - slightly larger
    legend_left = ax_left.legend(title='Frequency', fontsize=12, title_fontsize=13,
                                loc='upper right', frameon=True, framealpha=0.95,
                                edgecolor='black', facecolor='white',
                                borderpad=1.0,  # Slightly increase border padding
                                labelspacing=0.8,  # Slightly increase label spacing
                                handlelength=2.0,  # Slightly increase line length
                                handletextpad=0.5)  # Slightly increase text-line spacing
    
    # Set legend title and label fonts
    for text in legend_left.get_texts():
        text.set_fontfamily('Times New Roman')
    legend_left.get_title().set_fontfamily('Times New Roman')
    
    legend_left.get_frame().set_linewidth(1.5)
    legend_left.get_frame().set_boxstyle('round,pad=0.3')  # Slightly increase
    
    # ================== Right plot: ρ_V data points and fitted curve in linear scale (only τ=1-30) ==================
    ax_right = axes[1]
    
    for i, (folder_name, results) in enumerate(all_results.items()):
        # Extract data
        tau = results['tau']
        rho_V = results['rho_V']
        rho_V_fit = results['rho_V_fit']
        beta_val = results['beta']  # Modification: use β instead of γ
        R2_val = results['R2']
        
        # Get frequency string
        freq_str = results['freq_str']
        
        # Only show τ=1 to 30 data
        tau_mask = tau <= 30
        tau_30 = tau[tau_mask]
        rho_V_30 = rho_V[tau_mask]
        rho_V_fit_30 = rho_V_fit[tau_mask]
        
        # Plot fitted curve (only to 30)
        ax_right.plot(tau_30, rho_V_fit_30,
                     color=colors[i % len(colors)],
                     linestyle=line_styles[i % len(line_styles)],
                     linewidth=3,
                     alpha=0.9,
                     # Modification: display β instead of γ
                     label=f'{freq_str}: β={beta_val:.3f}, R²={R2_val:.3f}')
        
        # Plot actual data points, skip first point (if first point is abnormal)
        # Only plot from second point onward, and only to 30
        if len(tau_30) > 1:
            ax_right.scatter(tau_30[1:], rho_V_30[1:],
                           color=colors[i % len(colors)],
                           marker=markers[i % len(markers)],
                           s=60,  # Point size
                           alpha=0.7,
                           edgecolor='white',
                           linewidth=1.5,
                           zorder=5)  # Ensure points are above curves
    
    # Style right plot
    ax_right.set_xlabel('Lag τ', fontsize=14, fontweight='bold', fontfamily='Times New Roman')
    ax_right.set_ylabel(r'$\rho_V(\tau)$', fontsize=14, fontweight='bold', fontfamily='Times New Roman')
    ax_right.set_title(r'Normalized Autocorrelation $\rho_V(\tau)$ with Fitting ', 
                      fontsize=16, fontweight='bold', pad=25, fontfamily='Times New Roman')
    
    # Add grid
    ax_right.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Set axis range: x-axis only shows 1-30
    ax_right.set_xlim(0, 31)
    
    # Set x-axis ticks
    ax_right.set_xticks(np.arange(0, 31, 5))
    ax_right.set_xticks(np.arange(0, 31, 1), minor=True)
    
    # Add legend - annotate β and R² values, slightly enlarge legend box
    # Modification: legend title changed to β
    legend_right = ax_right.legend(title='Frequency (β, R²)', fontsize=11,  # Slightly reduce font to fit more content
                                  title_fontsize=13,  # Slightly increase title font
                                  loc='upper right',
                                  frameon=True, framealpha=0.95,
                                  edgecolor='black', facecolor='white',
                                  borderpad=1.0,  # Slightly increase border padding
                                  labelspacing=0.8,  # Slightly increase label spacing
                                  handlelength=2.0,  # Slightly increase line length
                                  handletextpad=0.5)  # Slightly increase text-line spacing
    
    # Set legend title and label fonts
    for text in legend_right.get_texts():
        text.set_fontfamily('Times New Roman')
    legend_right.get_title().set_fontfamily('Times New Roman')
    
    # Slightly enlarge legend box
    legend_right.get_frame().set_linewidth(1.5)
    legend_right.get_frame().set_boxstyle('round,pad=0.3')  # Slightly increase
    
    # Adjust subplot spacing
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Reserve space for overall title
    
    # Add overall title
    fig.suptitle(r'Autocorrelation Analysis: $\rho_B(\tau)$ and $\rho_V(\tau)$', 
                fontsize=20, fontweight='bold', y=0.98, fontfamily='Times New Roman')
    
    # Save figure
    save_path = os.path.join(output_dir, 'beautiful_dual_plot.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    
    # Save high-resolution version
    save_path_highres = os.path.join(output_dir, 'beautiful_dual_plot_highres.png')
    plt.savefig(save_path_highres, dpi=600, bbox_inches='tight', facecolor='white')
    
    # Save as PDF (suitable for publication)
    save_path_pdf = os.path.join(output_dir, 'beautiful_dual_plot.pdf')
    plt.savefig(save_path_pdf, bbox_inches='tight', facecolor='white')
    
    print(f"\nDual plot saved to:")
    print(f"  Standard resolution: {save_path}")
    print(f"  High resolution: {save_path_highres}")
    print(f"  PDF version: {save_path_pdf}")
    
    plt.show()
    
    return fig, axes

# ====================================================
# Create parameter table
# ====================================================

def create_parameter_table(all_results):
    """
    Create table containing gamma (γ=1-β), r_u, and r_f
    """
    
    # Create output directory
    output_dir = 'Parameter_Tables'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Prepare data
    table_data = []
    
    for folder_name, results in all_results.items():
        # Get frequency string
        freq_str = results['freq_str']
        
        # Extract required parameters
        row_data = {
            'Frequency': freq_str,
            'Folder': results['folder_name'],
            'N': results['N'],
            'p_hat': results['p_hat'],
            'r_u': results['r_u'],
            'r_f': results['r_f'],
            'beta': results['beta'],
            'gamma': results['gamma'],
            'A': results['A'],
            'K_ru': results['K_ru'],
            'R2': results['R2']
        }
        table_data.append(row_data)
    
    # Sort by frequency (1GHz, 2GHz, ...)
    table_data.sort(key=lambda x: int(x['Frequency'].replace('GHz', '')))
    
    # Create DataFrame
    summary_df = pd.DataFrame(table_data)
    
    # Set display format
    pd.set_option('display.precision', 4)
    
    # Print table
    print("\n" + "="*80)
    print("PARAMETER TABLE: γ = 1-β, r_u, r_f")
    print("="*80)
    print(summary_df.to_string(index=False))
    print("="*80)
    
    # Save as CSV file
    csv_file = os.path.join(output_dir, 'parameter_table.csv')
    summary_df.to_csv(csv_file, index=False, encoding='utf-8-sig')
    print(f"\nParameter table saved to: {csv_file}")
    
    # Save as LaTeX table - use pure ASCII characters
    latex_file = os.path.join(output_dir, 'parameter_table.tex')
    # Create column names without special characters
    latex_df = summary_df.copy()
    latex_df.columns = ['Frequency', 'Folder', 'N', 'p_hat', 'r_u', 'r_f', 
                       'beta', 'gamma', 'A', 'K_ru', 'R2']
    
    with open(latex_file, 'w', encoding='utf-8') as f:
        latex_str = latex_df.to_latex(index=False, float_format="%.4f")
        f.write(latex_str)
    print(f"LaTeX table saved to: {latex_file}")
    
    # Create beautiful HTML table
    html_file = os.path.join(output_dir, 'parameter_table.html')
    # Create display column names
    display_df = summary_df.copy()
    display_df.columns = ['Frequency', 'Folder', 'N', 'p̂', 'r_u', 'r_f', 
                         'β', 'γ = 1-β', 'A', 'K(r_u)', 'R²']
    
    html_content = display_df.to_html(index=False, float_format=lambda x: f'{x:.4f}')
    
    # Add CSS styling
    styled_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <style>
            table {{
                border-collapse: collapse;
                width: 100%;
                font-family: Arial, sans-serif;
                margin: 20px 0;
            }}
            th {{
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                padding: 12px;
                text-align: center;
                border: 1px solid #ddd;
            }}
            td {{
                padding: 10px;
                text-align: center;
                border: 1px solid #ddd;
            }}
            tr:nth-child(even) {{
                background-color: #f2f2f2;
            }}
            tr:hover {{
                background-color: #e6f7ff;
            }}
            .highlight {{
                background-color: #fffacd;
                font-weight: bold;
            }}
            caption {{
                font-size: 1.5em;
                margin: 10px;
                font-weight: bold;
                color: #333;
            }}
        </style>
    </head>
    <body>
        <h2>Platform Parameter Extraction Results</h2>
        <p>Analysis based on Algorithm 1: γ = 1-β, r_u, r_f</p>
        {html_content}
    </body>
    </html>
    """
    
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(styled_html)
    print(f"HTML table saved to: {html_file}")
    
    # Create simplified table (only key parameters)
    simplified_data = []
    for row in table_data:
        simplified_data.append({
            'Frequency': row['Frequency'],
            'r_u': row['r_u'],
            'r_f': row['r_f'],
            'beta': row['beta'],
            'gamma': row['gamma'],
            'R2': row['R2']
        })
    
    simplified_df = pd.DataFrame(simplified_data)
    
    # Save simplified table
    simplified_csv = os.path.join(output_dir, 'simplified_parameter_table.csv')
    simplified_df.to_csv(simplified_csv, index=False, encoding='utf-8-sig')
    print(f"Simplified parameter table saved to: {simplified_csv}")
    
    # Print simplified table
    print("\n" + "="*60)
    print("SIMPLIFIED PARAMETER TABLE (Key Parameters)")
    print("="*60)
    print(simplified_df.to_string(index=False))
    print("="*60)
    
    return summary_df, simplified_df

# ====================================================
# Create parameter visualization charts
# ====================================================

def create_parameter_visualization(all_results):
    """
    Create parameter visualization charts
    """
    
    # Create output directory
    output_dir = 'Parameter_Visualizations'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Prepare data
    frequencies = []
    ru_values = []
    rf_values = []
    beta_values = []
    gamma_values = []
    R2_values = []
    
    # Extract data and sort by frequency
    sorted_items = sorted(all_results.items(), 
                         key=lambda x: int(x[1]['freq_str'].replace('GHz', '')))
    
    for folder_name, results in sorted_items:
        frequencies.append(results['freq_str'])
        ru_values.append(results['r_u'])
        rf_values.append(results['r_f'])
        beta_values.append(results['beta'])
        gamma_values.append(results['gamma'])
        R2_values.append(results['R2'])
    
    # Define color scheme
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # Create multi-subplot visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Subplot 1: Comparison of r_u and r_f
    ax1 = axes[0, 0]
    x_pos = np.arange(len(frequencies))
    width = 0.35
    
    bars1 = ax1.bar(x_pos - width/2, ru_values, width, label='r_u', color=colors[0], alpha=0.8)
    bars2 = ax1.bar(x_pos + width/2, rf_values, width, label='r_f', color=colors[1], alpha=0.8)
    
    ax1.set_xlabel('Frequency', fontsize=12, fontweight='bold', fontfamily='Times New Roman')
    ax1.set_ylabel('Parameter Value', fontsize=12, fontweight='bold', fontfamily='Times New Roman')
    ax1.set_title('Bias Ratio (r_u) and Noise Ratio (r_f)', 
                 fontsize=14, fontweight='bold', pad=20, fontfamily='Times New Roman')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(frequencies, fontfamily='Times New Roman')
    ax1.legend(prop={'family': 'Times New Roman'})
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9, fontfamily='Times New Roman')
    
    # Subplot 2: Comparison of β and γ
    ax2 = axes[0, 1]
    bars3 = ax2.bar(x_pos - width/2, beta_values, width, label='β', color=colors[2], alpha=0.8)
    bars4 = ax2.bar(x_pos + width/2, gamma_values, width, label='γ = 1-β', color=colors[3], alpha=0.8)
    
    ax2.set_xlabel('Frequency', fontsize=12, fontweight='bold', fontfamily='Times New Roman')
    ax2.set_ylabel('Parameter Value', fontsize=12, fontweight='bold', fontfamily='Times New Roman')
    ax2.set_title('Power-Law Exponent (β) and γ = 1-β', 
                 fontsize=14, fontweight='bold', pad=20, fontfamily='Times New Roman')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(frequencies, fontfamily='Times New Roman')
    ax2.legend(prop={'family': 'Times New Roman'})
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bars in [bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9, fontfamily='Times New Roman')
    
    # Subplot 3: Relationship between γ and R²
    ax3 = axes[1, 0]
    scatter = ax3.scatter(gamma_values, R2_values, s=150, c=range(len(frequencies)), 
                         cmap='viridis', alpha=0.7, edgecolors='black', linewidth=1.5)
    
    # Add data point labels
    for i, (gamma, R2, freq) in enumerate(zip(gamma_values, R2_values, frequencies)):
        ax3.text(gamma, R2, f' {freq}', fontsize=11, fontweight='bold', va='center', fontfamily='Times New Roman')
    
    ax3.set_xlabel('γ', fontsize=12, fontweight='bold', fontfamily='Times New Roman')
    ax3.set_ylabel('R²', fontsize=12, fontweight='bold', fontfamily='Times New Roman')
    ax3.set_title('Relationship between γ and R²', 
                 fontsize=14, fontweight='bold', pad=20, fontfamily='Times New Roman')
    ax3.grid(True, alpha=0.3)
    
    # Subplot 4: Relationship between β and γ
    ax4 = axes[1, 1]
    # Theoretical relationship line: γ = 1 - β
    beta_range = np.linspace(min(beta_values)-0.1, max(beta_values)+0.1, 100)
    gamma_theory = 1 - beta_range
    ax4.plot(beta_range, gamma_theory, 'r--', alpha=0.7, linewidth=2, label='γ = 1 - β')
    
    # Actual data points
    scatter2 = ax4.scatter(beta_values, gamma_values, s=150, c=range(len(frequencies)), 
                          cmap='plasma', alpha=0.7, edgecolors='black', linewidth=1.5, label='Data')
    
    # Add data point labels
    for i, (beta, gamma, freq) in enumerate(zip(beta_values, gamma_values, frequencies)):
        ax4.text(beta, gamma, f' {freq}', fontsize=11, fontweight='bold', va='center', fontfamily='Times New Roman')
    
    ax4.set_xlabel('β', fontsize=12, fontweight='bold', fontfamily='Times New Roman')
    ax4.set_ylabel('γ', fontsize=12, fontweight='bold', fontfamily='Times New Roman')
    ax4.set_title('Relationship between β and γ', 
                 fontsize=14, fontweight='bold', pad=20, fontfamily='Times New Roman')
    ax4.legend(prop={'family': 'Times New Roman'})
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.suptitle('Platform Parameter Analysis by Frequency', 
                fontsize=16, fontweight='bold', y=0.98, fontfamily='Times New Roman')
    
    # Save figure
    save_path = os.path.join(output_dir, 'parameter_visualization.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\nParameter visualization saved to: {save_path}")
    
    plt.show()
    
    return fig

# ====================================================
# More flexible main function version
# ====================================================

def main_flexible():
    """More flexible main function version, directly looking for 1.bin in specified folders"""
    
    # Define folder list to analyze
    folder_list = [
        'V1.3_F1_T27',  # 1GHz
        'V1.3_F2_T27',  # 2GHz
        'V1.3_F3_T27',  # 3GHz
        'V1.3_F4_T27',  # 4GHz
        'V1.3_F5_T27'   # 5GHz
    ]
    
    # Check if folders exist
    existing_folders = []
    for folder in folder_list:
        if os.path.exists(folder):
            existing_folders.append(folder)
        else:
            print(f"WARNING: Folder {folder} does not exist, skipping")
    
    if not existing_folders:
        print("ERROR: No folders found!")
        return
    
    print("=" * 70)
    print("PLATFORM PARAMETER EXTRACTION (Algorithm 1)")
    print("=" * 70)
    print(f"Analyzing {len(existing_folders)} folders:")
    print("Directly looking for 1.bin in each folder")
    for folder in existing_folders:
        freq_str = extract_frequency_from_folder(folder)
        print(f"  - {folder} ({freq_str})")
    print()
    
    # Maximum lag
    max_lag = 100
    
    # Analyze all folders - look for 1.bin directly in each folder
    all_results = {}
    for folder_name in existing_folders:
        # Build file path directly in the folder
        file_path = os.path.join(folder_name, '1.bin')
        
        if not os.path.exists(file_path):
            print(f"ERROR: File {file_path} does not exist!")
            continue
        
        print(f"Analyzing: {folder_name}/1.bin")
        
        # Read binary file
        bits = read_binary_file(file_path)
        
        # Extract platform parameters according to Algorithm 1
        results = extract_platform_parameters(bits, max_lag)
        
        # Add folder information
        results['folder_name'] = os.path.basename(folder_name)
        results['subfolder'] = 'root'  # Since we're looking directly in folder
        results['freq_str'] = extract_frequency_from_folder(results['folder_name'])
        
        all_results[folder_name] = results
    
    if not all_results:
        print("ERROR: No valid data found!")
        return
    
    print("\n" + "=" * 70)
    print("ANALYSIS SUMMARY")
    print("=" * 70)
    print("Direct file analysis completed")
    print()
    
    print("\n" + "=" * 70)
    print("CREATING BEAUTIFUL DUAL PLOT")
    print("=" * 70)
    
    # Create professional dual plot
    create_beautiful_dual_plot(all_results)
    
    print("\n" + "=" * 70)
    print("CREATING PARAMETER TABLE")
    print("=" * 70)
    
    # Create parameter table
    summary_df, simplified_df = create_parameter_table(all_results)
    
    print("\n" + "=" * 70)
    print("CREATING PARAMETER VISUALIZATION")
    print("=" * 70)
    
    # Create parameter visualization
    create_parameter_visualization(all_results)
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE!")
    print("=" * 70)
    
    return all_results, summary_df, simplified_df

# Run main function
if __name__ == '__main__':
    all_results, summary_df, simplified_df = main_flexible()