import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches

def plot_nst_concept():
    # Setup professional style
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    plt.rcParams['mathtext.fontset'] = 'cm'
    plt.rcParams['font.size'] = 12
    
    fig = plt.figure(figsize=(12, 6))
    
    # --- Panel A: TTFS Encoding ---
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.set_title("(a) Time-to-First-Spike (TTFS) Encoding", fontsize=14, pad=20, loc='left')
    
    # Draw characteristic curve t = Tmax * (1 - x)
    x = np.linspace(0, 1, 100)
    t = 50 * (1 - x)
    
    ax1.plot(x, t, color='#333333', linewidth=2, label=r'$t = T_{max}(1 - x)$')
    
    # Examples
    examples = [(0.2, 40, 'Weak Feature\n(Late Spike)', '#e74c3c'), 
                (0.8, 10, 'Strong Feature\n(Early Spike)', '#2ecc71')]
    
    for val, time, label, color in examples:
        # Dashed lines
        ax1.plot([val, val], [0, time], linestyle='--', color=color, alpha=0.6)
        ax1.plot([0, val], [time, time], linestyle='--', color=color, alpha=0.6)
        # Point
        ax1.scatter([val], [time], s=100, color=color, zorder=5)
        # Spike representation on Y-axis
        ax1.arrow(-0.05, time, 0.05, 0, head_width=0, head_length=0, color=color, linewidth=2)
        ax1.text(val + 0.05, time + 2, label, fontsize=10, color=color, va='bottom')

    ax1.set_xlabel("Normalized Feature Value ($x$)", fontsize=12)
    ax1.set_ylabel("Spike Time ($t$)", fontsize=12)
    ax1.set_xlim(0, 1.1)
    ax1.set_ylim(0, 55)
    ax1.grid(True, linestyle=':', alpha=0.3)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # --- Panel B: Temporal Decision Logic ---
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.set_title("(b) Spiking Node Dynamics", fontsize=14, pad=20, loc='left')
    
    # Clean axis
    ax2.set_xlim(0, 60)
    ax2.set_ylim(0, 4)
    ax2.axis('off')
    
    # Draw timeline
    ax2.annotate('', xy=(55, 2), xytext=(5, 2), 
                 arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    ax2.text(56, 1.9, "Simulation Time ($t$)", fontsize=12)
    
    # Draw Tmax
    ax2.axvline(x=50, ymin=0.4, ymax=0.6, color='gray', linestyle='--')
    ax2.text(50, 2.5, r'$T_{max}$', ha='center')
    
    # Draw Threshold Theta
    theta = 25
    ax2.axvline(x=theta, ymin=0.3, ymax=0.7, color='#3498db', linewidth=3)
    ax2.text(theta, 3.2, r'Threshold $\theta_k$', color='#3498db', ha='center', fontsize=12, fontweight='bold')
    
    # Scenario 1: Early Spike (Active)
    t_spike = 15
    ax2.scatter([t_spike], [2], s=150, color='#2ecc71', marker='|', linewidth=4, zorder=10)
    ax2.text(t_spike, 1.5, r'Spike $t_x$', color='#2ecc71', ha='center', fontweight='bold')
    
    # Arrow for branch
    style = "Simple, tail_width=0.5, head_width=4, head_length=8"
    kw = dict(arrowstyle=style, color="#2ecc71")
    path1 = patches.FancyArrowPatch((t_spike, 2.1), (theta-2, 2.8), connectionstyle="arc3,rad=.3", **kw)
    ax2.add_patch(path1)
    ax2.text((t_spike+theta)/2, 2.9, "Trigger\nRight Child", color='#2ecc71', ha='center', fontsize=11, fontweight='bold')

    # Scenario 2: Late/No Spike (Passive) - Conceptual
    ax2.text(40, 1.5, "No Spike < $\theta$", color='#e74c3c', ha='center', alpha=0.6)
    
    kw_late = dict(arrowstyle=style, color="#e74c3c", alpha=0.6)
    path2 = patches.FancyArrowPatch((theta, 2.1), (45, 2.8), connectionstyle="arc3,rad=-.3", **kw_late)
    ax2.add_patch(path2)
    ax2.text(38, 2.9, "Timeout\nLeft Child", color='#e74c3c', ha='center', fontsize=11, fontweight='bold')
    
    # Add Equation
    ax2.text(30, 0.5, 
             r"$(B, t_{out}) = \{ \text{Right} \to t_x + \delta, \text{ if } t_x < \theta_k; \text{Left} \to \theta_k + \delta, \text{ else} \}$", 
             fontsize=12, ha='center', bbox=dict(facecolor='#f8f9fa', edgecolor='none', pad=10))

    plt.tight_layout()
    plt.savefig('nst_mechanism_concept.png', dpi=300, bbox_inches='tight')
    plt.savefig('nst_mechanism_concept.pdf', bbox_inches='tight')
    print("Figure generated: nst_mechanism_concept.png")

if __name__ == "__main__":
    plot_nst_concept()
