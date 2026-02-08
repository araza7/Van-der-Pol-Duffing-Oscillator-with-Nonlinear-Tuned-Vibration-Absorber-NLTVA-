import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# ==========================================
# 1. PARAMETERS 
# ==========================================
m1 = 1.0; k1 = 1.0; knl1 = 0.3; epsilon = 0.05

# Optimal Tuning (Eq 3.10 & 6.1)
gamma_opt = 1.0 / np.sqrt(1.0 + epsilon)
mu2_opt = 0.5 * np.sqrt(epsilon / (1.0 + epsilon))
alpha3 = knl1 / k1
beta3_opt = (epsilon / (1.0 + epsilon)**2) * alpha3

# Physical Values
m2 = epsilon * m1
omega_n1 = np.sqrt(k1 / m1)
omega_n2 = gamma_opt * omega_n1
k2 = m2 * omega_n2**2
c2 = 2 * mu2_opt * m2 * omega_n2
knl2 = beta3_opt * k1 * epsilon
mu1_limit = np.sqrt(epsilon) / 2.0

# ==========================================
# 2. EQUATIONS OF MOTION
# ==========================================
def vdpd_system(state, t, mu1, use_absorber):
    q1, d1, q2, d2 = state
    c1 = 2 * mu1 * np.sqrt(k1 * m1)
    
    f_damp = c1 * (q1**2 - 1.0) * d1
    f_stiff = k1 * q1 + knl1 * q1**3
    
    if use_absorber:
        qd = q1 - q2
        dd = d1 - d2
        f_abs = c2*dd + k2*qd + knl2*(qd**3)
    else:
        f_abs = 0.0

    dd1 = (-f_damp - f_stiff - f_abs) / m1
    dd2 = f_abs / m2 if use_absorber else 0.0
    return [d1, dd1, d2, dd2]

# ==========================================
# 3. SIMULATION (Pseudo-Continuation)
# ==========================================
mu_vals = np.linspace(0, 0.20, 50)
t_sim = np.linspace(0, 500, 5000)

amp_no = []
amp_yes = []

print("Running Simulation...")

# CASE 1: No Absorber
x0 = [0.1, 0, 0, 0]
for mu in mu_vals:
    sol = odeint(vdpd_system, x0, t_sim, args=(mu, False))
    steady = sol[-1000:, 0]
    amp_no.append((np.max(steady) - np.min(steady))/2)

# CASE 2: With NLTVA (Forward Sweep to catch the Jump)
current_state = [0.001, 0, 0, 0] 

for mu in mu_vals:
    # Nudge if stuck at perfect zero to allow instability to trigger
    if abs(current_state[0]) < 1e-5:
        current_state[0] = 1e-4

    sol = odeint(vdpd_system, current_state, t_sim, args=(mu, True))
    
    # Save state for next loop
    current_state = sol[-1, :]
    
    steady = sol[-1000:, 0]
    amp_yes.append((np.max(steady) - np.min(steady))/2)

# ==========================================
# 4. PLOTTING (Bifurcation + Time Series Only)
# ==========================================
# Time Series Calculation (at mu = 0.08)
t_ts = np.linspace(0, 150, 2000)
ts_u = odeint(vdpd_system, [0.1,0,0,0], t_ts, args=(0.08, False))
ts_c = odeint(vdpd_system, [0.1,0,0,0], t_ts, args=(0.08, True))

# Create 2 Subplots stacked vertically
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

# --- Plot 1: Bifurcation Diagram ---
ax1.plot(mu_vals, amp_no, 'k--', label='Primary Only (Uncontrolled)')
ax1.plot(mu_vals, amp_yes, 'r-o', linewidth=2, label='With NLTVA')
ax1.axvline(mu1_limit, color='b', linestyle=':', linewidth=2, label=r'Theoretical Limit $\sqrt{\epsilon}/2$')

# Annotations
ax1.text(0.01, 1.85, "VdPD Limit Cycle", fontweight='bold')

ax1.set_title(f'Bifurcation Diagram: Global Dynamics ', fontsize=14)
ax1.set_ylabel(r'Amplitude $q_1$', fontsize=12)
ax1.set_xlabel(r'Instability Parameter $\mu_1$', fontsize=12)
ax1.grid(True, alpha=0.3)
ax1.legend(loc='lower right')
ax1.set_xlim(0, 0.2)

# --- Plot 2: Time Response ---
ax2.plot(t_ts, ts_u[:,0], 'k--', alpha=0.4, label='No Absorber')
ax2.plot(t_ts, ts_c[:,0], 'r', linewidth=1.5, label='With NLTVA')

ax2.set_title(r'Time Response B3 = 0.01360', fontsize=14)
ax2.set_xlabel('Time', fontsize=12)
ax2.set_ylabel(r'Displacement $q_1$', fontsize=12)
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.tight_layout()
plt.show()
