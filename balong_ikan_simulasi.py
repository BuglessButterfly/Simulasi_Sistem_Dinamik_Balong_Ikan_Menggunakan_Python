"""
====================================================
SIMULASI SISTEM DINAMIK BALONG IKAN
====================================================
Nama  : Intan Telaumbanua
NIM   : 301230016
Kelas : 5B - Teknik Informatika
Mata Kuliah : Praktikum Pemodelan dan Simulasi

====================================================
TUJUAN SIMULASI
====================================================
1. Menganalisis dinamika biomassa ikan (B) dan kadar oksigen terlarut (DO)
   terhadap waktu menggunakan model sistem dinamik berbasis ODE.
2. Mengamati pengaruh perubahan input pakan (u) terhadap kestabilan sistem.
3. Mensimulasikan respons steady-state, step, dan sinusoida.
4. Mengidentifikasi karakteristik sistem dan pengaruh kontrol pakan.

====================================================
DESKRIPSI STUDI KASUS
====================================================
Balong ikan dengan volume 1000 m³ memiliki debit air masuk dan keluar konstan.
Kadar oksigen (DO) dalam air dipengaruhi oleh:
- suplai dari air masuk dan udara (reaerasi),
- serta konsumsi oleh respirasi ikan.
Biomassa ikan meningkat dengan pakan, namun juga mengonsumsi oksigen.

====================================================
ASUMSI-ASUMSI MODEL
====================================================
1. Air dalam kolam homogen (tidak ada gradien spasial → ODE).
2. Suhu dan tekanan dianggap konstan.
3. Pertumbuhan ikan mengikuti model logistik.
4. Tidak ada kehilangan massa ikan selain mortalitas alami.
5. Reaerasi mengikuti hukum linear terhadap perbedaan DO.
6. Debit air masuk sama dengan debit keluar (steady flow).
7. Tidak ada efek gangguan eksternal selain variasi input pakan.

====================================================
KLASIFIKASI VARIABEL
====================================================
Variabel Keadaan (State Variables):
- B(t) : Biomassa ikan (kg)
- DO(t): Dissolved Oxygen (mg/L)

Variabel Masukan (Input Variables):
- u(t): Input pakan ikan (% dari nilai nominal)
  • Dapat berupa konstanta, step, atau sinusoida.

Variabel Keluaran (Output Variables):
- B dan DO (diamati untuk menganalisis kestabilan sistem)

====================================================
DERAJAT KEBEBASAN (DOF)
====================================================
Jumlah variabel tak diketahui  : 2 (B, DO)
Jumlah persamaan diferensial   : 2
→ DOF = 0 → Sistem terdefinisi lengkap (well-posed).

====================================================
PERSAMAAN MODEL (Mass & Species Balance)
====================================================
1. Pertumbuhan biomassa ikan (model logistik + pakan):
   dB/dt = r * B * (1 - B/K) - m * B
   dengan r = r₀ (1 + α * u)

2. Perubahan kadar DO:
   dDO/dt = (Q/V) * (O2_in - DO) + k_reaer * (O2_sat - DO) - k_resp * B

====================================================
"""

# --- Import Library ---
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import pandas as pd

# --- Parameter Fisik dan Operasional Balong Ikan ---
V_m3 = 1000.0             # Volume kolam (m³)
V_L = V_m3 * 1000.0       # Volume dalam liter
Q_m3_per_day = 1.0        # Debit air masuk (m³/hari)
Q_L_per_day = Q_m3_per_day * 1000.0  # Debit air dalam liter/hari

O2_in = 8.0               # Kadar O2 air masuk (mg/L)
O2_sat = 9.0              # Kadar O2 jenuh (mg/L)
r0 = 0.15                 # Laju pertumbuhan dasar ikan (1/hari)
K = 500.0                 # Kapasitas maksimum (kg)
m = 0.02                  # Laju kematian ikan (1/hari)
alpha_feed = 0.4          # Pengaruh pakan terhadap pertumbuhan
k_resp = 0.01             # Konsumsi O2 per kg ikan (mg/L per kg)
k_reaer = 0.5             # Koefisien reaerasi (1/hari)

# ==============================================================
# FUNGSI MODEL SISTEM DINAMIK (ODE)
# ==============================================================

def pond_model(states, t, u_func):
    B, DO = states
    u = u_func(t)

    # Model pertumbuhan biomassa ikan
    r = r0 * (1 + alpha_feed * u)
    dB_dt = r * B * (1 - B / K) - m * B

    # Model kadar oksigen terlarut
    dilution = (Q_L_per_day / V_L) * (O2_in - DO)
    reaeration = k_reaer * (O2_sat - DO)
    respiration = -k_resp * B
    dDO_dt = dilution + reaeration + respiration

    return [dB_dt, dDO_dt]


# ==============================================================
# DEFINISI INPUT KONTROL (u)
# ==============================================================

def u_constant(u0):
    return lambda t: float(u0)

def u_step(u0, t_step):
    return lambda t: float(u0) if t >= t_step else 0.0

def u_sine(amplitude, freq):
    return lambda t: amplitude * np.sin(2 * np.pi * freq * t) + 1.0


# ==============================================================
# FUNGSI SIMULASI
# ==============================================================

def run_simulation(u_func, t_final=200.0, dt=0.1, initial_states=None):
    t = np.arange(0, t_final + dt, dt)
    if initial_states is None:
        initial_states = [50.0, 8.5]  # kondisi awal
    sol = odeint(pond_model, initial_states, t, args=(u_func,))
    B = sol[:, 0]
    DO = sol[:, 1]
    return t, B, DO


# ==============================================================
# PERHITUNGAN MANUAL STEADY STATE
# ==============================================================

u_manual = 1.0
r_manual = r0 * (1 + alpha_feed * u_manual)
B_star_manual = K * (1 - m / r_manual)
DO_star_manual = O2_in  # pendekatan awal

print("=== PERHITUNGAN MANUAL ===")
print(f"Perkiraan steady-state: B* = {B_star_manual:.2f} kg, DO* = {DO_star_manual:.2f} mg/L\n")


# ==============================================================
# SIMULASI STEADY STATE
# ==============================================================

print("Simulasi kondisi steady-state (u = 1.0)...")
u_ss = u_constant(1.0)
t_ss, B_ss, DO_ss = run_simulation(u_ss, t_final=400.0, dt=0.2)
B_star = B_ss[-1]
DO_star = DO_ss[-1]
print(f"Hasil steady-state simulasi: B* = {B_star:.2f} kg, DO* = {DO_star:.3f} mg/L\n")


# ==============================================================
# SIMULASI STEP INPUT
# ==============================================================

print("Simulasi step input (kenaikan pakan pada t = 50 hari)...")
u_step50 = u_step(2.0, t_step=50.0)
t_step, B_step, DO_step = run_simulation(u_step50, t_final=200.0, dt=0.1, initial_states=[B_star, DO_star])
u_values = np.array([u_step50(tt) for tt in t_step])

fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
axs[0].plot(t_step, B_step, 'b', label='Biomassa (kg)')
axs[0].plot(t_step, DO_step, 'g', label='DO (mg/L)')
axs[0].axhline(y=DO_star, color='gray', linestyle=':', label='Steady-state DO')
axs[0].set_ylabel('Nilai')
axs[0].legend()
axs[0].grid(True, linestyle='--', alpha=0.5)
axs[0].set_title('Respon Step Input pada Balong Ikan')

axs[1].plot(t_step, u_values, 'r--', label='Input Pakan (%)')
axs[1].set_xlabel('Waktu (hari)')
axs[1].set_ylabel('Kontrol u(t)')
axs[1].legend()
axs[1].grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('step_simulation_balongan.png')
plt.show()

# Simpan hasil
df_step = pd.DataFrame({'time': t_step, 'Biomass(kg)': B_step, 'DO(mg/L)': DO_step})
df_step.to_csv('pond_step_response.csv', index=False)
print("Data hasil simulasi step disimpan ke pond_step_response.csv\n")


# ==============================================================
# SIMULASI SINUSOIDAL INPUT
# ==============================================================

print("Simulasi input sinusoida (u(t) = 1 + 0.5sin(2πft))...")
u_sin = u_sine(amplitude=0.5, freq=0.01)
t_sin, B_sin, DO_sin = run_simulation(u_sin, t_final=200.0, dt=0.1, initial_states=[B_star, DO_star])

plt.figure(figsize=(8, 5))
plt.plot(t_sin, B_sin, 'b', label='Biomassa (kg)')
plt.plot(t_sin, DO_sin, 'g', label='DO (mg/L)')
plt.title('Simulasi Input Sinusoida pada Balong Ikan')
plt.xlabel('Waktu (hari)')
plt.ylabel('Nilai')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig('sine_simulation_balongan.png')
plt.show()


# ==============================================================
# ANALISIS OTOMATIS HASIL SIMULASI
# ==============================================================

print("\n=== ANALISIS HASIL SIMULASI ===")
if B_step[-1] > B_star:
    print("Biomassa ikan meningkat signifikan setelah penambahan pakan.")
else:
    print("Tidak terjadi peningkatan biomassa signifikan.")

if DO_step[-1] < DO_star:
    print("Kadar oksigen menurun akibat peningkatan aktivitas respirasi.")
else:
    print("Kadar oksigen relatif stabil setelah peningkatan pakan.")

print("\nSimulasi selesai dengan sukses ")
