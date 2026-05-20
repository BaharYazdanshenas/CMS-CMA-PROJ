import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
import matplotlib.cm as cm
import matplotlib.colors as colors


# -----------------------------
# Choose run file
# -----------------------------
run_folder = Path("CHNS_Drop_Schrinkage_N256_dt0.0005_Rin0.35_Rout_1.25_T5_nu0.5")        #Name of the folder in which the .npz simulation data is stored (same as .npz name)
data_file = run_folder / f"{run_folder.name}.npz"

data = np.load(data_file)

outdir = run_folder / "plots_1"
outdir.mkdir(exist_ok=True)


# -----------------------------
# Load data
# -----------------------------
frames_phi = data["frames_phi"]
frames_omega = data["frames_omega"]

x = data["x"]
y = data["y"]
kx = data["kx"]
ky = data["ky"]
k2 = data["k2"]

Lx = float(data["Lx"])
Ly = float(data["Ly"])
dt = float(data["dt"])

save_every = int(data["save_every"])
nframes = len(frames_phi)
frame_times = np.arange(nframes) * save_every * dt

time_cmap = cm.plasma
time_norm = colors.Normalize(vmin=frame_times[0], vmax=frame_times[-1])

time_hist = data["time_hist"]
mass_hist = data["mass_hist"]
energy_hist = data["energy_hist"]
kinetic_energy_hist = data["kinetic_energy_hist"]

neck_time_hist = data["neck_time_hist"]
inner_neck_width_hist = data["inner_neck_width_hist"]
outer_neck_width_hist = data["outer_neck_width_hist"]

x_cm_hist = data["x_cm_hist"]
y_cm_hist = data["y_cm_hist"]


LABEL_SIZE = 18
TICK_SIZE = 14
LEGEND_SIZE = 14

# -----------------------------
# Helper: spectra and velocity
# -----------------------------
def spectrum_grid(shape, Lx, Ly):
    Ny, Nx = shape
    dx = Lx / Nx
    dy = Ly / Ny

    kx = 2 * np.pi * np.fft.fftfreq(Nx, dx)
    ky = 2 * np.pi * np.fft.fftfreq(Ny, dy)

    Kx, Ky = np.meshgrid(kx, ky, indexing="xy")
    k_abs = np.sqrt(Kx**2 + Ky**2)

    return Kx, Ky, k_abs



def velocity_from_omega(omega, Lx, Ly):
    Kx, Ky, k2 = None, None, None

    Ny, Nx = omega.shape
    dx = Lx / Nx
    dy = Ly / Ny

    kx = 2 * np.pi * np.fft.fftfreq(Nx, dx)
    ky = 2 * np.pi * np.fft.fftfreq(Ny, dy)

    Kx, Ky = np.meshgrid(kx, ky, indexing="xy")
    k2 = Kx**2 + Ky**2

    omega_hat = np.fft.fft2(omega)

    psi_hat = np.zeros_like(omega_hat)
    nonzero = k2 != 0
    psi_hat[nonzero] = omega_hat[nonzero] / k2[nonzero]

    Ux_hat = 1j * Ky * psi_hat
    Uy_hat = -1j * Kx * psi_hat

    Ux = np.fft.ifft2(Ux_hat).real
    Uy = np.fft.ifft2(Uy_hat).real

    return Ux, Uy



def phase_spectrum_2d(phi):
    phi_hat = np.fft.fft2(phi)
    spec = np.abs(phi_hat)**2 / phi.size**2
    spec = np.fft.fftshift(spec)
    return np.log10(spec + 1e-30)


def kinetic_spectrum_2d(omega):
    Ux, Uy = velocity_from_omega(omega, Lx, Ly)

    Ux_hat = np.fft.fft2(Ux)
    Uy_hat = np.fft.fft2(Uy)

    spec = 0.5 * (np.abs(Ux_hat)**2 + np.abs(Uy_hat)**2) / omega.size**2
    spec = np.fft.fftshift(spec)

    return np.log10(spec + 1e-30)


# -----------------------------
# GIF: phase evolution
# -----------------------------
fig, ax = plt.subplots()
im = ax.imshow(
    frames_phi[0],
    origin="lower",
    cmap="RdBu_r",
    vmin=-1,
    vmax=1,
    extent=(0, Lx, 0, Ly),
)
plt.colorbar(im, ax=ax)

def update_phi(frame):
    ax.clear()
    im = ax.imshow(
        frames_phi[frame],
        origin="lower",
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        extent=(0, Lx, 0, Ly),
    )
    ax.contour(
        np.linspace(0, Lx, frames_phi[frame].shape[1]),
        np.linspace(0, Ly, frames_phi[frame].shape[0]),
        frames_phi[frame],
        levels=[0],
        colors="black",
        linewidths=1.2,
    )
    ax.set_title(f"Phase evolution, t={frame_times[frame]:.3g}")
    ax.tick_params(axis="both", labelsize=TICK_SIZE)
    return [im]

ani = animation.FuncAnimation(fig, update_phi, frames=len(frames_phi), interval=100)
ani.save(outdir / "phase_evolution.gif", writer="pillow", fps=10)
plt.close(fig)


# -----------------------------
# GIF: vorticity evolution
# -----------------------------
omega_max = np.nanmax(np.abs(frames_omega))

fig, ax = plt.subplots()
im = ax.imshow(
    frames_omega[0],
    origin="lower",
    cmap="PuOr",
    vmin=-omega_max,
    vmax=omega_max,
    extent=(0, Lx, 0, Ly),
)
plt.colorbar(im, ax=ax)

def update_omega(frame):
    im.set_data(frames_omega[frame])
    ax.set_title(f"Vorticity evolution, t={frame_times[frame]:.3g}")
    ax.tick_params(axis="both", labelsize=TICK_SIZE)
    return [im]

ani = animation.FuncAnimation(fig, update_omega, frames=len(frames_omega), interval=100)
ani.save(outdir / "vorticity_evolution.gif", writer="pillow", fps=10)
plt.close(fig)


# -----------------------------
# Scalar history Plots (Mass, Free Energy, Kinetic Energy, Outer Neck Width, Inner Neck Width)
# -----------------------------
def save_line_plot(xdata, ydata, xlabel, ylabel, title, filename):
    xdata = np.asarray(xdata)
    ydata = np.asarray(ydata)

    if len(xdata) != len(ydata):
        print(f"Skipping {filename}: len(x)={len(xdata)}, len(y)={len(ydata)}")
        return

    fig, ax = plt.subplots()
    ax.plot(xdata, ydata)
    ax.set_xlabel(xlabel, fontsize=LABEL_SIZE )
    ax.set_ylabel(ylabel, fontsize = LABEL_SIZE)
    ax.set_title(title)
    ax.tick_params(axis="both", labelsize=TICK_SIZE)
    ax.grid(True)
    fig.savefig(outdir / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


save_line_plot(time_hist, mass_hist, "Time", "Mean(phi)", "Mass Conservation", "mass_vs_time.png")
save_line_plot(time_hist, energy_hist, "Time", "Free energy", "Free Energy", "free_energy_vs_time.png")
save_line_plot(time_hist, kinetic_energy_hist, "time", "Kinetic Energy", "Kinetic Energy", "kinetic_energy_vs_time.png")

save_line_plot(neck_time_hist, outer_neck_width_hist, "Time", "Outer Neck Width", "Outer Neck Width", "outer_neck_width_vs_time.png")
save_line_plot(neck_time_hist, inner_neck_width_hist, "Time", "Inner Neck Width", "Inner Neck Width", "inner_neck_width_vs_time.png")

