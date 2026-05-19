import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
from pathlib import Path
import shutil
import sys



def run_variable_viscosity(nu_plus, nu_minus, T):     #Function for Simulating the Droplet Coalescence with Viscosity Contrast Across Phases - Define nu+ and nu- Viscoty and total simulation runtime

    start = time.time()
    run_name = f"N{N}_T{T}_dt{dt}_nuplus{nu_plus}_numinus{nu_minus}_M{M/Eps**2}eps2_eps{int(Eps/dx)}dx_Rout{R_out}_Rin{R_in}"
    outdir = Path(run_name)
    outdir.mkdir(parents=True, exist_ok=True)

    # Initial Parameters 
    N = 512
    Lx = Ly = 2* np.pi
    dx = Lx/N
    dy = Ly/N
    i = np.arange(N)
    j = np.arange(N)
    x = dx * i
    y = dy * j
    X, Y = np.meshgrid(x, y, indexing="xy")
    dt = 0.001
    nstep = int(T/dt)
    alpha = 0.0
    sigma = 1.0
    Eps = 3 * dx   # ca. 3 x dx
    M = Eps**2 / 2

    nu_ref = nu_minus


    #Building k values 
    freq_x = np.fft.fftfreq(N, dx)
    freq_y = np.fft.fftfreq(N, dy)
    kx = 2 * np.pi * freq_x
    ky = 2 * np.pi * freq_y
    Kx, Ky = np.meshgrid(kx, ky, indexing="xy")
    k2 = Kx**2 + Ky**2

    #Circular Cutoff Mask for Dealiasing    
    kc = (N/4) * (2 * np.pi/Lx)            
    mask = k2 > kc**2


    #phi_0 Definition for Two Droplets at 2 x Eps Distance 
    R_out = 0.8       #Radius of the Outer Droplets
    R_in = 0.0        #Radius of the Inner Droplets
    x1, y1 = Lx/2 - (R_out + Eps), Ly / 2
    x2, y2 = Lx/2 + (R_out + Eps), Ly / 2
    r1 = np.sqrt((X-x1)**2 + (Y - y1)**2)
    r2 = np.sqrt((X-x2)**2 + (Y - y2)**2)

    phi1 =  np.tanh((R_out - r1) / Eps) * np.tanh((r1 - R_in) / Eps)
    phi2 = np.tanh((R_out - r2) / Eps) * np.tanh((r2 - R_in) / Eps)
    phi_0 = np.maximum(phi1, phi2)
    phi_0_hat = np.fft.fft2(phi_0)


    #Initial Vortecity
    omega_0 = np.zeros((N,N))
    omega_hat = np.fft.fft2(omega_0)



    def velocity_U(omega_hat):                   #Computing Velocity U from Vorticity
        psi_hat = np.zeros_like(omega_hat)
        mask_zero = k2 == 0  
        psi_hat[~mask_zero] = omega_hat[~mask_zero] / k2[~mask_zero]

        #Recovering u from psi_hat
        Ux_hat = 1j * Ky * psi_hat
        Uy_hat = - 1j * Kx * psi_hat

        Ux_hat[mask] = 0
        Uy_hat[mask] = 0    

        Ux = np.fft.ifft2(Ux_hat).real
        Uy = np.fft.ifft2(Uy_hat).real

        return Ux, Uy


    def NL_RHS_CH(phi_hat, omega_hat):   #Nonlinear RHC Contribution from Cahn-Hilliard Equation 
        #Building Phi^3 in Real Space 
        phi = np.fft.ifft2(phi_hat).real
        phi3 = phi**3
        phi3_hat = np.fft.fft2(phi3)
        phi3_hat[mask] = 0       #dealiasing

        #Advection term
        phi_x_hat = 1j * Kx * phi_hat
        phi_y_hat = 1j * Ky * phi_hat
        phi_x = np.fft.ifft2(phi_x_hat).real
        phi_y = np.fft.ifft2(phi_y_hat).real
        Ux, Uy = velocity_U(omega_hat)
        Advec = Ux * phi_x + Uy * phi_y
        Advec_hat = np.fft.fft2(Advec)
        Advec_hat[mask] = 0        #dealiasing

        
        #Building Nonlinear Part
        NL_hat = - Advec_hat - M * k2 * 3/4 * sigma/Eps * phi3_hat   
        NL_hat[mask] = 0

        return NL_hat


    def variable_viscous_curl_hat(phi, omega_hat):
        Ux, Uy = velocity_U(omega_hat)

        Ux_hat = np.fft.fft2(Ux)
        Uy_hat = np.fft.fft2(Uy)

        Ux_x = np.fft.ifft2(1j * Kx * Ux_hat).real
        Ux_y = np.fft.ifft2(1j * Ky * Ux_hat).real
        Uy_x = np.fft.ifft2(1j * Kx * Uy_hat).real
        Uy_y = np.fft.ifft2(1j * Ky * Uy_hat).real

        nu_phi = nu_plus * 0.5 * (1.0 + phi) + nu_minus * 0.5 * (1.0 - phi)
        dnu = nu_phi - nu_ref

        tau_xx = dnu * (2.0 * Ux_x)
        tau_xy = dnu * (Ux_y + Uy_x)
        tau_yy = dnu * (2.0 * Uy_y)

        tau_xx_hat = np.fft.fft2(tau_xx)
        tau_xy_hat = np.fft.fft2(tau_xy)
        tau_yy_hat = np.fft.fft2(tau_yy)

        tau_xx_hat[mask] = 0
        tau_xy_hat[mask] = 0
        tau_yy_hat[mask] = 0

        Fx_hat = 1j * Kx * tau_xx_hat + 1j * Ky * tau_xy_hat
        Fy_hat = 1j * Kx * tau_xy_hat + 1j * Ky * tau_yy_hat

        visc_curl_hat = 1j * Kx * Fy_hat - 1j * Ky * Fx_hat
        visc_curl_hat[mask] = 0

        return visc_curl_hat


    def NL_RHS_NS(phi_hat, omega_hat):    #Nonlinear RC Contribution from Navier-Stokes Equation
        #Building Phi^3 in Real Space
        phi = np.fft.ifft2(phi_hat).real
        phi3 = phi**3
        phi3_hat = np.fft.fft2(phi3) 
        phi3_hat[mask] = 0              #dealiasing

        phi_x_hat = 1j * Kx * phi_hat
        phi_x = np.fft.ifft2(phi_x_hat).real
        phi_y_hat = 1j * Ky * phi_hat
        phi_y = np.fft.ifft2(phi_y_hat).real

        mu_hat = 3/2 * sigma * Eps * k2 * phi_hat + 3/4 * sigma/Eps * (phi3_hat - phi_hat)
        mu_hat[mask] = 0
        mu = np.fft.ifft2(mu_hat).real

        omega_x_hat = 1j * Kx * omega_hat
        omega_y_hat = 1j * Ky * omega_hat
        omega_x = np.fft.ifft2(omega_x_hat).real
        omega_y = np.fft.ifft2(omega_y_hat).real

        Ux, Uy = velocity_U(omega_hat)

        ST1 = mu * phi_x
        ST2 = mu * phi_y
        ST1_hat = np.fft.fft2(ST1)
        ST1_hat[mask] = 0
        ST2_hat = np.fft.fft2(ST2)
        ST2_hat[mask] = 0

        Advec = Ux * omega_x + Uy * omega_y
        Advec_hat = np.fft.fft2(Advec)
        Advec_hat[mask] = 0

        visc_corr_hat = variable_viscous_curl_hat(phi, omega_hat)

        nl_rhs_NS_hat = (
            - Advec_hat
            + 1j * (Kx * ST2_hat - Ky * ST1_hat)
            + visc_corr_hat
        )
        #dealiasing
        nl_rhs_NS_hat[mask] = 0

        return nl_rhs_NS_hat

    #Function for Neck Points Detection
    def zero_crossings_y(phi, x, y, x_probe):
        ix = np.argmin(np.abs(x - x_probe))
        profile = phi[:, ix]   # phi[y_index, x_index]

        crossings = []

        for j in range(len(y) - 1):
            f1 = profile[j]
            f2 = profile[j + 1]

            if f1 == 0:
                crossings.append(y[j])

            elif f1 * f2 < 0:
                # linear interpolation for phi=0
                y0 = y[j] - f1 * (y[j+1] - y[j]) / (f2 - f1)
                crossings.append(y0)

        return np.array(crossings), profile, ix



    #Setting the Stage for Semi-Implicit ETDRK2 Time-Stepping Scheme

    #CH Time-Stepping Coefficients
    L_phi = - 3/2 * M * sigma * Eps * k2**2 + 3/4 * sigma / Eps * M * k2
    E_phi = np.exp(L_phi*dt)
    Phi1_phi = np.zeros_like(k2)
    Phi2_phi = np.zeros_like(k2)
    mask_zero_phi = np.abs(L_phi) < 1e-14
    Phi1_phi[~mask_zero_phi] =  (E_phi[~mask_zero_phi] - 1) / L_phi[~mask_zero_phi]
    Phi1_phi[mask_zero_phi] = dt
    Phi2_phi[~mask_zero_phi] =  (E_phi[~mask_zero_phi] - 1 - L_phi[~mask_zero_phi] * dt) / (L_phi[~mask_zero_phi]**2 * dt)
    Phi2_phi[mask_zero_phi] = dt /2


    #NS Time-Stepping Coefficients
    L_omega = -nu_ref * k2 - alpha
    E_omega = np.exp(L_omega * dt)
    Phi1_omega = np.zeros_like(k2)
    Phi2_omega = np.zeros_like(k2)
    mask_zero_omega = np.abs(L_omega) < 1e-14
    Phi1_omega[~mask_zero_omega] = (E_omega[~mask_zero_omega] - 1) / L_omega[~mask_zero_omega]
    Phi1_omega[mask_zero_omega] = dt
    Phi2_omega[~mask_zero_omega] =  (E_omega[~mask_zero_omega] - 1 - L_omega[~mask_zero_omega] * dt) / (L_omega[~mask_zero_omega]**2 * dt)
    Phi2_omega[mask_zero_omega] = dt /2



    phi_hat = phi_0_hat.copy()

    #Preparing for Saving Data at Each Time-Step
    mass_hist = []
    energy_hist = []

    frames_phi = []
    frames_omega = []
    save_every = max(1, int(nstep/120))

    inner_neck_width_hist = []
    outer_neck_width_hist = []
    neck_time_hist = []
    profile_frames = []
    profile_times = []

    kinetic_energy_hist = []
    x_cm_hist = []
    y_cm_hist = []
    time_hist = []


    # ETDRK2 Time-Stepping
    for step in range(nstep):

        a_CH = phi_hat * E_phi + Phi1_phi * NL_RHS_CH(phi_hat, omega_hat)
        a_NS = omega_hat * E_omega + Phi1_omega * NL_RHS_NS(phi_hat, omega_hat)

        phi_hat_new = a_CH + Phi2_phi * (
            NL_RHS_CH(a_CH, a_NS) - NL_RHS_CH(phi_hat, omega_hat)
        )

        omega_hat_new = a_NS + Phi2_omega * (
            NL_RHS_NS(a_CH, a_NS) - NL_RHS_NS(phi_hat, omega_hat)
        )

        phi_hat = phi_hat_new
        omega_hat = omega_hat_new

        phi_x = np.fft.ifft2(1j * Kx * phi_hat).real
        phi_y = np.fft.ifft2(1j * Ky * phi_hat).real
        phi = np.fft.ifft2(phi_hat).real

        Ux, Uy = velocity_U(omega_hat)

        # if your viscosity is computed from phi, define it here for diagnostics
        h = 0.5 * (phi + 1.0)
        h = np.clip(h, 0.0, 1.0)
        nu = nu_minus + h * (nu_plus - nu_minus)

        # Middle Step Check Diagnostics
        if step % 100 == 0:
            print(
                step,
                "time =", step * dt,
                "phi min/max =", np.nanmin(phi), np.nanmax(phi),
                "u max =", np.nanmax(np.sqrt(Ux**2 + Uy**2)),
                "nu min/max =", np.nanmin(nu), np.nanmax(nu),
            )

        # Hard Failure Checks for Phi3 Overflow
        if not np.all(np.isfinite(phi)):
            raise FloatingPointError(f"phi became non-finite at step {step}")

        if np.max(np.abs(phi)) > 10:
            raise FloatingPointError(
                f"phi is blowing up at step {step}: "
                f"min={phi.min()}, max={phi.max()}"
            )

         #Computing and Storing the Simulation Data
        K_current = 0.5 * dx * dy * np.sum(Ux**2 + Uy**2)    #Kinetic Energy
        kinetic_energy_hist.append(K_current)

        F_current = dx * dy * np.sum(3/4 * sigma * ( Eps * (phi_x**2 + phi_y**2) + (phi**2 -1)**2 / (4 * sigma)))   #Free Energy
        time_hist.append(step * dt)     #Time
        mass_hist.append(np.mean(phi))     #Mass
        energy_hist.append(F_current)

        if step % save_every == 0:
            omega = np.fft.ifft2(omega_hat).real
            frames_phi.append(phi.copy())           #Phase
            frames_omega.append(omega.copy())       #Vorticity
            crossings, profile, ix = zero_crossings_y(phi, x, y, Lx/2)    #Finding Where Necks are

            crossings = np.sort(crossings)

            if len(crossings) == 2:
                outer_width = crossings[1] - crossings[0]
                inner_width = np.nan

            elif len(crossings) == 4:
                inner_width = crossings[2] - crossings[1]
                outer_width = crossings[3] - crossings[0]
            else:
                inner_width = np.nan
                outer_width = np.nan

            inner_neck_width_hist.append(inner_width)
            outer_neck_width_hist.append(outer_width)

            neck_time_hist.append(step * dt)
            profile_frames.append(profile.copy())
            profile_times.append(step * dt)
            



    end = time.time()
    runtime_seconds = end - start
    runtime_minutes = runtime_seconds / 60

    print("Simulation time:", runtime_minutes, "minutes")

    #Saving the Script

    try:
        script_path = Path(sys.argv[0]).resolve()

        if script_path.exists():
            shutil.copy(script_path, outdir / f"script.py")
        else:
            raise FileNotFoundError

    except Exception:
        print("WARNING: Could not automatically save script. Save manually.")


    phi_final = np.fft.ifft2(phi_hat).real
    omega_final = np.fft.ifft2(omega_hat).real


    #Saving Simulation Data
    np.savez(
        outdir / f"{run_name}.npz",

        runtime_seconds=runtime_seconds,
        runtime_minutes=runtime_minutes,

        N=N, T=T, Lx=Lx, Ly=Ly, dx=dx, dy=dy, dt=dt, nstep=nstep,
        nu_plus=nu_plus,
        nu_minus=nu_minus,
        nu_ref=nu_ref, alpha=alpha, sigma=sigma, Eps=Eps, M=M,
        R_out=R_out, R_in=R_in,
        save_every=save_every,

        x=x, y=y,

        mass_hist=np.array(mass_hist),
        energy_hist=np.array(energy_hist),
        neck_time_hist=np.array(neck_time_hist),
        inner_neck_width_hist=np.array(inner_neck_width_hist),
        outer_neck_width_hist=np.array(outer_neck_width_hist),

        frames_phi=np.array(frames_phi),
        frames_omega=np.array(frames_omega),

        profile_frames=np.array(profile_frames),
        profile_times=np.array(profile_times),

        phi_0=phi_0,
        phi_final=phi_final,
        omega_final=omega_final,

        time_hist=np.array(time_hist),
        kinetic_energy_hist=np.array(kinetic_energy_hist),

        kx=kx,
        ky=ky,
        k2=k2,
    )


def run_constant_viscosity(nu, T):     #Function for Simulating Droplet Coalescence with Constant Viscosity Across the Phases - Define Viscosity and Total Simulation Runtime
    start = time.time()


    # Initial Paparemers
    N = 512
    Lx = Ly = 2* np.pi
    dx = Lx/N
    dy = Ly/N
    i = np.arange(N)
    j = np.arange(N)
    x = dx * i
    y = dy * j
    X, Y = np.meshgrid(x, y, indexing="xy")
    dt = 0.0005
    nstep = int(T/dt)
    alpha = 0.0
    sigma = 1.0
    Eps = 3 * dx   # ca. 3 x dx
    M = Eps**2 / 2


    #Building k values 
    freq_x = np.fft.fftfreq(N, dx)
    freq_y = np.fft.fftfreq(N, dy)
    kx = 2 * np.pi * freq_x
    ky = 2 * np.pi * freq_y
    Kx, Ky = np.meshgrid(kx, ky, indexing="xy")
    k2 = Kx**2 + Ky**2

    #Circular Cutoff Mask for Dealiasing    
    kc = (N/4) * (2 * np.pi/Lx)            
    mask = k2 > kc**2


    #phi_0 Definition
    R_out = 0.8       #Radius of the Outer Droplets
    R_in = 0.0        #Radius of the Inner Droplets
    x1, y1 = Lx/2 - (R_out + Eps), Ly / 2
    x2, y2 = Lx/2 + (R_out + Eps), Ly / 2
    r1 = np.sqrt((X-x1)**2 + (Y - y1)**2)
    r2 = np.sqrt((X-x2)**2 + (Y - y2)**2)

    #Creating the Directory to Save the Simualtion Files
    run_name = f"N{N}_T{T}_dt{dt}_nu{nu}_M{M/Eps**2}eps2_eps{int(Eps/dx)}dx_Rout{R_out}_Rin{R_in}"
    outdir = Path(run_name)
    outdir.mkdir(parents=True, exist_ok=True)

    #Building Initial Phase Condition
    phi1 =  np.tanh((R_out - r1) / Eps) * np.tanh((r1 - R_in) / Eps)
    phi2 = np.tanh((R_out - r2) / Eps) * np.tanh((r2 - R_in) / Eps)
    phi_0 = np.maximum(phi1, phi2)
    phi_0_hat = np.fft.fft2(phi_0)


    #Initial vortecity
    omega_0 = np.zeros((N,N))
    omega_hat = np.fft.fft2(omega_0)



    def velocity_U(omega_hat):        #Computing Velocity U from Vorticity
        psi_hat = np.zeros_like(omega_hat)
        mask_zero = k2 == 0  
        psi_hat[~mask_zero] = omega_hat[~mask_zero] / k2[~mask_zero]

        #Recovering u from psi_hat
        Ux_hat = 1j * Ky * psi_hat
        Uy_hat = - 1j * Kx * psi_hat

        Ux = np.fft.ifft2(Ux_hat).real
        Uy = np.fft.ifft2(Uy_hat).real

        return Ux, Uy


    def NL_RHS_CH(phi_hat, omega_hat):      #Nonlinear RHS Contribution from Cahn-Hilliard Equation
        #Computing Phi^3 in Real Space
        phi = np.fft.ifft2(phi_hat).real
        phi3 = phi**3
        phi3_hat = np.fft.fft2(phi3)
        phi3_hat[mask] = 0       #dealiasing

        #Advection term
        phi_x_hat = 1j * Kx * phi_hat
        phi_y_hat = 1j * Ky * phi_hat
        phi_x = np.fft.ifft2(phi_x_hat).real
        phi_y = np.fft.ifft2(phi_y_hat).real
        Ux, Uy = velocity_U(omega_hat)
        Advec = Ux * phi_x + Uy * phi_y
        Advec_hat = np.fft.fft2(Advec)
        Advec_hat[mask] = 0        #dealiasing

        
        #Nonlinear part
        NL_hat = - Advec_hat - M * k2 * 3/4 * sigma/Eps * phi3_hat   
        NL_hat[mask] = 0

        return NL_hat


    def NL_RHS_NS(phi_hat, omega_hat):      #Nonlinear RHS Contribution from Navier-Stokes Equation
        #Computing Phi^3 in Real Space
        phi = np.fft.ifft2(phi_hat).real
        phi3 = phi**3
        phi3_hat = np.fft.fft2(phi3) 
        phi3_hat[mask] = 0      #dealiasing

        phi_x_hat = 1j * Kx * phi_hat
        phi_x = np.fft.ifft2(phi_x_hat).real
        phi_y_hat = 1j * Ky * phi_hat
        phi_y = np.fft.ifft2(phi_y_hat).real

        mu_hat = 3/2 * sigma * Eps * k2 * phi_hat + 3/4 * sigma/Eps * (phi3_hat - phi_hat)
        mu_hat[mask] = 0
        mu = np.fft.ifft2(mu_hat).real

        omega_x_hat = 1j * Kx * omega_hat
        omega_y_hat = 1j * Ky * omega_hat
        omega_x = np.fft.ifft2(omega_x_hat).real
        omega_y = np.fft.ifft2(omega_y_hat).real

        Ux, Uy = velocity_U(omega_hat)

        ST1 = mu * phi_x
        ST2 = mu * phi_y
        ST1_hat = np.fft.fft2(ST1)
        ST1_hat[mask] = 0
        ST2_hat = np.fft.fft2(ST2)
        ST2_hat[mask] = 0

        Advec = Ux * omega_x + Uy * omega_y
        Advec_hat = np.fft.fft2(Advec)
        Advec_hat[mask] = 0

        nl_rhs_NS_hat = - Advec_hat + 1j * (Kx * ST2_hat - Ky * ST1_hat)

        #dealiasing
        nl_rhs_NS_hat[mask] = 0

        return nl_rhs_NS_hat

    #Function for Neck Points Detection
    def zero_crossings_y(phi, x, y, x_probe):
        ix = np.argmin(np.abs(x - x_probe))
        profile = phi[:, ix]   # phi[y_index, x_index]

        crossings = []

        for j in range(len(y) - 1):
            f1 = profile[j]
            f2 = profile[j + 1]

            if f1 == 0:
                crossings.append(y[j])

            elif f1 * f2 < 0:
                # linear interpolation for phi=0
                y0 = y[j] - f1 * (y[j+1] - y[j]) / (f2 - f1)
                crossings.append(y0)

        return np.array(crossings), profile, ix

    
    #CH Time-Stepping Coefficients
    L_phi = - 3/2 * M * sigma * Eps * k2**2 + 3/4 * sigma / Eps * M * k2
    E_phi = np.exp(L_phi*dt)
    Phi1_phi = np.zeros_like(k2)
    Phi2_phi = np.zeros_like(k2)
    mask_zero_phi = np.abs(L_phi) < 1e-14
    Phi1_phi[~mask_zero_phi] =  (E_phi[~mask_zero_phi] - 1) / L_phi[~mask_zero_phi]
    Phi1_phi[mask_zero_phi] = dt
    Phi2_phi[~mask_zero_phi] =  (E_phi[~mask_zero_phi] - 1 - L_phi[~mask_zero_phi] * dt) / (L_phi[~mask_zero_phi]**2 * dt)
    Phi2_phi[mask_zero_phi] = dt /2


    #NS Time-Stepping Coefficients
    L_omega = -nu * k2 - alpha
    E_omega = np.exp(L_omega * dt)
    Phi1_omega = np.zeros_like(k2)
    Phi2_omega = np.zeros_like(k2)
    mask_zero_omega = np.abs(L_omega) < 1e-14
    Phi1_omega[~mask_zero_omega] = (E_omega[~mask_zero_omega] - 1) / L_omega[~mask_zero_omega]
    Phi1_omega[mask_zero_omega] = dt
    Phi2_omega[~mask_zero_omega] =  (E_omega[~mask_zero_omega] - 1 - L_omega[~mask_zero_omega] * dt) / (L_omega[~mask_zero_omega]**2 * dt)
    Phi2_omega[mask_zero_omega] = dt /2

    phi_hat = phi_0_hat.copy()

    #Preparing for Saving Data at Each Time-Step
    mass_hist = []
    energy_hist = []

    frames_phi = []
    frames_omega = []
    save_every = max(1, int(nstep/120))

    inner_neck_width_hist = []
    outer_neck_width_hist = []
    neck_time_hist = []
    profile_frames = []
    profile_times = []

    kinetic_energy_hist = []
    x_cm_hist = []
    y_cm_hist = []
    time_hist = []

    #ETDRK2 Time-Stepping
    for t in range(nstep):

        a_CH = phi_hat * E_phi + Phi1_phi * NL_RHS_CH(phi_hat, omega_hat)
        a_NS = omega_hat * E_omega + Phi1_omega * NL_RHS_NS(phi_hat, omega_hat)

        #CH Time-Stepping
        phi_hat_new = a_CH + Phi2_phi * (NL_RHS_CH(a_CH, a_NS) - NL_RHS_CH(phi_hat, omega_hat))

        #NS Time-Stepping 
        omega_hat_new = a_NS + Phi2_omega * (NL_RHS_NS(a_CH, a_NS) - NL_RHS_NS(phi_hat, omega_hat))
        
        phi_hat = phi_hat_new
        omega_hat = omega_hat_new

        phi_x = np.fft.ifft2(1j * Kx * phi_hat).real
        phi_y = np.fft.ifft2(1j * Ky * phi_hat).real
        phi = np.fft.ifft2(phi_hat).real

        Ux, Uy = velocity_U(omega_hat)

        K_current = 0.5 * dx * dy * np.sum(Ux**2 + Uy**2)
        kinetic_energy_hist.append(K_current)

        c_indicator = 0.5 * (phi + 1.0)
        mass_c = np.sum(c_indicator) * dx * dy

        if mass_c > 1e-14:
            x_cm = np.sum(X * c_indicator) * dx * dy / mass_c
            y_cm = np.sum(Y * c_indicator) * dx * dy / mass_c
        else:
            x_cm = np.nan
            y_cm = np.nan

        x_cm_hist.append(x_cm)
        y_cm_hist.append(y_cm)
        time_hist.append(t * dt)

        F_current = dx * dy * np.sum(3/4 * sigma * ( Eps * (phi_x**2 + phi_y**2) + (phi**2 -1)**2 / (4 * sigma)))
        mass_hist.append(np.mean(phi))
        energy_hist.append(F_current)

        if t % save_every == 0:
            omega = np.fft.ifft2(omega_hat).real
            frames_phi.append(phi.copy())
            frames_omega.append(omega.copy())
            crossings, profile, ix = zero_crossings_y(phi, x, y, Lx/2)

            crossings = np.sort(crossings)

            if len(crossings) == 2:
                outer_width = crossings[1] - crossings[0]
                inner_width = np.nan

            elif len(crossings) == 4:
                inner_width = crossings[2] - crossings[1]
                outer_width = crossings[3] - crossings[0]
            else:
                inner_width = np.nan
                outer_width = np.nan

            inner_neck_width_hist.append(inner_width)
            outer_neck_width_hist.append(outer_width)

            neck_time_hist.append(t * dt)
            profile_frames.append(profile.copy())
            profile_times.append(t * dt)
            



    end = time.time()
    runtime_seconds = end - start
    runtime_minutes = runtime_seconds / 60

    print("Simulation time:", runtime_minutes, "minutes")

    #Saving the Script

    try:
        script_path = Path(sys.argv[0]).resolve()

        if script_path.exists():
            shutil.copy(script_path, outdir / f"script.py")
        else:
            raise FileNotFoundError

    except Exception:
        print("WARNING: Could not automatically save script. Save manually.")


    phi_final = np.fft.ifft2(phi_hat).real
    omega_final = np.fft.ifft2(omega_hat).real

    np.savez(
        outdir / f"{run_name}.npz",

        runtime_seconds=runtime_seconds,
        runtime_minutes=runtime_minutes,

        N=N, T=T, Lx=Lx, Ly=Ly, dx=dx, dy=dy, dt=dt, nstep=nstep,
        nu=nu, alpha=alpha, sigma=sigma, Eps=Eps, M=M,
        R_out=R_out, R_in=R_in,
        save_every=save_every,

        x=x, y=y,

        mass_hist=np.array(mass_hist),
        energy_hist=np.array(energy_hist),
        neck_time_hist=np.array(neck_time_hist),
        inner_neck_width_hist=np.array(inner_neck_width_hist),
        outer_neck_width_hist=np.array(outer_neck_width_hist),

        frames_phi=np.array(frames_phi),
        frames_omega=np.array(frames_omega),

        profile_frames=np.array(profile_frames),
        profile_times=np.array(profile_times),

        phi_0=phi_0,
        phi_final=phi_final,
        omega_final=omega_final,

        time_hist=np.array(time_hist),
        kinetic_energy_hist=np.array(kinetic_energy_hist),
        x_cm_hist=np.array(x_cm_hist),
        y_cm_hist=np.array(y_cm_hist),

        kx=kx,
        ky=ky,
        k2=k2,
    )


viscosity_cases = [

    (0.5, 1.0, 16),

    (1.0, 0.5, 17),

    (1.0, 1.5, 25),

    (1.5, 1.0, 25),

] #Respectively nu_plus, nu_minus, T   for Viscosity-Contrast Simulation

nu_cases = [(0.5, 13), (1.0, 20), (1.5, 25)]  #Respectively nu, T for Constant Viscosity Simulation


for nu_plus, nu_minus, T in viscosity_cases:    #Run Simulations for Viscosity Contrast
    print(f"\nRunning variable viscosity: nu_plus={nu_plus}, nu_minus={nu_minus}, T={T}")
    try:
        run_variable_viscosity(nu_plus, nu_minus, T)
        print("Finished")
    except Exception as e:
        print("FAILED:", e)


for nu, T in nu_cases:              #Run Simulations for Constant Viscosity
    print(f"\nRunning constant viscosity: nu={nu}, T={T}")
    try:
        run_constant_viscosity(nu, T)
        print("Finished")
    except Exception as e:
        print("FAILED:", e)


