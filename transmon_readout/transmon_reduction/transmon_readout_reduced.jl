using ITensors, ITensorMPS, LinearAlgebra, SparseArrays, KrylovKit, CairoMakie



function transmon_hamiltonian(ECT,EJ,N_trunc = 3, transmon_trunc=41)
    # ===== Define operators in the full basis =====    
    charge = spdiagm(0 => ComplexF64[i - transmon_trunc÷2 - 1 for i in 1:transmon_trunc])
    cos_phi = spdiagm(1 => 0.5 * ones(ComplexF64, transmon_trunc-1), -1 => 0.5 * ones(ComplexF64, transmon_trunc-1))

    # ==== Full Hamiltonian ====
    H_full = 4 * ECT * charge*charge - EJ * cos_phi

    # ===== Diagonalize the full Hamiltonian =====
    _, vecs, _ = eigsolve(H_full, N_trunc, :SR) 

    # ===== Keep the first N_trunc levels =====     
    U = hcat(vecs[1:N_trunc]...)
    H_reduced = U' * Matrix(H_full) * U  #XXX : Probably not very fast to convert back and forth to Matrix
    charge_reduced = U' * Matrix(charge) * U
    return H_reduced, charge_reduced

end


# ----- Parameters ----
ECT = 0.3
ECR = 0.2
EJ = 20.0
EL = 15.0
ECoup = 0.02
N_trunc = 3

# ----- Deifining the custom operators -----
transmon_trunc=41

HT_reduced, charge_reduced = transmon_hamiltonian(ECT, EJ)

ITensors.space(::SiteType"CustomTransmon") = N_trunc
ITensors.op(::OpName"charge", ::SiteType"CustomTransmon") =
    charge_reduced
ITensors.op(::OpName"H", ::SiteType"CustomTransmon") =
    HT_reduced



function states_dmrg(ECR, ECoup, EL; nb_states = 4, resonator_trunc=40) #Transmon_trunc must be odd
    
    # === Initialize the sites and the OpSum ===
    T = siteind("CustomTransmon", 1)
    R = siteind("Boson", 2, dim = resonator_trunc)
    sites = [T, R]
    os = OpSum()

    # ===== Transmon Hamiltonian =====
    os += 1.0, "H", 1

    # ===== Resonator Hamiltonian =====
    omega_q = sqrt(8*ECR*EL)
    os += omega_q, "N", 2
    os += 0.5, "I", 2

    # ===== Coupling Hamiltonian =====
    phi_zpf_r = ((2 * ECR) / EL)^(1/4)
    os += -4 * ECoup / (2*1im*phi_zpf_r), "charge", 1, "A - Adag", 2
    

# ---- Computing the states ----
    H =  MPO(os, sites)

    # ==== DMRG Parameters ====
    nsweeps = 70
    maxdim = [10,10,10,20,20,40,80,100,200,200]
    cutoff = [1E-8]
    noise = [1E-7]
    weight = 40

    # ==== DMRG Computations ====
    psi0_init = random_mps(sites;linkdims=20)
    E0,psi0 = dmrg(H,psi0_init;nsweeps,maxdim,cutoff,outputlevel = 0)
    Psi = [psi0]
    Energies = [E0]
    for i in 1:(nb_states-1)
        psi_init = random_mps(sites;linkdims=20)
        energy,psi = dmrg(H,Psi, psi_init;nsweeps,maxdim,cutoff,noise,weight,outputlevel = 0)
        push!(Psi, psi)
        push!(Energies, energy)
    end 

    return Energies


end



# --- Parameters ---
nb_states = 6

# --- Sweep over EL to vary omega_r ---
EL_vals = range(0.1, 5, length=20)
Omega_r = sqrt.(8 * ECR .* EL_vals)

krylov_data = [Float64[] for _ in 1:nb_states]
dmrg_data = [Float64[] for _ in 1:nb_states]


# ---- Computiing the energies ----
for el in EL_vals
    # Krylov Calculation
    H_k = hamiltonian_tr(ECT, ECR, ECoup, EJ, el)
    vals_k, _, _ = eigsolve(H_k, nb_states, :SR)
    vals_k .-= vals_k[1] 
    
    # DMRG Calculation
    vals_d = states_dmrg(ECR, ECoup, el; nb_states=nb_states)
    vals_d .-= vals_d[1] 

    for i in 1:nb_states
        push!(krylov_data[i], vals_k[i])
        push!(dmrg_data[i], vals_d[i])
    end
    print(".")
end



# ---- Plotting with CairoMakie ------
fig = Figure(resolution = (800, 600), font = "DejaVu Sans")
ax = Axis(fig[1, 1], 
    xlabel = L"\omega_r(E_L) = \sqrt{8 E_{C_R} E_L} [GHz]", 
    ylabel = L"E_n - E_0 [GHz]",
    title = "Transmon-Resonator Energy Levels")

colors = Makie.wong_colors()

for i in 1:nb_states
    lines!(ax, Omega_r, krylov_data[i], 
        linestyle = :dash, 
        color = (colors[i], 0.5), 
        linewidth = 2,
        label = i == 1 ? "Krylov" : nothing)
    
    lines!(ax, Omega_r, dmrg_data[i], 
        linestyle = :solid, 
        color = (colors[i], 0.5), 
        linewidth = 2,
        label = i == 1 ? "DMRG" : nothing)
end

axislegend(ax, position = :lt)
save("./transmon_readout/transmon_reduction/energies_vs_omega_r.png", fig)