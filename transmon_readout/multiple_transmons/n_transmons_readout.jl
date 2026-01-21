using ITensors, ITensorMPS, LinearAlgebra, SparseArrays, KrylovKit, CairoMakie


N_trunc = 3

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
    H_reduced = U' * Matrix(H_full) * U  
    charge_reduced = U' * Matrix(charge) * U
    return H_reduced, charge_reduced

end


# ----- Parameters ----
ECT = 0.3
ECR = 0.2
EJ = 20.0
EL = 15.0
ECoup = 0.02
transmon_trunc=31

# ----- Deifining the custom operators -----
HT_reduced, charge_reduced = transmon_hamiltonian(ECT, EJ, N_trunc, transmon_trunc)



function states_dmrg_reduced(ECT, EJ, ECoup, ECR, EL; N_trunc=3, nb_states = 6, resonator_trunc=40) #Transmon_trunc must be odd

    N_transmons = length(ECT)
    @assert N_transmons == length(EJ) == length(ECoup) "Length of ECT, EJ and ECoup must be equal to the number of transmons"

    # === Initialize the sites and the OpSum ===
    sites = Array{Index}(undef, N_transmons + 1)
    for i in 2:N_transmons+1
        T_i = siteind("Boson", i, dim = typeof(N_trunc)==Int ? N_trunc : N_trunc[i-1])
        sites[i] = T_i
    end
    R = siteind("Boson", 1, dim = resonator_trunc)
    sites[1] = R
    os = OpSum()
    
    # ===== Resonator Hamiltonian =====
    omega_q = sqrt(8*ECR*EL)
    os += omega_q, "N", 1
    os += 0.5, "I", 1

    # ===== Transmon & Coupling Hamiltonians =====
    HT_i, charge_i = transmon_hamiltonian(ECT[1], EJ[1], typeof(N_trunc)==Int ? N_trunc : N_trunc[1], transmon_trunc)
    os += 1.0, HT_i, 2
    phi_zpf_r = ((2 * ECR) / EL)^(1/4)
    os += -4 * ECoup[1] / (2*1im*phi_zpf_r), "A - Adag", 1, charge_i, 2

    for i in 2:N_transmons
        HT_ip1, charge_ip1 = transmon_hamiltonian(ECT[i], EJ[i], typeof(N_trunc)==Int ? N_trunc : N_trunc[i], transmon_trunc)
        os += 1.0, HT_ip1, i+1
        os += -4 * ECoup[i], charge_i, i, charge_ip1, i+1
        charge_i = charge_ip1
        HT_i = HT_ip1
    end
    

# ---- Computing the states ----
    H =  MPO(os, sites)

    # ==== DMRG Parameters ====
    nsweeps = 50
    maxdim = [10,10,10,20,20,40,80,100,200,200]
    cutoff = [1E-8]
    noise = [1E-7]
    weight = 40

    # ==== DMRG Computations ====
    psi0_init = random_mps(sites;linkdims=10)
    E0,psi0 = dmrg(H,psi0_init;nsweeps,maxdim,cutoff,outputlevel = 0)
    Psi = [psi0]
    Energies = [E0]
    for i in 1:(nb_states-1)
        psi_init = random_mps(sites;linkdims=10)
        energy,psi = dmrg(H,Psi, psi_init;nsweeps,maxdim,cutoff,noise,weight,outputlevel = 0)
        push!(Psi, psi)
        push!(Energies, energy)
    end 

    return Energies, Psi, H


end






# ---- Energy levels of the Transmon Chain (varying EL) ----

# --- Fixed Parameters ---
ECT = [0.3, 0.2, 0.15, 0.1]
EJ = [50.0, 40.0, 20.0, 20.0]
ECR = 0.5
ECoup = [1E-4, 1E-4, 1E-5, 3E-4]
nb_states = 6
resonator_trunc = 30
N_trunc = [3, 3, 3, 5]

# === Sweep over EL (which changes omega_r only) ===
Omega_r = range(1.0, 20.0, length=60)

dmrg_data = [Float64[] for _ in 1:nb_states]

# ==== Computing the energies ====
for omega_r in Omega_r
    el = omega_r^2 / (8 * ECR)

    # DMRG Calculation
    Energies_dmrg, _, _ = states_dmrg_reduced(ECT, EJ, ECoup, ECR, el; N_trunc=N_trunc, nb_states=nb_states, resonator_trunc=resonator_trunc)
    Energies_dmrg .-= Energies_dmrg[1]

    for i in 1:nb_states
        push!(dmrg_data[i], Energies_dmrg[i])
    end

    print(".")
end

# ---- Plotting with CairoMakie ------
fig = Figure(size = (800, 600), font = "DejaVu Sans")
ax = Axis(fig[1, 1], 
    xlabel = L"\omega_r \ [GHz]", 
    ylabel = L"\chi \ [GHz]",
    title = "Transmon-Resonator Dispersive Shift (varying EL only)")

colors = Makie.wong_colors()

for i in 1:nb_states
    lines!(ax, Omega_r, dmrg_data[i], 
        linestyle = :solid, 
        color = (colors[i], 0.5), 
        linewidth = 2,
        label = i == 1 ? "DMRG" : nothing)
end

vlines!(ax, [sqrt(8 * ECT[i] * EJ[i]) - ECT[i] for i in eachindex(ECT)], linestyle = :dash, color = :black, linewidth = 1, label = "Qubit Frequencies")

axislegend(ax, position = :lt)
save("./transmon_readout/multiple_transmons/Energies_vs_EL_multiple_transmons.png", fig)