using ITensors, ITensorMPS, LinearAlgebra, SparseArrays, KrylovKit, CairoMakie



# ------ using the reduced hamiltonian -----
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


# ----- Transmon Parameters ----
N_trunc = 3
ECT = 0.1
EJ = 50.0
transmon_trunc=31
omega_q = sqrt(8 * ECT * EJ) - ECT 


# ----- Deifining the custom operators -----
HT_reduced, charge_reduced = transmon_hamiltonian(ECT, EJ, N_trunc, transmon_trunc)



function find_chi_reduced(ECR, ECoup, EL; nb_states = 6, resonator_trunc=40) 
    
    # === Initialize the sites and the OpSum ===
    T = siteind("Boson", 1, dim = N_trunc)
    R = siteind("Boson", 2, dim = resonator_trunc)
    sites = [T, R]
    os = OpSum()

    # ===== Transmon Hamiltonian =====
    os += 1.0, HT_reduced, 1

    # ===== Resonator Hamiltonian ===== 
    omega_r = sqrt(8*ECR*EL)
    os += omega_r, "N", 2
    os += omega_r * 0.5, "I", 2

    # ===== Coupling Hamiltonian =====
    phi_zpf_r = ((2 * ECR) / EL)^(1/4)
    os += -4 * ECoup / (2*1im*phi_zpf_r), charge_reduced, 1, "A - Adag", 2
    

# ---- Computing the states ----
    H =  MPO(os, sites)

    # ==== DMRG Parameters ====
    nsweeps = 55
    maxdim = [10,10,10,20,20,40]
    cutoff = [0.0]
    noise = [1E-7]
    weight = 90

    # ==== DMRG Computations ====
    psi0_init = MPS([state(T,"0"), state(R,"0")])
    E0,psi0 = dmrg(H,psi0_init;nsweeps,maxdim,cutoff,outputlevel = 0, eigsolve_krylovdim = 15)
    Psi = [psi0]
    Energies = [E0]
    for i in 1:(nb_states-1)
        psi_init = random_mps(sites;linkdims=5)
        _,psi = dmrg(H,Psi, psi_init;nsweeps,maxdim,cutoff,noise,weight,outputlevel = 0, eigsolve_krylovdim = 15)
        push!(Psi, psi)
        push!(Energies, real(inner(psi',H,psi)))
    end 

# ----- Labeling the states -----
    function find_state_match_index(Product_state_ind, True_evecs)

        # ===== Product state =====
        product_state_mps = MPS([state(T,Product_state_ind[1]), state(R, Product_state_ind[2])])


        # ===== Finding the best match =====
        best_ind = 1
        best_val = 0

        for (i, psi) in enumerate(True_evecs)
            overlap_mps = abs(inner(product_state_mps, psi))

            if best_val < overlap_mps
                best_ind = i
                best_val = overlap_mps
            end
        end

        return best_ind

    end

# ----- Finding chi ------
    omegaq = (
        Energies[find_state_match_index(["1", "0"], Psi)] 
        - Energies[find_state_match_index(["0", "0"], Psi)]
    )
    omegaq_plus_2chi = (
        Energies[find_state_match_index(["1", "1"], Psi)] 
        - Energies[find_state_match_index(["0", "1"], Psi)]
    )
    println("omegaq = $omegaq, 2*chi = $(omegaq_plus_2chi - omegaq)")
    return 0.5 * (omegaq_plus_2chi - omegaq)

end



function chi_analytical(ECT, EJ, ECR, EL, ECoup, delta)
    # Coupling strength
    phi_zpf_t = ((2 * ECT) / EJ)^(0.25)
    phi_zpf_r = ((2 * ECR) / EL)^(0.25)
    g = ECoup / (phi_zpf_t * phi_zpf_r)
    # Anharmonicity
    alpha = -ECT
    # Dispersive shift
    chi = g^2 * alpha / (delta * (delta + alpha))
    return chi
end





# ---- Chi as function of delta (varying only EL) ----

# --- Fixed Parameters ---
ECR = 0.5
ECoup = 1E-4
nb_states = 7
resonator_trunc = 30


# === Sweep over EL (which changes omega_r and thus delta) ===
Delta_range = range(-0.1, 0.2, length=80)

dmrg_data = Float64[]
analytical_data = Float64[]

# ==== Computing the energies ====
for delta in Delta_range

    omega_r = omega_q - delta
    local el = (omega_r^2) / (8 * ECR)
    # DMRG Calculation
    chi_d = find_chi_reduced(ECR, ECoup, el; nb_states=nb_states)
    chi_a = chi_analytical(ECT, EJ, ECR, el, ECoup, delta)

    push!(dmrg_data, chi_d)
    push!(analytical_data, chi_a)

    print(".")
end

# ---- Plotting with CairoMakie ------
fig = Figure(size = (800, 600), font = "DejaVu Sans")
ax = Axis(fig[1, 1], 
    xlabel = L"\delta=\omega_p - \omega_r [GHz]", 
    ylabel = L"\chi [GHz]",
    title = "Transmon-Resonator Dispersive Shift (varying E_L only)")

colors = Makie.wong_colors()

lines!(ax, Delta_range, dmrg_data, 
    linestyle = :solid, 
    color = (colors[1], 0.5), 
    linewidth = 2,
    label = "DMRG")

lines!(ax, Delta_range, analytical_data, 
    linestyle = :solid, 
    color = (colors[2], 0.5), 
    linewidth = 2,
    label = "Analytical")

vlines!(ax, [0.0, ECT], linestyle = :dash, color = (:black, 0.3))
ylims!(ax, minimum(analytical_data)*2.0, maximum(analytical_data)*2.0)

axislegend(ax, position = :lt)
save("./transmon_readout/reduction/chi_vs_delta.png", fig)
