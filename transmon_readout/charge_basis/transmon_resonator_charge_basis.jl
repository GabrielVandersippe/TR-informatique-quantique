using ITensors, ITensorMPS, LinearAlgebra, SparseArrays, KrylovKit, CairoMakie




# --- Hamiltonian function ---
function hamiltonian_tr(ECT, ECR, ECoup, EJ, EL; transmon_trunc=41, resonator_trunc=40)
    

    # ===== Transmon Hamiltonian =====

    charge = sparse(ComplexF64[(i == j) ? (i - transmon_trunc÷2 - 1) : 0.0 for i in 1:transmon_trunc, j in 1:transmon_trunc])
    cos_phi = sparse(ComplexF64[(i == j+1) || (i+1 == j) ? 0.5 : 0.0 for i in 1:transmon_trunc, j in 1:transmon_trunc])

    HT = 4 * ECT * charge*charge - EJ * cos_phi


    # ===== Resonator Hamiltonian =====
    destruction_r = spzeros(Float64, resonator_trunc, resonator_trunc)
    for n in 1:(resonator_trunc - 1)
        destruction_r[n, n + 1] = sqrt(n)
    end
    creation_r = destruction_r'
    HR = sqrt(8 * ECR * EL) * (creation_r * destruction_r + spdiagm(0 => ones(resonator_trunc)) * 0.5)


    # ===== Coupling Hamiltonian =====
    phizpf_r = ((2 * ECR) / EL)^(1/4)
    HC = -4 * ECoup / (2*1im*phizpf_r) * kron(charge, destruction_r - creation_r)


    # ===== Final Hamiltonian =====
    H = kron(HT, spdiagm(0 => ones(resonator_trunc))) +
        kron(spdiagm(0 => ones(transmon_trunc)), HR) +
        HC

    return H
end



# ----- Deifining the custom operators -----
transmon_trunc=41

ITensors.space(::SiteType"CustomTransmon") = transmon_trunc
ITensors.op(::OpName"charge", ::SiteType"CustomTransmon") =
    ComplexF64[(i == j) ? (i - transmon_trunc÷2 - 1) : 0.0 for i in 1:transmon_trunc, j in 1:transmon_trunc]
ITensors.op(::OpName"cosPhi", ::SiteType"CustomTransmon") =
    ComplexF64[(i == j+1) || (i+1 == j) ? 0.5 : 0.0 for i in 1:transmon_trunc, j in 1:transmon_trunc]

# ---- Computing the states ----
function states_dmrg(ECT, ECR, ECoup, EJ, EL; nb_states = 4, resonator_trunc=40) #Transmon_trunc must be odd
    
    # === Initialize the sites and the OpSum ===
    T = siteind("CustomTransmon", 1)
    R = siteind("Boson", 2, dim = resonator_trunc)
    sites = [T, R]
    os = OpSum()

    # ===== Transmon Hamiltonian =====

    os += 4*ECT, "charge * charge", 1
    os += -EJ, "cosPhi", 1

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


# ---- Energies as function of delta ----
# let
#     # --- Parameters ---
#     ECT = 0.1
#     ECR = 0.5
#     ECoup = 0.005
#     nb_states = 3

#     c = 7.5 # GHz, this is omega_p + omega_r / 2

#     # --- Sweep over delta ---
#     Delta = range(-10, 10, length=20)

#     krylov_data = [Float64[] for _ in 1:nb_states]
#     dmrg_data = [Float64[] for _ in 1:nb_states]


#     # ---- Computiing the energies ----
#     for delta in Delta
#         ej = ( (c + delta/2)^2 ) / (8 * ECT)
#         el = ( (c - delta/2)^2 ) / (8 * ECR)
#         # Krylov Calculation
#         H_k = hamiltonian_tr(ECT, ECR, ECoup, ej, el)
#         vals_k, _, _ = eigsolve(H_k, nb_states, :SR)
#         vals_k .-= vals_k[1] 
        
#         # DMRG Calculation
#         vals_d = states_dmrg(ECT, ECR, ECoup, ej, el; nb_states=nb_states)
#         vals_d .-= vals_d[1] 

#         for i in 1:nb_states
#             push!(krylov_data[i], vals_k[i])
#             push!(dmrg_data[i], vals_d[i])
#         end
#         print(".")
#     end



#     # ---- Plotting with CairoMakie ------
#     fig = Figure(size = (800, 600), font = "DejaVu Sans")
#     ax = Axis(fig[1, 1], 
#         xlabel = L"\delta=\omega_p - \omega_r [GHz]", 
#         ylabel = L"E_n - E_0 [GHz]",
#         title = "Transmon-Resonator Energy Levels")

#     colors = Makie.wong_colors()

#     for i in 1:nb_states
#         lines!(ax, Delta, krylov_data[i], 
#             linestyle = :dash, 
#             color = (colors[i], 0.5), 
#             linewidth = 2,
#             label = i == 1 ? "Krylov" : nothing)
        
#         lines!(ax, Delta, dmrg_data[i], 
#             linestyle = :solid, 
#             color = (colors[i], 0.5), 
#             linewidth = 2,
#             label = i == 1 ? "DMRG" : nothing)
#     end

#     hlines!(ax, c, color=:black, linestyle=:dash, linewidth=2, label=L"\frac{\omega_r + \omega_p}{2}")

#     axislegend(ax, position = :lt)
#     save("./transmon_readout/charge_basis/energies_vs_delta.png", fig)
# end



# ---- Energies as function of omega_r ----
# let
# # --- Changing Parameters ---
# EJ=30

# # --- Sweep over EJ to vary omega_p ---
# EL_vals = range(0.1, 5, length=20)
# Omega_r = sqrt.(8 * ECR .* EL_vals)

# krylov_data = [Float64[] for _ in 1:nb_states]
# dmrg_data = [Float64[] for _ in 1:nb_states]


# # ---- Computiing the energies ----
# for el in EL_vals
#     # Krylov Calculation
#     H_k = hamiltonian_tr(ECT, ECR, ECoup, EJ, el)
#     vals_k, _, _ = eigsolve(H_k, nb_states, :SR)
#     vals_k .-= vals_k[1] 
    
#     # DMRG Calculation
#     vals_d = states_dmrg(ECT, ECR, ECoup, EJ, el; nb_states=nb_states)
#     vals_d .-= vals_d[1] 

#     for i in 1:nb_states
#         push!(krylov_data[i], vals_k[i])
#         push!(dmrg_data[i], vals_d[i])
#     end
#     print(".")
# end



# # ---- Plotting with CairoMakie ------
# fig = Figure(resolution = (800, 600), font = "DejaVu Sans")
# ax = Axis(fig[1, 1], 
#     xlabel = L"\omega_r(E_L) = \sqrt{8 E_{C_R} E_L} [GHz]", 
#     ylabel = L"E_n - E_0 [GHz]",
#     title = "Transmon-Resonator Energy Levels")

# colors = Makie.wong_colors()

# for i in 1:nb_states
#     lines!(ax, Omega_r, krylov_data[i], 
#         linestyle = :dash, 
#         color = (colors[i], 0.5), 
#         linewidth = 2,
#         label = i == 1 ? "Krylov" : nothing)
    
#     lines!(ax, Omega_r, dmrg_data[i], 
#         linestyle = :solid, 
#         color = (colors[i], 0.5), 
#         linewidth = 2,
#         label = i == 1 ? "DMRG" : nothing)
# end

# axislegend(ax, position = :lt)
# save("./transmon_readout/charge_basis/energies_vs_omega_r.png", fig)
# end