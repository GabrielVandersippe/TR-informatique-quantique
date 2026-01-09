using ITensors, ITensorMPS, LinearAlgebra, SparseArrays, KrylovKit

# --- Projection function --- (UNUSED)
function operator_nlevel_trunc(nlevels, op, states)
    U = states[:, 1:nlevels]
    return U' * op * U
end



# --- Parameters ---
ECT = 0.1
ECR = 0.5
ECoup = 0.005
EJ = 50
EL = 0.5



# --- Hamiltonian function ---
function hamiltonian_tr(ECT, ECR, ECoup, EJ, EL; transmon_trunc=41, resonator_trunc=40)
    


    # ===== Transmon Hamiltonian =====
    charge = spzeros(Float64, transmon_trunc, transmon_trunc)
    cos_phi = spzeros(Float64, transmon_trunc, transmon_trunc)

    for i in 1:(transmon_trunc - 1)
        cos_phi[i, i + 1] = 1
        cos_phi[i + 1, i] = 1
        charge[i, i] = i - (transmon_trunc + 1) ÷ 2
    end

    charge[end, end] = transmon_trunc ÷ 2  

    HT = 4 * ECT * charge * charge - EJ / 2 * cos_phi

    # Evecs = eigen(Matrix(HT)).vectors 
    # HT_ndim = operator_nlevel_trunc(n_levels_transmon, Matrix(HT), Evecs)
    # charge_ndim = operator_nlevel_trunc(n_levels_transmon, Matrix(charge), Evecs)



    # ===== Resonator Hamiltonian =====
    destruction = spzeros(Float64, resonator_trunc, resonator_trunc)

    for n in 1:(resonator_trunc - 1)
        destruction[n, n + 1] = sqrt(n)
    end

    creation = destruction'

    HR = sqrt(8 * ECR * EL) * (creation * destruction + spdiagm(0 => ones(resonator_trunc)) * 0.5)



    # ===== Coupling Hamiltonian =====
    phizpf = ((2 * ECR) / EL)^(1/4)
    n_R = (destruction - creation) / (2im * phizpf)

    HC = -4 * ECoup * kron(charge, n_R)



    # ===== Final Hamiltonian =====
    H = kron(HT, spdiagm(0 => ones(resonator_trunc))) +
        kron(spdiagm(0 => ones(transmon_trunc)), HR) +
        HC

    return H
end




function states_dmrg(ECT, ECR, ECoup, EJ, EL; nb_states = 4, transmon_trunc=41, resonator_trunc=40) #Transmon_trunc must be odd
    
    # === Initialize the sites and the OpSum ===
    T = siteind("Boson", 1, dim = transmon_trunc)
    R = siteind("Boson", 2, dim = resonator_trunc)
    sites = [T, R]
    os = OpSum()

    # ===== Transmon Hamiltonian =====
    omega_p = sqrt(8*ECT*EJ)
    os += omega_p, "N", 1
    os += -ECT/2, "a * a * adag * adag", 1

    # ===== Resonator Hamiltonian =====
    omega_q = sqrt(8*ECR*EL)
    os += omega_q, "N", 2
    os += 0.5, "I", 2

    # ===== Coupling Hamiltonian =====
    os += -4 * ECoup, "N", 1, "N", 2
    


# ---- Computing the ground state ----

    H =  MPO(os, sites)

    # ==== DMRG Parameters ====
    nsweeps = 30
    maxdim = [10,10,10,20,20,40,80,100,200,200]
    cutoff = [1E-8]
    noise = [1E-6]
    weight = 20

    # ==== DMRG Computations ====
    psi0_init = random_mps(sites;linkdims=20)
    E0,psi0 = dmrg(H,psi0_init;nsweeps,maxdim,cutoff)
    Psi = [psi0]
    Energies = [E0]
    for i in 1:(nb_states-1)
        psi_init = random_mps(sites;linkdims=20)
        energy,psi = dmrg(H,Psi, psi_init;nsweeps,maxdim,cutoff,noise,weight)
        push!(Psi, psi)
        push!(Energies, energy)
    end 

    return Energies

end



# ====== Comparaison ======
let
    T = siteind("Boson", 1, dim = 41) #Transmon
    R = siteind("Boson", 2, dim = 40) #Resonator
    sites = [T, R]

    H = hamiltonian_tr(ECT, ECR, ECoup, EJ, EL)
    Energies_krylov, _, _ = eigsolve(H, 6, :SR)
    Energies_krylov .-= Energies_krylov[1]

    Energies_dmrg = states_dmrg(ECT, ECR, ECoup, EJ, EL; nb_states=6)
    Energies_dmrg .-= Energies_dmrg[1]

    for i in 1:6
        println("E$(i-1): Krylov: $(Energies_krylov[i]) ; DMRG: $(Energies_dmrg[i])")
    end
end