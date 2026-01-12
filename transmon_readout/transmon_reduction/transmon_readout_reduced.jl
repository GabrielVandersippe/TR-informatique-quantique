using ITensors, ITensorMPS, LinearAlgebra, SparseArrays, KrylovKit, CairoMakie



function transmon_hamiltonian(ECT,EJ,N_trunc = 3, transmon_trunc=41)
    # ===== Define operators in the full basis =====
    charge = ComplexF64[(i == j) ? (i - transmon_trunc÷2 - 1) : 0.0 for i in 1:transmon_trunc, j in 1:transmon_trunc]
    cos_phi = ComplexF64[(i == j+1) || (i+1 == j) ? 0.5 : 0.0 for i in 1:transmon_trunc, j in 1:transmon_trunc]

    # ==== Full Hamiltonian ====
    H_full = 4 * ECT * charge*charge - EJ * cos_phi

    # ===== Diagonalize the full Hamiltonian =====
    _, evecs = eigen(H_full)

    # ===== Keep the first N_trunc levels =====
    U = evecs[:, 1:N_trunc]
    H_reduced = U' * H_full * U
    charge_reduced = U' * charge * U
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