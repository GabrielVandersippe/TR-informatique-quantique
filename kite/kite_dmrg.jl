using ITensors, ITensorMPS, LinearAlgebra, HypergeometricFunctions


DEFAULT_DIMS = (19, 32, 32, 5)

ECs_GHz=0.072472
EL_GHz=1.269
ECJ_GHz=4.9895
EJ_GHz=17.501
eps=0.05702
ECc_GHz=0.003989
f_r_GHz=4.337


# ============================================================
# Utils
# ============================================================

function genlaguerre(n::Int, a::Int, x::Float64)
    """Generalized Laguerre polynomial L_n^a(x)"""
    return binomial(n + a, n) * pFq((-n,), (a + 1,), x)
end


function factorial_fraction(n::Union{Int, BigInt}, m::Union{Int, BigInt})
    """n!/m!"""
    m = big(m)
    n = big(n)
    if m == 0
        return factorial(n) #XXX to avoid overflow for large n
    end
    if n <= m
        return 1
    else
        return n * factorial_fraction(n - 1, m)
    end
end


function displacement_ij(p, i, j)
    """
    return <i|exp(i*p*(a + a_dag))|j> = <i|D(i*p)|j> 

    uses Glauber-Cahill formula for displacement with alpha = i*p
    """
    n, m = min(i, j), max(i, j)
    return (
        (1im*p)^(m-n)
        / sqrt(factorial_fraction(m, n)) 
        * exp(-0.5 * p^2)
        * genlaguerre(n, m - n, p^2)
    )
end


function fock_basis_displacement_operator(d::Int, p::Float64)
    """Displacement operator matrix in fock basis of dimension d with alpha = i*p"""
    D = zeros(ComplexF64, d, d)
    for i in 0:d-1
        for j in 0:d-1
            D[i+1, j+1] = displacement_ij(p, i, j)
        end
    end
    return D
end


function phi_zpf(EC_GHz::Float64, EL_GHz::Float64)
    """Zero point fluctuations of phi operator"""
    return (2 * EC_GHz / EL_GHz) ^ 0.25
end


function compute_EC_matrix(
    ECs_GHz::Float64, 
    ECJ_GHz::Float64, 
    ECc_GHz::Float64, 
    fr_GHz::Float64,
    nr_zpf::Float64, 
    eps::Float64
    )
    """Compute the EC matrix for the coupled system"""

    ECr_GHz = fr_GHz / nr_zpf^2 / 16

    a00 = a01 = ECs_GHz
    a11 = 0.5 * ECJ_GHz / (1 - eps^2) + ECs_GHz
    a22 = 0.5 * ECJ_GHz / (1 - eps^2)
    a12 = -0.5 * ECJ_GHz * eps / (1 - eps^2)
    a33 = ECr_GHz
    a03 = a13 = ECc_GHz

    return [
        [a00 a01 0.0 a03];
        [a01 a11 a12 a13];
        [0.0 a12 a22 0.0];
        [a03 a13 0.0 a33];
    ]
end


# ============================================================
# Matrices in charge basis #XXX Right now these are full matrices. Try later with Diagonal/ Tridiagonal for faster arithmetics
# ============================================================

function charge_basis_identity(d::Int)
    @assert d % 2 == 1 "dim must be odd"
    return diagm(ones(ComplexF64, d))
end


function charge_basis_charge_operator(d::Int)
    @assert d % 2 == 1 "dim must be odd"
    half_dim = (d - 1) ÷ 2
    vals = collect(-half_dim:half_dim)
    return diagm(ComplexF64.(vals))
end


function charge_basis_harmonic_hamiltonian(d::Int, ECT_GHz::Float64) #XXX Recomputes charge matrix
    n = charge_basis_charge_operator(d)
    return 4*ECT_GHz*n*n
end


function charge_basis_cos_operator(d::Int)
    """<i|cos(phi)|j> = 0.5 if |i-j| = 1, 0 otherwise"""
    @assert d % 2 == 1 "dim must be odd"
    return 0.5 * (diagm(1 => ones(ComplexF64, d - 1)) + diagm(-1 => ones(ComplexF64, d - 1)))
end


function charge_basis_sin_operator(d::Int)
    """<j-1|sin(phi)|j> = 0.5 * 1j, <j+1|sin(phi)|j> = -0.5 * 1j, 0 otherwise"""
    @assert d % 2 == 1 "dim must be odd"
    return 0.5 * 1im * (diagm(1 => ones(ComplexF64, d - 1)) - diagm(-1 => ones(ComplexF64, d - 1)))
end



# ============================================================
# Matrices in fock basis #XXX Right now these are full matrices. Try later with Diagonal/ Tridiagonal for faster arithmetics
# ============================================================

# XXX Perhaps it could work with just I and take up less space
function fock_basis_identity(d::Int)
    return diagm(ones(ComplexF64, d))
end


function fock_basis_number_operator(d::Int)
    vals = collect(0:d-1)
    return diagm(ComplexF64.(vals))
end


function fock_basis_harmonic_hamiltonian(d::Int, f_GHz::Float64)
    n = fock_basis_number_operator(d)
    return f_GHz * (n + 0.5 * I)
end


function fock_basis_phi_operator(d::Int, phi_zpf::Float64)
    """<i|phi|j> = phi_zpf * (a + a_dag)

    <j-1|a|j> = sqrt(j)
    <j+1|a_dag|j> = sqrt(j+1)

    """
    vals_up = sqrt.(collect(1:d-1))
    vals_down = sqrt.(collect(0:d-2))
    return phi_zpf * (diagm(1 => ComplexF64.(vals_up)) + diagm(-1 => ComplexF64.(vals_down)))
end


function fock_basis_charge_operator(d::Int, phi_zpf::Float64)
    """<i|charge_n|j> = i * 0.5 * (a_dag - a) / phi_zpf

    <j-1|a|j> = sqrt(j)
    <j+1|a_dag|j> = sqrt(j+1)

    """
    vals_up = sqrt.(collect(1:d-1))
    vals_down = sqrt.(collect(0:d-2))
    return 1im * (diagm(-1 => ComplexF64.(vals_down)) - diagm(1 => ComplexF64.(vals_up))) / (2 * phi_zpf)
end


function fock_basis_cos_operator(d::Int, phi_zpf::Float64)
    """<i|cos(phi)|j> using displacement operator"""
    return real(fock_basis_displacement_operator(d, phi_zpf))
end


function fock_basis_sin_operator(d::Int, phi_zpf::Float64)
    """<i|sin(phi)|j> using displacement operator"""
    return imag(fock_basis_displacement_operator(d, phi_zpf))
end



# ============================================================
# Building the Hamiltonian
# ============================================================

function build_static_operators(
    dims::Tuple{Int,Int,Int,Int},
    EC_vec,
    EL_vec)
    """
    Build static operators for DMRG simulation
    dims: tuple of dimensions for each variable (phi, phi_sum, phi_diff, phi_r) 
    """ 

    # Variable 0 (regular phi, in charge basis)
    N0 = charge_basis_charge_operator(dims[1])
    C0 = charge_basis_cos_operator(dims[1])
    S0 = charge_basis_sin_operator(dims[1])


    # Variable 1 (phi_sum, in fock basis)
    phi_1_zpf = phi_zpf(EC_vec[2], EL_vec[2])
    N1 = fock_basis_charge_operator(dims[2], phi_1_zpf)
    C1 = fock_basis_cos_operator(dims[2], phi_1_zpf)
    S1 = fock_basis_sin_operator(dims[2], phi_1_zpf)

    # Variable 2 (phi_diff, in fock basis)
    phi_2_zpf = phi_zpf(EC_vec[3], EL_vec[3])
    N2 = fock_basis_charge_operator(dims[3], phi_2_zpf)
    C2 = fock_basis_cos_operator(dims[3], phi_2_zpf)
    S2 = fock_basis_sin_operator(dims[3], phi_2_zpf)

    #Resonator variable (in fock basis)
    phi__r_zpf = phi_zpf(EC_vec[4], EL_vec[4])
    NR = fock_basis_charge_operator(dims[4], phi__r_zpf)

    return N0, C0, S0, N1, C1, S1, N2, C2, S2, NR
end



function create_hamiltonian(
        dims,
        ECs_GHz, 
        ECJ_GHz, 
        ECc_GHz, 
        f_r_GHz,
        nr_zpf, 
        eps,
        EL_GHz,
        EJ_GHz,
        ng,
        phi_ext)

    # Initializing the matrices for each variable


    # === Initializing the sites ===
    Site0 = siteind("Boson", 1, dim=dims[1]) #phi
    Site1 = siteind("Boson", 2, dim=dims[2]) #phi_sum
    Site2 = siteind("Boson", 3, dim=dims[3]) #phi_diff
    SiteR = siteind("Boson", 4, dim=dims[4]) #phi_r

    sites = [Site0, Site1, Site2, SiteR]
    os = OpSum()

    # === Initializing the matrices ===
    EC_mat = compute_EC_matrix(
        ECs_GHz, 
        ECJ_GHz, 
        ECc_GHz, 
        f_r_GHz,
        nr_zpf, 
        eps
    )

    EL_r_GHz = 2 * f_r_GHz * nr_zpf^2

    EC_diag = diag(EC_mat)
    EL_diag = [
        0.0;
        2*EL_GHz;
        2*EL_GHz;
        EL_r_GHz;
    ]

    f_1_GHz = sqrt(8 * EC_diag[2] * EL_diag[2])
    f_2_GHz = sqrt(8 * EC_diag[3] * EL_diag[3]) 

    # === Initializing static operators ===
    N0, C0, S0, N1, C1, S1, N2, C2, S2, N_R = build_static_operators(dims, diag(EC_mat), EL_diag)

    # === Harmonic Hamiltonians ===

    # H0 
    os += 4*EC_mat[1,1], N0*N0, 1
    # H1 
    os += f_1_GHz, "N", 2
    os += 0.5*f_1_GHz, "I", 2
    # H2 
    os += f_2_GHz, "N", 3
    os += 0.5*f_2_GHz, "I", 3
    # Hr
    os += f_r_GHz, "N", 4
    os += 0.5*f_r_GHz, "I", 4


    # === Coupling terms ===

    #  N_i N_j terms 
    os += 8*EC_mat[1,2], N0, 1, N1, 2
    os += 8*EC_mat[2,3], N1, 2, N2, 3

    # ng coupling to qubit 
    os += EC_mat[1,1]*ng, N0, 1
    os += EC_mat[1,2]*ng, N1, 2

    # ng coupling to resonator
    os += EC_mat[1,4]*ng, N0, 1

    # Coupling to resonator
    os += 8*EC_mat[1,4], N0, 1, N_R, 4
    os += 8*EC_mat[2,4], N1, 2, N_R, 4


    # === Cosine and Sine terms ===
    os += -2*EJ_GHz*cos(phi_ext/2), C0, 1, C1, 2, C2, 3
    os += 2*EJ_GHz*sin(phi_ext/2), C0, 1, C1, 2, S2, 3
    os += -2*EJ_GHz*cos(phi_ext/2), S0, 1, S1, 2, C2, 3
    os += 2*EJ_GHz*sin(phi_ext/2), S0, 1, S1, 2, S2, 3

    return MPO(os, sites)

end



# ============================================================
# Computing the states
# ============================================================


# Observer to stop DMRG early if energy converged
mutable struct EnergyObserver <: AbstractObserver
    energy_tol::Float64
    last_energy::Float64

    EnergyObserver(energy_tol::Float64=0.0) = new(energy_tol, 1000.0)
end

#Overloading the checkdone! method
function ITensorMPS.checkdone!(obs::EnergyObserver; kwargs...)
    energy=kwargs[:energy]
    if abs(energy - obs.last_energy) < obs.energy_tol
        return true
    else
        obs.last_energy = energy
        return false
    end
end


# Computing eigenstates with DMRG
function eigenstates_hamiltonian(H::MPO, n_levels::Int, precision::Float64=1E-6)
    """Compute the first n_levels eigenvalues and eigenvectors of the Hamiltonian H given as MPO"""
# ==== DMRG Parameters ====
    nsweeps = 60
    maxdim = [10,10,10,20,20,40,60]
    cutoff = [1E-9]
    noise = [1E-7]
    weight = 40

    sites = [siteinds(H)[i][2] for i in 1:4]

    obs = EnergyObserver(precision)

    # ==== DMRG Computations ====
    psi0_init = random_mps(sites;linkdims=10) #TODO : improve initial guess
    E0,psi0 = dmrg(H,psi0_init;nsweeps,maxdim,cutoff,observer=obs,outputlevel = 1, eigsolve_krylovdim = 6)
    Psi = [psi0]
    Energies = [E0]
    for _ in 1:(n_levels-1)
        psi_init = random_mps(sites;linkdims=10) #TODO : improve initial guess
        _,psi = dmrg(H, Psi, psi_init;nsweeps,maxdim,cutoff,noise,weight,observer=obs,outputlevel = 1, eigsolve_krylovdim = 6)
        push!(Psi, psi)
        push!(Energies, real(inner(psi',H,psi)))
    end 
    return Energies.-Energies[1], Psi
end