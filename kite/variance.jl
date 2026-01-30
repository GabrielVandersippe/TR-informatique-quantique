include("kite_dmrg.jl")
include("../plotting.jl")
using CSV, DataFrames

## ==== Parameters =====
DEFAULT_DIMS = (9,16,16,5) #(19, 32, 32, 5)  # Default dimensions for the kite model

ECs_GHz=0.072472
EL_GHz=1.269
ECJ_GHz=4.9895
EJ_GHz=17.501
eps=0.05702
ECc_GHz=0.003989
f_r_GHz=4.337
n_r_zpf=2.0

n_g = 0.5
phi_ext_list = range(0.0, stop=1.0, length=9)

precision = 1E-15
nb_states = 4



function compute_variance(psi::MPS, H::MPO)
    """Compute the variance of the energy for a given MPS psi and Hamiltonian H, defined as:
    sigma = <psi|H^2|psi> - <psi|H|psi>^2
    """
    E = real(inner(psi', H, psi))
    Hpsi = H*psi
    E2 = real(inner(Hpsi, Hpsi))
    return abs(E2 - E^2)
end

# Computing the variances
sigmas = [Float64[] for _ in 1:nb_states]  

for phi_ext in phi_ext_list
    println("Computing for phi_ext = $phi_ext")

    H = create_hamiltonian(DEFAULT_DIMS, ECs_GHz, EL_GHz, ECJ_GHz, EJ_GHz, eps, ECc_GHz, f_r_GHz, n_r_zpf, n_g, phi_ext)
    energies, states = eigenstates_hamiltonian(H, nb_states, precision)
    
    # Compute variances
    for (i, psi) in enumerate(states)
        push!(sigmas[i], compute_variance(psi, H))
    end
end


# Plotting the variances
plot_list(phi_ext_list, sigmas; labels=["State $i" for i in 1:nb_states], xlabel=L"\varphi_{\mathrm{ext}}", ylabel="Variance", title=L"\text{Energy Variance vs }\varphi_{\mathrm{ext}} \text{ for First States}")

df = DataFrame(phi_ext = collect(phi_ext_list))
for i in 1:nb_states
    df[!, "sigma_state_$i"] = sigmas[i]
end
CSV.write("kite/variances/all_sigmas.csv", df)