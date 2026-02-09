include("kite_dmrg_reordered.jl")
include("../plotting.jl")

DEFAULT_DIMS = (9,16,16,5) #(19, 32, 32, 5)

ECs_GHz=0.072472
EL_GHz=1.269
ECJ_GHz=4.9895
EJ_GHz=17.501
eps=0.05702
ECc_GHz=0.003989
f_r_GHz=4.337
n_r_zpf=2.0

n_g = 0.5
phi_ext_list = range(0.0, stop=2*pi, length=15)

precision = 1E-10
nb_states = 4

# Computing the energies
energies_list = [Float64[] for _ in 1:nb_states]  

for phi_ext in phi_ext_list
    println("Computing for phi_ext = $phi_ext")

    H = create_hamiltonian(DEFAULT_DIMS, ECs_GHz, ECJ_GHz, ECc_GHz, f_r_GHz, n_r_zpf, eps, EL_GHz, EJ_GHz, n_g, phi_ext)
    energies, _ = eigenstates_hamiltonian(H, nb_states, precision)
    println("Energies: ", energies)
    # Compute energies
    for (i, energy) in enumerate(energies)
        push!(energies_list[i], energy)
    end
end

# Plotting the energies
plot_list(phi_ext_list, energies_list; labels=["State $i" for i in 1:nb_states], xlabel=L"\varphi_{\mathrm{ext}}", ylabel="Energy", title=L"\text{Energy vs }\varphi_{\mathrm{ext}} \text{ for First States}")

df = DataFrame(phi_ext = collect(phi_ext_list))
for i in 1:nb_states
    df[!, "energy_state_$i"] = energies_list[i]
end
CSV.write("kite/data/all_energies_reordered.csv", df)