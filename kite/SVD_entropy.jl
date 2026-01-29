include("kite_dmrg.jl")
include("../plotting.jl")


## ==== Parameters =====
DEFAULT_DIMS = (9,16,16,4) #(19, 32, 32, 5)  # Default dimensions for the kite model

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

precision = 1E-8
nb_states = 4



function vn_entropy(psi::MPS, bond::Int)
    """Compute the von Neumann entropy across a given bond for MPS psi."""
    psiortho = orthogonalize(psi, bond)
    _,S,_ = svd(psiortho[bond], (linkinds(psiortho, bond-1)..., siteinds(psiortho, bond)...))
    SvN = 0.0
    for n=1:dim(S, 1)
        p = S[n,n]^2
        SvN -= p * log(p)
    end
    return SvN, diag(S)
end



# Computing the Entropies
entropies = [[Float64[] for _ in 1:nb_states] for _ in 1:3]
S_vec = [[Float64[] for _ in 1:3] for _ in 1:2]  

for phi_ext in phi_ext_list
    println("Computing for phi_ext = $phi_ext")

    H = create_hamiltonian(DEFAULT_DIMS, ECs_GHz, EL_GHz, ECJ_GHz, EJ_GHz, eps, ECc_GHz, f_r_GHz, n_r_zpf, n_g, phi_ext)
    _, states = eigenstates_hamiltonian(H, nb_states, precision)
    
    # Compute variances
    for (i, psi) in enumerate(states)
        push!(entropies[1][i], vn_entropy(psi, 1)[1])
        push!(entropies[2][i], vn_entropy(psi, 2)[1])
        push!(entropies[3][i], vn_entropy(psi, 3)[1])
        if phi_ext == 0.5 && i==1
            S_vec[1][1] = vn_entropy(psi, 1)[2]
            S_vec[1][2] = vn_entropy(psi, 2)[2]
            S_vec[1][3] = vn_entropy(psi, 3)[2]
        end
        if phi_ext == 0.5 && i==4
            S_vec[2][1] = vn_entropy(psi, 1)[2]
            S_vec[2][2] = vn_entropy(psi, 2)[2]
            S_vec[2][3] = vn_entropy(psi, 3)[2]
        end
    end


end


# ====== Plotting the Entropies ======
fig = Figure(size = (1200, 1200)) 
bonds_to_plot = [1, 2, 3]

Label(fig[1, 2], L"\text{Von Neumann Entropy vs }\varphi_{\mathrm{ext}} \text{ for First States}", 
      fontsize = 20,
      font = :bold,
      halign = :center)

Label(fig[3, 2], L"\text{Singular values at }\varphi_{\mathrm{ext}} = 0.5 \text{ for State 2}", 
      fontsize = 20,
      font = :bold,
      halign = :center)

Label(fig[5, 2], L"\text{Singular values at }\varphi_{\mathrm{ext}} = 0.5 \text{ for State 4}", 
      fontsize = 20,
      font = :bold,
      halign = :center)

for (idx, b) in enumerate(bonds_to_plot)

    ax = Axis(fig[2, idx], 
              xlabel=L"\varphi_{\mathrm{ext}}", 
              ylabel= idx == 1 ? "Entropy" : "", 
              title="Bond $b")

    for s in 1:nb_states
        lines!(ax, phi_ext_list, entropies[b][s], label="State $s")
    end
    if idx == 3
        axislegend(ax, position = :rt)
    end


    ax_bar_1 = Axis(fig[4, idx], 
            yscale = log10,
            ylabel= idx == 1 ? L"\sigma_i" : "",
            xticks = 1:length(S_vec[1][idx]))
    
    barplot!(ax_bar_1, 1:length(S_vec[1][idx]), S_vec[1][idx], 
        color = Pattern('/'), 
        strokewidth = 1, 
        strokecolor = :black,
        width = 0.3)


    ax_bar_2 = Axis(fig[6, idx], 
            yscale = log10,
            ylabel= idx == 1 ? L"\sigma_i" : "",
            xticks = 1:length(S_vec[2][idx]))
    
    barplot!(ax_bar_2, 1:length(S_vec[2][idx]), S_vec[2][idx], 
        color = Pattern('/'), 
        strokewidth = 1, 
        strokecolor = :black,
        width = 0.3)

end

display(fig)