using DataFrames, CSV, CairoMakie

df = CSV.read("kite/data/entropies_results.csv", DataFrame)




# ====== Plotting the Entropies ======
colors = ["#ffb0a3", "#ff6973", "#15788c", "#00b9be"]
spine_hex = "#46425e"

fig = Figure(size = (400, 400), backgroundcolor = :transparent) 

ax = Axis(fig[1, 1], 
            topspinevisible = false,
            rightspinevisible = false,
            xgridvisible = false,
            ygridvisible = false,
            backgroundcolor = :transparent,
            spinewidth = 2,
            leftspinecolor = spine_hex,
            bottomspinecolor = spine_hex,
            xtickwidth = 2,
            ytickwidth = 2,
            xtickcolor = spine_hex,
            ytickcolor = spine_hex,
            xticklabelcolor = spine_hex,
            yticklabelcolor = spine_hex,
            xticklabelfont = :bold,
            yticklabelfont = :bold,
            xticks = ([0, pi/2, pi, 3pi/2, 2pi], 
                      ["0", "π/2", "π", "3π/2", "2π"]))


for s in 1:nb_states
    column_name = "Bond3_State$(s)"
    lines!(ax, df.phi_ext, df[!, column_name], color = colors[s], linewidth=2, alpha=0.7)
end



display(fig)
save("kite/plots/von_Neumann_entropy_bond_3.svg", fig)