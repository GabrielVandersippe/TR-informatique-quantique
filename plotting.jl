using CairoMakie

function plot_list(x, ys; labels=[], xlabel="x", ylabel="y", title="Plot", savepath=nothing)
    fig = Figure(size = (800, 600))
    ax = Axis(fig[1, 1], xlabel=xlabel, ylabel=ylabel, title=title)
    
    for (i, y) in enumerate(ys)
        label = isempty(labels) ? nothing : labels[i]
        lines!(ax, x, y, label=label)
    end
    
    axislegend(ax)
    display(fig)

    if savepath !== nothing
        save(savepath, fig)
    end
end