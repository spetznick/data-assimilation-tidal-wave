#
# 1d shallow water model
#
# solves
# dh/dt + D du/dx = 0
# du/dt + g dh/dx + f*u = 0
#
# staggered discretiztation in space and central in time
#
# o -> o -> o -> o ->   # staggering
# L u  h u  h u  h  R   # element
# 1 2  3 4  5 6  7  8   # index in state vector ; counting starts at 1
#
# m=1/2, 3/2, ...
#  u[n+1,m] + 0.5 g dt/dx ( h[n+1,m+1/2] - h[n+1,m-1/2]) + 0.5 dt f u[n+1,m]
#  = u[n  ,m] - 0.5 g dt/dx ( h[n  ,m+1/2] - h[n  ,m-1/2]) - 0.5 dt f u[n  ,m]
# m=1,2,3,...
#  h[n+1,m] + 0.5 D dt/dx ( u[n+1,m+1/2] - u[n+1,m-1/2])
#  = h[n  ,m] - 0.5 D dt/dx ( u[n  ,m+1/2] - u[n  ,m-1/2])

using Interpolations
using Plots
using Dates
using LinearAlgebra
using Statistics
using Latexify
using DataFrames

plot_maps = false #true or false - plotting makes the runs much slower
build_latex_tables = true #true or false - build latex tables

minutes_to_seconds = 60.0
hours_to_seconds = 60.0 * 60.0
days_to_seconds = 24.0 * 60.0 * 60.0
seconds_to_hours = 1.0 / hours_to_seconds

include("function_file.jl")

wavelengths = []

function read_series(filename::String)
    infile = open(filename)
    times = DateTime[]
    values = Float64[]
    for line in eachline(infile)
        #println(line)
        if Base.startswith(line, "#") || length(line) <= 1
            continue
        end
        parts = split(line)
        push!(times, DateTime(parts[1], "yyyymmddHHMM"))
        push!(values, parse(Float64, parts[2]))
    end
    close(infile)
    return (times, values)
end

function settings()
    s = Dict() #hashmap to  use s['g'] as s.g in matlab
    # Constants
    s["g"] = 9.81 # acceleration of gravity
    s["D"] = 20.0 # Depth
    s["f"] = 1 / (0.06 * days_to_seconds) # damping time scale
    L = 100.e3 # length of the estuary
    s["L"] = L
    n = 100 #number of cells
    s["n"] = n
    # Grid(staggered water levels at 0 (boundary) dx 2dx ... (n-1)dx
    #      velocities at dx/2, 3dx/2, (n-1/2)dx
    dx = L / (n + 0.5)
    s["dx"] = dx
    x_h = range(0, L - dx, length=n)
    s["x_h"] = x_h
    s["x_u"] = x_h .+ 0.5
    # initial condition
    s["h_0"] = zeros(Float64, n)
    s["u_0"] = zeros(Float64, n)
    # time
    t_f = 2.0 * days_to_seconds #end of simulation
    dt = 10.0 * minutes_to_seconds
    s["dt"] = dt
    reftime = DateTime("201312050000", "yyyymmddHHMM") #times in secs relative
    s["reftime"] = reftime
    t = collect(dt * (1:round(t_f / dt))) #expand to numbers with collect
    s["t"] = t
    #boundary (western water level)
    # read from file
    (bound_times, bound_values) = read_series("tide_cadzand.txt")
    bound_t = zeros(Float64, length(bound_times))
    for i = 1:length(bound_times)
        bound_t[i] = (bound_times[i] - reftime) / Dates.Millisecond(1000)
    end
    s["t_left"] = bound_t
    itp = LinearInterpolation(bound_t, bound_values)
    s["h_left"] = itp(t)
    return s
end

function initialize(s) #return (x,t) at initial time
    #compute initial fields and cache some things for speed
    h_0 = s["h_0"]
    u_0 = s["u_0"]
    n = s["n"]
    x = zeros(2 * n) #order h[0],u[0],...h[n],u[n]
    x[1:2:end] = u_0[:]
    x[2:2:end] = h_0[:]
    #time
    t = s["t"]
    reftime = s["reftime"]
    dt = s["dt"]
    sec = Dates.Second(1)
    times = []# reftime+sec*t
    for i = 1:length(t)
        push!(times, ((i * dt) * sec) + reftime)
    end
    s["times"] = times
    #initialize coefficients
    # create matrices in form A*x_new=B*x+alpha
    # A and B are tri-diagonal sparse matrices
    Adata_l = zeros(Float64, 2n - 1) #lower diagonal of tridiagonal
    Adata_d = zeros(Float64, 2n)
    Adata_r = zeros(Float64, 2n - 1)
    Bdata_l = zeros(Float64, 2 * n - 1)
    Bdata_d = zeros(Float64, 2 * n)
    Bdata_r = zeros(Float64, 2 * n - 1)
    #left boundary
    Adata_d[1] = 1.0
    #right boundary
    Adata_d[2*n] = 1.0
    # i=1,3,5,... du/dt  + g dh/sx + f u = 0
    #  u[n+1,m] + 0.5 g dt/dx ( h[n+1,m+1/2] - h[n+1,m-1/2]) + 0.5 dt f u[n+1,m]
    # = u[n  ,m] - 0.5 g dt/dx ( h[n  ,m+1/2] - h[n  ,m-1/2]) - 0.5 dt f u[n  ,m]
    g = s["g"]
    dx = s["dx"]
    f = s["f"]
    temp1 = 0.5 * g * dt / dx
    temp2 = 0.5 * f * dt
    for i = 2:2:(2*n-1)
        Adata_l[i-1] = -temp1
        Adata_d[i] = 1.0 + temp2
        Adata_r[i] = +temp1
        Bdata_l[i-1] = +temp1
        Bdata_d[i] = 1.0 - temp2
        Bdata_r[i] = -temp1
    end
    # i=2,4,6,... dh/dt + D du/dx = 0
    #  h[n+1,m] + 0.5 D dt/dx ( u[n+1,m+1/2] - u[n+1,m-1/2])
    # = h[n  ,m] - 0.5 D dt/dx ( u[n  ,m+1/2] - u[n  ,m-1/2])
    D = s["D"]
    temp1 = 0.5 * D * dt / dx
    for i = 3:2:(2*n-1)
        Adata_l[i-1] = -temp1
        Adata_d[i] = 1.0
        Adata_r[i] = +temp1
        Bdata_l[i-1] = +temp1
        Bdata_d[i] = 1.0
        Bdata_r[i] = -temp1
    end
    # build sparse matrix
    A = Tridiagonal(Adata_l, Adata_d, Adata_r)
    B = Tridiagonal(Bdata_l, Bdata_d, Bdata_r)
    s["A"] = A #cache for later use
    s["B"] = B
    return (x, t[1])
end

function timestep(x, i, settings) #return x one timestep later
    # take one timestep
    A = settings["A"]
    B = settings["B"]
    rhs = B * x
    rhs[1] = settings["h_left"][i] #left boundary
    newx = A \ rhs
    return newx
end

function plot_state(x, i, s)
    println("plotting a map.")
    #plot all waterlevels and velocities at one time
    xh = 0.001 * s["x_h"]
    p1 = plot(xh, x[1:2:end], ylabel="h", ylims=(-3.0, 5.0), legend=false)
    xu = 0.001 * s["x_u"]
    p2 = plot(xu, x[2:2:end], ylabel="u", ylims=(-2.0, 3.0), xlabel="x [km]", legend=false)
    p = plot(p1, p2, layout=(2, 1))
    savefig(p, "figures/fig_map_$(string(i,pad=3)).png")
    sleep(0.05) #slow down a bit or the plotting backend starts complaining.
    #This is a bug and will probably be solved soon.
end

function plot_series(t, series_data, s, obs_data)
    # plot timeseries from model and observations
    loc_names = s["loc_names"][2:5]
    nseries = length(loc_names)
    println("nseries:", nseries)
    plots = []
    for i = 1:nseries
        #fig=PyPlot.figure(i+1)
        p = plot(seconds_to_hours .* t, series_data[i, :], linecolor=:blue, ylabel="Waterlevel [m]", label=["model"], dpi=1000, foreground_color_legend=nothing, size=(800, 600), legend=:topleft)
        ntimes = min(length(t), size(obs_data, 2))
        plot!(p, seconds_to_hours .* t[1:ntimes], obs_data[i, 1:ntimes], linecolor=:black, label=["model", "measured"])
        title!(p, loc_names[i])
        xlabel!(p, "time [hours]")
        push!(plots, p)
        sleep(0.05) #Slow down to avoid that that the plotting backend starts complaining. This is a bug and should be fixed soon.
    end
    p = plot(plots..., layout=(2, 2))
    savefig(p, replace("figures/waterlevel_wave1d.png", " " => "_"))
    savefig(p, replace("figures/waterlevel_wave1d.pdf", " " => "_"))
end

function simulate()
    # for plots
    # locations of observations
    s = settings()
    L = s["L"]
    dx = s["dx"]
    xlocs_waterlevel = [0.0 * L, 0.25 * L, 0.5 * L, 0.75 * L, 0.99 * L]
    xlocs_velocity = [0.0 * L, 0.25 * L, 0.5 * L, 0.75 * L]
    ilocs = vcat(map(x -> round(Int, x), xlocs_waterlevel ./ dx) .* 2 .+ 1, map(x -> round(Int, x), xlocs_velocity ./ dx) .* 2 .+ 2)

    loc_names = String[]
    names = ["Cadzand", "Vlissingen", "Terneuzen", "Hansweert", "Bath"]
    for i = 1:length(xlocs_waterlevel)
        push!(loc_names, "Waterlevel at x=$(0.001*xlocs_waterlevel[i]) km $(names[i])")
    end
    for i = 1:length(xlocs_velocity)
        push!(loc_names, "Velocity at x=$(0.001*xlocs_velocity[i]) km $(names[i])")
    end
    s["xlocs_waterlevel"] = xlocs_waterlevel
    s["xlocs_velocity"] = xlocs_velocity
    s["ilocs"] = ilocs
    s["loc_names"] = loc_names

    (x, t0) = initialize(s)
    t = s["t"]
    times = s["times"]
    series_data = zeros(Float64, length(ilocs), length(t))
    nt = length(t)
    for i = 1:nt
        # println("timestep $(i), $(round(i/nt*100,digits=1)) %")
        x = timestep(x, i, s)
        if plot_maps == true
            plot_state(x, i, s) #Show spatial plot.
            #Very instructive, but turn off for production
        end
        series_data[:, i] = x[ilocs]
    end
    #load observations
    (obs_times, obs_values) = read_series("tide_cadzand.txt")
    observed_data = zeros(Float64, length(ilocs), length(obs_times))
    observed_data[1, :] = obs_values[:]
    (obs_times, obs_values) = read_series("tide_vlissingen.txt")
    observed_data[2, :] = obs_values[:]
    (obs_times, obs_values) = read_series("tide_terneuzen.txt")
    observed_data[3, :] = obs_values[:]
    (obs_times, obs_values) = read_series("tide_hansweert.txt")
    observed_data[4, :] = obs_values[:]
    (obs_times, obs_values) = read_series("tide_bath.txt")
    observed_data[5, :] = obs_values[:]

    #plot timeseries
    plot_series(t, series_data, s, observed_data)


    # compute Statistics
    index_start = 62 # start at second rising tide
    biases = zeros(Float64, length(names))
    rmses = zeros(Float64, length(names))
    names = ["Cadzand", "Vlissingen", "Terneuzen", "Hansweert", "Bath"]
    biases = bias_at_locations(series_data[:, index_start:end], observed_data[:, index_start+1:end], names) # ignore first data points due to dynamic behaviour in the beginning
    rmses = rmse_at_locations(series_data[:, index_start:end], observed_data[:, index_start+1:end], names) # ignore first data points due to dynamic behaviour in the beginning
    if build_latex_tables
        build_latex_table_bias_rmse(biases, rmses, names; filename="q3_table")
    end
    println("All figures have been saved to files.")
    if plot_maps
        println("You can plotting of maps off by setting plot_maps to false.")
        println("This will make the computation much faster.")
    else
        println("You can plot maps by setting plot_maps to true.")
    end

    println("Wave speed: $(compute_wave_propagation_speed(series_data, s))")
    # return wavelengths, series_data
end

function compute_local_max_indices(x)
    indices_local_maxima::Vector{Int32} = []
    previous_value::Float64 = x[1]
    for i = 2:length(x)-1
        if x[i] > previous_value && x[i] > x[i+1]
            push!(indices_local_maxima, i)
        end
        previous_value = x[i]
    end
    return indices_local_maxima
end

function compute_wave_propagation_speed(series_data, s::Dict)
    indices_local_maxima_left = compute_local_max_indices(series_data[1, :])
    indices_local_maxima_right = compute_local_max_indices(series_data[5, :])
    start_idx = 1
    indices_after_lhs_max = indices_local_maxima_right[indices_local_maxima_right.>indices_local_maxima_left[start_idx]]
    xlocs_waterlevel = s["xlocs_waterlevel"]
    L = xlocs_waterlevel[end] - xlocs_waterlevel[1]
    dt = s["dt"]
    wave_speed = L / ((indices_after_lhs_max[1] - indices_local_maxima_left[start_idx]) * dt)
    return wave_speed
end

simulate()