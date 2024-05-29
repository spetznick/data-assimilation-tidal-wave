#About getting started
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
    # velocities at dx/2, 3dx/2, (n-1/2)dx
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
    println(reftime)
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
    L = s["L"]
    dx = s["dx"]
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

    xlocs_tide = [0.0 * L, 0.25 * L, 0.5 * L, 0.75 * L, 0.99 * L]
    xlocs_waterlevel = [0.0 * L, 0.25 * L, 0.5 * L, 0.75 * L]
    ilocs = vcat(map(x -> round(Int, x), xlocs_tide ./ dx) .* 2 .+ 1, map(x -> round(Int, x), xlocs_waterlevel ./ dx) .* 2 .+ 2)
    loc_names = String[]
    s["ilocs"] = ilocs
    names = ["Cadzand", "Vlissingen", "Terneuzen", "Hansweert", "Bath"]
    s["names"] = names
    for i = 1:length(xlocs_tide)
        push!(loc_names, "Waterlevel at x=$(0.001*xlocs_tide[i]) km $(names[i])")
    end
    for i = 1:length(xlocs_waterlevel)
        push!(loc_names, "Velocity at x=$(0.001*xlocs_waterlevel[i]) km $(names[i])")
    end
    s["loc_names"] = loc_names
    return (x, t[1])
end

function load_observations(settings)
    ilocs = settings["ilocs"]
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

    return observed_data
end

#About plotting
function plot_state(x, i, s; cov_data, enkf=false)
    # println("plotting a map.")
    if enkf
        enkf_suffix = "_ENKF"
    else
        enkf_suffix = ""
    end
    #plot all waterlevels and velocities at one time
    xh = 0.001 * s["x_h"]
    p1 = plot(xh, x[1:2:end], ylabel="h", ylims=(-3.0, 5.0), legend=false, ribbon=cov_data, fillalpha=0.2, fillcolor=:blue)
    xu = 0.001 * s["x_u"]
    p2 = plot(xu, x[2:2:end], ylabel="u", ylims=(-2.0, 3.0), xlabel="x [km]", legend=false, ribbon=cov_data, fillalpha=0.2, fillcolor=:blue)
    p = plot(p1, p2, layout=(2, 1))
    savefig(p, "figures/fig_map$(enkf_suffix)_$(string(i,pad=3)).png")
    sleep(0.05)
end

function plot_series(t, series_data, s, obs_data, loc_names, mode)

    println(loc_names)
    nseries = length(loc_names)
    if mode["enkf"]
        enkf_suffix = "_ENKF"
    else
        enkf_suffix = ""
    end
    for i = 1:nseries
        #fig=PyPlot.figure(i+1)
        # Variances are so small that they are not visible in the plot for subsequent plots
        p = plot(seconds_to_hours .* t, series_data[i, :], linecolor=:blue, label=["model"])
        ntimes = min(length(t), size(obs_data, 2))
        plot!(p, seconds_to_hours .* t[1:ntimes], obs_data[i, 1:ntimes], linecolor=:black, label=["model", "measured"])
        title!(p, loc_names[i])
        xlabel!(p, "time [hours]")
        savefig(p, replace("figures/$(loc_names[i])$(enkf_suffix).png", " " => "_"))
        sleep(0.05) #Slow down to avoid that that the plotting backend starts complaining. This is a bug and should be fixed soon.
    end
end

function plot_state_for_gif(x, cov_data, s, observed_data, time,mode)
    #plot all waterlevels and velocities at one time
    #prepare observed data for plotting
    ilocs = s["ilocs"][mode["location_used"]]
    observed_data = observed_data[:, :]

    xh = 0.001 * s["x_h"]
    xu = 0.001 * s["x_u"]
    p1 = plot()
    p2 = plot()
    p1 = scatter!(p1, ilocs/2, observed_data[:,time+1], legend = true, ylims=(-3.0, 5.0), color=:red, markersize=2, label="measurment data")
    for i in 1:size(x, 2)
        #println(observed_data[:, i])
        p1 = plot!(p1, xh, x[1:2:end-1, i], ylabel="h", ylims=(-3.0, 5.0), legend=false, ribbon=cov_data, fillalpha=0.2, fillcolor=:blue)
        p2 = plot!(p2, xu, x[2:2:end, i], ylabel="u", ylims=(-2.0, 3.0), xlabel="x [km]", legend=false, ribbon=cov_data, fillalpha=0.2, fillcolor=:blue)

        # Add observed data to the plots
    end
    p = plot(p1, p2, layout=(2, 1))
    return p
end

#About Statistics
function bias_at_locations(data1, data2, names)
    println("Computing bias at locations.")
    #names = ["Cadzand", "Vlissingen", "Terneuzen", "Hansweert", "Bath"]
    biases = zeros(Float64, length(names))
    nseries = length(names)

    println("nseries: $(nseries)")
    for i = 1:nseries
        biases[i] = compute_bias(data1[i, :], data2[i, :], names[i])
    end
    return biases
end

function rmse_at_locations(data1, data2, names)
    #names = ["Cadzand", "Vlissingen", "Terneuzen", "Hansweert", "Bath"]
    nseries = length(names)
    rmses = zeros(Float64, length(names))
    for i = 1:nseries
        rmses[i] = compute_rmse(data1[i, :], data2[i, :], names[i])
    end
    return rmses
end

function compute_bias(data1, data2, label)
    residual = data1 - data2[2:end, :]
    bias = Statistics.std(residual)
    # println("Bias at $(label): $(bias)")
    return bias
end

function compute_rmse(data1, data2, label)
    residual = data1 - data2[2:end, :]
    rmse = 1 / length(residual) * sqrt(sum(residual .^ 2))
    # println("RMSE at $(label): $(rmse)")
    return rmse
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
    L = s["L"]
    dt = s["dt"]
    wave_speed = L / ((indices_after_lhs_max[1] - indices_local_maxima_left[start_idx]) * dt)
    return wave_speed
end

function build_latex_table_bias_rmse(biases, rmses, mode, names)
    df = DataFrame(Locations=names, biases=biases, rmses=rmses)
    table = latexify(df, env=:table)
    if mode["enkf"]
        enkf_suffix = "_ENKF"
    else
        enkf_suffix = ""
    end
    open("tables/q3_bias_rmse_table$(enkf_suffix).txt", "w") do io
        println(io, table)
    end
end

function compute_statistics(series_data, observed_data, names, mode, index_start)
    # Initialize arrays for biases and rmses
    biases = zeros(Float64, length(names))
    rmses = zeros(Float64, length(names))

    # Compute biases and rmses starting from index_start
    biases = bias_at_locations(series_data[:, index_start:end], observed_data[:, index_start:end], names)
    rmses = rmse_at_locations(series_data[:, index_start:end], observed_data[:, index_start:end], names)

    # Optionally build LaTeX table
    if mode["build_latex_tables"]
        build_latex_table_bias_rmse(biases, rmses, mode, names)
    end
end

