using Distributions
using Interpolations
using Plots
using Dates
using LinearAlgebra
using Statistics
using Latexify
using DataFrames
using CSV
using JLD
# using GLM
using Polynomials

include("function_file.jl")


mode = Dict(
    "plot_maps" => false,  # true or false - plotting makes the runs much slower
    "build_latex_tables" => false,  # true or false - build latex tables
    "latex_table_filename" => nothing,
    "use_ensembles" => false,  # true or false - run simulations as ensemble
    "n_ensemble" => 50, # number of ensemble members
    "location_used" => 2:5,  # locations used in the analysis
    "measurement_noise" => 10e-2, # measurement noise 10e-2 is good value because of dimensions and shit
    "system_noise" => 0.2,  # system noise # 0.2 is our calculated value
    "use_Kalman" => false,  # do Kalman stuff
    "create_data" => true,
    "alpha" => exp(-10 / (6 * 60)),
    #    tide = 1
    #    waterlevel = 2
    #    synthetic = 3
    #    no_observation = 0
    "with_observation_data" => synthetic, # choose which observations to use during assimilation
    "observation_file" => "groundtruth_0.2.jld", # filename of the observation file
    "gif_filename" => "fig_map_synthetic.gif",
    "assimilate_left_bc" => use_cadzand,
    "initialisation" => nothing
)

minutes_to_seconds = 60.0
hours_to_seconds = 60.0 * 60.0
days_to_seconds = 24.0 * 60.0 * 60.0
seconds_to_hours = 1.0 / hours_to_seconds

function timestep(X, u, A_inv, B, mode)
    # Initialize e vector and replicate it to match the size of X
    e = zeros(Float64, size(X, 1))
    w = repeat(e, 1, size(X, 2))

    # Compute U and replicate it to match the size of X
    if !(mode["assimilate_left_bc"] == keep_ensembles_apart)
        # input is given
        U = repeat(A_inv * u, 1, size(X, 2))
    else
        # input is not given
        U = zeros(Float64, size(X, 1), size(X, 2))
        U[1, :] = X[1, :]
        U = A_inv * U
    end
    # Compute M
    M = A_inv * B

    # Update the first row of w with random values
    alpha_squared = mode["alpha"]^2
    w[end, :] = sqrt(1 - alpha_squared) * mode["system_noise"] * randn(size(w, 2))
    # Compute W
    W = A_inv * w
    # Update state estimates
    X_new = M * X + U + W

    # Calculate the mean of the new state estimates
    x_mean = mean(X_new, dims=2)

    # Compute the L matrix
    L_matrix = (X_new .- x_mean) / sqrt(size(X_new, 2) - 1)

    # Compute the covariance matrix
    P = L_matrix * L_matrix'

    return X_new, P, L_matrix
end


function update_kalman(X_, L_matrix, H, observations, mode)
    # Compute the big Psi matrix
    big_psi_matrix = H * L_matrix

    # Compute the Kalman gain matrix
    K_K = L_matrix * big_psi_matrix' * inv(big_psi_matrix * big_psi_matrix' + mode["measurement_noise"] * I)

    # Reshape and repeat the observations to match the size of H * X_new
    observations = repeat(reshape(observations, size(observations, 1), 1), 1, size(H * X_, 2))

    # Update the state estimates with the Kalman gain
    X = X_ + K_K * (observations - (H * X_) + mode["measurement_noise"] * randn(size(observations)))

    return X
end

function build_measurement_matrix(settings, mode)
    ilocs = settings["ilocs"]
    size_x = 2 * settings["n"]
    H = zeros(Float64, length(mode["location_used"]), size_x + 1)
    j = 0
    for i = mode["location_used"]
        j += 1
        H[j, ilocs[i]] = 1.0
    end
    return H
end


function simulate_enkf(settings, mode, counter_for_prediction=0)
    names = settings["names"]
    names = names[mode["location_used"]]
    t = settings["t"]

    A, B, full_state_data, X = construct_system!(settings, mode)

    A_inv = inv(A)
    u = zeros(Float64, size(B)[1])
    cov_data = zeros(Float64, size(A)[2] - 1, length(t))

    H = build_measurement_matrix(settings, mode)

    observed_data = load_observations(settings, mode)

    nt = length(t)

    for i = 1:nt
        #check whether we want to forecast or not
        if nt - counter_for_prediction < i
            mode["use_Kalman"] = false
            mode["assimilate_left_bc"] = use_cadzand
        end
        # if we assimilate all
        if mode["assimilate_left_bc"] == use_cadzand
            u[1] = settings["h_left"][i]
        elseif mode["assimilate_left_bc"] == use_mean
            u[1] = mean(X[1, :])
        elseif mode["assimilate_left_bc"] == keep_ensembles_apart || mode["assimilate_left_bc"] == use_zero
            u[1] = 0.0
        else
            throw(ArgumentError("Invalid assimilation mode"))
        end

        # timestep
        X, P, L_matrix = timestep(X, u, A_inv, B, mode)
        # Kalman update
        if mode["use_Kalman"]
            X = update_kalman(X, L_matrix, H, observed_data[:, i], mode)
        end

        if mode["use_ensembles"]
            full_state_data[:, :, i] = X[1:end-1, :]
            cov_data[:, i] = diag(P[1:end-1, 1:end-1])
        else
            full_state_data[:, i] = X
        end
    end

    return full_state_data, observed_data, cov_data
end


"""
    Retrieve shape of the full state data and collapse it to a 2D array on the second dimension
    by taking the mean of the ensemble members.
    If the full state data is a 2D array, the function will return the full state data.
    If the mode parameters indicate that the full state data is a 3D array but the
    use_ensembles parameter is disabled, the function will throw an error.
    If the mode parameters indicate that the full state data is a 2D array but the
    use_ensembles parameter is enabled, the function will throw an error.

    Pass the optional parameter ilocs to return only the locations specified in the ilocs array
    as indices on the state vector.
"""
function collapse_full_state_data(full_state_data, mode; indices_like_ilocs=nothing)
    if isnothing(indices_like_ilocs)
        indices_like_ilocs = 1:size(full_state_data, 1)
    end
    if ndims(full_state_data) == 2 && !mode["use_ensembles"]
        return full_state_data[indices_like_ilocs, :]
    elseif ndims(full_state_data) == 2 && mode["use_ensembles"]
        throw(ArgumentError("full_state_data is not a 3D array but ensemble mode is enabled"))
    elseif ndims(full_state_data) == 3 && !mode["use_ensembles"]
        throw(ArgumentError("full_state_data is a 3D array but ensemble mode is disabled"))
    end

    series_data = zeros(Float64, size(full_state_data, 1), size(full_state_data, 3))

    series_data = mean(full_state_data, dims=2)[:, 1, :]

    return series_data[indices_like_ilocs, :]
end

function run_ensemble_nokf_compare_to_measurements()
    mode["num_ensembles"] = 50
    mode["create_data"] = false
    mode["use_ensembles"] = true
    mode["use_Kalman"] = true
    mode["with_observation_data"] = synthetic
    mode["gif_filename"] = "enkf_bias_rmse"
    mode["observation_file"] = "synthetic_0.2.jld"
    mode["assimilate_left_bc"] = use_cadzand
    mode["build_latex_tables"] = true  # true or false - build latex tables
    mode["latex_table_filename"] = "q6_table_enkf_bias_rmse"
    mode["plot_maps"] = true
    mode["location_used"] = 2:5

    settings = create_settings()
    _ = initialize!(settings)

    full_state_data, observed_data, cov_data = simulate_enkf(settings, mode)

    series_data = collapse_full_state_data(full_state_data, mode)
    plot_series_with_name(series_data, observed_data, settings, mode, "enkf_bias_rmse", 0.0)

    # Compute error statistics
    ilocs = settings["ilocs"][mode["location_used"]]
    H_x = series_data[ilocs, :]
    index_start = 62 # start at second rising tide
    names = ["Bath", "Vlissingen", "Terneuzen", "Hansweert", "Bath"]
    compute_statistics(H_x, observed_data[1:end, 2:end], names[mode["location_used"]], mode, size(series_data)[2] - index_start)

    for i in mode["location_used"]
        println("Max std for $(names[i]) found at $(findmax(sqrt.(cov_data[settings["ilocs"][i], :])))")
    end
    anim = @animate for i ∈ 1:(length(settings["t"])-1)
        plot_state_for_gif(series_data[:, i], cov_data, settings, observed_data, i, mode)
    end

    gif(anim, "figures/$(mode["gif_filename"]).gif", fps=10)
    println("gif saved at $(mode["gif_filename"])")
end


function run_synthetic_versus_ensemble_enkf()
    mode["create_data"] = false
    mode["use_ensembles"] = true
    mode["use_Kalman"] = true
    mode["with_observation_data"] = synthetic
    mode["observation_file"] = "synthetic_0.2.jld"
    mode["gif_filename"] = "enkf_versus_synthetic_use_cadzand"
    mode["assimilate_left_bc"] = use_cadzand

    settings = create_settings()
    _ = initialize!(settings)

    full_state_data, observed_data, cov_data = simulate_enkf(settings, mode)

    series_data = collapse_full_state_data(full_state_data, mode)
    plot_series_with_name(series_data, observed_data, settings, mode, "enkf_versus_synthetic_use_cadzand")

    anim = @animate for i ∈ 1:(length(settings["t"])-1)
        plot_state_for_gif(series_data[:, i], cov_data, settings, observed_data, i, mode)
    end

    gif(anim, "figures/$(mode["gif_filename"]).gif", fps=10)
    println("gif saved at $(mode["gif_filename"])")
end


function run_synthetic_versus_ensemble_enkf_nobc_info()
    mode["create_data"] = false
    mode["use_ensembles"] = true
    mode["use_Kalman"] = true
    mode["with_observation_data"] = synthetic
    mode["observation_file"] = "synthetic_0.2.jld"
    mode["gif_filename"] = "enkf_versus_synthetic_nobc_use_zero"
    mode["assimilate_left_bc"] = use_zero

    settings = create_settings()
    _ = initialize!(settings)

    full_state_data, observed_data, cov_data = simulate_enkf(settings, mode)
    series_data = collapse_full_state_data(full_state_data, mode)
    plot_series_with_name(series_data, observed_data, settings, mode, "enkf_versus_synthetic_nobc_use_zero")

    anim = @animate for i ∈ 1:(length(settings["t"])-1)
        plot_state_for_gif(full_state_data[:, :, i], cov_data, settings, observed_data, i, mode)
    end

    gif(anim, "figures/$(mode["gif_filename"]).gif", fps=10)
    println("gif saved at $(mode["gif_filename"])")
end


function run_synthetic_versus_ensemble_nokf()
    mode["create_data"] = false
    mode["use_ensembles"] = true
    mode["use_Kalman"] = false
    mode["with_observation_data"] = synthetic
    mode["observation_file"] = "synthetic_0.2.jld"
    mode["gif_filename"] = "fig_map_ensemble_versus_synthetic_nobcinfo_0_keep"
    mode["assimilate_left_bc"] = keep_ensembles_apart

    settings = create_settings()
    _ = initialize!(settings)

    full_state_data, observed_data, cov_data = simulate_enkf(settings, mode)

    anim = @animate for i ∈ 1:(length(settings["t"])-1)
        plot_state_for_gif(full_state_data[:, :, i], cov_data, settings, observed_data, i, mode)
    end

    gif(anim, "figures/$(mode["gif_filename"]).gif", fps=10)
    println("gif saved at $(mode["gif_filename"])")
end


function run_simulation_and_create_synthetic_data()
    mode["create_data"] = true
    mode["use_ensembles"] = false
    mode["use_Kalman"] = false
    mode["with_observation_data"] = no_observation
    mode["gif_filename"] = "fig_map_synthetic"

    settings = create_settings()
    _ = initialize!(settings)

    full_state_data, _, cov_data = simulate_enkf(settings, mode)
    H = build_measurement_matrix(settings, mode)[:, 1:end-1]

    if mode["create_data"]
        println("synthetic experiment, save data", size(full_state_data))
        title = "synthetic_$(mode["system_noise"]).jld"
        save(title, "synthetic data", full_state_data)
    end

    anim = @animate for i ∈ 1:(length(settings["t"])-1)
        measurements = H * full_state_data
        plot_state_for_gif(full_state_data[:, i], cov_data, settings, measurements, i, mode)
    end

    gif(anim, "figures/$(mode["gif_filename"]).gif", fps=10)
    println("gif saved at $(mode["gif_filename"])")
end


function run_ensemble_enkf_in_storm(counter_for_prediction, plot_gif)
    mode["create_data"] = false
    mode["use_ensembles"] = true
    mode["use_Kalman"] = true
    mode["with_observation_data"] = waterlevel
    mode["gif_filename"] = "enkf_in_storm"
    mode["assimilate_left_bc"] = use_cadzand# keep_ensembles_apart
    settings = create_settings()
    _ = initialize!(settings)

    if counter_for_prediction != 0
        add = "_$((288-counter_for_prediction)*10/60)_h"
    else
        add = ""
    end

    full_state_data, observed_data, cov_data = simulate_enkf(settings, mode, counter_for_prediction)
    series_data = collapse_full_state_data(full_state_data, mode)
    plot_series_with_name(series_data, observed_data, settings, mode, "enkf_in_storm$(add)", counter_for_prediction)

    mean = collapse_full_state_data(full_state_data, mode)
    mean_error = 0#mean_squared_error(mean, observed_data)
    if plot_gif
        anim = @animate for i ∈ 1:(length(settings["t"])-1)
            plot_state_for_gif(full_state_data[:, :, i], cov_data, settings, observed_data, i, mode)
        end

        gif(anim, "figures/$(mode["gif_filename"]).gif", fps=10)
        println("gif saved at $(mode["gif_filename"])")
    end
    return full_state_data, observed_data, mean_error
end


function comparison_prediciton_noKalman(counter_for_prediction, plot_gif)
    mode["create_data"] = false
    mode["use_ensembles"] = true
    mode["use_Kalman"] = false
    mode["with_observation_data"] = waterlevel
    mode["gif_filename"] = "no_enkf_in_storm"
    mode["assimilate_left_bc"] = use_cadzand #keep_ensembles_apart
    settings = create_settings()
    _ = initialize!(settings)

    if counter_for_prediction != 0
        add = "_$((288-counter_for_prediction)*10/60)_h"
    else
        add = ""
    end
    full_state_data, observed_data, cov_data = simulate_enkf(settings, mode, counter_for_prediction)
    series_data = collapse_full_state_data(full_state_data, mode)
    plot_series_with_name(series_data, observed_data, settings, mode, "no enkf_in_storm$(add)", counter_for_prediction)

    mean = collapse_full_state_data(full_state_data, mode)
    mean_error = 0#mean_squared_error(mean, observed_data)
    if plot_gif
        anim = @animate for i ∈ 1:(length(settings["t"])-1)
            plot_state_for_gif(full_state_data[:, :, i], cov_data, settings, observed_data, i, mode)
        end

        gif(anim, "figures/$(mode["gif_filename"]).gif", fps=10)
        println("gif saved at $(mode["gif_filename"])")
    end
    return full_state_data, observed_data, mean_error
end

function run_different_num_enkfs()
    mode["create_data"] = false
    mode["use_ensembles"] = true
    mode["use_Kalman"] = true
    mode["with_observation_data"] = synthetic
    mode["observation_file"] = "synthetic_0.2.jld"
    mode["gif_filename"] = "enkf_varying_membersize_synthetic_hansweert"
    mode["assimilate_left_bc"] = use_cadzand
    settings = create_settings()
    _ = initialize!(settings)

    # Allocate memory for the series data arrays
    num_ensembles = [3, 4, 5, 6, 8, 10, 12, 15]
    biases_all_en_members = zeros(Float64, length(num_ensembles), length(settings["t"]))
    rmses_all_en_members = zeros(Float64, length(num_ensembles), length(settings["t"]))
    for i = 1:length(num_ensembles)
        mode["n_ensemble"] = num_ensembles[i]
        full_state_data, observed_data, cov_data = simulate_enkf(settings, mode)
        series_data = collapse_full_state_data(full_state_data, mode)
        ilocs = settings["ilocs"][mode["location_used"][3:3]]
        biases_all_en_members[i, :], rmses_all_en_members[i, :] = compute_bias_and_rmse_over_time(series_data[ilocs, :], observed_data[4:4, :])
    end
    plot_bias_and_rmse_over_time(biases_all_en_members, rmses_all_en_members, num_ensembles, "enkf_varying_membersize_synthetic_hansweert", settings)
end

function run_different_num_enkfs_compare_whole_state()
    mode["create_data"] = false
    mode["use_ensembles"] = true
    mode["use_Kalman"] = true
    mode["with_observation_data"] = synthetic
    mode["observation_file"] = "synthetic_0.2.jld"
    mode["gif_filename"] = "enkf_varying_membersize_synthetic_height_state_keep_ensembles_apart"
    mode["assimilate_left_bc"] = keep_ensembles_apart
    settings = create_settings()
    _ = initialize!(settings)

    # Allocate memory for the series data arrays
    num_ensembles = rand(4:15, 100)
    # num_ensembles = [3, 10, 50]
    observed_data = load(mode["observation_file"], "synthetic data")
    biases_all_en_members = zeros(Float64, length(num_ensembles), length(settings["t"]))
    rmses_all_en_members = zeros(Float64, length(num_ensembles), length(settings["t"]))
    for (i, val) in enumerate(num_ensembles)
        mode["n_ensemble"] = val
        full_state_data, _, cov_data = simulate_enkf(settings, mode)
        series_data = collapse_full_state_data(full_state_data, mode)
        biases_all_en_members[i, :], rmses_all_en_members[i, :] = compute_bias_and_rmse_over_time(series_data[1:2:end, :], observed_data[1:2:end, :])
    end
    # plot_bias_and_rmse_over_time(biases_all_en_members, rmses_all_en_members, num_ensembles, "enkf_varying_membersize_synthetic_height_state_keep_ensembles_apart", settings)
    biases_sum_over_time = sum(abs.(biases_all_en_members), dims=2)
    rmses_sum_over_time = sum(abs.(rmses_all_en_members), dims=2)
    # Fit the linear regression model
    # Calculate the mean of x and y
    x = log.(2, Array(num_ensembles))
    y = log.(2, biases_sum_over_time[:, 1])
    y2 = log.(2, rmses_sum_over_time[:, 1])
    # Define the polynomial degree
    degree = 1
    # f = fit(xs, ys) # degree = length(xs) - 1
    f = Polynomials.fit(x, y, degree) # degree = 2
    print(map(x -> round(x, digits=4), f))

    p1 = scatter()
    p2 = scatter()
    scatter!(p1, x, y, label="Log2(Sum of biases over number of ensembles [m])")
    plot!(p1, f, extrema(x)..., label="Linear regression")
    scatter!(p2, x, y2, label="Log2(Sum of rmses over number of ensembles [m])", xlabel="log2(Number of ensembles)")
    p = plot(p1, p2, layout=(2, 1))
    savefig(p, "figures/enkf_varying_membersize_synthetic_height_state_keep_ensembles_apart_linfit.pdf")
    # Compute linear regression for the rate of convergence on the bias
end

function run_different_init_state_as_zero()
    mode["create_data"] = false
    mode["use_ensembles"] = true
    mode["use_Kalman"] = true
    mode["with_observation_data"] = synthetic
    mode["observation_file"] = "synthetic_0.2.jld"
    mode["gif_filename"] = "enkf_versus_synthetic_use_cadzand_4_init"
    mode["assimilate_left_bc"] = use_cadzand
    mode["build_latex_tables"] = true  # true or false - build latex tables
    mode["latex_table_filename"] = "q8_bias_and_rmse"
    mode["initialisation"] = zero

    settings = create_settings()
    _ = initialize!(settings)

    full_state_data, observed_data, cov_data = simulate_enkf(settings, mode)

    series_data = collapse_full_state_data(full_state_data, mode)
    plot_series_with_name(series_data, observed_data, settings, mode, "enkf_versus_synthetic_use_cadzand_zero_init")

    ilocs = settings["ilocs"][mode["location_used"]]
    H_x = series_data[ilocs, :]
    index_start = 20 # start at second rising tide
    names = ["Bath", "Vlissingen", "Terneuzen", "Hansweert", "Bath"]
    compute_statistics(H_x, observed_data[1:end, 2:end], names[mode["location_used"]], mode, size(series_data)[2] - index_start)


    anim = @animate for i ∈ 1:(length(settings["t"])-1)
        plot_state_for_gif(series_data[:, i], cov_data, settings, observed_data, i, mode)
    end

    gif(anim, "figures/$(mode["gif_filename"]).gif", fps=10)
    println("gif saved at $(mode["gif_filename"])")
end

function run_enkf_twin_in_storm()
    mode["create_data"] = false
    mode["use_ensembles"] = false
    mode["use_Kalman"] = false
    mode["with_observation_data"] = waterlevel
    mode["gif_filename"] = "noenkf_and_enkf_in_storm"
    mode["assimilate_left_bc"] = use_cadzand# keep_ensembles_apart
    mode["build_latex_tables"] = true  # true or false - build latex tables
    mode["latex_table_filename"] = "q9_bias_and_rmse_noenkf"
    settings = create_settings()
    _ = initialize!(settings)

    full_state_data_1, observed_data_1, cov_data = simulate_enkf(settings, mode, 0.0)

    series_data_1 = collapse_full_state_data(full_state_data_1, mode)

    ilocs = settings["ilocs"][mode["location_used"]]
    H_x = series_data_1[ilocs, :]
    index_start = 62 # start at second rising tide
    names = ["Bath", "Vlissingen", "Terneuzen", "Hansweert", "Bath"]
    compute_statistics(H_x, observed_data_1[1:end, 2:end], names[mode["location_used"]], mode, size(series_data_1)[2] - index_start)

    mode["use_ensembles"] = true
    mode["use_Kalman"] = true
    mode["with_observation_data"] = waterlevel
    mode["assimilate_left_bc"] = use_cadzand
    mode["build_latex_tables"] = true  # true or false - build latex tables
    mode["latex_table_filename"] = "q9_bias_and_rmse_enkf"
    settings = create_settings()
    _ = initialize!(settings)

    full_state_data_2, observed_data_2, cov_data = simulate_enkf(settings, mode, 0.0)

    series_data_2 = collapse_full_state_data(full_state_data_2, mode)

    ilocs = settings["ilocs"][mode["location_used"]]
    H_x_2 = series_data_2[ilocs, :]
    index_start = 62 # start at second rising tide
    names = ["Bath", "Vlissingen", "Terneuzen", "Hansweert", "Bath"]
    compute_statistics(H_x_2, observed_data_2[1:end, 2:end], names[mode["location_used"]], mode, size(series_data_2)[2] - index_start)


    name = "noenkf_and_enkf_in_storm"
    println("Plot at locations. ", name)
    t = settings["t"]
    ilocs = settings["ilocs"][mode["location_used"]]
    loc_names = settings["loc_names"][mode["location_used"]]
    loc_names = replace.(loc_names, "Waterlevel at " => "")


    nseries = length(loc_names)
    series_data_1 = series_data_1[ilocs, :]
    series_data_2 = series_data_2[ilocs, :]
    ntimes = min(length(t), size(observed_data_2, 2))
    plots = []
    for i = nseries-3:nseries
        p = plot(seconds_to_hours .* t, series_data_1[i, :], linecolor=:blue, ylabel="Waterlevel [m]", label="no EnKF", dpi=1000, foreground_color_legend=nothing, size=(800, 600), legend=:topleft)
        plot!(p, seconds_to_hours .* t, series_data_2[i, :], linecolor=:red, ylabel="Waterlevel [m]", label="EnKF", dpi=1000, foreground_color_legend=nothing, size=(800, 600), legend=:topleft)
        plot!(p, seconds_to_hours .* t[1:ntimes], observed_data_2[i, 1:ntimes], linecolor=:black, label="observations")
        title!(p, loc_names[i])
        xlabel!(p, "time [hours]")
        push!(plots, p)
        sleep(0.05) # Slow down to avoid that the plotting backend starts complaining. This is a bug and should be fixed soon.
    end

    # Create a separate legend plot
    legend_plot = plot(legend=true)
    plot!(legend_plot, [NaN, NaN], linecolor=[:blue, :black])

    p_combined = plot(plots..., layout=(2, 2))
    savefig(p_combined, replace("figures/$(mode["gif_filename"]).png", " " => "_"))
    savefig(p_combined, replace("figures/$(mode["gif_filename"]).pdf", " " => "_"))

end

# en = [120, 123, 126, 129, 132, 135, 138, 144, 150] #120 entsprich ab stunde 28 haben wir keinen assimilation mehr. danach entsprechen 3 weitere zeitschritte, dass wir weitere 30 min vorher keine ass mehr haben.
# error_ENKF = zeros(Float64, length(en))
# error_NoENKF = zeros(Float64, length(en))
# s = create_settings()
# _ = initialize!(s)

# for n in en
#     state_data_Strom_ENKF, observed_data_Strom_ENKF, error_ENKF_ = run_ensemble_enkf_in_storm(n, false)
#     comparison_data_storm_noENKF, _, error_NoENKF_ = comparison_prediciton_noKalman(n, false)
#     """error_ENKF[n] = error_ENKF_
#     error_NoENKF[n] = error_NoENKF_"""

#     compare_forecasting(state_data_Strom_ENKF, comparison_data_storm_noENKF, observed_data_Strom_ENKF, s, mode, "comparison_forecasting_$((288-n)*10/60)_h", n)
# end

run_enkf_twin_in_storm()
##########################
## Hier noch ne func die den error in einem plot plotted