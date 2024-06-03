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

include("function_file.jl")


mode = Dict(
    "plot_maps" => false,  # true or false - plotting makes the runs much slower
    "build_latex_tables" => false,  # true or false - build latex tables
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
    X = X_ + K_K * (observations - (H * X_))

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


function simulate_enkf(settings, mode)
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
    println(size(series_data))
    println(size(mean(full_state_data, dims=2)[:, 1, :]))
    series_data = mean(full_state_data, dims=2)[:, 1, :]

    return series_data[indices_like_ilocs, :]
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

    anim = @animate for i ∈ 1:(length(s["t"])-1)
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

    anim = @animate for i ∈ 1:(length(s["t"])-1)
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

    anim = @animate for i ∈ 1:(length(s["t"])-1)
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

    anim = @animate for i ∈ 1:(length(s["t"])-1)
        measurements = H * full_state_data
        plot_state_for_gif(full_state_data[:, i], cov_data, settings, measurements, i, mode)
    end

    gif(anim, "figures/$(mode["gif_filename"]).gif", fps=10)
    println("gif saved at $(mode["gif_filename"])")
end

function run_ensemble_enkf_in_storm()
    mode["create_data"] = false
    mode["use_ensembles"] = true
    mode["use_Kalman"] = true
    mode["with_observation_data"] = waterlevel
    mode["gif_filename"] = "enkf_in_storm"
    mode["assimilate_left_bc"] = keep_ensembles_apart

    settings = create_settings()
    _ = initialize!(settings)

    full_state_data, observed_data, cov_data = simulate_enkf(settings, mode)
    series_data = collapse_full_state_data(full_state_data, mode)
    plot_series_with_name(series_data, observed_data, settings, mode, "enkf_in_storm")

    anim = @animate for i ∈ 1:(length(s["t"])-1)
        plot_state_for_gif(full_state_data[:, :, i], cov_data, settings, observed_data, i, mode)
    end

    gif(anim, "figures/$(mode["gif_filename"]).gif", fps=10)
    println("gif saved at $(mode["gif_filename"])")
end

run_ensemble_enkf_in_storm()

