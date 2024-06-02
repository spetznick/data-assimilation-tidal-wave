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
    "build_latex_tables" => true,  # true or false - build latex tables
    "use_ensembles" => true,  # true or false - run simulations as ensemble
    "n_ensemble" => 50, # number of ensemble members
    "location_used" => 2:5,  # locations used in the analysis
    "measurement_noise" => 10e-2, # measurement noise 10e-2 is good value because of dimensions and shit
    "system_noise" => 0.2,  # system noise # 0.2 is our calculated value
    "use_Kalman" => true,  # do Kalman stuff
    "create_data" => false,
    "alpha" => exp(-10 / (6 * 60)),
    # "run twin experiment" => false,
    "with_observation_data" => tide, # choose which observations to use during assimilation
    "observation_file" => "groundtruth_0.2.jld", # filename of the observation file
)

minutes_to_seconds = 60.0
hours_to_seconds = 60.0 * 60.0
days_to_seconds = 24.0 * 60.0 * 60.0
seconds_to_hours = 1.0 / hours_to_seconds

function update_ENKF_newest(X, A_inv, B, u, H, observations, mode)
    # Initialize e vector and replicate it to match the size of X

    e = zeros(Float64, size(X, 1))
    w = repeat(e, 1, size(X, 2))

    # Compute U and replicate it to match the size of X
    U = repeat(A_inv * u, 1, size(X, 2))

    # Compute M
    M = A_inv * B

    # Update the first row of w with random values
    alpha_squared = mode["alpha"]^2
    w[end, :] = sqrt(1 - alpha_squared) * mode["system_noise"] * randn(size(w, 2))
    # Compute W
    W = A_inv * w
    # Update state estimates
    x_new = M * X + U + W

    # Calculate the mean of the new state estimates
    x_mean = mean(x_new, dims=2)

    # Compute the L matrix
    L_matrix = (x_new .- x_mean) / sqrt(size(x_new, 2) - 1)

    # Compute the covariance matrix
    P = L_matrix * L_matrix'

    if mode["use_Kalman"]
        # Compute the big Psi matrix
        big_psi_matrix = H * L_matrix

        # Compute the Kalman gain matrix
        K_K = L_matrix * big_psi_matrix' * inv(big_psi_matrix * big_psi_matrix' + mode["measurement_noise"] * I)

        # Reshape and repeat the observations to match the size of H * x_new
        observations = repeat(reshape(observations, size(observations, 1), 1), 1, size(H * x_new, 2))

        # Update the state estimates with the Kalman gain
        x_new = x_new + K_K * (observations - (H * x_new))

        return x_new, P
    else
        return x_new, P
    end
end

function simulate_ENKF(mode)

    s = settings()
    (x, t0) = initialize(s)
    ilocs = s["ilocs"]
    names = s["names"]
    names = names[mode["location_used"]]
    t = s["t"]

    A, B, full_state_data, X = construct_system(x, s, mode)

    A_inv = inv(A)

    u = zeros(Float64, size(B)[1])

    series_data = zeros(Float64, length(ilocs), length(t))
    cov_data = zeros(Float64, length(x), length(t))

    nt = length(t)
    H = zeros(Float64, length(mode["location_used"]), length(x) + 1)
    j = 0
    for i = mode["location_used"]
        j += 1
        H[j, ilocs[i]] = 1.0
    end

    observed_data = load_observations(s, mode)

    for i = 1:nt
        u[1] = s["h_left"][i]

        X, P = update_ENKF_newest(X, A_inv, B, u, H, observed_data[:, i], mode)
        x = mean(X[1:end-1, :], dims=2)

        if mode["plot_maps"]
            plot_state(x, i, s; cov_data=diag(P[1:end-1, 1:end-1]), enkf=mode["use_Kalman"]) #Show spatial plot.
            #Very instructive, but turn off for production
        end
        series_data[:, i] = x[ilocs]
        if mode["use_ensembles"]
            full_state_data[:, :, i] = X[1:end-1, :]
            cov_data[:, i] = diag(P[1:end-1, 1:end-1])
        else
            full_state_data[:, i] = X
        end
    end

    plot_series(t, series_data, s, observed_data, names, mode)

    compute_statistics(series_data, observed_data, names, mode, 225) #die letzten 225 punkte werden für die statistik verwendet

    if mode["create_data"]
        println("Twin experiment, save data", size(full_state_data))
        title = "twin_$(mode["system_noise"]).jld"
        save(title, "Twin Data", full_state_data)
    end

    return full_state_data, observed_data, cov_data, s
end

full_state_data, observed_data, cov_data, s = simulate_ENKF(mode)
anim = @animate for i ∈ 1:(length(s["t"])-1)
    if mode["use_ensembles"]
        plot_state_for_gif(full_state_data[:, :, i], cov_data, s, observed_data, i, mode)
    else
        plot_state_for_gif(full_state_data[:, i], cov_data, s, observed_data, i, mode)
    end
end

gif(anim, "figures/fig_map_enkf.gif", fps=10)
