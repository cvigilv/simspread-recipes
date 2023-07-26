#!/usr/local/bin/julia

using ArgParse
using CUDA
using DelimitedFiles
using LinearAlgebra
using NamedArrays
using ProgressMeter
using SimSpread

function main(args)
    # Argument parsing {{{
    configs = ArgParseSettings()

    add_arg_group!(configs, "I/O:")
    @add_arg_table! configs begin
        "--dt-train"
        help = "Training drug-target adjacency matrix"
        required = true
        action = :store_arg
        arg_type = String
        "--dt-test"
        help = "Testing drug-target adjacency matrix"
        required = true
        action = :store_arg
        arg_type = String
        "--dd-train"
        help = "Training drug-drug similarity matrix"
        required = true
        action = :store_arg
        arg_type = String
        "--dd-test"
        help = "Testing drug-drug similarity matrix"
        required = true
        action = :store_arg
        arg_type = String
        "--output-file", "-o"
        help = "Predicted test drug-target interactions"
        required = true
        action = :store_arg
        arg_type = String
    end

    add_arg_group!(configs, "wl-SimSpread parameters:")
    @add_arg_table! configs begin
        "--weighted", "-w"
        help = "Similarity matrix featurization weighting scheme"
        action = :store_true
        "--resolution", "-r"
        help = "z-Cutoff resolution"
        action = :store_arg
        arg_type = Float64
        default = 0.1
    end

    add_arg_group!(configs, "Miscellaneous:")
    @add_arg_table! configs begin
        "--gpu"
        help = "GPU acceleration"
        action = :store_true
        "--gpu-id"
        help = "GPU to use"
        action = :store_arg
        arg_type = Int64
        default = 0
    end

    parsed_args = parse_args(args, configs)

    # Store arguments to variables
    weighted = parsed_args["weighted"]
    step = parsed_args["resolution"]
    template = parsed_args["output-file"]
    # gpu = parse_args["gpu"]
    # }}}

    # Load matrices to memory
    DT₀ = read_namedmatrix(parsed_args["dt-train"])
    DT₁ = read_namedmatrix(parsed_args["dt-test"])
    DD₀ = read_namedmatrix(parsed_args["dd-train"])
    DD₁ = read_namedmatrix(parsed_args["dd-test"])

    D₀ = names(DT₀, 1)
    T₀ = names(DT₀, 2)
    D₁ = names(DT₁, 1)
    T₁ = names(DT₁, 2)

    if length(D₀ ∩ D₁) > 0
        println("Training and test sets share $(length(D₀ ∩ D₁)) entries, adding prefix to \
            training set!")
        setnames!(DT₀, ["s_$f" for f in D₀], 1)
        setnames!(DD₀, ["s_$f" for f in D₀], 1)
    end

    @show size(DT₀)
    @show size(DT₁)
    @show size(DD₀)
    @show size(DD₁)

    # Predict interactions
    pbar = Progress(Int64((length(0:step:1))); desc="α = 0.0")
    for α in 0:step:1.0
        DF₀ = featurize(DD₀, α, weighted)
        DF₁ = featurize(DD₁, α, weighted)
        A, B = construct((DT₀, DT₁), (DF₀, DF₁))
        R = predict(A, B, (names(A, 1), names(A, 2)); GPU=CUDA.functional())
        save(template * "_$(replace(string(α), '.'=>""))_N1.out", R, DT₁)
        next!(
            pbar;
            desc="α = $(rpad(α, 4, '0'))"
        )
    end
end

main(ARGS)
