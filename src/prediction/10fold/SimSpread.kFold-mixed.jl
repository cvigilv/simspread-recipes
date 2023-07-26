#!/usr/local/bin/julia

using ArgParse
using CUDA
using DelimitedFiles
using LinearAlgebra
using ProgressMeter
using SimSpread
using NamedArrays

function merge(M::AbstractMatrix, N::AbstractMatrix)
    # Get number of columns and rows per matrix
    Mr, Mc = size(M)
    Nr, Nc = size(N)

    # Get names of matrix indices
    Mr_names = names(M, 1)
    Mc_names = names(M, 2)
    Nr_names = names(N, 1)
    Nc_names = names(N, 2)

    # Construct empty matrix to fill merged matrix
    Zmn = zeros(Mr, Nc)
    Znm = zeros(Nr, Mc)

    # @show size(M)
    # @show size(N)
    # @show size(Zmn)
    # @show size(Znm)

    # Construct matrix
    A = [M.array Zmn; Znm N.array]
    A = NamedArray(A)
    setnames!(A, vcat(Mr_names, Nr_names), 1)
    setnames!(A, vcat(Mc_names, Nc_names), 2)

    return A
end

function merge(M::AbstractMatrix, N::AbstractMatrix, MN::AbstractMatrix)
    # Get number of columns and rows per matrix
    Mr, Mc = size(M)
    Nr, Nc = size(N)
    size(MN)

    # Get names of matrix indices
    Mr_names = names(M, 1)
    Mc_names = names(M, 2)
    Nr_names = names(N, 1)
    Nc_names = names(N, 2)

    # Construct matrix
    A = [M.array MN.array; MN.array' N.array]
    A = NamedArray(A)
    setnames!(A, vcat(Mr_names, Nr_names), 1)
    setnames!(A, vcat(Mc_names, Nc_names), 2)

    return A
end

function main(args)
    # Argument parsing {za{{
    configs = ArgParseSettings()

    add_arg_group!(configs, "I/O:")
    @add_arg_table! configs begin
        "--MT"
        help = "M vs T drug-target adjacency matrix"
        required = true
        action = :store_arg
        arg_type = String
        "--NR"
        help = "N vs R drug-target adjacency matrix"
        required = true
        action = :store_arg
        arg_type = String
        "--MM"
        help = "M vs M compound similarity matrix"
        required = true
        action = :store_arg
        arg_type = String
        "--NN"
        help = "N vs N compound similarity matrix"
        required = true
        action = :store_arg
        arg_type = String
        "--MN"
        help = "M vs N compound similarity matrix"
        required = true
        action = :store_arg
        arg_type = String
        "--output-file", "-o"
        help = "Predicted test drug-target interactions"
        required = true
        action = :store_arg
        arg_type = String
    end

    add_arg_group!(configs, "SimSpread parameters:")
    @add_arg_table! configs begin
        "--weighted", "-w"
        help = "Featurization weighting scheme"
        action = :store_true
        "--resolution", "-r"
        help = "Cutoff resolution"
        action = :store_arg
        arg_type = Float64
        default = 0.05
    end
    add_arg_group!(configs, "Cross-Validation:")
    @add_arg_table! configs begin
        "--splits"
        help = "Number of splits in dataset"
        action = :store_arg
        arg_type = Int64
        default = 10
        "--iterations"
        help = "Number of times to repeat cross-validation"
        action = :store_arg
        arg_type = Int64
        default = 5
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
    nsplits = parsed_args["splits"]
    iterations = parsed_args["iterations"]
    if parsed_args["gpu"] && CUDA.functional()
        usegpu = true
        CUDA.device!(parsed_args["gpu-id"])
    else
        usegpu = false
    end

    # Load matrices to memory
    DT1 = read_namedmatrix(parsed_args["MT"], '\t')
    DT2 = read_namedmatrix(parsed_args["NR"], '\t')
    DD1 = read_namedmatrix(parsed_args["MM"], ' ')
    DD2 = read_namedmatrix(parsed_args["NN"], ' ')
    DD3 = read_namedmatrix(parsed_args["MN"], ' ')
    y = merge(DT1, DT2)
    X = merge(DD1, DD2, DD3)

    pbar = Progress(
        Int64(iterations * nsplits * (length(0:step:1)));
        desc="I 1/$(iterations); F 1/$(nsplits); α = 0.00"
    )
    for iter in 1:1:iterations
        splits = split(y, nsplits; seed=iter)
        for (fold_idx, test_idx) in enumerate(splits)
            for α in 0:step:1.0
                # Split dataset in training and testing sets
                train_idx = [s for s in names(y, 1) if s ∉ test_idx]

                ytrain = y[train_idx, :]
                ytest  = y[test_idx, :]
                Xtrain = X[train_idx, train_idx]
                Xtest  = X[test_idx, train_idx]

                # Construct feature-source-target graph for predictions
                Xtrain′ = featurize(Xtrain, α, weighted)
                Xtest′  = featurize(Xtest, α, weighted)
                G = construct(ytrain, ytest, Xtrain′, Xtest′)

                # Predict drug-target interactions of testing set
                yhat = predict(G, ytest; GPU=usegpu)
                clean!(yhat, first(G), ytest)

                # Convert predictions matrix into a data frame
                save(
                    template * "_$(replace(string(α), '.'=>""))_N$(iter).out",
                    fold_idx,
                    yhat,
                    ytest;
                    delimiter=", "
                )

                next!(
                    pbar;
                    desc="I$(iter)/$(iterations); F$(lpad(fold_idx,2,"0"))/$(nsplits); α = $(rpad(α,4,"0"))"
                )
            end
        end
    end
    ProgressMeter.finish!(pbar; desc="DONE!")
end

main(ARGS)
