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
    @show Mr, Mc = size(M)
    @show Nr, Nc = size(N)

    # Get names of matrix indices
    Mr_names = names(M, 1)
    Mc_names = names(M, 2)
    Nr_names = names(N, 1)
    Nc_names = names(N, 2)

    # Construct empty matrix to fill merged matrix
    Zmn = zeros(Mr, Nc)
    Znm = zeros(Nr, Mc)

    @show size(M)
    @show size(N)
    @show size(Zmn)
    @show size(Znm)

    # Construct matrix
    A = [M.array Zmn; Znm N.array]
    A = NamedArray(A)
    setnames!(A, vcat(Mr_names, Nr_names), 1)
    setnames!(A, vcat(Mc_names, Nc_names), 2)

    return A
end

function merge(M::AbstractMatrix, N::AbstractMatrix, MN::AbstractMatrix)
    # Get number of columns and rows per matrix
    @show Mr, Mc = size(M)
    @show Nr, Nc = size(N)
    @show size(MN)

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
    # Argument parsing {{{
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
    if parsed_args["gpu"] && CUDA.functional()
        usegpu = true
        CUDA.device!(parsed_args["gpu-id"])
    else
        usegpu = false
    end
    # }}}

    # Load matrices to memory
    DT1 = read_namedmatrix(parsed_args["MT"], '\t')
    DT2 = read_namedmatrix(parsed_args["NR"], '\t')
    DD1 = read_namedmatrix(parsed_args["MM"], ' ')
    DD2 = read_namedmatrix(parsed_args["NN"], ' ')
    DD3 = read_namedmatrix(parsed_args["MN"], ' ')
    DT = merge(DT1, DT2)
    DD = merge(DD1, DD2, DD3)

    # Predict interactions
    pbar = Progress(length(0:step:1); desc="α = 0.00")
    for α in 0:step:1.0
        DF = featurize(DD, α, weighted)
        I = construct(DT, DF)
        R = predict(I, DT; GPU=usegpu)
        clean!(R, I, DT)
        save(template * "_$(replace(string(α), '.'=>"")).out", R, DT)

        next!(
            pbar;
            desc="α = $(rpad(α, 4, "0"))"
        )
    end
    ProgressMeter.finish!(pbar; desc="DONE!")
end

main(ARGS)
