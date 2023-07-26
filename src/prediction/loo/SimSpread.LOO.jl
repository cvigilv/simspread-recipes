#!/usr/local/bin/julia

using ArgParse
using CUDA
using DelimitedFiles
using LinearAlgebra
using ProgressMeter
using SimSpreadHelper

function main(args)
    # Argument parsing {{{
    configs = ArgParseSettings()

    add_arg_group!(configs, "I/O:")
    @add_arg_table! configs begin
        "--dt"
        help = "Drug-target adjacency matrix"
        required = true
        action = :store_arg
        arg_type = String
        "--dd"
        help = "Drug-drug similarity matrix"
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
    DT = read_namedmatrix(parsed_args["dt"], '\t')
    DD = read_namedmatrix(parsed_args["dd"])
    @show Nd = size(DT, 1)
    @show Nt = size(DT, 2)

    # Predict interactions
    pbar = Progress(Int64(Nd * (length(0:step:1))); desc="Fold 1/$(Nd) :: α = 0.0")
    Cgroups = split(DT, Nd)
    for α in 0:step:1.0
        for foldᵢ in eachindex(Cgroups)
            Cᵢ = Cgroups[foldᵢ]                                 # Get fold compounds
            DF = featurize(DD, α, weighted)                     # Featurize similarity matrix
            I = prepare!(DT, DF, Cᵢ)                            # Prepare graph used in SimSpread
            R = predict(I, DT; GPU=usegpu)                      # Predict drug-target interactions
            clean!(R, first(I), DT)                             # Remove errors caused by CV
            save(template * "_$(α).out", foldᵢ, Cᵢ, R, DT)      # Save to file
            next!(pbar; desc="Fold $(foldᵢ)/$(Nd) :: α = $(α)") # Advance counter
        end
    end
    ProgressMeter.finish!(pbar; desc="DONE!")
end

main(ARGS)
