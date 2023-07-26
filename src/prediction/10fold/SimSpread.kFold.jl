#!/usr/local/bin/julia

using ArgParse
using CUDA
using DelimitedFiles
using LinearAlgebra
using ProgressMeter
using SimSpread

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
        default = 10
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
    # }}}

    # Load matrices to memory
    y = read_namedmatrix(parsed_args["dt"], '\t')
    X = read_namedmatrix(parsed_args["dd"])

    pbar = Progress(
        Int64(iterations * nsplits * (length(0:step:1)));
        desc="I 1/$(iterations); F 1/$(nsplits); α = 0.00"
    )
    for iter in 1:1:iterations
        splits = split(y, nsplits; seed=1 + iter)
        for (fold_idx, test_idx) in enumerate(splits)
            for α in 0:step:1.0
                # Split dataset in training and testing sets
                train_idx = [s for s in names(y, 1) if s ∉ test_idx]

                ytrain = y[train_idx, :]
                Xtrain = X[train_idx, train_idx]
                ytest = y[test_idx, :]
                Xtest = X[test_idx, train_idx]

                # Featurization of similarity matrices
                Xtrain′, Xtest′ = map(D -> featurize(D, α, weighted), [Xtrain, Xtest])

                # Construct feature-source-target graph for predictions
                G = construct(ytrain, ytest, Xtrain′, Xtest′)

                # Predict drug-target interactions of testing set
                yhat = predict(G, ytest)
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
