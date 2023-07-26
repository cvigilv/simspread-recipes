#!/usr/local/bin/julia

using ArgParse
using DelimitedFiles
using LinearAlgebra
using NamedArrays
using Random
using SimSpread
using ProgressMeter
using CUDA


"""
    split(G::AbstractMatrix, ngroups::Int64; seed::Int64 = 1)

Split all possible `DT` (given by Nd × Nt) into `k` groups for cross-validation.

# Arguments
- `DT::AbstractMatrix`: Drug-Target rectangular adjacency matrix.
- `k::Int64`: Number of groups to use in data splitting.
- `rng::Int64`: Seed used for data splitting.

"""
function split(G::AbstractMatrix, ngroups::Int64; seed::Int64=1)
    E = Tuple.(findall(!iszero, G))

    # Assign fold to edges of graph
    shuffle!(MersenneTwister(seed), E)
    groups = [[] for _ in 1:ngroups]

    for (i, eᵢ) in enumerate(E)
        foldᵢ = mod(i, ngroups) + 1
        push!(groups[foldᵢ], eᵢ)
    end

    return groups
end

"""
    prepare(DD::AbstractMatrix, DT::AbstractMatrix, Eᵢ::AbstractVector)

Prepare matrices to use in cross-validation scheme. Here we eliminate all the edges from the
test set (denoted with `E`) and delete self-loops for the drugs considered in the matrices.

# Arguments
- `DD::AbstractMatrix`: Drug-Drug similarity matrix.
- `DT::AbstractMatrix`: Drug-Target rectangular adjacency matrix.
- `E::AbstractVector`: Test set edges to delete from `DT`.
"""
function prepare(DD::AbstractMatrix, DT::AbstractMatrix, Eᵢ::AbstractVector)
    # Remove self-loops in DD
    DD′ = deepcopy(DD)
    DD′[diagind(DD′)] .= 0

    # Remove all drug-target interactions for given compounds (C)
    DT₁ = deepcopy(DT)
    DT′ = similar(DT)
    DT′ .= 0
    for eᵢ in Eᵢ
        s, t = eᵢ
        DT₁[s, t] = 0
        DT′[s, t] = DT[s, t]
    end

    return DD′, DT′, DT₁
end

"""
    predict(DD::AbstractMatrix, DT::AbstractMatrix)

Predict all possible drug-target interactions using 1-nearest-neighbour algorithm, using as
distance matrix the `DD` similarity matrix.

# Arguments
- `DD::AbstractMatrix`: Drug-Drug similarity matrix.
- `DT::AbstractMatrix`: Drug-Target rectangular adjacency matrix.

# Implementation

Here we use the similarity matrix instead of a distance matrix in the nearest-neighbour
algorithm, therefore in order to eliminate overhead produces by element-wise conversion of
a similarity metric to a distance metric and the lack of a standarized method for given
conversion, we search for the most similar neighbour, in other words, we search for the
farthest neighbour in the `DD` matrix.
"""
function predict(DD::AbstractMatrix, DT::AbstractMatrix)
    # Initialize matrix for storing results
    Nd, Nt = size(DT)
    R = zeros(Nd, Nt)

    # Calculate Nearest-Neighbour per drug
    for C in 1:Nd
        F₁ = CuArray(DD[:, C]) .* CuArray(DT)
        R[C, :] = mapslices(maximum, Matrix{Float64}(F₁); dims=1)
    end

    return R
end

"""
    clean(R::AbstractMatrix, DT::AbstractMatrix)

Flag all drug-target interactions predictions that weren't able to be predicted by the
method because of limitations in the data splitting procedure. The flag is hardcoded to be 
the value `-99`.

# Arguments
- `R::AbstractMatrix`: Predicted drug-target rectangular adjacency matrix.
- `DT::AbstractMatrix`: Drug-Target rectangular adjacency matrix.

# Implementation

Drug-Target interaction dataset are frequently sparse, therefore there is a chance that we
will encounter targets that have only one annotated interaction with drugs. This targets
will be splitted incorrectly in the data splitting procedure, producing predictions with a 
score of 0 (zero) when in reality they can't be predicted. This cases are flagged in order
to exclude them from evaluation.
"""
function clean(R::AbstractArray, DT::AbstractArray)
    R′ = deepcopy(R)

    # Get degrees for nodes in bipartite graph A
    Kₜ = k(DT')

    # Flag predictions for all targets with degree == 0
    # NOTE: This are the cases that are imposible to predict due to data splitting
    # limitations, therefore we need to ignore them.
    for (tᵢ, k) in enumerate(Kₜ)
        if k == 0
            # @warn "Target #$(tᵢ) becomes disconnected in data splitting, flagging predictions."
            R′[:, tᵢ] .= -99
        end
    end

    return R′
end

"""
    save(R::AbstractMatrix, DT::AbstractMatrix, E::AbstractVector, fold::Int64, fout::String)

Save the drug-target interactions predictions as a CSV (comma separated value) file.

# Arguments
- `R::AbstractMatrix`: Predicted drug-target rectangular adjacency matrix.
- `DT::AbstractMatrix`: Drug-Target rectangular adjacency matrix.
- `E::AbstractVector`: Test set edges deleted from `DT`.
- `fold::Int64`: Group number.
- `fout::String`: File path of output CSV file.
"""
function save(R::AbstractMatrix, DT::AbstractMatrix, E::AbstractArray, foldᵢ::Int64, fout::String)
    L = unique([nₛ for (nₛ, nₜ) in E])
    T = 1:size(DT, 2)

    open(fout, "a+") do f
        for nₛ in L
            for nₜ in T
                formatted_output = "$foldᵢ, $nₛ, $nₜ, $(R[nₛ, nₜ]), $(DT[nₛ, nₜ])"
                write(f, formatted_output * "\n")
            end
        end
    end
end

function main(args)
    configs = ArgParseSettings()

    add_arg_group!(configs, "I/O options:")
    @add_arg_table! configs begin
        "--dt"
        arg_type = String
        action = :store_arg
        help = "Drug-Target adjacency matrix"
        required = true
        "--dd"
        arg_type = String
        action = :store_arg
        help = "Drug-Drug similarity matrix"
        required = true
        "--k-folds"
        arg_type = Int64
        action = :store_arg
        help = "Number of folds used in data splitting"
        required = false
        default = 10
        "--seed"
        arg_type = Int64
        action = :store_arg
        help = "Seed used for data splitting"
        required = false
        default = 1
        "--k-iterations"
        arg_type = Int64
        action = :store_arg
        help = "Number of iterations"
        required = false
        default = 5
        "-o"
        arg_type = String
        action = :store_arg
        help = "File path for predictions"
        required = true
    end

    parsed_args = parse_args(args, configs)

    # Store arguments to variables
    kfolds = parsed_args["k-folds"]
    seed = parsed_args["seed"]
    iterations = parsed_args["k-iterations"]
    fout = parsed_args["o"]

    # Load matrices to memory
    DT = read_namedmatrix(parsed_args["dt"], '\t').array
    DD = read_namedmatrix(parsed_args["dd"], ' ').array

    # Predict drug-target interactions
    for iter in 1:iterations
        Eₜ = split(DT, kfolds; seed=seed + iter)
        @showprogress for foldᵢ in 1:length(Eₜ)
            Eᵢ = Eₜ[foldᵢ]
            DD′, DT′, DT₁ = prepare(DD, DT, Eᵢ)
            R = predict(DD′, DT₁)
            R′ = clean(R, DT₁)
            save(R′, DT′, Eᵢ, foldᵢ, replace(fout, "NN" => "N$(iter)"))
        end
    end
end

main(ARGS)
