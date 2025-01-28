using Random, LinearAlgebra
using HDF5

function write_to_h5(
  name::AbstractString,
  couplings,
  biases;
  metadata::Dict=Dict(),
  deflate::Int=0,
  shuffle::Bool=true,
) where {T}
  @assert size(biases, 1) == size(couplings, 1) == size(couplings, 2)
  g = create_group(parent, name)
  attributes(g)["sparsity"] = "dense"
  attributes(g)["type"] = "IsingModel"
  g["biases", shuffle=shuffle, deflate=deflate] = model.biases
  L = size(model.biases, 1)
  write(g, "L", L)
  write(g, "num_batches", size(model.couplings, 3))
  chunk_size = min(L, 100)
  write(
    g,
    "couplings",
    couplings;
    deflate=deflate,
    shuffle=shuffle,
    chunk=(chunk_size, chunk_size),
  )
  if !isempty(metadata)
    write(g, "metadata", JSON.json(metadata))
  end
end

struct RandomWeightedMax3SAT{T<:AbstractFloat}
    type::Type{T}
    size::Int
    rng::AbstractRNG

    function RandomWeightedMax3SAT{T}(size::Int, rng::AbstractRNG) where {T<:AbstractFloat}
        return new{T}(T, size, rng)
    end

    function RandomWeightedMax3SAT{T}(size::Int, seed::Int) where {T<:AbstractFloat}
        return new{T}(T, size, initialize_rng(seed))
    end

    function RandomWeightedMax3SAT{T}(size::Int) where {T<:AbstractFloat}
        return new{T}(T, size, initialize_rng())
    end
end


function generate(g::RandomWeightedMax3SAT{T}) where {T}
    # random weights
    # the problem is to maximize the sum of the weights of the satisfied clauses
    # so we build instance with -ω to turn it into a minimization problem
    ω = -rand(g.rng, T, g.size - 2)
    # random clause structure
    c = rand([0, 1], g.size)

    # each third order monomial gets a new auxiliary spin -> L - 2 auxiliary spins for OBC
    n_spins = g.size + (g.size - 2)
    couplings = zeros(T, n_spins, n_spins)
    biases = zeros(T, n_spins)
    reduction_offset = 0.0
    constant_energy_term = 0.0

    # there are no just 1 or 2 body terms
    for i = 1:(g.size-2)
        i1, i2, i3 = i, i + 1, i + 2
        aux_i = g.size + i

        constant_energy_term += ω[i] / 8.0
        reduction_offset += 3.0 * abs(ω[i]) / 8.0

        biases[i1] += ω[i] / 8.0 * (-1.0)^c[i1] * (1 + (-1)^(c[i2] + c[i3]))
        biases[i2] += ω[i] / 8.0 * (-1.0)^c[i2] * (1 + (-1)^(c[i1] + c[i3]))
        biases[i3] += ω[i] / 8.0 * (-1.0)^c[i3] * (1 + (-1)^(c[i1] + c[i2]))
        biases[aux_i] += ω[i] / 4.0 * (-1.0)^(c[i1] + c[i2] + c[i3])

        couplings[i1, i2] += ω[i] / 8.0 * (-1.0)^(c[i1] + c[i2]) + abs(ω[i]) / 8.0
        couplings[i1, i3] += ω[i] / 8.0 * (-1.0)^(c[i1] + c[i3]) + abs(ω[i]) / 8.0
        couplings[i2, i3] += ω[i] / 8.0 * (-1.0)^(c[i2] + c[i3]) + abs(ω[i]) / 8.0
        couplings[i1, aux_i] += abs(ω[i]) / 4.0
        couplings[i2, aux_i] += abs(ω[i]) / 4.0
        couplings[i3, aux_i] += abs(ω[i]) / 4.0
    end
    couplings .+= couplings'

    if length(biases) <= 1000
        metadata = Dict(
            "reduction_offset" => reduction_offset,
            "constant_energy_term" => constant_energy_term,
            "omega" => -ω, # saves the original weights from maximization problem
            "clauses" => c,
        )
    else
        metadata = Dict(
            "reduction_offset" => reduction_offset,
            "constant_energy_term" => constant_energy_term,
        )
    end
    return couplings, biases, metadata
end

initialize_rng(seed::Int) = Random.default_rng(seed)
initialize_rng() = Random.default_rng()

to_gen = [(10, 10), (100, 10), (156, 10), (433, 10), (1000, 10)]
path = "$(@__DIR__)/instances"
if !ispath(path)
  mkpath(path)
end

for (s, rep) ∈ to_gen
    gen = RandomWeightedMax3SAT{Float64}(s)
    for i = 1:rep
        println("Running s=$s, rep=$i")
        couplings, biases, metadata = generate(gen)
        filename = "randomWeightedMax3SAT_dense_L=$(s)_$(i).h5"
        write_to_h5(filename, couplings, biases; metadata)
    end
end
