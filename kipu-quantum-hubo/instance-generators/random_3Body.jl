using Random, LinearAlgebra
using SparseArrays
using ProgressMeter
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

struct Random3BodyLinearGenerator{T<:AbstractFloat}
  type::Type{T}
  size::Int
  rng::AbstractRNG

  function Random3BodyLinearGenerator{T}(
    size::Int,
    rng::AbstractRNG,
  ) where {T<:AbstractFloat}
    return new{T}(T, size, rng)
  end

  function Random3BodyLinearGenerator{T}(size::Int, seed::Int) where {T<:AbstractFloat}
    return new{T}(T, size, initialize_rng(seed))
  end

  function Random3BodyLinearGenerator{T}(size::Int) where {T<:AbstractFloat}
    return new{T}(T, size, initialize_rng())
  end
end

function generate(g::Random3BodyLinearGenerator{T}) where {T}
  h = randn(g.rng, T, g.size)
  J = randn(g.rng, T, g.size - 1)
  K = randn(g.rng, T, g.size - 2)
  sgn_K = sign.(K)

  println("Number of threads: $(Threads.nthreads())")
  # each third order monomial gets a new auxiliary spin -> L - 2 auxiliary spins for OBC
  n_spins = g.size + (g.size - 2)
  reduction_offset = [0.0 for _ = 1:Threads.nthreads()]

  couplings = [spzeros(T, n_spins, n_spins) for _ = 1:Threads.nthreads()]
  biases = [zeros(T, n_spins) for _ = 1:Threads.nthreads()]

  @showprogress Threads.@threads :static for i = 1:(g.size-2)
    i1, i2, i3 = i, i + 1, i + 2
    aux_i = g.size + i
    threadid = Threads.threadid()

    biases[threadid][i1] += h[i1] + sgn_K[i1] * abs(K[i1])
    biases[threadid][i2] += sgn_K[i1] * abs(K[i1])
    biases[threadid][i3] += sgn_K[i1] * abs(K[i1])
    biases[threadid][aux_i] += 2 * sgn_K[i1] * abs(K[i1])

    couplings[threadid][i1, i2] += J[i1] + abs(K[i1])
    couplings[threadid][i1, i3] += abs(K[i1])
    couplings[threadid][i2, i3] += abs(K[i1])
    couplings[threadid][i1, aux_i] += 2 * abs(K[i1])
    couplings[threadid][i2, aux_i] += 2 * abs(K[i1])
    couplings[threadid][i3, aux_i] += 2 * abs(K[i1])

    reduction_offset[threadid] += 3 * abs(K[i1])
  end

  reduction_offset = sum(reduction_offset)
  biases = sum(biases)
  couplings = sum(couplings)
  biases[g.size-1] += h[g.size-1]
  biases[g.size] += h[g.size]
  couplings[g.size-1, g.size] += J[g.size-1]
  couplings .+= couplings'

  if length(biases) <= 1000
    metadata = Dict(
      "reduction_offset" => reduction_offset,
      "h" => h,
      "J" => J,
      "K" => K,
    )
  else
    metadata = Dict("reduction_offset" => reduction_offset)
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
  gen = Random3BodyLinearGenerator{Float64}(s)
  for i = 1:rep
    println("Running s=$s, rep=$i")
    tm = @elapsed couplings, biases, metadata = generate(gen)
    println("Finished in $(tm) seconds")
    filename = "random3BodyIsing_dense_L=$(s)_$(i).h5"
    write_to_h5(filename, couplings, biases; metadata)
  end
end
