using DataFrames
using CSV
using SparseArrays
using LinearAlgebra
using Graphs
using CUDA
using GenericTensorNetworks
using TropicalGEMM

function to_qubo3(J::AbstractMatrix{T}, h::AbstractVector{T}) where {T}
  b = dropdims(sum(J, dims=2), dims=2)
  Q = J + Diagonal(T(0.5) .* (h .- b))
  Q .*= T(4)
  Q
end
function to_ising(Q::AbstractMatrix{T}) where {T}
  h = vec(sum(Q, dims=2)) .+ diag(Q)
  J = Q ./ T(4)
  J[diagind(J)] .= zero(T)
  J, h ./ T(4)
end

path_to_instances = "$(@__DIR__)/../../beit/data/instances/ocean/"

instances = readdir(path_to_instances)
sort!(instances, by=x -> parse(Int64, split(x, "_")[3]))
println(instances)

chimera_size = [parse(Int64, split(x, "_")[3]) for x in instances]
optimizers = [TreeSA()]# , GreedyMethod(), KaHyParBipartite(; sc_target=25), SABipartite()]
opt_names = ["TreeSA"] # "Greedy", "KaHyParBipartite", "SABipartite"]

filename_best = "TTN_chimera_CPU_TGEMM_energy_only.csv"
for i in eachindex(instances)
  instance = instances[i]
  # read line by line
  println("Loading ", instance)
  lines = readlines(path_to_instances * instance)
  I = Int64[]
  J = Int64[]
  V = Float64[]
  for line in lines
    line = split(line, " ")
    push!(I, parse(Int64, line[1]))
    push!(J, parse(Int64, line[2]))
    push!(V, parse(Float64, line[3]))
  end
  Q = sparse(I .+ 1, J .+ 1, V)
  Q .= (Q + Q')
  Q[diagind(Q)] ./= 2.0
  Q = Matrix{Float32}(Q)
  J, h = to_ising(Q)
  G = SimpleGraph(J)
  JG = [J[e.src, e.dst] for e in edges(G)]
  spinglass = SpinGlass(G, JG, h)
  for j in eachindex(optimizers)
    problem = GenericTensorNetwork(spinglass; optimizer=optimizers[j])
    
    # Energy computation
    tts = @elapsed res = solve(problem, SizeMin(), usecuda=false)
    Emin = res[1]
    println("Emin from TTN CPU TGEMM and $(string(opt_names[j])): ", Emin, " in $(tts) s")
    datapoint = (; instance=instance, num_var=size(J, 1), optimizer=opt_names[j], TTN_energy=Emin, runtime=tts)
    row_df = DataFrame(pairs(datapoint))
    CSV.write(filename_best, row_df; append=true, writeheader=!isfile(filename_best))

    # State computation
    # tts = @elapsed res = solve(problem, SingleConfigMin(), usecuda=true)
    # CUDA.@allowscalar state = read_config(res[])
    # Emin = energy(problem.problem, 1 .- 2 .* Int.(state))
    # println("Emin from TTN GPU and $(string(opt_names[j])): ", Emin, " in $(tts) s")
    # datapoint = (; instance=instance, num_var=size(J, 1), optimizer=opt_names[j], TTN_energy=Emin, runtime=tts, state=string(state))
    # row_df = DataFrame(pairs(datapoint))
    # CSV.write(filename_best, row_df; append=true, writeheader=!isfile(filename_best))
    # CUDA.reclaim()
  end
end
