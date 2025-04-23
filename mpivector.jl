# ./mpiexecjl -n 2 julia mpivector.jl

using MPI
using LinearAlgebra
using Test
using Krylov
import Krylov.FloatOrComplex

struct MPIVector{T, V<:AbstractVector{T}} <: AbstractVector{T}
    data::V
    comm::MPI.Comm
    global_len::Int
    data_range::UnitRange{Int}
end

function Base.similar(v::MPIVector)
    v2 = MPIVector(copy(v.data), v.comm, v.global_len, v.data_range)
    return v2
end

function Base.getindex(v::MPIVector, idx)
    if idx in v.data_range
    	return v.data[idx - data_range.start + 1]
    end
end

Base.length(v::MPIVector) = v.global_len
Base.eltype(v::MPIVector{T}) where T = T

function MPIVector(global_vec::AbstractVector{T}, comm::MPI.Comm = MPI.COMM_WORLD) where T
    size, rank = MPI.Comm_size(comm), MPI.Comm_rank(comm)
    n = length(global_vec)
    chunk = div(n + size - 1, size)
    istart = rank*chunk + 1
    iend   = min((rank+1)*chunk, n)
    data = global_vec[istart:iend]
    return MPIVector{T, typeof(data)}(data, comm, n, istart:iend)
end

function MPIVector(::Type{T}, global_len::Int, comm::MPI.Comm = MPI.COMM_WORLD) where T
    size, rank = MPI.Comm_size(comm), MPI.Comm_rank(comm)
    chunk = div(global_len + size - 1, size)
    istart = rank*chunk + 1
    iend = min((rank+1)*chunk, global_len)
    data = zeros(T, iend-istart+1)
    return MPIVector{T, typeof(data)}(data, comm, global_len, istart:iend)
end

function Krylov.kdot(n::Integer, x::MPIVector{T}, y::MPIVector{T}) where T <: FloatOrComplex
    data_dot = sum(x.data .* y.data)
    res = MPI.Allreduce(data_dot, +, x.comm)
    return res
end

function Krylov.knorm(n::Integer, x::MPIVector{T}) where T <: FloatOrComplex
	res = Krylov.kdot(n, x, x)
    return sqrt(res)
end

function Krylov.kscal!(n::Integer, s::T, x::MPIVector{T}) where T <: FloatOrComplex
	x.data .*= s
    return x
end

function Krylov.kdiv!(n::Integer, x::MPIVector{T}, s::T) where T <: FloatOrComplex
	x.data ./= s
    return x
end

function Krylov.kaxpy!(n::Integer, s::T, x::MPIVector{T}, y::MPIVector{T}) where T <: FloatOrComplex
	y.data .+= s .* x.data
    return y
end

function Krylov.kaxpby!(n::Integer, s::T, x::MPIVector{T}, t::T, y::MPIVector{T}) where T <: FloatOrComplex
	y.data .= s .* x.data .+ t .* y.data
    return y
end

function Krylov.kcopy!(n::Integer, y::MPIVector{T}, x::MPIVector{T}) where T <: FloatOrComplex
	y.data .= x.data
    return y
end

function Krylov.kscalcopy!(n::Integer, y::MPIVector{T}, s::T, x::MPIVector{T}) where T <: FloatOrComplex
	y.data .= x.data .* s
    return y
end

function Krylov.kdivcopy!(n::Integer, y::MPIVector{T}, x::MPIVector{T}, s::T) where T <: FloatOrComplex
	y.data .= x.data ./ s
    return y
end

function Krylov.kfill!(x::MPIVector{T}, val::T) where T <: FloatOrComplex
	x.data .= val
    return x
end

# function LinearAlgebra.dot(x::MPIVector{T}, y::MPIVector{T}) where T
#     data_dot = sum(x.data .* y.data)
#     return MPI.Allreduce(data_dot, +, x.comm)
# end

# function LinearAlgebra.norm(x::MPIVector{T}) where T
#     return sqrt(dot(x, x))
# end

# function LinearAlgebra.axpy!(y::MPIVector{T}, a::T, x::MPIVector{T}) where T
#     y.data .+= a .* x.data
#     return y
# end

struct MPIOperator{T}
    m::Int
    n::Int
end

# Define size and element type for the operator
Base.size(A::MPIOperator) = (A.m, A.n)
Base.eltype(A::MPIOperator{T}) where T = T

function LinearAlgebra.mul!(y::MPIVector{Float64}, A::MPIOperator{Float64}, u::MPIVector{Float64})
	y.data .= u.data_range .* u.data
    return y
end

MPI.Init()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
n = 10
global_v1 = collect(1:n) .|> Float64
global_v2 = collect(n+1:2*n) .|> Float64
mpi_v1 = MPIVector(global_v1)
mpi_v2 = MPIVector(global_v2)
Krylov.kdot(n, mpi_v1, mpi_v2) == dot(global_v1, global_v2)
@test Krylov.knorm(n, mpi_v1) == norm(global_v1)
@test Krylov.knorm(n, mpi_v2) == norm(global_v2)
output = Krylov.kaxpy!(n, 5.0, mpi_v1, mpi_v2)
global_v3 = global_v2 + 5.0 * global_v1
@test Krylov.knorm(n, mpi_v2) == norm(global_v3)
println("Rank: $rank")
for i = 0:MPI.Comm_size(comm)-1
	if i == rank
		println("Local data: ", mpi_v2.data)
		println("Global data: ", global_v3)
	end	
	MPI.Barrier(comm)
end

A_global = Float64.(Diagonal(1:n))
A_mpi = MPIOperator{Float64}(n, n)
b_global = Float64.(collect(n:-1:1))
x_global, _ = cg(A_global, b_global)
b_mpi = MPIVector(b_global)
kc = KrylovConstructor(b_mpi)
workspace = CgWorkspace(kc)
Krylov.cg!(workspace, A_mpi, b_mpi)
x_mpi = workspace.x

for i = 0:MPI.Comm_size(comm)-1
	if i == rank
		println("Local data: ", x_mpi.data)
		println("Global data: ", x_global)
	end	
	MPI.Barrier(comm)
end

# MPIVector(Float64, 10)


MPI.Finalize()
