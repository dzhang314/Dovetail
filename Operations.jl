module Operations

########################################################################################


export AbstractOperation, Add, Sub, Sqr, Mul, MulAdd, FMA, Inv, Div, Sqrt, InvSqrt


abstract type AbstractOperation{NR} end


struct Add{NX,NY,NR} <: AbstractOperation{NR} end
struct Sub{NX,NY,NR} <: AbstractOperation{NR} end
struct Sqr{NX,NR} <: AbstractOperation{NR} end
struct Mul{NX,NY,NR} <: AbstractOperation{NR} end
struct MulAdd{NX,NY,NZ,NR} <: AbstractOperation{NR} end
struct FMA{NX,NY,NZ,NR} <: AbstractOperation{NR} end
struct Inv{NX,NR} <: AbstractOperation{NR} end
struct Div{NX,NY,NR} <: AbstractOperation{NR} end
struct Sqrt{NX,NR} <: AbstractOperation{NR} end
struct InvSqrt{NX,NR} <: AbstractOperation{NR} end


########################################################################################


export input_widths, output_width, in_domain, evaluate_operation


@inline input_widths(::Type{Add{NX,NY,NR}}) where {NX,NY,NR} = (NX, NY)
@inline input_widths(::Type{Sub{NX,NY,NR}}) where {NX,NY,NR} = (NX, NY)
@inline input_widths(::Type{Sqr{NX,NR}}) where {NX,NR} = (NX,)
@inline input_widths(::Type{Mul{NX,NY,NR}}) where {NX,NY,NR} = (NX, NY)
@inline input_widths(::Type{MulAdd{NX,NY,NZ,NR}}) where {NX,NY,NZ,NR} = (NX, NY, NZ)
@inline input_widths(::Type{FMA{NX,NY,NZ,NR}}) where {NX,NY,NZ,NR} = (NX, NY, NZ)
@inline input_widths(::Type{Inv{NX,NR}}) where {NX,NR} = (NX,)
@inline input_widths(::Type{Div{NX,NY,NR}}) where {NX,NY,NR} = (NX, NY)
@inline input_widths(::Type{Sqrt{NX,NR}}) where {NX,NR} = (NX,)
@inline input_widths(::Type{InvSqrt{NX,NR}}) where {NX,NR} = (NX,)


@inline output_width(::Type{O}) where {NR,O<:AbstractOperation{NR}} = NR


@inline in_domain(::Type{<:Add}, ::NTuple{2}) = true
@inline in_domain(::Type{<:Sub}, ::NTuple{2}) = true
@inline in_domain(::Type{<:Sqr}, ::NTuple{1}) = true
@inline in_domain(::Type{<:Mul}, ::NTuple{2}) = true
@inline in_domain(::Type{<:MulAdd}, (x, y, z)::NTuple{3}) =
    iszero(x) | iszero(y) | iszero(z) | (xor(signbit(x), signbit(y)) == signbit(z))
@inline in_domain(::Type{<:FMA}, ::NTuple{3}) = true
@inline in_domain(::Type{<:Inv}, (x,)::NTuple{1}) = !iszero(x)
@inline in_domain(::Type{<:Div}, (_, y)::NTuple{2}) = !iszero(y)
@inline in_domain(::Type{<:Sqrt}, (x,)::NTuple{1}) = x > zero(x)
@inline in_domain(::Type{<:InvSqrt}, (x,)::NTuple{1}) = x > zero(x)


@inline evaluate_operation(::Type{<:Add}, (x, y)::NTuple{2}) = x + y
@inline evaluate_operation(::Type{<:Sub}, (x, y)::NTuple{2}) = x - y
@inline evaluate_operation(::Type{<:Sqr}, (x,)::NTuple{1}) = abs2(x)
@inline evaluate_operation(::Type{<:Mul}, (x, y)::NTuple{2}) = x * y
@inline evaluate_operation(::Type{<:MulAdd}, (x, y, z)::NTuple{3}) = x * y + z
@inline evaluate_operation(::Type{<:FMA}, (x, y, z)::NTuple{3}) = fma(x, y, z)
@inline evaluate_operation(::Type{<:Inv}, (x,)::NTuple{1}) = inv(x)
@inline evaluate_operation(::Type{<:Div}, (x, y)::NTuple{2}) = x / y
@inline evaluate_operation(::Type{<:Sqrt}, (x,)::NTuple{1}) = sqrt(x)
@inline evaluate_operation(::Type{<:InvSqrt}, (x,)::NTuple{1}) = inv(sqrt(x))


########################################################################################

end # module Operations
