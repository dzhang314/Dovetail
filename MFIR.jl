module MFIR

using Base.MPFR: libmpfr, MPFRRoundingMode, MPFRRoundNearest
using Core.Intrinsics: sqrt_llvm
using SIMD: Vec

######################################################################## MFIR OPERATIONS


export MFIROperation, MFIR_NEG,
    MFIR_ADD, MFIR_TWO_SUM, MFIR_FAST_TWO_SUM,
    MFIR_SUB, MFIR_TWO_DIFF, MFIR_FAST_TWO_DIFF,
    MFIR_SQR, MFIR_MUL, MFIR_FMA, MFIR_FMS, MFIR_FNMA, MFIR_FNMS,
    MFIR_TWO_SQR, MFIR_TWO_PROD,
    MFIR_INV, MFIR_DIV, MFIR_SQRT, MFIR_INV_SQRT,
    num_inputs, num_outputs


@enum MFIROperation::UInt16 begin
    MFIR_NEG
    MFIR_ADD
    MFIR_TWO_SUM
    MFIR_FAST_TWO_SUM
    MFIR_SUB
    MFIR_TWO_DIFF
    MFIR_FAST_TWO_DIFF
    MFIR_SQR
    MFIR_MUL
    MFIR_FMA
    MFIR_FMS
    MFIR_FNMA
    MFIR_FNMS
    MFIR_TWO_SQR
    MFIR_TWO_PROD
    MFIR_INV
    MFIR_DIV
    MFIR_SQRT
    MFIR_INV_SQRT
end


@inline function num_inputs(op::MFIROperation)
    if ((op == MFIR_NEG) | (op == MFIR_SQR) | (op == MFIR_TWO_SQR) |
        (op == MFIR_INV) | (op == MFIR_SQRT) | (op == MFIR_INV_SQRT))
        return 1
    elseif ((op == MFIR_FMA) | (op == MFIR_FMS) | (op == MFIR_FNMA) | (op == MFIR_FNMS))
        return 3
    else
        return 2
    end
end


@inline function num_outputs(op::MFIROperation)
    if ((op == MFIR_TWO_SUM) | (op == MFIR_TWO_DIFF) |
        (op == MFIR_FAST_TWO_SUM) | (op == MFIR_FAST_TWO_DIFF) |
        (op == MFIR_TWO_SQR) | (op == MFIR_TWO_PROD))
        return 2
    else
        return 1
    end
end


############################################################# INSTRUCTION DATA STRUCTURE


export MFIRInstruction, num_inputs, num_outputs, normalize


struct MFIRInstruction
    op::MFIROperation
    args::NTuple{3,UInt16}
end


const NULL_ARG = zero(UInt16)

@inline MFIRInstruction(op::MFIROperation, x::Integer) =
    MFIRInstruction(op, (x % UInt16, NULL_ARG, NULL_ARG))

@inline MFIRInstruction(op::MFIROperation, x::Integer, y::Integer) =
    MFIRInstruction(op, (x % UInt16, y % UInt16, NULL_ARG))

@inline MFIRInstruction(op::MFIROperation, x::Integer, y::Integer, z::Integer) =
    MFIRInstruction(op, (x % UInt16, y % UInt16, z % UInt16))


@inline function Base.isvalid(insn::MFIRInstruction, num_reg::Int)
    if signbit(num_reg) | !(MFIR_NEG <= insn.op <= MFIR_INV_SQRT)
        return false
    end
    n = num_inputs(insn.op)
    x, y, z = insn.args
    if n == 1
        return (1 <= x <= num_reg) & iszero(y) & iszero(z)
    elseif n == 2
        return (1 <= x <= num_reg) & (1 <= y <= num_reg) & iszero(z)
    elseif n == 3
        return (1 <= x <= num_reg) & (1 <= y <= num_reg) & (1 <= z <= num_reg)
    else
        return false
    end
end


@inline num_inputs(insn::MFIRInstruction) = num_inputs(insn.op)
@inline num_outputs(insn::MFIRInstruction) = num_outputs(insn.op)


@inline function normalize(insn::MFIRInstruction)
    op = insn.op
    x, y, z = insn.args
    if ((op == MFIR_ADD) | (op == MFIR_TWO_SUM) |
        (op == MFIR_MUL) | (op == MFIR_TWO_PROD) |
        (op == MFIR_FMA) | (op == MFIR_FMS) | (op == MFIR_FNMA) | (op == MFIR_FNMS))
        return MFIRInstruction(op, (minmax(x, y)..., z))
    else
        return insn
    end
end


################################################################## INSTRUCTION EXECUTION


export two_sum, fast_two_sum, two_diff, fast_two_diff, two_prod, unsafe_sqrt, inv_sqrt,
    execute!


@inline function two_sum(x::T, y::T) where {T}
    s = x + y
    x_prime = s - y
    y_prime = s - x_prime
    x_err = x - x_prime
    y_err = y - y_prime
    e = x_err + y_err
    return (s, e)
end


@inline function fast_two_sum(x::T, y::T) where {T}
    s = x + y
    y_prime = s - x
    e = y - y_prime
    return (s, e)
end


@inline function two_diff(x::T, y::T) where {T}
    d = x - y
    y_prime = x - d
    x_prime = d + y_prime
    x_err = x - x_prime
    y_err = y_prime - y
    e = x_err + y_err
    return (d, e)
end


@inline function fast_two_diff(x::T, y::T) where {T}
    d = x - y
    y_prime = x - d
    e = y_prime - y
    return (d, e)
end


@inline function two_prod(x::T, y::T) where {T}
    p = x * y
    e = fma(x, y, -p)
    return (p, e)
end


@inline unsafe_sqrt(x::Float16) = sqrt_llvm(x)
@inline unsafe_sqrt(x::Float32) = sqrt_llvm(x)
@inline unsafe_sqrt(x::Float64) = sqrt_llvm(x)

function unsafe_sqrt(x::BigFloat)
    result = BigFloat()
    ccall((:mpfr_sqrt, libmpfr),
        Cint, (Ref{BigFloat}, Ref{BigFloat}, MPFRRoundingMode),
        result, x, MPFRRoundNearest)
    return result
end

@inline unsafe_sqrt(x::Any) = sqrt(x)


@inline inv_sqrt(x::Float16) = Float16(unsafe_sqrt(inv(Float32(x))))
@inline inv_sqrt(x::Float32) = Float32(unsafe_sqrt(inv(Float64(x))))
@inline inv_sqrt(x::Vec{N,Float16}) where {N} =
    convert(Vec{N,Float16}, unsafe_sqrt(inv(convert(Vec{N,Float32}, x))))
@inline inv_sqrt(x::Vec{N,Float32}) where {N} =
    convert(Vec{N,Float32}, unsafe_sqrt(inv(convert(Vec{N,Float64}, x))))

# See "High-level algorithms for correctly-rounded reciprocal square roots",
# Table II and Section III.B.3. Freely available at: hal.science/hal-03728088
# Borges, Jeannerod, and Muller, ARITH 2022, DOI 10.1109/ARITH54963.2022.00013
@inline function inv_sqrt(x::Float64)
    r = inv(x)
    y = unsafe_sqrt(r)
    h = -0.5 * x
    t = fma(h, r, 0.5)
    s = fma(y, y, -r)
    c = fma(h, s, t)
    y = fma(y, c, y)
    # There are two exceptional values that Newton iteration fails to correct.
    key = reinterpret(UInt64, x) & 0x001FFFFFFFFFFFFF
    offset = UInt64((key == 0x000C562B857453DD) | (key == 0x000FFFFFFFFFFFFE))
    return reinterpret(Float64, reinterpret(UInt64, y) + offset)
end

@inline function inv_sqrt(x::Vec{N,Float64}) where {N}
    r = inv(x)
    y = unsafe_sqrt(r)
    h = -0.5 * x
    t = fma(h, r, 0.5)
    s = fma(y, y, -r)
    c = fma(h, s, t)
    y = fma(y, c, y)
    # There are two exceptional values that Newton iteration fails to correct.
    key = reinterpret(Vec{N,UInt64}, x) & 0x001FFFFFFFFFFFFF
    offset = convert(Vec{N,UInt64},
        (key == 0x000C562B857453DD) | (key == 0x000FFFFFFFFFFFFE))
    return reinterpret(Vec{N,Float64}, reinterpret(Vec{N,UInt64}, y) + offset)
end

function inv_sqrt(x::BigFloat)
    result = BigFloat()
    ccall((:mpfr_rec_sqrt, libmpfr),
        Cint, (Ref{BigFloat}, Ref{BigFloat}, MPFRRoundingMode),
        result, x, MPFRRoundNearest)
    return result
end

@inline inv_sqrt(x::Any) = sqrt(inv(x))


@inline function store_pair!(v::AbstractVector{T}, i::Int, (a, b)::Tuple{T,T}) where {T}
    @static if Sys.ARCH === :aarch64
        # This seemingly unnecessarily complicated formula works around an LLVM
        # bug causing suboptimal code generation on Apple M-series processors.
        j = ifelse(i > 0, i + 1, i)
    else
        j = i + 1
    end
    @inbounds v[i] = a
    @inbounds v[j] = b
    return v
end


@inline function execute!(v::AbstractVector, reg::Int, insn::MFIRInstruction)
    op = insn.op
    x, y, z = insn.args
    @inbounds if op == MFIR_NEG
        v[reg] = -v[x]
    elseif op == MFIR_ADD
        v[reg] = v[x] + v[y]
    elseif op == MFIR_TWO_SUM
        store_pair!(v, reg, two_sum(v[x], v[y]))
    elseif op == MFIR_FAST_TWO_SUM
        store_pair!(v, reg, fast_two_sum(v[x], v[y]))
    elseif op == MFIR_SUB
        v[reg] = v[x] - v[y]
    elseif op == MFIR_TWO_DIFF
        store_pair!(v, reg, two_diff(v[x], v[y]))
    elseif op == MFIR_FAST_TWO_DIFF
        store_pair!(v, reg, fast_two_diff(v[x], v[y]))
    elseif op == MFIR_SQR
        v[reg] = v[x] * v[x]
    elseif op == MFIR_MUL
        v[reg] = v[x] * v[y]
    elseif op == MFIR_FMA
        v[reg] = fma(v[x], v[y], v[z])
    elseif op == MFIR_FMS
        v[reg] = fma(v[x], v[y], -v[z])
    elseif op == MFIR_FNMA
        v[reg] = fma(-v[x], v[y], v[z])
    elseif op == MFIR_FNMS
        v[reg] = fma(-v[x], v[y], -v[z])
    elseif op == MFIR_TWO_SQR
        store_pair!(v, reg, two_prod(v[x], v[x]))
    elseif op == MFIR_TWO_PROD
        store_pair!(v, reg, two_prod(v[x], v[y]))
    elseif op == MFIR_INV
        v[reg] = inv(v[x])
    elseif op == MFIR_DIV
        v[reg] = v[x] / v[y]
    elseif op == MFIR_SQRT
        v[reg] = unsafe_sqrt(v[x])
    elseif op == MFIR_INV_SQRT
        v[reg] = inv_sqrt(v[x])
    else
        @assert false
    end
    return v
end


################################################################# PROGRAM DATA STRUCTURE


export MFIRProgram, num_registers, use_counts, normalize


struct MFIRProgram
    num_inputs::Int
    instructions::Vector{MFIRInstruction}
    result_ranges::Vector{UnitRange{UInt16}}
    output_indices::Vector{UInt16}
end


function Base.:(==)(P::MFIRProgram, Q::MFIRProgram)
    return (P === Q) || (
        (P.num_inputs == Q.num_inputs) &&
        (P.instructions == Q.instructions) &&
        (P.output_indices == Q.output_indices))
end


function Base.hash(P::MFIRProgram, h::UInt)
    h = hash(P.num_inputs, h)
    @inbounds for insn in P.instructions
        h = hash(reinterpret(UInt64, insn), h)
    end
    @inbounds for out_reg in P.output_indices
        h = hash(out_reg, h)
    end
    return h
end


function MFIRProgram(
    num_inputs::Integer,
    instructions::Vector{MFIRInstruction},
    output_indices::AbstractVector{<:Integer},
)
    reg_hi = UInt16(num_inputs) # UInt16 range check
    result_ranges = Vector{UnitRange{UInt16}}(undef, length(instructions))
    for (i, insn) in enumerate(instructions)
        @assert isvalid(insn, Int(reg_hi))
        reg_lo = reg_hi + one(UInt16)
        reg_hi = UInt16(reg_hi + num_outputs(insn)) # UInt16 range check
        @inbounds result_ranges[i] = reg_lo:reg_hi
    end
    num_reg = reg_hi
    @assert all(1 <= out_reg <= num_reg for out_reg in output_indices)
    return MFIRProgram(Int(num_inputs), instructions, result_ranges,
        convert(Vector{UInt16}, output_indices))
end


function Base.isvalid(P::MFIRProgram)
    if !(0 <= P.num_inputs <= typemax(UInt16))
        return false
    end
    if length(P.instructions) != length(P.result_ranges)
        return false
    end
    num_reg = P.num_inputs
    @inbounds for i in eachindex(P.instructions)
        insn = P.instructions[i]
        reg_range = P.result_ranges[i]
        if !isvalid(insn, num_reg)
            return false
        end
        if reg_range.start != num_reg + 1
            return false
        end
        num_reg += num_outputs(insn)
        if reg_range.stop != num_reg
            return false
        end
    end
    for out_reg in P.output_indices
        if !(1 <= out_reg <= num_reg)
            return false
        end
    end
    return true
end


@inline num_registers(P::MFIRProgram) =
    isempty(P.result_ranges) ? P.num_inputs : Int(P.result_ranges[end].stop)


@inline num_registers(P::MFIRProgram, i::Integer) =
    iszero(i) ? P.num_inputs : Int(P.result_ranges[i].stop)


function use_counts(P::MFIRProgram)
    result = zeros(Int, num_registers(P))
    @inbounds for insn in P.instructions
        for reg in insn.args
            if !iszero(reg)
                result[reg] += 1
            end
        end
    end
    @inbounds for out_reg in P.output_indices
        result[out_reg] += 1
    end
    return result
end


function normalize(P::MFIRProgram)
    reg_live = falses(num_registers(P))
    @inbounds for out_reg in P.output_indices
        reg_live[out_reg] = true
    end
    insn_live = falses(length(P.instructions))
    @inbounds for i = length(P.instructions):-1:1
        live = false
        for out_reg in P.result_ranges[i]
            live |= reg_live[out_reg]
        end
        if live
            insn_live[i] = true
            insn = P.instructions[i]
            for reg in insn.args
                if !iszero(reg)
                    reg_live[reg] = true
                end
            end
        end
    end
    reg_map = zeros(UInt16, length(reg_live))
    @inbounds for reg = 1:P.num_inputs
        reg_map[reg] = UInt16(reg)
    end
    instructions = Vector{MFIRInstruction}(undef, count(insn_live))
    next_reg = UInt16(P.num_inputs)
    j = 0
    @inbounds for i in eachindex(P.instructions)
        if insn_live[i]
            insn = P.instructions[i]
            x, y, z = insn.args
            args = (
                iszero(x) ? NULL_ARG : reg_map[x],
                iszero(y) ? NULL_ARG : reg_map[y],
                iszero(z) ? NULL_ARG : reg_map[z],
            )
            instructions[j+=1] = normalize(MFIRInstruction(insn.op, args))
            for old_reg in P.result_ranges[i]
                reg_map[old_reg] = (next_reg += one(UInt16))
            end
        end
    end
    return MFIRProgram(P.num_inputs, instructions,
        UInt16[reg_map[out_reg] for out_reg in P.output_indices])
end


###################################################################### PROGRAM EXECUTION


export execute!


@inline function execute!(v::Vector{T}, P::MFIRProgram, inputs::NTuple{N,T}) where {T,N}
    @assert length(v) >= num_registers(P)
    @assert P.num_inputs == N
    @simd ivdep for reg = 1:N
        @inbounds v[reg] = inputs[reg]
    end
    reg = N + 1
    @inbounds for insn in P.instructions
        execute!(v, reg, insn)
        reg += num_outputs(insn)
    end
    return v
end


####################################################################### PROGRAM ANALYSIS


export FLOPCounter, count_flops


struct FLOPCounter
    neg::Int
    add::Int
    mul::Int
    fma::Int
    rcp::Int
    div::Int
    sqrt::Int
    rsqrt::Int
end


Base.zero(::Type{FLOPCounter}) = FLOPCounter(0, 0, 0, 0, 0, 0, 0, 0)


Base.:+(A::FLOPCounter, B::FLOPCounter) = FLOPCounter(
    A.neg + B.neg,
    A.add + B.add,
    A.mul + B.mul,
    A.fma + B.fma,
    A.rcp + B.rcp,
    A.div + B.div,
    A.sqrt + B.sqrt,
    A.rsqrt + B.rsqrt,
)


@inline count_flops(op::MFIROperation) =
    op == MFIR_NEG ? FLOPCounter(1, 0, 0, 0, 0, 0, 0, 0) :
    op == MFIR_ADD ? FLOPCounter(0, 1, 0, 0, 0, 0, 0, 0) :
    op == MFIR_TWO_SUM ? FLOPCounter(0, 6, 0, 0, 0, 0, 0, 0) :
    op == MFIR_FAST_TWO_SUM ? FLOPCounter(0, 3, 0, 0, 0, 0, 0, 0) :
    op == MFIR_SUB ? FLOPCounter(0, 1, 0, 0, 0, 0, 0, 0) :
    op == MFIR_TWO_DIFF ? FLOPCounter(0, 6, 0, 0, 0, 0, 0, 0) :
    op == MFIR_FAST_TWO_DIFF ? FLOPCounter(0, 3, 0, 0, 0, 0, 0, 0) :
    op == MFIR_SQR ? FLOPCounter(0, 0, 1, 0, 0, 0, 0, 0) :
    op == MFIR_MUL ? FLOPCounter(0, 0, 1, 0, 0, 0, 0, 0) :
    op == MFIR_FMA ? FLOPCounter(0, 0, 0, 1, 0, 0, 0, 0) :
    op == MFIR_FMS ? FLOPCounter(0, 0, 0, 1, 0, 0, 0, 0) :
    op == MFIR_FNMA ? FLOPCounter(0, 0, 0, 1, 0, 0, 0, 0) :
    op == MFIR_FNMS ? FLOPCounter(0, 0, 0, 1, 0, 0, 0, 0) :
    op == MFIR_TWO_SQR ? FLOPCounter(0, 0, 1, 1, 0, 0, 0, 0) :
    op == MFIR_TWO_PROD ? FLOPCounter(0, 0, 1, 1, 0, 0, 0, 0) :
    op == MFIR_INV ? FLOPCounter(0, 0, 0, 0, 1, 0, 0, 0) :
    op == MFIR_DIV ? FLOPCounter(0, 0, 0, 0, 0, 1, 0, 0) :
    op == MFIR_SQRT ? FLOPCounter(0, 0, 0, 0, 0, 0, 1, 0) :
    op == MFIR_INV_SQRT ? FLOPCounter(0, 0, 0, 0, 0, 0, 0, 1) :
    @assert false


function count_flops(P::MFIRProgram)
    result = zero(FLOPCounter)
    @inbounds for insn in P.instructions
        result += count_flops(insn.op)
    end
    return result
end


####################################################################### PROGRAM MUTATION


export change_outputs, append_instruction, replace_instruction, remove_instruction


function change_outputs(P::MFIRProgram, output_indices::AbstractVector{<:Integer})
    num_reg = num_registers(P)
    @assert all(1 <= out_reg <= num_reg for out_reg in output_indices)
    return MFIRProgram(P.num_inputs, P.instructions, P.result_ranges,
        convert(Vector{UInt16}, output_indices))
end


function append_instruction(P::MFIRProgram, insn::MFIRInstruction)
    num_reg = num_registers(P)
    @assert isvalid(insn, num_reg)
    instructions = Vector{MFIRInstruction}(undef, length(P.instructions) + 1)
    copyto!(instructions, P.instructions)
    instructions[end] = insn
    result_ranges = Vector{UnitRange{UInt16}}(undef, length(P.result_ranges) + 1)
    copyto!(result_ranges, P.result_ranges)
    result_ranges[end] = UInt16(num_reg+1):UInt16(num_reg+num_outputs(insn))
    return MFIRProgram(P.num_inputs, instructions, result_ranges, P.output_indices)
end


function replace_instruction(P::MFIRProgram, i::Int, insn::MFIRInstruction)
    @assert num_outputs(insn) == num_outputs(P.instructions[i])
    @assert isvalid(insn, num_registers(P, i - 1))
    instructions = copy(P.instructions)
    @inbounds instructions[i] = insn
    return MFIRProgram(P.num_inputs, instructions, P.result_ranges, P.output_indices)
end


function remove_instruction(
    P::MFIRProgram,
    i::Int,
    replacements::NTuple{N,UInt16},
) where {N}
    @assert num_outputs(P.instructions[i]) == N
    @inbounds removed_base_reg = P.result_ranges[i].start
    @assert all(1 <= rep_reg < removed_base_reg for rep_reg in replacements)
    @inline remap(reg::UInt16) =
        reg < removed_base_reg ? reg :
        reg >= removed_base_reg + N ? reg - UInt16(N) :
        @inbounds(replacements[reg-(removed_base_reg-1)])
    num_insn = length(P.instructions)
    instructions = Vector{MFIRInstruction}(undef, num_insn - 1)
    copyto!(instructions, 1, P.instructions, 1, i - 1)
    result_ranges = Vector{UnitRange{UInt16}}(undef, num_insn - 1)
    copyto!(result_ranges, 1, P.result_ranges, 1, i - 1)
    @inbounds for j = (i+1):num_insn
        insn = P.instructions[j]
        instructions[j-1] = MFIRInstruction(insn.op, map(remap, insn.args))
        reg_range = P.result_ranges[j]
        result_ranges[j-1] = (reg_range.start-UInt16(N)):(reg_range.stop-UInt16(N))
    end
    return MFIRProgram(P.num_inputs, instructions, result_ranges,
        UInt16[remap(out_reg) for out_reg in P.output_indices])
end


########################################################################################

end # module MFIR
