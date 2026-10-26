from __future__ import annotations

import cutlass
import cutlass.cute as cute


@cute.jit
def per_key_bias_score_mod(
    score,
    b_idx,
    h_idx,
    q_idx,
    kv_idx,
    seqlen_info,
    aux_tensors,
):
    key_bias = aux_tensors[0]
    index_fragment = cute.make_rmem_tensor(1, cutlass.Int32)
    index_fragment.store(kv_idx)
    bias_fragment = cute.make_rmem_tensor(1, key_bias.element_type)
    bias_fragment[0] = key_bias[index_fragment[0]]
    return score + bias_fragment.load().to(cutlass.Float32)


@cute.jit
def per_key_bias_score_mod_backward(
    gradient,
    score,
    b_idx,
    h_idx,
    q_idx,
    kv_idx,
    seqlen_info,
    aux_tensors,
):

    return gradient


@cute.jit
def action_group_bias_score_mod(
    score,
    b_idx,
    h_idx,
    q_idx,
    kv_idx,
    seqlen_info,
    aux_tensors,
):

    action_bias = aux_tensors[0]
    index_fragment = cute.make_rmem_tensor(1, cutlass.Int32)
    index_fragment[0] = 0
    bias_fragment = cute.make_rmem_tensor(1, action_bias.element_type)
    bias_fragment[0] = action_bias[index_fragment[0]]
    bias = bias_fragment.load().to(cutlass.Float32)
    return score + cute.where(
        kv_idx >= seqlen_info.seqlen_q,
        bias,
        bias * cutlass.Float32(0.0),
    )


@cute.jit
def action_group_bias_score_mod_backward(
    gradient,
    score,
    b_idx,
    h_idx,
    q_idx,
    kv_idx,
    seqlen_info,
    aux_tensors,
):
    return gradient
