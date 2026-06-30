from pathlib import Path


path = Path("/workspace/policy_pred/models/nanochat_vendor/nanochat/flash_attention.py")
text = path.read_text()

text = text.replace(
    "    window = window_size[0]\n\n    # Full context, same length",
    "    window = window_size[0]\n\n"
    "    if enable_gqa:\n"
    "        repeat = q.size(-3) // k.size(-3)\n"
    "        k = k.repeat_interleave(repeat, dim=-3)\n"
    "        v = v.repeat_interleave(repeat, dim=-3)\n\n"
    "    # Full context, same length",
)
text = text.replace(
    "F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=enable_gqa)",
    "F.scaled_dot_product_attention(q, k, v, is_causal=True)",
)
text = text.replace(
    "F.scaled_dot_product_attention(q, k, v, is_causal=False, enable_gqa=enable_gqa)",
    "F.scaled_dot_product_attention(q, k, v, is_causal=False)",
)
text = text.replace(
    "    \n"
    "    if enable_gqa:\n"
    "        repeat = q.size(-3) // k.size(-3)\n"
    "        k = k.repeat_interleave(repeat, dim=-3)\n"
    "        v = v.repeat_interleave(repeat, dim=-3)\n"
    "    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)",
    "    \n"
    "    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)",
)

path.write_text(text)
print(f"patched {path}")
