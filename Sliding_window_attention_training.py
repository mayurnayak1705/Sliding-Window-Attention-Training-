import torch
import torch.nn as nn
import torch.nn.functional as f

class BalancedALiBi(nn.Module):
    def __init__(self, num_heads):
        super().__init__()

        self.num_heads = num_heads
        slopes = self._get_slopes(num_heads)
        half = num_heads // 2
        signs = torch.ones(num_heads)
        signs[half:] = -1
        slopes = slopes * signs
        self.register_buffer("slopes",slopes.view(num_heads, 1, 1),persistent=False)

    def _get_slopes(self, n_heads):
        return torch.tensor([2 ** (-8 * i / n_heads) for i in range(n_heads)],dtype=torch.float32)

    def forward(self, seq_len):
        pos = torch.arange(seq_len)
        distance = (pos[None, :] - pos[:, None])
        bias = self.slopes * distance
        return bias
    


class RotaryPositionalEmbedding(nn.Module):
  def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
      super().__init__()
      self.theta = theta
      self.d_k = d_k
      self.max_seq_len = max_seq_len

      d = d_k // 2
      dens = torch.arange(0, d, dtype=torch.float32, device=device)
      inv_freq = 1.0 / (theta ** (2 * dens / d_k))  
      pos = torch.arange(max_seq_len, dtype=torch.float32, device=device)  
      angles = pos[:, None] * inv_freq[None, :]     
      
      self.register_buffer('cos', torch.cos(angles), persistent=False)  
      self.register_buffer('sin', torch.sin(angles), persistent=False) 

  def forward(self, x, token_positions=None):
      original_shape = x.shape
      *prefix, seq_len, d_k = x.shape
      cos = self.cos[:seq_len, :]  
      sin = self.sin[:seq_len, :]  
      for _ in prefix:
          cos = cos.unsqueeze(0)
          sin = sin.unsqueeze(0)
      cos = cos.expand(*prefix, seq_len, d_k // 2)
      sin = sin.expand(*prefix, seq_len, d_k // 2)
      x_even = x[..., 0::2]
      x_odd = x[..., 1::2]
      x_rotated_even = x_even * cos - x_odd * sin
      x_rotated_odd = x_even * sin + x_odd * cos
      x_out = torch.stack([x_rotated_even, x_rotated_odd], dim=-1).flatten(-2)

      return x_out



class MHA(nn.Module):
    def __init__(self,d_in,d_q,d_v,d_heads,window_size,max_seq_len=4096,rope_theta=10000.0):
        super().__init__()

        self.d_in = d_in
        self.d_q = d_q
        self.d_v = d_v
        self.heads = d_heads
        self.window_size = window_size

        self.w_q = nn.Parameter(torch.rand(d_in, d_q))
        self.w_k = nn.Parameter(torch.rand(d_in, d_q))
        self.w_v = nn.Parameter(torch.rand(d_in, d_v))
        self.W_o = nn.Parameter(torch.rand(d_v, d_in))

        # RoPE
        self.rope = RotaryPositionalEmbedding(theta=rope_theta,d_k=d_q // d_heads,max_seq_len=max_seq_len)

        # Balanced ALiBi
        self.alibi = BalancedALiBi(d_heads)

    def forward(self, x):

        B, T, _ = x.shape

        q = x @ self.w_q
        k = x @ self.w_k
        v = x @ self.w_v

        d_head_q = self.d_q // self.heads
        d_head_v = self.d_v // self.heads

        Q = q.view(B, T, self.heads, d_head_q).transpose(1, 2)
        K = k.view(B, T, self.heads, d_head_q).transpose(1, 2)
        V = v.view(B, T, self.heads, d_head_v).transpose(1, 2)


        # Apply RoPE
        Q = self.rope(Q)
        K = self.rope(K)

        scores = Q @ K.transpose(-2, -1)
        scores = scores / (d_head_q ** 0.5)


        # Add Balanced ALiBi
        alibi_bias = self.alibi(seq_len=T)

        scores = scores + alibi_bias.unsqueeze(0)

        # Sliding Window + Causal Mask
        idx = torch.arange(T)

        mask = ((idx[None, :] <= idx[:, None]) & (idx[None, :] >= idx[:, None] - (self.window_size - 1)))

        scores = scores.masked_fill(~mask,float("-inf"))

        #Sigmoid instead of SF
        attn = torch.sigmoid(scores)
        attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-8)

        values = attn @ V
        values = (values.transpose(1, 2).contiguous().view(B, T, self.d_v))
        out = values @ self.W_o

        return out