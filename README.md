
```python
    
    def taylor_softmax(self, x, order=2):
        taylor_approx = 1.0
        for i in range(1, order + 1):
            factorial_i = torch.exp(torch.lgamma(torch.tensor(i + 1, dtype=torch.float32)))
            taylor_approx += x**i / factorial_i
        return taylor_approx / torch.sum(taylor_approx, dim=-1, keepdim=True)

    def taylor_softmax_2nd_order(x):
        exp_approx = 1 + x + (x**2) / 2
        return exp_approx / torch.sum(exp_approx, dim=-1, keepdim=True)


### replace softmax

    def forward(self, x, xa = None, mask = None):

        q = self.que(self.ln(x))
        k, v = self.kv(self.ln(x if xa is None else xa)).chunk(2, dim=-1)

        q, k, v = map(lambda t: rearrange(t, 'b c (h d) -> b h c d', h = self.head), (q, k, v))
        scale = q.shape[-1] ** -0.5

        qk = einsum('b h k d, b h q d -> b h k q', q, k) * scale 
       #  qk = torch.nn.functional.softmax(qk, dim=-1)
        qk = self.taylor_softmax(qk, order=2)        

        wv = einsum('b h k q, b h q d -> b h k d', qk, v) 
        wv = rearrange(wv, 'b h c d -> b c (h d)')
        out = self.out(wv)
        return out

```

