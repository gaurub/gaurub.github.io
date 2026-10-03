---
layout: post
title: "Learning CuTe DSL"
date: 2026-09-29 07:55:23 +0000
---

Edit: This is now a multi-part worklog.

Part 1: this post.  
Part 2: [Here]({% post_url 2026-09-30-kernel-2-shared-memory-caching %}).

A few months ago, I came across CuTe DSL and thought it was a great idea to make GPU programming more accessible. Little did I know that while I was spared from the idiosyncracies of C++, I still had to know about types, lanes, warps, blocks, grids, swizzles and all the other goodness that comes along with it. Thankfully, the C experience from back in the day was very helpful and I’ve been able to get my head around it pretty well. I am writing this post facto, so there may be some slips in memory, but I’m trying to tell it as close to “as it happened” as possible.

## 1. The Plan

My first thought was that I’d re-implement all the kernels from siboehm’s beautifully written post [How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog](https://siboehm.com/articles/22/CUDA-MMM). I learned a lot from that post, so full credit to him. However, after writing about 2–3 kernels from his list, I started branching out to answer my own questions, which is the kind of learning I love doing. I’ll be writing a series of posts documenting the entire journey and will update the list I’ll create at the top so articles are easier to find.

## 2. The Execution

Of course, if I just used Claude Code, I would learn absolutely nothing, so the first order of business was to write a CLAUDE.MD that would actually answer questions, clarify doubts, but never give me whole answers and never give me code. I would like to thank the staff of Stanford’s CS336 class for providing a great CLAUDE.MD that worked beautifully. For me, figuring things out, answering my own questions by thinking long and hard is the best way to learn, so this was essential.

Anyway, I started off by writing the most naive matmul kernel I could and compared it to the performance of cuBLAS. Naturally, it came out terrible, but here it is in all it’s glory ( or shame if you will ). I will cover this line by line later.

```python
"""Kernel 1: naive. One thread per output element, no reuse."""
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

from registry import register


@cute.kernel
def naive_kernel(gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor):
    tidx, tidy, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()
    bdimx, bdimy, _ = cute.arch.block_dim()
    m, n = gA.shape
    _, p = gB.shape
    col = bidx * bdimx + tidx
    row = bidy * bdimy + tidy
    if row < m and col < p:
        acc = cutlass.Float32(0.0)
        for i in cutlass.range(n):
            acc += gA[row, i].to(cutlass.Float32) * gB[i, col].to(cutlass.Float32)
        gC[row, col] = acc.to(gC.element_type)


@cute.jit
def naive_jit(mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor):
    m, n = mA.shape
    _, p = mB.shape
    naive_kernel(mA, mB, mC).launch(
        grid=((p + 15) // 16, (m + 15) // 16, 1), block=(16, 16, 1))


@register("naive", budget_ms=60.0)
def _(a, b, c):
    a_, b_, c_ = (from_dlpack(t, assumed_align=16) for t in (a, b, c))
    f = cute.compile(naive_jit, a_, b_, c_)
    return (lambda: f(a_, b_, c_)), c
```

OK, let’s look at the JIT method naive_jit and the kernel, naive_kernel a little more closely:

```python
m, n = mA.shape
_, p = mB.shape
naive_kernel(mA, mB, mC).launch(
grid=((p + 15) // 16, (m + 15) // 16, 1), block=(16, 16, 1))
```

I wanted grids that were 16 x16 ( at the time ). If I had to redo this after learning everything I have from all the other kernels I’ve written, I would launch the block as (256, 1, 1) and then resize the grid as needed. Why I would do this will become clear later, but for now, suffice it to say that it makes things a LOT easier.

Anyway, this code block basically gets the shapes of the operands and launches the kernel with block and grid shapes that are intended to cover the whole matrix.

```python
tidx, tidy, _ = cute.arch.thread_idx()
bidx, bidy, _ = cute.arch.block_idx()
bdimx, bdimy, _ = cute.arch.block_dim()
```

This block of code ( or something similar ) will be at the top of almost every kernel, since we need to know which “thread” we are and we need to know which block we’re in. The block_dim will become less important later.

```python
m, n = gA.shape
_, p = gB.shape
col = bidx * bdimx + tidx
row = bidy * bdimy + tidy
```

This gives us the shape of the operands again ( in retrospect, I could have passed them in as a parameter) and calculates which “col” of B and which “row” of A we’re going to loop over.

```python
if row < m and col < p:
    acc = cutlass.Float32(0.0)
    for i in cutlass.range(n):
        acc += gA[row, i].to(cutlass.Float32) * gB[i, col].to(cutlass.Float32)
    gC[row, col] = acc.to(gC.element_type)
```

The if statements just make sure we don’t over the “ends” of any matrix that doesn’t fit neatly into our blocks. The runtime is going to create blocks of 16 x 16 whether the matrix needs it or not. If you have a 5 x 5 matrix, you’re still going to get a 16 x 16 block, so you have to check if you’re going over the ends.

acc is the variable we accumulate the results in. We then loop over the “reduction dimension” and multiply each element of A in row by each element of B in col. We then set the element in the result, gC[row, col] to the value we get. Notice we cast the values from gA and gB to cutlass.Float32 since our accumulator is a cutlass.Float32 and our operands gA and gB are in cutlass.BFloat16. When setting the result in gC we have to cast back to a cutlass.BFloat16. In order to not lose generality or to avoid encountering issues if the operands are a different datatype, I use gC.element_type to ensure the code works for a wide range of operand datatypes.

## 3. The Result

As expected, this is extremely slow, since it’s literally the worst matmul algorithm there is. We’re going to work to make this quicker over time ( just like siboehm ), but also start asking other questions and learning by answering them.

The speed runs show us exactly what we expect:

| Size (M = N = K) | Naive (ms) | Naive (TFLOP/s) | cuBLAS (ms) | cuBLAS (TFLOP/s) | cuBLAS speedup |
|-----------------:|-----------:|----------------:|------------:|-----------------:|---------------:|
| 512              | 0.039      | 7.0             | 0.006       | 47.1             | 6.8×           |
| 1024             | 0.264      | 8.1             | 0.010       | 222.4            | 27.4×          |
| 2048             | 2.083      | 8.2             | 0.051       | 334.4            | 40.5×          |
| 4096             | 16.895     | 8.1             | 0.320       | 428.9            | 52.7×          |

*Tested on an Nvidia RTX Pro 6000 Blackwell with bf16 inputs. Each is the median of 5 repeats of 10 iterations, after 3 warmup runs*

This was tested solely on square matrices, but we’re only really going for speed tests here, so it should be fine.

Next, we’ll move onto making this quicker, by loading things into shared memory and re-using values as many times as possible.
