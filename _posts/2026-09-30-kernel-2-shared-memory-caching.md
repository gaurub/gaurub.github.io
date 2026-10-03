---
layout: post
title: "Kernel 2: Shared memory caching"
date: 2026-09-30 11:01:14 +0000
---

We’re skipping memory coalescing, which I’ll cover in more detail in a different post later. For now, if you load elements as they were in the previous thread, with a row-major layout, you’re good. I’ll cover layouts in a separate post, since I still make a mistake with those sometimes.

The basic idea with this kernel is to load values into shared memory, which provides much quicker access than global memory. Each SM has shared memory blocks on chip, allowing us to load elements from A and B once and use them as many times as possible to avoid hitting global memory every time we need a value from either A or B.

In order to get really used to what’s going on ( since it’s a paradigm we’re going to be following for the remaining kernels ), I created a visualizer that I could use to picture things happening so I could get a clearer picture of everything. I’m still using 2D blocks ( 16 x 16 ), which I would not do if I were to go back, but it’s still in this code, so it is what it is.

{% include visualizer.html src="tile-walk.html" title="Kernel 2 Tile Walk" %}

I urge you to play around with the visualization. While everything is simple right now, it gets a little harder to wrap your head around once a single thread starts loading multiple values and even more interesting when a single thread starts calculating multiple output values. We’ll be walking through those with visualizations like this as well.

```python
smem = SmemAllocator()

layout = cute.make_layout((TILE, TILE), stride=(TILE, 1))

As = smem.allocate_tensor(gA.element_type, layout)
Bs = smem.allocate_tensor(gB.element_type, layout)
```

This block of code instantiates an SmemAllocator, defines a layout for our “tiles” and then allocates the shared memory we’re going to use to keep our values handy when we’re using them to compute. The stride=(TILE, 1) argument basically gives us a layout in row-major order. We’ll get more into layouts in a separate post. We then use the SmemAllocator to allocate two buffers in shared memory, one for our entries from A and one for our entries from B. Notice the types and the byte alignment specification.

```python
tidx, tidy, _ = cute.arch.thread_idx()
bidx, bidy, _ = cute.arch.block_idx()
bdimx, bdimy, _ = cute.arch.block_dim()

m, n = gA.shape
_, p = gB.shape
```

We’re then getting our usual values: thread ID, block ID and block dimensions. Again, if I had to do this again, I’d declare a (256, 1, 1) block, but that realization only came later for me, so I’m writing it out as it happened.

```python
 row = bidy * TILE + tidy
 col = bidx * TILE + tidx
```

So each thread will be on a fixed row in A and a fixed column in B since we’re “hopping” across the row and down the column in the reduction dimension. I elaborate on this a bit below.

```python
 hops = (n + TILE - 1) // TILE

 Pvalue = cutlass.Float32(0.0)
```

The hops is the crucial variable here. It defines how many times the tile will “move” across the matrix. If you look at the visualizer above, you see a tile only grabs a 16x16 slice ( as written ). In order to grab a full row of A and a full column of B, that tile will have to move across the rows of A and down the columns of B. We are defining how many times it has to move to grab all the columns of A (since we grab columns of A when moving across a row). Since the reduction dimension MUST be equal in a matrix multiply, that will be equal to the number of rows we must grab when going down a column of B.

We initialize the “accumulator”, i.e. the variable named Pvalue, since that is where we’ll be storing all of our interim sums.

```python
for i in range(hops):
   aCol = i * TILE + tidx
   aRow = i * TILE + tidy

   As[tidy, tidx] = gA[row, aCol] if (row < m and aCol < n) else gA.element_type(0.0)
   Bs[tidy, tidx] = gB[aRow, col] if (aRow < n and col < p) else gB.element_type(0.0)

   cute.arch.sync_threads()

   for j in range(TILE):
        Pvalue += As[tidy, j].to(cutlass.Float32) * Bs[j, tidx].to(cutlass.Float32)

    cute.arch.sync_threads()
```

This block of code is the meat of the calculation. First, we’re going to loop hops times. During each iteration, we’re first going to find out which row of A and which column of B we’re going to load into shared memory. The row and col itself are already calculated previously and we just have to calculate which ones to load for **this** particular hop. We then actually load those values into shared memory using our handy tidy and tidx values. These allow us to index into the shared memory buffers since we’ve defined our block sizes appropriately. We then synchronize across threads since we don’t want anything to happen until all values are loaded.

The next part is just iterating over the As and Bs shared memory buffers to calculate the product of As[tidy, j] and Bs[j, tidx]. We cast both of these to cutlass.Float32 since our Pvalue accumulator is of that type. We then synchronize threads once more to make sure we don’t “hop” before all the threads have completed their required calculations.

```python
if row < m and col < p:
    gC[row, col] = Pvalue.to(gC.element_type)
```

Finally, at the end, we store the value in Pvalue to the row, col in gC, our output buffer.

Below is the entire code for this kernel without commentary.

```python
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from cutlass.memory import SmemAllocator

from registry import register

TILE = 16


@cute.kernel
def tiled_mm_kernel(gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor):
    smem = SmemAllocator()

    layout = cute.make_layout((TILE, TILE), stride=(TILE, 1))
    #layout = cute.make_layout((TILE, TILE))
    As = smem.allocate_tensor(gA.element_type, layout, byte_alignment=16)
    Bs = smem.allocate_tensor(gB.element_type, layout, byte_alignment=16)

    tidx, tidy, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()
    bdimx, bdimy, _ = cute.arch.block_dim()

    m, n = gA.shape
    _, p = gB.shape

    row = bidy * TILE + tidy
    col = bidx * TILE + tidx

    hops = (n + TILE - 1) // TILE
    Pvalue = cutlass.Float32(0.0)
    for i in range(hops):
        aCol = i * TILE + tidx
        aRow = i * TILE + tidy

        As[tidy, tidx] = gA[row, aCol] if (row < m and aCol < n) else gA.element_type(0.0)
        Bs[tidy, tidx] = gB[aRow, col] if (aRow < n and col < p) else gB.element_type(0.0)

        cute.arch.sync_threads()

        for j in range(TILE):
            Pvalue += As[tidy, j].to(cutlass.Float32) * Bs[j, tidx].to(cutlass.Float32)

        cute.arch.sync_threads()

    if row < m and col < p:
        gC[row, col] = Pvalue.to(gC.element_type)

@cute.jit
def tiled_mm(mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor):
    m, n = mA.shape
    _, p = mB.shape
    block=(TILE, TILE, 1)
    grid=((p + TILE - 1 ) // TILE, (m + TILE - 1) // TILE, 1)
    tiled_mm_kernel(mA, mB, mC).launch(grid=grid, block=block)
```
