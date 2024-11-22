# MiniTorch Module 3

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module3.html


You will need to modify `tensor_functions.py` slightly in this assignment.

* Tests:

```
python run_tests.py
```

* Note:

Several of the tests for this assignment will only run if you are on a GPU machine and will not
run on github's test infrastructure. Please follow the instructions to setup up a colab machine
to run these tests.

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/tensor_data.py minitorch/tensor_functions.py minitorch/tensor_ops.py minitorch/operators.py minitorch/scalar.py minitorch/scalar_functions.py minitorch/module.py minitorch/autodiff.py minitorch/module.py project/run_manual.py project/run_scalar.py project/run_tensor.py minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/tensor.py minitorch/datasets.py minitorch/testing.py minitorch/optim.py


# Diagnostic Output for task3_1 and task3_2
```
MAP

================================================================================
 Parallel Accelerator Optimizing:  Function tensor_map.<locals>._map, /Users/and
ychang/Documents/projects/minitorch/mod3-andyrochi/minitorch/fast_ops.py (185)
================================================================================


Parallel loop listing for  Function tensor_map.<locals>._map, /Users/andychang/Documents/projects/minitorch/mod3-andyrochi/minitorch/fast_ops.py (185)
-------------------------------------------------------------------------|loop #ID
    def _map(                                                            |
        out: Storage,                                                    |
        out_shape: Shape,                                                |
        out_strides: Strides,                                            |
        in_storage: Storage,                                             |
        in_shape: Shape,                                                 |
        in_strides: Strides,                                             |
    ) -> None:                                                           |
        # TODO: Implement for Task 3.1.                                  |
        if is_aligned(out_strides, in_strides, out_shape, in_shape):     |
            for i in prange(len(out)):-----------------------------------| #2
                out[i] = fn(in_storage[i])                               |
            return                                                       |
                                                                         |
        for i in prange(len(out)):---------------------------------------| #3
            out_index: Index = np.zeros(MAX_DIMS, np.int32)--------------| #0
            in_index: Index = np.zeros(MAX_DIMS, np.int32)---------------| #1
            to_index(i, out_shape, out_index)                            |
            broadcast_index(out_index, out_shape, in_shape, in_index)    |
            o = index_to_position(out_index, out_strides)                |
            j = index_to_position(in_index, in_strides)                  |
            out[o] = fn(in_storage[j])                                   |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...

Fused loop summary:
+--0 has the following loops fused into it:
   +--1 (fused)
Following the attempted fusion of parallel for-loops there are 3 parallel for-
loop(s) (originating from loops labelled: #2, #3, #0).
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...

+--3 is a parallel loop
   +--0 --> rewritten as a serial loop
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--3 (parallel)
   +--0 (parallel)
   +--1 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--3 (parallel)
   +--0 (serial, fused with loop(s): 1)



Parallel region 0 (loop #3) had 1 loop(s) fused and 1 loop(s) serialized as part
 of the larger parallel loop (#3).
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at /Users/andychang/Documents
/projects/minitorch/mod3-andyrochi/minitorch/fast_ops.py (200) is hoisted out of
 the parallel loop labelled #3 (it will be performed before the loop is executed
 and reused inside the loop):
   Allocation:: out_index: Index = np.zeros(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at /Users/andychang/Documents
/projects/minitorch/mod3-andyrochi/minitorch/fast_ops.py (201) is hoisted out of
 the parallel loop labelled #3 (it will be performed before the loop is executed
 and reused inside the loop):
   Allocation:: in_index: Index = np.zeros(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
None
ZIP

================================================================================
 Parallel Accelerator Optimizing:  Function tensor_zip.<locals>._zip, /Users/and
ychang/Documents/projects/minitorch/mod3-andyrochi/minitorch/fast_ops.py (234)
================================================================================


Parallel loop listing for  Function tensor_zip.<locals>._zip, /Users/andychang/Documents/projects/minitorch/mod3-andyrochi/minitorch/fast_ops.py (234)
-------------------------------------------------------------------------------------|loop #ID
    def _zip(                                                                        |
        out: Storage,                                                                |
        out_shape: Shape,                                                            |
        out_strides: Strides,                                                        |
        a_storage: Storage,                                                          |
        a_shape: Shape,                                                              |
        a_strides: Strides,                                                          |
        b_storage: Storage,                                                          |
        b_shape: Shape,                                                              |
        b_strides: Strides,                                                          |
    ) -> None:                                                                       |
        # TODO: Implement for Task 3.1.                                              |
        if is_aligned(out_strides, a_strides, out_shape, a_shape) and is_aligned(    |
            out_strides, b_strides, out_shape, b_shape                               |
        ):                                                                           |
            for i in prange(len(out)):-----------------------------------------------| #7
                out[i] = fn(a_storage[i], b_storage[i])                              |
            return                                                                   |
                                                                                     |
        for i in prange(len(out)):---------------------------------------------------| #8
            out_index: Index = np.zeros(MAX_DIMS, np.int32)--------------------------| #4
            a_index: Index = np.zeros(MAX_DIMS, np.int32)----------------------------| #5
            b_index: Index = np.zeros(MAX_DIMS, np.int32)----------------------------| #6
            to_index(i, out_shape, out_index)                                        |
            o = index_to_position(out_index, out_strides)                            |
            broadcast_index(out_index, out_shape, a_shape, a_index)                  |
            j = index_to_position(a_index, a_strides)                                |
            broadcast_index(out_index, out_shape, b_shape, b_index)                  |
            k = index_to_position(b_index, b_strides)                                |
            out[o] = fn(a_storage[j], b_storage[k])                                  |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...

Fused loop summary:
+--4 has the following loops fused into it:
   +--5 (fused)
   +--6 (fused)
Following the attempted fusion of parallel for-loops there are 3 parallel for-
loop(s) (originating from loops labelled: #7, #8, #4).
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...

+--8 is a parallel loop
   +--4 --> rewritten as a serial loop
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--8 (parallel)
   +--4 (parallel)
   +--5 (parallel)
   +--6 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--8 (parallel)
   +--4 (serial, fused with loop(s): 5, 6)



Parallel region 0 (loop #8) had 2 loop(s) fused and 1 loop(s) serialized as part
 of the larger parallel loop (#8).
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at /Users/andychang/Documents
/projects/minitorch/mod3-andyrochi/minitorch/fast_ops.py (254) is hoisted out of
 the parallel loop labelled #8 (it will be performed before the loop is executed
 and reused inside the loop):
   Allocation:: out_index: Index = np.zeros(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at /Users/andychang/Documents
/projects/minitorch/mod3-andyrochi/minitorch/fast_ops.py (255) is hoisted out of
 the parallel loop labelled #8 (it will be performed before the loop is executed
 and reused inside the loop):
   Allocation:: a_index: Index = np.zeros(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at /Users/andychang/Documents
/projects/minitorch/mod3-andyrochi/minitorch/fast_ops.py (256) is hoisted out of
 the parallel loop labelled #8 (it will be performed before the loop is executed
 and reused inside the loop):
   Allocation:: b_index: Index = np.zeros(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
None
REDUCE

================================================================================
 Parallel Accelerator Optimizing:  Function tensor_reduce.<locals>._reduce, /Use
rs/andychang/Documents/projects/minitorch/mod3-andyrochi/minitorch/fast_ops.py
(289)
================================================================================


Parallel loop listing for  Function tensor_reduce.<locals>._reduce, /Users/andychang/Documents/projects/minitorch/mod3-andyrochi/minitorch/fast_ops.py (289)
----------------------------------------------------------------|loop #ID
    def _reduce(                                                |
        out: Storage,                                           |
        out_shape: Shape,                                       |
        out_strides: Strides,                                   |
        a_storage: Storage,                                     |
        a_shape: Shape,                                         |
        a_strides: Strides,                                     |
        reduce_dim: int,                                        |
    ) -> None:                                                  |
        # TODO: Implement for Task 3.1.                         |
        reduce_size = a_shape[reduce_dim]                       |
        for i in prange(len(out)):------------------------------| #10
            out_index: Index = np.zeros(MAX_DIMS, np.int32)-----| #9
            to_index(i, out_shape, out_index)                   |
            o = index_to_position(out_index, out_strides)       |
            temp = out[o]                                       |
            out_index[reduce_dim] = 0                           |
            j_base = index_to_position(out_index, a_strides)    |
            for s in range(reduce_size):                        |
                j = int(j_base + a_strides[reduce_dim] * s)     |
                temp = fn(temp, a_storage[j])                   |
            out[o] = temp                                       |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 2 parallel for-
loop(s) (originating from loops labelled: #10, #9).
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...

+--10 is a parallel loop
   +--9 --> rewritten as a serial loop
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--10 (parallel)
   +--9 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--10 (parallel)
   +--9 (serial)



Parallel region 0 (loop #10) had 0 loop(s) fused and 1 loop(s) serialized as
part of the larger parallel loop (#10).
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at /Users/andychang/Documents
/projects/minitorch/mod3-andyrochi/minitorch/fast_ops.py (301) is hoisted out of
 the parallel loop labelled #10 (it will be performed before the loop is
executed and reused inside the loop):
   Allocation:: out_index: Index = np.zeros(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
None
MATRIX MULTIPLY

================================================================================
 Parallel Accelerator Optimizing:  Function _tensor_matrix_multiply, /Users/andy
chang/Documents/projects/minitorch/mod3-andyrochi/minitorch/fast_ops.py (315)
================================================================================


Parallel loop listing for  Function _tensor_matrix_multiply, /Users/andychang/Documents/projects/minitorch/mod3-andyrochi/minitorch/fast_ops.py (315)
--------------------------------------------------------------------------------------------|loop #ID
def _tensor_matrix_multiply(                                                                |
    out: Storage,                                                                           |
    out_shape: Shape,                                                                       |
    out_strides: Strides,                                                                   |
    a_storage: Storage,                                                                     |
    a_shape: Shape,                                                                         |
    a_strides: Strides,                                                                     |
    b_storage: Storage,                                                                     |
    b_shape: Shape,                                                                         |
    b_strides: Strides,                                                                     |
) -> None:                                                                                  |
    """NUMBA tensor matrix multiply function.                                               |
                                                                                            |
    Should work for any tensor shapes that broadcast as long as                             |
                                                                                            |
    ```                                                                                     |
    assert a_shape[-1] == b_shape[-2]                                                       |
    ```                                                                                     |
                                                                                            |
    Optimizations:                                                                          |
                                                                                            |
    * Outer loop in parallel                                                                |
    * No index buffers or function calls                                                    |
    * Inner loop should have no global writes, 1 multiply.                                  |
                                                                                            |
                                                                                            |
    Args:                                                                                   |
    ----                                                                                    |
        out (Storage): storage for `out` tensor                                             |
        out_shape (Shape): shape for `out` tensor                                           |
        out_strides (Strides): strides for `out` tensor                                     |
        a_storage (Storage): storage for `a` tensor                                         |
        a_shape (Shape): shape for `a` tensor                                               |
        a_strides (Strides): strides for `a` tensor                                         |
        b_storage (Storage): storage for `b` tensor                                         |
        b_shape (Shape): shape for `b` tensor                                               |
        b_strides (Strides): strides for `b` tensor                                         |
                                                                                            |
    Returns:                                                                                |
    -------                                                                                 |
        None : Fills in `out`                                                               |
                                                                                            |
    """                                                                                     |
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0                                  |
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0                                  |
                                                                                            |
    # TODO: Implement for Task 3.2.                                                         |
    assert a_shape[-1] == b_shape[-2]                                                       |
                                                                                            |
    for i in prange(len(out)):--------------------------------------------------------------| #11
        # matrix index                                                                      |
        out_0 = i // (out_shape[-2] * out_shape[-1])                                        |
        out_1 = (i % (out_shape[-2] * out_shape[-1])) // out_shape[-1]                      |
        out_2 = i % out_shape[-1]                                                           |
                                                                                            |
        out_i = out_0 * out_strides[0] + out_1 * out_strides[1] + out_2 * out_strides[2]    |
        a_start = out_0 * a_batch_stride + out_1 * a_strides[-2]                            |
        b_start = out_0 * b_batch_stride + out_2 * b_strides[-1]                            |
        # sum over reduced dimension                                                        |
        temp = 0                                                                            |
        for j in range(a_shape[-1]):                                                        |
            a_i = a_start + j * a_strides[-1]                                               |
            b_i = b_start + j * b_strides[-2]                                               |
            temp += a_storage[a_i] * b_storage[b_i]                                         |
        out[out_i] = temp                                                                   |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 1 parallel for-
loop(s) (originating from loops labelled: #11).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
No allocation hoisting found
None

```

# Task 3_4

## Hidden=100 Split Dataset
**CPU**

Average time per epoch: 0.16768265342712402s
```
Epoch    0 | loss  6.30263 | correct 33  | time 18.1381s
Epoch   10 | loss  4.59244 | correct 36  | time 0.1246s
Epoch   20 | loss  4.01475 | correct 33  | time 0.1156s
Epoch   30 | loss  4.67928 | correct 45  | time 0.1146s
Epoch   40 | loss  3.44027 | correct 46  | time 0.1156s
Epoch   50 | loss  3.00204 | correct 48  | time 0.1161s
Epoch   60 | loss  1.80957 | correct 48  | time 0.2701s
Epoch   70 | loss  0.89136 | correct 47  | time 0.1144s
Epoch   80 | loss  2.07281 | correct 48  | time 0.1161s
Epoch   90 | loss  2.93252 | correct 50  | time 0.1177s
Epoch  100 | loss  2.11063 | correct 50  | time 0.1175s
Epoch  110 | loss  1.16841 | correct 49  | time 0.1178s
Epoch  120 | loss  1.96589 | correct 49  | time 0.1540s
Epoch  130 | loss  1.10930 | correct 49  | time 0.1167s
Epoch  140 | loss  1.36004 | correct 49  | time 0.1170s
Epoch  150 | loss  1.17644 | correct 49  | time 0.1187s
Epoch  160 | loss  1.37316 | correct 49  | time 0.2094s
Epoch  170 | loss  1.36451 | correct 50  | time 0.1148s
Epoch  180 | loss  0.19618 | correct 47  | time 0.1140s
Epoch  190 | loss  0.94940 | correct 50  | time 0.1141s
Epoch  200 | loss  1.30178 | correct 49  | time 0.1181s
Epoch  210 | loss  0.45300 | correct 48  | time 0.1146s
Epoch  220 | loss  0.56713 | correct 49  | time 0.1174s
Epoch  230 | loss  0.46659 | correct 49  | time 0.1159s
Epoch  240 | loss  1.33311 | correct 50  | time 0.1318s
Epoch  250 | loss  2.40920 | correct 47  | time 0.2702s
Epoch  260 | loss  1.15473 | correct 49  | time 0.2119s
Epoch  270 | loss  0.72215 | correct 49  | time 0.1142s
Epoch  280 | loss  1.40837 | correct 50  | time 0.1161s
Epoch  290 | loss  0.43633 | correct 50  | time 0.1165s
Epoch  300 | loss  0.58836 | correct 50  | time 0.1290s
Epoch  310 | loss  0.10710 | correct 50  | time 0.1163s
Epoch  320 | loss  1.12276 | correct 50  | time 0.1139s
Epoch  330 | loss  1.16407 | correct 49  | time 0.1152s
Epoch  340 | loss  0.22880 | correct 47  | time 0.1172s
Epoch  350 | loss  1.47903 | correct 50  | time 0.2405s
Epoch  360 | loss  0.74166 | correct 49  | time 0.1277s
Epoch  370 | loss  1.18424 | correct 49  | time 0.1154s
Epoch  380 | loss  1.34720 | correct 49  | time 0.1260s
Epoch  390 | loss  0.21780 | correct 49  | time 0.1151s
Epoch  400 | loss  0.55588 | correct 49  | time 0.1135s
Epoch  410 | loss  0.72839 | correct 50  | time 0.1147s
Epoch  420 | loss  0.79172 | correct 50  | time 0.1293s
Epoch  430 | loss  0.13787 | correct 49  | time 0.1156s
Epoch  440 | loss  2.24387 | correct 48  | time 0.1149s
Epoch  450 | loss  0.10307 | correct 49  | time 0.2219s
Epoch  460 | loss  0.21960 | correct 49  | time 0.1133s
Epoch  470 | loss  0.07558 | correct 49  | time 0.1162s
Epoch  480 | loss  1.51691 | correct 47  | time 0.1149s
Epoch  490 | loss  0.98830 | correct 49  | time 0.1432s
Epoch  499 | loss  0.93818 | correct 50  | time 0.1160s
Average time per epoch:  0.16768265342712402
```

**GPU**

Average time per epoch: 1.9713713564872741s
```
Epoch    0 | loss  8.24566 | correct 32  | time 4.6446s
Epoch   10 | loss  4.99586 | correct 32  | time 1.7911s
Epoch   20 | loss  4.91305 | correct 40  | time 1.8953s
Epoch   30 | loss  5.13396 | correct 42  | time 1.7961s
Epoch   40 | loss  2.97881 | correct 40  | time 1.8539s
Epoch   50 | loss  3.73369 | correct 48  | time 2.5602s
Epoch   60 | loss  7.36139 | correct 40  | time 1.8792s
Epoch   70 | loss  2.48868 | correct 45  | time 1.8497s
Epoch   80 | loss  4.31109 | correct 43  | time 1.8178s
Epoch   90 | loss  3.65239 | correct 47  | time 1.8216s
Epoch  100 | loss  1.25482 | correct 49  | time 2.4526s
Epoch  110 | loss  2.45641 | correct 49  | time 2.6098s
Epoch  120 | loss  2.34214 | correct 49  | time 1.8069s
Epoch  130 | loss  1.45839 | correct 50  | time 2.3749s
Epoch  140 | loss  1.97303 | correct 47  | time 1.8628s
Epoch  150 | loss  0.46042 | correct 48  | time 1.8113s
Epoch  160 | loss  1.51592 | correct 48  | time 1.8113s
Epoch  170 | loss  1.83917 | correct 47  | time 1.8124s
Epoch  180 | loss  0.77097 | correct 50  | time 2.6344s
Epoch  190 | loss  0.10024 | correct 49  | time 1.8031s
Epoch  200 | loss  0.71316 | correct 49  | time 1.8214s
Epoch  210 | loss  0.91885 | correct 50  | time 1.8691s
Epoch  220 | loss  0.85468 | correct 50  | time 1.8162s
Epoch  230 | loss  0.96510 | correct 50  | time 1.8273s
Epoch  240 | loss  0.49429 | correct 50  | time 1.8174s
Epoch  250 | loss  0.22459 | correct 49  | time 1.8766s
Epoch  260 | loss  0.97530 | correct 49  | time 2.4936s
Epoch  270 | loss  0.43523 | correct 50  | time 1.8180s
Epoch  280 | loss  0.69628 | correct 50  | time 1.8786s
Epoch  290 | loss  0.69677 | correct 50  | time 1.8100s
Epoch  300 | loss  1.02373 | correct 50  | time 1.8543s
Epoch  310 | loss  0.52704 | correct 50  | time 2.4491s
Epoch  320 | loss  0.83295 | correct 49  | time 1.8931s
Epoch  330 | loss  1.08668 | correct 50  | time 1.8545s
Epoch  340 | loss  0.63454 | correct 50  | time 2.1029s
Epoch  350 | loss  1.02785 | correct 50  | time 1.8149s
Epoch  360 | loss  0.58935 | correct 50  | time 2.2850s
Epoch  370 | loss  0.17661 | correct 50  | time 1.8019s
Epoch  380 | loss  0.05708 | correct 47  | time 1.8286s
Epoch  390 | loss  0.48242 | correct 50  | time 2.2976s
Epoch  400 | loss  1.38738 | correct 49  | time 1.8099s
Epoch  410 | loss  0.49167 | correct 50  | time 1.8088s
Epoch  420 | loss  0.22154 | correct 50  | time 1.8211s
Epoch  430 | loss  0.60033 | correct 50  | time 1.8831s
Epoch  440 | loss  1.35166 | correct 50  | time 2.5437s
Epoch  450 | loss  0.30221 | correct 50  | time 1.8091s
Epoch  460 | loss  0.12074 | correct 50  | time 1.8247s
Epoch  470 | loss  0.08714 | correct 50  | time 2.0695s
Epoch  480 | loss  1.21963 | correct 50  | time 1.8079s
Epoch  490 | loss  0.31200 | correct 50  | time 2.1142s
Epoch  499 | loss  0.09991 | correct 50  | time 1.8007s
Average time per epoch:  1.9713713564872741
```

## Hidden=100 Simple Dataset

CPU

Average time per epoch:  0.1713330774307251s
```
Epoch    0 | loss  4.58434 | correct 39  | time 19.5411s
Epoch   10 | loss  1.91783 | correct 48  | time 0.1220s
Epoch   20 | loss  1.05980 | correct 48  | time 0.1200s
Epoch   30 | loss  1.17617 | correct 48  | time 0.1265s
Epoch   40 | loss  1.04635 | correct 50  | time 0.1309s
Epoch   50 | loss  2.48122 | correct 47  | time 0.1406s
Epoch   60 | loss  0.59751 | correct 50  | time 0.1249s
Epoch   70 | loss  0.84667 | correct 48  | time 0.2130s
Epoch   80 | loss  0.51342 | correct 50  | time 0.1251s
Epoch   90 | loss  1.09729 | correct 48  | time 0.1250s
Epoch  100 | loss  0.72946 | correct 50  | time 0.1201s
Epoch  110 | loss  0.46728 | correct 50  | time 0.1344s
Epoch  120 | loss  0.29516 | correct 50  | time 0.1167s
Epoch  130 | loss  0.46381 | correct 50  | time 0.1155s
Epoch  140 | loss  0.08354 | correct 49  | time 0.1322s
Epoch  150 | loss  0.35170 | correct 50  | time 0.1169s
Epoch  160 | loss  0.65014 | correct 50  | time 0.2051s
Epoch  170 | loss  0.01565 | correct 50  | time 0.1159s
Epoch  180 | loss  0.21451 | correct 50  | time 0.1161s
Epoch  190 | loss  0.61830 | correct 50  | time 0.1292s
Epoch  200 | loss  0.19906 | correct 50  | time 0.1259s
Epoch  210 | loss  0.48719 | correct 50  | time 0.1156s
Epoch  220 | loss  0.88823 | correct 50  | time 0.1164s
Epoch  230 | loss  0.14312 | correct 50  | time 0.1166s
Epoch  240 | loss  0.41328 | correct 50  | time 0.1179s
Epoch  250 | loss  0.21762 | correct 50  | time 0.2699s
Epoch  260 | loss  0.43380 | correct 50  | time 0.1183s
Epoch  270 | loss  0.38766 | correct 50  | time 0.1146s
Epoch  280 | loss  0.51087 | correct 50  | time 0.1178s
Epoch  290 | loss  0.58627 | correct 50  | time 0.1153s
Epoch  300 | loss  0.30660 | correct 50  | time 0.1167s
Epoch  310 | loss  0.17974 | correct 50  | time 0.1267s
Epoch  320 | loss  0.12566 | correct 50  | time 0.1157s
Epoch  330 | loss  0.03988 | correct 50  | time 0.1157s
Epoch  340 | loss  0.01805 | correct 50  | time 0.1549s
Epoch  350 | loss  0.04543 | correct 50  | time 0.2650s
Epoch  360 | loss  0.01914 | correct 50  | time 0.1167s
Epoch  370 | loss  0.28637 | correct 50  | time 0.1230s
Epoch  380 | loss  0.11257 | correct 50  | time 0.1172s
Epoch  390 | loss  0.04241 | correct 50  | time 0.1154s
Epoch  400 | loss  0.26103 | correct 50  | time 0.1158s
Epoch  410 | loss  0.23377 | correct 50  | time 0.1158s
Epoch  420 | loss  0.06340 | correct 50  | time 0.1170s
Epoch  430 | loss  0.62987 | correct 50  | time 0.1343s
Epoch  440 | loss  0.02296 | correct 50  | time 0.2516s
Epoch  450 | loss  0.04652 | correct 50  | time 0.1181s
Epoch  460 | loss  0.06631 | correct 50  | time 0.1224s
Epoch  470 | loss  0.02936 | correct 50  | time 0.1177s
Epoch  480 | loss  0.15983 | correct 50  | time 0.1164s
Epoch  490 | loss  0.00796 | correct 50  | time 0.1296s
Epoch  499 | loss  0.19655 | correct 50  | time 0.1237s
Average time per epoch:  0.1713330774307251
```

GPU

Average time per epoch:  1.9595617141723634s
```
Epoch    0 | loss  5.03550 | correct 48  | time 5.2272s
Epoch   10 | loss  1.69084 | correct 49  | time 1.8320s
Epoch   20 | loss  0.88179 | correct 49  | time 1.8553s
Epoch   30 | loss  0.98482 | correct 49  | time 2.4416s
Epoch   40 | loss  0.31078 | correct 49  | time 1.8134s
Epoch   50 | loss  0.07342 | correct 49  | time 1.8018s
Epoch   60 | loss  1.02487 | correct 49  | time 1.8608s
Epoch   70 | loss  0.25909 | correct 49  | time 1.8033s
Epoch   80 | loss  0.85012 | correct 49  | time 2.3156s
Epoch   90 | loss  0.03449 | correct 49  | time 1.7995s
Epoch  100 | loss  0.64267 | correct 50  | time 1.8817s
Epoch  110 | loss  0.18327 | correct 49  | time 1.9327s
Epoch  120 | loss  0.90199 | correct 50  | time 1.8100s
Epoch  130 | loss  0.25974 | correct 50  | time 2.0663s
Epoch  140 | loss  0.69306 | correct 49  | time 1.8633s
Epoch  150 | loss  0.07695 | correct 50  | time 1.8157s
Epoch  160 | loss  0.05811 | correct 49  | time 2.1711s
Epoch  170 | loss  0.77698 | correct 50  | time 1.8053s
Epoch  180 | loss  0.23205 | correct 50  | time 1.8994s
Epoch  190 | loss  0.73111 | correct 50  | time 2.1675s
Epoch  200 | loss  0.07896 | correct 50  | time 1.7999s
Epoch  210 | loss  0.27524 | correct 50  | time 2.6752s
Epoch  220 | loss  0.00583 | correct 49  | time 1.8229s
Epoch  230 | loss  0.60512 | correct 50  | time 1.7973s
Epoch  240 | loss  0.61118 | correct 50  | time 1.7932s
Epoch  250 | loss  0.15247 | correct 50  | time 1.8677s
Epoch  260 | loss  0.03055 | correct 49  | time 2.5635s
Epoch  270 | loss  0.17781 | correct 50  | time 1.8107s
Epoch  280 | loss  0.70018 | correct 50  | time 1.8991s
Epoch  290 | loss  0.04042 | correct 50  | time 1.7935s
Epoch  300 | loss  0.03128 | correct 50  | time 1.8039s
Epoch  310 | loss  0.10753 | correct 50  | time 2.1491s
Epoch  320 | loss  0.07071 | correct 50  | time 1.8780s
Epoch  330 | loss  0.07134 | correct 50  | time 1.7951s
Epoch  340 | loss  0.03611 | correct 50  | time 2.0537s
Epoch  350 | loss  0.56580 | correct 50  | time 1.8014s
Epoch  360 | loss  0.05502 | correct 50  | time 2.1289s
Epoch  370 | loss  0.59572 | correct 50  | time 1.8186s
Epoch  380 | loss  0.65649 | correct 50  | time 1.8157s
Epoch  390 | loss  0.71847 | correct 50  | time 2.1441s
Epoch  400 | loss  0.01319 | correct 50  | time 1.7786s
Epoch  410 | loss  0.10282 | correct 50  | time 2.1140s
Epoch  420 | loss  0.48008 | correct 50  | time 1.8032s
Epoch  430 | loss  0.60033 | correct 50  | time 1.8623s
Epoch  440 | loss  0.06375 | correct 50  | time 2.3583s
Epoch  450 | loss  0.56886 | correct 50  | time 1.8124s
Epoch  460 | loss  0.53101 | correct 50  | time 1.7988s
Epoch  470 | loss  0.44980 | correct 50  | time 1.8933s
Epoch  480 | loss  0.61193 | correct 50  | time 1.8287s
Epoch  490 | loss  0.34155 | correct 50  | time 2.3685s
Epoch  499 | loss  0.00426 | correct 50  | time 1.8131s
Average time per epoch:  1.9595617141723634
```

## Hidden=100 XOR Dataset

CPU

Average time per epoch:  0.16871760082244874s
```
Epoch    0 | loss  5.84421 | correct 35  | time 17.7512s
Epoch   10 | loss  4.63925 | correct 39  | time 0.1166s
Epoch   20 | loss  5.62795 | correct 42  | time 0.1157s
Epoch   30 | loss  3.76341 | correct 42  | time 0.1164s
Epoch   40 | loss  5.52191 | correct 42  | time 0.1174s
Epoch   50 | loss  2.65806 | correct 44  | time 0.1161s
Epoch   60 | loss  5.86599 | correct 46  | time 0.1276s
Epoch   70 | loss  4.84378 | correct 44  | time 0.1171s
Epoch   80 | loss  1.98581 | correct 48  | time 0.1177s
Epoch   90 | loss  3.20568 | correct 47  | time 0.2660s
Epoch  100 | loss  1.50005 | correct 47  | time 0.1172s
Epoch  110 | loss  1.48623 | correct 48  | time 0.1178s
Epoch  120 | loss  1.65422 | correct 48  | time 0.1166s
Epoch  130 | loss  2.65912 | correct 48  | time 0.1178s
Epoch  140 | loss  0.93938 | correct 48  | time 0.1178s
Epoch  150 | loss  2.38433 | correct 48  | time 0.1186s
Epoch  160 | loss  1.79846 | correct 48  | time 0.1190s
Epoch  170 | loss  1.12592 | correct 49  | time 0.1150s
Epoch  180 | loss  0.66492 | correct 48  | time 0.1415s
Epoch  190 | loss  0.63555 | correct 48  | time 0.2253s
Epoch  200 | loss  1.53521 | correct 49  | time 0.1154s
Epoch  210 | loss  1.15253 | correct 48  | time 0.1161s
Epoch  220 | loss  1.28250 | correct 50  | time 0.1145s
Epoch  230 | loss  1.81124 | correct 48  | time 0.1139s
Epoch  240 | loss  1.04082 | correct 50  | time 0.1431s
Epoch  250 | loss  1.06944 | correct 48  | time 0.1158s
Epoch  260 | loss  1.64670 | correct 50  | time 0.1163s
Epoch  270 | loss  1.07213 | correct 50  | time 0.1150s
Epoch  280 | loss  0.32577 | correct 50  | time 0.2685s
Epoch  290 | loss  0.62600 | correct 50  | time 0.1171s
Epoch  300 | loss  0.10381 | correct 50  | time 0.1161s
Epoch  310 | loss  1.23920 | correct 48  | time 0.1184s
Epoch  320 | loss  1.54240 | correct 48  | time 0.1157s
Epoch  330 | loss  1.19802 | correct 50  | time 0.1202s
Epoch  340 | loss  0.03720 | correct 48  | time 0.1165s
Epoch  350 | loss  0.83830 | correct 50  | time 0.1190s
Epoch  360 | loss  0.67229 | correct 50  | time 0.1178s
Epoch  370 | loss  0.54666 | correct 50  | time 0.2188s
Epoch  380 | loss  0.07443 | correct 50  | time 0.1181s
Epoch  390 | loss  0.12677 | correct 48  | time 0.1182s
Epoch  400 | loss  0.19279 | correct 50  | time 0.1190s
Epoch  410 | loss  0.18594 | correct 48  | time 0.1188s
Epoch  420 | loss  0.16547 | correct 50  | time 0.1201s
Epoch  430 | loss  0.18817 | correct 50  | time 0.1196s
Epoch  440 | loss  0.92150 | correct 50  | time 0.1276s
Epoch  450 | loss  0.90311 | correct 50  | time 0.1158s
Epoch  460 | loss  0.07536 | correct 50  | time 0.1315s
Epoch  470 | loss  0.73342 | correct 50  | time 0.2418s
Epoch  480 | loss  0.02098 | correct 50  | time 0.1202s
Epoch  490 | loss  0.06997 | correct 48  | time 0.1173s
Epoch  499 | loss  0.03598 | correct 50  | time 0.1149s
Average time per epoch:  0.16871760082244874
```

GPU

Average time per epoch:  1.9411767544746399s
```
Epoch    0 | loss  7.06605 | correct 31  | time 3.6100s
Epoch   10 | loss  4.89749 | correct 46  | time 1.7856s
Epoch   20 | loss  5.09013 | correct 44  | time 2.4103s
Epoch   30 | loss  2.65768 | correct 40  | time 1.7941s
Epoch   40 | loss  2.10173 | correct 41  | time 1.8086s
Epoch   50 | loss  2.81551 | correct 48  | time 2.7875s
Epoch   60 | loss  1.67127 | correct 48  | time 1.8418s
Epoch   70 | loss  2.02051 | correct 41  | time 2.2809s
Epoch   80 | loss  1.75154 | correct 47  | time 1.7793s
Epoch   90 | loss  2.62378 | correct 49  | time 1.8015s
Epoch  100 | loss  1.92518 | correct 49  | time 1.8513s
Epoch  110 | loss  0.69146 | correct 50  | time 1.7868s
Epoch  120 | loss  1.99742 | correct 46  | time 2.5173s
Epoch  130 | loss  1.39902 | correct 50  | time 1.7845s
Epoch  140 | loss  1.44240 | correct 47  | time 1.8902s
Epoch  150 | loss  1.61594 | correct 47  | time 1.7727s
Epoch  160 | loss  1.88583 | correct 47  | time 1.7861s
Epoch  170 | loss  0.44606 | correct 49  | time 2.5818s
Epoch  180 | loss  1.98029 | correct 50  | time 1.8586s
Epoch  190 | loss  0.69171 | correct 50  | time 1.7859s
Epoch  200 | loss  1.04941 | correct 50  | time 1.7797s
Epoch  210 | loss  1.75668 | correct 47  | time 1.8347s
Epoch  220 | loss  1.16435 | correct 50  | time 2.5797s
Epoch  230 | loss  1.13215 | correct 50  | time 1.7957s
Epoch  240 | loss  0.90221 | correct 50  | time 1.7848s
Epoch  250 | loss  1.52437 | correct 50  | time 1.8636s
Epoch  260 | loss  0.67454 | correct 50  | time 1.7827s
Epoch  270 | loss  0.26074 | correct 50  | time 2.3143s
Epoch  280 | loss  0.29093 | correct 50  | time 1.8494s
Epoch  290 | loss  0.16919 | correct 50  | time 1.9915s
Epoch  300 | loss  0.34503 | correct 50  | time 1.7709s
Epoch  310 | loss  0.38581 | correct 50  | time 1.8266s
Epoch  320 | loss  0.34863 | correct 50  | time 2.4227s
Epoch  330 | loss  0.23710 | correct 50  | time 1.7965s
Epoch  340 | loss  1.08480 | correct 50  | time 1.9001s
Epoch  350 | loss  0.91562 | correct 50  | time 1.7814s
Epoch  360 | loss  0.03000 | correct 50  | time 1.8403s
Epoch  370 | loss  0.46782 | correct 50  | time 1.9644s
Epoch  380 | loss  1.05265 | correct 50  | time 1.7891s
Epoch  390 | loss  0.19091 | correct 50  | time 2.3897s
Epoch  400 | loss  0.54707 | correct 50  | time 1.7737s
Epoch  410 | loss  0.06568 | correct 50  | time 1.8062s
Epoch  420 | loss  0.27175 | correct 50  | time 1.8168s
Epoch  430 | loss  0.25460 | correct 50  | time 1.8658s
Epoch  440 | loss  0.07177 | correct 50  | time 2.3285s
Epoch  450 | loss  0.52242 | correct 50  | time 1.8196s
Epoch  460 | loss  0.24600 | correct 50  | time 1.7841s
Epoch  470 | loss  0.27517 | correct 50  | time 1.8422s
Epoch  480 | loss  0.10665 | correct 50  | time 1.7822s
Epoch  490 | loss  0.10806 | correct 50  | time 2.3798s
Epoch  499 | loss  0.19444 | correct 50  | time 1.7794s
Average time per epoch:  1.9411767544746399
```

## Hidden=200 Split Dataset

CPU

Average time per epoch:  0.31794249200820923s
```
Epoch    0 | loss  11.79425 | correct 25  | time 18.7277s
Epoch   10 | loss  2.14989 | correct 42  | time 0.2480s
Epoch   20 | loss  4.44321 | correct 42  | time 0.2464s
Epoch   30 | loss  1.56469 | correct 47  | time 0.2459s
Epoch   40 | loss  1.27526 | correct 49  | time 0.2473s
Epoch   50 | loss  0.98707 | correct 49  | time 0.4937s
Epoch   60 | loss  2.36928 | correct 49  | time 0.2493s
Epoch   70 | loss  0.71757 | correct 50  | time 0.2442s
Epoch   80 | loss  0.82688 | correct 50  | time 0.2625s
Epoch   90 | loss  1.01086 | correct 49  | time 0.2459s
Epoch  100 | loss  0.76525 | correct 50  | time 0.2469s
Epoch  110 | loss  1.04454 | correct 50  | time 0.2487s
Epoch  120 | loss  0.70389 | correct 50  | time 0.2573s
Epoch  130 | loss  0.22254 | correct 50  | time 0.2471s
Epoch  140 | loss  0.23863 | correct 50  | time 0.3538s
Epoch  150 | loss  0.16777 | correct 50  | time 0.2473s
Epoch  160 | loss  0.14595 | correct 50  | time 0.2594s
Epoch  170 | loss  0.35805 | correct 50  | time 0.2472s
Epoch  180 | loss  0.19178 | correct 50  | time 0.4407s
Epoch  190 | loss  0.16255 | correct 50  | time 0.2435s
Epoch  200 | loss  0.20502 | correct 50  | time 0.2541s
Epoch  210 | loss  0.22238 | correct 50  | time 0.2438s
Epoch  220 | loss  0.17258 | correct 50  | time 0.2454s
Epoch  230 | loss  0.18799 | correct 50  | time 0.2485s
Epoch  240 | loss  0.09245 | correct 50  | time 0.2501s
Epoch  250 | loss  0.07614 | correct 50  | time 0.2482s
Epoch  260 | loss  0.17120 | correct 50  | time 0.2439s
Epoch  270 | loss  0.14938 | correct 50  | time 0.2497s
Epoch  280 | loss  0.23633 | correct 50  | time 0.2456s
Epoch  290 | loss  0.08907 | correct 50  | time 0.2483s
Epoch  300 | loss  0.15066 | correct 50  | time 0.2470s
Epoch  310 | loss  0.05263 | correct 50  | time 0.4575s
Epoch  320 | loss  0.04908 | correct 50  | time 0.2596s
Epoch  330 | loss  0.16295 | correct 50  | time 0.2469s
Epoch  340 | loss  0.05431 | correct 50  | time 0.2471s
Epoch  350 | loss  0.05425 | correct 50  | time 0.2446s
Epoch  360 | loss  0.15323 | correct 50  | time 0.2459s
Epoch  370 | loss  0.02593 | correct 50  | time 0.2490s
Epoch  380 | loss  0.09232 | correct 50  | time 0.2504s
Epoch  390 | loss  0.04657 | correct 50  | time 0.2619s
Epoch  400 | loss  0.13165 | correct 50  | time 0.3596s
Epoch  410 | loss  0.04693 | correct 50  | time 0.2460s
Epoch  420 | loss  0.07471 | correct 50  | time 0.2434s
Epoch  430 | loss  0.08519 | correct 50  | time 0.2581s
Epoch  440 | loss  0.07059 | correct 50  | time 0.4328s
Epoch  450 | loss  0.10057 | correct 50  | time 0.2502s
Epoch  460 | loss  0.05337 | correct 50  | time 0.2481s
Epoch  470 | loss  0.01923 | correct 50  | time 0.2594s
Epoch  480 | loss  0.03455 | correct 50  | time 0.2460s
Epoch  490 | loss  0.08967 | correct 50  | time 0.2556s
Epoch  499 | loss  0.04734 | correct 50  | time 0.2435s
Average time per epoch:  0.31794249200820923
```

GPU

Average time per epoch:  2.0138186378479004s
```
Epoch    0 | loss  20.78925 | correct 28  | time 3.7117s
Epoch   10 | loss  5.99725 | correct 43  | time 1.8591s
Epoch   20 | loss  2.87234 | correct 38  | time 2.6932s
Epoch   30 | loss  2.87402 | correct 43  | time 1.8816s
Epoch   40 | loss  1.48890 | correct 46  | time 1.8777s
Epoch   50 | loss  2.76980 | correct 49  | time 2.6349s
Epoch   60 | loss  2.39332 | correct 47  | time 1.9195s
Epoch   70 | loss  1.14462 | correct 48  | time 1.8713s
Epoch   80 | loss  2.21461 | correct 49  | time 2.6263s
Epoch   90 | loss  0.85223 | correct 46  | time 1.8661s
Epoch  100 | loss  1.93042 | correct 49  | time 1.9293s
Epoch  110 | loss  1.50511 | correct 49  | time 2.4802s
Epoch  120 | loss  1.11678 | correct 49  | time 1.8550s
Epoch  130 | loss  0.46788 | correct 49  | time 1.9097s
Epoch  140 | loss  2.30835 | correct 49  | time 2.2412s
Epoch  150 | loss  0.12526 | correct 47  | time 1.8627s
Epoch  160 | loss  0.26340 | correct 49  | time 1.8611s
Epoch  170 | loss  0.39486 | correct 50  | time 2.0271s
Epoch  180 | loss  0.66273 | correct 50  | time 1.8563s
Epoch  190 | loss  0.32530 | correct 50  | time 1.8618s
Epoch  200 | loss  1.70309 | correct 49  | time 2.0675s
Epoch  210 | loss  0.66741 | correct 50  | time 1.9040s
Epoch  220 | loss  1.39959 | correct 50  | time 1.8686s
Epoch  230 | loss  1.78957 | correct 50  | time 1.8416s
Epoch  240 | loss  0.58561 | correct 50  | time 1.9050s
Epoch  250 | loss  1.12848 | correct 49  | time 2.1244s
Epoch  260 | loss  0.93458 | correct 50  | time 1.8383s
Epoch  270 | loss  0.27379 | correct 50  | time 1.8584s
Epoch  280 | loss  1.13929 | correct 50  | time 2.4688s
Epoch  290 | loss  0.32527 | correct 50  | time 1.9210s
Epoch  300 | loss  0.47378 | correct 50  | time 1.8558s
Epoch  310 | loss  0.07349 | correct 50  | time 2.4416s
Epoch  320 | loss  0.20982 | correct 49  | time 1.9040s
Epoch  330 | loss  1.65469 | correct 47  | time 1.8575s
Epoch  340 | loss  0.99634 | correct 50  | time 2.6500s
Epoch  350 | loss  0.02268 | correct 50  | time 1.8301s
Epoch  360 | loss  1.37292 | correct 49  | time 1.9123s
Epoch  370 | loss  0.00383 | correct 50  | time 2.1532s
Epoch  380 | loss  0.29739 | correct 50  | time 1.8543s
Epoch  390 | loss  0.03350 | correct 50  | time 1.8427s
Epoch  400 | loss  0.86462 | correct 50  | time 1.9054s
Epoch  410 | loss  0.44861 | correct 50  | time 1.8692s
Epoch  420 | loss  0.02700 | correct 50  | time 2.0096s
Epoch  430 | loss  0.18889 | correct 50  | time 1.8501s
Epoch  440 | loss  0.71643 | correct 50  | time 1.9152s
Epoch  450 | loss  0.27310 | correct 50  | time 1.9728s
Epoch  460 | loss  0.51851 | correct 50  | time 1.8571s
Epoch  470 | loss  0.38242 | correct 50  | time 1.9049s
Epoch  480 | loss  0.24465 | correct 50  | time 2.4322s
Epoch  490 | loss  0.11029 | correct 50  | time 1.8472s
Epoch  499 | loss  0.56829 | correct 50  | time 1.9262s
Average time per epoch:  2.0138186378479004
```

## Hidden=200 Simple Dataset
CPU

Average time per epoch:  0.31458006954193113s
```
Epoch    0 | loss  3.80867 | correct 43  | time 17.9506s
Epoch   10 | loss  0.49284 | correct 46  | time 0.2485s
Epoch   20 | loss  2.99546 | correct 42  | time 0.2484s
Epoch   30 | loss  1.51356 | correct 49  | time 0.3501s
Epoch   40 | loss  0.83100 | correct 50  | time 0.2469s
Epoch   50 | loss  0.77569 | correct 50  | time 0.2462s
Epoch   60 | loss  0.71066 | correct 47  | time 0.2602s
Epoch   70 | loss  0.25101 | correct 46  | time 0.2443s
Epoch   80 | loss  0.64121 | correct 50  | time 0.2484s
Epoch   90 | loss  0.27030 | correct 50  | time 0.2466s
Epoch  100 | loss  0.08123 | correct 49  | time 0.2457s
Epoch  110 | loss  0.10702 | correct 50  | time 0.2481s
Epoch  120 | loss  0.35000 | correct 50  | time 0.5064s
Epoch  130 | loss  0.61688 | correct 50  | time 0.2490s
Epoch  140 | loss  0.47873 | correct 50  | time 0.2481s
Epoch  150 | loss  0.15310 | correct 50  | time 0.2470s
Epoch  160 | loss  0.64364 | correct 50  | time 0.2537s
Epoch  170 | loss  0.27547 | correct 50  | time 0.2466s
Epoch  180 | loss  0.29054 | correct 50  | time 0.2476s
Epoch  190 | loss  0.24962 | correct 50  | time 0.2461s
Epoch  200 | loss  0.08807 | correct 50  | time 0.2528s
Epoch  210 | loss  0.02058 | correct 50  | time 0.4776s
Epoch  220 | loss  0.49029 | correct 50  | time 0.2470s
Epoch  230 | loss  0.44206 | correct 50  | time 0.2585s
Epoch  240 | loss  0.52742 | correct 50  | time 0.2483s
Epoch  250 | loss  0.25786 | correct 50  | time 0.2882s
Epoch  260 | loss  0.17521 | correct 50  | time 0.2471s
Epoch  270 | loss  0.09398 | correct 50  | time 0.2581s
Epoch  280 | loss  0.00557 | correct 50  | time 0.2499s
Epoch  290 | loss  0.00045 | correct 50  | time 0.2584s
Epoch  300 | loss  0.02093 | correct 50  | time 0.2526s
Epoch  310 | loss  0.11086 | correct 50  | time 0.2534s
Epoch  320 | loss  0.00834 | correct 50  | time 0.2448s
Epoch  330 | loss  0.07812 | correct 50  | time 0.2552s
Epoch  340 | loss  0.11386 | correct 50  | time 0.5144s
Epoch  350 | loss  0.04981 | correct 50  | time 0.2485s
Epoch  360 | loss  0.01292 | correct 50  | time 0.2495s
Epoch  370 | loss  0.28731 | correct 50  | time 0.2479s
Epoch  380 | loss  0.30376 | correct 50  | time 0.2463s
Epoch  390 | loss  0.03117 | correct 50  | time 0.4572s
Epoch  400 | loss  0.27117 | correct 50  | time 0.2440s
Epoch  410 | loss  0.01155 | correct 50  | time 0.2517s
Epoch  420 | loss  0.04942 | correct 50  | time 0.2438s
Epoch  430 | loss  0.03977 | correct 50  | time 0.4187s
Epoch  440 | loss  0.00451 | correct 50  | time 0.2527s
Epoch  450 | loss  0.03067 | correct 50  | time 0.2470s
Epoch  460 | loss  0.17532 | correct 50  | time 0.2545s
Epoch  470 | loss  0.00013 | correct 50  | time 0.4828s
Epoch  480 | loss  0.01875 | correct 50  | time 0.2477s
Epoch  490 | loss  0.20906 | correct 50  | time 0.2438s
Epoch  499 | loss  0.18789 | correct 50  | time 0.2588s
Average time per epoch:  0.31458006954193113
```

GPU:

Average time per epoch:  2.0255869150161745s
```
Epoch    0 | loss  6.38512 | correct 40  | time 3.7583s
Epoch   10 | loss  1.28540 | correct 49  | time 1.8662s
Epoch   20 | loss  0.19066 | correct 48  | time 2.7089s
Epoch   30 | loss  1.45822 | correct 48  | time 1.8693s
Epoch   40 | loss  0.77241 | correct 49  | time 1.8776s
Epoch   50 | loss  0.24327 | correct 50  | time 2.6747s
Epoch   60 | loss  0.56334 | correct 50  | time 1.9569s
Epoch   70 | loss  0.13416 | correct 50  | time 1.8654s
Epoch   80 | loss  0.32247 | correct 50  | time 2.6377s
Epoch   90 | loss  0.73659 | correct 50  | time 1.8891s
Epoch  100 | loss  0.98404 | correct 50  | time 1.9132s
Epoch  110 | loss  0.53956 | correct 50  | time 2.5864s
Epoch  120 | loss  0.07900 | correct 50  | time 1.9061s
Epoch  130 | loss  0.82126 | correct 50  | time 1.9061s
Epoch  140 | loss  0.02416 | correct 50  | time 2.4712s
Epoch  150 | loss  0.28826 | correct 50  | time 1.8711s
Epoch  160 | loss  0.52444 | correct 50  | time 1.8855s
Epoch  170 | loss  0.03322 | correct 50  | time 2.4162s
Epoch  180 | loss  0.38634 | correct 50  | time 1.8513s
Epoch  190 | loss  0.43058 | correct 50  | time 1.8524s
Epoch  200 | loss  0.09172 | correct 50  | time 2.6566s
Epoch  210 | loss  0.39712 | correct 50  | time 1.9177s
Epoch  220 | loss  0.35487 | correct 50  | time 1.8657s
Epoch  230 | loss  0.00345 | correct 50  | time 2.3916s
Epoch  240 | loss  0.07649 | correct 50  | time 1.8490s
Epoch  250 | loss  0.22833 | correct 50  | time 1.9294s
Epoch  260 | loss  0.07042 | correct 50  | time 2.0832s
Epoch  270 | loss  0.06536 | correct 50  | time 1.8689s
Epoch  280 | loss  0.00806 | correct 50  | time 1.8663s
Epoch  290 | loss  0.03468 | correct 50  | time 2.0724s
Epoch  300 | loss  0.06076 | correct 50  | time 1.8643s
Epoch  310 | loss  0.16436 | correct 50  | time 1.8804s
Epoch  320 | loss  0.07110 | correct 50  | time 2.0128s
Epoch  330 | loss  0.02050 | correct 50  | time 1.8578s
Epoch  340 | loss  0.01842 | correct 50  | time 1.8795s
Epoch  350 | loss  0.39384 | correct 50  | time 1.8474s
Epoch  360 | loss  0.02172 | correct 50  | time 1.9306s
Epoch  370 | loss  0.01883 | correct 50  | time 2.0331s
Epoch  380 | loss  0.05427 | correct 50  | time 1.8587s
Epoch  390 | loss  0.27633 | correct 50  | time 1.8614s
Epoch  400 | loss  0.04692 | correct 50  | time 2.3383s
Epoch  410 | loss  0.00915 | correct 50  | time 1.8670s
Epoch  420 | loss  0.13713 | correct 50  | time 1.8601s
Epoch  430 | loss  0.17931 | correct 50  | time 3.2032s
Epoch  440 | loss  0.01254 | correct 50  | time 1.9256s
Epoch  450 | loss  0.11034 | correct 50  | time 1.8643s
Epoch  460 | loss  0.00074 | correct 50  | time 2.5021s
Epoch  470 | loss  0.00166 | correct 50  | time 1.9203s
Epoch  480 | loss  0.14157 | correct 50  | time 1.8626s
Epoch  490 | loss  0.20752 | correct 50  | time 2.2687s
Epoch  499 | loss  0.15406 | correct 50  | time 1.9229s
Average time per epoch:  2.0255869150161745
```

## Hidden=200 XOR Dataset

CPU

Average time per epoch:  0.3137641406059265s
```
Epoch    0 | loss  13.90547 | correct 30  | time 17.9027s
Epoch   10 | loss  5.93472 | correct 40  | time 0.2497s
Epoch   20 | loss  4.10273 | correct 44  | time 0.2449s
Epoch   30 | loss  3.56813 | correct 42  | time 0.2474s
Epoch   40 | loss  2.17285 | correct 47  | time 0.3994s
Epoch   50 | loss  2.45942 | correct 43  | time 0.2567s
Epoch   60 | loss  1.86029 | correct 48  | time 0.2463s
Epoch   70 | loss  1.67957 | correct 48  | time 0.2464s
Epoch   80 | loss  2.06625 | correct 48  | time 0.2505s
Epoch   90 | loss  1.82988 | correct 48  | time 0.2450s
Epoch  100 | loss  1.90738 | correct 48  | time 0.2491s
Epoch  110 | loss  2.01860 | correct 48  | time 0.2480s
Epoch  120 | loss  1.96516 | correct 49  | time 0.2666s
Epoch  130 | loss  0.96397 | correct 50  | time 0.4848s
Epoch  140 | loss  1.01789 | correct 47  | time 0.2614s
Epoch  150 | loss  2.01243 | correct 49  | time 0.2480s
Epoch  160 | loss  1.51150 | correct 50  | time 0.2624s
Epoch  170 | loss  1.20326 | correct 49  | time 0.2493s
Epoch  180 | loss  0.45157 | correct 50  | time 0.2496s
Epoch  190 | loss  2.58662 | correct 49  | time 0.2784s
Epoch  200 | loss  2.26614 | correct 49  | time 0.2560s
Epoch  210 | loss  1.76831 | correct 49  | time 0.2460s
Epoch  220 | loss  1.00195 | correct 50  | time 0.2514s
Epoch  230 | loss  1.57658 | correct 48  | time 0.2479s
Epoch  240 | loss  2.00287 | correct 46  | time 0.2554s
Epoch  250 | loss  1.07269 | correct 50  | time 0.2472s
Epoch  260 | loss  2.64392 | correct 45  | time 0.5531s
Epoch  270 | loss  0.21971 | correct 49  | time 0.2738s
Epoch  280 | loss  1.69598 | correct 49  | time 0.2449s
Epoch  290 | loss  0.23147 | correct 50  | time 0.2526s
Epoch  300 | loss  0.91488 | correct 49  | time 0.2453s
Epoch  310 | loss  0.96923 | correct 50  | time 0.2464s
Epoch  320 | loss  0.11865 | correct 48  | time 0.2468s
Epoch  330 | loss  0.98320 | correct 47  | time 0.2458s
Epoch  340 | loss  1.44712 | correct 47  | time 0.2478s
Epoch  350 | loss  0.22217 | correct 50  | time 0.3535s
Epoch  360 | loss  0.44098 | correct 50  | time 0.2528s
Epoch  370 | loss  1.92970 | correct 48  | time 0.2470s
Epoch  380 | loss  0.93146 | correct 48  | time 0.2508s
Epoch  390 | loss  0.09513 | correct 50  | time 0.5375s
Epoch  400 | loss  0.84990 | correct 50  | time 0.2488s
Epoch  410 | loss  0.79305 | correct 50  | time 0.2498s
Epoch  420 | loss  1.40367 | correct 49  | time 0.2463s
Epoch  430 | loss  1.23655 | correct 50  | time 0.2497s
Epoch  440 | loss  0.66894 | correct 49  | time 0.2480s
Epoch  450 | loss  1.19610 | correct 50  | time 0.2484s
Epoch  460 | loss  0.32076 | correct 50  | time 0.2491s
Epoch  470 | loss  0.47728 | correct 48  | time 0.2467s
Epoch  480 | loss  0.08787 | correct 50  | time 0.5060s
Epoch  490 | loss  1.11984 | correct 48  | time 0.2559s
Epoch  499 | loss  1.02846 | correct 49  | time 0.2574s
Average time per epoch:  0.3137641406059265
```

GPU

Average time per epoch:  2.0086645045280456s
```
Epoch    0 | loss  5.47798 | correct 28  | time 3.8426s
Epoch   10 | loss  8.57031 | correct 25  | time 1.8452s
Epoch   20 | loss  1.93777 | correct 44  | time 1.9059s
Epoch   30 | loss  2.56951 | correct 44  | time 1.8220s
Epoch   40 | loss  1.97871 | correct 44  | time 1.8767s
Epoch   50 | loss  2.96338 | correct 47  | time 2.0396s
Epoch   60 | loss  2.41812 | correct 49  | time 1.9049s
Epoch   70 | loss  1.99626 | correct 48  | time 1.8377s
Epoch   80 | loss  1.86260 | correct 48  | time 2.6389s
Epoch   90 | loss  0.43920 | correct 48  | time 1.8599s
Epoch  100 | loss  0.37424 | correct 48  | time 1.9138s
Epoch  110 | loss  1.59454 | correct 47  | time 2.1528s
Epoch  120 | loss  0.94759 | correct 50  | time 1.8335s
Epoch  130 | loss  1.29688 | correct 48  | time 1.9077s
Epoch  140 | loss  0.75288 | correct 50  | time 1.9142s
Epoch  150 | loss  1.10412 | correct 47  | time 1.8465s
Epoch  160 | loss  0.33069 | correct 50  | time 1.9111s
Epoch  170 | loss  0.20836 | correct 50  | time 1.9221s
Epoch  180 | loss  0.31590 | correct 50  | time 1.8362s
Epoch  190 | loss  0.28807 | correct 48  | time 2.5004s
Epoch  200 | loss  0.79165 | correct 50  | time 1.8460s
Epoch  210 | loss  0.27076 | correct 50  | time 1.9316s
Epoch  220 | loss  1.27568 | correct 50  | time 2.5036s
Epoch  230 | loss  0.68431 | correct 48  | time 1.8338s
Epoch  240 | loss  0.77095 | correct 50  | time 1.8326s
Epoch  250 | loss  0.51764 | correct 49  | time 1.9764s
Epoch  260 | loss  0.08894 | correct 46  | time 1.8436s
Epoch  270 | loss  0.25974 | correct 50  | time 1.8460s
Epoch  280 | loss  0.29032 | correct 50  | time 1.8617s
Epoch  290 | loss  0.25563 | correct 49  | time 1.9429s
Epoch  300 | loss  0.25479 | correct 50  | time 2.2516s
Epoch  310 | loss  0.33388 | correct 50  | time 2.1118s
Epoch  320 | loss  0.33404 | correct 50  | time 1.9075s
Epoch  330 | loss  0.78554 | correct 50  | time 2.4213s
Epoch  340 | loss  1.48816 | correct 48  | time 1.8504s
Epoch  350 | loss  0.19626 | correct 50  | time 1.8541s
Epoch  360 | loss  0.08202 | correct 50  | time 2.7037s
Epoch  370 | loss  0.17508 | correct 50  | time 1.8356s
Epoch  380 | loss  0.37687 | correct 50  | time 1.8456s
Epoch  390 | loss  0.23081 | correct 50  | time 2.1501s
Epoch  400 | loss  0.55709 | correct 50  | time 1.9281s
Epoch  410 | loss  0.89846 | correct 50  | time 1.9019s
Epoch  420 | loss  0.20685 | correct 50  | time 1.8469s
Epoch  430 | loss  0.05449 | correct 50  | time 1.8404s
Epoch  440 | loss  1.11768 | correct 50  | time 1.9124s
Epoch  450 | loss  0.21264 | correct 50  | time 1.8803s
Epoch  460 | loss  0.06688 | correct 50  | time 1.8576s
Epoch  470 | loss  0.41257 | correct 50  | time 2.2116s
Epoch  480 | loss  0.12014 | correct 50  | time 1.8625s
Epoch  490 | loss  0.03013 | correct 50  | time 1.8455s
Epoch  499 | loss  0.19986 | correct 50  | time 1.9334s
Average time per epoch:  2.0086645045280456
```