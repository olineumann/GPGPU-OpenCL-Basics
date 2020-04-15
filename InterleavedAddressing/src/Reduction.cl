
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__kernel void Reduction_InterleavedAddressing(__global uint* array, uint stride)
{
	// calculate position of worker
	unsigned int pos = get_global_id(0) * stride * 2;

	// increment array at position with position plus stride
	array[pos] = array[pos] + array[pos + stride];
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__kernel void Reduction_SequentialAddressing(__global uint* array, uint stride)
{
	// get global id
	unsigned int id = get_global_id(0);

	// increment array at global id with global id plus stride
	array[id] = array[id] + array[id + stride];
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__kernel void Reduction_Decomp(const __global uint* inArray, __global uint* outArray, uint N, __local uint* localBlock)
{
	// get group and local id
	unsigned int id = get_local_id(0);
	unsigned int group_id = get_group_id(0);

	// first reduction step with sequential adressing
	localBlock[id] = inArray[group_id * N * 2 + id] + inArray[group_id * N * 2 + N + id];

	// further reduction on local memory using interleaved adressing
	for (unsigned int stride = 1; stride < N; stride = stride * 2)
	{
		barrier(CLK_LOCAL_MEM_FENCE);
		if (id * stride * 2 + stride < N)
		{
			int sum = localBlock[id * stride * 2] + localBlock[id * stride * 2 + stride];
			localBlock[id * stride * 2] = sum;
		}
	}

	// write result back to global memory
	if (id == 0) {
		outArray[group_id] = localBlock[0];
	}
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__kernel void Reduction_DecompUnroll(const __global uint* inArray, __global uint* outArray, uint N, __local uint* localBlock)
{
	// not needed in GPGPU
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__kernel void Reduction_DecompAtomics(const __global uint* inArray, __global uint* outArray, uint N, __local uint* localSum)
{
	// get group and local id
	unsigned int id = get_local_id(0);
	unsigned int group_id = get_group_id(0);

	// calculate first reduce step in sequential way because we need to read from global memory
	unsigned int sum = inArray[group_id * N * 2 + id] + inArray[group_id * N * 2 + N + id];

	// set localSum[0] to zero because it is not initialised
	if (id == 0)
	{
		localSum[0] = 0;
	}

	// wait that all workers have localSum[0] equals zero
	barrier(CLK_LOCAL_MEM_FENCE);

	// perform next reduce steps with atomic_add
	atomic_add(localSum, sum);

	// wait for all atomic_add
	barrier(CLK_LOCAL_MEM_FENCE);

	// write result back to global memory
	if (id == 0)
	{
		outArray[group_id] = localSum[0];
	}
}
