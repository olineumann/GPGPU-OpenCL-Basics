
// Rotate the matrix CLOCKWISE

// Naive implementation: move the elements of the matrix directly to their destinations
// this will cause unaligned memory accessed which - as we will see - should be avoided on the GPU

__kernel void MatrixRotNaive(__global const float* M, __global float* MR, uint SizeX, uint SizeY)
{
    // Get 2D global index
	int2 GID;
	GID.x = get_global_id(0);
	GID.y = get_global_id(1);

	// Check if worker needs to rotate item
	if (GID.x < SizeX && GID.y < SizeY)
	{
        // Rotate item of matrix
        MR[GID.x * SizeY + (SizeY - GID.y - 1)] = M[GID.y * SizeX + GID.x];
	}
}

// This kernel does the same thing, however, the local memory is used to
// transform a small chunk of the matrix locally
// then write it back after synchronization in a coalesced access pattern

__kernel void MatrixRotOptimized(__global const float* M, __global float* MR,
				uint SizeX, uint SizeY,	__local float* block)
{
    // Get 2D local and global index
	int2 GID;
	GID.x = get_global_id(0);
	GID.y = get_global_id(1);

	int2 LID;
	LID.x = get_local_id(0);
	LID.y = get_local_id(1);

	// Load block into shared memory
	if (GID.x < SizeX && GID.y < SizeY)
	{
		block[LID.y * get_local_size(0) + LID.x] = M[GID.y * SizeX + GID.x];
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	// Check if worker needs to rotate item
	if (LID.y == 0 && GID.x < SizeX)
	{
		for (int i = get_local_size(1)-1; i >= 0; i--)
		{
				MR[GID.x * SizeY + (SizeY - GID.y - i - 1)] = block[i * get_local_size(0) + LID.x];
		}
	}
}

__kernel void MatrixRotOptimized_all(__global const float* M, __global float* MR,
				uint SizeX, uint SizeY,	__local float* block)
{
    // Get 2D local and global index
	int2 GID;
	GID.x = get_global_id(0);
	GID.y = get_global_id(1);

	int2 LID;
	LID.x = get_local_id(0);
	LID.y = get_local_id(1);

	// Load block into shared memory
	if (GID.x < SizeX && GID.y < SizeY)
	{
		block[LID.y * get_local_size(0) + LID.x] = M[GID.y * SizeX + GID.x];
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	// Check if worker needs to rotate item
	if (GID.x < SizeX && GID.y < SizeY)
	{
			MR[GID.x * SizeY + (SizeY - GID.y - 1)] = block[LID.y * get_local_size(0) + LID.x];
	}
}

__kernel void MatrixRotOptimized_x(__global const float* M, __global float* MR,
				uint SizeX, uint SizeY,	__local float* block)
{
    // Get 2D local and global index
	int2 GID;
	GID.x = get_global_id(0);
	GID.y = get_global_id(1);

	int2 LID;
	LID.x = get_local_id(0);
	LID.y = get_local_id(1);

	// Load block into shared memory
	if (GID.x < SizeX && GID.y < SizeY)
	{
		block[LID.y * get_local_size(0) + LID.x] = M[GID.y * SizeX + GID.x];
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	// Check if worker needs to rotate item
	if (LID.y == 0)
	{
		for (int i = get_local_size(1)-1; i >= 0; i--)
		{
			MR[GID.x * SizeY + (SizeY - GID.y - i - 1)] = block[i * get_local_size(0) + LID.x];
		}
	}
}

__kernel void MatrixRotOptimized_y(__global const float* M, __global float* MR,
				uint SizeX, uint SizeY,	__local float* block)
{
    // Get 2D local and global index
	int2 GID;
	GID.x = get_global_id(0);
	GID.y = get_global_id(1);

	int2 LID;
	LID.x = get_local_id(0);
	LID.y = get_local_id(1);

	// Load block into shared memory
	if (GID.x < SizeX && GID.y < SizeY)
	{
		block[LID.y * get_local_size(0) + LID.x] = M[GID.y * SizeX + GID.x];
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	// Check if worker needs to rotate item
	if (LID.x == 0 && GID.y < SizeY)
	{
		for (int i = 0; i < get_local_size(0); i++)
		{
			MR[(GID.x+i) * SizeY + (SizeY - GID.y - 1)] = block[LID.y * get_local_size(0) + i];
		}
	}
}
