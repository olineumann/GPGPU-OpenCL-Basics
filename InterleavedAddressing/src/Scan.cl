


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__kernel void Scan_Naive(const __global uint* inArray, __global uint* outArray, uint N, uint offset)
{
	// get global id
	int id = get_global_id(0);

	unsigned int sum;

	// if global id is within array size, worker needs to set value on output array
	if (id < N)
	{
		sum = inArray[id];
	}
	// if global id - offset is within array size, worker needs to add value to sum
	if (id >= offset)
	{
		sum += inArray[id - offset];
	}

	// write sum back to output array
	outArray[id] = sum;
}

// Why did we not have conflicts in the Reduction? Because of the sequential addressing (here we use interleaved => we have conflicts).

#define UNROLL
#define NUM_BANKS			32
#define NUM_BANKS_LOG		5
#define SIMD_GROUP_SIZE		32

// Bank conflicts
#define AVOID_BANK_CONFLICTS
#ifdef AVOID_BANK_CONFLICTS
	// TO DO: define your conflict-free macro here
#else
	#define OFFSET(A) (A)
#endif

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__kernel void Scan_WorkEfficient(__global uint* array, __global uint* higherLevelArray, __local uint* localBlock)
{
	// get local id, size and position
	unsigned int id = get_local_id(0);
	unsigned int size = get_local_size(0);
	unsigned int pos = get_group_id(0) * size * 2 + id;

	// load data into local memory sequential for better performance
	localBlock[id] = array[pos];
	localBlock[id + size] = array[pos + size];

	// wait for local writes are done
	barrier(CLK_LOCAL_MEM_FENCE);

	// perfom up sweep on local array
	for (unsigned int step = 2; step <= size * 2; step = step * 2)
	{
		// check whether worker has something to do
		if (id < (size * 2) / step)
		{
			// perform up sweep step
			unsigned int index = (id + 1) * step - 1;
			localBlock[index] += localBlock[index - step / 2];

		}

		// wait for sweep step completed
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	// set last element to zero for exclusive prefix sum
	if (id == 0)
	{
		localBlock[size * 2 - 1] = 0;
	}

	// wait for write of last element done
	barrier(CLK_LOCAL_MEM_FENCE);

	// perform down sweep
	for (unsigned int step = size * 2; step > 1; step = step / 2)
	{
			// check whether worker has something to do
			if (id < (size * 2) / step)
			{
				// perform single down sweep step
				unsigned int index = (id + 1) * step - 1;
				unsigned int left_value = localBlock[index - step / 2];
				unsigned int right_value = localBlock[index];
				// left child
				localBlock[index - step / 2] = right_value;
				// right child
				localBlock[index] = left_value + right_value;
			}

			// wait for sweep step completed;
			barrier(CLK_LOCAL_MEM_FENCE);
	}

	// read global sequential array and add to local array for inclusive prefix sum
	localBlock[id] += array[pos];
	localBlock[id + size] += array[pos + size];
	barrier(CLK_LOCAL_MEM_FENCE);

	// write result back sequential
	array[pos] = localBlock[id];
	array[pos + size] = localBlock[id + size];

	// write group sum in higher level array for later composition of result
	if (id == 0) {
		higherLevelArray[get_group_id(0)] = localBlock[size * 2 - 1];
	}
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__kernel void Scan_WorkEfficientAdd(__global uint* higherLevelArray, __global uint* array, __local uint* localBlock)
{
	// Kernel that should add the group PPS to the local PPS (Figure 14)
	// set local and group id and calculate position to read and write from
	unsigned int id = get_local_id(0);
	unsigned int group = get_group_id(0);
	unsigned int pos = id + (group + 2) * get_local_size(0);

	// load group part of global array into local memory
	localBlock[id] = array[pos];

	// add group sum to local array
	localBlock[id] += higherLevelArray[group/2];

	// write result to global array
	array[pos] = localBlock[id];
}
