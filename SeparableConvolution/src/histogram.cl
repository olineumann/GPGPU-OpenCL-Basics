
__kernel void
set_array_to_constant(
	__global int *array,
	int num_elements,
	int val
)
{
	// There is no need to touch this kernel
	if(get_global_id(0) < num_elements)
		array[get_global_id(0)] = val;
}

__kernel void
compute_histogram(
	__global int *histogram,   // accumulate histogram here
	__global const float *img, // input image
	int width,                 // image width
	int height,                // image height
	int pitch,                 // image pitch
	int num_hist_bins          // number of histogram bins
)
{
	// get pixel position to calculate histogram data from
	// for that global id in x and y direction is used
	int x = get_global_id(0);
	int y = get_global_id(1);

	// check whether there is something to to for worker
	if (x < width && y < height)
	{
		// x, y in image dimensions -> need to add histogram data
		int value = img[y * pitch + x] * num_hist_bins;
		value = min(num_hist_bins - 1, max(0, value));
		atomic_add(&histogram[value], 1);
	}

}

__kernel void
compute_histogram_local_memory(
	__global int *histogram,   // accumulate histogram here
	__global const float *img, // input image
	int width,                 // image width
	int height,                // image height
	int pitch,                 // image pitch
	int num_hist_bins,         // number of histogram bins
	__local int *local_hist
)
{
	// get pixel position to calculate histogram data in local memory
	int x = get_global_id(0);
	int y = get_global_id(1);

	// calculate an id of worker by using local ids to initialise
	// local histogram and write it back later
	int id = get_local_id(0) + get_local_id(1) * get_local_size(0);
	int size = get_local_size(0) * get_local_size(1);

	// read image value
	int value = -1;
	if (x < width && y < height)
	{
		value = img[y * pitch + x] * num_hist_bins;
		value = min(num_hist_bins - 1, max(0, value));
	}

	// initialise local histogram
	// using for loop because local work size
	// could be lower than number of histogram bins
	for (int i=id; i < num_hist_bins; i = i + size)
	{
		local_hist[i] = 0;
	}

	// wait for local histogram initialised
	barrier(CLK_LOCAL_MEM_FENCE);

	// increment local histogram
	if (value >= 0)
	{
		atomic_add(&local_hist[value], 1);
	}

	// wait that all histogram data is written to local memory
	barrier(CLK_LOCAL_MEM_FENCE);


	// add local histogram to global by writing it back
	// using for loop because local work size
	// could be lower than number of histogram bins
	for (int i=id; i < num_hist_bins; i = i + size)
	{
		atomic_add(&histogram[i], local_hist[i]);
	}
}
