/*
We assume a 3x3 (radius: 1) convolution kernel, which is not separable.
Each work-group will process a (TILE_X x TILE_Y) tile of the image.
For coalescing, TILE_X should be multiple of 16.

Instead of examining the image border for each kernel, we recommend to pad the image
to be the multiple of the given tile-size.
*/

//should be multiple of 32 on Fermi and 16 on pre-Fermi...
#define TILE_X 32

#define TILE_Y 16
// d_Dst is the convolution of d_Src with the kernel c_Kernel
// c_Kernel is assumed to be a float[11] array of the 3x3 convolution constants, one multiplier (for normalization) and an offset (in this order!)
// With & Height are the image dimensions (should be multiple of the tile size)
__kernel __attribute__((reqd_work_group_size(TILE_X, TILE_Y, 1)))
void Convolution(
				__global float* d_Dst,
				__global const float* d_Src,
				__constant float* c_Kernel,
				uint Width,  // Use width to check for image bounds
				uint Height,
				uint Pitch   // Use pitch for offsetting between lines
				)
{
	// OpenCL allows to allocate the local memory from 'inside' the kernel (without using the clSetKernelArg() call)
	// in a similar way to standard C.
	// the size of the local memory necessary for the convolution is the tile size + the halo area
	__local float tile[TILE_Y + 2][TILE_X + 2];
	const int2 pos = {
		pos.x = get_global_id(0),
		pos.y = get_global_id(1)
	};

	const int2 id = {
		id.x = get_local_id(0),
		id.y = get_local_id(1)
	};

	const int2 reference_point = {
		reference_point.x = get_group_id(0) * TILE_X,
		reference_point.y = get_group_id(1) * TILE_Y
	};
	
	// numerate worker like:
	// 0, 							1, 			..., 			TILE_X + 1
	// TILE_X + 2, 							..., 			2*TILE_X + 3 = TILE_X + 2 + TILE_X + 1
	// n*(TILE_X + 2), 					..., 			(n+1)*TILE_X + 2*n + 1 = n*(TILE_X + 2) + TILE_X + 1

	// initialise local number of worker items
	int n = id.x + id.y * (TILE_X);
	if (n < TILE_X + 2)
	{
		// wirte first top bar of halo area
		int x = reference_point.x - 1 + n;
		int y = reference_point.y - 1;

		if (y >= 0 && x >= 0 && x < Width)
		{
			// halo area within image dimensions
			tile[0][n] = d_Src[y * Pitch + x];
		}
		else
		{
			// halo area outside image dimensions
			tile[0][n] = 0;
		}
	}
	else if (n >= TILE_X + 2 && n < TILE_X + 2 + TILE_Y * 2)
	{
		// write sides of halo area
		if ((n - TILE_X - 2) % 2 == 0)
		{
			// left side
			int x = reference_point.x - 1;
			int y = reference_point.y + (n - TILE_X - 2) / 2;

			if (x > 0 && y < Height)
			{
				// halo area within image dimensions
				tile[(n - TILE_X - 2) / 2 + 1][0] = d_Src[x + y * Pitch];
			}
			else
			{
				// halo area outside image dimensions
				tile[(n - TILE_X - 2) / 2 + 1][0] = 0;
			}
		}
		else
		{
			// right side
			int x = reference_point.x + TILE_X;
			int y = reference_point.y + (n - TILE_X - 2) / 2;

			if (x < Width && y < Height)
			{
				// halo area within image dimensions
				tile[(n - TILE_X - 2) / 2 + 1][TILE_X + 1] = d_Src[x + y * Pitch];
			}
			else
			{
				// halo area outside image dimensions
				tile[(n - TILE_X - 2) / 2 + 1][TILE_X + 1] = 0;
			}
		}
	}
	else if (n >= TILE_X + 2 + TILE_Y * 2 && n < TILE_X + 2 + TILE_Y * 2 + TILE_X + 2)
	{
		// bottom part of halo areaoutside
		int x = reference_point.x + n - (TILE_X + 2 + TILE_Y * 2) - 1;
		int y = reference_point.y + TILE_Y;


		if (y < Height && x < Width)
		{
			// halo area within image dimensions
			tile[TILE_Y + 1][n - (TILE_X + 2 + TILE_Y * 2)] = d_Src[x + y * Pitch];
		}
		else
		{
			// halo area outside image dimensions
			tile[TILE_Y + 1][n - (TILE_X + 2 + TILE_Y * 2)] = 0;
		}
	}

	// write tile image into tile or otherwise 0 if out of image bounds
	if (pos.x < Width && pos.y < Height)
	{
		// tile area inside image dimensions
		tile[id.y + 1][id.x + 1] = d_Src[pos.y * Pitch + pos.x];
	}
	else
	{
		// tile area outside image dimensions
		tile[id.y + 1][id.x + 1] = 0;
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	// calculate convolution value
	double value = 0.0;
	for(int offsetY = 0; offsetY < 3; offsetY++)
	{
		int y = id.y + offsetY;
		for(int offsetX = 0; offsetX < 3; offsetX++)
		{
			int x = id.x + offsetX;
			value += tile[y][x] * c_Kernel[offsetX + offsetY * 3];
		}
	}

	// write result back to d_Dst
	if (pos.x < Width && pos.y < Height)
	{
		d_Dst[pos.y * Pitch + pos.x] = value * c_Kernel[9] + c_Kernel[10];
	}
}
