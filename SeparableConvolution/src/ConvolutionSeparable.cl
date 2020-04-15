//Each thread load exactly one halo pixel
//Thus, we assume that the halo size is not larger than the
//dimension of the work-group in the direction of the kernel

//to efficiently reduce the memory transfer overhead of the global memory
// (each pixel is lodaded multiple times at high overlaps)
// one work-item will compute RESULT_STEPS pixels

//for unrolling loops, these values have to be known at compile time

/* These macros will be defined dynamically during building the program

#define KERNEL_RADIUS 2

//horizontal kernel
#define H_GROUPSIZE_X  32
#define H_GROUPSIZE_Y  4
#define H_RESULT_STEPS  2

//vertical kernel
#define V_GROUPSIZE_X  32
#define V_GROUPSIZE_Y  16
#define V_RESULT_STEPS  3

*/

#define KERNEL_LENGTH (2 * KERNEL_RADIUS + 1)

//////////////////////////////////////////////////////////////////////////////////////////////////////
// Horizontal convolution filter

/*
c_Kernel stores 2 * KERNEL_RADIUS + 1 weights, use these during the convolution
*/

//require matching work-group size
__kernel __attribute__((reqd_work_group_size(H_GROUPSIZE_X, H_GROUPSIZE_Y, 1)))
void ConvHorizontal(
   __global float* d_Dst,
   __global const float* d_Src,
   __constant float* c_Kernel,
   int Width,
   int Pitch
   )
{
  //The size of the local memory: one value for each work-item.
  //We even load unused pixels to the halo area, to keep the code and local memory access simple.
  //Since these loads are coalesced, they introduce no overhead, except for slightly redundant local memory allocation.
  //Each work-item loads H_RESULT_STEPS values + 2 halo values
  __local float tile[H_GROUPSIZE_Y][(H_RESULT_STEPS + 2) * H_GROUPSIZE_X];

  // initialise global position which is first pixel to convolve
  const int posX = get_group_id(0) * H_RESULT_STEPS * H_GROUPSIZE_X + get_local_id(0);
  const int posY = get_global_id(1);

  // initialise local base x and y used to work on tile
  const int baseX = get_local_id(0) + H_GROUPSIZE_X;
  const int baseY = get_local_id(1);
  const int offset = H_GROUPSIZE_X;

  // load halo region and image pixels
  for (int i = -offset; baseX + i < (H_RESULT_STEPS + 2) * offset; i += offset)
  {
    // in my opinion check for IMAGE_HEIGHT is also needed (e.g. work size 16x16 image size 17x17)
    if (posX + i > 0  && posX + i < Width)
    {
     tile[baseY][baseX + i] = d_Src[posX + i + posY * Pitch];
    }
    else
    {
     tile[baseY][baseX + i] = 0;
    }
  }

  // sync after local memory writes
  barrier(CLK_LOCAL_MEM_FENCE);

  // convolve pixels
  for (int i = 0; baseX + i < (H_RESULT_STEPS + 1) * offset; i += offset)
  {
   double value = 0.0;
   for (int r = -KERNEL_RADIUS; r <= KERNEL_RADIUS; r++)
   {
     value += tile[baseY][baseX + i + r] * c_Kernel[KERNEL_RADIUS - r];
   }

   // write result back
   // in my opinion check for IMAGE_HEIGHT is also needed (e.g. work size 16x16 image size 17x17)
   if (posX + i < Width)
   {
     d_Dst[posX + i + posY * Pitch] = value;
   }
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////
// Vertical convolution filter

//require matching work-group size
__kernel __attribute__((reqd_work_group_size(V_GROUPSIZE_X, V_GROUPSIZE_Y, 1)))
void ConvVertical(
   __global float* d_Dst,
   __global const float* d_Src,
   __constant float* c_Kernel,
   int Height,
   int Pitch
   )
{
  __local float tile[(V_RESULT_STEPS + 2) * V_GROUPSIZE_Y][V_GROUPSIZE_X];

  const int posX = get_global_id(0);
  const int posY = get_group_id(1) * V_RESULT_STEPS * V_GROUPSIZE_Y + get_local_id(1);

  // initialise local base x and y used to work on tile
  const int baseX = get_local_id(0);
  const int baseY = get_local_id(1) + V_GROUPSIZE_Y;
  const int offset = V_GROUPSIZE_Y;

  // load halo region and image pixels
  for (int i = -offset; baseY + i < (V_RESULT_STEPS + 2) * offset; i += offset)
  {
    // in my opinion check for IMAGE_WIDTH is also needed (e.g. work size 16x16 image size 20x20)
    if (posY + i > 0  && posY + i < Height)
    {
      tile[baseY + i][baseX] = d_Src[posX + (posY + i) * Pitch];
    }
    else
    {
      tile[baseY + i][baseX] = 0;
    }
  }

  // wait for all writes to local memory are done
  barrier(CLK_LOCAL_MEM_FENCE);

  // convolve pixels and write results back
  for (int i = 0; baseY + i < (V_RESULT_STEPS + 1) * offset; i += offset)
  {
    float value = 0.0;
    for (int r = -KERNEL_RADIUS; r <= KERNEL_RADIUS; r++)
    {
      value += tile[baseY + i + r][baseX] * c_Kernel[KERNEL_RADIUS - r];
    }

    // write result back
    // in my opinion check for IMAGE_WIDTH is also needed (e.g. work size 16x16 image size 17x17)
    if (posY + i < Height)
    {
      d_Dst[posX + (posY + i) * Pitch] = value;
    }
  }
}
