
// kernel code function to add two arrays (second one reverse)
__kernel void VecAdd(__global const int* a, __global const int* b,
                        __global int* c, int numElements)
{
    int GID = get_global_id(0);
    if (GID < numElements)
    {
        // GID is lower then number of elements,
        // so to integers need to be added.
        c[GID] = a[GID] + b[numElements - GID - 1];
    }
}
