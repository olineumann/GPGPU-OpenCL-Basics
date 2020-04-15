

float4 cross3(float4 a, float4 b) {
	float4 c;
	c.x = a.y * b.z - b.y * a.z;
	c.y = a.z * b.x - b.z * a.x;
	c.z = a.x * b.y - b.x * a.y;
	c.w = 0.f;
	return c;
}

float dot3(float4 a, float4 b) {
	return a.x*b.x + a.y*b.y + a.z*b.z;
}


#define EPSILON 0.001f

#define LEFT_EYE (float4)(0.5f - 0.17f, 0.66f, 0.5f - 0.21f, 100.0f)
#define RIGHT_EYE (float4)(0.5f - 0.05f, 0.66f, 0.5f - 0.24f, 100.0f)
#define MOUTH (float4)(0.5f - 0.11f, 0.48f, 0.5f - 0.21f, 100.0f)
#define MONKEY_NORMAL normalize((float4)(-0.6f, 0.4f, -1.2f, 0.0f))

// This function expects two points defining a ray (x0 and x1)
// and three vertices stored in v1, v2, and v3 (the last component is not used)
// it returns true if an intersection is found and sets the isectT and isectN
// with the intersection ray parameter and the normal at the intersection point.
bool LineTriangleIntersection(float4 x0, float4 x1,
															float4 v1, float4 v2, float4 v3,
															float *isectT, float4 *isectN) {

	float4 dir = x1 - x0;
	dir.w = 0.f;

	float4 e1 = v2 - v1;
	float4 e2 = v3 - v1;
	e1.w = 0.f;
	e2.w = 0.f;

	float4 s1 = cross3(dir, e2);
	float divisor = dot3(s1, e1);
	if (divisor == 0.f)
		return false;
	float invDivisor = 1.f / divisor;

	// Compute first barycentric coordinate
	float4 d = x0 - v1;
	float b1 = dot3(d, s1) * invDivisor;
	if (b1 < -EPSILON || b1 > 1.f + EPSILON)
		return false;

	// Compute second barycentric coordinate
	float4 s2 = cross3(d, e1);
	float b2 = dot3(dir, s2) * invDivisor;
	if (b2 < -EPSILON || b1 + b2 > 1.f + EPSILON)
		return false;

	// Compute _t_ to intersection point
	float t = dot3(e2, s2) * invDivisor;
	if (t < -EPSILON || t > 1.f + EPSILON)
		return false;

	// Store the closest found intersection so far
	*isectT = t;
	*isectN = cross3(e1, e2);
	*isectN = normalize(*isectN);
	return true;

}


bool CheckCollisions(	float4 x0, float4 x1,
											__global float4 *gTriangleSoup,
											__local float4* lTriangleCache,	// The cache should hold as many vertices as the number of threads (therefore the number of triangles is nThreads/3)
											uint nTriangles,
											float  *t,
											float4 *n) {
	// Each vertex of a triangle is stored as a float4, the last component is not used.
	// gTriangleSoup contains vertices of triangles in the following layout:
	// --------------------------------------------------------------
	// | t0_v0 | t0_v1 | t0_v2 | t1_v0 | t1_v1 | t1_v2 | t2_v0 | ...
	// --------------------------------------------------------------

	unsigned int cache_size = get_local_size(0);
	unsigned int id = get_local_id(0);
	unsigned int processed_vertices = 0;
	bool found_collision = false;

	while (processed_vertices < 3 * nTriangles) {
		// load n vertices coalesced where n equals cache_size
		if (processed_vertices + id < 3 * nTriangles) {
			lTriangleCache[id] = gTriangleSoup[processed_vertices + id];
		}

		// wait for all results are written
		barrier(CLK_LOCAL_MEM_FENCE);

		// check cache_size/3 triangles for collisions
		for (int i = 0; i < cache_size / 3; i++)
		{
			float4 v0 = lTriangleCache[3 * i + 0];
			v0.w = 0;
			float4 v1 = lTriangleCache[3 * i + 1];
			v1.w = 0;
			float4 v2 = lTriangleCache[3 * i + 2];
			v2.w = 0;

			// check if collision is found and t and n has to be updated
			// if
			float t_new;
			float4 n_new;
			if (LineTriangleIntersection(x0, x1, v0, v1, v2, &t_new, &n_new) && (!found_collision || t_new < *t)) {
				*t = t_new;
				*n = n_new;
				found_collision = true;
			}
		}

		// wait for other worker who may need triangles stored in cache
		barrier(CLK_LOCAL_MEM_FENCE);

		processed_vertices += cache_size;
	}

	return found_collision;
}


float random(ulong seed) {
        seed = (seed * 0x5DEECE66DL + 0xBL) &((1L << 48) - 1);
        return seed >> 16;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// This is the integration kernel. Implement the missing functionality
//
// Input data:
// gAlive         - Field of flag indicating whether the particle with that index is alive (!= 0) or dead (0). You will have to modify this
// gForceField    - 3D texture with the  force field
// sampler        - 3D texture sampler for the force field (see usage below)
// nParticles     - Number of input particles
// nTriangles     - Number of triangles in the scene (for collision detection)
// lTriangleCache - Local memory cache to be used during collision detection for the triangles
// gTriangleSoup  - The triangles in the scene (layout see the description of CheckCollisions())
// gPosLife       - Position (xyz) and remaining lifetime (w) of a particle
// gVelMass       - Velocity vector (xyz) and the mass (w) of a particle
// dT             - The timestep for the integration (the has to be subtracted from the remaining lifetime of each particle)
//
// Output data:
// gAlive   - Updated alive flags
// gPosLife - Updated position and lifetime
// gVelMass - Updated position and mass
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__kernel void Integrate(__global uint *gAlive,
						__read_only image3d_t gForceField,
						sampler_t sampler,
						uint nParticles,
						uint nTriangles,
						__local float4 *lTriangleCache,
						__global float4 *gTriangleSoup,
						__global float4 *gPosLife,
						__global float4 *gVelMass,
						float dT
						) {

	float4 gAccel = (float4)(0.f, -9.81f, 0.f, 0.f);

	// Verlet Velocity Integration
	float4 x0 = gPosLife[get_global_id(0)];
	float4 v0 = gVelMass[get_global_id(0)];

	float mass = v0.w;
	float life = x0.w;

	float4 lookUp = x0;
	lookUp.w = 0.f;

	float4 F0 = read_imagef(gForceField, sampler, lookUp);
	F0.w = 0;

	// Initialise acceleration a0 and a1 which is a(t) and a(t+dt)
	float4 a0 = F0 / mass + gAccel;
	//a0 = gAccel;

	//// Verlet velocity integration
	// Position: x(t + dt) = x(t) + v(t)dt + 0.5a(t)(dt)^2
	float4 x1 = x0 + v0 * dT + 0.5f * a0 * dT * dT;

	// Check for collisions and correct the position and velocity of the particle if it collides with a triangle
	// - Don't forget to offset the particles from the surface a little bit, otherwise they might get stuck in it.
	// - Dampen the velocity (e.g. by a factor of 0.7) to simulate dissipation of the energy.
	float t;
	float4 normal;
	if (CheckCollisions(x0, x1, gTriangleSoup, lTriangleCache, nTriangles, &t, &normal)) {
		// adjust x1 by x1 = (x1 - x0) t + normal * [c]
		x1 = x0 + (x1 - x0) * t + normal * 0.001f;

		// mirror v0 along normal
		v0 = v0 - 2 * dot3(v0, normal) * normal;

		// damping velocity
		v0 *= 0.66f;
	}

	// Update force field for new position
	lookUp = x1;
	lookUp.w = 0.0f;
	F0 = read_imagef(gForceField, sampler, lookUp);

	// Add force of forcefield to a(t+dt)
	float4 a1 = F0 / mass + gAccel;
	//a1 = gAccel;

	// Velocity: v(t + dt) = v(t) + 0.5 (a(t) + a(t+dt)) dt
	float4 v1;
	v1 = v0 + 0.5f * (a0 + a1) * dT;
	v1.w = 0;

	// set life of particle
	x1.w = life - 0.1 * length(v1);
	// set mass of particle
	v1.w = mass;

	// Kill the particle if its life is < 0.0 by setting the corresponding flag in gAlive to 0.
	if (x0.w < 0)	{
		//gAlive[get_global_id(0)] = 0;
		float seed = random(get_global_id(0));
		float r1 = random(seed) / (1024 * 1024 * 1024 * 2.0f);
		float r2 = random(r1) / (1024 * 1024 * 1024 * 2.0f);
		float r3 = random(r2) / (1024 * 1024 * 1024 * 2.0f);
		float r4 = random(r3) / (1024 * 1024 * 1024 * 2.0f);

		if (get_global_id(0) % 3 == 0)
			x1 = LEFT_EYE;
		if (get_global_id(0) % 3 == 1)
			x1 = RIGHT_EYE;
		if (get_global_id(0) % 3 == 2)
			x1 = MOUTH;

		v1 = v1 * 0.1f + 2 * r4 * dot3(MONKEY_NORMAL, (float4)(r1, r2, r3, 0.0f));
		v1.w = mass;
	}

	// Update position/life and velocity/mass
	gPosLife[get_global_id(0)] = x1;
	gVelMass[get_global_id(0)] = v1;
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__kernel void Clear(__global float4* gPosLife, __global float4* gVelMass) {
	uint GID = get_global_id(0);
	gPosLife[GID] = 0.f;
	gVelMass[GID] = 0.f;
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__kernel void Reorganize(	__global uint* gAlive, __global uint* gRank,
							__global float4* gPosLifeIn,  __global float4* gVelMassIn,
							__global float4* gPosLifeOut, __global float4* gVelMassOut) {


	// Re-order the particles according to the gRank obtained from the parallel prefix sum
	// ADD YOUR CODE HERE
	// TODO
}
