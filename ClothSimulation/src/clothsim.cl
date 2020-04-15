#define DAMPING 0.02f

#define G_ACCEL (float4)(0.f, -9.81f, 0.f, 0.f)

#define WEIGHT_ORTHO	0.138f
#define WEIGHT_DIAG		0.097f
#define WEIGHT_ORTHO_2	0.069f
#define WEIGHT_DIAG_2	0.048f


#define ROOT_OF_2 1.4142135f
#define DOUBLE_ROOT_OF_2 2.8284271f

///////////////////////////////////////////////////////////////////////////////
// The integration kernel
// Input data:
// width and height - the dimensions of the particle grid
// d_pos - the most recent position of the cloth particle while...
// d_prevPos - ...contains the position from the previous iteration.
// elapsedTime      - contains the elapsed time since the previous invocation of the kernel,
// prevElapsedTime  - contains the previous time step.
// simulationTime   - contains the time elapsed since the start of the simulation (useful for wind)
// All time values are given in seconds.
//
// Output data:
// d_prevPos - Input data from d_pos must be copied to this array
// d_pos     - Updated positions
///////////////////////////////////////////////////////////////////////////////
__kernel void Integrate(unsigned int width,
					unsigned int height,
					__global float4* d_pos,
					__global float4* d_prevPos,
					float elapsedTime,
					float prevElapsedTime,
					float simulationTime) {
  // Make sure the work-item does not map outside the cloth
  if(get_global_id(0) >= width || get_global_id(1) >= height) {
    return;
  }

  unsigned int particleID = get_global_id(0) + get_global_id(1) * width;
  // This is just to keep every 8th particle of the first row attached to the bar
  if(particleID > width-1 || ( particleID & ( 7 )) != 0) {
    // read most recent and previous position
    float4 xprev = d_prevPos[particleID];
    float4 x0 = d_pos[particleID];

    // set current velocity and acceleration
    float4 v0 = (float4) (0.0f, 0.0f, 0.0f, 0.0f);
    if (prevElapsedTime > 0) {
      v0 = (x0 - xprev) * (1 / prevElapsedTime);
    }
    float4 a0 = (float4)(sin(simulationTime * 2) / 2 + 1, 0.0f, cos(simulationTime * 3) - 2, 0.0f) * 0.3f + G_ACCEL;

  	//// Verlet position integration
  	// Position: x(t + dt) = x(t) + v(t)dt + 0.5a(t)(dt)^2
    float dt = elapsedTime;
  	float4 x1 = x0 + v0 * dt + 0.5f * a0 * dt * dt;

    d_prevPos[particleID] = x0;
    d_pos[particleID] = x1;
  }
}



///////////////////////////////////////////////////////////////////////////////
// Input data:
// pos1 and pos2 - The positions of two particles
// restDistance  - the distance between the given particles at rest
//
// Return data:
// correction vector for particle 1
///////////////////////////////////////////////////////////////////////////////
  float4 SatisfyConstraint(float4 pos1,
						 float4 pos2,
						 float restDistance){
	float4 toNeighbor = pos2 - pos1;
	return (toNeighbor - normalize(toNeighbor) * restDistance);
}

///////////////////////////////////////////////////////////////////////////////
// Input data:
// width and height - the dimensions of the particle grid
// restDistance     - the distance between two orthogonally neighboring particles at rest
// d_posIn          - the input positions
//
// Output data:
// d_posOut - new positions must be written here
///////////////////////////////////////////////////////////////////////////////

#define TILE_X 16
#define TILE_Y 16
#define HALOSIZE 2

__kernel __attribute__((reqd_work_group_size(TILE_X, TILE_Y, 1)))
__kernel void SatisfyConstraints(unsigned int width,
                  							unsigned int height,
                  							float restDistance,
                  							__global float4* d_posOut,
                  							__global float4 const * d_posIn) {

  if(get_global_id(0) >= width || get_global_id(1) >= height) {
	  return;
  }

	const int2 lid = {
		lid.x = get_local_id(0),
		lid.y = get_local_id(1)
	};
	const int2 gid = {
		gid.x = get_global_id(0),
		gid.y = get_global_id(1)
	};
	const int2 hid = {
		hid.x = get_local_id(0) + HALOSIZE,
		hid.y = get_local_id(1) + HALOSIZE
	};
  __local float4 tile[TILE_Y + 2 * HALOSIZE][TILE_X + 2 * HALOSIZE];

  //// Load Halo Region
  // load top and bottom halo region
  if (lid.y < HALOSIZE) {
    if (gid.x < width && gid.y >= 2 && gid.y < height) {
      tile[lid.y][lid.x + HALOSIZE] = d_posIn[gid.x + (gid.y - 2) * width];
    } else {
      tile[lid.y][lid.x + HALOSIZE] = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
    }
  }
  if (lid.y - TILE_Y + HALOSIZE >= 0) {
    if (gid.x < width && gid.y < height - 2) {
      tile[lid.y + 2 * HALOSIZE][lid.x + HALOSIZE] = d_posIn[gid.x + (gid.y + 2) * width];
    } else {
      tile[lid.y + 2 * HALOSIZE][lid.x + HALOSIZE] = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
    }
  }

  // load left and right halo region
  if (lid.x < HALOSIZE) {
    if (gid.x >= 2 && gid.x < width && gid.y < height) {
      tile[lid.y + HALOSIZE][lid.x] = d_posIn[gid.x - 2 + gid.y * width];
    } else {
      tile[lid.y + HALOSIZE][lid.x] = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
    }
  }
  if (lid.x - TILE_X + HALOSIZE >= 0) {
    if (gid.x < width - 2 && gid.y < height) {
      tile[lid.y + HALOSIZE][lid.x + 2 * HALOSIZE] = d_posIn[gid.x + 2 + gid.y * width];
    } else {
      tile[lid.y + HALOSIZE][lid.x + 2 * HALOSIZE] = (float4)(1.0f, 2.0f, 3.0f, 0.0f);
    }
  }

  // load corner halos
  if (lid.x < HALOSIZE && lid.y < HALOSIZE) {
    if (gid.x - 2 > 0 && gid.x < width && gid.y - 2 > 0 && gid.y < height) {
      tile[lid.y][lid.x] = d_posIn[gid.x - 2 + (gid.y - 2) * width];
    } else {
      tile[lid.y][lid.x] = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
    }
  }
  if (lid.x - TILE_X + 2 >= 0 && lid.y < HALOSIZE) {
    if (gid.x + 2 < width && gid.y - 2 > 0 && gid.y < height) {
      tile[lid.y][lid.x + 2 * HALOSIZE] = d_posIn[gid.x + 2 + (gid.y - 2) * width];
    } else {
      tile[lid.y][lid.x + 2 * HALOSIZE] = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
    }
  }
  if (lid.x < HALOSIZE && lid.y - TILE_Y + 2 >= 0) {
    if (gid.x - 2 > 0 && gid.x < width && gid.y + 2 < height) {
      tile[lid.y + 2 * HALOSIZE][lid.x] = d_posIn[gid.x - 2 + (gid.y + 2) * width];
    } else {
      tile[lid.y + 2 * HALOSIZE][lid.x] = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
    }
  }
  if (lid.x - TILE_X + 2 >= 0 && lid.y - TILE_Y + 2 >= 0) {
    if (gid.x + 2 < width && gid.y + 2 < height) {
      tile[lid.y + 2 * HALOSIZE][lid.x + 2 * HALOSIZE] = d_posIn[gid.x + 2 + (gid.y + 2) * width];
    } else {
      tile[lid.y + 2 * HALOSIZE][lid.x + 2 * HALOSIZE] = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
    }
  }

  //// Load Inner Particle Region
  if (gid.x < width && gid.y < height) {
    tile[hid.y][hid.x] = d_posIn[gid.x + gid.y * width];
  }

  // wait for particles are written in local
  barrier(CLK_LOCAL_MEM_FENCE);

	// if (gid.x == 16 && gid.y == 16) {
	// 	for (int y = 0; y < TILE_X + 2 * HALOSIZE; y++) {
	// 		printf("[");
	// 		for (int x = 0; x < TILE_Y + 2 * HALOSIZE; x++) {
	// 			printf("%d, ", convert_int(tile[y][x].x * 100));
	// 		}
	// 		printf("]\n");
	// 	}
	// 	printf("END!\n\n");
	// }

	// Satisfy all the constraints (structural, shear, and bend).
	// You can use weights defined at the beginning of this file.
  if (gid.x < width && gid.y < height) {
    float4 pos = tile[hid.y][hid.x];
		float4 neighbor = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
    float4 correct_vector = (float4)(0.0f, 0.0f, 0.0f, 0.0f);

    // structural
    if (gid.x > 0) {
      neighbor = tile[hid.y][hid.x - 1];
      correct_vector += SatisfyConstraint(pos, neighbor, restDistance) * WEIGHT_ORTHO;
    }
    if (gid.y < height - 1) {
      neighbor = tile[hid.y + 1][hid.x];
      correct_vector += SatisfyConstraint(pos, neighbor, restDistance) * WEIGHT_ORTHO;
    }
    if (gid.x < width - 1) {
      neighbor = tile[hid.y][hid.x + 1];
      correct_vector += SatisfyConstraint(pos, neighbor, restDistance) * WEIGHT_ORTHO;
    }
    if (gid.y > 0) {
      neighbor = tile[hid.y - 1][hid.x];
      correct_vector += SatisfyConstraint(pos, neighbor, restDistance) * WEIGHT_ORTHO;
    }

    // shear
    if (gid.x > 0 && gid.y < height - 1) {
      neighbor = tile[hid.y + 1][hid.x - 1];
      correct_vector += SatisfyConstraint(pos, neighbor, ROOT_OF_2 * restDistance) * WEIGHT_DIAG;
    }
    if (gid.x < width - 1 && gid.y < height - 1) {
      neighbor = tile[hid.y + 1][hid.x + 1];
      correct_vector += SatisfyConstraint(pos, neighbor, ROOT_OF_2 * restDistance) * WEIGHT_DIAG;
    }
    if (gid.x < width - 1 && gid.y > 0) {
      neighbor = tile[hid.y - 1][hid.x + 1];
      correct_vector += SatisfyConstraint(pos, neighbor, ROOT_OF_2 * restDistance) * WEIGHT_DIAG;
    }
    if (gid.x > 0 && gid.y > 0) {
      neighbor = tile[hid.y - 1][hid.x - 1];
      correct_vector += SatisfyConstraint(pos, neighbor, ROOT_OF_2 * restDistance) * WEIGHT_DIAG;
    }

    // bend
    for (int i = 0; i < 3; i++) {
      float4 b1 = tile[hid.y - 2 + 2 * i][hid.x - 2];
      float4 b2 = tile[hid.y - 2 + 2 * i][hid.x + 2];
      float4 b3 = tile[hid.y - 2 + 2 * i][hid.x];

			float weight = WEIGHT_DIAG_2;
			float dist = 2 * ROOT_OF_2 * restDistance;
      if (i==1) {
				weight = WEIGHT_ORTHO_2;
				dist = 2 * restDistance;
			} else {

			}

			// left
      if ((gid.y - 2 + 2 * i) >= 0 && (gid.y - 2 + 2 * i) < height && gid.x > 1 && gid.x < width) {
        correct_vector += SatisfyConstraint(pos, b1, dist) * weight;
			}
			// right
			if ((gid.y - 2 + 2 * i) >= 0 && (gid.y - 2 + 2 * i) < height && gid.x < width - 2) {
				correct_vector += SatisfyConstraint(pos, b2, dist) * weight;
			}
			// mid
			if (!(i==1) && (gid.y - 2 + 2 * i) >= 0 && (gid.y - 2 + 2 * i) < height && gid.x < width) {
				correct_vector += SatisfyConstraint(pos, b3, 2 * ROOT_OF_2 * restDistance) * WEIGHT_ORTHO_2;
			}
    }
		// correct pos
  	unsigned int particleID = get_global_id(0) + get_global_id(1) * width;
		if (particleID > width-1 || ( particleID & ( 7 )) != 0) {
	    if (length(correct_vector) < restDistance/2) {
	      pos += correct_vector;
	    } else {
	      pos += (restDistance / 2) * normalize(correct_vector);
	    }
		}

    // write result back
		d_posOut[gid.x + gid.y * width] = pos;
  }
}


///////////////////////////////////////////////////////////////////////////////
// Input data:
// width and height - the dimensions of the particle grid
// d_pos            - the input positions
// spherePos        - The position of the sphere (xyz)
// sphereRad        - The radius of the sphere
//
// Output data:
// d_pos            - The updated positions
///////////////////////////////////////////////////////////////////////////////
__kernel void CheckCollisions(unsigned int width,
								unsigned int height,
								__global float4* d_pos,
								float4 spherePos,
								float sphereRad) {

  const uint2 gid = (uint2)(get_global_id(0), get_global_id(1));

	// Find whether the particle is inside the sphere.
	// If so, push it outside.
	// check if worker has to do something
  if (gid.x < width && gid.y < height) {
		// calculate distance vector between sphere center and particle
    float4 distance_vector = d_pos[gid.x + gid.y * width] - spherePos;

		// check whether distance is lower than sphere radius
		// if so: push particle outside
    if (length(distance_vector) < sphereRad) {
        d_pos[gid.x + gid.y * width] = spherePos + sphereRad * normalize(distance_vector);
    }
  }
}

///////////////////////////////////////////////////////////////////////////////
// There is no need to change this function!
///////////////////////////////////////////////////////////////////////////////
float4 CalcTriangleNormal( float4 p1, float4 p2, float4 p3) {
    float4 v1 = p2-p1;
    float4 v2 = p3-p1;

    return cross( v1, v2);
}

///////////////////////////////////////////////////////////////////////////////
// There is no need to change this kernel!
///////////////////////////////////////////////////////////////////////////////
__kernel void ComputeNormals(unsigned int width,
								unsigned int height,
								__global float4* d_pos,
								__global float4* d_normal){

    int particleID = get_global_id(0) + get_global_id(1) * width;
    float4 normal = (float4)( 0.0f, 0.0f, 0.0f, 0.0f);

    int minX, maxX, minY, maxY, cntX, cntY;
    minX = max( (int)(0), (int)(get_global_id(0)-1));
    maxX = min( (int)(width-1), (int)(get_global_id(0)+1));
    minY = max( (int)(0), (int)(get_global_id(1)-1));
    maxY = min( (int)(height-1), (int)(get_global_id(1)+1));

    for( cntX = minX; cntX < maxX; ++cntX) {
        for( cntY = minY; cntY < maxY; ++cntY) {
            normal += normalize( CalcTriangleNormal(
                d_pos[(cntX+1)+width*(cntY)],
                d_pos[(cntX)+width*(cntY)],
                d_pos[(cntX)+width*(cntY+1)]));
            normal += normalize( CalcTriangleNormal(
                d_pos[(cntX+1)+width*(cntY+1)],
                d_pos[(cntX+1)+width*(cntY)],
                d_pos[(cntX)+width*(cntY+1)]));
        }
    }
    d_normal[particleID] = normalize( normal);
}
