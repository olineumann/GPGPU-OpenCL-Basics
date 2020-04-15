/******************************************************************************
GPU Computing / GPGPU Praktikum source code.

******************************************************************************/

#include "CAssignment1.h"

#include "CSimpleArraysTask.h"
#include "CMatrixRotateTask.h"

#include <iostream>

using namespace std;

#define NUM_TASKS 1048576

///////////////////////////////////////////////////////////////////////////////
// CAssignment1

bool CAssignment1::DoCompute()
{
	// Task 1: simple array addition.
	cout << endl << endl << "Running vector addition example..." << endl << endl;
	{
		size_t localWorkSize[3] = {64, 1, 1};
		CSimpleArraysTask task(NUM_TASKS);
		RunComputeTask(task, localWorkSize);
	}
	{
		size_t localWorkSize[3] = {128, 1, 1};
		CSimpleArraysTask task(NUM_TASKS);
		RunComputeTask(task, localWorkSize);
	}
	{
		size_t localWorkSize[3] = {192, 1, 1};
		CSimpleArraysTask task(NUM_TASKS);
		RunComputeTask(task, localWorkSize);
	}
	{
		size_t localWorkSize[3] = {256, 1, 1};
		CSimpleArraysTask task(NUM_TASKS);
		RunComputeTask(task, localWorkSize);
	}
	{
		size_t localWorkSize[3] = {320, 1, 1};
		CSimpleArraysTask task(NUM_TASKS);
		RunComputeTask(task, localWorkSize);
	}
	{
		size_t localWorkSize[3] = {384, 1, 1};
		CSimpleArraysTask task(NUM_TASKS);
		RunComputeTask(task, localWorkSize);
	}
	{
		size_t localWorkSize[3] = {448, 1, 1};
		CSimpleArraysTask task(NUM_TASKS);
		RunComputeTask(task, localWorkSize);
	}
	{
		size_t localWorkSize[3] = {512, 1, 1};
		CSimpleArraysTask task(NUM_TASKS);
		RunComputeTask(task, localWorkSize);
	}
	{
		size_t localWorkSize[3] = {576, 1, 1};
		CSimpleArraysTask task(NUM_TASKS);
		RunComputeTask(task, localWorkSize);
	}
	{
		size_t localWorkSize[3] = {640, 1, 1};
		CSimpleArraysTask task(NUM_TASKS);
		RunComputeTask(task, localWorkSize);
	}
	{
		size_t localWorkSize[3] = {704, 1, 1};
		CSimpleArraysTask task(NUM_TASKS);
		RunComputeTask(task, localWorkSize);
	}
	{
		size_t localWorkSize[3] = {768, 1, 1};
		CSimpleArraysTask task(NUM_TASKS);
		RunComputeTask(task, localWorkSize);
	}
	{
		size_t localWorkSize[3] = {832, 1, 1};
		CSimpleArraysTask task(NUM_TASKS);
		RunComputeTask(task, localWorkSize);
	}
	{
		size_t localWorkSize[3] = {896, 1, 1};
		CSimpleArraysTask task(NUM_TASKS);
		RunComputeTask(task, localWorkSize);
	}
	{
		size_t localWorkSize[3] = {960, 1, 1};
		CSimpleArraysTask task(NUM_TASKS);
		RunComputeTask(task, localWorkSize);
	}
	{
		size_t localWorkSize[3] = {1024, 1, 1};
		CSimpleArraysTask task(NUM_TASKS);
		RunComputeTask(task, localWorkSize);
	}

	// Task 2: matrix rotation.
	cout << endl << endl << "Running matrix rotation example..." << endl << endl;
	{
		size_t localWorkSize[3] = {32, 16, 1};
		CMatrixRotateTask task(33, 17);
		RunComputeTask(task, localWorkSize);
	}
	{
		size_t localWorkSize[3] = {32, 16, 1};
		CMatrixRotateTask task(2048, 1024);
		RunComputeTask(task, localWorkSize);
	}
	{
		size_t localWorkSize[3] = {32, 16, 1};
		CMatrixRotateTask task(2048, 1025);
		RunComputeTask(task, localWorkSize);
	}
	{
		size_t localWorkSize[3] = {32, 16, 1};
		CMatrixRotateTask task(2049, 1024);
		RunComputeTask(task, localWorkSize);
	}
	{
		size_t localWorkSize[3] = {32, 16, 1};
		CMatrixRotateTask task(2049, 1025);
		RunComputeTask(task, localWorkSize);
	}

	return true;
}

///////////////////////////////////////////////////////////////////////////////