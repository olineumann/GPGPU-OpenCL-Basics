/******************************************************************************
GPU Computing / GPGPU Praktikum source code.

******************************************************************************/

#include "CAssignment2.h"

#include "CReductionTask.h"
#include "CScanTask.h"

#include <iostream>

using namespace std;

///////////////////////////////////////////////////////////////////////////////
// CAssignment2

bool CAssignment2::DoCompute()
{
	// Task 1: parallel reduction
	cout<<"########################################"<<endl;
	cout<<"Running parallel reduction task..."<<endl<<endl;
	{
		size_t LocalWorkSize[3] = {256, 1, 1};
		CReductionTask reduction(1024 * 1024 * 16);
		RunComputeTask(reduction, LocalWorkSize);

		// cout << "######################################   64   ######################################\n";
		// LocalWorkSize[0] = 64;
		// reduction = CReductionTask(1024 * 1024 * 1);
		// RunComputeTask(reduction, LocalWorkSize);
		// reduction = CReductionTask(1024 * 1024 * 2);
		// RunComputeTask(reduction, LocalWorkSize);
		// reduction = CReductionTask(1024 * 1024 * 4);
		// RunComputeTask(reduction, LocalWorkSize);
		// reduction = CReductionTask(1024 * 1024 * 8);
		// RunComputeTask(reduction, LocalWorkSize);
		// reduction = CReductionTask(1024 * 1024 * 16);
		// RunComputeTask(reduction, LocalWorkSize);
		// cout << "######################################  128   ######################################\n";
		// LocalWorkSize[0] = 128;
		// reduction = CReductionTask(1024 * 1024 * 1);
		// RunComputeTask(reduction, LocalWorkSize);
		// reduction = CReductionTask(1024 * 1024 * 2);
		// RunComputeTask(reduction, LocalWorkSize);
		// reduction = CReductionTask(1024 * 1024 * 4);
		// RunComputeTask(reduction, LocalWorkSize);
		// reduction = CReductionTask(1024 * 1024 * 8);
		// RunComputeTask(reduction, LocalWorkSize);
		// reduction = CReductionTask(1024 * 1024 * 16);
		// RunComputeTask(reduction, LocalWorkSize);
		// cout << "######################################  256   ######################################\n";
		// LocalWorkSize[0] = 256;
		// reduction = CReductionTask(1024 * 1024 * 1);
		// RunComputeTask(reduction, LocalWorkSize);
		// reduction = CReductionTask(1024 * 1024 * 2);
		// RunComputeTask(reduction, LocalWorkSize);
		// reduction = CReductionTask(1024 * 1024 * 4);
		// RunComputeTask(reduction, LocalWorkSize);
		// reduction = CReductionTask(1024 * 1024 * 8);
		// RunComputeTask(reduction, LocalWorkSize);
		// reduction = CReductionTask(1024 * 1024 * 16);
		// RunComputeTask(reduction, LocalWorkSize);
		// cout << "######################################  512   ######################################\n";
		// LocalWorkSize[0] = 512;
		// reduction = CReductionTask(1024 * 1024 * 1);
		// RunComputeTask(reduction, LocalWorkSize);
		// reduction = CReductionTask(1024 * 1024 * 2);
		// RunComputeTask(reduction, LocalWorkSize);
		// reduction = CReductionTask(1024 * 1024 * 4);
		// RunComputeTask(reduction, LocalWorkSize);
		// reduction = CReductionTask(1024 * 1024 * 8);
		// RunComputeTask(reduction, LocalWorkSize);
		// reduction = CReductionTask(1024 * 1024 * 16);
		// RunComputeTask(reduction, LocalWorkSize);
		// cout << "######################################  1024  ######################################\n";
		// LocalWorkSize[0] = 1024;
		// reduction = CReductionTask(1024 * 1024 * 1);
		// RunComputeTask(reduction, LocalWorkSize);
		// reduction = CReductionTask(1024 * 1024 * 2);
		// RunComputeTask(reduction, LocalWorkSize);
		// reduction = CReductionTask(1024 * 1024 * 4);
		// RunComputeTask(reduction, LocalWorkSize);
		// reduction = CReductionTask(1024 * 1024 * 8);
		// RunComputeTask(reduction, LocalWorkSize);
		// reduction = CReductionTask(1024 * 1024 * 16);
		// RunComputeTask(reduction, LocalWorkSize);

	}

	// Task 2: parallel prefix sum
	cout << "########################################"<<endl;
	cout<<"Running parallel prefix sum task..."<<endl<<endl;
	{
		size_t LocalWorkSize[3] = {256, 1, 1};
		CScanTask scan(1024*1024*16 , LocalWorkSize[0]);
		RunComputeTask(scan, LocalWorkSize);
	}


	return true;
}

///////////////////////////////////////////////////////////////////////////////
