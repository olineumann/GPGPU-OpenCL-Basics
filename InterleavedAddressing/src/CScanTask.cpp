/******************************************************************************
GPU Computing / GPGPU Praktikum source code.

******************************************************************************/

#include "CScanTask.h"

#include "../Common/CLUtil.h"
#include "../Common/CTimer.h"

#include <string.h>

using namespace std;

// number of banks in the local memory. This can be used to avoid bank conflicts
// but we also need to allocate more local memory for that.
#define NUM_BANKS	32

///////////////////////////////////////////////////////////////////////////////
// CScanTask

// only useful for debug info
const string g_kernelNames[2] =
{
	"scanNaive",
	"scanWorkEfficient"
};

CScanTask::CScanTask(size_t ArraySize, size_t MinLocalWorkSize)
	: m_N(ArraySize), m_hArray(NULL), m_hResultCPU(NULL), m_hResultGPU(NULL),
	m_dPingArray(NULL), m_dPongArray(NULL),
	m_Program(NULL),
	m_ScanNaiveKernel(NULL), m_ScanWorkEfficientKernel(NULL), m_ScanWorkEfficientAddKernel(NULL)
{
	// compute the number of levels that we need for the work-efficient algorithm

	m_MinLocalWorkSize = MinLocalWorkSize;

	m_nLevels = 2;
	size_t N = ArraySize;
	while (N > 2 * m_MinLocalWorkSize)
	{
		N /= 2 * m_MinLocalWorkSize;
		m_nLevels++;
	}

	// Reset validation results
	for (int i = 0; i < (int)ARRAYLEN(m_bValidationResults); i++)
		m_bValidationResults[i] = false;
}

CScanTask::~CScanTask()
{
	ReleaseResources();
}

bool CScanTask::InitResources(cl_device_id Device, cl_context Context)
{
	//CPU resources
	m_hArray	 = new unsigned int[m_N];
	m_hResultCPU = new unsigned int[m_N];
	m_hResultGPU = new unsigned int[m_N];

	//fill the array with some values
	for(unsigned int i = 0; i < m_N; i++)
		//m_hArray[i] = 1;			// Use this for debugging
		m_hArray[i] = rand() & 15;

	//device resources
	// ping-pong buffers
	cl_int clError, clError2;
	m_dPingArray = clCreateBuffer(Context, CL_MEM_READ_WRITE, sizeof(cl_uint) * m_N, NULL, &clError2);
	clError = clError2;
	m_dPongArray = clCreateBuffer(Context, CL_MEM_READ_WRITE, sizeof(cl_uint) * m_N, NULL, &clError2);
	clError |= clError2;

	// level buffer
	m_dLevelArrays = new cl_mem[m_nLevels];
	unsigned int N = m_N;
	for (unsigned int i = 0; i < m_nLevels; i++) {
		m_dLevelArrays[i] = clCreateBuffer(Context, CL_MEM_READ_WRITE, sizeof(cl_uint) * N, NULL, &clError2);
		clError |= clError2;
		N = max(N / (2 * m_MinLocalWorkSize), m_MinLocalWorkSize);
	}
	V_RETURN_FALSE_CL(clError, "Error allocating device arrays");

	//load and compile kernels
	string programCode;

	CLUtil::LoadProgramSourceToMemory("Scan.cl", programCode);
	m_Program = CLUtil::BuildCLProgramFromMemory(Device, Context, programCode);
	if(m_Program == nullptr) return false;

	//create kernels
	m_ScanNaiveKernel = clCreateKernel(m_Program, "Scan_Naive", &clError);
	V_RETURN_FALSE_CL(clError, "Failed to create kernel.");

	m_ScanWorkEfficientKernel = clCreateKernel(m_Program, "Scan_WorkEfficient", &clError);
	V_RETURN_FALSE_CL(clError, "Failed to create kernel.");

	m_ScanWorkEfficientAddKernel = clCreateKernel(m_Program, "Scan_WorkEfficientAdd", &clError);
	V_RETURN_FALSE_CL(clError, "Failed to create kernel.");

	return true;
}

void CScanTask::ReleaseResources()
{
	// host resources
	SAFE_DELETE_ARRAY(m_hArray);

	SAFE_DELETE_ARRAY(m_hResultCPU);
	SAFE_DELETE_ARRAY(m_hResultGPU);

	// device resources
	SAFE_RELEASE_MEMOBJECT(m_dPingArray);
	SAFE_RELEASE_MEMOBJECT(m_dPongArray);

	if(m_dLevelArrays)
		for (unsigned int i = 0; i < m_nLevels; i++) {
			SAFE_RELEASE_MEMOBJECT(m_dLevelArrays[i]);
		}
	SAFE_DELETE_ARRAY(m_dLevelArrays);

	SAFE_RELEASE_KERNEL(m_ScanNaiveKernel);
	SAFE_RELEASE_KERNEL(m_ScanWorkEfficientKernel);
	SAFE_RELEASE_KERNEL(m_ScanWorkEfficientAddKernel);

	SAFE_RELEASE_PROGRAM(m_Program);
}

void CScanTask::ComputeGPU(cl_context Context, cl_command_queue CommandQueue, size_t LocalWorkSize[3])
{
	cout << endl;

	ValidateTask(Context, CommandQueue, LocalWorkSize, 0);
	ValidateTask(Context, CommandQueue, LocalWorkSize, 1);

	cout << endl;

	TestPerformance(Context, CommandQueue, LocalWorkSize, 0);
	TestPerformance(Context, CommandQueue, LocalWorkSize, 1);

	cout << endl;
}

void CScanTask::ComputeCPU()
{
	CTimer timer;
	timer.Start();

	unsigned int nIterations = 100;
	for(unsigned int j = 0; j < nIterations; j++) {
		unsigned int sum = 0;
		for(unsigned int i = 0; i < m_N; i++) {
			sum += m_hArray[i];
			m_hResultCPU[i] = sum;
		}
	}

	timer.Stop();
	double ms = timer.GetElapsedMilliseconds() / double(nIterations);
	cout << "  average time: " << ms << " ms, throughput: " << 1.0e-6 * (double)m_N / ms << " Gelem/s" <<endl;
}

bool CScanTask::ValidateResults()
{
	bool success = true;

	for(int i = 0; i < 2; i++)
		if(!m_bValidationResults[i])
		{
			cout<<"Validation of reduction kernel "<<g_kernelNames[i]<<" failed." << endl;
			success = false;
		}

	return success;
}

void CScanTask::Scan_Naive(cl_context Context, cl_command_queue CommandQueue, size_t LocalWorkSize[3])
{
	for (unsigned int offset = 1; offset < m_N; offset = offset * 2)
	{
		cl_int cl_error;
		cl_error = clSetKernelArg(m_ScanNaiveKernel, 0, sizeof(cl_mem), (void*) &m_dPingArray);
		cl_error = clSetKernelArg(m_ScanNaiveKernel, 1, sizeof(cl_mem), (void*) &m_dPongArray);
		cl_error = clSetKernelArg(m_ScanNaiveKernel, 2, sizeof(cl_uint), (void*) &m_N);
		cl_error = clSetKernelArg(m_ScanNaiveKernel, 3, sizeof(cl_uint), (void*) &offset);
		V_RETURN_CL(cl_error, "Failed to set kernel arguments in 'Scan_Naive'.");

		// get local and global work size
		size_t localWorkSize = LocalWorkSize[0];
		size_t globalWorkSize = m_N;

		if (globalWorkSize < localWorkSize)
		{
			localWorkSize = globalWorkSize;
		}

		// run kernel
	  cl_error = clEnqueueNDRangeKernel(CommandQueue, m_ScanNaiveKernel, 1, NULL, &globalWorkSize, &localWorkSize, 0, NULL, NULL);
		V_RETURN_CL(cl_error, "Failed to run kernel in 'Scan_Naive'.");

		// swap ping and pong array
		swap(m_dPingArray, m_dPongArray);
	}
}

void CScanTask::Scan_WorkEfficient(cl_context Context, cl_command_queue CommandQueue, size_t LocalWorkSize[3])
{
	// get local and global work size
	size_t localWorkSize = LocalWorkSize[0];
	size_t globalWorkSize = m_N / 2;
	// loop over levels to compute decomposed pps where i+1 input for higher level
	for (unsigned int i = 0; i < m_nLevels - 1; i++) {
		// set kernel arguments for decomposed pps
		cl_int cl_error;
		cl_error = clSetKernelArg(m_ScanWorkEfficientKernel, 0, sizeof(cl_mem), (void*) &m_dLevelArrays[i]);
		cl_error = clSetKernelArg(m_ScanWorkEfficientKernel, 1, sizeof(cl_mem), (void*) &m_dLevelArrays[i + 1]);
		cl_error = clSetKernelArg(m_ScanWorkEfficientKernel, 2, localWorkSize * 2 * sizeof(cl_uint), NULL);
		V_RETURN_CL(cl_error, "Failed to set kernel arguments in 'Scan_WorkEfficient'.");

		// run decomposed pps kernel
		if (globalWorkSize < localWorkSize)
	  	cl_error = clEnqueueNDRangeKernel(CommandQueue, m_ScanWorkEfficientKernel, 1, NULL, &globalWorkSize, &globalWorkSize, 0, NULL, NULL);
		else
	  	cl_error = clEnqueueNDRangeKernel(CommandQueue, m_ScanWorkEfficientKernel, 1, NULL, &globalWorkSize, &localWorkSize, 0, NULL, NULL);
		V_RETURN_CL(cl_error, "Failed to run kernel in 'Scan_WorkEfficient'.");

		// change global work size if there is another step
		if (i != m_nLevels - 2)
		{
			globalWorkSize /= 2 * localWorkSize;
		}
	}

	// multiply global work size with 2 because add kernel always needs twice as much worker
	globalWorkSize *= 2;

	// loop over levels backward to propagate results back with Scan_WorkEfficientAdd
	for (unsigned int i = m_nLevels - 2; i > 0; i--) {
		// change global and local work size
		globalWorkSize *= 2 * localWorkSize;

		// set kernel arguments for compose pps with add
		cl_int cl_error;
		cl_error = clSetKernelArg(m_ScanWorkEfficientAddKernel, 0, sizeof(cl_mem), (void*) &m_dLevelArrays[i]);
		cl_error = clSetKernelArg(m_ScanWorkEfficientAddKernel, 1, sizeof(cl_mem), (void*) &m_dLevelArrays[i - 1]);
		cl_error = clSetKernelArg(m_ScanWorkEfficientAddKernel, 2, localWorkSize * sizeof(cl_uint), NULL);
		V_RETURN_CL(cl_error, "Failed to set kernel arguments in 'Scan_WorkEfficientAdd'.");

		// run add kernel
		if (globalWorkSize < localWorkSize)
	  	cl_error = clEnqueueNDRangeKernel(CommandQueue, m_ScanWorkEfficientAddKernel, 1, NULL, &globalWorkSize, &globalWorkSize, 0, NULL, NULL);
		else
	  	cl_error = clEnqueueNDRangeKernel(CommandQueue, m_ScanWorkEfficientAddKernel, 1, NULL, &globalWorkSize, &localWorkSize, 0, NULL, NULL);
		V_RETURN_CL(cl_error, "Failed to run kernel in 'Scan_WorkEfficientAdd'.");
	}
}

void CScanTask::ValidateTask(cl_context Context, cl_command_queue CommandQueue, size_t LocalWorkSize[3], unsigned int Task)
{
	//run selected task
	switch (Task){
		case 0:
			V_RETURN_CL(clEnqueueWriteBuffer(CommandQueue, m_dPingArray, CL_FALSE, 0, m_N * sizeof(cl_uint), m_hArray, 0, NULL, NULL), "Error copying data from host to device!");
			Scan_Naive(Context, CommandQueue, LocalWorkSize);
			V_RETURN_CL(clEnqueueReadBuffer(CommandQueue, m_dPingArray, CL_TRUE, 0, m_N * sizeof(cl_uint), m_hResultGPU, 0, NULL, NULL), "Error reading data from device!");
			break;
		case 1:
			V_RETURN_CL(clEnqueueWriteBuffer(CommandQueue, m_dLevelArrays[0], CL_FALSE, 0, m_N * sizeof(cl_uint), m_hArray, 0, NULL, NULL), "Error copying data from host to device!");
			Scan_WorkEfficient(Context, CommandQueue, LocalWorkSize);
			V_RETURN_CL(clEnqueueReadBuffer(CommandQueue, m_dLevelArrays[0], CL_TRUE, 0, m_N * sizeof(cl_uint), m_hResultGPU, 0, NULL, NULL), "Error reading data from device!");
			break;
	}

	// validate results
	m_bValidationResults[Task] =( memcmp(m_hResultCPU, m_hResultGPU, m_N * sizeof(unsigned int)) == 0);
}

void CScanTask::TestPerformance(cl_context Context, cl_command_queue CommandQueue, size_t LocalWorkSize[3], unsigned int Task)
{
	cout << "Testing performance of task " << g_kernelNames[Task] << endl;

	//write input data to the GPU
	V_RETURN_CL(clEnqueueWriteBuffer(CommandQueue, m_dPingArray, CL_FALSE, 0, m_N * sizeof(cl_uint), m_hArray, 0, NULL, NULL), "Error copying data from host to device!");
	//finish all before we start meassuring the time
	V_RETURN_CL(clFinish(CommandQueue), "Error finishing the queue!");

	CTimer timer;
	timer.Start();

	//run the kernel N times
	unsigned int nIterations = 100;
	for(unsigned int i = 0; i < nIterations; i++) {
		//run selected task
		switch (Task){
			case 0:
				Scan_Naive(Context, CommandQueue, LocalWorkSize);
				break;
			case 1:
				Scan_WorkEfficient(Context, CommandQueue, LocalWorkSize);
				break;
		}
	}

	//wait until the command queue is empty again
	V_RETURN_CL(clFinish(CommandQueue), "Error finishing the queue!");

	timer.Stop();

	double ms = timer.GetElapsedMilliseconds() / double(nIterations);
	cout << "  average time: " << ms << " ms, throughput: " << 1.0e-6 * (double)m_N / ms << " Gelem/s" <<endl;
}



///////////////////////////////////////////////////////////////////////////////
