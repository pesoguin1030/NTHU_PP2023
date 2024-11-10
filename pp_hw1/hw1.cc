#include "mpi.h"
#include "stdio.h"
#include "stdlib.h"
#include <boost/sort/spreadsort/spreadsort.hpp>
#define	isOdd(x) ((x) & 1)
#define isEven(x) (!((x) & 1))

float *receiveBuffer;
float *temp;
MPI_Comm customWorld = MPI_COMM_WORLD;

void exchangeWithright(float* localBuffer, int rankID, int sendCount, int recvCount) {

	MPI_Sendrecv(localBuffer, sendCount, MPI_FLOAT, rankID + 1, 1, receiveBuffer, recvCount, MPI_FLOAT, rankID + 1, 0, customWorld, MPI_STATUS_IGNORE);

    //preserve samller half
	int x = 0, y = 0;
	for (int i = 0; i < sendCount; i++) {
		if (y==recvCount || (y < recvCount && x < sendCount && localBuffer[x] <= receiveBuffer[y])) 
			temp[i] = localBuffer[x++];
		else 
			temp[i] = receiveBuffer[y++];
	}	
	for (int i = 0; i<sendCount; i++)
		localBuffer[i] = temp[i];
}

void exchangeWithleft(float* localBuffer, int rankID, int sendCount, int recvCount) {

	MPI_Sendrecv(localBuffer, sendCount, MPI_FLOAT, rankID - 1, 0, receiveBuffer, recvCount, MPI_FLOAT, rankID - 1, 1, customWorld, MPI_STATUS_IGNORE);

    //preserve larger half
	int x = sendCount - 1, y = recvCount - 1;
	for (int i = sendCount - 1; i >= 0; i--) {
		if (y<0 || (y>=0 && x>=0 && localBuffer[x] >= receiveBuffer[y])) 
			temp[i] = localBuffer[x--];
		else 
			temp[i] = receiveBuffer[y--];
	}
	for (int i = 0; i<sendCount; i++)
		localBuffer[i] = temp[i];
}


int main(int argc, char *argv[]) {

	int dataPerProcess = 0;		//data amount per process
	int lastProcessCount = 0;
	int generalProcesscount = 0;
	int N = atoi(argv[1]);
	MPI_Group oldGroup, newGroup;
	MPI_Offset offset;

	//MPI Init
	MPI_Init(&argc, &argv);
    int numOfProcess, rankID;
	MPI_Comm_size(customWorld, &numOfProcess);
	MPI_Comm_rank(customWorld, &rankID);

	// if the number of process is larger than N
	if (N < numOfProcess) {
		// obtain the group of processes in the world comm
		MPI_Comm_group(customWorld, &oldGroup);

		// remove unnecessary processes
		int ranges[][3] = { { N, numOfProcess - 1, 1 } };
		MPI_Group_range_excl(oldGroup, 1, ranges, &newGroup);
		MPI_Comm_create(customWorld, newGroup, &customWorld);

		// create new comm
		if (customWorld == MPI_COMM_NULL) {
			MPI_Finalize();
			exit(0);
		}
		numOfProcess = N;
	}
    
	// scatter the data to each process
	dataPerProcess  = (int)(ceil((double)N/(double)numOfProcess));
	generalProcesscount = dataPerProcess;
	lastProcessCount = N- (dataPerProcess * (numOfProcess-1));
	offset   = rankID * dataPerProcess * sizeof(float);
	if(rankID == numOfProcess -1) dataPerProcess = lastProcessCount;

	// create memory space
	float *localBuffer;
	localBuffer = (float*)malloc(dataPerProcess * sizeof(float));
	receiveBuffer = (float*)malloc(generalProcesscount * sizeof(float));
	temp = (float*)malloc(generalProcesscount * sizeof(float));
	
	// file read
	MPI_File fileInput, fileOutput;
	MPI_File_open(customWorld, argv[2], MPI_MODE_RDONLY, MPI_INFO_NULL, &fileInput);
	MPI_File_read_at(fileInput, offset, localBuffer, dataPerProcess, MPI_FLOAT,  MPI_STATUS_IGNORE);
	MPI_File_close(&fileInput);

	//in-process sort
    boost::sort::spreadsort::spreadsort(localBuffer, localBuffer + dataPerProcess);

	// between process odd-even sort
	for (int i = 0; i < numOfProcess ; i++) {
		//odd-phase
		if (isOdd(i)) {
			if (isOdd(rankID)) {
				if (rankID == numOfProcess - 2) 
					exchangeWithright(localBuffer, rankID, generalProcesscount, lastProcessCount);
				else if (rankID != numOfProcess - 1) 
					exchangeWithright(localBuffer, rankID, generalProcesscount, generalProcesscount);
			}

			if (isEven(rankID)) {
				if (rankID == numOfProcess - 1)
					exchangeWithleft(localBuffer, rankID, lastProcessCount, generalProcesscount);
				else if (rankID != 0) 
					exchangeWithleft(localBuffer, rankID, generalProcesscount, generalProcesscount);
			}
		}
		//even-phase
		if (isEven(i)) {
			if (isOdd(rankID)) {
				if (rankID == numOfProcess - 1) 
					exchangeWithleft(localBuffer, rankID, lastProcessCount, generalProcesscount);
				else 
					exchangeWithleft(localBuffer, rankID, generalProcesscount, generalProcesscount);
			}

			if (isEven(rankID)) {
				if (rankID == numOfProcess - 2) 
					exchangeWithright(localBuffer, rankID, generalProcesscount, lastProcessCount);
				else if (rankID != numOfProcess - 1) 
					exchangeWithright(localBuffer, rankID, generalProcesscount, generalProcesscount);
			}
		}		
	}

	MPI_File_open(customWorld, argv[3], MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fileOutput);
	MPI_File_write_at(fileOutput, offset, localBuffer, dataPerProcess, MPI_FLOAT, MPI_STATUS_IGNORE);
	MPI_File_close(&fileOutput);

	free(localBuffer);
	free(receiveBuffer);
	free(temp);
	
	MPI_Finalize();
	return 0;
}