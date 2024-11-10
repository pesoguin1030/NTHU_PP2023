#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <mpi.h>

typedef unsigned long long ull;

void initialize_mpi(int *rank, int *size) {
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, rank);
    MPI_Comm_size(MPI_COMM_WORLD, size);
}

ull get_pixel_count(ull r, ull mpiRank, ull mpiSize) {
    ull start = mpiRank * r / mpiSize;
    ull end = (mpiRank + 1) * r / mpiSize;
    ull rSqr = r * r;
    ull pixels = 0;

#pragma omp parallel for reduction(+:pixels)
    for (ull x = start; x < end; x++) {
        ull y = ceil(sqrtl(rSqr - x * x));
        pixels += y;
    }

    return pixels;
}

int main(int argc, char** argv) {
    int mpiRank, mpiSize;
    initialize_mpi(&mpiRank, &mpiSize);

    if (argc != 3) {
        if (mpiRank == 0) { // Only master process outputs the error
            fprintf(stderr, "Usage: %s <r> <k>\n", argv[0]);
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1; // This line will only be reached if MPI_Abort is not effective (which is unlikely)
    }

    ull r = atoll(argv[1]);
    ull k = atoll(argv[2]);

    ull localPixels = get_pixel_count(r, mpiRank, mpiSize);
    
    ull totalPixels;
    MPI_Reduce(&localPixels, &totalPixels, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Finalize();

    if(mpiRank == 0) {
        printf("%llu\n", (4 * (totalPixels % k)) % k);
    }

    return 0;
}