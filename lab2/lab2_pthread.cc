#include <cstdio>
#include <cmath>
#include <pthread.h>
#include <iostream>
#include <vector>
#include <sched.h>

typedef unsigned long long ull;
using namespace std;
ull pixels = 0;

struct ThreadArgs {
    int threadID;
    ull r;
    ull rSqr;
    ull threadNum;
    pthread_mutex_t* mutex;
};

void* calcPixels(void* tArgs) {
    ThreadArgs* args = static_cast<ThreadArgs*>(tArgs);
    ull tmpPixels = 0;
    ull start = args->threadID * (args->r / args->threadNum);
    ull end = (args->threadID + 1) * (args->r / args->threadNum);

    if (args->threadID == args->threadNum - 1) {
        end = args->r;
    }

    for (ull x = start; x < end; x++) {
        ull y = ceil(sqrtl(args->rSqr - x * x));
        tmpPixels += y;
    }

    pthread_mutex_lock(args->mutex);
    extern ull pixels;
    pixels += tmpPixels;
    pthread_mutex_unlock(args->mutex);

    delete args;
    pthread_exit(NULL);
}

int main(int argc, char** argv) {
    if (argc != 3) {
        cerr << "Usage: " << argv[0] << " <r> <k>" << endl;
        return 1;
    }

    ull r = atoll(argv[1]);
    ull k = atoll(argv[2]);
    ull rSqr = r * r;

    cpu_set_t cpuSet;
    sched_getaffinity(0, sizeof(cpuSet), &cpuSet);
    ull cpuNum = CPU_COUNT(&cpuSet);

    ull threadNum = cpuNum * 2;
    vector<pthread_t> threads(threadNum);
    pthread_mutex_t pixelMutex;
    pthread_mutex_init(&pixelMutex, NULL);

    for (int i = 0; i < threadNum; i++) {
        ThreadArgs* args = new ThreadArgs{ i, r, rSqr, threadNum, &pixelMutex };
        int rc = pthread_create(&threads[i], NULL, calcPixels, static_cast<void*>(args));
        if (rc) {
            cerr << "ERROR; return code from pthread_create() is " << rc << endl;
            exit(-1);
        }
    }

    for (int i = 0; i < threadNum; i++) {
        pthread_join(threads[i], NULL);
    }

    pthread_mutex_destroy(&pixelMutex);
    cout << (4 * (pixels % k)) % k << endl;

    return 0;
}