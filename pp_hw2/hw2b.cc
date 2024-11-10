#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <png.h>
#include <omp.h>
#include <sched.h>

int cpu_num;

typedef struct {
    int start_row;
    int end_row;
} Task;

typedef struct {
    double left, right, lower, upper;
    int width, height, iters;
    int* image;
    int start_row, end_row;
} mandelbrot_args;

void write_png(const char* filename, int iters, int width, int height, const int* buffer) {
    FILE* fp = fopen(filename, "wb");
    assert(fp);
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    assert(png_ptr);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    assert(info_ptr);
    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_set_filter(png_ptr, 0, PNG_NO_FILTERS);
    png_write_info(png_ptr, info_ptr);
    png_set_compression_level(png_ptr, 1);
    size_t row_size = 3 * width * sizeof(png_byte);
    png_bytep row = (png_bytep)malloc(row_size);
    for (int y = 0; y < height; ++y) {
        memset(row, 0, row_size);
        for (int x = 0; x < width; ++x) {
            //int p = buffer[(height - 1 - y) * width + x];
            int p = buffer[(height - 1 - y) * width + x];
            png_bytep color = row + x * 3;
            if (p != iters) {
                if (p & 16) {
                    color[0] = 240;
                    color[1] = color[2] = p % 16 * 16;
                } else {
                    color[0] = p % 16 * 16;
                }
            }
        }
        png_write_row(png_ptr, row);
    }
    free(row);
    png_write_end(png_ptr, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
}

void compute_mandelbrot(mandelbrot_args* args) {
    #pragma omp parallel for schedule(dynamic)
    for (int j = args->start_row; j  < args->end_row; ++j) {
        for (int i = 0; i < args->width; ++i) {
            double y0 = j * ((args->upper - args->lower) / args->height) + args->lower;
            double x0 = i * ((args->right - args->left) / args->width) + args->left;

            int repeats = 0;
            double x = 0;
            double y = 0;
            double length_squared = 0;
            while (repeats < args->iters && length_squared < 4) {
                double temp = x * x - y * y + x0;
                y = 2 * x * y + y0;
                x = temp;
                length_squared = x * x + y * y;
                ++repeats;
            }
            args->image[(j - args->start_row) * args->width + i] = repeats;
        }
    }
}

int main(int argc, char** argv) {
    // Initialize the MPI environment
    MPI_Init(&argc, &argv);

    // Get the number of processes and the rank of this process
    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    cpu_num = CPU_COUNT(&cpu_set);

    // Parameters (you might want to parse them from command line arguments)
    const char* filename = argv[1];
    int iters = strtol(argv[2], 0, 10);
    double left = strtod(argv[3], 0);
    double right = strtod(argv[4], 0);
    double lower = strtod(argv[5], 0);
    double upper = strtod(argv[6], 0);
    int width = strtol(argv[7], 0, 10);
    int height = strtol(argv[8], 0, 10);

    int rows_per_task = 10; // or some other appropriate value


    if (world_rank == 0) {
        // Master process logic
        const int num_tasks = (height + rows_per_task - 1) / rows_per_task;
        int next_task = 0;

        while (next_task < num_tasks) {
            // Receive a task request from any worker
            int worker_rank;
            MPI_Recv(&worker_rank, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            // Send the next task to the requesting worker
            Task task;
            task.start_row = next_task * rows_per_task;
            task.end_row = (next_task + 1) * rows_per_task;
            if (task.end_row > height) {
                task.end_row = height;
            }
            //printf("Sending task: start_row = %d, end_row = %d to worker %d\n", task.start_row, task.end_row, worker_rank);
            MPI_Send(&task, 2, MPI_INT, worker_rank, 0, MPI_COMM_WORLD);

            next_task++;
        }

        

        //end send

        // new one
        // Allocate memory for the full image
        int* full_image = (int*)malloc(width * height * sizeof(int));
        MPI_Status status;

        // Receive results from workers
        for (int i = 1; i < num_tasks+1; ++i) {
            // Receive the task details first to know where to place the data
            Task task;
            //MPI_Recv(&task, 2, MPI_INT, i, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&task, 2, MPI_INT, MPI_ANY_SOURCE, 1, MPI_COMM_WORLD, &status);
            //printf("Received task: start_row = %d, end_row = %d\n", task.start_row, task.end_row);
            
            //MPI_Recv(&task, sizeof(Task), MPI_BYTE, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
            // Calculate the number of rows and the offset in the full image
            int rows = task.end_row - task.start_row;
            /*if (rows <= 0) {
                fprintf(stderr, "Error: Invalid row count received from worker %d: start_row = %d, end_row = %d\n", status.MPI_SOURCE, task.start_row, task.end_row);
                continue; // Skip this iteration
            }*/

            int offset = task.start_row * width;

            // Receive the computed data from the worker
            MPI_Recv(full_image + offset, rows* width, MPI_INT, status.MPI_SOURCE, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }        

        // Write the full image to a file
        write_png(filename, iters, width, height, full_image);

        // Free the allocated memory
        free(full_image);

        // Send a termination signal to all workers
        Task terminate_task = {-1, -1};
        for (int i = 1; i < world_size; ++i) {
            MPI_Send(&terminate_task, 2, MPI_INT, i, 0, MPI_COMM_WORLD);
        }
    } else {
        // Worker process logic 
        while (true) {
            // Request a task from the master
            MPI_Send(&world_rank, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);

            // Receive the task (row range) from the master
            Task task;
            MPI_Recv(&task, 2, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            //printf("Worker %d received task: start_row = %d, end_row = %d\n", world_rank, task.start_row, task.end_row);

            // Check for termination signal
            if (task.start_row == -1) {
                break;
            }

            // Allocate memory for the local computation
            int local_height = task.end_row - task.start_row;
            int* local_image = (int*)malloc(width * local_height * sizeof(int));

            // Prepare the arguments for the Mandelbrot computation
            mandelbrot_args args = {left, right, lower, upper, width, height, iters, local_image, task.start_row, task.end_row};

            // Compute the Mandelbrot set in parallel using OpenMP
            compute_mandelbrot(&args);

            // Send the task information back to the master
            MPI_Send(&task, 2, MPI_INT, 0, 1, MPI_COMM_WORLD);

            // Send the results back to the master
            // Assuming the master process is prepared to receive the results immediately
            MPI_Send(local_image, width * local_height, MPI_INT, 0, 1, MPI_COMM_WORLD);

            // Free the allocated memory
            free(local_image);
        }
    }

    MPI_Finalize();
    return 0;
}