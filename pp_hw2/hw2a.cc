#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define PNG_NO_SETJMP
#include <sched.h>
#include <assert.h>
#include <png.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
//#define max(a,b) (((a)>(b))?(a):(b))

typedef struct {
    int start_line;
    int end_line;
    double left;
    double right;
    double lower;
    double upper;
    int width;
    int height;
    int iters;
    int* image;
} task_t;

typedef struct {
    task_t* tasks;
    int total_tasks;
    int next_task;
    pthread_mutex_t mutex;
} work_queue_t;

void work_queue_init(work_queue_t* queue, int total_tasks) {
    queue->tasks = (task_t*)malloc(total_tasks * sizeof(task_t));
    queue->total_tasks = total_tasks;
    queue->next_task = 0;
    pthread_mutex_init(&queue->mutex, NULL);
}

bool work_queue_get_task(work_queue_t* queue, task_t* task) {
    pthread_mutex_lock(&queue->mutex);
    if (queue->next_task >= queue->total_tasks) {
        pthread_mutex_unlock(&queue->mutex);
        return false;
    }
    *task = queue->tasks[queue->next_task++];
    pthread_mutex_unlock(&queue->mutex);
    return true;
}

void work_queue_destroy(work_queue_t* queue) {
    free(queue->tasks);
    pthread_mutex_destroy(&queue->mutex);
}

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

void* compute_mandelbrot(void* arg) {
    work_queue_t* queue = (work_queue_t*)arg;
    task_t task;

    while (work_queue_get_task(queue, &task)) {
        for (int j = task.start_line; j < task.end_line; ++j) {
            double y0 = (j * ((task.upper - task.lower) / task.height)) + task.lower;
            for (int i = 0; i < task.width; ++i) {
                double x0 = (i * ((task.right - task.left) / task.width)) + task.left;

                int repeats = 0;
                double x = 0;
                double y = 0;
                double length_squared = 0;
                while (repeats < task.iters && length_squared < 4) {
                    double temp = x * x - y * y + x0;
                    y = 2 * x * y + y0;
                    x = temp;
                    length_squared = x * x + y * y;
                    ++repeats;
                }
                task.image[j * task.width + i] = repeats;
            }
        }
    }

    return NULL;
}

int main(int argc, char** argv) {

    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    assert(argc == 9);
    const char* filename = argv[1];
    int iters = strtol(argv[2], 0, 10);
    double left = strtod(argv[3], 0);
    double right = strtod(argv[4], 0);
    double lower = strtod(argv[5], 0);
    double upper = strtod(argv[6], 0);
    int width = strtol(argv[7], 0, 10);
    int height = strtol(argv[8], 0, 10);

    /* allocate memory for image */
    int* image = (int*)malloc(width * height * sizeof(int));
    assert(image);

    // Determine the number of tasks
    int num_threads = CPU_COUNT(&cpu_set); 
    //int num_tasks = max(height / 10, num_threads * 2);
    int num_tasks = height;
    
    // Initialize work queue
    work_queue_t queue;
    work_queue_init(&queue, num_tasks);

    // Populate tasks in the queue
    int lines_per_task = height / num_tasks;
    for (int i = 0; i < num_tasks; ++i) {
        queue.tasks[i].start_line = i * lines_per_task;
        queue.tasks[i].end_line = (i == num_tasks - 1) ? height : (i + 1) * lines_per_task;
        queue.tasks[i].left = left;
        queue.tasks[i].right = right;
        queue.tasks[i].lower = lower;
        queue.tasks[i].upper = upper;
        queue.tasks[i].width = width;
        queue.tasks[i].height = height;
        queue.tasks[i].iters = iters;
        queue.tasks[i].image = image;
    }

    // Create threads
    pthread_t threads[num_threads];
    for (int i = 0; i < num_threads; ++i) {
        pthread_create(&threads[i], NULL, compute_mandelbrot, &queue);
    }

    // Join threads
    for (int i = 0; i < num_threads; ++i) {
        pthread_join(threads[i], NULL);
    }

    // Write the image to a file
    write_png(filename, iters, width, height, image);

    // Cleanup
    free(image);
    work_queue_destroy(&queue);

    return 0;
}