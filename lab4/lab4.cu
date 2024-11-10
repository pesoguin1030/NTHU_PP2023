#include <iostream>
#include <cstdlib>
#include <cassert>
#include <zlib.h>
#include <png.h>

#define Z 2
#define Y 5
#define X 5
#define xB 2
#define yB 2
#define SCALE 8

#define T 256

int read_png(const char* filename, unsigned char** image, unsigned* height, 
             unsigned* width, unsigned* channels) {

    unsigned char sig[8];
    FILE* infile;
    infile = fopen(filename, "rb");

    fread(sig, 1, 8, infile);
    if (!png_check_sig(sig, 8))
        return 1;   /* bad signature */

    png_structp png_ptr;
    png_infop info_ptr;

    png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png_ptr)
        return 4;   /* out of memory */
  
    info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr) {
        png_destroy_read_struct(&png_ptr, NULL, NULL);
        return 4;   /* out of memory */
    }

    png_init_io(png_ptr, infile);
    png_set_sig_bytes(png_ptr, 8);
    png_read_info(png_ptr, info_ptr);
    int bit_depth, color_type;
    png_get_IHDR(png_ptr, info_ptr, width, height, &bit_depth, &color_type, NULL, NULL, NULL);

    png_uint_32  i, rowbytes;
    png_bytep  row_pointers[*height];
    png_read_update_info(png_ptr, info_ptr);
    rowbytes = png_get_rowbytes(png_ptr, info_ptr);
    *channels = (int) png_get_channels(png_ptr, info_ptr);

    if ((*image = (unsigned char *) malloc(rowbytes * *height)) == NULL) {
        png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
        return 3;
    }

    for (i = 0;  i < *height;  ++i)
        row_pointers[i] = *image + i * rowbytes;
    png_read_image(png_ptr, row_pointers);
    png_read_end(png_ptr, NULL);
    return 0;
}

void write_png(const char* filename, png_bytep image, const unsigned height, const unsigned width, 
               const unsigned channels) {
    FILE* fp = fopen(filename, "wb");
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr, width, height, 8,
                 PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_set_filter(png_ptr, 0, PNG_NO_FILTERS);
    png_write_info(png_ptr, info_ptr);
    png_set_compression_level(png_ptr, 1);

    png_bytep row_ptr[height];
    for (int i = 0; i < height; ++ i) {
        row_ptr[i] = image + i * width * channels * sizeof(unsigned char);
    }
    png_write_image(png_ptr, row_ptr);
    png_write_end(png_ptr, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
}

__constant__ char mask[Z][Y][X] = { { { -1, -4, -6, -4, -1 },
                                        { -2, -8, -12, -8, -2 },
                                        { 0, 0, 0, 0, 0 },
                                        { 2, 8, 12, 8, 2 },
                                        { 1, 4, 6, 4, 1 } },
                                      { { -1, -2, 0, 2, 1 },
                                        { -4, -8, 0, 8, 4 },
                                        { -6, -12, 0, 12, 6 },
                                        { -4, -8, 0, 8, 4 },
                                        { -1, -2, 0, 2, 1 } } };


__global__ void sobel(unsigned char* src, unsigned char* dst, unsigned height, unsigned width, unsigned channels) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= width) return;
    int x = tid, idx = threadIdx.x, now = yB;

    // Shared memory for the Sobel mask and source image
    __shared__ char sharedMask[Z][Y][X];
    __shared__ unsigned char sharedSrc[Y][T + xB + xB][3];

    // Initialize shared memory for the Sobel mask
    for (int z = 0; z < Z; ++z) {
        for (int y = 0; y < Y; ++y) {
            for (int x = 0; x < X; ++x) {
                sharedMask[z][y][x] = mask[z][y][x];
            }
        }
    }

    // Load source image into shared memory
    for (int v = -yB; v <= yB; ++v) {
        for (int u = -xB; u <= xB; ++u) {
            int flag = x + u >= 0 && x + u < width && v >= 0 && v < height;
            for (int c = 0; c < 3; ++c) {
                sharedSrc[v + yB][u + (idx + xB)][c] = flag ? src[channels * (width * (0 + v) + (x + u)) + c] : 0;
            }
        }
    }

    // Apply Sobel filter
    for (int y = 0; y < height; ++y) {
        float val[2][3] = {0}; // For storing the convolution results

        __syncthreads();
        for (int v = -yB; v <= yB; ++v) {
            for (int u = -xB; u <= xB; ++u) {
                for (int z = 0; z < Z; ++z) {
                    for (int c = 0; c < 3; ++c) {
                        unsigned char pixelValue = sharedSrc[(v + now + Y) % Y][u + (idx + xB)][c];
                        val[z][c] += pixelValue * sharedMask[z][u + xB][v + yB];
                    }
                }
            }
        }

        // Update shared memory for the next iteration
        for (int u = -xB; u <= xB; ++u) {
            int flag = x + u >= 0 && x + u < width && y + 1 + yB >= 0 && y + 1 + yB < height;
            for (int c = 0; c < 3; ++c) {
                sharedSrc[(-yB + now + Y) % Y][u + (idx + xB)][c] = flag ? src[channels * (width * (y + 1 + yB) + (x + u)) + c] : 0;
            }
        }
        now = (now + 1) % Y;

        // Write the result to the destination image
        for (int c = 0; c < 3; ++c) {
            float magnitude = sqrt(val[0][c] * val[0][c] + val[1][c] * val[1][c]) / SCALE;
            dst[channels * (width * y + x) + c] = min(255.0f, magnitude);
        }
    }
}

int main(int argc, char **argv) {
    assert(argc == 3);
    unsigned height, width, channels;
    unsigned char *src = NULL, *dst;
    unsigned char *dsrc, *ddst;

    /* read the image to src, and get height, width, channels */
    if (read_png(argv[1], &src, &height, &width, &channels)) {
        std::cerr << "Error in read png" << std::endl;
        return -1;
    }

    // decide to use how many blocks and threads
    const int num_threads = T;
    const int num_blocks = width / num_threads + 1;

    dst = (unsigned char *)malloc(height * width * channels * sizeof(unsigned char));
    cudaHostRegister(src, height * width * channels * sizeof(unsigned char), cudaHostRegisterDefault);

    // cudaMalloc(...) for device src and device dst
    cudaMalloc(&dsrc, height * width * channels * sizeof(unsigned char));
    cudaMalloc(&ddst, height * width * channels * sizeof(unsigned char));

    // cudaMemcpy(...) copy source image to device (mask matrix if necessary)
    cudaMemcpy(dsrc, src, height * width * channels * sizeof(unsigned char), cudaMemcpyHostToDevice);

    // launch cuda kernel
    sobel <<<num_blocks, num_threads>>> (dsrc, ddst, height, width, channels);

    // cudaMemcpy(...) copy result image to host
    cudaMemcpy(dst, ddst, height * width * channels * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    write_png(argv[2], dst, height, width, channels);
    free(src);
    free(dst);
    cudaFree(dsrc);
    cudaFree(ddst);
    return 0;
}
