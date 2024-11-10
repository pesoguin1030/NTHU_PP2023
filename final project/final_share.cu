#include <iostream>
#include <cstdlib>
#include <cassert>
#include <zlib.h>
#include <png.h>
#include <sys/time.h>
#define US_PER_SEC 1000000

#define BLOCK_SIZE 32
#define max_iterations 500
#define convergence_threshold 1.0f
#define bandwidth 40
#define gaussian_sigma 10


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

// __global__ void meanshift(unsigned char* src, unsigned char* dst, int height, int width, int channels){
//     __shared__ float shared_data[BLOCK_SIZE][3];  // 假设最大颜色通道为3

//     int tx = threadIdx.x;
//     int ty = threadIdx.y;
//     int bx = blockIdx.x;
//     int by = blockIdx.y;

//     int idx = bx * blockDim.x + tx;
//     int idy = by * blockDim.y + ty;

//     if (idx >= width || idy >= height) return;

//     int pixelPos = (idy * width + idx) * channels;
//     float shift_x = 0.0, shift_y = 0.0;
//     float r, g, b;

//     // 初始位置的颜色
//     r = src[pixelPos];
//     g = src[pixelPos + 1];
//     b = src[pixelPos + 2];

//     for (int iter = 0; iter < max_iterations; ++iter) {
//         float sum_x = 0.0, sum_y = 0.0;
//         float sum_r = 0.0, sum_g = 0.0, sum_b = 0.0;
//         float weight_sum = 0.0;

//         // 在带宽内遍历像素
//         for (int dy = -bandwidth; dy <= bandwidth; ++dy) {
//             for (int dx = -bandwidth; dx <= bandwidth; ++dx) {
//                 int new_x = idx + dx;
//                 int new_y = idy + dy;

//                 // 确保索引在图像范围内
//                 if (new_x >= 0 && new_x < width && new_y >= 0 && new_y < height) {
//                     int newPixelPos = (new_y * width + new_x) * channels;

//                     // 计算颜色距离
//                     float r_diff = r - src[newPixelPos];
//                     float g_diff = g - src[newPixelPos + 1];
//                     float b_diff = b - src[newPixelPos + 2];
//                     float color_distance = r_diff * r_diff + g_diff * g_diff + b_diff * b_diff;

//                     // 使用高斯核函数计算权重
//                     float weight = expf(-color_distance / (2 * gaussian_sigma * gaussian_sigma));

//                     sum_x += new_x * weight;
//                     sum_y += new_y * weight;
//                     sum_r += src[newPixelPos] * weight;
//                     sum_g += src[newPixelPos + 1] * weight;
//                     sum_b += src[newPixelPos + 2] * weight;
//                     weight_sum += weight;
//                 }
//             }
//         }

//         // 计算新的颜色和位置
//         r = sum_r / weight_sum;
//         g = sum_g / weight_sum;
//         b = sum_b / weight_sum;
//         shift_x = (sum_x / weight_sum) - idx;
//         shift_y = (sum_y / weight_sum) - idy;

//         // 检查是否达到收敛条件
//         if (shift_x * shift_x + shift_y * shift_y < convergence_threshold * convergence_threshold) {
//             break;
//         }
//     }

//     // 更新目标图像
//     dst[pixelPos] = static_cast<unsigned char>(r);
//     dst[pixelPos + 1] = static_cast<unsigned char>(g);
//     dst[pixelPos + 2] = static_cast<unsigned char>(b);
// }

// __global__ void meanshift(unsigned char* src, unsigned char* dst, int height, int width, int channels){
//     __shared__ float shared_data[BLOCK_SIZE + 2 * bandwidth][BLOCK_SIZE + 2 * bandwidth][3];  // 擴大共享內存以包括邊界像素

//     int tx = threadIdx.x;
//     int ty = threadIdx.y;
//     int bx = blockIdx.x;
//     int by = blockIdx.y;

//     int idx = bx * blockDim.x + tx;
//     int idy = by * blockDim.y + ty;

//     int sharedIdx = tx + bandwidth;
//     int sharedIdy = ty + bandwidth;

//     // 將目標像素加載到共享內存
//     if (idx < width && idy < height) {
//         int pixelPos = (idy * width + idx) * channels;
//         for (int c = 0; c < channels; c++) {
//             shared_data[sharedIdx][sharedIdy][c] = src[pixelPos + c];
//         }
//     }

//     // 加載周邊像素到共享內存（包括邊界檢查）
//     for (int dy = -bandwidth; dy <= bandwidth; dy++) {
//         for (int dx = -bandwidth; dx <= bandwidth; dx++) {
//             int global_x = idx + dx;
//             int global_y = idy + dy;
//             int shared_x = sharedIdx + dx;
//             int shared_y = sharedIdy + dy;

//             if (shared_x >= 0 && shared_x < BLOCK_SIZE + 2 * bandwidth &&
//                 shared_y >= 0 && shared_y < BLOCK_SIZE + 2 * bandwidth) {
//                 if (global_x >= 0 && global_x < width && global_y >= 0 && global_y < height) {
//                     int global_pixel_pos = (global_y * width + global_x) * channels;
//                     for (int c = 0; c < channels; c++) {
//                         shared_data[shared_x][shared_y][c] = src[global_pixel_pos + c];
//                     }
//                 } else {
//                     // 如果是圖像外的像素，設置為0或其他合適的默認值
//                     for (int c = 0; c < channels; c++) {
//                         shared_data[shared_x][shared_y][c] = 0;
//                     }
//                 }
//             }
//         }
//     }

//     __syncthreads();  // 確保共享內存加載完成

//     float shift_x = 0.0, shift_y = 0.0;
//     float r, g, b;
//     r = shared_data[sharedIdx][sharedIdy][0];
//     g = shared_data[sharedIdx][sharedIdy][1];
//     b = shared_data[sharedIdx][sharedIdy][2];

//     for (int iter = 0; iter < max_iterations; ++iter) {
//         float sum_x = 0.0, sum_y = 0.0;
//         float sum_r = 0.0, sum_g = 0.0, sum_b = 0.0;
//         float weight_sum = 0.0;

//         // 在带宽内遍历像素
//         for (int dy = -bandwidth; dy <= bandwidth; ++dy) {
//             for (int dx = -bandwidth; dx <= bandwidth; ++dx) {
//                 int new_x = idx + dx;
//                 int new_y = idy + dy;
//                 int shared_new_x = sharedIdx + dx;
//                 int shared_new_y = sharedIdy + dy;

//                 // 确保索引在共享内存的范围内
//                 if (shared_new_x >= 0 && shared_new_x < BLOCK_SIZE + 2 * bandwidth && shared_new_y >= 0 && shared_new_y < BLOCK_SIZE + 2 * bandwidth) {
//                     float r_diff = r - shared_data[shared_new_x][shared_new_y][0];
//                     float g_diff = g - shared_data[shared_new_x][shared_new_y][1];
//                     float b_diff = b - shared_data[shared_new_x][shared_new_y][2];
//                     float color_distance = r_diff * r_diff + g_diff * g_diff + b_diff * b_diff;

//                     float weight = expf(-color_distance / (2 * gaussian_sigma * gaussian_sigma));

//                     sum_x += new_x * weight;
//                     sum_y += new_y * weight;
//                     sum_r += shared_data[shared_new_x][shared_new_y][0] * weight;
//                     sum_g += shared_data[shared_new_x][shared_new_y][1] * weight;
//                     sum_b += shared_data[shared_new_x][shared_new_y][2] * weight;
//                     weight_sum += weight;
//                 }
//             }
//         }

//         r = sum_r / weight_sum;
//         g = sum_g / weight_sum;
//         b = sum_b / weight_sum;
//         shift_x = (sum_x / weight_sum) - idx;
//         shift_y = (sum_y / weight_sum) - idy;

//         if (shift_x * shift_x + shift_y * shift_y < convergence_threshold * convergence_threshold) {
//             break;
//         }
//     }

//     if (idx < width && idy < height) {
//         int pixelPos = (idy * width + idx) * channels;
//         dst[pixelPos] = static_cast<unsigned char>(r);
//         dst[pixelPos + 1] = static_cast<unsigned char>(g);
//         dst[pixelPos + 2] = static_cast<unsigned char>(b);
//     }
// }

__global__ void meanshift(unsigned char* src, unsigned char* dst, int height, int width, int channels){
    __shared__ float shared_data[BLOCK_SIZE][BLOCK_SIZE][3];  // 只存儲核心區塊的像素

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int idx = bx * blockDim.x + tx;
    int idy = by * blockDim.y + ty;

    // 將核心區塊的像素加載到共享內存
    if (idx < width && idy < height) {
        int pixelPos = (idy * width + idx) * channels;
        for (int c = 0; c < channels; c++) {
            shared_data[tx][ty][c] = src[pixelPos + c];
        }
    }

    __syncthreads();  // 確保共享內存加載完成

    if (idx >= width || idy >= height) return;

    float shift_x = 0.0, shift_y = 0.0;
    float r, g, b;
    // 從共享內存讀取初始顏色
    r = shared_data[tx][ty][0];
    g = shared_data[tx][ty][1];
    b = shared_data[tx][ty][2];

    for (int iter = 0; iter < max_iterations; ++iter) {
        float sum_x = 0.0, sum_y = 0.0;
        float sum_r = 0.0, sum_g = 0.0, sum_b = 0.0;
        float weight_sum = 0.0;

        // 在带宽内遍历像素
        for (int dy = -bandwidth; dy <= bandwidth; ++dy) {
            for (int dx = -bandwidth; dx <= bandwidth; ++dx) {
                int new_x = idx + dx;
                int new_y = idy + dy;
                int shared_new_x = tx + dx;
                int shared_new_y = ty + dy;

                float r_diff, g_diff, b_diff, color_distance, weight;
                int global_pixel_pos = (new_y * width + new_x) * channels;  // 在此處定義global_pixel_pos

                
                r_diff = r - shared_data[shared_new_x][shared_new_y][0];
                g_diff = g - shared_data[shared_new_x][shared_new_y][1];
                b_diff = b - shared_data[shared_new_x][shared_new_y][2];
                // 檢查是否可以從共享內存讀取
                // if (shared_new_x >= 0 && shared_new_x < BLOCK_SIZE && shared_new_y >= 0 && shared_new_y < BLOCK_SIZE) {
                //     r_diff = r - shared_data[shared_new_x][shared_new_y][0];
                //     g_diff = g - shared_data[shared_new_x][shared_new_y][1];
                //     b_diff = b - shared_data[shared_new_x][shared_new_y][2];
                // } else {
                //     // 從全局內存讀取
                //     if (new_x >= 0 && new_x < width && new_y >= 0 && new_y < height) {
                //         int global_pixel_pos = (new_y * width + new_x) * channels;
                //         r_diff = r - src[global_pixel_pos];
                //         g_diff = g - src[global_pixel_pos + 1];
                //         b_diff = b - src[global_pixel_pos + 2];
                //     } else {
                //         // 超出圖像邊界，跳過
                //         continue;
                //     }
                // }

                color_distance = r_diff * r_diff + g_diff * g_diff + b_diff * b_diff;
                weight = expf(-color_distance / (2 * gaussian_sigma * gaussian_sigma));

                sum_x += new_x * weight;
                sum_y += new_y * weight;
                sum_r += (shared_new_x >= 0 && shared_new_x < BLOCK_SIZE && shared_new_y >= 0 && shared_new_y < BLOCK_SIZE) ? shared_data[shared_new_x][shared_new_y][0] * weight : src[global_pixel_pos] * weight;
                sum_g += (shared_new_x >= 0 && shared_new_x < BLOCK_SIZE && shared_new_y >= 0 && shared_new_y < BLOCK_SIZE) ? shared_data[shared_new_x][shared_new_y][1] * weight : src[global_pixel_pos + 1] * weight;
                sum_b += (shared_new_x >= 0 && shared_new_x < BLOCK_SIZE && shared_new_y >= 0 && shared_new_y < BLOCK_SIZE) ? shared_data[shared_new_x][shared_new_y][2] * weight : src[global_pixel_pos + 2] * weight;
                weight_sum += weight;
            }
        }

        r = sum_r / weight_sum;
        g = sum_g / weight_sum;
        b = sum_b / weight_sum;
        shift_x = (sum_x / weight_sum) - idx;
        shift_y = (sum_y / weight_sum) - idy;

        if (shift_x * shift_x + shift_y * shift_y < convergence_threshold * convergence_threshold) {
            break;
        }
    }

    // 更新目標圖像
    if (idx < width && idy < height) {
        int pixelPos = (idy * width + idx) * channels;
        dst[pixelPos] = static_cast<unsigned char>(r);
        dst[pixelPos + 1] = static_cast<unsigned char>(g);
        dst[pixelPos + 2] = static_cast<unsigned char>(b);
    }
}


int main(int argc, char **argv) {
    struct timeval start, end;
    double time;
    gettimeofday(&start, NULL);


    assert(argc == 3);
    unsigned height, width, channels;
    unsigned char *src = NULL, *dst;
    unsigned char *dsrc, *ddst;

    // 读取图像
    if (read_png(argv[1], &src, &height, &width, &channels)) {
        std::cerr << "Error reading PNG file" << std::endl;
        return -1;
    }

    // 分配内存
    dst = (unsigned char *)malloc(height * width * channels * sizeof(unsigned char));
    cudaHostRegister(src, height * width * channels * sizeof(unsigned char), cudaHostRegisterDefault);

    // 在设备上分配内存
    cudaMalloc(&dsrc, height * width * channels * sizeof(unsigned char));
    cudaMalloc(&ddst, height * width * channels * sizeof(unsigned char));

    // 将源图像复制到设备
    cudaMemcpy(dsrc, src, height * width * channels * sizeof(unsigned char), cudaMemcpyHostToDevice);

    // 设置CUDA核的尺寸和数量
    dim3 threadsPerBlock(32,32);
    dim3 numBlocks(ceil(width / 32), ceil(height / 32));

    // 启动MeanShift CUDA核心函数
    meanshift <<<numBlocks, threadsPerBlock>>> (dsrc, ddst, height, width, channels);

    // 将结果复制回主机
    cudaMemcpy(dst, ddst, height * width * channels * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    // 将处理后的图像保存到文件
    write_png(argv[2], dst, height, width, channels);

    // 清理
    free(src);
    free(dst);
    cudaFree(dsrc);
    cudaFree(ddst);

    gettimeofday(&end, NULL);
    time = (double)(US_PER_SEC*(end.tv_sec-start.tv_sec)+(end.tv_usec-start.tv_usec))/US_PER_SEC;
    printf("Time: %.2lf\n", time);
    return 0;

    return 0;
}
