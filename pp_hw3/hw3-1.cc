#include <stdio.h>
#include <stdlib.h>
#include <sched.h>
#include <omp.h>

#define min(a,b) ((a<b)?a:b)

const int INF = 1073741823;
const int V = 6010;
void input(char* inFileName);
void output(char* outFileName);

void block_FW(int B);
int ceil(int a, int b);
void cal(int B, int Round, int block_start_x, int block_start_y, int block_width, int block_height);

int n, m, cpu_num;
static int Dist[V][V];

int main(int argc, char* argv[]) {
    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    cpu_num = CPU_COUNT(&cpu_set);

    input(argv[1]);
    int B = 64;
    block_FW(B);
    output(argv[2]);
    return 0;
}

void input(char* infile) {
    FILE* file = fopen(infile, "rb");
    fread(&n, sizeof(int), 1, file);
    fread(&m, sizeof(int), 1, file);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; j++){
            if (i == j) {
                Dist[i][j] = 0;
            } else {
                Dist[i][j] = INF;
            }
        }
    }

    int pair[3];
    for (int i = 0; i < m; ++i) {
        fread(pair, sizeof(int), 3, file);
        Dist[pair[0]][pair[1]] = pair[2];
    }
    fclose(file);
}

void output(char* outFileName) {
    FILE* outfile = fopen(outFileName, "w");
    for (int i = 0; i < n; ++i) {
        fwrite(Dist[i], sizeof(int), n, outfile);
    }
    fclose(outfile);
}

int ceil(int a, int b) { return (a + b - 1) / b; }

void block_FW(int B) {
    int round = ceil(n, B);
    for (int r = 0; r < round; ++r) {
        /* Phase 1*/
        cal(B, r, r, r, 1, 1);
        /* Phase 2*/
        cal(B, r, r, 0, round, 1);
        cal(B, r, 0, r, 1, round);
        /* Phase 3*/
        cal(B, r, 0, 0, round, round);
    }
}

void cal(int B, int Round, int block_start_x, int block_start_y, int block_width, int block_height) {
    int block_end_x = block_start_x + block_height;
    int block_end_y = block_start_y + block_width;
    //int k_start = Round * B, k_end = min((Round + 1) * B, n);

    #pragma omp parallel for num_threads(cpu_num) schedule(dynamic)
    for (int b_i = block_start_x; b_i < block_end_x; ++b_i) {
        for (int b_j = block_start_y; b_j < block_end_y; ++b_j) {
            for (int k = Round * B; k < (Round + 1) * B && k < n; ++k) {
                int block_internal_start_x = b_i * B;
                int block_internal_end_x = min((b_i + 1) * B, n);
                int block_internal_start_y = b_j * B;
                int block_internal_end_y = min((b_j + 1) * B, n);

                for (int i = block_internal_start_x; i < block_internal_end_x; ++i) {
                    int dist_ik = Dist[i][k];
                    for (int j = block_internal_start_y; j < block_internal_end_y; ++j) {
                        Dist[i][j] = min(Dist[i][j], dist_ik + Dist[k][j]);
                    }
                }
            }
        }
    }
}