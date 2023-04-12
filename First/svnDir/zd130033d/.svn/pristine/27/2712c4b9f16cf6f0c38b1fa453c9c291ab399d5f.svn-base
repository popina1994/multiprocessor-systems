#define _CRT_SECURE_NO_WARNINGS

#include <stdio.h>
# include <stdlib.h>
# include <stdio.h>
# include <math.h>
# include <time.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

# include <string.h>

#define NUM_OF_GPU_THREADS 1024
#define NUM_DIR 4
#define TILE_WIDTH 32
#define TILE_HEIGHT 32

const double ACCURACY = 0.01;

int main(int argc, char *argv[]);
double cpu_time(void);

__device__ __constant__ int mGPU;
__device__ __constant__ int nGPU;
__device__ __constant__ double epsilonGPU;
__device__ __constant__ int tilesWidthGPU;
__device__ __constant__ double sideVal;

__device__ void calcRowCol(int& row, int& col)
{
    row = TILE_WIDTH * blockIdx.x + threadIdx.x;
    col = TILE_HEIGHT * blockIdx.y + threadIdx.y;
}

__device__ int calcIndex(int uRowIdx, int uColIdx, int dispX, int dispY)
{
    return (uRowIdx + 1 + dispX) * (TILE_WIDTH + 2) + uColIdx + dispY + 1;
}

__device__ int linIndex(int row, int col)
{
    return row * nGPU + col;
}

__global__ void updateMatrixKernel(double * w, double *diffTiles, double *wOut)
{
    extern __shared__ double u[];
    double val = 0.0;
    int row, col, idx, idxTile;
    int rowBegin, colBegin;
    double *diff;
    const int DISP_ROW[NUM_DIR] = { 0, 0, -1, 1 };
    const int DISP_COL[NUM_DIR] = { -1, 1, 0, 0 };
    double oldVal = 0.0;

    diff = &u[ (TILE_WIDTH + 2) * (TILE_HEIGHT + 2)];
    idxTile = blockIdx.x * tilesWidthGPU + blockIdx.y;

    rowBegin = TILE_WIDTH * blockIdx.x;
    colBegin = TILE_HEIGHT * blockIdx.y;

    row = rowBegin + threadIdx.x;
    col = colBegin + threadIdx.y;


    // Copy vals to shared memory.
    //
    if ( (row < mGPU) && (col < nGPU))
    {
        u[calcIndex(threadIdx.x, threadIdx.y, 0, 0)] = w[linIndex(row, col)];
        oldVal = u[calcIndex(threadIdx.x, threadIdx.y, 0, 0)];
    }

    //printf("Stara vrednost %d %d %.10lf\n", row, col, oldVal);

    //printf("Vrsta: %d Kolona %d x: %d y; %d \n", row, col, threadIdx.x, threadIdx.y);

    switch (threadIdx.x)
    {
        case 0:
            if ( (rowBegin - 1 >= 0) && ( (colBegin + threadIdx.y) < nGPU) )
            {
                u[calcIndex(0, threadIdx.y, -1, 0)] =
                    w[linIndex(rowBegin-1, colBegin + threadIdx.y)];
            }
            break;
        case 1:
            if ((rowBegin + TILE_HEIGHT < mGPU) && ((colBegin + threadIdx.y) < nGPU))
            {
                u[calcIndex(TILE_HEIGHT, threadIdx.y, 0, 0)] =
                    w[linIndex(rowBegin + TILE_HEIGHT, colBegin + threadIdx.y)];


            }
            break;
        case 2:
            if ((rowBegin + threadIdx.y < mGPU) && (colBegin - 1 >= 0))
            {
                u[calcIndex(threadIdx.y, 0, 0, -1)] =
                    w[linIndex(rowBegin + threadIdx.y, colBegin-1)];
            }
            break;
        case 3:
            if ((rowBegin + threadIdx.y < mGPU) && (colBegin + TILE_WIDTH < nGPU))
            {
                u[calcIndex(threadIdx.y, TILE_WIDTH, 0, 0)] =
                    w[linIndex(rowBegin + threadIdx.y, colBegin + TILE_WIDTH)];
            }
            break;
        default:
            break;;
    }

    __syncthreads();

    if ( (row > 0) && (row < mGPU - 1) && (col > 0) && (col < nGPU - 1))
    {
        for (idx = 0; idx < NUM_DIR; idx++)
        {
            //printf("Racun : %.10lf\n", u[calcIndex(threadIdx.x, threadIdx.y,
                //DISP_ROW[idx], DISP_COL[idx])]);
            val += u[calcIndex(threadIdx.x, threadIdx.y,
                DISP_ROW[idx], DISP_COL[idx])];
            if ( (threadIdx.x == 1 && threadIdx.y == 0 && idxTile == 1))
            {
                /*printf("Vrednost indes %d %d u : %lf %d %d %.10lf\n", calcIndex(threadIdx.x, threadIdx.y,
                    DISP_ROW[idx], DISP_COL[idx]), idx, u[calcIndex(threadIdx.x, threadIdx.y,
                    DISP_ROW[idx], DISP_COL[idx])], blockIdx.x, blockIdx.y, oldVal);
                    */
            }
        }
        val /= 4.0;

        //printf("Vrednost val : %lf\n", val);
        wOut[linIndex(row, col)] = val;

        diff[TILE_WIDTH * threadIdx.x + threadIdx.y] = fabs(val - oldVal);
    }
    else
    {
        if ( (row >= 0) && (row < mGPU) && (col >= 0) && (col < nGPU))
        {
            wOut[linIndex(row, col)] = u[calcIndex(threadIdx.x, threadIdx.y, 0, 0)];
        }

        diff[TILE_WIDTH * threadIdx.x + threadIdx.y] = 0.0;
    }

#if 0
    if (threadIdx.x == 1 && threadIdx.y == 1)
    {
        printf("Vrednost %.10lf \n", val);
        printf("Row %d col %d\n", row, col);
        printf("Vrednost ostatka :%.10lf %.10lf %.10lf %.10lf\n", u[calcIndex(0, threadIdx.y, -1, 0)], u[calcIndex(TILE_HEIGHT, threadIdx.y, 0, 0)],
            u[calcIndex(threadIdx.y, colBegin, 0, -1)], u[calcIndex(threadIdx.y, colBegin, 0, TILE_WIDTH)]);
    }
#endif
    idx = threadIdx.x * TILE_WIDTH + threadIdx.y;

    //printf("Razlika : diff %d %.10lf\n", idx, diff[idx]);

    if (diff[idx] > 100)
    {

        //printf("Razlika %d %.10lf %d %d %d\n", idx, diff[idx], threadIdx.x, threadIdx.y, idxTile);
    }


    __syncthreads();
    for (unsigned int s = NUM_OF_GPU_THREADS / 2; s > 0; s >>= 1) {
        if (idx < s) {
            diff[idx] = diff[idx] > diff[idx + s] ? diff[idx] : diff[idx + s];
            if (idxTile == 0)
            {
                //printf("Idx : %d %d %dVrednost %.10lf\n", threadIdx.x, threadIdx.y, idx, diff[idx]);
            }

        }
        __syncthreads();
    }

    if (idx == 0)
    {

        diffTiles[idxTile] = diff[0];
        //printf("Vrednost diffTiles:%d %lf\n", idxTile, diffTiles[idxTile]);

    }
}


int main(int argc, char *argv[])
{
    int M;
    int N;

    double ctime;
    double ctime1;
    double ctime2;
    double diff;
    double epsilon;
    FILE *fp;
    int i;
    int iterations;
    int iterations_print;
    int j;
    double mean;
    char output_file_seq[80];
    char output_file_par[80];
    int success;

    double **u;
    double **wSeq;
    double **wPar;

    int col;
    int row;
    double *tmpMemory;

    if (argc != 5) {
        printf("Wrong number of arguments!\n");
        return 1;
    }
    else {
        success = sscanf(argv[1], "%d", &M);
        success += sscanf(argv[2], "%d", &N);
        success += sscanf(argv[3], "%lf", &epsilon);
        success += sscanf(argv[4], "%s", output_file_seq);

        if (success != 4) {
            printf("Wrong arguments!\n");
            return 2;
        }
    }

    printf("\n");
    printf("HEATED_PLATE\n");
    printf("  C version\n");
    printf("  A program to solve for the steady state temperature distribution\n");
    printf("  over a rectangular plate.\n");
    printf("\n");
    printf("  Spatial grid of %d by %d points.\n", M, N);
    printf("\n");
    printf("  The iteration will be repeated until the change is <= %f\n", epsilon);
    diff = epsilon;
    printf("\n");
    printf("  The steady state solution will be written to %s.\n", output_file_seq);



    u = (double **)malloc(M * sizeof(double*));
    tmpMemory = (double*)malloc(M * N * sizeof(double));
    for (i = 0; i < M; i++)
        u[i] = (double *)&(tmpMemory[i * N]);

    wSeq = (double **)malloc(M * sizeof(double*));
    tmpMemory = (double*)malloc(M * N * sizeof(double));
    for (i = 0; i < M; i++)
        wSeq[i] = (double *)&(tmpMemory[i * N]);

    wPar = (double **)malloc(M * sizeof(double*));
    tmpMemory = (double*)malloc(M * N * sizeof(double));
    for (i = 0; i < M; i++)
        wPar[i] = (double *)&(tmpMemory[i * N]);

    /*
    Set the boundary values, which don't change.
    */
    for (i = 1; i < M - 1; i++)
    {
        wSeq[i][0] = 100.0;
    }
    for (i = 1; i < M - 1; i++)
    {
        wSeq[i][N - 1] = 100.0;
    }
    for (j = 0; j < N; j++)
    {
        wSeq[M - 1][j] = 100.0;
    }
    for (j = 0; j < N; j++)
    {
        wSeq[0][j] = 0.0;
    }
    /*
    Average the boundary values, to come up with a reasonable
    initial value for the interior.
    */
    mean = 0.0;
    for (i = 1; i < M - 1; i++)
    {
        mean = mean + wSeq[i][0];
    }
    for (i = 1; i < M - 1; i++)
    {
        mean = mean + wSeq[i][N - 1];
    }
    for (j = 0; j < N; j++)
    {
        mean = mean + wSeq[M - 1][j];
    }
    for (j = 0; j < N; j++)
    {
        mean = mean + wSeq[0][j];
    }
    mean = mean / (double)(2 * M + 2 * N - 4);
    printf("Mean is %.10lf\n", mean);
    /*
    Initialize the interior solution to the mean value.
    */
    for (i = 1; i < M - 1; i++)
    {
        for (j = 1; j < N - 1; j++)
        {
            wSeq[i][j] = mean;
        }
    }
    /*
    iterate until the  new solution W differs from the old solution U
    by no more than EPSILON.
    */
    iterations = 0;
    iterations_print = 1;
    printf("\n");
    printf(" Iteration  Change\n");
    printf("\n");
    ctime1 = cpu_time();

    while (epsilon <= diff)
    {
        /*
        Save the old solution in U.
        */
        for (i = 0; i < M; i++)
        {
            for (j = 0; j < N; j++)
            {
                u[i][j] = wSeq[i][j];
            }
        }
        /*
        Determine the new estimate of the solution at the interior points.
        The new solution W is the average of north, south, east and west neighbors.
        */
        diff = 0.0;
        for (i = 1; i < M - 1; i++)
        {
            for (j = 1; j < N - 1; j++)
            {
                wSeq[i][j] = (u[i - 1][j] + u[i + 1][j] + u[i][j - 1] + u[i][j + 1]) / 4.0;

                if (diff < fabs(wSeq[i][j] - u[i][j]))
                {
                    diff = fabs(wSeq[i][j] - u[i][j]);
                }
            }
        }

        iterations++;
        //break;
        if (iterations == iterations_print)
        {
            printf("  %8d  %f\n", iterations, diff);
            iterations_print = 2 * iterations_print;
        }
    }
    ctime2 = cpu_time();
    ctime = ctime2 - ctime1;

    printf("\n");
    printf("  %8d  %f\n", iterations, diff);
    printf("\n");
    printf("  Error tolerance achieved.\n");
    printf("  CPU sequential time = %f\n", ctime);
    /*
    Write the solution to the output file.
    */
    fp = fopen(output_file_seq, "w");

    fprintf(fp, "%d\n", M);
    fprintf(fp, "%d\n", N);

    for (i = 0; i < M; i++)
    {
        for (j = 0; j < N; j++)
        {
            fprintf(fp, "%6.2f ", wSeq[i][j]);
        }
        fputc('\n', fp);
    }
    fclose(fp);

    printf("\n");
    printf("  Solution written to the output file %s\n", output_file_seq);
    /*
    All done!
    */
    printf("\n");
    printf("HEATED_PLATE:\n");
    printf(" Sequential end of execution.\n");

    strcpy(output_file_par, "par");
    strcat(output_file_par, output_file_seq);


    /*
    Set the boundary values, which don't change.
    */
    for (i = 1; i < M - 1; i++)
    {
        wPar[i][0] = 100.0;
    }
    for (i = 1; i < M - 1; i++)
    {
        wPar[i][N - 1] = 100.0;
    }
    for (j = 0; j < N; j++)
    {
        wPar[M - 1][j] = 100.0;
    }
    for (j = 0; j < N; j++)
    {
        wPar[0][j] = 0.0;
    }
    /*
    Average the boundary values, to come up with a reasonable
    initial value for the interior.
    */
    mean = 0.0;
    for (i = 1; i < M - 1; i++)
    {
        mean = mean + wPar[i][0];
    }
    for (i = 1; i < M - 1; i++)
    {
        mean = mean + wPar[i][N - 1];
    }
    for (j = 0; j < N; j++)
    {
        mean = mean + wPar[M - 1][j];
    }
    for (j = 0; j < N; j++)
    {
        mean = mean + wPar[0][j];
    }
    mean = mean / (double)(2 * M + 2 * N - 4);
    /*
    Initialize the interior solution to the mean value.
    */
    for (i = 1; i < M - 1; i++)
    {
        for (j = 1; j < N - 1; j++)
        {
            wPar[i][j] = mean;
        }
    }
    /*
    iterate until the  new solution W differs from the old solution U
    by no more than EPSILON.
    */

    iterations = 0;
    iterations_print = 1;
    printf("\n");
    printf(" Iteration  Change\n");
    printf("\n");

    cudaError_t cuda_error;




    diff = epsilon;

    double *wParGPU;
    double *wOutGPU;
    double *diffTiles;
    double *wParDummy = (double*)malloc(sizeof(double) * M * N);
    for (row = 0; row < M; row ++ )
    {
        for (col = 0; col < N; col ++)
        {
            wParDummy[row * N + col] = wPar[row][col];
        }
    }

    if (cudaSuccess != cudaMalloc(&wParGPU, sizeof(double) * M * N))
    {
        printf("Nije uspesna alokacija niza\n");
        exit(1);
    }
    ctime1 = cpu_time();
    if (cudaSuccess != cudaMalloc(&wOutGPU, sizeof(double) * M * N))
    {
        printf("Nije uspesna alokacija niza\n");
        exit(1);
    }

    if (cudaSuccess != cudaMemcpyToSymbol(mGPU, &M, sizeof(int)))
    {
        printf("Nije uspesno kopiranje simbola\n");
    }

    if (cudaSuccess != cudaMemcpyToSymbol(nGPU, &N, sizeof(int)))
    {
        printf("Nije uspesno kopiranje simbola\n");
    }

    if (cudaSuccess != cudaMemcpyToSymbol(epsilonGPU, &epsilon, sizeof(double)))
    {
        printf("Nije uspesno kopiranje simbola\n");
    }

    if (cudaSuccess != cudaMemcpyToSymbol(sideVal, &wPar[0][0], sizeof(double)))
    {
        printf("Nije uspesno kopiranje simbola\n");
    }



    cudaMemcpy(wParGPU, wParDummy, sizeof(double) * M * N,
                            cudaMemcpyHostToDevice);

    // Copiramo u njega ceo wPar
    // Mora se imati odredjeni broj niti (1s024) koji ce raditi potreban posao racunanja i reduce i tako dok ne spadne razlika.
    //


    int numWidh = ((N + TILE_WIDTH - 1) / TILE_WIDTH);
    int numHeight = ((M + TILE_HEIGHT - 1) / TILE_HEIGHT);
    int numTiles = numHeight * numWidh;
    double *diffCPU;

    if (cudaSuccess != cudaMemcpyToSymbol(tilesWidthGPU, &numWidh, sizeof(int)))
    {
        printf("Nije uspesno kopiranje simbola\n");
    }

    if (cudaSuccess != cudaMalloc(&diffTiles, sizeof(double) * numTiles))
    {
        printf("Nije uspesna alokacija niza\n");
        exit(1);
    }

    diffCPU = (double*)malloc(sizeof(double) * numTiles);



    while (epsilon <= diff)
    {
        updateMatrixKernel
            <<< dim3(numHeight, numWidh),
            dim3(TILE_WIDTH, TILE_HEIGHT),
            sizeof(double) *
            ((TILE_WIDTH + 2)* (TILE_HEIGHT + 2)
                + NUM_OF_GPU_THREADS)
            >>> (wParGPU, diffTiles, wOutGPU);

        cudaMemcpy(diffCPU, diffTiles, sizeof(double)*numTiles, cudaMemcpyDeviceToHost);

        double *tmp = wOutGPU;
        wOutGPU = wParGPU;
        wParGPU = tmp;

        diff = 0.0;
        for (int idx = 0; idx < numTiles; idx ++)
        {
            diff = diff > diffCPU[idx] ? diff : diffCPU[idx];
        }
        iterations++;
        //break;
        if (iterations == iterations_print)
        {
            printf("  %8d  %f\n", iterations, diff);
            iterations_print = 2 * iterations_print;
        }
    }

    cudaMemcpy(wParDummy, wParGPU, sizeof(double) * M * N,
        cudaMemcpyDeviceToHost);

    for (row = 0; row < M; row++)
    {
        for (col = 0; col < N; col++)
        {
            wPar[row][col] = wParDummy[row * N + col];
        }
    }

    ctime2 = cpu_time();
    ctime = ctime2 - ctime1;
    printf("\n");
    printf("  %8d  %f\n", iterations, diff);
    printf("\n");
    printf("  Error tolerance achieved.\n");
    printf("  CPU parallel time = %f\n", ctime);
    /*
    Write the solution to the output file.
    */
    fp = fopen(output_file_par, "w");

    fprintf(fp, "%d\n", M);
    fprintf(fp, "%d\n", N);

    for (i = 0; i < M; i++)
    {
        for (j = 0; j < N; j++)
        {
            fprintf(fp, "%6.2f ", wPar[i][j]);
        }
        fputc('\n', fp);
    }
    fclose(fp);

    printf("\n");
    printf("  Solution written to the output file %s\n", output_file_par);
    /*
    All done!
    */
    printf("\n");
    printf("HEATED_PLATE:\n");
    printf("Parallel   end of execution.\n");

    int testPassed = 1;
    for (i = 0; i < M; i++)
        for (j = 0; j < N; j++) {
            if ( fabs(wSeq[i][j] - wPar[i][j]) > ACCURACY) {
                testPassed = 0;
                break;
            }
        }
    if (testPassed) {
        printf("Test PASSED\n");
    }
    else {
        printf("Test FAILED");
    }

    return 0;

}

double cpu_time(void)
{
    double value;

    value = (double)clock() / (double)CLOCKS_PER_SEC;

    return value;
}
