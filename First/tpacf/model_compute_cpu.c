#include <sys/time.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "omp.h"

#include "model.h"

int doCompute(struct cartesian *data1, int n1, struct cartesian *data2,
         int n2, int doSelf, long long *data_bins,
         int nbins, float *binb, int isParallel)
{
    int i, j;


    if (isParallel)
    {
        int lenBuffer = (nbins + 2);
        size_t memsize = (lenBuffer)*sizeof(long long);
        long long* threadDataBins = (long long*)malloc(memsize * MAX_THREADS);
        bzero(threadDataBins, memsize * MAX_THREADS);
        if (doSelf)
        {
            n2 = n1;
            data2 = data1;
        }

        #pragma omp parallel for num_threads(MAX_THREADS) private(i, j)\
        default(none) shared(doSelf, data_bins, threadDataBins, data1, data2, binb,\
            nbins, lenBuffer, n2, n1)
        for (i = 0; i < ((doSelf) ? n1-1 : n1); i++)
        {
            const register float xi = data1[i].x;
            const register float yi = data1[i].y;
            const register float zi = data1[i].z;

             for (j = ((doSelf) ? i+1 : 0); j < n2; j++)
             {
                register float dot = xi * data2[j].x + yi * data2[j].y +
                   zi * data2[j].z;

                // run binary search
                register int min = 0;
                register int max = nbins;
                register int k, indx;

                while (max > min+1)
                {
                    k = (min + max) / 2;
                    if (dot >= binb[k])
                    {
                        max = k;
                    }
                    else
                    {
                        min = k;
                    }
                 };

                int threadId = omp_get_thread_num();
                if (dot >= binb[min])
                {
                    threadDataBins[threadId * lenBuffer + min] ++;
                }
                else if (dot < binb[max])
                {
                    threadDataBins[max + 1 + threadId * lenBuffer] ++;
                }
                else
                {
                    threadDataBins[max + threadId * lenBuffer] ++;
                }
            }
        }
        int threadId, histoIdx;
        for (threadId = 0; threadId < MAX_THREADS; threadId ++)
        {
            for (histoIdx = 0; histoIdx < lenBuffer; histoIdx ++)
            {
                data_bins[histoIdx] += threadDataBins[threadId * lenBuffer + histoIdx];
            }
        }
        free(threadDataBins);
    }
    // Not parallel.
    else
    {
        if (doSelf)
        {
            n2 = n1;
            data2 = data1;
        }
        for (i = 0; i < ((doSelf) ? n1-1 : n1); i++)
        {
            const register float xi = data1[i].x;
            const register float yi = data1[i].y;
            const register float zi = data1[i].z;

             for (j = ((doSelf) ? i+1 : 0); j < n2; j++)
             {
                register float dot = xi * data2[j].x + yi * data2[j].y +
                   zi * data2[j].z;

                // run binary search
                register int min = 0;
                register int max = nbins;
                register int k, indx;

                while (max > min+1)
                {
                    k = (min + max) / 2;
                    if (dot >= binb[k])
                    {
                        max = k;
                    }
                    else
                    {
                        min = k;
                    }
                 };

                 if (dot >= binb[min])
                 {
                     data_bins[min] += 1; /*k = min;*/
                 }
                 else if (dot < binb[max])
                 {
                     data_bins[max+1] += 1; /*k = max+1;*/
                 }
                 else
                 {
                     data_bins[max] += 1; /*k = max;*/
                 }
            }
        }

    }

    return 0;
}

