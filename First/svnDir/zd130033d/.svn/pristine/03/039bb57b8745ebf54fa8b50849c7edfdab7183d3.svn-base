#include <stdio.h>
#include <stdlib.h>
#include <strings.h>
#include <unistd.h>
#include <sys/time.h>
#include <math.h>
#include <string.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>


typedef struct _options_
{
  char *data_name;
  char *random_name;
  int random_count;
  int npoints;
  char *output_name;
} options;

void usage(char *name);
void parse_args(int argc, char **argv, options* args);

#ifdef __cplusplus
extern "C" {
#endif

#include <unistd.h>

/* Command line parameters for benchmarks */
struct pb_Parameters {
  char *outFile;    /* If not NULL, the raw output of the
         * computation should be saved to this
         * file. The string is owned. */
  char **inpFiles;    /* A NULL-terminated array of strings
         * holding the input file(s) for the
         * computation.  The array and strings
         * are owned. */
};

/* Read command-line parameters.
 *
 * The argc and argv parameters to main are read, and any parameters
 * interpreted by this function are removed from the argument list.
 *
 * A new instance of struct pb_Parameters is returned.
 * If there is an error, then an error message is printed on stderr
 * and NULL is returned.
 */
struct pb_Parameters *
pb_ReadParameters(int *_argc, char **argv);

/* Free an instance of struct pb_Parameters.
 */
void
pb_FreeParameters(struct pb_Parameters *p);

/* Count the number of input files in a pb_Parameters instance.
 */
int
pb_Parameters_CountInputs(struct pb_Parameters *p);

/* A time or duration. */
#if _POSIX_VERSION >= 200112L
typedef unsigned long long pb_Timestamp; /* time in microseconds */
#else
# error "Timestamps not implemented"
#endif

enum pb_TimerState {
  pb_Timer_STOPPED,
  pb_Timer_RUNNING,
};

struct pb_Timer {
  enum pb_TimerState state;
  pb_Timestamp elapsed;   /* Amount of time elapsed so far */
  pb_Timestamp init;    /* Beginning of the current time interval,
         * if state is RUNNING.  End of the last
         * recorded time interfal otherwise.  */
};

/* Reset a timer.
 * Use this to initialize a timer or to clear
 * its elapsed time.  The reset timer is stopped.
 */
void
pb_ResetTimer(struct pb_Timer *timer);

/* Start a timer.  The timer is set to RUNNING mode and
 * time elapsed while the timer is running is added to
 * the timer.
 * The timer should not already be running.
 */
void
pb_StartTimer(struct pb_Timer *timer);

/* Stop a timer.
 * This stops adding elapsed time to the timer.
 * The timer should not already be stopped.
 */
void
pb_StopTimer(struct pb_Timer *timer);

/* Get the elapsed time in seconds. */
double
pb_GetElapsedTime(struct pb_Timer *timer);

/* Execution time is assigned to one of these categories. */
enum pb_TimerID {
  pb_TimerID_NONE = 0,
  pb_TimerID_IO,    /* Time spent in input/output */
  pb_TimerID_KERNEL,    /* Time spent computing on the device,
         * recorded asynchronously */
  pb_TimerID_COPY,    /* Time spent synchronously moving data
         * to/from device and allocating/freeing
         * memory on the device */
  pb_TimerID_DRIVER,    /* Time spent in the host interacting with the
         * driver, primarily for recording the time
                                 * spent queueing asynchronous operations */
  pb_TimerID_COPY_ASYNC,  /* Time spent in asynchronous transfers */
  pb_TimerID_COMPUTE,   /* Time for all program execution other
         * than parsing command line arguments,
         * I/O, kernel, and copy */
  pb_TimerID_OVERLAP,   /* Time double-counted in asynchronous and
         * host activity: automatically filled in,
         * not intended for direct usage */
  pb_TimerID_LAST   /* Number of timer IDs */
};

/* Dynamic list of asynchronously tracked times between events */
struct pb_async_time_marker_list {
  char *label; // actually just a pointer to a string
  enum pb_TimerID timerID;  /* The ID to which the interval beginning
                                 * with this marker should be attributed */
  void * marker;
  //cudaEvent_t marker;     /* The driver event for this marker */
  struct pb_async_time_marker_list *next;
};

struct pb_SubTimer {
  char *label;
  struct pb_Timer timer;
  struct pb_SubTimer *next;
};

struct pb_SubTimerList {
  struct pb_SubTimer *current;
  struct pb_SubTimer *subtimer_list;
};

/* A set of timers for recording execution times. */
struct pb_TimerSet {
  enum pb_TimerID current;
  struct pb_async_time_marker_list* async_markers;
  pb_Timestamp async_begin;
  pb_Timestamp wall_begin;
  struct pb_Timer timers[pb_TimerID_LAST];
  struct pb_SubTimerList *sub_timer_list[pb_TimerID_LAST];
};

/* Reset all timers in the set. */
void
pb_InitializeTimerSet(struct pb_TimerSet *timers);

void
pb_AddSubTimer(struct pb_TimerSet *timers, char *label, enum pb_TimerID pb_Category);

/* Select which timer the next interval of time should be accounted
 * to. The selected timer is started and other timers are stopped.
 * Using pb_TimerID_NONE stops all timers. */
void
pb_SwitchToTimer(struct pb_TimerSet *timers, enum pb_TimerID timer);

void
pb_SwitchToSubTimer(struct pb_TimerSet *timers, char *label, enum pb_TimerID category);

/* Print timer values to standard output. */
void
pb_PrintTimerSet(struct pb_TimerSet *timers);

/* Release timer resources */
void
pb_DestroyTimerSet(struct pb_TimerSet * timers);

void
pb_SetOpenCL(void *clContextPtr, void *clCommandQueuePtr);

#ifdef __cplusplus
}
#endif

#define D2R M_PI/180.0
#define R2D 180.0/M_PI
#define R2AM 60.0*180.0/M_PI

#define bins_per_dec 5
#define min_arcmin 1.0
#define max_arcmin 10000.0

#define NUM_BINS 20
#define MAX_THREADS 8
typedef unsigned long hist_t;

struct spherical
{
  float ra, dec;  // latitude, longitude pair
};

struct cartesian
{
  float x, y, z;  // cartesian coodrinates
};

int readdatafile(char *fname, struct cartesian *data, int npoints);

int doCompute(struct cartesian *data1, int n1, struct cartesian *data2,
        int n2, int doSelf, long long *data_bins,
        int nbins, float *binb, int isParallel);

void initBinB(struct pb_TimerSet *timers);



extern char *optarg;

void usage(char *name)
{
  printf("Usage: %s <-d data_file_name> <-r rnd_file_name> "
   "<-m rnd_count> <-p count> <-o file_name>\n", name);
  exit(0);
}

void parse_args(int argc, char **argv, options* args)
{
  int c;

  args->data_name = NULL;
  args->random_name = NULL;
  args->random_count = 0;
  args->npoints = 0;
  args->output_name = NULL;

  while ((c = getopt(argc, argv, "d:n:r:p:o:")) != EOF)
    {
      switch (c)
  {
        case 'd':
          args->data_name = optarg;
          break;
        case 'r':
          args->random_name = optarg;
          break;
        case 'n':
          args->random_count = atoi(optarg);
          break;
        case 'o':
          args->output_name = optarg;
          break;
        case 'p':
          args->npoints = atol(optarg);
          break;
        default:
          usage(argv[0]);
  }
    }
}

bool allocated = false;


//const float ACCURACY = 0.01;
struct cartesian *data1GPU;
struct cartesian *data2GPU;

unsigned long long *data_binsGPU;
float *binbGPU;
__device__ __constant__  int doSelfGPU;
__device__ __constant__  int n1GPU;
__device__ __constant__  int n2GPU;
__device__ __constant__  int nbinsGPU;
const int NUM_OF_THREADS = 1024;

__global__ void calcHisto(struct cartesian *data1, struct cartesian *data2, unsigned long long *data_bins, float *binb)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j;
	int limit = doSelfGPU ? (n1GPU - 1) : n1GPU;
	double dot;
	int min;
	int max;
	int k;
	int idx;

	if (i == 0) {
		//printf("Vrednost doSelfGPU ; %d\n", doSelfGPU);
		//printf("Vrednost n1GPU : %d\n", n1GPU);
		//printf("Vrednost n2GPU : %d\n", n2GPU);
		//printf("Vredonst nbinsGPU: %d\n", nbinsGPU);
		//printf("Vrednost binbsGPU : %.10f\n", binb[nbinsGPU]);

		for (idx =0; idx < n1GPU; idx ++) {
			//printf("Vrednost data1GPU : %d x: %.10f y :%.10f z : %.10f\n", idx, data1[idx].x, data1[idx].y, data1[idx].z);
		}

		for (idx =0; idx < n2GPU; idx ++) {
			//printf("Vrednost data2GPU : %d x: %.10f y :%.10f z : %.10f\n", idx, data2[idx].x, data2[idx].y, data2[idx].z);
		}
	}

	if (i < limit)
	{
		//printf("Vrednost i GPU %d\n", i);
		const double xi = data1[i].x;
		const double yi = data1[i].y;
		const double zi = data1[i].z;

		 for (j = ((doSelfGPU) ? i+1 : 0); j < n2GPU; j++)
		 {
			dot = xi * data2[j].x + yi * data2[j].y +
			   zi * data2[j].z;

			// run binary search
			min = 0;
			max = nbinsGPU;
			while (max > min+1)
			{
				k = (min + max) / 2;

				//if (fabs(dot - binb[k]) >= ACCURACY)
				if (dot >= binb[k])
				{
					max = k;
				}
				else
				{
					min = k;
				}
			 };
#if 0
			if (i == 54 && j == 64) {
				printf("GPU vrednost dot %.20f min %d max %d \n", dot, min, max);
				printf("GPU binb min : %.20f\n", binb[min]);
				printf("GPU binb max : %.10f\n", binb[max]);
			}
#endif
			 //if (fabs(dot - binb[min]) >= ACCURACY)
			 idx = -1;
			 if (dot >= binb[min])
			 {
				 idx = min;
				 //atomicAdd(&data_bins[min], 1);
				 //data_bins[min] += 1; /*k = min;*/
			 }
			 //else if ( fabs(binb[max] - dot) > ACCURACY)
			 else if (dot < binb[max])
			 {
				 idx = max + 1;
				 //atomicAdd(&data_bins[max+1], 1);
				 //data_bins[max+1] += 1; /*k = max+1;*/
			 }
			 else
			 {
				 idx = max;
				 //atomicAdd(&data_bins[max], 1);
				 //data_bins[max] += 1; /*k = max;*/
			 }
#if 0
			 if (idx == 0)
			 {
				 printf("GPU Vrednost i %d j %d %lf\n", i, j, dot);
				 //printf("data1GPU x : %.10f y: %.10f z: %.10f\n", data1[i].x, data1[i].y, data1[i].z);
				 //printf("data2GPU x : %.10f y: %.10f z: %.10f\n", data2[j].x, data2[j].y, data2[i].z);
			 }
#endif
			 atomicAdd(&data_bins[idx], 1);
		}

	}
}




int doCompute(struct cartesian *data1, int n1, struct cartesian *data2,
         int n2, int doSelf, long long *data_bins,
         int nbins, float *binb, int isParallel)
{


	int i, j;
	int idx;
	int lenBuffer = (nbins + 2);
	size_t memsize = (lenBuffer)*sizeof(unsigned long long);

    if (isParallel)
    {
    	if (doSelf)
    	{
    		n2 = n1;
    		data2 = data1;
    	}

    	if (!allocated)
    	{
    		cudaMalloc(&data1GPU, sizeof(struct cartesian) * n1);
    		cudaMalloc(&data2GPU, sizeof(struct cartesian) * n1);
    		cudaMalloc(&data_binsGPU, memsize);
        	cudaMemcpyToSymbol(nbinsGPU, &nbins, sizeof(nbins));
        	cudaMalloc(&binbGPU, sizeof(float) * (nbins+1));
    		allocated = true;
    	}

		// Kopiraj promenljivu memoriju .
    	//
    	cudaMemcpy(data1GPU, data1, sizeof(struct cartesian) * n1, cudaMemcpyHostToHost);
    	cudaMemcpy(data2GPU, data2, sizeof(struct cartesian) * n2, cudaMemcpyHostToDevice);
    	cudaMemcpy(binbGPU, binb, sizeof(float) * (nbins + 1), cudaMemcpyHostToDevice);
    	cudaMemset(data_binsGPU, 0, memsize);

    	// Kopiraj konstante na GPU.
    	//
    	cudaMemcpyToSymbol(doSelfGPU, &doSelf, sizeof(doSelf));
    	cudaMemcpyToSymbol(n1GPU, &n1, sizeof(n1));
    	cudaMemcpyToSymbol(n2GPU, &n2, sizeof(n2));

    	int limit = doSelf ? n1 - 1 : n1;



    	calcHisto<<< limit / NUM_OF_THREADS +1, NUM_OF_THREADS>>>(data1GPU, data2GPU, data_binsGPU, binbGPU);

		cudaDeviceSynchronize();
    	cudaMemcpy(data_bins, data_binsGPU, memsize, cudaMemcpyDeviceToHost);


    }
    // Not parallel.
    else
    {
        if (doSelf)
        {
            n2 = n1;
            data2 = data1;
        }
        bzero(data_bins, memsize);
#if 0
		printf("Vrednost doSelf ; %d\n", doSelf);
		printf("Vrednost n1 : %d\n", n1);
		printf("Vrednost n2 : %d\n", n2);
		printf("Vredonst nbins: %d\n", nbins);
    	printf("Vrednost binbs : %.10f\n", binb[nbins]);
#endif
    	for (int idx = 0; idx < n1; idx ++) {
    		//printf("Vrednost data1 : %d x: %.10f y :%.10f z : %.10f\n", idx, data1[idx].x, data1[idx].y, data1[idx].z);
    	}


		for (int idx =0; idx < n2; idx ++) {
			//printf("Vrednost data2: %d x: %.10f y :%.10f z : %.10f\n", idx, data2[idx].x, data2[idx].y, data2[idx].z);
		}

        for (i = 0; i < ((doSelf) ? n1-1 : n1); i++)
        {
            const register double xi = data1[i].x;
            const register double yi = data1[i].y;
            const register double zi = data1[i].z;

             for (j = ((doSelf) ? i+1 : 0); j < n2; j++)
             {
                register double dot = xi * data2[j].x + yi * data2[j].y +
                   zi * data2[j].z;

                // run binary search
                register int min = 0;
                register int max = nbins;
                register int k;

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
#if 0

    			if (i == 54 && j == 64) {
    				printf("CPU vrednost dot %.20f min %d max %d \n", dot, min, max);
    				printf("CPU binb min : %.10f\n", binb[min]);
    				printf("CPU binb max : %.10f\n", binb[max]);
    			}
#endif
    			if (dot >= binb[min])
                 {
                	 idx = min;
                     //data_bins[min] += 1; /*k = min;*/
                 }
                 else if (dot < binb[max])
                 {
                	 idx = max + 1;
                     //data_bins[max+1] += 1; /*k = max+1;*/
                 }
                 else
                 {
                	 idx = max;
                     //data_bins[max] += 1; /*k = max;*/
                 }
#if 0
    			if (idx == 0) {
    				 //printf("Dodao sam te\n");
                	 printf("Vrednost i %d j %d\n", i, j);
    				 //printf("data1 x : %.10f y: %.10f z: %.10f\n", data1[i].x, data1[i].y, data1[i].z);
    				 //printf("data2 x : %.10f y: %.10f z: %.10f\n", data2[j].x, data2[j].y, data2[j].z);
                 }
#endif
                 data_bins[idx] ++;
            }
        }

    }

    return 0;
}


int readdatafile(char *fname, struct cartesian *data, int npoints)
{
  FILE *infile;
  int lcount = 0;
  float ra, dec;

  if ((infile = fopen(fname, "r")) == NULL)
    {
      fprintf(stderr, "Unable to open data file %s for reading\n", fname);
      return lcount;
    }

  for (lcount = 0; lcount < npoints; lcount++)
    {
      if (fscanf(infile, "%f %f", &ra, &dec) != 2)
  break;

        {
          // data conversion
    float rarad = D2R * ra;
    float decrad = D2R * dec;
    float cd = cos(decrad);

    data[lcount].x = cos(rarad) * cd;
    data[lcount].y = sin(rarad) * cd;
    data[lcount].z = sin(decrad);
        }
    }

  fclose(infile);

  return lcount;
}



#if _POSIX_VERSION >= 200112L
# include <sys/time.h>
#endif

/* Free an array of owned strings. */
static void
free_string_array(char **string_array)
{
  char **p;

  if (!string_array) return;
  for (p = string_array; *p; p++) free(*p);
  free(string_array);
}

/* Parse a comma-delimited list of strings into an
 * array of strings. */
static char **
read_string_array(char *in)
{
  char **ret;
  int i;
  int count;      /* Number of items in the input */
  char *substring;    /* Current substring within 'in' */

  /* Count the number of items in the string */
  count = 1;
  for (i = 0; in[i]; i++) if (in[i] == ',') count++;

  /* Allocate storage */
  ret = (char **)malloc((count + 1) * sizeof(char *));

  /* Create copies of the strings from the list */
  substring = in;
  for (i = 0; i < count; i++) {
    char *substring_end;
    int substring_length;

    /* Find length of substring */
    for (substring_end = substring;
   (*substring_end != ',') && (*substring_end != 0);
   substring_end++);

    substring_length = substring_end - substring;

    /* Allocate memory and copy the substring */
    ret[i] = (char *)malloc(substring_length + 1);
    memcpy(ret[i], substring, substring_length);
    ret[i][substring_length] = 0;

    /* go to next substring */
    substring = substring_end + 1;
  }
  ret[i] = NULL;    /* Write the sentinel value */

  return ret;
}

struct argparse {
  int argc;     /* Number of arguments.  Mutable. */
  char **argv;      /* Argument values.  Immutable. */

  int argn;     /* Current argument number. */
  char **argv_get;    /* Argument value being read. */
  char **argv_put;    /* Argument value being written.
         * argv_put <= argv_get. */
};

static void
initialize_argparse(struct argparse *ap, int argc, char **argv)
{
  ap->argc = argc;
  ap->argn = 0;
  ap->argv_get = ap->argv_put = ap->argv = argv;
}

static void
finalize_argparse(struct argparse *ap)
{
  /* Move the remaining arguments */
  for(; ap->argn < ap->argc; ap->argn++)
    *ap->argv_put++ = *ap->argv_get++;
}

/* Delete the current argument. */
static void
delete_argument(struct argparse *ap)
{
  if (ap->argn >= ap->argc) {
    fprintf(stderr, "delete_argument\n");
  }
  ap->argc--;
  ap->argv_get++;
}

/* Go to the next argument.  Also, move the current argument to its
 * final location in argv. */
static void
next_argument(struct argparse *ap)
{
  if (ap->argn >= ap->argc) {
    fprintf(stderr, "next_argument\n");
  }
  /* Move argument to its new location. */
  *ap->argv_put++ = *ap->argv_get++;
  ap->argn++;
}

static int
is_end_of_arguments(struct argparse *ap)
{
  return ap->argn == ap->argc;
}

static char *
get_argument(struct argparse *ap)
{
  return *ap->argv_get;
}

static char *
consume_argument(struct argparse *ap)
{
  char *ret = get_argument(ap);
  delete_argument(ap);
  return ret;
}

struct pb_Parameters *
pb_ReadParameters(int *_argc, char **argv)
{
  char *err_message;
  struct argparse ap;
  struct pb_Parameters *ret =
    (struct pb_Parameters *)malloc(sizeof(struct pb_Parameters));

  /* Initialize the parameters structure */
  ret->outFile = NULL;
  ret->inpFiles = (char **)malloc(sizeof(char *));
  ret->inpFiles[0] = NULL;

  /* Each argument */
  initialize_argparse(&ap, *_argc, argv);
  while(!is_end_of_arguments(&ap)) {
    char *arg = get_argument(&ap);

    /* Single-character flag */
    if ((arg[0] == '-') && (arg[1] != 0) && (arg[2] == 0)) {
      delete_argument(&ap); /* This argument is consumed here */

      switch(arg[1]) {
      case 'o':     /* Output file name */
  if (is_end_of_arguments(&ap))
    {
      err_message = "Expecting file name after '-o'\n";
      goto error;
    }
  free(ret->outFile);
  ret->outFile = strdup(consume_argument(&ap));
  break;
      case 'i':     /* Input file name */
  if (is_end_of_arguments(&ap))
    {
      err_message = "Expecting file name after '-i'\n";
      goto error;
    }
  ret->inpFiles = read_string_array(consume_argument(&ap));
  break;
      case '-':     /* End of options */
  goto end_of_options;
      default:
  err_message = "Unexpected command-line parameter\n";
  goto error;
      }
    }
    else {
      /* Other parameters are ignored */
      next_argument(&ap);
    }
  } /* end for each argument */

 end_of_options:
  *_argc = ap.argc;   /* Save the modified argc value */
  finalize_argparse(&ap);

  return ret;

 error:
  fputs(err_message, stderr);
  pb_FreeParameters(ret);
  return NULL;
}

void
pb_FreeParameters(struct pb_Parameters *p)
{

  free(p->outFile);
  free_string_array(p->inpFiles);
  free(p);
}

int
pb_Parameters_CountInputs(struct pb_Parameters *p)
{
  int n;

  for (n = 0; p->inpFiles[n]; n++);
  return n;
}

/*****************************************************************************/
/* Timer routines */

static void
accumulate_time(pb_Timestamp *accum,
    pb_Timestamp start,
    pb_Timestamp end)
{
#if _POSIX_VERSION >= 200112L
  *accum += end - start;
#else
# error "Timestamps not implemented for this system"
#endif
}

#if _POSIX_VERSION >= 200112L
static pb_Timestamp get_time()
{
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (pb_Timestamp) (tv.tv_sec * 1000000LL + tv.tv_usec);
}
#else
# error "no supported time libraries are available on this platform"
#endif

void
pb_ResetTimer(struct pb_Timer *timer)
{
  timer->state = pb_Timer_STOPPED;

#if _POSIX_VERSION >= 200112L
  timer->elapsed = 0;
#else
# error "pb_ResetTimer: not implemented for this system"
#endif
}

void
pb_StartTimer(struct pb_Timer *timer)
{
  if (timer->state != pb_Timer_STOPPED) {
    fputs("Ignoring attempt to start a running timer\n", stderr);
    return;
  }

  timer->state = pb_Timer_RUNNING;

#if _POSIX_VERSION >= 200112L
  {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    timer->init = tv.tv_sec * 1000000LL + tv.tv_usec;
  }
#else
# error "pb_StartTimer: not implemented for this system"
#endif
}

void
pb_StartTimerAndSubTimer(struct pb_Timer *timer, struct pb_Timer *subtimer)
{
  unsigned int numNotStopped = 0x3; // 11
  if (timer->state != pb_Timer_STOPPED) {
    fputs("Warning: Timer was not stopped\n", stderr);
    numNotStopped &= 0x1; // Zero out 2^1
  }
  if (subtimer->state != pb_Timer_STOPPED) {
    fputs("Warning: Subtimer was not stopped\n", stderr);
    numNotStopped &= 0x2; // Zero out 2^0
  }
  if (numNotStopped == 0x0) {
    fputs("Ignoring attempt to start running timer and subtimer\n", stderr);
    return;
  }

  timer->state = pb_Timer_RUNNING;
  subtimer->state = pb_Timer_RUNNING;

#if _POSIX_VERSION >= 200112L
  {
    struct timeval tv;
    gettimeofday(&tv, NULL);

    if (numNotStopped & 0x2) {
      timer->init = tv.tv_sec * 1000000LL + tv.tv_usec;
    }

    if (numNotStopped & 0x1) {
      subtimer->init = tv.tv_sec * 1000000LL + tv.tv_usec;
    }
  }
#else
# error "pb_StartTimer: not implemented for this system"
#endif

}

void
pb_StopTimer(struct pb_Timer *timer)
{

  pb_Timestamp fini;

  if (timer->state != pb_Timer_RUNNING) {
    fputs("Ignoring attempt to stop a stopped timer\n", stderr);
    return;
  }

  timer->state = pb_Timer_STOPPED;

#if _POSIX_VERSION >= 200112L
  {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    fini = tv.tv_sec * 1000000LL + tv.tv_usec;
  }
#else
# error "pb_StopTimer: not implemented for this system"
#endif

  accumulate_time(&timer->elapsed, timer->init, fini);
  timer->init = fini;

}

void pb_StopTimerAndSubTimer(struct pb_Timer *timer, struct pb_Timer *subtimer) {

  pb_Timestamp fini;

  unsigned int numNotRunning = 0x3; // 0b11
  if (timer->state != pb_Timer_RUNNING) {
    fputs("Warning: Timer was not running\n", stderr);
    numNotRunning &= 0x1; // Zero out 2^1
  }
  if (subtimer->state != pb_Timer_RUNNING) {
    fputs("Warning: Subtimer was not running\n", stderr);
    numNotRunning &= 0x2; // Zero out 2^0
  }
  if (numNotRunning == 0x0) {
    fputs("Ignoring attempt to stop stopped timer and subtimer\n", stderr);
    return;
  }


  timer->state = pb_Timer_STOPPED;
  subtimer->state = pb_Timer_STOPPED;

#if _POSIX_VERSION >= 200112L
  {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    fini = tv.tv_sec * 1000000LL + tv.tv_usec;
  }
#else
# error "pb_StopTimer: not implemented for this system"
#endif

  if (numNotRunning & 0x2) {
    accumulate_time(&timer->elapsed, timer->init, fini);
    timer->init = fini;
  }

  if (numNotRunning & 0x1) {
    accumulate_time(&subtimer->elapsed, subtimer->init, fini);
    subtimer->init = fini;
  }

}

/* Get the elapsed time in seconds. */
double
pb_GetElapsedTime(struct pb_Timer *timer)
{
  double ret;

  if (timer->state != pb_Timer_STOPPED) {
    fputs("Elapsed time from a running timer is inaccurate\n", stderr);
  }

#if _POSIX_VERSION >= 200112L
  ret = timer->elapsed / 1e6;
#else
# error "pb_GetElapsedTime: not implemented for this system"
#endif
  return ret;
}

void
pb_InitializeTimerSet(struct pb_TimerSet *timers)
{
  int n;

  timers->wall_begin = get_time();

  timers->current = pb_TimerID_NONE;

  timers->async_markers = NULL;


  for (n = 0; n < pb_TimerID_LAST; n++) {
    pb_ResetTimer(&timers->timers[n]);
    timers->sub_timer_list[n] = NULL; // free first?
  }
}

void
pb_AddSubTimer(struct pb_TimerSet *timers, char *label, enum pb_TimerID pb_Category) {

  struct pb_SubTimer *subtimer = (struct pb_SubTimer *) malloc
    (sizeof(struct pb_SubTimer));

  int len = strlen(label);

  subtimer->label = (char *) malloc (sizeof(char)*(len+1));
  sprintf(subtimer->label, "%s\0", label);

  pb_ResetTimer(&subtimer->timer);
  subtimer->next = NULL;

  struct pb_SubTimerList *subtimerlist = timers->sub_timer_list[pb_Category];
  if (subtimerlist == NULL) {
    subtimerlist = (struct pb_SubTimerList *) malloc
      (sizeof(struct pb_SubTimerList));
    subtimerlist->subtimer_list = subtimer;
    timers->sub_timer_list[pb_Category] = subtimerlist;
  } else {
    // Append to list
    struct pb_SubTimer *element = subtimerlist->subtimer_list;
    while (element->next != NULL) {
      element = element->next;
    }
    element->next = subtimer;
  }

}

void
pb_SwitchToSubTimer(struct pb_TimerSet *timers, char *label, enum pb_TimerID category)
{

// switchToSub( NULL, NONE
// switchToSub( NULL, some
// switchToSub( some, some
// switchToSub( some, NONE -- tries to find "some" in NONE's sublist, which won't be printed

  struct pb_Timer *topLevelToStop = NULL;
  if (timers->current != category && timers->current != pb_TimerID_NONE) {
    // Switching to subtimer in a different category needs to stop the top-level current, different categoried timer.
    // NONE shouldn't have a timer associated with it, so exclude from branch
    topLevelToStop = &timers->timers[timers->current];
  }

  struct pb_SubTimerList *subtimerlist = timers->sub_timer_list[timers->current];
  struct pb_SubTimer *curr = (subtimerlist == NULL) ? NULL : subtimerlist->current;

  if (timers->current != pb_TimerID_NONE) {
    if (curr != NULL && topLevelToStop != NULL) {
      pb_StopTimerAndSubTimer(topLevelToStop, &curr->timer);
    } else if (curr != NULL) {
      pb_StopTimer(&curr->timer);
    } else {
      pb_StopTimer(topLevelToStop);
    }
  }

  subtimerlist = timers->sub_timer_list[category];
  struct pb_SubTimer *subtimer = NULL;

  if (label != NULL) {
    subtimer = subtimerlist->subtimer_list;
    while (subtimer != NULL) {
      if (strcmp(subtimer->label, label) == 0) {
        break;
      } else {
        subtimer = subtimer->next;
      }
    }
  }

  if (category != pb_TimerID_NONE) {

    if (subtimerlist != NULL) {
      subtimerlist->current = subtimer;
    }

    if (category != timers->current && subtimer != NULL) {
      pb_StartTimerAndSubTimer(&timers->timers[category], &subtimer->timer);
    } else if (subtimer != NULL) {
      // Same category, different non-NULL subtimer
      pb_StartTimer(&subtimer->timer);
    } else{
      // Different category, but no subtimer (not found or specified as NULL) -- unprefered way of setting topLevel timer
      pb_StartTimer(&timers->timers[category]);
    }
  }

  timers->current = category;

}

void
pb_SwitchToTimer(struct pb_TimerSet *timers, enum pb_TimerID timer)
{
  /* Stop the currently running timer */
  if (timers->current != pb_TimerID_NONE) {
    struct pb_SubTimer *currSubTimer = NULL;
    struct pb_SubTimerList *subtimerlist = timers->sub_timer_list[timers->current];

    if ( subtimerlist != NULL) {
      currSubTimer = timers->sub_timer_list[timers->current]->current;
    }
    if ( currSubTimer!= NULL) {
      pb_StopTimerAndSubTimer(&timers->timers[timers->current], &currSubTimer->timer);
    } else {
      pb_StopTimer(&timers->timers[timers->current]);
    }

  }

  timers->current = timer;

  if (timer != pb_TimerID_NONE) {
    pb_StartTimer(&timers->timers[timer]);
  }
}

void
pb_PrintTimerSet(struct pb_TimerSet *timers)
{

  pb_Timestamp wall_end = get_time();

  struct pb_Timer *t = timers->timers;
  struct pb_SubTimer* sub = NULL;

  int maxSubLength;

  const char *categories[] = {
    "IO", "Kernel", "Copy", "Driver", "Copy Async", "Compute"
  };

  const int maxCategoryLength = 10;

  int i;
  for(i = 1; i < pb_TimerID_LAST-1; ++i) { // exclude NONE and OVRELAP from this format
    if(pb_GetElapsedTime(&t[i]) != 0) {

      // Print Category Timer
      printf("%-*s: %f\n", maxCategoryLength, categories[i-1], pb_GetElapsedTime(&t[i]));

      if (timers->sub_timer_list[i] != NULL) {
        sub = timers->sub_timer_list[i]->subtimer_list;
        maxSubLength = 0;
        while (sub != NULL) {
          // Find longest SubTimer label
          if (strlen(sub->label) > maxSubLength) {
            maxSubLength = strlen(sub->label);
          }
          sub = sub->next;
        }

        // Fit to Categories
        if (maxSubLength <= maxCategoryLength) {
         maxSubLength = maxCategoryLength;
        }

        sub = timers->sub_timer_list[i]->subtimer_list;

        // Print SubTimers
        while (sub != NULL) {
          printf(" -%-*s: %f\n", maxSubLength, sub->label, pb_GetElapsedTime(&sub->timer));
          sub = sub->next;
        }
      }
    }
  }

  if(pb_GetElapsedTime(&t[pb_TimerID_OVERLAP]) != 0)
    printf("CPU/Kernel Overlap: %f\n", pb_GetElapsedTime(&t[pb_TimerID_OVERLAP]));

  float walltime = (wall_end - timers->wall_begin)/ 1e6;
  printf("Timer Wall Time: %f\n", walltime);

}

void pb_DestroyTimerSet(struct pb_TimerSet * timers)
{
  /* clean up all of the async event markers */
  struct pb_async_time_marker_list ** event = &(timers->async_markers);
  while( *event != NULL) {
    struct pb_async_time_marker_list ** next = &((*event)->next);
    free(*event);
    (*event) = NULL;
    event = next;
  }

  int i = 0;
  for(i = 0; i < pb_TimerID_LAST; ++i) {
    if (timers->sub_timer_list[i] != NULL) {
      struct pb_SubTimer *subtimer = timers->sub_timer_list[i]->subtimer_list;
      struct pb_SubTimer *prev = NULL;
      while (subtimer != NULL) {
        free(subtimer->label);
        prev = subtimer;
        subtimer = subtimer->next;
        free(prev);
      }
      free(timers->sub_timer_list[i]);
    }
  }
}

void CompareAndSet(long long *first, long long *second, size_t memsize, int *isEqual)
{
  int idx;
  int idx2;
  for (idx = 0; idx < memsize / sizeof(long long); idx ++)
  {
    if (first[idx] != second[idx])
    {

      printf("Nisu jednaki memSize : %d idx %llu first %llu second %llu \n",
                                    memsize, idx, first[idx], second[idx]);
      for (int idx2 = idx + 1; idx2 < memsize / sizeof(long long); idx2++)
      {
    	  printf("Vrednost prvog %llu drugog %llu\n", first[idx2], second[idx2]);
      }
      *isEqual = 0;
      return;
    }
  }
}


int main( int argc, char **argv )
{
  struct pb_TimerSet timers;
  struct pb_Parameters *params;
  int rf, k, nbins, npd, npr;
  float *binb;
  //long long *DD, *RRS, *DRS;
  long long *DDseq, *RRSseq, *DRSseq;
  long long *DDpar, *RRSpar, *DRSpar;
  size_t memsize;
  struct cartesian *data, *random;
  FILE *outfile;

  char *seqStr = "Sequential";
  char *parStr = "Parallel";
  char *mixedStr = "Mixed";

  int isEqual = 1;

  pb_InitializeTimerSet( &timers );
  pb_AddSubTimer(&timers, seqStr, pb_TimerID_COMPUTE);
  pb_AddSubTimer(&timers, parStr, pb_TimerID_COMPUTE);
  pb_AddSubTimer(&timers, mixedStr, pb_TimerID_COMPUTE);

  params = pb_ReadParameters( &argc, argv );

  options args;
  parse_args( argc, argv, &args );

  pb_SwitchToSubTimer(&timers, mixedStr,  pb_TimerID_COMPUTE);
  nbins = (int)floor(bins_per_dec * (log10(max_arcmin) -
					 log10(min_arcmin)));
  memsize = (nbins+2)*sizeof(long long);

  // memory for bin boundaries
  binb = (float *)malloc((nbins+1)*sizeof(float));
  if (binb == NULL)
    {
      fprintf(stderr, "Unable to allocate memory\n");
      exit(-1);
    }
  printf("Velicina nbins %d\n", nbins+1);
  for (k = 0; k < nbins+1; k++)
  {
      binb[k] = cos(pow(10, log10(min_arcmin) +
			k*1.0/bins_per_dec) / 60.0*D2R);
  }

  // memory for DD
  pb_SwitchToSubTimer(&timers, seqStr,  pb_TimerID_COMPUTE);
  DDseq = (long long*)malloc(memsize);

  pb_SwitchToSubTimer(&timers, parStr,  pb_TimerID_COMPUTE);
  DDpar = (long long*)malloc(memsize);

  pb_SwitchToSubTimer(&timers, mixedStr,  pb_TimerID_COMPUTE);
  if ( (DDseq == NULL) || (DDpar == NULL))
  {
    fprintf(stderr, "Unable to allocate memory\n");
    exit(-1);
  }

  pb_SwitchToSubTimer(&timers, parStr,  pb_TimerID_COMPUTE);
  bzero(DDpar, memsize);

  pb_SwitchToSubTimer(&timers, seqStr,  pb_TimerID_COMPUTE);
  bzero(DDseq, memsize);

  // memory for RR
  RRSseq = (long long*)malloc(memsize);

  pb_SwitchToSubTimer(&timers, parStr,  pb_TimerID_COMPUTE);
  RRSpar = (long long*)malloc(memsize);

  pb_SwitchToSubTimer(&timers, mixedStr,  pb_TimerID_COMPUTE);
  if ( (RRSseq == NULL) || (RRSpar == NULL))
    {
      fprintf(stderr, "Unable to allocate memory\n");
      exit(-1);
    }

  pb_SwitchToSubTimer(&timers, parStr,  pb_TimerID_COMPUTE);
  bzero(RRSpar, memsize);

  pb_SwitchToSubTimer(&timers, seqStr,  pb_TimerID_COMPUTE);
  bzero(RRSseq, memsize);
  // memory for DR
  DRSseq = (long long*)malloc(memsize);

  pb_SwitchToSubTimer(&timers, parStr,  pb_TimerID_COMPUTE);
  DRSpar = (long long*)malloc(memsize);

  pb_SwitchToSubTimer(&timers, mixedStr,  pb_TimerID_COMPUTE);
  if ( (DRSseq == NULL) || (DRSpar == NULL))
  {
      fprintf(stderr, "Unable to allocate memory\n");
      exit(-1);
  }

  pb_SwitchToSubTimer(&timers, seqStr,  pb_TimerID_COMPUTE);
  bzero(DRSseq, memsize);
  pb_SwitchToSubTimer(&timers, parStr,  pb_TimerID_COMPUTE);
  bzero(DRSpar, memsize);

  pb_SwitchToSubTimer(&timers, mixedStr,  pb_TimerID_COMPUTE);
  // memory for input data
  data = (struct cartesian*)malloc
    (args.npoints* sizeof(struct cartesian));
  if (data == NULL)
    {
      fprintf(stderr,
	      "Unable to allocate memory for % data points (#1)\n",
	      args.npoints);
      return(0);
    }

  random = (struct cartesian*)malloc
    (args.npoints*sizeof(struct cartesian));
  if (random == NULL)
    {
      fprintf(stderr,
	      "Unable to allocate memory for % data points (#2)\n",
	      args.npoints);
      return(0);
    }

  printf("Min distance: %f arcmin\n", min_arcmin);
  printf("Max distance: %f arcmin\n", max_arcmin);
  printf("Bins per dec: %i\n", bins_per_dec);
  printf("Total bins  : %i\n", nbins);
  // read data file
  pb_SwitchToTimer( &timers, pb_TimerID_IO );
  // >>>>>> BEGIN *****
  npd = readdatafile(params->inpFiles[0], data, args.npoints);
  // ***** END   <<<<<
  pb_SwitchToSubTimer(&timers, mixedStr,  pb_TimerID_COMPUTE);
  if (npd != args.npoints)
    {
      fprintf(stderr,
	      "Error: read %i data points out of %i\n",
	      npd, args.npoints);
      return(0);
    }


    // compute DD
    //printf("\nDD\n");
    pb_SwitchToSubTimer(&timers, seqStr,  pb_TimerID_COMPUTE);
    doCompute(data, npd, NULL, 0, 1, DDseq, nbins, binb, 0/*isParallel*/);
    pb_SwitchToSubTimer(&timers, parStr,  pb_TimerID_COMPUTE);
    doCompute(data, npd, NULL, 0, 1, DDpar, nbins, binb, 1/*isParallel*/);
    pb_SwitchToSubTimer(&timers, mixedStr,  pb_TimerID_COMPUTE);
    CompareAndSet(DDseq, DDpar, memsize, &isEqual);
  // loop through random data files
    for (rf = 0; rf < args.random_count; rf++)
    {
        // read random file
        pb_SwitchToTimer( &timers, pb_TimerID_IO );
        npr = readdatafile(params->inpFiles[rf+1], random, args.npoints);
        pb_SwitchToSubTimer(&timers, mixedStr,  pb_TimerID_COMPUTE);
        if (npr != args.npoints)
        {
      	  fprintf(stderr,
      		  "Error: read %i random points out of %i in file %s\n",
      		  npr, args.npoints, params->inpFiles[rf+1]);
      	  return(0);
        }

      //printf("\nRR\n");
      // compute RR
      pb_SwitchToSubTimer(&timers, seqStr,  pb_TimerID_COMPUTE);
      doCompute(random, npr, NULL, 0, 1, RRSseq, nbins, binb, 0/*isParallel*/);
      pb_SwitchToSubTimer(&timers, parStr,  pb_TimerID_COMPUTE);
      doCompute(random, npr, NULL, 0, 1, RRSpar, nbins, binb, 1/*isParallel*/);
      pb_SwitchToSubTimer(&timers, mixedStr,  pb_TimerID_COMPUTE);
      CompareAndSet(RRSseq, RRSpar, memsize, &isEqual);

      //printf("\nDRS\n");
      pb_SwitchToSubTimer(&timers, seqStr,  pb_TimerID_COMPUTE);
      doCompute(data, npd, random, npr, 0, DRSseq, nbins, binb, 0/*isParallel*/);
      pb_SwitchToSubTimer(&timers, parStr,  pb_TimerID_COMPUTE);
      doCompute(data, npd, random, npr, 0, DRSpar, nbins, binb, 1/*isParallel*/);
      pb_SwitchToSubTimer(&timers, mixedStr,  pb_TimerID_COMPUTE);
      CompareAndSet(DRSseq, DRSpar, memsize, &isEqual);
      pb_SwitchToTimer( &timers, pb_TimerID_IO );
    }

  // compute and output results
  if ((outfile = fopen(params->outFile, "w")) == NULL)
    {
      fprintf(stderr,
	      "Unable to open output file %s for writing, assuming stdout\n",
	      params->outFile);
      outfile = stdout;
    }

  // ****BEGIN ****
    pb_SwitchToTimer( &timers, pb_TimerID_IO );
    for (k = 1; k < nbins+1; k++)
    {
      fprintf(outfile, "%d\n%d\n%d\n", DDseq[k], DRSseq[k], RRSseq[k]);
    }
    fprintf(outfile, "End of Sequential output\n");

  for (k = 1; k < nbins+1; k++)
    {
      fprintf(outfile, "%d\n%d\n%d\n", DDpar[k], DRSpar[k], RRSpar[k]);

    }
    fprintf(outfile, "End of Parallel output\n");
  if(outfile != stdout)
    fclose(outfile);

  // free memory
  free(data);
  free(random);
  free(binb);

  pb_SwitchToSubTimer(&timers, parStr,  pb_TimerID_COMPUTE);
  free(DDseq);
  free(RRSseq);
  free(DRSseq);

  pb_SwitchToSubTimer(&timers, seqStr,  pb_TimerID_COMPUTE);
  free(DDpar);
  free(RRSpar);
  free(DRSpar);

   // ****END****
  pb_SwitchToTimer( &timers, pb_TimerID_NONE );
  if (isEqual)
  {
    printf (" Test PASSED\n");
  }
  else
  {
    printf(" Test FAILED\n");
  }
  pb_PrintTimerSet( &timers );
  pb_FreeParameters( params );
}

