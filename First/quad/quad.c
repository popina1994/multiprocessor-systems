# include <stdlib.h>
# include <stdio.h>
# include <math.h>
# include <time.h>
# include <omp.h>

int main ( int argc, char *argv[] );
double f ( double x );

int main ( int argc, char *argv[] ) {

  // First limit of integral.
  //
  double a;
  // Second limit of integral.
  //
  double b;
  // Error in calculation.
  //
  double error;
  // Exact value in calculation.
  //
  double exact = 0.49936338107645674464;
  int i;
  // Number of iteration.
  //
  int n;
  double total;
  double totalSeq;
  double wtimeSeq;
  double x;
  double wtimePar;
  if (argc != 4) {
    n = 10000000;
    a = 0.0;
    b = 10.0;
  } else {
    n = atoi(argv[1]);
    a = atoi(argv[2]);
    b = atoi(argv[3]);
  }

  printf ( "\n" );
  printf ( "QUAD:\n" );
  printf ( "  Estimate the integral of f(x) from A to B.\n" );
  printf ( "  f(x) = 50 / ( pi * ( 2500 * x * x + 1 ) ).\n" );
  printf ( "\n" );
  printf ( "  A        = %f\n", a );
  printf ( "  B        = %f\n", b );
  printf ( "  N        = %d\n", n );

  wtimeSeq  = omp_get_wtime ( );

  total = 0.0;

  for ( i = 0; i < n; i++ )
  {
    x = ( ( double ) ( n - i - 1 ) * a + ( double ) ( i ) * b ) / ( double ) ( n - 1 );
    totalSeq = totalSeq + f ( x );
  }

  wtimeSeq = omp_get_wtime ( ) - wtimeSeq;

  totalSeq = ( b - a ) * totalSeq / ( double ) n;
  error = fabs ( totalSeq - exact );

  printf ( "\n" );
   printf ( "  Estimate = %24.16f\n", totalSeq );
  // printf ( "  Error    = %e\n", error );

  printf ( "  Normal end of execution.\n" );
  printf ( "  Time     = %f\n", wtimeSeq );
  printf ( "\n" );

  // Parallel block.
  //
  const int N = 8;
  const double ACCURACY = 0.01;
  int numThreads;
  int size;
  for (numThreads = 1; numThreads <= N; numThreads <<= 1)
  {
    wtimePar = omp_get_wtime ( );

    total = 0.0;
    size = n / numThreads;
    #pragma omp  parallel num_threads(numThreads) private(i, x)\
	reduction(+:total)
    {
	int myId = omp_get_thread_num();
	for ( i = myId * size; i  < ((numThreads == myId + 1) ? n : (myId + 1) * size); i++ )
        {
          x = ( ( double ) ( n - i - 1 ) * a + ( double ) ( i ) * b ) / ( double ) ( n - 1 );
	  total = total + f(x);
        }
    }
    wtimePar  = omp_get_wtime ( ) - wtimePar;

    total = ( b - a ) * total / ( double ) n;
    error = fabs ( total - exact );

    printf ( "  Estimate = %24.16f\n", total );
    // printf ( "  Error    = %e\n", error );

    printf ( "  End of paralle execution with %d number of threads.\n", numThreads);
    if ( fabs ( total  - totalSeq) <=  ACCURACY)
    {
      printf ( "  Test PASSED\n" );
    }
    else
    {
      printf( "  Test FAILED\n" );
    }

    printf ( "  Time     = %f\n", wtimePar );
    printf ( "\n" );
  }

  return 0;
}

double f ( double x ) {
  double pi = 3.141592653589793;
  double value;

  value = 50.0 / ( pi * ( 2500.0 * x * x + 1.0 ) );

  return value;
}

