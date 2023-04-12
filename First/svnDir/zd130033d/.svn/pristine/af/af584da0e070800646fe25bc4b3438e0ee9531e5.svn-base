# include <stdlib.h>
# include <stdio.h>
# include <math.h>
# include <time.h>
# include <omp.h>
# include <mpi.h>

int main(int argc, char *argv[]);
double f(double x);

const int NUM_GROUPS = 4;
const int MASTER = 0;
const double ACCURACY = 0.01;

int main(int argc, char *argv[]) {
	double a;
	double b;
	double error;
	double exact = 0.49936338107645674464;
	int i;
	int n;
	double totalSeq;
	double totalParSend;
	double totalPar;
	double wtime;
	double x;
	int numGroups;
	int rank;

	MPI_Init(&argc, &argv);

	MPI_Comm_size(MPI_COMM_WORLD, &numGroups);
	if (numGroups != NUM_GROUPS) {
		MPI_Abort(MPI_COMM_WORLD, -1);
	}
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	if (rank == MASTER) {

		if (argc != 4) {
			n = 10000000;
			a = 0.0;
			b = 10.0;
		}
		else {
			n = atoi(argv[1]);
			a = atoi(argv[2]);
			b = atoi(argv[3]);
		}

		printf("\n");
		printf("QUAD:\n");
		printf("  Estimate the integral of f(x) from A to B.\n");
		printf("  f(x) = 50 / ( pi * ( 2500 * x * x + 1 ) ).\n");
		printf("\n");
		printf("  A        = %f\n", a);
		printf("  B        = %f\n", b);
		printf("  N        = %d\n", n);
		printf("  Exact    = %24.16f\n", exact);

		wtime = omp_get_wtime();

		totalSeq = 0.0;

		for (i = 0; i < n; i++)
		{
			x = ((double)(n - i - 1) * a + (double)(i)* b) / (double)(n - 1);
			totalSeq = totalSeq + f(x);
		}

		wtime = omp_get_wtime() - wtime;

		totalSeq = (b - a) * totalSeq / (double)n;
		error = fabs(totalSeq - exact);
		printf("  Sequential Time     = %f\n", wtime);

	}
	wtime = omp_get_wtime();

	totalParSend = 0.0;

	if (rank == MASTER) {
		MPI_Bcast(&n, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
		MPI_Bcast(&a, 1, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);
		MPI_Bcast(&b, 1, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);
	}
	else {
		MPI_Bcast(&n, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
		MPI_Bcast(&a, 1, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);
		MPI_Bcast(&b, 1, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);
	}


	for (i = rank * (n / NUM_GROUPS); i < (rank + 1) * (n / NUM_GROUPS); i++)
	{
		x = ((double)(n - i - 1) * a + (double)(i)* b) / (double)(n - 1);
		totalParSend = totalParSend + f(x);
	}
	MPI_Reduce(&totalParSend, &totalPar, 1, MPI_DOUBLE, MPI_SUM, MASTER, MPI_COMM_WORLD);
	
	wtime = omp_get_wtime() - wtime;

	if (MASTER == rank) {
		
		totalPar = (b - a) * totalPar / (double)n;
		error = fabs(totalPar - exact);
		printf("Parallel time     = %f\n", wtime);

		if (fabs(totalPar - totalSeq) < ACCURACY) {
			printf("Test PASSED");
		}
		else {
			printf("Test FAILED");
		}
	}
	MPI_Finalize();
	return 0;
}

double f(double x) {
	double pi = 3.141592653589793;
	double value;

	value = 50.0 / (pi * (2500.0 * x * x + 1.0));

	return value;
}

