# C compiler
CC = mpicc
CC_FLAGS = -g -fopenmp
LD_FLAGS = -lm

tpacf:
	$(CC) $(CC_FLAGS) main.c   -o tpacf $(LD_FLAGS)

clean:
	rm -f tpacf *.txt
