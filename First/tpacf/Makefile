# C compiler
CC = gcc
CC_FLAGS = -g -fopenmp -O2
LD_FLAGS = -lm

tpacf:
	$(CC) $(CC_FLAGS) main.c -o tpacf $(LD_FLAGS)

clean:
	rm -f tpacf *.txt
