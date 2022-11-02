MPI_LSTDFLG = -lstdc++ -lm -lgsl -lgslcblas
MPI_INCLUDE = -I/usr/include/
MPI_LIB = -L/usr/lib/
MPI_OBJS = logitEstimates

all:	${MPI_OBJS}
	rm -f *.o

matrices.o: matrices.cpp matrices.h
	mpic++ -g -c matrices.cpp -o matrices.o ${MPI_INCLUDE} ${MPI_LIB}

logitEstimates.o: logitEstimates.cpp matrices.h
	mpic++ -g -c logitEstimates.cpp -o logitEstimates.o ${MPI_INCLUDE} ${MPI_LIB}

logitEstimates: logitEstimates.o matrices.o
	mpic++ logitEstimates.o matrices.o -o logitEstimates ${MPI_LIB} ${MPI_LSTDFLG}

clean:
	rm -f *.o
	rm -f ${MPI_OBJS}