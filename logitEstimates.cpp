/*
 This program computes the estimates of several logistic regressions in parallel.
 Compile the program using the makefile provided.
 
 Run the program using the command:

 mpirun -np 10 logitEstimates 
*/


#include "matrices.h"
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>
#include <mpi.h>
#include <iomanip>

// For MPI communication
#define GETR2	1
#define SHUTDOWNTAG	0

// Used to determine PRIMARY or REPLICA
static int myrank;

// Global variables
int n = 148; //sample size
int p = 61; //number of variables
gsl_matrix* data = gsl_matrix_alloc(n,p);
gsl_vector* Y = gsl_vector_alloc(n);
//Linked list
typedef struct myRegression* LPRegression;
typedef struct myRegression Regression;

struct myRegression
{
	int explanatory; //index of x
	double LaplaceApprox; //Laplace approximation
	double mcIntegration; //Monte Carlo integration
	double betaBayes0; // beta0 from M-H algo
	double betaBayes1; // beta1 from M-H algo

	LPRegression Next; //link to the next regression
};
// declare head regression
LPRegression reg = new Regression;
int nMaxRegs = 5;

// Function Declarations
void primary();
void replica(int primaryname);
gsl_vector* getcoefNR(int explanatory);
double getLaplaceApprox(int explanatory, gsl_vector* betaMode);
double MCIntegration(int explanatory, int NumberOfIterations);
gsl_vector* getPosteriorMeans(int explanatory, gsl_vector* betaMode, int NumberOfIterations);
void AddRegression(int nMaxRegs,LPRegression regressions,int explanatory,double LaplaceApprox,double mcIntegration,double betaBayes0,double betaBayes1);
void SaveRegressions(char* filename,LPRegression regressions);
void DeleteLastRegression(LPRegression regressions);
void DeleteAllRegressions(LPRegression regressions);

int main(int argc, char* argv[])
{
	///////////////////////////
	// START THE MPI SESSION //
	///////////////////////////
	MPI_Init(&argc, &argv);

	/////////////////////////////////////
	// What is the ID for the process? //   
	/////////////////////////////////////
	MPI_Comm_rank(MPI_COMM_WORLD,&myrank);

	char datafilename[] = "534finalprojectdata.txt";
	
	//initialize head regression
	reg->Next = NULL;
	
	//read the data
	FILE* datafile = fopen(datafilename,"r");
	if(NULL==datafile)
	{
		fprintf(stderr,"Cannot open data file [%s]\n",datafilename);
		return(0);
	}
	if(0!=gsl_matrix_fscanf(datafile,data))
	{
		fprintf(stderr,"File [%s] does not have the required format.\n",datafilename);
		return(0);
	}
	fclose(datafile);

	gsl_matrix_get_col(Y,data,60);//change 60 later

	// Branch off to primary or replica function
	// Primary has ID == 0, the replicas are then in order 1,2,3,...

	if(myrank==0)
	{
	  primary();
	}
	else
	{
	  replica(myrank);
	}

	// clean memory
	gsl_vector_free(Y);
	gsl_matrix_free(data);
	DeleteAllRegressions(reg);
	delete reg;

	// Finalize the MPI session
	MPI_Finalize();

	return(0);
}

void primary()
{
   int var;		// to loop over the variables
   int rank;		// another looping variable
   int ntasks;		// the total number of replicas
   int jobsRunning;	// how many replicas we have working
   int work[1];		// information to send to the replicas
   double workresults[5]; // info received from the replicas: var, Laplace, MC, beta0, beta1
   FILE* fout;		// the output file
   MPI_Status status;	// MPI information

	char outfilename[] = "estimates_best5.txt";
   fout = fopen("estimates.txt","w");

   // Find out how many replicas there are
   MPI_Comm_size(MPI_COMM_WORLD,&ntasks);

   fprintf(stdout, "Total Number of processors = %d\n",ntasks);

   // Now loop through the variables and compute the R2 values in
   // parallel
   jobsRunning = 1;

   for(var=0; var<p-1; var++)
   {
      // This will tell a replica which variable to work on
      work[0] = var;

      if(jobsRunning < ntasks) // Do we have an available processor?
      {
         // Send out a work request
         MPI_Send(&work, 	// the vector with the variable
		            1, 		// the size of the vector
		            MPI_INT,	// the type of the vector
                  jobsRunning,	// the ID of the replica to use
                  GETR2,	// tells the replica what to do
                  MPI_COMM_WORLD); // send the request out to anyone
				   // who is available
         printf("Primary sends out work request [%d] to replica [%d]\n",
                work[0],jobsRunning);

         // Increase the # of processors in use
         jobsRunning++;

      }
      else // all the processors are in use!
      {
         MPI_Recv(workresults,	// where to store the results
 		            5,		// the size of the vector
		            MPI_DOUBLE,	// the type of the vector
	 	            MPI_ANY_SOURCE,
		            MPI_ANY_TAG, 	
		            MPI_COMM_WORLD,
		            &status);     // lets us know which processor
				// returned these results

         printf("Primary has received the result of work request [%d] from replica [%d]\n",
                (int) workresults[0],status.MPI_SOURCE);
 
         // Print out the results
         fprintf(fout, "%d, Laplace=%f, MC=%f, beta0=%f, beta1=%f\n", (int)workresults[0]+1, workresults[1],
         	workresults[2], workresults[3], workresults[4]);

         AddRegression(nMaxRegs, reg,(int)workresults[0]+1,workresults[1],workresults[2],workresults[3], workresults[4]);
         printf("Primary sends out work request [%d] to replica [%d]\n",
                work[0],status.MPI_SOURCE);

         // Send out a new work order to the processors that just
         // returned
         MPI_Send(&work,
                  1,
                  MPI_INT,
                  status.MPI_SOURCE, // the replica that just returned
                  GETR2,
                  MPI_COMM_WORLD); 
      } // using all the processors
   } // loop over all the variables


   ///////////////////////////////////////////////////////////////
   // NOTE: we still have some work requests out that need to be
   // collected. Collect those results now.
   ///////////////////////////////////////////////////////////////

   // loop over all the replicas
   for(rank=1; rank<jobsRunning; rank++)
   {
      MPI_Recv(workresults,
               5,
               MPI_DOUBLE,
               MPI_ANY_SOURCE,	// whoever is ready to report back
               MPI_ANY_TAG,
               MPI_COMM_WORLD,
               &status);

       printf("Primary has received the result of work request [%d]\n",
                (int) workresults[0]);
 
      //save the results received
      fprintf(fout, "%d, Laplace=%f, MC=%f, beta0=%f, beta1=%f\n", (int)workresults[0]+1, workresults[1],
         	workresults[2], workresults[3], workresults[4]);
      AddRegression(nMaxRegs, reg,(int)workresults[0]+1,workresults[1],workresults[2],workresults[3], workresults[4]);
   }

   printf("Tell the replicas to shutdown.\n");

   // Shut down the replica processes
   for(rank=1; rank<ntasks; rank++)
   {
      printf("Primary is shutting down replica [%d]\n",rank);
      MPI_Send(0,
	            0,
               MPI_INT,
               rank,		// shutdown this particular node
               SHUTDOWNTAG,		// tell it to shutdown
	       MPI_COMM_WORLD);
   }

   printf("got to the end of Primary code\n");

   fclose(fout);

   SaveRegressions(outfilename, reg);
   // return to the main function
   return;
  
}

void replica(int replicaname)
{
   int work[1];			// the input from primary
   double workresults[5];	// the output for primary
   MPI_Status status;		// for MPI communication

   // the replica listens for instructions...
   int notDone = 1;
   while(notDone)
   {
      printf("Replica %d is waiting\n",replicaname);
      MPI_Recv(&work, // the input from primary
	            1,		// the size of the input
	            MPI_INT,		// the type of the input
               0,		// from the PRIMARY node (rank=0)
               MPI_ANY_TAG,	// any type of order is fine
               MPI_COMM_WORLD,
               &status);
      printf("Replica %d just received smth\n",replicaname);

      // switch on the type of work request
      switch(status.MPI_TAG)
      {
   		case GETR2:
		{	
			// Get the R2 value for this variable
            // ...and save it in the results vector

       		printf("Replica %d has received work request [%d]\n",
                  replicaname,work[0]);
          	
          	gsl_vector* betaMode = getcoefNR(work[0]);
			
	        workresults[1] = getLaplaceApprox(work[0], betaMode);
	        workresults[2] = MCIntegration(work[0], 10000);
	        gsl_vector* betaBayes = getPosteriorMeans(work[0], betaMode, 10000);
	        workresults[3] = gsl_vector_get(betaBayes,0);
	        workresults[4] = gsl_vector_get(betaBayes,1);

            // tell the primary what variable you're returning
            workresults[0] = (double)work[0];

            // Send the results
            MPI_Send(&workresults,
                     5,
                     MPI_DOUBLE,
                     0,		// send it to primary
                     0,		// doesn't need a TAG
                     MPI_COMM_WORLD);

            printf("Replica %d finished processing work request [%d]\n",
                   replicaname,work[0]);

            //free memory
            gsl_vector_free(betaMode);
            gsl_vector_free(betaBayes);

            break;
        }
     	case SHUTDOWNTAG:
   		{
   			printf("Replica %d was told to shutdown\n",replicaname);
            return;
     	}
     	default:
        {
            notDone = 0;
            printf("The replica code should never get here.\n");
            return;
        }
      }
   }

   // No memory to clean up, so just return to the main function
   return;
}
///////////////////////////////////////////////////

//computes pi_i = P(y_i = 1 | x_i)
gsl_vector* getPi(gsl_vector* x, gsl_vector* beta){
	int n = x->size;
	gsl_vector* one = gsl_vector_alloc(n);
	int i;
	for(i=0;i<n;i++){ //vector with all ones
		gsl_vector_set(one,i,1);
	}

	gsl_matrix* x0 = gsl_matrix_alloc(n,2); //design matrix
	gsl_matrix_set_col(x0, 0, one);
	gsl_matrix_set_col(x0, 1, x);

	gsl_vector* pi = gsl_vector_alloc(n);
	gsl_blas_dgemv(CblasNoTrans, 1.0, x0, beta, 0.0, pi);

	double tmp;
	for(i=0;i<n;i++){ //inverseLogit
		tmp = gsl_vector_get(pi,i);
		gsl_vector_set(pi,i,exp(tmp)/(1+exp(tmp)));
	}

	//free memory
	gsl_vector_free(one);
	gsl_matrix_free(x0);

	return(pi);
}

gsl_vector* getPi2(gsl_vector* x, gsl_vector* beta){
	int n = x->size;
	gsl_vector* one = gsl_vector_alloc(n);
	int i;
	for(i=0;i<n;i++){ //vector with all ones
		gsl_vector_set(one,i,1);
	}

	gsl_matrix* x0 = gsl_matrix_alloc(n,2); //design matrix
	gsl_matrix_set_col(x0, 0, one);
	gsl_matrix_set_col(x0, 1, x);

	gsl_vector* pi = gsl_vector_alloc(n);
	gsl_blas_dgemv(CblasNoTrans, 1.0, x0, beta, 0.0, pi);

	double tmp;
	for(i=0;i<n;i++){ //inverseLogit
		tmp = gsl_vector_get(pi,i);
		gsl_vector_set(pi,i,exp(tmp)/pow((1+exp(tmp)),2));
	}

	//free memory
	gsl_vector_free(one);
	gsl_matrix_free(x0);

	return(pi);
}

//logistic log-likelihood
double logisticLoglik(gsl_vector* y, gsl_vector* x, gsl_vector* beta){
	int n = x->size;
	gsl_vector* p = getPi(x, beta);

	int i;
	double yi, pi;
	double lik = 0.0;
	for(i=0;i<n;i++){
		yi = gsl_vector_get(y,i);
		pi = gsl_vector_get(p,i);
		lik += yi*log(pi)+(1-yi)*log(1-pi);
	}

	//free memory
	gsl_vector_free(p);

	return(lik);
}

gsl_matrix* getHessian(gsl_vector* x, gsl_vector* beta){
	gsl_matrix* hessian = gsl_matrix_alloc(2,2);
	gsl_vector* Pi2 = getPi2(x, beta);

	int i;
	double h00 = -1.0; // -1-sum(Pi2)
	double h01 = 0.0;
	double h11 = -1.0; // -1-sum(Pi2*x^2)
	double pi2, xi;
	for(i=0;i<n;i++){
		pi2 = gsl_vector_get(Pi2,i);
		xi = gsl_vector_get(x,i);
		h00 -= pi2;
		h01 -= pi2*xi;
		h11 -= pi2*xi*xi;
	}

	gsl_matrix_set(hessian,0,0,h00);
	gsl_matrix_set(hessian,0,1,h01);
	gsl_matrix_set(hessian,1,0,h01);
	gsl_matrix_set(hessian,1,1,h11);

	//free memory
	gsl_vector_free(Pi2);

	return(hessian);
}

gsl_vector* getGradient(gsl_vector* y, gsl_vector* x, gsl_vector* beta){
	gsl_vector* gradient = gsl_vector_alloc(2);
	gsl_vector* Pi = getPi(x, beta);

	int i;
	double g0 = -gsl_vector_get(beta,0); // -beta0+sum(y-Pi)
	double g1 = -gsl_vector_get(beta,1); // -beta0+sum((y-Pi)*x)

	double pi, xi, yi;
	for(i=0;i<n;i++){
		pi = gsl_vector_get(Pi,i);
		xi = gsl_vector_get(x,i);
		yi = gsl_vector_get(y,i);
		g0 += (yi - pi);
		g1 += (yi - pi)*xi;
	}

	gsl_vector_set(gradient,0,g0);
	gsl_vector_set(gradient,1,g1);

	//free memory
	gsl_vector_free(Pi);

	return(gradient);
}


gsl_vector* getcoefNR(int explanatory){
	double threshold = 1e-6;
	int max_iter = 10000;
	gsl_vector* beta = gsl_vector_alloc(2);
	gsl_vector* x = gsl_vector_alloc(n);
	gsl_matrix_get_col(x, data, explanatory); //change explanatory-1 later?

	//initialize beta at (0,0)
	gsl_vector_set(beta,0,0);
	gsl_vector_set(beta,1,0);

	//current value of log-likelihood
	double currentLoglik = logisticLoglik(Y,x,beta);

	gsl_vector* newBeta = gsl_vector_alloc(2);
	int i;
	// gsl_matrix* H = gsl_matrix_alloc(2,2);
	// gsl_matrix* Hinv = gsl_matrix_alloc(2,2); // H inverse
	// gsl_vector* G = gsl_vector_alloc(2);
	// gsl_vector* update = gsl_vector_alloc(2);
	double newLoglik;
	for(i=0;i<max_iter;i++){
		gsl_matrix* H = getHessian(x, beta);
		gsl_vector* G = getGradient(Y, x, beta);
		gsl_matrix* Hinv = inverse(H);
		gsl_vector* update = gsl_vector_alloc(2);

		gsl_blas_dgemv(CblasNoTrans, 1.0, Hinv, G, 0.0, update);

		//update newBeta
		gsl_vector_set(newBeta,0,gsl_vector_get(beta,0)-gsl_vector_get(update,0));
		gsl_vector_set(newBeta,1,gsl_vector_get(beta,1)-gsl_vector_get(update,1));
		newLoglik = logisticLoglik(Y,x,newBeta);

		//have to free memory before any breaks in the loop
		gsl_matrix_free(H);
		gsl_matrix_free(Hinv);
		gsl_vector_free(G);
		gsl_vector_free(update);

		if(newLoglik<currentLoglik){
			fprintf(stderr,"CODING ERROR!\n");
			break;
		}
		gsl_vector_memcpy(beta, newBeta);
		
		//stop if the log-likelihood doesn't improve by too much
		if(newLoglik-currentLoglik<threshold){ break; }
		currentLoglik = newLoglik;
	}
	//free memory
	gsl_vector_free(x);
	gsl_vector_free(newBeta);

	return(beta);
}

double getLaplaceApprox(int explanatory, gsl_vector* betaMode){
	gsl_vector* x = gsl_vector_alloc(n);
	gsl_matrix_get_col(x, data, explanatory); //change explanatory-1 later?

	double beta0 = gsl_vector_get(betaMode, 0);
	double beta1 = gsl_vector_get(betaMode, 1);

	gsl_matrix* H = getHessian(x, betaMode);

	

	double logPD = -0.5*(beta0*beta0+beta1*beta1)+logisticLoglik(Y,x,betaMode)-0.5*logdet(H);
	//*our logdet can handle non-PSD matrices

	//free memory
	gsl_vector_free(x);
	gsl_matrix_free(H);

	return(logPD);
}

//this function lower triangle Cholesky decomposition matrix of a matrix K
gsl_matrix* makeCholesky(gsl_matrix* K){
	int p = K->size1;
	gsl_matrix* decomp = gsl_matrix_alloc(p,p);//copy so that K won't change after the func
	if(GSL_SUCCESS!=gsl_matrix_memcpy(decomp,K))
	{
		printf("GSL failed to copy a matrix.\n");
		exit(1);
	}
	if(GSL_SUCCESS!=gsl_linalg_cholesky_decomp(decomp))
    {
        printf("GSL failed Cholesky decomposition.\n");
        exit(1);
    }

    int i,j;
    //set upper part of decomp to zero
    for(i=0;i<p-1;i++){
		for(j=i+1;j<p;j++){
			gsl_matrix_set(decomp,i,j,0.0);
		}
	}

    return(decomp);
}



void randomMVN(gsl_rng* mystream, gsl_matrix* samples, gsl_matrix* sigma){
	int n = samples->size1; //number of multivariate gaussian samples n*p
	int p = sigma->size1;
	int i,j;
	gsl_vector* tempvec = gsl_vector_alloc(p); //for storing a particular Z-sample
	gsl_matrix* zsamples = gsl_matrix_alloc(p,n);//storing n Z-samples
	for(i=0;i<n;i++){
		for(j=0;j<p;j++){
			gsl_vector_set(tempvec,j,gsl_ran_ugaussian(mystream));
		}
		gsl_matrix_set_col(zsamples, i, tempvec);
	}

	gsl_matrix* decomp = makeCholesky(sigma);
	gsl_matrix* samples_t = gsl_matrix_alloc(p,n);
	gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, decomp, zsamples, 0.0, samples_t);
	gsl_matrix* samples_temp = transposematrix(samples_t);
	if(GSL_SUCCESS!=gsl_matrix_memcpy(samples,samples_temp))
	{
		printf("GSL failed to copy a matrix.\n");
		exit(1);
	}

	//free memory
	gsl_vector_free(tempvec);
	gsl_matrix_free(decomp);
	gsl_matrix_free(zsamples);
	gsl_matrix_free(samples_t);
	gsl_matrix_free(samples_temp);

	return;	
}

double MCIntegration(int explanatory, int NumberOfIterations){
	gsl_vector* x = gsl_vector_alloc(n);
	gsl_matrix_get_col(x, data, explanatory); //change explanatory-1 later?

	gsl_matrix* samples = gsl_matrix_alloc(NumberOfIterations,2);//draw 10000 samples
	gsl_rng* r = gsl_rng_alloc(gsl_rng_default);

	gsl_matrix* sigma = gsl_matrix_alloc(2,2);
	gsl_matrix_set(sigma,0,0,1);
	gsl_matrix_set(sigma,1,0,0);
	gsl_matrix_set(sigma,0,1,0);
	gsl_matrix_set(sigma,1,1,1);
	randomMVN(r, samples, sigma);

	gsl_vector* betaSample = gsl_vector_alloc(2);
	double logmarglik = 0.0;
	int i;
	for(i=0;i<NumberOfIterations;i++){
		gsl_matrix_get_row(betaSample, samples, i);
		logmarglik += exp(logisticLoglik(Y,x,betaSample));
	}

	//free memory
	gsl_rng_free(r);
	gsl_vector_free(x);
	gsl_vector_free(betaSample);
	gsl_matrix_free(samples);
	gsl_matrix_free(sigma);


	return(log(logmarglik)-log(NumberOfIterations));
}

//For M-H algo

//For comparing l* (ignoring the constant log(2pi))
double lStar(gsl_vector* x, gsl_vector* beta){
	double beta0 = gsl_vector_get(beta,0);
	double beta1 = gsl_vector_get(beta,1);
	return(-0.5*(beta0*beta0+beta1*beta1)+logisticLoglik(Y,x,beta));
}

//performs one iteration of the M-H algo
void mhLogisticRegression(gsl_vector* x, gsl_vector* beta, gsl_vector* betaNext, gsl_matrix* invNegHessian){
	gsl_vector* betaCandidate = gsl_vector_alloc(2);
	gsl_matrix* samples = gsl_matrix_alloc(1,2);//draw 1 sample of N(0,invNegHessian)

	gsl_rng* r = gsl_rng_alloc(gsl_rng_default);
	randomMVN(r, samples, invNegHessian);

	gsl_vector_set(betaCandidate,0,gsl_vector_get(beta,0)+gsl_matrix_get(samples,0,0));
	gsl_vector_set(betaCandidate,1,gsl_vector_get(beta,1)+gsl_matrix_get(samples,0,1));

	double currentLStar = lStar(x, beta);
	double candidateLStar = lStar(x, betaCandidate);

	//free memory before the function returns
	gsl_matrix_free(samples);
	if(candidateLStar>=currentLStar){
		gsl_vector_memcpy(betaNext, betaCandidate);
	}

	double u = gsl_rng_uniform(r);
	if(u<=exp(candidateLStar - currentLStar)){
		gsl_vector_memcpy(betaNext, betaCandidate);
	}

	gsl_vector_memcpy(betaNext, beta);
	//free memory
	gsl_rng_free(r);
	gsl_vector_free(betaCandidate);
	return;
}

gsl_vector* getPosteriorMeans(int explanatory, gsl_vector* betaMode, int NumberOfIterations){
	gsl_vector* betaBayes = gsl_vector_alloc(2);
	gsl_vector* betaCurrent = gsl_vector_alloc(2);
	gsl_vector* x = gsl_vector_alloc(n);
	gsl_matrix_get_col(x, data, explanatory); //change explanatory-1 later?

	gsl_vector_memcpy(betaCurrent, betaMode);

	gsl_matrix* H = getHessian(x, betaMode);
	gsl_matrix* invH = inverse(H);
	gsl_matrix* invNegHessian = gsl_matrix_alloc(2,2);

	gsl_matrix_set(invNegHessian,0,0,-gsl_matrix_get(invH,0,0));
	gsl_matrix_set(invNegHessian,0,1,-gsl_matrix_get(invH,0,1));
	gsl_matrix_set(invNegHessian,1,0,-gsl_matrix_get(invH,1,0));
	gsl_matrix_set(invNegHessian,1,1,-gsl_matrix_get(invH,1,1));

	double betaBayes0 = 0.0;
	double betaBayes1 = 0.0;
	int i;
	for(i=0;i<NumberOfIterations;i++){
		mhLogisticRegression(x,betaCurrent,betaCurrent,invNegHessian);
		betaBayes0 += gsl_vector_get(betaCurrent,0);
		betaBayes1 += gsl_vector_get(betaCurrent,1);
	}

	gsl_vector_set(betaBayes,0,betaBayes0/NumberOfIterations);
	gsl_vector_set(betaBayes,1,betaBayes1/NumberOfIterations);

	//free memory
	gsl_vector_free(betaCurrent);
	gsl_vector_free(x);
	gsl_matrix_free(H);
	gsl_matrix_free(invH);
	gsl_matrix_free(invNegHessian);

	return(betaBayes);
}

//this functions counts the number of elements in a linked list
int getCount(LPRegression regressions)
{
    int count = 0; // Initialize count
    LPRegression current = regressions; // Initialize current
    while (current != NULL)
    {
        count++;
        current = current->Next;
    }
    return count;
}


//this function adds a new regression to a list of at most nMaxRegs.
//Here "regressions" represents the head of the list,
//with subsequent arguments being the estimates
void AddRegression(int nMaxRegs,LPRegression regressions,int explanatory,double LaplaceApprox,double mcIntegration,double betaBayes0,double betaBayes1)
{
  int i;
  LPRegression p = regressions;
  LPRegression pnext = p->Next;

  while(NULL!=pnext)
  {
     //return if we have previously found this regression
     if(explanatory == pnext->explanatory)
     {
        return;
     }

     //go to the next element in the list if the current
     //regression has a larger Monte Carlo integration than
     //the new regression A
     if(pnext->mcIntegration>mcIntegration)
     {
        p = pnext;
        pnext = p->Next;
     }
     else //otherwise stop; this is where we insert the new regression
     {
        break;
     }
  }

  //create a new element of the list
  LPRegression newp = new Regression;
  newp->explanatory = explanatory;
  newp->LaplaceApprox = LaplaceApprox;
  newp->mcIntegration = mcIntegration;
  newp->betaBayes0 = betaBayes0;
  newp->betaBayes1 = betaBayes1;
  

  //insert the new element in the list
  p->Next = newp;
  newp->Next = pnext;

  if(getCount(regressions)>nMaxRegs+1) // +1 for head
  {
    DeleteLastRegression(regressions); // delete the last (smallest marglik) element
  }

  return;
}

//this function deletes all the elements of the list
//with the head "regressions"
//remark that the head is not touched
void DeleteAllRegressions(LPRegression regressions)
{
  //this is the first regression
  LPRegression p = regressions->Next;
  LPRegression pnext;

  while(NULL!=p)
  {
    //save the link to the next element of p
    pnext = p->Next;

    p->Next = NULL;
    delete p;

    //move to the next element
    p = pnext;
  }

  return;
}

//this function deletes the last element of the list
//with the head "regressions"
//again, the head is not touched
void DeleteLastRegression(LPRegression regressions)
{
  //this is the element before the first regression
  LPRegression pprev = regressions;
  //this is the first regression
  LPRegression p = regressions->Next;

  //if the list does not have any elements, return
  if(NULL==p)
  {
     return;
  }

  //the last element of the list is the only
  //element that has the "Next" field equal to NULL
  while(NULL!=p->Next)
  {
    pprev = p;
    p = p->Next;
  }
  
  //now "p" should give the last element
  //delete it

  p->Next = NULL;
  delete p;

  //now the previous element in the list
  //becomes the last element
  pprev->Next = NULL;

  return;
}


//this function saves the regressions in the list with
//head "regressions" in a file with name "filename"
void SaveRegressions(char* filename,LPRegression regressions)
{
  int i;
  //open the output file
  FILE* out = fopen(filename,"w");
	
  if(NULL==out)
  {
    printf("Cannot open output file [%s]\n",filename);
    exit(1);
  }

  //this is the first regression
  LPRegression p = regressions->Next;
  while(NULL!=p)
  {
    //print the log marginal likelhood and the number of predictors
    fprintf(out,"index = %d, Laplace = %.5lf, MC = %.5lf, beta0 = %.5lf, beta1 = %.5lf\n",
    	p->explanatory,p->LaplaceApprox,p->mcIntegration,p->betaBayes0,p->betaBayes1);
    

    //go to the next regression
    p = p->Next;
  }

  //close the output file
  fclose(out);

  return;
}
