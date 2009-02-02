// LinAlg.cpp : CPU Conjugate Gradient Implementation
// Jan & Cory - CUDA 6.963 - 1/23/09

#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <stdlib.h> // for  atoi()
#include "cuTimer.h"

/* Include cblas */
#include "cblas.h" 

/* Includes, cuda */
#include "cublas.h" 
#include <cutil.h>

float* d_A = 0, *d_Xin=0, *d_Xout=0 ; // make them global for eaiser access

//........Prototypes
typedef void (*Av)(int n, float* a, float* xin, float *xout); // Heavy-duty computation
void AvCPU(int n, float* a, float* xin, float *xout);
void AvCUBLAS(int n, float* a, float* xin, float *xout);

void  conjGradCPU( void (*pAv)(int , float*, float* , float *),
	       int n, float* a, float* b, float* x, float tol, float &eps, int &niter);
void  conjGradBLAS(int n, float* a, float* b, float* x, float tol, float &eps, int &niter) ;
void  conjGradCUBLAS(int n, float* a, float* b, float* x, float tol, float &eps, int &niter) ;

FILE *fd=0; // input file with {A,B,X}, produced in python
void readABX(float* A, float* B, float* X,int n);


//==========================================================
//==========================================================
//Main Function
int main(int argc, char *argv[]) {
   float parMag=2.;

   if(argc!=4 ) {
    printf("Usage: %s  [size]  [method] [nTask] \n", argv[0]);
    return 0;
  }
  char *methodN[4]={"CPU","BLAS","CPU+CUBLAS","CUBLAS"};

  int size=atoi(argv[1]);
  int kMethod=atoi(argv[2]);
  int nTask=atoi(argv[3]);

  printf("===ConjGrad Calculation===\n  size=%d method=%s  nTask=%d\n",size,methodN[kMethod-1],nTask);
  assert(size>0);
  assert(kMethod>=1 && kMethod<=4); // CPU, Blas, mixed, GPU
 
 
   //........ CHOSE  CUDA DEVICE .......
   int num_devices;
   cudaGetDeviceCount(&num_devices);
   printf("	Cuda Device Count: %d\n",num_devices);
   int dev=1;  // use 0 for GTX200, 1 for Tesla @ karken1
   cudaSetDevice(dev);
   //....... Query device properties
   cudaDeviceProp deviceProp;
   CUDA_SAFE_CALL(cudaGetDeviceProperties(&deviceProp, dev));
   float dev_globMem= deviceProp.totalGlobalMem/1024./1024./1024.;
   printf("\nDevice %d: \"%s\"  tot glob mem=%.1f GB\n", dev, deviceProp.name,dev_globMem);


  float* h_A;
  cudaMallocHost((void **)&h_A, size*size*sizeof(float));
  
  float* B	= (float*)malloc(size*sizeof(float));
  float* Xref	= (float*)malloc(size*sizeof(float));
  float* X	= (float*)malloc(size*sizeof(float));
  

  /* Initialize CUBLAS */
  printf("CUBLAS initialization...\n");
  cublasStatus  status = cublasInit();
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! CUBLAS initialization error\n");
    return EXIT_FAILURE;
  }

  /* Allocate device memory for the matrices */
  status = cublasAlloc(size*size, sizeof(d_A[0]), (void**)&d_A);
  printf("status = %d\n", status);
  printf("CUBLAS_STATUS_SUCCESS = %d\n", CUBLAS_STATUS_SUCCESS);
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! device memory allocation error (A)\n");
    return EXIT_FAILURE;
  }
  status = cublasAlloc(size, sizeof(d_Xin[0]), (void**)&d_Xin);
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! device memory allocation error (Xin)\n");
    return EXIT_FAILURE;
  }
  status = cublasAlloc(size, sizeof(d_Xout[0]), (void**)&d_Xout);
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! device memory allocation error (Xout)\n");
    return EXIT_FAILURE;
  } 


  int nCycle=1;
  if(size>16)nCycle += 2048*2048/size/size;
  else  nCycle=10000;
  //....... start timed processing ................ 

  for(int jTask=0; jTask<nTask; jTask++ ) {//tttttttttttttttttttt START  
  readABX(h_A, B, Xref,size);


  printf("size=%d task=%d Generating Solution xRef[]...\n", size ,jTask);

  float eps=-1;
  int nIter=-1;
  float tol=-7;

  float tAtot=0, tBtot=0;
  //ccccccccccccccccccccccccccccccccccccccccccc START
  for(int jCycle=0; jCycle<nCycle; jCycle++) { 
  // generate true answer xRef[] & starting point x[]
  for(int i=0; i<size; i++) {// starting vector close to true solution
     X[i] = Xref[i]+parMag*(1.-2.*rand()/(float)RAND_MAX);
  }

  cuResetTimer();  
  if(kMethod>=3) { // any involving GPU
    /* Initialize the device matrices with the host matrices */
    status = cublasSetVector(size*size, sizeof(h_A[0]), h_A, 1, d_A, 1);
    if (status != CUBLAS_STATUS_SUCCESS) {
      fprintf (stderr, "!!!! device access error (write A)\n");
      return EXIT_FAILURE;
    }
  }
  tAtot+=cuGetTimer();
  //...... matrix A ready for use .......

  switch( kMethod) {
    case 1: //:::::::::::::::::::::::: CPU ::::::::::::::::::::::
       conjGradCPU( AvCPU,size,h_A,B,X,tol,eps,nIter);
       break;
    case 2: //::::::::::::::::::::::: PURE CPU BLAS :::::::::::::
        conjGradBLAS(size,h_A,B,X,tol,eps,nIter);
        break;
    case 3: //::::::::::::::::::::: Mixed CPU+GPU-CUBLAS ::::::::
        conjGradCPU( AvCUBLAS,size,h_A,B,X,tol,eps,nIter);
        break;
    case 4:  //:::::::::::::::::::: PURE GPU CUBLAS :::::::::::::
       conjGradCUBLAS(size,d_A,B,X,tol,eps,nIter);
       break;
    default: assert(1==2); // wrong key value
  }
  tBtot+=cuGetTimer();
  }// ccccccccccccccccccccccccccccccccccc END 

  printf("method,  size, traial,Tmem(ms),Tmath(ms),lastEps,lastNiter,nCycle,devivce, globmem(GB)\n");
  printf("#%s, %d,    %d,    %.3f,   %.3f,  %.3f,  %d, %d, %s, %.1f\n",methodN[kMethod-1],size,jTask, tAtot/nCycle, tBtot/nCycle, eps,  nIter,nCycle, deviceProp.name,dev_globMem); 

  if(jTask==0) { //Print out results
  printf("\nexact & fitted slution for %s\n",methodN[kMethod-1]);
  for(int i=0; i<size; i++) {
    printf("<i=%2d>  (xRef=%8.3f  xFit=%8.3f)   d1=%8.3f \n",i,Xref[i],X[i], Xref[i]-X[i]);
    if(i>10) break;
    }
  }
 }//tttttttttttttttttttttttttttttttttttttt END 
  
}

//===========================================
//===========================================
//===========================================

//Heavy computation on local CPU
void AvCPU(int n, float* a, float* xin, float *xout) {
  for(int j=0; j<n; j++) {
    float sum = 0; //initialize
    for(int i=0; i<n; i++) {
      sum += a[i+j*n] * xin[i];
    }
    xout[j]=sum;
  }
}


//===========================================
//===========================================
//===========================================
//Conjugate Gradient Procedure
void  conjGradCPU( void (*pAv)(int , float*, float* , float *),
           int n, float* a, float* b, float* x, float tol, float &eps, int &niter) {
         //(Av-func, size , [a]{x} = {b}	, starting	, tolerance)	
  int len=n*sizeof(float);
  float *r= (float*)malloc(len); // working space
  float *s= (float*)malloc(len); // working space
  float *u= (float*)malloc(len); // working space
  eps=-1;
  niter=-2;
  
  (pAv)(n,a,x,u); // HEAVY COMPUTING

  for(int i=0;i<n;i++) // r = b - Av(x);  s=r.copy()
    s[i]=r[i]=b[i]-u[i];
  
  //......... iterations ............
  int k;
  for (k=0; k<n; k++) {
    (pAv)(n,a,s,u); // HEAVY COMPUTING //u = Av(s)
    float alpha1 =0., alpha2=0;
    for(int i=0;i<n;i++) {
      alpha1+=s[i]*r[i];
      alpha2+=s[i]*u[i];
    }
    float alpha=alpha1/alpha2;
  
    for(int i=0;i<n;i++) // 	//x = x+alpha*s
      x[i]=x[i]+alpha*s[i];
    //printf("al=%f s=%f newX=%f\n",alpha, s[0],x[0]);  
    (pAv)(n,a,x,r); // HEAVY COMPUTING , use r[] as temp storage
    
    float sum2=0;
    for(int i=0;i<n;i++) { // r = b - Av(x); eps
      r[i] =b[i]-r[i]; // compute new r
      sum2+=r[i]*r[i];
    }
    eps=sqrt(sum2);
   //printf("conjGrad: k=%d eps=%.3f\n",k,eps);
    if(eps < tol)      break;
    float beta1=0, beta2=0;
    for(int i=0;i<n;i++) {// compute beta
      beta1+=r[i]*u[i];
      beta2+=s[i]*u[i];
    }
    float beta=beta1/beta2;
    for(int i=0;i<n;i++) // 
      s[i]=r[i]-beta*s[i]; //s = r + beta*s    
  }// end of iterations
  niter=k;

  //......... free memory at the end
  free(u);
  free(s);
  free(r);
 
}


//-------------------------------------------
//Cublas Version of the Code
void  conjGradCUBLAS(int n, float* d_a, float* b, float* x, float tol, float &eps, int &niter) {

	//Initialize our variables and transfer to the device!
	
	float* d_b;
	float* d_x;
	float* d_r;
	float* d_s;
	float* d_u;
	float* d_z;
	
	eps=-1;
	niter=-3;


	cublasAlloc(n, sizeof(float), (void**)&d_b);
	cublasAlloc(n, sizeof(float), (void**)&d_x);
	cublasAlloc(n, sizeof(float), (void**)&d_r);
	cublasAlloc(n, sizeof(float), (void**)&d_s);
	cublasAlloc(n, sizeof(float), (void**)&d_u);
	cublasAlloc(n, sizeof(float), (void**)&d_z); //backup for b;
  

	cublasSetVector(n, sizeof(float), b, 1, d_b, 1);
	cublasSetVector(n, sizeof(float), x, 1, d_x, 1);
	cublasSetVector(n, sizeof(float), b, 1, d_z, 1); //z is a b backup
	
	//----------------------
	
	//n=len(b)
	//r = b-Av(x)
	//s = r.copy()
	
	cublasSgemv('n',n,n,-1,d_a,n,d_x,1,1,d_b,1); //b=-Ax+b
	cublasScopy(n,d_b,1,d_r,1); // r=b
	cublasScopy(n,d_b,1,d_s,1); // s=b
	cublasScopy(n, d_z, 1, d_b, 1); //reset b

	//-----------------------
	
	//For i in range(n)

	int k;
	for(k=0; k<n; k++) {
		cublasSgemv('n',n,n,1,d_a,n,d_s,1,0,d_u,1);  //u=Av(s)
		float alpha = cublasSdot(n, d_s, 1, d_r, 1) / cublasSdot(n, d_s, 1, d_u, 1); //alpha = dot(s,r)/dot(s,u)
		cublasSaxpy(n, alpha, d_s, 1, d_x, 1);  //x = x + alpha * s

		//r = b - Av(x)
		cublasSgemv('n',n,n,-1,d_a,n,d_x,1,1,d_b,1); //b=-Ax+b
		cublasScopy(n,d_b,1,d_r,1); // r=b
		cublasScopy(n, d_z, 1, d_b, 1);; //reset b
		
		eps = sqrt(cublasSdot(n, d_r, 1, d_r, 1));
		if(eps<tol) {
			break;
		} else {
			float beta = -cublasSdot(n, d_r, 1, d_u, 1) / cublasSdot(n, d_s, 1, d_u, 1);
			cublasSscal(n,beta,d_s,1); //s=beta*s
			cublasSaxpy(n,1,d_r,1,d_s,1);
		}
		
	}
	niter=k;

	cublasGetVector(n, sizeof(float), d_x, 1, x, 1); 

	cublasFree(d_b);
	cublasFree(d_x);
	cublasFree(d_r);
	cublasFree(d_s);
	cublasFree(d_u);
	cublasFree(d_z);

}


///////////////////////////////////////////////////////////////////////////////////////////
//BLAS Version

void conjGradBLAS(int n, float* a, float* b, float* x, float tol, float &eps, int &niter) {

	//Initialize our variables and transfer to the device!
	
	float* r= (float*)malloc(n*sizeof(float));
	float* s= (float*)malloc(n*sizeof(float));
	float* u= (float*)malloc(n*sizeof(float));
	float* z= (float*)malloc(n*sizeof(float));
	
	eps=-1;
        niter=-2;

	cblas_scopy(n,b,1,z,1);
	//cblas_SetVector(n, sizeof(float), b, 1, z, 1); //z is a b backup
	
	//----------------------
	
	//n=len(b)
	//r = b-Av(x)
	//s = r.copy()
	
	cblas_sgemv(CblasRowMajor, CblasNoTrans,n,n,-1,a,n,x,1,1,b,1); //b=-Ax+b
	cblas_scopy(n,b,1,r,1); // r=b
	cblas_scopy(n,b,1,s,1); // s=b
	cblas_scopy(n, z, 1, b, 1); //reset b

	//-----------------------
	
	//For i in range(n)

        int k; 
	for(k=0; k<n; k++) {
		cblas_sgemv(CblasRowMajor,CblasNoTrans,n,n,1,a,n,s,1,0,u,1);  //u=Av(s)
		float alpha = cblas_sdot(n, s, 1, r, 1) / cblas_sdot(n, s, 1, u, 1); //alpha = dot(s,r)/dot(s,u)
		cblas_saxpy(n, alpha, s, 1, x, 1);  //x = x + alpha * s

		//r = b - Av(x)
		cblas_sgemv(CblasRowMajor,CblasNoTrans,n,n,-1,a,n,x,1,1,b,1); //b=-Ax+b
		cblas_scopy(n,b,1,r,1); // r=b
		cblas_scopy(n,z,1,b,1); //reset b
		
		eps = sqrt(cblas_sdot(n, r, 1, r, 1));
		if(eps<tol) {
			break;
		} else {
			float beta = -cblas_sdot(n, r, 1, u, 1) / cblas_sdot(n, s, 1, u, 1);
			cblas_sscal(n,beta,s,1); //s=beta*s
			cblas_saxpy(n,1,r,1,s,1);
		}
		
	}

	niter=k;
	free(r);
	free(s);
	free(u);
	free(z);
}



//===========================================
//===========================================
//===========================================

//Heavy computation, using CUBLAS
void AvCUBLAS(int N, float* h_A, float* h_Xin, float *h_Xout) {

  /* h_A matrix was already loaded in init */
  
  /* Initialize the device matrices with Xin host matrics */
  cublasStatus   status = cublasSetVector(N, sizeof(h_Xin[0]), h_Xin, 1, d_Xin, 1);
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! device access error (write Xin)\n");
    assert(1==2);
  }
  float tcub2= cuGetTimer();  
  // printf("CUBAS copy Xin->dev   time=%.2f uSec\n", tcub2*1000.); 

  /* Clear last error */
  cublasGetError();

  /* Performs operation using cublas */
  cublasSgemm('n', 'n', N, 1, N, 1.0f, d_A, N, d_Xin, N, 0., d_Xout, N);
  status = cublasGetError();
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! kernel execution error.\n");
    assert(1==2);
  }


  /* Read the result back */
  status = cublasGetVector(N, sizeof(h_Xout[0]), d_Xout, 1, h_Xout, 1);
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! device access error (read Xout)\n");
    assert(1==2);
  }

}


//========================================
//========================================
//Helper Functions
void readABX( float* A, float* B,  float* X,int size) {
  if(fd==0) {
    char *pathIn="./dataA/";
    char fname[1000];
  
    sprintf(fname,"%s/conjData_%d.bin",pathIn,size);
    printf("Open Matrix file'%s'\n",fname);
    fd=fopen( fname,"r"); 
    if (!fd) {
         printf( "Unable to open file, exit\n"); 
         exit(1); // terminate with error
    }
  } 

  int ret=fread(A,sizeof(float),size*size,fd);      
  printf("matrix A got %d values\n",ret);
  assert(ret==size*size);
   int retB=fread(B,sizeof(float),size,fd);      
   int retX=fread(X,sizeof(float),size,fd);      
   printf("read   B + X has dim %d+%d\n",retB,retX);
  assert(retB==size);
  assert(retX==size);
#if 0   
   for(int i=0;i<size*size;i++) {
      printf("i=%d val=%f \n",i,A[i]);
   }
   for(int i=0;i<size;i++) {
      printf("i=%d B=%f  X=%f \n",i,B[i],X[i]);
   }
#endif
}

