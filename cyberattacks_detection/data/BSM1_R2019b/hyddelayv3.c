/*
 * hyddelay is a C-file S-function for first order reaction of flow and conc.  
 * In this version hyddelayv3 the loads are first calculated and the first
 * order reaction is used for the load and flow. After this the concentrations
 * recalculated based on the delayed flow and load. Note that the state
 * vector internally here represents mass. Better according to Jens Alex.
 * State no 14 (TSS) is a dummy state to maintain consistence with the normal
 * vector size used in BSM1.
 *
 * Copyright: Ulf Jeppsson, IEA, Lund University, Lund, Sweden
 */

#define S_FUNCTION_NAME hyddelayv3

#include "simstruc.h"

#define XINIT   ssGetArg(S,0)
#define PAR     ssGetArg(S,1)
#define T       ssGetArg(S,2)

/*
 * mdlInitializeSizes - initialize the sizes array
 */
static void mdlInitializeSizes(SimStruct *S)
{
    ssSetNumContStates(    S, 15); /*15 number of continuous states           */
    ssSetNumDiscStates(    S, 0);   /* number of discrete states             */
    ssSetNumInputs(        S, 15);   /* number of inputs                      */
    ssSetNumOutputs(       S, 15);   /* number of outputs                     */
    ssSetDirectFeedThrough(S, 0);   /* direct feedthrough flag               */
    ssSetNumSampleTimes(   S, 1);   /* number of sample times                */
    ssSetNumSFcnParams(    S, 3);   /* number of input arguments             */
    ssSetNumRWork(         S, 0);   /* number of real work vector elements   */
    ssSetNumIWork(         S, 0);   /* number of integer work vector elements*/
    ssSetNumPWork(         S, 0);   /* number of pointer work vector elements*/
}

/*
 * mdlInitializeSampleTimes - initialize the sample times array
 */
static void mdlInitializeSampleTimes(SimStruct *S)
{
    ssSetSampleTime(S, 0, CONTINUOUS_SAMPLE_TIME);
    ssSetOffsetTime(S, 0, 0.0);
}


/*
 * mdlInitializeConditions - initialize the states
 */
static void mdlInitializeConditions(double *x0, SimStruct *S)
{
int i;

for (i = 0; i < 15; i++) {
   x0[i] = mxGetPr(XINIT)[i];
}
}

/*
 * mdlOutputs - compute the outputs
 */

static void mdlOutputs(double *y, double *x, double *u, SimStruct *S, int tid)
{
  double X_I2TSS, X_S2TSS, X_BH2TSS, X_BA2TSS, X_P2TSS;
  int i;

  X_I2TSS = mxGetPr(PAR)[19];
  X_S2TSS = mxGetPr(PAR)[20];
  X_BH2TSS = mxGetPr(PAR)[21];
  X_BA2TSS = mxGetPr(PAR)[22];
  X_P2TSS = mxGetPr(PAR)[23];

  for (i = 0; i < 13; i++) {
      y[i] = x[i]/x[14];
  }

  y[13]=(X_I2TSS*x[2]+X_S2TSS*x[3]+X_BH2TSS*x[4]+X_BA2TSS*x[5]+X_P2TSS*x[6])/x[14];
  y[14]=x[14];
}

/*
 * mdlUpdate - perform action at major integration time step
 */

static void mdlUpdate(double *x, double *u, SimStruct *S, int tid)
{
}

/*
 * mdlDerivatives - compute the derivatives
 */
static void mdlDerivatives(double *dx, double *x, double *u, SimStruct *S, int tid)
{
int i;
double timeconst;

timeconst = mxGetPr(T)[0];
if (timeconst > 0.000001) {
  dx[0] = (u[0]*u[14]-x[0])/timeconst;
  dx[1] = (u[1]*u[14]-x[1])/timeconst;
  dx[2] = (u[2]*u[14]-x[2])/timeconst;
  dx[3] = (u[3]*u[14]-x[3])/timeconst;
  dx[4] = (u[4]*u[14]-x[4])/timeconst;
  dx[5] = (u[5]*u[14]-x[5])/timeconst;
  dx[6] = (u[6]*u[14]-x[6])/timeconst;
  dx[7] = (u[7]*u[14]-x[7])/timeconst;
  dx[8] = (u[8]*u[14]-x[8])/timeconst;
  dx[9] = (u[9]*u[14]-x[9])/timeconst;
  dx[10] = (u[10]*u[14]-x[10])/timeconst;
  dx[11] = (u[11]*u[14]-x[11])/timeconst;
  dx[12] = (u[12]*u[14]-x[12])/timeconst;
  dx[13] = (u[13]*u[14]-x[13])/timeconst;
  dx[14] = (u[14]-x[14])/timeconst; }
else {
  dx[0] = 0;
  dx[1] = 0;
  dx[2] = 0;
  dx[3] = 0;
  dx[4] = 0;
  dx[5] = 0;
  dx[6] = 0;
  dx[7] = 0;
  dx[8] = 0;
  dx[9] = 0;
  dx[10] = 0;
  dx[11] = 0;
  dx[12] = 0;
  dx[13] = 0; 
  dx[14] = 0; 
  x[0] = u[0]*u[14];
  x[1] = u[1]*u[14];
  x[2] = u[2]*u[14];
  x[3] = u[3]*u[14];
  x[4] = u[4]*u[14];
  x[5] = u[5]*u[14];
  x[6] = u[6]*u[14];
  x[7] = u[7]*u[14];
  x[8] = u[8]*u[14];
  x[9] = u[9]*u[14];
  x[10] = u[10]*u[14];
  x[11] = u[11]*u[14];
  x[12] = u[12]*u[14];
  x[13] = u[13]*u[14];
  x[14] = u[14];
}
}


/*
 * mdlTerminate - called when the simulation is terminated.
 */
static void mdlTerminate(SimStruct *S)
{
}

#ifdef	MATLAB_MEX_FILE    /* Is this file being compiled as a MEX-file? */
#include "simulink.c"      /* MEX-file interface mechanism */
#else
#include "cg_sfun.h"       /* Code generation registration function */
#endif
