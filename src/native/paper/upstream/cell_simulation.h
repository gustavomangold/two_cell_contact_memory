#ifndef cell_simulation_HEADER
#define cell_simulation_HEADER

#include <stdlib.h>  
#include <string.h>  //for strcmp()
#include <stdint.h>  //to use uint64_t
#include <time.h>
#include <math.h>
#include <assert.h>
#include <gsl/gsl_multimin.h>
#include <gsl/gsl_vector.h>
#include "global.h"
#include "base_types2D.h"
#include "xoshiro_rng.h"
#include "cell_simulation.h"



double getNextParameter(FILE * file, char * parameterName);

void SetParameters(int argc, char** argv);

void AllocArrays();

//Potential Energy to be minimised
double fHarmonicPotential(const gsl_vector *v, void *params);

//Gradient of the potential (For gsl_multimin energy minimisation of initial positons)
void dfHarmonicPotential(const gsl_vector *v, void *params, gsl_vector *df);

//Potential and Gradient together (For gsl_multimin energy minimisation of initial positons)
void fdfHarmonicPotential(const gsl_vector *x, void * params, double * f, gsl_vector *df);

void InitialisePositions();

void InitialiseColor();

void InitialiseAngles();

void MeasurePositions(real time);

void SetUpJob();

//Begin: Support Functions for Single Step
real HarmonicForce(real distance);

real LennardJonesForce(real distance);

real GetAngle(VecR r);

void ComputeInteractions();

void EulerMaruyamaR();

void EulerMaruyamaTheta();

void EnforcePeriodicBoundares();

void UpdatePersistence();

//End: Support Functions for Single Step
void SingleStep (int stepIdx);

void writeTracks();

void cleanup();


#endif
