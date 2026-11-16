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
#include "agent_simulation_config.h"
#include "stdbool.h"


//Gunnar's malloc macro, gives error if space can't be assigned (e.g. too big -> NULL returned from OS) and keeps taps on the ammount of space used 
long long int total_malloced=0LL;
#define MALLOC(a,n) {if ((a=malloc((n)*sizeof(*a)))==NULL) { fprintf(stderr, "Failed to malloc %i items of size %i (%lli bytes) for variable %s in line %i of %s\n", (int)(n), (int)sizeof(*a), (long long int)((n)*sizeof(*a)), #a, __LINE__, __FILE__); } else { total_malloced+=((long long int)((n)*sizeof(*a)));} }
//The verbose version prints the space used so far
#define MALLOC_VERBOSE(a,n) {if ((a=malloc((n)*sizeof(*a)))==NULL) { fprintf(stderr, "Failed to malloc %i items of size %i (%lli bytes) for variable %s in line %i of %s\n", (int)(n), (int)sizeof(*a), (long long int)((n)*sizeof(*a)), #a, __LINE__, __FILE__); } else { total_malloced+=((long long int)((n)*sizeof(*a))); printf("# Info: %lli bytes malloced for %s, %lli in total so far.\n", (long long int)((n)*sizeof(*a)), #a, total_malloced); } }

double getNextParameter(FILE * file, char * parameterName){
    char line[MAX_LINE_LENGTH] = {0};
    char name[MAX_STRING_LENGTH];
    char value[MAX_STRING_LENGTH];
    if (fgets(line,MAX_LINE_LENGTH,file)) {
        sscanf(line, "%s : %s", name, value);
    } 
    char * ptr; //Throughaway, needed to call strtod
    double val = strtod(value,&ptr);
    printf("%s = %f \n",name,val);
    assert(strcmp(name,parameterName)==0); //Do the names match? If not it's an invalid parameter or parameter out of order or not enough parameters
    return val;
}

void SetParameters(int argc, char** argv){
    char PATH[500];
    char * trackFileSuffix;
    if(argc==2){ //Tracks file name unspecified
        strcpy(PATH,argv[1]);
        trackFileSuffix = "_tracks.csv";
    } else if (argc==3){ //Tracks file name specified
        strcpy(PATH,argv[1]);
        trackFileSuffix = argv[2];
    } else {
        assert(false); // main needs at least one and at most two command line inputs
    }
    
    // Files
    paramFile = fopen(PATH,"r+");
    char tmp[500];
    strcpy(tmp, PATH);
    tracksFile = fopen(strcat(tmp,trackFileSuffix),"w");

    printf("Parameter file: "); printf(PATH); printf("\n");

    #ifdef TRACK_VELOCITIES
    strcat(PATH, "_velocities");
    velocityTracksFile = fopen(strcat(PATH,trackFileSuffix),"w");
    //for some reason the strcat(...) in the previous line changes trackFileSuffix to "velocities". Why???
    #endif

    printf("Initialise paramters:\n");
    
    //Read out parameters
    stepLimit = getNextParameter(paramFile, "stepLimit");
    stepDuration = getNextParameter(paramFile, "stepDuration");
    skipSteps = getNextParameter(paramFile, "skipSteps"); 
    measurementInterval = getNextParameter(paramFile, "measurementInterval");
    nGreenParticles = getNextParameter(paramFile, "nGreenParticles");
    nRedParticles = getNextParameter(paramFile, "nRedParticles");
    real areaFraction = getNextParameter(paramFile, "areaFraction");
    redD = getNextParameter(paramFile, "redD");
    greenD = getNextParameter(paramFile, "greenD");
    greenPersistentD = getNextParameter(paramFile, "greenPersistentD");
    k = getNextParameter(paramFile, "k");
    tau = getNextParameter(paramFile, "tau");
    Pe = getNextParameter(paramFile, "Pe");
    potentialRange = getNextParameter(paramFile, "potentialRange");
    LennardJones = getNextParameter(paramFile, "LennardJones");
    turnAround = getNextParameter(paramFile, "turnAround");
    redRedAdhesionMult = getNextParameter(paramFile,"redRedAdhesionMult");
    greenGreenAdhesionMutl = getNextParameter(paramFile,"greenGreenAdhesionMutl");
    redGreenAdhesionMult = getNextParameter(paramFile,"redGreenAdhesionMult");
    // Barricades
    //measurementInterval must be a multiple of stepDuration 
    assert((measurementInterval/stepDuration)-round(measurementInterval/stepDuration)< 1e-4);
    //The Simulation duration must be a multiple of measurementInterval
    assert((stepLimit-skipSteps)*(stepDuration/measurementInterval)-round((stepLimit-skipSteps)*(stepDuration/measurementInterval))<1e-4); 
    //Can't skip more steps than we have
    assert(stepLimit>=skipSteps);
    //Need it least 20 steps, otherwise the loading bar divides by zero
    assert(stepLimit>=20);

    //Initialise secondary parameters
    nParticles = nGreenParticles + nRedParticles;
    #ifdef INITIAL_PHASE_SEGREGATION
    //Qudratic packing, ideally the number of particles is the square of an integter
    real length = ceil(sqrt(nParticles));
    #else
    //Setup Square area to fit the area fraction; each particle covers area of size pi*0.5^2 = 0.785398
    real length = sqrt(nParticles*0.785398/areaFraction);
    #endif
    printf("length=%f \n",length);
    region = (VecR) {.x = length,
                     .y = length};  //Size of the simulation region
    cells.x = region.x/(potentialRange+CELL_SIZE_EXTENSION); //Smallest cells that are larger than interaction range
    cells.y = region.y/(potentialRange+CELL_SIZE_EXTENSION);
    nMeasurements = round((stepLimit-skipSteps)*(stepDuration/measurementInterval)) + 1; //+1 one because we do the first and the last step 
    seed = SEED;//time(NULL);
    rng = xoshiro256ss_init(seed);
    srand(seed);

    //append the file
    fprintf(paramFile, "nParticles              : %i\n",nParticles);
    fprintf(paramFile, "Length                  : %f\n",length);
    fprintf(paramFile, "Seed                    : %i \n",seed);

    //Set up log file
    fprintf(paramFile, "\n");
    time_t tm;
    time(&tm);
    fprintf(paramFile,"LOGFILE: %s",ctime(&tm));  
}

void AllocArrays(){
    //Particle array
    MALLOC(particles, nParticles); //particles = (particle *) malloc(nParticles*(sizeof(particle)));
    //Linked list for cells
    MALLOC(cellList, (nParticles + VProd(cells)));//cellList = (int *) malloc((nParticles + VProd(cells))*sizeof(int));
    MALLOC_VERBOSE(neighbourList, 2*MAX_NEIGHBOUR_PAIRS);
};

//Potential Energy to be minimised
double fHarmonicPotential(const gsl_vector *v, void *params){
    VecR * positions;
    positions = (VecR *) malloc(nParticles*sizeof(VecR));
    for(int particleIdx=0; particleIdx<nParticles; particleIdx++){
        positions[particleIdx].x = gsl_vector_get(v, 2*particleIdx);
        positions[particleIdx].y = gsl_vector_get(v, 2*particleIdx + 1);
    }
    double * para = (double *) params;
    double potentialRange = para[0];

    double V = 0;
    VecR deltaR;
    real distance;
    for(int pIdx1 = 0; pIdx1 < nParticles; pIdx1++){
        for(int pIdx2 = pIdx1+1; pIdx2 < nParticles; pIdx2++){
            VSub(deltaR, positions[pIdx1],positions[pIdx2]);
            VWrapAllTang(deltaR); //Periodic Boundary conditions
            distance = sqrt(VLenSq(deltaR));
            if(distance < potentialRange){
                V += (distance-1)*(distance-1);
            }
        }
    }
    free(positions);
    return V;
}

//Gradient of the potential (For gsl_multimin energy minimisation of initial positons)
void dfHarmonicPotential(const gsl_vector *v, void *params, gsl_vector *df){
    //ToDo: Take care of divergence at 0
    VecR * positions, *gradient;
    positions = (VecR *) malloc(nParticles*sizeof(VecR));
    gradient = (VecR *) malloc(nParticles*sizeof(VecR));
    for(int particleIdx=0; particleIdx<nParticles; particleIdx++){
        positions[particleIdx].x = gsl_vector_get(v, 2*particleIdx);
        positions[particleIdx].y = gsl_vector_get(v, 2*particleIdx + 1);
    }
    double * para = (double *) params;
    double potentialRange = para[0]; //Interaction range
    VecR deltaR;
    real distance;
    //Calculate gradient
    for(int gradientIdx = 0; gradientIdx < nParticles; gradientIdx++){
        VSet(gradient[gradientIdx],0,0);
        for(int particleIdx = 0; particleIdx < nParticles; particleIdx++){
            if(particleIdx != gradientIdx){
                VSub(deltaR, positions[gradientIdx],positions[particleIdx]);
                VWrapAllTang(deltaR); //Periodic Boundary conditions
                distance = sqrt(VLenSq(deltaR));
                if(distance==0){ //Avoid undefined behaviour and just send the particle in some direction
                    VecR direction = {.x = 1, .y = 1};
                    VSAdd(gradient[gradientIdx], gradient[gradientIdx], -2, direction);
                }
                else if(distance < potentialRange){
                    VSAdd(gradient[gradientIdx], gradient[gradientIdx], 2*(1-1/distance), deltaR);
                }
            }
        }
    }
    //Write gradient to gsl vector (For gsl_multimin energy minimisation of initial positons)
    for(int gradientIdx = 0; gradientIdx < nParticles; gradientIdx++){
        gsl_vector_set(df, 2*gradientIdx    , gradient[gradientIdx].x);
        gsl_vector_set(df, 2*gradientIdx + 1, gradient[gradientIdx].y);
    }
    free(positions);
    free(gradient);
}

//Potential and Gradient together (For gsl_multimin energy minimisation of initial positons)
void fdfHarmonicPotential(const gsl_vector *x, void * params, double * f, gsl_vector *df){
    *f = fHarmonicPotential(x, params);
    dfHarmonicPotential(x, params, df);
}

void InitialisePositions(){    
    //Assign random initial positions
    double uniform1, uniform2;
    for(int particleIdx = 0; particleIdx < nParticles; particleIdx++){
        uniform1 = xoshire256ss_uniform(&rng);
        uniform2 = xoshire256ss_uniform(&rng);

        #ifdef INITIAL_BLOB
        real box_size = sqrt(nGreenParticles*M_PI/4); //Box with area equal to total area of green Particles
        if(particles[particleIdx].color==0){ // Place red particles at the edge
            particles[particleIdx].r.x = region.x*uniform1;
            particles[particleIdx].r.y = region.y*uniform2;
            // Shift particles out of the central box
            if(particles[particleIdx].r.x< 0.5*region.x+0.5*box_size
                && particles[particleIdx].r.x> 0.5*region.x-0.5*box_size
                && particles[particleIdx].r.y< 0.5*region.y+0.5*box_size
                && particles[particleIdx].r.y> 0.5*region.y-0.5*box_size){
                    particles[particleIdx].r.x += box_size;
                    // particles[particleIdx].r.y += region.y;
                }
        } else { // Blob green particles at the center
            particles[particleIdx].r.x = 0.5*region.x+(uniform1-0.5)*box_size;
            particles[particleIdx].r.y = 0.5*region.y+(uniform2-0.5)*box_size;
        }
        #elif defined INITIAL_PHASE_SEGREGATION
        int rowLength = ceil(sqrt(nParticles));
        int xPosition = particleIdx % rowLength;
        int yPosition = particleIdx / rowLength;
        particles[particleIdx].r.x = xPosition + 0.5;
        particles[particleIdx].r.y = yPosition + 0.5;
        #else
        particles[particleIdx].r.x = region.x*uniform1;
        particles[particleIdx].r.y = region.y*uniform2;
        #endif
    }
    

    //Minimise Energy of initial positions with GSL conjugate gradient

    //Parameters
    double stepSize = STEP_SIZE; //Initial step size
    double lineTol = LINE_TOL; //Error tolerance for the line minimisation in a given direction
    double tol = TOL; //Error tolerance for the overall minimisation
    int maxIter = MAX_ITER;

    //Initialise function to minimise
    double params[1] = {1.0};
    gsl_multimin_function_fdf func;
    func.f      =   &fHarmonicPotential;
    func.df     =  &dfHarmonicPotential;
    func.fdf    = &fdfHarmonicPotential;
    func.n      = 2*nParticles; //Number of variables
    func.params = (void*) params;

    //starting point
    gsl_vector *x;
    x = gsl_vector_alloc(2*nParticles);
    
    for(int particleIdx = 0; particleIdx < nParticles; particleIdx++){
        gsl_vector_set(x,2*particleIdx,     particles[particleIdx].r.x);
        gsl_vector_set(x,2*particleIdx + 1, particles[particleIdx].r.y);
    }

    //Initialise minimizer
    const gsl_multimin_fdfminimizer_type *T;
    gsl_multimin_fdfminimizer *s;
    T = gsl_multimin_fdfminimizer_conjugate_fr; //Fletcher-Reeves conjugate gradient algorithm
    s = gsl_multimin_fdfminimizer_alloc (T, 2*nParticles); 
    gsl_multimin_fdfminimizer_set (s, &func, x, stepSize, lineTol);

    printf("Initial Potential Energy: %f \n",s->f);
    fprintf(paramFile,"Initial Potential Energy: %f \n",s->f);

    //Minimise potential energy of particle configuration
    size_t iter = 0;
    int status;
    do{
    iter++;
    status = gsl_multimin_fdfminimizer_iterate (s);
    
    if (status)
        break;

    status = gsl_multimin_test_gradient (s->gradient, tol);

    // if (status == GSL_SUCCESS)
    //     printf ("Minimum found at:\n");

    // printf ("%5d (%.3f %.3f) (%.3f %.3f) %10.5f\n", iter,
    //         gsl_vector_get (s->x, 0),
    //         gsl_vector_get (s->x, 1),
    //         gsl_vector_get (s->x, 2),
    //         gsl_vector_get (s->x, 3),
    //         s->f);

    } while (status == GSL_CONTINUE && iter < maxIter);

    printf("Potential energy after %i/%i iterations: %f \n", iter, maxIter, s->f);
    fprintf(paramFile,"Potential energy after %i/%i iterations: %f \n", iter, maxIter, s->f);

    //Copy new starting positions
    VecR pos;
    for(int particleIdx = 0; particleIdx < nParticles; particleIdx++){
        pos.x = gsl_vector_get(s->x,2*particleIdx); 
        pos.y = gsl_vector_get(s->x,2*particleIdx+1); 
        VWrapAll(pos); //Enforce periodic boundary conditions
        particles[particleIdx].r = pos;
    }

    gsl_multimin_fdfminimizer_free (s);
    gsl_vector_free (x);
};

void InitialiseInternalStates(){
    //Green particles
    for(int particleIdx = 0; particleIdx < nGreenParticles; particleIdx++){
        particles[particleIdx].D = greenD;
        particles[particleIdx].decayTimer = 0;
        particles[particleIdx].color = 1;
        particles[particleIdx].lastContact = -1;
        particles[particleIdx].contactTime = -1;
        particles[particleIdx].cilCooldown = -1;
    }
    //Red particles
    for(int particleIdx = nGreenParticles; particleIdx < nGreenParticles + nRedParticles; particleIdx++){
        particles[particleIdx].D = redD;
        particles[particleIdx].decayTimer = 0;
        particles[particleIdx].color = 0;
        particles[particleIdx].lastContact = -1;
        particles[particleIdx].contactTime = -1;
        particles[particleIdx].cilCooldown = -1;
    }
};

void InitialiseAngles(){
    //Set theta randomly in [0, 2 pi] and set decay time to 0
    real sum = 0;
    for(int particleIdx=0; particleIdx < nParticles; particleIdx++){
        particles[particleIdx].theta = (float) rand()/(float)(RAND_MAX/(2*M_PI));
        // sum += particles[particleIdx].theta;
    }
    // //set the average of theta to zero
    // for(int particleIdx=0; particleIdx < nParticles; particleIdx++){
    //     particles[particleIdx].theta -= sum/nParticles;
    // }
};

void MeasurePositions(real time){
    //Times
    fprintf(tracksFile, "%f", time);
    for(int particleIdx = 0; particleIdx<nParticles; particleIdx++){
        //Print color
        if(particles[particleIdx].color ==0){
            fprintf(tracksFile,", red");
        } else if (particles[particleIdx].color ==1){
            fprintf(tracksFile,", green");
        } else if (particles[particleIdx].color == 2){
            fprintf(tracksFile,", greenPlus");
        }
        //Print positions
        fprintf(tracksFile,", %f, %f",particles[particleIdx].r.x, \
                                      particles[particleIdx].r.y);
    }
    fprintf(tracksFile,"\n");
};

void MeasureVelocities(real time){
    //Times
    fprintf(velocityTracksFile, "%f", time);
    real angle;
    for(int particleIdx = 0; particleIdx<nParticles; particleIdx++){
        //Print the angle at which the particles went persistent
        angle = particles[particleIdx].theta;
        while(angle > M_PI){angle -=2*M_PI;}
        while(angle < -M_PI){angle += 2*M_PI;}
        //Print positions

        fprintf(velocityTracksFile,", %f", angle);
    }
    fprintf(velocityTracksFile,"\n");
};

void BuildNeighbourList(){
    VecR cellWidth;
    VecI cellIdx;
    int linearCellIdx;
    VDiv(cellWidth, region, cells);

     //Write linked list
    for(int n = nParticles; n < nParticles + VProd(cells); n++) cellList[n] = -1; //Reset list values
    for(int particleIdx = 0; particleIdx < nParticles; particleIdx++){
        VDiv(cellIdx,particles[particleIdx].r,cellWidth);
        linearCellIdx = VLinear(cellIdx, cells) + nParticles;//Linearise matrix index to vector index
        cellList[particleIdx] = cellList[linearCellIdx];
        cellList[linearCellIdx] = particleIdx;
        // Explaination: After this every cellIndex (cellList[nParticle:nParticle+nCells])
        // links to the first particle in that cell, that then links to the next particle,
        // until the particle chain finishes with -1
    }

    //Create Neighbourhood list
    VecR deltaR;
    real distance, forceMagnitude, springForce;
    VecI cellIdx1, cellIdx2;
    int linearCellIdx1, linearCellIdx2;
    VecI offset[] = {{0,0},{0,1},{-1,0},{-1,1},{1,1}}; // Define cell index offsets that need to be scanned (only half of neighbours to avoid double counting)
    int nOffsets = 5;
    real neighbourRadiusSq = (potentialRange+CELL_SIZE_EXTENSION)*(potentialRange+CELL_SIZE_EXTENSION); //<= cellWidth^2
    nNeighbourPairs = 0;
    //Go through all cells
    for(int cellYIdx = 0; cellYIdx < cells.y; cellYIdx++){
        for(int cellXIdx = 0; cellXIdx < cells.x; cellXIdx++){
            VSet(cellIdx1,cellXIdx,cellYIdx);
            //Go through this cell and its neighbouring cells
            for(int offsetIdx = 0; offsetIdx < nOffsets; offsetIdx++){
                VSAdd(cellIdx2, cellIdx1, 1, offset[offsetIdx]);
                VCellWrapAll(cellIdx2); //Periodic boundaries for cells
                linearCellIdx1 = VLinear(cellIdx1, cells) + nParticles;
                linearCellIdx2 = VLinear(cellIdx2, cells) + nParticles;
                //Go through all particles in the cells
                for(int pIdx1 = cellList[linearCellIdx1]; pIdx1 >=0; pIdx1 = cellList[pIdx1]){
                    for(int pIdx2 = cellList[linearCellIdx2]; pIdx2 >=0; pIdx2 = cellList[pIdx2]){
                        if(pIdx1!=pIdx2){//(linearCellIdx1<linearCellIdx2 || (linearCellIdx1==linearCellIdx2 && pIdx2 < pIdx1)){ //Avoid double counting
                            VSub(deltaR, particles[pIdx1].r,particles[pIdx2].r);
                            VWrapAllTang(deltaR); //Apply periodic boundary condition
                            if(VLenSq(deltaR) < neighbourRadiusSq){ //WARNING: Assumes square cells/ regions
                                assert(nNeighbourPairs <= MAX_NEIGHBOUR_PAIRS); // We can't exceed the size of the neighbour list
                                neighbourList[2*nNeighbourPairs] = pIdx1;
                                neighbourList[2*nNeighbourPairs+1] = pIdx2;
                                nNeighbourPairs++;
                            }
                        }
                    }
                }
            }
        }
    }
    // printf("Neighbour list:");
    // for(int n=0; n<nNeighbourPairs; n++){printf("(%d,%d), ", neighbourList[2*n], neighbourList[2*n+1]);}
    // printf("\n");
    // printf("nNeighbourPairs = %d\n", nNeighbourPairs);

}

void SetUpJob(){
    AllocArrays();
    InitialiseInternalStates();
    InitialisePositions();
    InitialiseAngles();
    MeasurePositions(0);
    #ifdef TRACK_VELOCITIES
    MeasureVelocities(0);
    #endif
    
}

//Begin: Support Functions for Single Step
real HarmonicForce(real distance){
    return -k/distance*(distance-1); //WARNING: this needs to be changed if we don't measure in units of potentialRange
}

real LennardJonesForce(real distance){
    //WARNING: needs to be changed if cellRadius!=1
    double invDistance = 1/(distance); 
    double invDistance2 = invDistance*invDistance;
    double invDistance6=invDistance2*invDistance2*invDistance2;
    return 12*k*invDistance6*(invDistance6-1)*invDistance2;
}

real GetAngle(VecR r){
    double theta;
    if(r.x>=0){
        theta = atan(r.y/r.x);
    } else {
        theta = atan(r.y/r.x)+M_PI ;
    }
    return theta;
}

void ChangeDirection(int pIdx1, int pIdx2, VecR deltaR){
    //Particles turn away from each other after contact
    
    real theta = GetAngle(deltaR);
    real deltaTheta1, deltaTheta2;


    #ifdef TURN_AROUND_VARIATION 
    //Randomise the turn-around directions to match the distributions seen in experiment

    double noise[2];
    xoshiro256ss_normal(noise, &rng);
    double varianceHom = 2.04692214673895; // = 117.276/360*2*pi
    double varianceHet = 1.6236798031303248; // = 93.03/360*2*pi
    // deltaTheta1 = theta + 2*(xoshire256ss_uniform(&rng)-0.5)*TURN_AROUND_VARIATION - particles[pIdx1].theta;
    // deltaTheta2 = theta + M_PI +2*(xoshire256ss_uniform(&rng)-0.5)*TURN_AROUND_VARIATION - particles[pIdx1].theta;
    if(    (particles[pIdx1].color==0 && particles[pIdx2].color==0) //both red
        || (particles[pIdx1].color!=0 && particles[pIdx2].color!=0))  {// both greeen
        // Homotypic contact
        noise[0] = noise[0]*sqrt(varianceHom);
        noise[1] = noise[1]*sqrt(varianceHom);
        //printf("Hom Angles %f, %f \n", noise[0]/2/M_PI*360,noise[1]/2/M_PI*36);
    } else {
        // Heterotypic contact
        noise[0] = noise[0]*sqrt(varianceHet);
        noise[1] = noise[1]*sqrt(varianceHet);
        //printf("Het Angles %f, %f \n", noise[0]/2/M_PI*360,noise[1]/2/M_PI*360);
    }
    
    deltaTheta1 = theta - particles[pIdx1].theta + noise[0];
    deltaTheta2 = theta + M_PI - particles[pIdx2].theta + noise[1];
    #else
    deltaTheta1 = theta - particles[pIdx1].theta;
    deltaTheta2 = theta + M_PI - particles[pIdx2].theta;
    #endif
    
    //Align velocity with the new direction based on how big turnAround is
    //NOte: Checks for cilCooldown; If CIL_COOLDOWN_DURATION is not set cilCooldown=-1 always
    #ifdef ONLY_GREEN_CIL
    if(particles[pIdx1].color!=0 && particles[pIdx1].cilCooldown<simulationTime){ particles[pIdx1].theta += deltaTheta1*turnAround;}
    if(particles[pIdx2].color!=0 && particles[pIdx2].cilCooldown<simulationTime){ particles[pIdx2].theta += deltaTheta2*turnAround;}
    #else
    if(particles[pIdx1].cilCooldown<=simulationTime){particles[pIdx1].theta += deltaTheta1*turnAround;}
    if(particles[pIdx2].cilCooldown<=simulationTime){particles[pIdx2].theta += deltaTheta2*turnAround;}
    #endif

    // Refresh CIL cooldown
    #ifdef CIL_COOLDOWN_DURATION
    particles[pIdx1].cilCooldown = simulationTime+CIL_COOLDOWN_DURATION;
    particles[pIdx2].cilCooldown = simulationTime+CIL_COOLDOWN_DURATION;
    #endif  

}

void AddContact(int pIdx1, int pIdx2){
    //Adds the contactIdx to the new particle, if it doesn't already have a contact.
    if((particles[pIdx1].lastContact == -1)&&(particles[pIdx2].lastContact==-1)){ //No partner already
        particles[pIdx1].lastContact = pIdx2;
        particles[pIdx1].contactTime = simulationTime;
        particles[pIdx2].lastContact = pIdx1;
        particles[pIdx2].contactTime = simulationTime;
        // printf("Contact index added: %d, %d, %d, %d, %f \n",particles[0].lastContact, particles[1].lastContact, particles[2].lastContact, particles[3].lastContact, simulationTime);
    }
    
}

void ComputeInteractions(){
    //Build Linked list
    VecR cellWidth;
    VecI cellIdx;
    int linearCellIdx;
    VDiv(cellWidth, region, cells);

    //Delete old force values
    for(int particleIdx = 0; particleIdx < nParticles; particleIdx++) VZero(particles[particleIdx].force);

    //Compute interactions:
    VecR deltaR;
    real distance, forceMagnitude, springForce;
    // real maxForce = 0;
    int pairIdx, pIdx1, pIdx2;
    for(pairIdx = 0; pairIdx < nNeighbourPairs; pairIdx++){
        pIdx1 = neighbourList[2*pairIdx];
        pIdx2 = neighbourList[2*pairIdx+1];
        VSub(deltaR, particles[pIdx1].r,particles[pIdx2].r);
        VWrapAllTang(deltaR); //Apply periodic boundary condition
        // printf("Distance: (%f,%f)",deltaR.x,deltaR.y);
        distance = sqrt(VLenSq(deltaR));
        if(distance < 1){ //Do particles touch? (Warning: assumes particle radius = 1)
            #ifdef DIFFERENTIAL_CIL
            //Add contacts index to both particles if they are of different type
            if ((particles[pIdx1].color==0 && particles[pIdx2].color!=0) || (particles[pIdx1].color!=0 && particles[pIdx2].color==0)){
                AddContact(pIdx1, pIdx2);
                nCollisions += 1;
            }
            #else
            AddContact(pIdx1, pIdx2);
            nCollisions += 1;
            #endif

            //Repulsive forces
            if(LennardJones==1){
                forceMagnitude = LennardJonesForce(distance);
            } else {
                forceMagnitude = HarmonicForce(distance);
            }
            VVSAdd(particles[pIdx1].force,forceMagnitude,deltaR);
            VVSAdd(particles[pIdx2].force,-forceMagnitude,deltaR);

        } else if(distance < potentialRange){ //Are particles within attraction range, but not touching?
            // Attractive forces
            if(LennardJones==1){
                forceMagnitude = LennardJonesForce(distance);
            } else {
                forceMagnitude = HarmonicForce(distance);
            }
            
            //Are these two red particles?
            if(particles[pIdx1].color==0 && particles[pIdx2].color==0){
                VVSAdd(particles[pIdx1].force,redRedAdhesionMult*forceMagnitude,deltaR);
                VVSAdd(particles[pIdx2].force,-redRedAdhesionMult*forceMagnitude,deltaR);
            } //Are these two green particles?
            else if((particles[pIdx1].color==1 || particles[pIdx1].color==2) && (particles[pIdx2].color==1 || particles[pIdx2].color==2)){
                VVSAdd(particles[pIdx1].force,greenGreenAdhesionMutl*forceMagnitude,deltaR);
                VVSAdd(particles[pIdx2].force,-greenGreenAdhesionMutl*forceMagnitude,deltaR);
            } // There are opposite-color particles
            else {
                VVSAdd(particles[pIdx1].force,redGreenAdhesionMult*forceMagnitude,deltaR);
                VVSAdd(particles[pIdx2].force,-redGreenAdhesionMult*forceMagnitude,deltaR);
            }
        }
        // if(fabs(forceMagnitude)>maxForce){
        //     maxForce = fabs(forceMagnitude);
        // }
        

    }
    // printf("Max force: %f, nCollisions: %d \n",maxForce, nCollisions);
    // fflush(stdout);
                            

    #ifdef STICKY_CONTACTS
    // Harmonic springs between particle pairs
    // Note: This needs to be done outside the cellIdx loop, because otherwise the springs "snap" as 
    // when the particles seperate further than the cell radius 
    for(int pIdx=0; pIdx < nParticles; pIdx++){ 
        if(particles[pIdx].lastContact!=-1){
            VSub(deltaR, particles[pIdx].r,particles[particles[pIdx].lastContact].r);
            VWrapAllTang(deltaR); //Apply periodic boundary condition
            distance = sqrt(VLenSq(deltaR));

            springForce = -10*k/distance*(distance-1); //Use stiff spring with 5*k
            VVSAdd(particles[pIdx].force,springForce,deltaR);
            #ifdef DEBUG
            int contactIdx = particles[pIdx].lastContact;
            assert(particles[contactIdx].lastContact==pIdx); //Check that contacts are reciprocal
            #endif
            
        }
    }
    #endif
    
}

void EulerMaruyamaR(){
    real rootStepDuration = sqrt(stepDuration); //ToDo: Avoid repeat calls
    real root2 = sqrt(2);
    //Updates the positions r of the particle based on the forces calculated in ComputeInteractions
    VecR direction;
    VecR displacement;
    real maxDisplacementSq=0;
    double noise[2];

    for(int particleIdx=0; particleIdx < nParticles; particleIdx++){
        VZero(displacement);
        //Self propulsion
        direction.x = cos(particles[particleIdx].theta); //ToDo: Can we get less calls to sin/ cos here?
        direction.y = sin(particles[particleIdx].theta); //WARNING: not 3d compatible
        VVSAdd(displacement,stepDuration*Pe,direction);

        //Forces
        VVSAdd(displacement,stepDuration,particles[particleIdx].force);

        //Noise (ToDo: Vectorise random number generation)
        xoshiro256ss_normal(noise, &rng);
        displacement.x += DIFFUSION_STRENGTH*rootStepDuration*root2*noise[0];
        displacement.y += DIFFUSION_STRENGTH*rootStepDuration*root2*noise[1];

        VVSAdd(particles[particleIdx].r,1,displacement);

        maxDisplacementSq = MAX(maxDisplacementSq, VLenSq(displacement));
    }

    maxTotalDisplacement += sqrt(maxDisplacementSq);
    if(maxTotalDisplacement >= 0.5 * CELL_SIZE_EXTENSION){// Can a new particle have moved within interaction distance?
        updateNeighbourList = 1;
    }

    #ifdef DEBUG
    if(maxDisplacementSq>MAX_STEP_DISPLACEMENT*MAX_STEP_DISPLACEMENT){
        double max = MAX_STEP_DISPLACEMENT;
        printf("Error: The displacment of %f was larger the the allowed maximum of %f\n",sqrt(maxDisplacementSq),max);
        fflush(stdout);
        assert(0);  //Exceeds maximum displacement
    }
    #endif
}

void EulerMaruyamaTheta(){
    real rootStepDuration = sqrt(stepDuration); //ToDo: Avoid repeat calls
    real root2 = sqrt(2);
    //Updates the angles of the particles
    double noise[2];
    int noiseCount = 0; //Takes care of having to generate random numbers in pairs
    for(int particleIdx=0; particleIdx < nParticles; particleIdx++){
        noiseCount = noiseCount % 2;
        if (noiseCount == 0) xoshiro256ss_normal(noise, &rng);
        particles[particleIdx].theta += rootStepDuration*root2*particles[particleIdx].D*noise[noiseCount];
        noiseCount++;
    }
}

void EnforcePeriodicBoundaries(){
    double xMax = 2*region.x;
    double yMax = 2*region.y;
    for(int particleIdx=0; particleIdx < nParticles; particleIdx++){
        //Check for large jumps
        assert(abs(particles[particleIdx].r.x)<xMax); //Step is larger than the entire box, reduce stepDuration!
        assert(abs(particles[particleIdx].r.y)<yMax);
        //Position
        VWrapAll(particles[particleIdx].r);
    }
}

void UpdateInternalStates(){
    VecR deltaR;
    real distance;
    for(int particleIdx=0; particleIdx < nParticles; particleIdx++){
        // Forward decay timer
        if (particles[particleIdx].decayTimer>0){
            particles[particleIdx].decayTimer -= stepDuration;
        }
        // Update persistence
        if(particles[particleIdx].color==2){
            if(particles[particleIdx].decayTimer <= 0){
                particles[particleIdx].color = 1; 
                particles[particleIdx].D = greenD;
                particles[particleIdx].decayTimer = 0;
            }
        }
        #ifdef NON_DIFFERENTIAL_PERSISTENCE
        // Update persistence for red particles as well
        if(particles[particleIdx].color==0){
            if(particles[particleIdx].decayTimer <= 0){
                particles[particleIdx].D = redD;
                particles[particleIdx].decayTimer = 0;
            }
        }
        #endif

        // Have any contacts matured?
        if ((particles[particleIdx].lastContact!=-1) && (simulationTime-particles[particleIdx].contactTime>CIL_DELAY)){
            int contactIdx = particles[particleIdx].lastContact;
            // printf("Contacts matured: %d, %d \n", particleIdx, contactIdx);
            VSub(deltaR, particles[particleIdx].r,particles[contactIdx].r);
            VWrapAllTang(deltaR);
            
            // Perform CIL
            ChangeDirection(particleIdx,contactIdx,deltaR);
            

            // Change persistence
            if (particles[particleIdx].color == 1 && particles[contactIdx].color==0){
                particles[particleIdx].color = 2; 
                particles[particleIdx].D = greenPersistentD;
                particles[particleIdx].decayTimer = tau;
            }
            if (particles[contactIdx].color == 1 && particles[particleIdx].color==0){
                particles[contactIdx].color = 2; 
                particles[contactIdx].D = greenPersistentD;
                particles[contactIdx].decayTimer = tau;     
            }

            
            #ifdef NON_DIFFERENTIAL_PERSISTENCE
            // Red particles also get increased persistence
            if (particles[particleIdx].color == 0 && particles[contactIdx].color!=0){
                particles[particleIdx].D = greenPersistentD;
                particles[particleIdx].decayTimer = tau;
            }
            if (particles[contactIdx].color == 0 && particles[particleIdx].color!=0){
                particles[contactIdx].D = greenPersistentD;
                particles[contactIdx].decayTimer = tau; 
            }
            #endif

            
            
            #ifdef MEASURE_COLLISION_ANGLE
            //Print the angle at which the particles went persistent
            real angle = particles[particleIdx].theta - particles[particles[particleIdx].lastContact].theta;
            while(angle > M_PI){angle -=2*M_PI;}
            while(angle < -M_PI){angle += 2*M_PI;}
            collisionAngle += fabs(angle*360/2/M_PI);
            collisionDuration +=(simulationTime-particles[particleIdx].contactTime);
            #endif //MEASURE_COLLISION_ANGLE

            // Clean up
            particles[particleIdx].lastContact = -1;
            particles[particleIdx].contactTime = -1;
            particles[contactIdx].lastContact = -1;
            particles[contactIdx].contactTime = -1;
        }
    }
}

//End: Support Functions for Single Step
void SingleStep (int stepIdx){
    if(updateNeighbourList==1){
        updateNeighbourList = 0;
        maxTotalDisplacement = 0;
        BuildNeighbourList();
    }
    ComputeInteractions();
    EulerMaruyamaR();
    EulerMaruyamaTheta();
    EnforcePeriodicBoundaries();
    UpdateInternalStates();
    int measurementSteps = round(measurementInterval/stepDuration); 
    if(((stepIdx > skipSteps) && (stepIdx % measurementSteps == 0)) ){ //Time for measurement? 
        MeasurePositions(stepIdx*stepDuration);
        #ifdef TRACK_VELOCITIES
        MeasureVelocities(stepIdx*stepDuration);
        #endif
    } else if (stepIdx==(stepLimit-1)){ //Save final step (initial position is measured by SetUpJob()
        MeasurePositions(stepIdx*stepDuration);
        #ifdef TRACK_VELOCITIES
        MeasureVelocities(stepIdx*stepDuration);
        #endif
    }
    fflush(stdout); //Flushes stdout
    fflush(paramFile); //Flushes the buffer
}

void cleanup(){
    //Close files
    fclose(tracksFile);
    fclose(paramFile);

    //Free memory
    free(particles);
    free(cellList);
    
}

