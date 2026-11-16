#ifndef global_variables
#define global_variables


#include "base_types2D.h"
#include "agent_simulation_config.h"

//Simulation Parameters
extern int stepLimit;
extern real stepDuration;
extern int skipSteps; //only after these steps will tracks be recorded (saves space)
extern real measurementInterval; //at the end of each interval we measure all observables
extern int nParticles;
extern int nGreenParticles;
extern int nRedParticles;
extern VecR region;  //Size of the simulation region
extern real redD, greenD, greenPersistentD; //rotational diffusion constants 
extern real k;  //potential strength
extern real tau; //decay time of the persistent state
extern real Pe; //self-propulsion velocity
extern real potentialRange; //interaction distance, sigma>1s creates stickyness
extern int nMeasurements;
extern VecI initialCellsGreen; //2D vector containing the number of particles in the x and y direction
extern VecI initialCellsRed;
extern int LennardJones; //bool that says whether if we use Lennard Jones potential instead of harmonic interactions 
extern real turnAround; // Bool that flags wether the particles move away from each other after colision
extern real redRedAdhesionMult; // Multiplies the adhesive component of the force between these particle types
extern real greenGreenAdhesionMutl;
extern real redGreenAdhesionMult;

// Data structures
extern uint64_t seed;
extern struct xoshiro256ss_state rng;
extern particle * particles; //First half is green, second half red
extern FILE * paramFile;
extern FILE * tracksFile;
#ifdef TRACK_VELOCITIES
extern FILE * velocityTracksFile;
#endif
extern VecI cells; //Number of cells dividing the space in each coordinate for interaction computation
extern int * cellList; //"Linked list" of length nParticle+nCells, first entries contain 
//indices of next particle in the cell, the last entries contain indices to first 
//particle in that cell
extern int * neighbourList; // List of int pairs that denote the to particles that are neighbours
extern int updateNeighbourList; //Boolean, when set it triggers a recalculation of the neighbour list.
extern int nNeighbourPairs; // Current length of the neighbour list
// Extra observables
extern real simulationTime; //So that any function can access the current time of the simulation
#ifdef MEASURE_COLLISION_ANGLE
extern real collisionAngle;
extern unsigned long int nCollisions;
extern real collisionDuration;
#endif //MEASURE_COLLISION_ANGLE
extern real maxTotalDisplacement; //Keeps track of the maximum any particle has moved, triggers updates of neighbour list


#endif