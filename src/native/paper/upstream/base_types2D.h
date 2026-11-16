#ifndef base_types
#define base_types
#include "agent_simulation_config.h"
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  Data types ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

typedef double real;

//2-dimensional vector
typedef struct { 
    real x,y;
} VecR;

typedef struct { 
    int x,y;
} VecI;

//active brownian motion particle
typedef struct {
    VecR r; // position of particle
    real theta; // angle of velocity direction
    VecR force; //force displacing the particle this step
    real D; // Rotational diffusion constant
    real decayTimer; //Time it takes to go back to less persistent state
    int color; //red = 0, green =1 , persistent green = 2
    int lastContact; //tracks index of the last particle this one has been in contact with
    real contactTime; //Tracks the time of collision
    real cilCooldown; // While coming down no CIL is performed (persistence change still possible)
} particle ;

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  General  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#define MIN(i, j) (((i) < (j)) ? (i) : (j))
#define MAX(i, j) (((i) > (j)) ? (i) : (j))

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  Linear algebra ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


#define VDot(v1, v2) \
     ((v1).x * (v2).x + (v1).y * (v2).y)
#define VSAdd(v1, v2, s3, v3) \
    ((v1).x = (v2).x + (s3) * (v3).x, \
    (v1).y = (v2).y + (s3) * (v3).y)
#define VSub(v1, v2, v3) \
    (v1).x = (v2).x - (v3).x, \
    (v1).y = (v2).y - (v3).y
#define VSet(v, sx, sy) \
    (v).x = sx, \
    (v).y = sy 
#define VSetAll(v, s)       VSet (v, s, s)
#define VZero(v)            VSetAll (v, 0)
#define VVSAdd(v1, s2, v2)  VSAdd (v1, v1, s2, v2)
#define VLenSq(v)           VDot (v, v)

#define VMul(v1, v2, v3) \
    (v1).x = (v2).x * (v3).x, \
    (v1).y = (v2).y * (v3).y

#define VScalMult(v1,s2,v2) \
    (v1).x = s2*(v2).x, \
    (v1).y = s2*(v2).y

#define VProd(v1) ((v1).x * (v1).y)

#define VDiv(v1, v2, v3) \
    (v1).x = (v2).x / (v3).x, \
    (v1).y = (v2).y /(v3).y


#define VLinear(p,s)  \
    (p).y*(s).x+(p).x
//Linearises a pair of row and column indices p into a single index based on the 
// row length s.x





//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  Periodic boundaries ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//Wraps tangent vectors, i.e. those that leave the top right quadrant
//assumes (rectangular) boundary is given as VecR region
#define VWrapTang(v, t) \
    while (v.t >= 0.5 * region.t) v.t -= region.t; \
    while (v.t < -0.5 * region.t) v.t += region.t;
#define VWrapAllTang(v) \
    {VWrapTang (v, x); \
    VWrapTang (v, y);}
//Wraps position vectors, that only live in the top right quadrant
#define VWrap(v,t)\
    while(v.t >= region.t) v.t -= region.t; \
    while(v.t < 0) v.t += region.t;
#define VWrapAll(v) \
    {VWrap(v,x); \
     VWrap(v,y);}

#define VCellWrap(cellIdx,t)    \
    if (cellIdx.t >= cells.t) { \
        cellIdx.t = 0;          \
    } else if (cellIdx.t < 0) { \
        cellIdx.t = cells.t - 1;\
    }

//Requires cells to be defined there
#define VCellWrapAll(cellIdx)   \
    {VCellWrap (cellIdx,x);     \
    VCellWrap (cellIdx,y);      \
    }


#endif