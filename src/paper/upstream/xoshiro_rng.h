#ifndef xoshiro_rng
#define xoshiro_rng

#include <stdint.h>
#include <math.h>

//from Xorshift wikipedia - xoshiro256**, fast, passes bigcrush, period of 2^256 − 1	
// See Sebastiano's site for referece (https://prng.di.unimi.it/)
struct xoshiro256ss_state {
	uint64_t s[4];
};

uint64_t rol64(uint64_t x, int k);

uint64_t xoshiro256ss(struct xoshiro256ss_state *state);


//Splitmix is only here to create initial state from integer seed (xoshiro has 4-integer state)

struct splitmix64_state {
	uint64_t s;
};

uint64_t splitmix64(struct splitmix64_state *state);

struct xoshiro256ss_state xoshiro256ss_init(uint64_t seed);


//Convert output to float uniform on [0,1)
double xoshire256ss_uniform(struct xoshiro256ss_state *state);

//Get a pair of normal distributed RVs via Box-Mueller algorithm
void xoshiro256ss_normal(double * Z, struct xoshiro256ss_state * state);


#endif