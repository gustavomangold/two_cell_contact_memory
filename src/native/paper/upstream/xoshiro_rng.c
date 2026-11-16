#ifndef xoshiro_rng
#define xoshiro_rng

#include <stdint.h>
#include <math.h>

//from Xorshift wikipedia - xoshiro256**, fast, passes bigcrush, period of 2^256 − 1	
// See Sebastiano's site for referece (https://prng.di.unimi.it/)
struct xoshiro256ss_state {
	uint64_t s[4];
};

uint64_t rol64(uint64_t x, int k)
{
	return (x << k) | (x >> (64 - k));
}

uint64_t xoshiro256ss(struct xoshiro256ss_state *state)
{
	uint64_t *s = state->s;
	uint64_t const result = rol64(s[1] * 5, 7) * 9;
	uint64_t const t = s[1] << 17;

	s[2] ^= s[0];
	s[3] ^= s[1];
	s[1] ^= s[2];
	s[0] ^= s[3];

	s[2] ^= t;
	s[3] = rol64(s[3], 45);

	return result;
}


//Splitmix is only here to create initial state from integer seed (xoshiro has 4-integer state)

struct splitmix64_state {
	uint64_t s;
};

uint64_t splitmix64(struct splitmix64_state *state) {
	uint64_t result = (state->s += 0x9E3779B97f4A7C15);
	result = (result ^ (result >> 30)) * 0xBF58476D1CE4E5B9;
	result = (result ^ (result >> 27)) * 0x94D049BB133111EB;
	return result ^ (result >> 31);
}

struct xoshiro256ss_state xoshiro256ss_init(uint64_t seed) {
	struct splitmix64_state smstate = {seed};
	struct xoshiro256ss_state result = {0};

	uint64_t tmp = splitmix64(&smstate);
	result.s[0] = (uint64_t)tmp;
	result.s[1] = (uint64_t)(tmp >> 32);

	tmp = splitmix64(&smstate);
	result.s[2] = (uint64_t)tmp;
	result.s[3] = (uint64_t)(tmp >> 32);

    //first value always really small, so skip:
    tmp = xoshiro256ss(&result);
	return result;
}


//Convert output to float uniform on [0,1)
double xoshire256ss_uniform(struct xoshiro256ss_state *state){
    uint64_t i = xoshiro256ss(state);
    double f = (i >> 11) * 0x1.0p-53; 
    //This conversion guarantees that all dyadic rationals of the form k / 2−53 will be equally likely
    return f;
};

//Get a pair of normal distributed RVs via Box-Mueller algorithm
void xoshiro256ss_normal(double * Z, struct xoshiro256ss_state * state){
    //Expects array Z with at least too entries
    double U1 = xoshire256ss_uniform(state);
    double U2 = xoshire256ss_uniform(state);
    Z[0] = sqrt(-2*log(U1))*cos(2*M_PI*U2);
    Z[1] = sqrt(-2*log(U1))*sin(2*M_PI*U2);
    //return Z;
}


#endif