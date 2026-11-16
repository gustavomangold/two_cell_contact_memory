/* Boundary-only additions to the paper engine. Pair forces, noise and contact
   updates stay in the inherited functions; no additional random draws. */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <gsl/gsl_vector.h>
#include "upstream/global.h"

static double wall_stiffness;
void ArenaBaseSetParameters(int argc, char **argv);
void ArenaParticleInteractions(void);
double ArenaParticlePotential(const gsl_vector *v, void *params);
void ArenaParticleGradient(const gsl_vector *v, void *params, gsl_vector *df);

void SetParameters(int argc, char **argv) {
    ArenaBaseSetParameters(argc, argv);
    const char *value = getenv("PAPER_WALL_STIFFNESS");
    wall_stiffness = value ? strtod(value, NULL) : 30000.;
    if (!isfinite(wall_stiffness) || wall_stiffness <= 0 || k <= 0) exit(2);
    printf("Circular arena: diameter=%g, wall stiffness=%g; no PBC\n", region.x, wall_stiffness);
}

/* L is the outer diameter; disk centers touch the wall at radius (L-1)/2. */
static double wall_overlap(double x, double y, double *nx, double *ny) {
    x -= region.x / 2.;
    y -= region.y / 2.;
    double radius = hypot(x, y);
    *nx = radius > 0 ? x / radius : 0;
    *ny = radius > 0 ? y / radius : 0;
    return fmax(0., radius - (region.x - 1.) / 2.);
}

void ComputeInteractions(void) {
    ArenaParticleInteractions();
    for (int i = 0; i < nParticles; i++) {
        double nx, ny;
        double overlap = wall_overlap(particles[i].r.x, particles[i].r.y, &nx, &ny);
        particles[i].force.x -= wall_stiffness * overlap * nx;
        particles[i].force.y -= wall_stiffness * overlap * ny;
    }
}

/* Keep the same GSL overlap minimizer, adding the matching wall potential. */
double fHarmonicPotential(const gsl_vector *v, void *params) {
    double energy = ArenaParticlePotential(v, params);
    for (int i = 0; i < nParticles; i++) {
        double nx, ny;
        double overlap = wall_overlap(gsl_vector_get(v, 2*i), gsl_vector_get(v, 2*i+1), &nx, &ny);
        energy += wall_stiffness / k * overlap * overlap;
    }
    return energy;
}

void dfHarmonicPotential(const gsl_vector *v, void *params, gsl_vector *df) {
    ArenaParticleGradient(v, params, df);
    for (int i = 0; i < nParticles; i++) {
        double nx, ny;
        double overlap = wall_overlap(gsl_vector_get(v, 2*i), gsl_vector_get(v, 2*i+1), &nx, &ny);
        double gradient = 2 * wall_stiffness / k * overlap;
        gsl_vector_set(df, 2*i, gsl_vector_get(df, 2*i) + gradient * nx);
        gsl_vector_set(df, 2*i+1, gsl_vector_get(df, 2*i+1) + gradient * ny);
    }
}
