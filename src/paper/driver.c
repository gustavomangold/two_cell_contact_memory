/* Compact measurements around the unmodified upstream dynamics. */
#define main upstream_main
#include "upstream/main.c"
#undef main

void BuildNeighbourList(void);

static void write_particle(FILE *file, int i, int initial) {
    double values[] = {particles[i].r.x, particles[i].r.y, particles[i].theta,
                       particles[i].decayTimer, particles[i].color};
    if (fwrite(values, sizeof(double), initial ? 3 : 5, file) != (initial ? 3 : 5)) exit(2);
}

int main(int argc, char **argv) {
    if (!getenv("PAPER_SEED")) { fprintf(stderr, "PAPER_SEED is required\n"); return 2; }
    SetParameters(argc, argv);
    /* Upstream still makes every measurement call; suppress raw trajectory I/O. */
    fclose(tracksFile);
    tracksFile = fopen("/dev/null", "w");
    #ifdef TRACK_VELOCITIES
    fclose(velocityTracksFile);
    velocityTracksFile = fopen("/dev/null", "w");
    #endif
    if (!tracksFile) return 2;
    SetUpJob();

    FILE *edgesFile = fopen("edges.txt", "r"), *result = fopen("result.bin", "wb");
    if (!edgesFile || !result) return 2;
    int windows;
    if (fscanf(edgesFile, "%d", &windows) != 1 || windows < 1) return 2;
    int *edges = calloc(windows + 1, sizeof(int));
    double *sums = calloc(3 * windows, sizeof(double));
    unsigned char *contact = calloc(nParticles, 1);
    if (!edges || !sums || !contact) return 2;
    for (int i = 0; i <= windows; i++) if (fscanf(edgesFile, "%d", &edges[i]) != 1) return 2;
    fclose(edgesFile);
    if (edges[0] != 0 || edges[windows] != stepLimit + 1) return 2;
    for (int i = 0; i < nParticles; i++) write_particle(result, i, 1);

    int window = 0;
    /* Preserve the original inclusive loop: stepIdx=0,...,stepLimit. */
    for (int stepIdx = 0; stepIdx <= stepLimit; stepIdx++) {
        simulationTime = stepIdx * stepDuration;
        if (stepIdx == edges[window + 1]) window++;
        /* Same prelude as SingleStep; moved forward only to observe old positions.
           No extra RNG draws or force/state updates. SingleStep sees flag=0. */
        if (updateNeighbourList == 1) {
            updateNeighbourList = 0;
            maxTotalDisplacement = 0;
            BuildNeighbourList();
        }
        memset(contact, 0, nParticles);
        for (int pair = 0; pair < nNeighbourPairs; pair++) {
            int i = neighbourList[2 * pair], j = neighbourList[2 * pair + 1];
            if ((particles[i].color == 0) == (particles[j].color == 0)) continue;
            VecR delta;
            VSub(delta, particles[i].r, particles[j].r);
            VWrapAllTang(delta);
            if (VLenSq(delta) < 1.) contact[i] = contact[j] = 1;
        }
        for (int i = 0; i < nParticles; i++) {
            sums[3 * window + (i < nGreenParticles ? 0 : 1)] += contact[i];
            if (i < nGreenParticles && particles[i].color == 2) sums[3 * window + 2]++;
        }
        SingleStep(stepIdx);
        if (stepIdx % (stepLimit / 20) == 0) { printf("step %d/%d\n", stepIdx, stepLimit); fflush(stdout); }
    }
    for (int i = 0; i < nParticles; i++) {
        if (!isfinite(particles[i].r.x) || !isfinite(particles[i].r.y) || !isfinite(particles[i].theta)) {
            fprintf(stderr, "Nonfinite particle state\n"); return 2;
        }
        write_particle(result, i, 0);
    }
    if (fwrite(sums, sizeof(double), 3 * windows, result) != 3 * windows) return 2;
    if (fwrite(rng.s, sizeof(uint64_t), 4, result) != 4) return 2;
    fclose(result);
    fclose(tracksFile);
    #ifdef TRACK_VELOCITIES
    fclose(velocityTracksFile);
    #endif
    fclose(paramFile);
    free(edges); free(sums); free(contact);
    return 0;
}
