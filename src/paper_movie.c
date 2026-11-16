/* Appended to the shared driver helpers in a temporary GIF-only build. */
static VecR *movie_previous, *movie_unwrapped;

/* Called immediately before the original periodic wrap. Reads positions only. */
void MovieObserveBeforeWrap(void) {
    for (int i = 0; i < nParticles; i++) {
        movie_unwrapped[i].x += particles[i].r.x - movie_previous[i].x;
        movie_unwrapped[i].y += particles[i].r.y - movie_previous[i].y;
    }
}

static void movie_frame(FILE *file, int step) {
    double time = step * stepDuration;
    if (fwrite(&time, sizeof(double), 1, file) != 1) exit(2);
    for (int i = 0; i < nParticles; i++) {
        double values[] = {particles[i].r.x, particles[i].r.y, particles[i].theta,
                           particles[i].decayTimer, particles[i].color,
                           movie_unwrapped[i].x, movie_unwrapped[i].y};
        if (fwrite(values, sizeof(double), 7, file) != 7) exit(2);
    }
}

int main(int argc, char **argv) {
    if (!getenv("PAPER_SEED")) return 2;
    SetParameters(argc, argv);
    if (cells.x < 1) cells.x = 1;
    if (cells.y < 1) cells.y = 1;
    fclose(tracksFile); tracksFile = fopen("/dev/null", "w");
    #ifdef TRACK_VELOCITIES
    fclose(velocityTracksFile); velocityTracksFile = fopen("/dev/null", "w");
    #endif
    SetUpJob();
    movie_previous = calloc(nParticles, sizeof(VecR));
    movie_unwrapped = calloc(nParticles, sizeof(VecR));
    if (!movie_previous || !movie_unwrapped) return 2;
    for (int i = 0; i < nParticles; i++) movie_unwrapped[i] = particles[i].r;
    FILE *plan = fopen("frames.txt", "r"), *movie = fopen("movie.bin", "wb");
    if (!plan || !movie) return 2;
    int frames;
    if (fscanf(plan, "%d", &frames) != 1 || frames < 2) return 2;
    int *steps = calloc(frames, sizeof(int));
    if (!steps) return 2;
    for (int i = 0; i < frames; i++) if (fscanf(plan, "%d", &steps[i]) != 1) return 2;
    fclose(plan);
    int frame = 0;
    if (steps[0] == 0) { movie_frame(movie, 0); frame++; }
    for (int step = 0; step <= stepLimit; step++) {
        simulationTime = step * stepDuration;
        for (int i = 0; i < nParticles; i++) movie_previous[i] = particles[i].r;
        SingleStep(step);
        if (frame < frames && step + 1 == steps[frame]) {
            movie_frame(movie, step + 1); frame++;
        }
        if (step % (stepLimit / 20) == 0) { printf("step %d/%d\n", step, stepLimit); fflush(stdout); }
    }
    fclose(movie);
    if (frame != frames) return 2;
    FILE *state = fopen("rng.bin", "wb");
    if (!state || fwrite(rng.s, sizeof(uint64_t), 4, state) != 4) return 2;
    fclose(state);
    free(steps); free(movie_previous); free(movie_unwrapped);
    return 0;
}
