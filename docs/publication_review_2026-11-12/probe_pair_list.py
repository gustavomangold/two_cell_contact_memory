"""Reproduce the force audit in a temporary build; never modify the engine."""
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path('/home/gustavo/Documents/doutorado/repositorios/two_cell_contact_memory')
PROBE = r'''
#define main original_main
#include "upstream/main.c"
#undef main
void BuildNeighbourList(void);

int main(void) {
    nParticles = 2;
    nGreenParticles = 2;
    nRedParticles = 0;
    region.x = region.y = 8.;
    cells.x = cells.y = 4;
    potentialRange = 1.;
    k = 1000.;
    LennardJones = 0;
    AllocArrays();
    for (int i = 0; i < 2; i++) {
        particles[i].color = 1;
        particles[i].lastContact = -1;
    }
    for (int translated = 0; translated < 2; translated++) {
        particles[0].r.x = .1 + translated * 1.8;
        particles[1].r.x = .3 + translated * 1.8;
        particles[0].r.y = particles[1].r.y = .1;
        BuildNeighbourList();
        ComputeInteractions();
        printf("AUDIT %d %d %.17g\n", translated, nNeighbourPairs, particles[0].force.x);
    }
    return 0;
}
'''

with tempfile.TemporaryDirectory(prefix='contact_memory_force_probe_') as directory:
    work = Path(directory)
    shutil.copytree(ROOT / 'src/paper/upstream', work / 'upstream')
    (work / 'probe.c').write_text(PROBE)
    subprocess.run(['cc', '-O2', '-std=gnu11', 'probe.c',
                    'upstream/cell_simulation.c', 'upstream/xoshiro_rng.c',
                    '-lgsl', '-lgslcblas', '-lm', '-o', 'probe'], cwd=work, check=True)
    result = subprocess.run([str(work / 'probe')], cwd=work, check=True,
                            capture_output=True, text=True)
    for line in result.stdout.splitlines():
        if line.startswith('AUDIT '):
            print(line)
print('Same separation 0.2; translated together across a bin edge. No simulation files changed.')
