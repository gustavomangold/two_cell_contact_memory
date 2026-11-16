"""Prepare a temporary circular-boundary build of the paper C engine."""

import hashlib
from pathlib import Path
import re
import shutil

from paper_engine import prepare_source


def arena_source_hash():
    digest = hashlib.sha256()
    for suffix in [".py", ".c"]:
        digest.update(Path(__file__).with_suffix(suffix).read_bytes())
    return digest.hexdigest()


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"Unexpected upstream arena patch: {old}")
    return text.replace(old, new)


def prepare_arena_source(destination):
    prepare_source(destination)
    shutil.copy2(Path(__file__).with_suffix(".c"), destination / "arena.c")
    header = destination / "upstream/base_types2D.h"
    text = header.read_text()
    end = text.rfind("#endif")
    header.write_text(text[:end] + "\n#undef VWrapAllTang\n#define VWrapAllTang(v) ((void)0)\n"
                      "#undef VWrapAll\n#define VWrapAll(v) ((void)0)\n" + text[end:])
    code = destination / "upstream/cell_simulation.c"
    text = code.read_text()
    for old, new in [
        ("void SetParameters(", "void ArenaBaseSetParameters("),
        ("void ComputeInteractions(){", "void ArenaParticleInteractions(){"),
        ("double fHarmonicPotential(", "double ArenaParticlePotential("),
        ("void dfHarmonicPotential(", "void ArenaParticleGradient("),
        ("sqrt(nParticles*0.785398/areaFraction)", "sqrt(nParticles/areaFraction)"),
        ("        particles[particleIdx].r.x = region.x*uniform1;\n        particles[particleIdx].r.y = region.y*uniform2;",
         "        particles[particleIdx].r.x = region.x/2 + (region.x/2-.5)*sqrt(uniform1)*cos(2*M_PI*uniform2);\n"
         "        particles[particleIdx].r.y = region.y/2 + (region.y/2-.5)*sqrt(uniform1)*sin(2*M_PI*uniform2);"),
        ("VCellWrapAll(cellIdx2);", "if (cellIdx2.x < 0 || cellIdx2.x >= cells.x || cellIdx2.y < 0 || cellIdx2.y >= cells.y) continue;"),
        ("VDiv(cellIdx,particles[particleIdx].r,cellWidth);",
         "VDiv(cellIdx,particles[particleIdx].r,cellWidth);\n"
         "        cellIdx.x = MAX(0, MIN(cells.x-1, cellIdx.x));\n"
         "        cellIdx.y = MAX(0, MIN(cells.y-1, cellIdx.y));"),
    ]:
        text = replace_once(text, old, new)
    # Wall forces act during the position update. There is no wrap or projection.
    text, count = re.subn(r"void EnforcePeriodicBoundaries\(\)\{.*?\n\}",
                         "void EnforcePeriodicBoundaries(){}", text, flags=re.DOTALL)
    if count != 1:
        raise ValueError("Unexpected upstream periodic-boundary function")
    code.write_text(text)
