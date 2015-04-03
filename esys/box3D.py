# import the division module for compatibility between Python 2 and Python 3
from __future__ import division
# import the appropriate ESyS-Particle modules:
from esys.lsm import *
from esys.lsm.util import Vec3, BoundingBox
from esys.lsm.geometry import CubicBlock
from hist_and_snaps import hist_and_snaps
from numpy.random import randn

# instantiate a simulation object
# and initialise the neighbour search algorithm:
sim = LsmMpi(numWorkerProcesses=1, mpiDimList=[1, 1, 1])
sim.initNeighbourSearch(
    particleType="NRotSphere",
    gridSpacing=2.5,
    verletDist=0.5
)

# set the number of timesteps and timestep increment:
sim.setNumTimeSteps(3000)
sim.setTimeStepSize(0.001)
L = 20
Ld = L + 1
# specify the spatial domain for the simulation:
domain = BoundingBox(Vec3(-Ld, -Ld, -Ld), Vec3(Ld, Ld, Ld))
sim.setSpatialDomain(
    bBox=domain, circDimList=[True, False, False])
# add a cube of particles to the domain:
cube = CubicBlock(dimCount=[20, 20, 20], radius=0.1)
sim.createParticles(cube)

for n in range(sim.getNumParticles()):
    sim.setParticleVelocity(id=n, Velocity=Vec3(randn(), randn(), randn()))

# specify the type of interactions between colliding particles:
sim.createInteractionGroup(
    NRotElasticPrms(
        name="elastic_repulsion",
        normalK=10000.0,
        scaling=True
    )
)

walls = ["bottom", "top", "right", "left", "back", "forward"]
origins = [
    Vec3(0, 1, 0), Vec3(0, -1, 0),
    Vec3(-1, 0, 0), Vec3(1, 0, 0),
    Vec3(0, 0, 1), Vec3(0, 0, -1)]
posns = [
    Vec3(0, -L, 0), Vec3(0, L, 0),
    Vec3(L, 0, 0), Vec3(-L, 0, 0),
    Vec3(0, 0, -L), Vec3(0, 0, L)]

for n, w in enumerate(walls):
    sim.createWall(
        name=w,
        posn=posns[n],
        normal=origins[n]
    )
    sim.createInteractionGroup(
        NRotElasticWallPrms(
            name=w + "repel",
            wallName=w,
            normalK=10000.0
        )
    )

## add local viscosity to simulate air resistance:
#sim.createInteractionGroup(
    #LinDampingPrms(
        #name="linDamping",
        #viscosity=0.001,
        #maxIterations=100
    #)
#)



# add Runnable post processing:
povcam = hist_and_snaps(sim=sim, interval=100, resolution=100)
sim.addPreTimeStepRunnable(povcam)

sim.run()
