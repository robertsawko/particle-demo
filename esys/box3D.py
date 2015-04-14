# import the division module for compatibility between Python 2 and Python 3
from __future__ import division
# import the appropriate ESyS-Particle modules:
from esys.lsm import *
from esys.lsm.util import Vec3, BoundingBox
from esys.lsm.geometry import CubicBlock
from my_post_processing import bulk_parameters, velocity_and_position_output, POVsnaps
from numpy.random import randn
from numpy.random import rand
from numpy.random import seed
from numpy import sqrt


def uniform(L=10):
    return Vec3(
        2*L*rand() - L,
        2*L*rand() - L,
        2*L*rand() - L,
        )

# instantiate a simulation object
# and initialise the neighbour search algorithm:
sim = LsmMpi(numWorkerProcesses=4, mpiDimList=[2, 2, 1])
sim.initNeighbourSearch(
    particleType="NRotSphere",
    gridSpacing=0.2,
    verletDist=0.1
)

# set the number of timesteps and timestep increment:
sim.setNumTimeSteps(5000000)
sim.setTimeStepSize(0.0001)
L = 10
Ld = L + 1
kT = 0.1
a = sqrt(kT)

# specify the spatial domain for the simulation:
domain = BoundingBox(Vec3(-Ld, -Ld, -Ld), Vec3(Ld, Ld, Ld))
sim.setSpatialDomain(
    bBox=domain, circDimList=[False, False, False])
# add a cube of particles to the domain:
#cube = CubicBlock(dimCount=[20, 20, 20], radius=0.05)
#sim.createParticles(cube)
N = 100
seed(1234)
for n in range(N):
    p = NRotSphere(id=n, posn=uniform(), radius=0.05, mass=1.0)
    sim.createParticle(p)

for n in range(sim.getNumParticles()):
    sim.setParticleVelocity(
        id=n,
        Velocity=Vec3(a*randn(), a*randn(), a*randn()))

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
# sim.createInteractionGroup(
    # LinDampingPrms(
        # name="linDamping",
        # viscosity=0.001,
        # maxIterations=100
    # )
# )

# add Runnable post processing:
povcam = POVsnaps(sim=sim, interval=1000)
sim.addPreTimeStepRunnable(povcam)
#velpdf = velocity_pdfs(sim=sim, interval=1000)
#sim.addPreTimeStepRunnable(velpdf)
bulk = bulk_parameters(sim=sim, interval=1000)
sim.addPreTimeStepRunnable(bulk)
v_and_x = velocity_and_position_output(sim=sim, interval=1000)
sim.addPreTimeStepRunnable(v_and_x)

sim.run()
