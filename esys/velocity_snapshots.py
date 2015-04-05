from esys.lsm import *
from esys.lsm.util import Vec3
from esys.lsm.vis import povray
import csv
from numpy import array


class velocity_snapshots (Runnable):
    def __init__(self, sim, interval):
        Runnable.__init__(self)
        self.sim = sim
        self.interval = interval
        self.count = 0
        self.configure()
        f = open("sim.csv".format(self.count), 'w')
        f.close()

    def configure(
        self,
        lookAt=Vec3(0, 0, 0),
        camPosn=Vec3(0, 0, 20),
        zoomFactor=0.1,
        imageSize=[800, 600]
    ):
        self.lookAt = lookAt
        self.camPosn = camPosn
        self.zoomFactor = zoomFactor
        self.imageSize = imageSize

    def run(self):
        time_step = self.sim.getTimeStep()
        if (time_step == 1 or (time_step % self.interval) == 0):
            print(
                "Post-processing time step {0}...".format(time_step),
                end="",
                flush=True)
            self.velocity_write()
            self.snapshot()
            self.count += 1
            print(" done.")

    def velocity_write(self):
        vs = array([
            pp.getLinearVelocity().norm()
            for pp in self.sim.getParticleList()
        ])

        with open("sim.csv".format(self.count), 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter=' ')
            writer.writerow([
                self.count,
                self.sim.getTimeStep() * self.sim.getTimeStepSize(),
                # Division as each one of the component is scaled by kT
                sum(vs**2) / len(vs) / 3.0])

        with open("pdf/v-{0:04d}.csv".format(self.count), 'w') as csvfile:
            writer = csv.writer(csvfile, delimiter=' ')
            for v in vs:
                writer.writerow([v])

    def snapshot(self):
        pkg = povray
        Scene = pkg.Scene()
        plist = self.sim.getParticleList()
        for pp in plist:
            povsphere = pkg.Sphere(pp.getPosn(), pp.getRadius())
            povsphere.apply(pkg.Colors.Red)
            Scene.add(povsphere)
        camera = Scene.getCamera()
        camera.setLookAt(self.lookAt)
        camera.setPosn(self.camPosn)
        camera.setZoom(self.zoomFactor)
        fname = "animation/snap_{0:04d}.png".format(self.count)
        Scene.render(
            offScreen=True,
            interactive=False,
            fileName=fname,
            size=self.imageSize
        )
