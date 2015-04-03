from esys.lsm import *
from esys.lsm.util import Vec3
from esys.lsm.vis import povray
from statsmodels.nonparametric.kernel_density import KDEMultivariate
from numpy import linspace
import csv


class hist_and_snaps (Runnable):
    def __init__(self, sim, interval, resolution=100):
        Runnable.__init__(self)
        self.sim = sim
        self.interval = interval
        self.resolution = resolution
        self.maxV = 5
        self.count = 0
        self.configure()

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
            self.histogram()
            self.snapshot()
            self.count += 1
            print(" done.")

    def histogram(self):
        x = linspace(0, self.maxV, self.resolution)

        v = [
            pp.getLinearVelocity().norm()
            for pp in self.sim.getParticleList()
        ]
        kde = KDEMultivariate(v, bw='normal_reference', var_type='c')
        with open("v-pdf{0:04d}.csv".format(self.count), 'w') as csvfile:
            writer = csv.writer(csvfile, delimiter=' ')
            for n in range(self.resolution):
                writer.writerow([x[n], kde.pdf(x)[n]])

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
