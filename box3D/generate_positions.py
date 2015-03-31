from mako.template import Template
from numpy.random import uniform
from numpy import array

positions = [
        array([uniform(-0.5, 0.5), uniform(-0.5, 0.5), uniform(-0.5, 0.5)])
        for n in xrange(10)]

fh = open('constant/kinematicCloudPositions', 'w')
template = Template(filename='constant/kinematicCloudPositions.mako')
print >>fh, template.render(positions=positions)
