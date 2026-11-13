"""3-D animation adaptation of the tracking-task schematic's racing quadrotor.

The standalone TikZ source is bundled at
assets/paper_sources/original_diagram/figures/core_pivot_current.tex.
The swept hexagonal fuselage, crossed arms, motor spacing, three-blade rotors
and nose camera are retained. Depth is illustrative; the paper contains a 2-D drawing,
not an attitude or vehicle model. The PIVOT branding is omitted so all
controllers share the identical unbranded asset.

Local x points forward and z points up. Geometry units are illustrative.
Near-black surfaces and coloured contours remain legible on black;
the body is deliberately not filled with the algorithm colour.
"""
from pathlib import Path
import numpy as np
from matplotlib.colors import to_rgb
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from demo.geometry import Mesh, closed


def bezier(p0, p1, p2, p3, n=9):
    t = np.linspace(0, 1, n)[:, None]
    return (1-t)**3*np.array(p0)+3*(1-t)**2*t*np.array(p1)+3*(1-t)*t*t*np.array(p2)+t**3*np.array(p3)


def build_mesh():
    """Preserve the TikZ planform and give it a shallow bevelled extrusion."""
    mesh = Mesh()
    motors = np.array([[-.78, -.54, .01], [.78, .54, .01],
                       [-.78, .54, .01], [.78, -.54, .01]])
    for motor in motors:
        end = motor[:2]
        direction = end / np.linalg.norm(end)
        side = np.array([-direction[1], direction[0]])*.044
        root = direction*.08
        mesh.prism([root-side, end-side, end+side, root+side], -.035, .025, 'frame', 'silver')
        mesh.line([[*root, .03], [*end, .03]], 'accent')
        mesh.cylinder(motor, .17, .075, 'motor', n=16, edge='silver')
        mesh.cylinder(motor+[0, 0, .075], .125, .045, 'silver', n=16, edge='accent')
        mesh.cylinder(motor+[0, 0, .12], .060, .025, 'frame', n=12)

    # Same six vertices as the paper, with a narrow bevel for readable depth.
    plan = np.array([[-.48, -.19], [.18, -.22], [.55, 0],
                     [.18, .22], [-.48, .19], [-.62, 0]])
    lower = np.column_stack([plan, np.full(6, -.065)])
    shoulder = np.column_stack([plan, np.full(6, .075)])
    top = np.column_stack([plan*.86, np.full(6, .145)])
    mesh.face(lower[::-1], 'dark')
    mesh.face(top, 'hull')
    for i in range(6):
        j = (i+1) % 6
        mesh.face([lower[i], lower[j], shoulder[j], shoulder[i]], 'dark')
        mesh.face([shoulder[i], shoulder[j], top[j], top[i]], 'silver')
    mesh.line(closed(shoulder), 'accent')
    mesh.line(closed(top), 'silver')
    # A slim longitudinal crease substitutes for a tiny illegible logo.
    mesh.line([[-.46, 0, .149], [.38, 0, .149]], 'silver')

    # Triangular nose camera housing and a circular forward-facing lens.
    mesh.prism([[.46, 0], [.66, -.11], [.66, .11]], -.025, .08, 'frame', 'silver')
    mesh.cylinder([.658, 0, .025], .066, .022, 'lens', n=18, axis='x', edge='accent')
    mesh.cylinder([.682, .014, .043], .012, .002, 'white', n=10, axis='x')
    for x, y, r in [(-.49, .13, .024), (-.49, -.13, .024), (.31, .15, .021), (.31, -.15, .021)]:
        mesh.cylinder([x, y, .14], r, .01, 'white' if x < 0 else 'accent', n=10)
    return mesh, motors


class LatexStyleDrone:
    """Reusable collections, suitable for Matplotlib FuncAnimation."""
    def __init__(self, ax, color='#25D6BF', scale=.46, outline=False, finish='silver'):
        self.ax = ax
        self.scale = float(scale)
        self.color = np.array(to_rgb(color))
        self.outline = bool(outline)
        if finish not in ('silver', 'dark'):
            raise ValueError('finish must be silver or dark')
        self.finish = finish
        self.mesh, self.motors = build_mesh()
        self.palette = {'frame': np.array([.25, .30, .34]),
                        'motor': np.array([.27, .32, .36]),
                        'dark': np.array([.11, .15, .19]),
                        'hull': np.array([.47, .55, .61]),
                        'silver': np.array([.69, .76, .80]),
                        'lens': np.array([.04, .12, .15]),
                        'white': np.array([.88, .94, .98]),
                        'accent': self.color}
        if finish == 'dark':
            # Preserve the paper's geometry without its silver body finish.
            self.palette.update(frame=np.array([.025, .035, .045]),
                                motor=np.array([.035, .045, .055]),
                                dark=np.array([.008, .012, .016]),
                                hull=np.array([.012, .018, .022]),
                                silver=np.array([.035, .045, .055]))
        self.surface = Poly3DCollection([], edgecolors='none', zorder=6)
        self.edges = Line3DCollection([], linewidths=1.05, zorder=7)
        self.props = Poly3DCollection([], edgecolors='none', zorder=8)
        self.rims = Line3DCollection([], linewidths=.8, zorder=9)
        self.artists = [self.surface, self.edges, self.props, self.rims]
        for artist in self.artists:
            ax.add_collection3d(artist, autolim=False)
            artist.set_clip_on(False)
        self.static_colors = np.array([self.palette[x] for x in self.mesh.materials])
        self.normals = []
        for f in self.mesh.faces:
            normal = np.cross(f[1]-f[0], f[2]-f[0])
            self.normals.append(normal/max(np.linalg.norm(normal), 1e-12))
        self.normals = np.array(self.normals)
        self.light = np.array([-.3, -.4, .85])
        self.light /= np.linalg.norm(self.light)
        upper = bezier([.045, .017], [.19, .070], [.32, .060], [.40, 0])
        lower = bezier([.40, 0], [.31, -.055], [.18, -.050], [.045, -.017])
        self.blade = np.vstack([upper, lower[1:]])
        self.set_pose([0, 0, 0])

    def set_pose(self, position, yaw=0., rotor_phase=0., pitch=0., roll=0.):
        cy, sy = np.cos(yaw), np.sin(yaw)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cr, sr = np.cos(roll), np.sin(roll)
        rotation = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]]) @ np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]]) @ np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
        position = np.asarray(position, float)
        def transform(points):
            return np.asarray(points) @ rotation.T * self.scale + position
        lighting = .58+.42*np.clip((self.normals@rotation.T)@self.light, 0, 1)
        self.surface.set_verts([] if self.outline else [transform(x) for x in self.mesh.faces])
        self.surface.set_facecolors(self.static_colors*lighting[:, None])
        self.edges.set_segments([transform(x) for x in self.mesh.edges])
        self.edges.set_color([self.color*(1. if x == 'accent' else .65)
                             if self.finish == 'dark' else self.palette[x]
                             for x in self.mesh.edge_materials])
        faces, rims = [], []
        spins = [1, 1, -1, -1]  # Same diagonal pairs as the TikZ drawing.
        angles = np.linspace(0, 2*np.pi, 49)
        for motor, spin in zip(self.motors, spins):
            center = motor+[0, 0, .15]
            ring = center+np.column_stack([.4*np.cos(angles), .4*np.sin(angles), np.zeros(len(angles))])
            rims.append(transform(ring))
            for j in range(3):
                theta = spin*float(rotor_phase)+j*2*np.pi/3
                matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
                points = center+np.column_stack([self.blade@matrix.T, np.zeros(len(self.blade))])
                faces.append(transform(points))
                rims.append(transform(closed(points)))
        self.props.set_verts([] if self.outline else faces)
        self.props.set_facecolors([(*(.7*self.color+.3*np.ones(3)), .22)])
        self.rims.set_segments(rims)
        self.rims.set_colors([(*self.color, .4 if i % 4 == 0 else .85) for i in range(len(rims))])
        return self.artists
