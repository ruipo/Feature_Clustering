import numpy as n, pylab as p

def _angle_to_point(point, centre):
    '''calculate angle in 2-D between points and x axis'''
    delta = point - centre
    res = n.arctan(delta[1] / delta[0])
    if delta[0] < 0:
        res += n.pi
    return res

def _draw_triangle(p1, p2, p3, **kwargs):
    tmp = n.vstack((p1,p2,p3))
    x,y = [x[0] for x in zip(tmp.transpose())]
    p.fill(x,y, **kwargs)

def area_of_triangle(p1, p2, p3):
    '''calculate area of any triangle given co-ordinates of the corners'''
    return n.linalg.norm(n.cross((p2 - p1), (p3 - p1)))/2.


def convex_hull(points, graphic=False, smidgen=0.0075):
    '''
    Calculate subset of points that make a convex hull around points
    Recursively eliminates points that lie inside two neighbouring points until only convex hull is remaining.

    :Parameters:
    points : ndarray (2 x m)
    array of points for which to find hull
    graphic : bool
    use pylab to show progress?
    smidgen : float
    offset for graphic number labels - useful values depend on your data range

    :Returns:
    hull_points : ndarray (2 x n)
    convex hull surrounding points
    '''

    if graphic:
        p.clf()
        p.plot(points[0], points[1], 'ro')
    n_pts = points.shape[1]
    assert(n_pts > 5)
    centre = points.mean(1)
    if graphic: p.plot((centre[0],),(centre[1],),'bo')
    angles = n.apply_along_axis(_angle_to_point, 0, points, centre)
    pts_ord = points[:,angles.argsort()]
    if graphic:
        for i in xrange(n_pts):
            p.text(pts_ord[0,i] + smidgen, pts_ord[1,i] + smidgen, \
                   '%d' % i)
    pts = [x[0] for x in zip(pts_ord.transpose())]
    prev_pts = len(pts) + 1
    k = 0
    while prev_pts > n_pts:
        prev_pts = n_pts
        n_pts = len(pts)
        if graphic: p.gca().patches = []
        i = -2
        while i < (n_pts - 2):
            Aij = area_of_triangle(centre, pts[i],     pts[(i + 1) % n_pts])
            Ajk = area_of_triangle(centre, pts[(i + 1) % n_pts], \
                                   pts[(i + 2) % n_pts])
            Aik = area_of_triangle(centre, pts[i],     pts[(i + 2) % n_pts])
            if graphic:
                _draw_triangle(centre, pts[i], pts[(i + 1) % n_pts], \
                               facecolor='blue', alpha = 0.2)
                _draw_triangle(centre, pts[(i + 1) % n_pts], \
                               pts[(i + 2) % n_pts], \
                               facecolor='green', alpha = 0.2)
                _draw_triangle(centre, pts[i], pts[(i + 2) % n_pts], \
                               facecolor='red', alpha = 0.2)
            if Aij + Ajk < Aik:
                if graphic: p.plot((pts[i + 1][0],),(pts[i + 1][1],),'go')
                del pts[i+1]
            i += 1
            n_pts = len(pts)
        k += 1
    return n.asarray(pts)







if __name__ == "__main__":

    import scipy.interpolate as interpolate

#    fig = p.figure(figsize=(10,10))

    theta = 2*n.pi*n.random.rand(1000)
    r = n.random.rand(1000)**0.5
    x,y = r*p.cos(theta),r*p.sin(theta)

    points = n.ndarray((2,len(x)))
    points[0,:],points[1,:] = x,y

    scale = 1.03
    hull_pts = scale*convex_hull(points)

    p.plot(x,y,'ko')

    x,y = [],[]
    convex = scale*hull_pts
    for point in convex:
        x.append(point[0])
        y.append(point[1])
    x.append(convex[0][0])
    y.append(convex[0][1])

    x,y = n.array(x),n.array(y)

#Taken from https://stackoverflow.com/questions/14344099/numpy-scipy-smooth-spline-representation-of-an-arbitrary-contour-flength
    nt = n.linspace(0, 1, 100)
    t = n.zeros(x.shape)
    t[1:] = n.sqrt((x[1:] - x[:-1])**2 + (y[1:] - y[:-1])**2)
    t = n.cumsum(t)
    t /= t[-1]
    x2 = interpolate.spline(t, x, nt)
    y2 = interpolate.spline(t, y, nt)
    p.plot(x2, y2,'r--',linewidth=2)

    p.show()