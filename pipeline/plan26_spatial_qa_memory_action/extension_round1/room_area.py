"""Sim-only floor-footprint truth; never infer Action necessity from area GT."""
import argparse, json, math, pathlib

def cross(a,b,c):return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
def distance_segment(p,a,b):
    d=sum((y-x)**2 for x,y in zip(a,b))
    t=max(0,min(1,sum((x-u)*(v-u) for x,u,v in zip(p,a,b))/d)) if d else 0
    return math.dist(p,[x+t*(y-x) for x,y in zip(a,b)])
def intersects(a,b,c,d):
    ab=(cross(a,b,c),cross(a,b,d));cd=(cross(c,d,a),cross(c,d,b))
    if ab[0]*ab[1]<0 and cd[0]*cd[1]<0:return True
    return any(abs(v)<1e-9 and distance_segment(p,x,y)<1e-9 for v,p,x,y in [(ab[0],c,a,b),(ab[1],d,a,b),(cd[0],a,c,d),(cd[1],b,c,d)])
def polygon(points):
    ps=[list(map(float,p)) for p in points]
    if ps and ps[0]==ps[-1]:ps.pop()
    if len(ps)<3 or any(len(p)!=2 or not all(math.isfinite(v) for v in p) for p in ps):raise ValueError('invalid_polygon')
    if len(set(map(tuple,ps)))!=len(ps):raise ValueError('duplicate_vertex')
    edges=list(zip(ps,ps[1:]+ps[:1]))
    for i,(a,b) in enumerate(edges):
        for j,(c,d) in enumerate(edges):
            if j<=i+1 or (i==0 and j==len(edges)-1):continue
            if intersects(a,b,c,d):raise ValueError('self_intersection')
    signed=sum(a[0]*b[1]-b[0]*a[1] for a,b in edges)/2
    if abs(signed)<1e-8:raise ValueError('zero_area')
    # A separate triangle-fan sum provides a second area calculation.
    fan=abs(sum(cross(ps[0],ps[i],ps[i+1]) for i in range(1,len(ps)-1)))/2
    assert math.isclose(abs(signed),fan,rel_tol=1e-9,abs_tol=1e-8)
    return ps,abs(signed)
def contains(ps,p,margin=0):
    if min(distance_segment(p,a,b) for a,b in zip(ps,ps[1:]+ps[:1]))<=max(1e-9,margin):return False
    x,y=p;inside=False
    for a,b in zip(ps,ps[1:]+ps[:1]):
        if (a[1]>y)!=(b[1]>y) and x<(b[0]-a[0])*(y-a[1])/(b[1]-a[1])+a[0]:inside=not inside
    return inside
def validate_rooms(rooms):
    good={};rejected=[]
    for name,r in rooms.items():
        try:
            if r.get('holes'):raise ValueError('holes_unsupported')
            ps,area=polygon(r.get('outline',[]))
            provided=float(r['area'])
            if not math.isfinite(provided) or not math.isclose(area,provided,rel_tol=1e-4,abs_tol=1e-4):raise ValueError('stored_area_disagrees')
            good[name]={'polygon':ps,'area_m2':area,'stored_area_m2':provided}
        except (ValueError,TypeError,KeyError) as e:rejected.append({'room':name,'reason':str(e)})
    return good,rejected
