"""Automatically extract CAD door goals for the supplied HOB second-floor plan.
Usage: python auto_detect_doors.py "path/to/engineering final as build.dxf"
The output is for desktop simulation, not validated for physical driving.
"""
from pathlib import Path
from collections import defaultdict, Counter
import argparse, json, math
import numpy as np
import cv2
import ezdxf
from PIL import Image, ImageDraw

ROOT=Path(__file__).resolve().parent

def extract(dxf_path):
    data=json.loads((ROOT/'floor2_map.json').read_text(encoding='utf-8'))
    doc=ezdxf.readfile(dxf_path); model=doc.modelspace()
    xmin,xmax,ymin,ymax=data['bounds']; resolution=25
    W=int((xmax-xmin)/resolution)+1; H=int((ymax-ymin)/resolution)+1
    wall=np.zeros((H,W),np.uint8)
    wall_layers={'-SECOND','2434-A-02-G-Brg1ST$0$G-B-BEARING WALL','WINDOWS','0'}
    def vec(p):return np.array([p[0],p[1]],dtype=float)
    def pixel(p):return round((p[0]-xmin)/resolution),round((ymax-p[1])/resolution)
    def draw(a,b):cv2.line(wall,pixel(a),pixel(b),255,2)
    def in_bounds(p, margin=0):return xmin-margin<=p[0]<=xmax+margin and ymin-margin<=p[1]<=ymax+margin
    segments=defaultdict(list)
    for entity in model:
        if entity.dxf.layer not in wall_layers:continue
        if entity.dxftype()=='LINE' and entity.dxf.linetype!='HIDDEN':
            a,b=vec(entity.dxf.start),vec(entity.dxf.end); draw(a,b)
            if in_bounds(a,2000) and in_bounds(b,2000):
                for axis in (0,1):
                    if abs(a[1-axis]-b[1-axis])<8 and abs(a[axis]-b[axis])>100:
                        z=round((a[1-axis]+b[1-axis])/50)*25
                        segments[(axis,z)].append(sorted([a[axis],b[axis]]))
        elif entity.dxftype()=='LWPOLYLINE':
            pts=list(entity.get_points('xy'))
            for a,b in zip(pts,pts[1:]+([pts[0]] if entity.closed else [])):draw(a,b)
    blocks=[]
    for entity in model.query('INSERT'):
        if not entity.dxf.name.startswith('DOOR_'):continue
        arc=next((a for a in doc.blocks[entity.dxf.name] if a.dxftype()=='ARC'),None)
        if arc is None:continue
        mat=entity.matrix44()
        # In this CAD family the leaf is drawn along local X; the closed
        # opening follows the arc end (local Y), including mirrored inserts.
        hinge=vec(mat.transform(arc.dxf.center)); closed=vec(mat.transform(arc.end_point)); opened=vec(mat.transform(arc.start_point))
        if not in_bounds(hinge):continue
        axis=closed-hinge
        if np.linalg.norm(axis)<100:continue
        n=np.array([-axis[1],axis[0]])/np.linalg.norm(axis)
        n*=np.sign(np.dot(n,opened-hinge)); door=(hinge+closed)/2
        draw(hinge-axis*.4,closed+axis*.4)
        blocks.append({'door':door,'normal':n,'source':'DXF '+entity.dxf.name+' handle '+entity.dxf.handle,'kind':'block'})
    # Additional openings: matched gaps on both faces of a wall.
    gaps=[]
    for (axis,z),vals in segments.items():
        merged=[]
        for lo,hi in sorted(vals):
            if merged and lo<merged[-1][1]+30:merged[-1][1]=max(merged[-1][1],hi)
            else:merged.append([lo,hi])
        for (_,lo),(hi,_) in zip(merged,merged[1:]):
            if 650<hi-lo<1600:gaps.append((axis,z,lo,hi))
    extra=[]
    for i,(axis,z,lo,hi) in enumerate(gaps):
        for ax2,z2,lo2,hi2 in gaps[i+1:]:
            if axis!=ax2 or not 75<abs(z-z2)<500 or abs(lo-lo2)>180 or abs(hi-hi2)>180:continue
            p=np.zeros(2);p[axis]=(lo+hi+lo2+hi2)/4;p[1-axis]=(z+z2)/2
            if any(np.linalg.norm(p-d['door'])<500 for d in blocks+extra):continue
            n=np.zeros(2);n[1-axis]=1
            extra.append({'door':p,'normal':n,'source':f'DXF paired wall opening ({axis}, {z}, {z2}, {round(lo)}, {round(hi)})','kind':'gap'})
    nodes=np.array(data['nodes'],dtype=float); edges=data['edges']
    def hits(a,b):
        n=max(2,int(np.linalg.norm(np.array(b)-a)/resolution)+1)
        xy=np.linspace(a,b,n); x=np.rint((xy[:,0]-xmin)/resolution).astype(int); y=np.rint((ymax-xy[:,1])/resolution).astype(int)
        if min(x)<0 or max(x)>=W or min(y)<0 or max(y)>=H:return n
        return int(np.count_nonzero(wall[y,x]))
    def anchors(p):
        for a,b in edges:
            u,v=nodes[a],nodes[b]; t=np.clip(np.dot(p-u,v-u)/np.dot(v-u,v-u),0,1); q=u+t*(v-u)
            yield q,float(np.linalg.norm(q-p))
    def options(room, doors):
        p=vec((room['x'],room['y'])); result=[]
        for d in doors:
            door=d['door']; dist=float(np.linalg.norm(p-door))
            if dist>8500:continue
            for sign in (1,-1):
                n=d['normal']*sign
                if np.dot(p-door,n)<400 or hits(p,door+n*600)!=0:continue
                stop=door-n*900
                aa=[(ad,q) for q,ad in anchors(stop) if ad<=5500 and np.dot(q-door,n)<-300 and hits(stop,q)==0]
                anchor=min(aa,key=lambda x:x[0]) if aa else None
                score=dist+(0 if sign==1 else 500)+(0 if anchor is None else .3*anchor[0])
                result.append((score,d,stop,anchor))
        return sorted(result,key=lambda x:x[0])
    counts=Counter(r['code'] for r in data['rooms']); seen=Counter(); result={}; report={}; occupied=[]
    def arr(x):return [float(v) for v in x]
    for room in data['rooms']:
        code=room['code'];seen[code]+=1
        label=code if counts[code]==1 else f'{code} (location {seen[code]})'
        opts=options(room,blocks)
        direct=[o for o in opts if o[3] is not None]
        if not direct:direct=[o for o in options(room,extra) if o[3] is not None]
        if direct:
            _,d,stop,anchor=direct[0]
            if any(np.linalg.norm(d['door']-p)<250 for p in occupied):
                report[label]={'status':'ambiguous_shared_door','source':d['source'],'door':arr(d['door'])};continue
            occupied.append(d['door'])
            result[label]={'door':arr(d['door']),'stop':arr(stop),'anchor':arr(anchor[1]),'source':d['source']+'; automatically matched; simulation only'}
            report[label]={'status':'auto_'+d['kind'],'source':d['source'],'door':arr(d['door']),'stop':arr(stop)}
        elif opts:
            _,d,stop,_=opts[0]
            report[label]={'status':'door_candidate_no_direct_corridor_route','door':arr(d['door']),'source':d['source']}
        else:report[label]={'status':'unresolved_no_supported_door_match'}
    # Inner rooms: select a verified corridor-facing parent on the opposite
    # side of the detected interior door. The room-number family disambiguates
    # suites with more than one connecting opening (e.g. 2231 / 2240).
    labels=list(report)
    room_by_label=dict(zip(labels,data['rooms']))
    for label,room in room_by_label.items():
        if label in result or report[label]['status']!='door_candidate_no_direct_corridor_route':continue
        parent_code=room['code'][:3]+'0'
        if parent_code==label or parent_code not in result:continue
        parent=room_by_label[parent_code]
        p=vec((room['x'],room['y'])); q=vec((parent['x'],parent['y']))
        for _,d,stop,anchor in options(room,blocks):
            if anchor is not None:continue
            door=d['door']; n=(door-stop)/900
            if np.dot(q-door,n)>=-400:continue
            if hits(q,door-n*600)!=0:continue
            exterior=result[parent_code]
            result[label]={**exterior,'shared_with':parent_code,'verification_code':parent['code'],'internal_door':arr(door),'source':exterior['source']+'; shared suite entrance inferred from interior door '+d['source']}
            report[label]={'status':'shared_exterior_door','shared_with':parent_code,'verification_code':parent['code'],'door':exterior['door'],'stop':exterior['stop'],'internal_door':arr(door),'source':result[label]['source']}
            break
    # Plan-specific recovery: 2680 has an L-shaped interior. A direct
    # label-to-door line crosses a partition, so use local free-space connectivity.
    def connected_local(a,b,margin=1800):
        x1,y1=pixel(a); x2,y2=pixel(b); m=round(margin/resolution)
        left=max(0,min(x1,x2)-m); top=max(0,min(y1,y2)-m)
        right=min(W,max(x1,x2)+m+1); bottom=min(H,max(y1,y2)+m+1)
        free=(wall[top:bottom,left:right]==0).astype(np.uint8)
        _,lab=cv2.connectedComponents(free,connectivity=4)
        u,v=lab[y1-top,x1-left],lab[y2-top,x2-left]
        return bool(u and u==v)
    if '2680' not in result:
        room=room_by_label['2680']; p=vec((room['x'],room['y']))
        for d in blocks:
            if not d['source'].endswith('handle 2F4BB'):continue
            door=d['door']
            for sign in (1,-1):
                n=d['normal']*sign; stop=door-n*900
                if np.dot(p-door,n)<400 or not connected_local(p,door+n*600):continue
                aa=[(ad,q) for q,ad in anchors(stop) if ad<=5500 and np.dot(q-door,n)<-300 and hits(stop,q)==0]
                if not aa:continue
                _,anchor=min(aa,key=lambda x:x[0])
                result['2680']={'door':arr(door),'stop':arr(stop),'anchor':arr(anchor),'source':d['source']+'; L-shaped room connectivity; plan-specific recovery; simulation only'}
                report['2680']={'status':'auto_block_bent_interior','door':arr(door),'stop':arr(stop),'source':result['2680']['source']}
                break
    # 2270/2271: common open entrance bounded by the two CAD wall ends.
    # These handles identify the opening on this particular supplied drawing,
    # not a generic classifier. No actual door block or external sign is assumed.
    if all(k not in result for k in ('2270','2271')):
        jambs=[doc.entitydb.get(h) for h in ('4818','480C')]
        if all(e is not None and e.dxftype()=='LINE' and e.dxf.layer=='-SECOND' for e in jambs):
            ends=[max((vec(e.dxf.start),vec(e.dxf.end)),key=lambda q:q[1]) for e in jambs]
            door=(ends[0]+ends[1])/2; stop=door+np.array([0.,900.])
            aa=[(ad,q) for q,ad in anchors(stop) if ad<2000 and q[1]>=door[1] and hits(stop,q)==0]
            inside=door-np.array([0.,600.])
            rooms_ok=all(connected_local(vec((room_by_label[k]['x'],room_by_label[k]['y'])),inside) for k in ('2270','2271'))
            if aa and rooms_ok and hits(ends[0]*.1+ends[1]*.9,ends[0]*.9+ends[1]*.1)==0 and hits(stop,inside)==0:
                _,anchor=min(aa,key=lambda x:x[0])
                for label in ('2270','2271'):
                    source='DXF open shared entrance bounded by LINE handles 4818 and 480C; plan-specific inference; external sign unknown'
                    result[label]={'door':arr(door),'stop':arr(stop),'anchor':arr(anchor),'source':source,'entrance_type':'shared_open_access','verification_code':label}
                    report[label]={'status':'shared_open_access','door':arr(door),'stop':arr(stop),'source':source,'verification_code':label,'external_sign_unconfirmed':True}
    (ROOT/'door_positions.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    summary={'door_block_inserts_in_bounds':len(blocks),'paired_opening_candidates':len(extra),'room_labels':len(data['rooms']),'auto_goals':len(result),'statuses':dict(Counter(r['status'] for r in report.values())),'rooms':report,'note':'CAD geometric inference, not physical clearance verification. Main corridor graph inherited from V5.'}
    (ROOT/'door_detection_report.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    im=Image.open(ROOT/'floor2_map.png').convert('RGB'); im.thumbnail((2400,1200)); dr=ImageDraw.Draw(im); w,h=im.size
    def ip(p):return ((p[0]-xmin)/(xmax-xmin)*w,(ymax-p[1])/(ymax-ymin)*h)
    for label,d in result.items():
        x,y=ip(d['door']); sx,sy=ip(d['stop']); ax,ay=ip(d['anchor']);dr.line((ax,ay,sx,sy),fill='#168649',width=2); dr.line((sx,sy,x,y),fill='#c2392d',width=1);dr.ellipse((x-3,y-3,x+3,y+3),fill='#c2392d');dr.ellipse((sx-3,sy-3,sx+3,sy+3),fill='#168649')
    for room,label in zip(data['rooms'],report):
        if label not in result:
            x,y=ip((room['x'],room['y'])); dr.ellipse((x-6,y-6,x+6,y+6),outline='#e88714',width=3)
    im.save(ROOT/'automatic_doors_preview.png')
    print(json.dumps({k:v for k,v in summary.items() if k!='rooms'},indent=2))
    print('Unresolved:', ', '.join(k for k in report if k not in result))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('dxf',type=Path);args=parser.parse_args();extract(args.dxf)
