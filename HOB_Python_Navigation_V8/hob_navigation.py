"""HOB floor 2: desktop simulation only. No motor commands are sent."""
from collections import deque
import heapq
import json
import math
import os
import queue
import re
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import ttk, messagebox
from PIL import Image, ImageTk

ROOT = Path(__file__).resolve().parent
DATA = json.loads((ROOT / 'floor2_map.json').read_text(encoding='utf-8'))
BASE_NODES = [tuple(p) for p in DATA['nodes']]
BASE_EDGES = DATA['edges']
NODES = BASE_NODES[:]
EDGES = [tuple(e) for e in BASE_EDGES]
BASE_PARTS = {}
DOOR_EDGES = {}
DOORS = {}
DOOR_FILE = ROOT / 'door_positions.json'
ROOMS = DATA['rooms']
CODES = {r['code'] for r in ROOMS}
ROOM_LABELS = {}
for i, room in enumerate(ROOMS):
    code = room['code']
    same = [j for j, r in enumerate(ROOMS) if r['code'] == code]
    label = code if len(same) == 1 else f'{code} (location {same.index(i)+1})'
    ROOM_LABELS[label] = i


def project(p, edge=None, *, base=False):
    """Return (edge index, fraction along edge, projected point, distance)."""
    best = None
    nodes, edges = (BASE_NODES, BASE_EDGES) if base else (NODES, EDGES)
    for k, (i, j) in enumerate(edges):
        if edge is not None and k != edge:
            continue
        a, b = nodes[i], nodes[j]
        dx, dy = b[0]-a[0], b[1]-a[1]
        t = max(0., min(1., ((p[0]-a[0])*dx+(p[1]-a[1])*dy)/(dx*dx+dy*dy)))
        q = (a[0]+t*dx, a[1]+t*dy)
        candidate = (k, t, q, math.dist(p, q))
        if best is None or candidate[3] < best[3]:
            best = candidate
    return best


def astar(start, goal):
    """Split graph edges at the exact current and target positions; A*."""
    pts = NODES + [start[2], goal[2]]
    s, g = len(pts)-2, len(pts)-1
    adj = [[] for _ in pts]
    for k, (a,b) in enumerate(EDGES):
        split = [(0., a), (1., b)]
        if start[0] == k: split.append((start[1], s))
        if goal[0] == k: split.append((goal[1], g))
        split.sort()
        for (_,u), (_,v) in zip(split, split[1:]):
            w = math.dist(pts[u],pts[v])
            adj[u].append((v,w)); adj[v].append((u,w))
    cost, prev = {s:0.}, {}
    pending = [(math.dist(pts[s],pts[g]),0.,s)]
    while pending:
        _, c, u = heapq.heappop(pending)
        if c > cost[u]: continue
        if u == g:
            path = [pts[u]]
            while u != s:
                u = prev[u]; path.append(pts[u])
            return list(reversed(path))
        for v,w in adj[u]:
            n = c+w
            if n < cost.get(v,math.inf):
                cost[v],prev[v] = n,u
                heapq.heappush(pending,(n+math.dist(pts[v],pts[g]),n,v))
    return []


def rebuild_graph():
    # Split original corridor edges at the calibrated connections. Door spurs
    # are real graph edges, so the next route starts at the exact previous stop.
    NODES[:] = BASE_NODES
    EDGES.clear(); BASE_PARTS.clear(); DOOR_EDGES.clear()
    for k, (a,b) in enumerate(BASE_EDGES):
        split = [(0.,a),(1.,b)]
        branches = []
        for i, d in DOORS.items():
            e, t, anchor, _ = project(d['anchor'], base=True)
            if e != k: continue
            match = next((node for frac,node in split if abs(frac-t)<1e-9),None)
            if match is None:
                match=len(NODES); NODES.append(anchor); split.append((t,match))
            branches.append((i,match,tuple(d['stop'])))
        split.sort(); BASE_PARTS[k]=[]
        for (_,u),(_,v) in zip(split,split[1:]):
            BASE_PARTS[k].append(len(EDGES)); EDGES.append((u,v))
        shared_branches = {}
        for i,node,stop in branches:
            key=(node,stop)
            if key in shared_branches:
                DOOR_EDGES[i]=shared_branches[key]; continue
            if math.dist(NODES[node],stop)<1e-6:
                DOOR_EDGES[i]=None
            else:
                end=len(NODES); NODES.append(stop)
                DOOR_EDGES[i]=len(EDGES); EDGES.append((node,end))
            shared_branches[key]=DOOR_EDGES[i]


def load_doors():
    if DOOR_FILE.exists():
        raw=json.loads(DOOR_FILE.read_text(encoding='utf-8'))
        for label,d in raw.items():
            if label not in ROOM_LABELS: continue
            if not all(len(d[k])==2 and all(math.isfinite(v) for v in d[k]) for k in ('door','stop','anchor')):
                raise ValueError('Invalid saved door: '+label)
            if project(d['anchor'],base=True)[3]>1e-4: raise ValueError('Invalid corridor anchor: '+label)
            DOORS[ROOM_LABELS[label]]=d
    rebuild_graph()


def save_door(index,door,stop,anchor):
    anchor=project(anchor,base=True)[2]
    if math.dist(door,stop)<1 or math.dist(door,stop)>2500:
        raise ValueError('Select a stop just outside the door (not the door itself).')
    if math.dist(anchor,stop)>6000: raise ValueError('Choose a closer corridor connection.')
    new=dict(DOORS); new[index]={'door':list(door),'stop':list(stop),'anchor':list(anchor),'source':'manual map calibration'}
    labels={v:k for k,v in ROOM_LABELS.items()}
    tmp=DOOR_FILE.with_suffix('.tmp')
    tmp.write_text(json.dumps({labels[i]:d for i,d in new.items()},indent=2),encoding='utf-8')
    tmp.replace(DOOR_FILE)
    DOORS.clear(); DOORS.update(new); rebuild_graph()


def room_goal(index):
    if index in DOORS:
        p=DOORS[index]['stop']; edge=DOOR_EDGES[index]
        return project(p,edge) if edge is not None else project(p)
    raise ValueError('Room has no saved door goal.')


def verification_code(index):
    """The exterior sign can differ from the requested internal room."""
    return DOORS[index].get('verification_code',ROOMS[index]['code'])


load_doors()


class Navigator:
    """FIFO room tasks; return to a fixed HOME when the queue empties."""
    def __init__(self):
        self.position = None
        self.home = None
        self.trail = []
        self.remaining = []
        self.waiting = deque()
        self.completed = []
        self.task_id = 0  # generation token for camera results
        self.target = None
        self.goal = None
        self.state = 'IDLE'

    def place(self, p):
        if self.state not in ('IDLE', 'AT_HOME', 'STOPPED'):
            raise ValueError('Stop before placing the robot.')
        self.task_id += 1
        self.position = project(p,base=True)[2]
        if self.home is None:
            self.home = self.position
        self.trail = [self.position]
        self.remaining = []
        self.target = self.goal = None
        self.state = 'AT_HOME' if math.dist(self.position,self.home) < 1e-6 else 'IDLE'

    def set_home(self):
        if self.position is None or self.state not in ('IDLE','AT_HOME','STOPPED'):
            raise ValueError('Place or stop the robot before setting HOME.')
        if project(self.position,base=True)[3]>1e-5:
            raise ValueError('Set HOME on the main corridor network.')
        self.home = self.position
        if self.state != 'STOPPED': self.state = 'AT_HOME'

    def _start(self, index):
        goal = room_goal(index) if index is not None else project(self.home)
        path = astar(project(self.position), goal)
        if not path:
            raise ValueError('No route found. Robot remains stopped.')
        self.task_id += 1
        self.target, self.goal = index, goal[2]
        self.remaining = path[1:]
        self.state = 'MOVING' if index is not None else 'RETURNING_HOME'
        return self.task_id

    def new_task(self, index):
        if self.position is None: raise ValueError('Set robot position first.')
        if not isinstance(index,int) or not 0 <= index < len(ROOMS):
            raise ValueError('Invalid room.')
        if index not in DOORS:
            raise ValueError('No direct CAD door route was detected for this room. See door_detection_report.json.')
        if self.state in ('MOVING','VERIFYING','UNVERIFIED','BLOCKED'):
            self.waiting.append(index)
            return None
        # New room commands interrupt HOME return, not another room task.
        return self._start(index)

    def dispatch_next(self):
        if self.state not in ('VERIFIED','BLOCKED'):
            return
        try:
            if self.waiting:
                self._start(self.waiting[0])
                self.waiting.popleft()
            elif math.dist(self.position,self.home) < 1e-6:
                self.target = None; self.goal = self.home; self.remaining = []
                self.state = 'AT_HOME'
            else:
                self._start(None)
        except ValueError:
            self.target = None; self.goal = None; self.remaining = []
            self.state = 'BLOCKED'
            raise

    def advance(self,distance):
        if self.state not in ('MOVING','RETURNING_HOME'): return False
        distance = max(0.,distance)
        while self.remaining:
            p = self.remaining[0]
            d = math.dist(self.position,p)
            if d <= distance + 1e-8:
                self.position = p; self.remaining.pop(0)
                self.trail.append(p); distance = max(0.,distance-d)
            else:
                if distance <= 0: break
                t = distance/d
                self.position = tuple(a+(b-a)*t for a,b in zip(self.position,p))
                self.trail.append(self.position)
                break
        if not self.remaining:
            self.state = 'AT_HOME' if self.state == 'RETURNING_HOME' else 'VERIFYING'
            return True
        return False

    def verify(self,task_id,code):
        if task_id != self.task_id or self.state != 'VERIFYING': return False
        if code == verification_code(self.target):
            self.completed.append(self.target)
            self.state = 'VERIFIED'
            self.dispatch_next()
        else:
            self.state = 'UNVERIFIED'
        return True

    def retry(self):
        if self.state == 'UNVERIFIED':
            self.task_id += 1
            self.state = 'VERIFYING'
        elif self.state == 'BLOCKED':
            self.dispatch_next()

    def stop(self):
        self.task_id += 1
        self.remaining = []
        self.waiting.clear()
        self.target = self.goal = None
        self.state = 'STOPPED'


def recognized_codes(text):
    return set(re.findall(r'(?<![A-Za-z0-9])(?:B[12]\d{3}|G0\d{3}|[1-4]\d{3})(?![A-Za-z0-9])', text.upper()))


def camera_verify(source,expected,cancel):
    """Use a local webcam index or uStreamer snapshot URL; require two matching frames."""
    import cv2
    import pytesseract
    import urllib.request
    from io import BytesIO
    exe = os.environ.get('TESSERACT_CMD')
    if exe: pytesseract.pytesseract.tesseract_cmd = exe
    elif os.name == 'nt':
        default = Path(r'C:\Program Files\Tesseract-OCR\tesseract.exe')
        if default.exists(): pytesseract.pytesseract.tesseract_cmd = str(default)
    cap = None
    observations = []
    try:
        if source.isdigit():
            cap = cv2.VideoCapture(int(source))
            if not cap.isOpened(): raise RuntimeError('Cannot open camera.')
        elif not source.startswith(('http://','https://')):
            raise ValueError('Use a webcam index (0) or an HTTP snapshot URL.')
        for _ in range(4):
            if cancel.is_set(): return None, 'Cancelled'
            if cap is not None:
                ok, frame = cap.read()
                if not ok: raise RuntimeError('Cannot read camera frame.')
                img = Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
            else:
                with urllib.request.urlopen(source,timeout=4) as response:
                    payload = response.read(12_000_001)
                    if len(payload)>12_000_000: raise ValueError('Snapshot too large.')
                with Image.open(BytesIO(payload)) as original:
                    img = original.convert('RGB')
            if cancel.is_set(): return None, 'Cancelled'
            text = pytesseract.image_to_string(img,lang='eng',config='--psm 11',timeout=5)
            codes = recognized_codes(text)
            observations.append(codes)
            if len(observations)>=2 and observations[-1]==observations[-2]=={expected}:
                return expected, f'OCR matched {expected} in two frames.'
            if cancel.wait(.25): return None,'Cancelled'
        found = sorted(set().union(*observations))
        return None,'No confirmed match. OCR saw: '+(', '.join(found) or 'no room code')
    finally:
        if cap is not None: cap.release()


class App:
    def __init__(self,root):
        self.root=root; root.title('HOB Robot V8 | Shared entrances + FIFO + HOME'); root.geometry('1250x780')
        self.nav=Navigator(); self.results=queue.Queue(); self.worker=None; self.cancel=None; self.pending=None
        self.image=Image.open(ROOT/'floor2_map.png').convert('RGB')
        self.last=time.monotonic(); self.transform=(1.,0.,0.)
        self.room=tk.StringVar(value='2030'); self.mode=tk.StringVar(value='Simulation'); self.source=tk.StringVar(value='0')
        self.speed=tk.DoubleVar(value=6500); self.network=tk.BooleanVar(value=True); self.status=tk.StringVar(value='Click a corridor to place the robot.')
        self.calibration=None; self.cal_points=[]; self.batch=False
        self.show_doors=tk.BooleanVar(value=True)
        self.zoom=1.0; self.center=None; self.drag=None
        self.verification_mode=None; self.observed=tk.StringVar(); self.task=tk.StringVar(value='No active task')
        self.queue_text=tk.StringVar(value='Queue: empty')
        self.history_text=tk.StringVar(value='Completed: none')
        self.home_file=ROOT/'home_position.json'
        try:
            saved=json.loads(self.home_file.read_text(encoding='utf-8'))
            p=tuple(float(x) for x in saved['home'])
            if len(p)!=2 or not all(math.isfinite(x) for x in p) or project(p,base=True)[3]>1e-5:
                raise ValueError('Invalid HOME')
            self.nav.place(p)
            self.status.set('Saved HOME loaded. Simulation starts at HOME.')
        except (OSError,ValueError,TypeError,KeyError):
            self.status.set('Click a corridor to set initial position and HOME.')
        bar=ttk.Frame(root,padding=8); bar.pack(fill='x')
        ttk.Label(bar,text='Destination').pack(side='left')
        self.combo=ttk.Combobox(bar,textvariable=self.room,values=list(ROOM_LABELS),width=21); self.combo.pack(side='left',padx=6)
        self.combo.bind('<Return>',lambda _:self.send())
        ttk.Button(bar,text='Add task',command=self.send).pack(side='left',padx=4)
        ttk.Button(bar,text='STOP',command=self.stop).pack(side='left',padx=4)
        ttk.Button(bar,text='Set HOME here',command=self.set_home).pack(side='left',padx=4)
        ttk.Checkbutton(bar,text='Corridor network',variable=self.network,command=self.draw).pack(side='left',padx=10)
        ttk.Label(bar,text='Simulation speed').pack(side='left')
        ttk.Scale(bar,from_=1000,to=18000,variable=self.speed).pack(side='left',fill='x',expand=True,padx=5)
        camera=ttk.Frame(root,padding=(8,0,8,8)); camera.pack(fill='x')
        ttk.Label(camera,text='Verification:').pack(side='left')
        ttk.Combobox(camera,textvariable=self.mode,values=['Simulation','Camera OCR'],state='readonly',width=13).pack(side='left',padx=6)
        ttk.Label(camera,text='Camera index / snapshot URL:').pack(side='left')
        ttk.Entry(camera,textvariable=self.source,width=32).pack(side='left',padx=6)
        ttk.Label(camera,text='Simulated OCR:').pack(side='left')
        ttk.Entry(camera,textvariable=self.observed,width=8).pack(side='left',padx=4)
        ttk.Button(camera,text='Submit OCR',command=self.manual_verify).pack(side='left')
        ttk.Button(camera,text='Retry verification',command=self.retry).pack(side='left',padx=5)
        doors=ttk.Frame(root,padding=(8,0,8,4)); doors.pack(fill='x')
        ttk.Checkbutton(doors,text='Show detected doors',variable=self.show_doors,command=self.draw).pack(side='left',padx=4)
        ttk.Button(doors,text='Detection report',command=self.show_report).pack(side='left',padx=4)
        ttk.Button(doors,text='Fit map',command=self.fit_map).pack(side='left',padx=4)
        self.door_status=tk.StringVar(); ttk.Label(doors,textvariable=self.door_status).pack(side='left',padx=8)
        self.room.trace_add('write',lambda *_: (self.refresh_door_status(), self.draw() if hasattr(self,'canvas') else None))
        self.refresh_door_status()
        ttk.Label(root,textvariable=self.task,font=('TkDefaultFont',12,'bold'),padding=7).pack(fill='x')
        ttk.Label(root,textvariable=self.status,wraplength=1150,padding=7).pack(fill='x')
        ttk.Label(root,textvariable=self.queue_text,wraplength=1150,padding=4).pack(fill='x')
        ttk.Label(root,textvariable=self.history_text,wraplength=1150,padding=4).pack(fill='x')
        self.canvas=tk.Canvas(root,bg='#e6e9ed',highlightthickness=0); self.canvas.pack(fill='both',expand=True,padx=8)
        self.canvas.bind('<Configure>',self.resize); self.canvas.bind('<Button-1>',self.place)
        self.canvas.bind('<MouseWheel>',self.wheel)
        self.canvas.bind('<Button-4>',lambda e:self.wheel(e,1))
        self.canvas.bind('<Button-5>',lambda e:self.wheel(e,-1))
        self.canvas.bind('<ButtonPress-3>',self.pan_start)
        self.canvas.bind('<B3-Motion>',self.pan_move)
        self.canvas.bind('<ButtonRelease-3>',lambda e:setattr(self,'drag',None))
        ttk.Label(root,text='Blue: travelled | Green: route/stop | Orange: robot | Purple: room label | Red: door | HOME: teal | Wheel: zoom | Right-drag: pan | Simulation only',padding=8).pack(fill='x')
        root.protocol('WM_DELETE_WINDOW',self.close)
        self.refresh_task()
        root.after(35,self.tick)

    def screen(self,p):
        xmin,xmax,ymin,ymax=DATA['bounds']; s,ox,oy=self.transform
        return ox+(p[0]-xmin)*s,oy+(ymax-p[1])*s

    def world(self,x,y):
        xmin,xmax,ymin,ymax=DATA['bounds']; s,ox,oy=self.transform
        return xmin+(x-ox)/s,ymax-(y-oy)/s

    def resize(self,event=None):
        xmin,xmax,ymin,ymax=DATA['bounds']
        w,h=max(1,self.canvas.winfo_width()),max(1,self.canvas.winfo_height())
        if event is not None: w,h=max(1,event.width),max(1,event.height)
        s=min(w/(xmax-xmin),h/(ymax-ymin))*self.zoom
        cx,cy=self.center or ((xmin+xmax)/2,(ymin+ymax)/2)
        ox=w/2-(cx-xmin)*s; oy=h/2-(ymax-cy)*s
        self.transform=(s,ox,oy)
        iw,ih=self.image.size
        extent=((-ox)/s*iw/(xmax-xmin),(-oy)/s*ih/(ymax-ymin),
                (w-ox)/s*iw/(xmax-xmin),(h-oy)/s*ih/(ymax-ymin))
        view=self.image.transform((w,h),Image.Transform.EXTENT,extent,
                                  resample=Image.Resampling.BILINEAR,fillcolor='#e6e9ed')
        self.photo=ImageTk.PhotoImage(view)
        self.canvas.delete('all'); self.canvas.create_image(0,0,anchor='nw',image=self.photo)
        self.draw()

    def fit_map(self):
        self.zoom=1.; self.center=None; self.resize()

    def wheel(self,event,direction=None):
        direction=direction if direction is not None else (1 if event.delta>0 else -1)
        px,py=self.world(event.x,event.y)
        old=self.zoom; self.zoom=max(1.,min(8.,old*(1.25 if direction>0 else .8)))
        if self.zoom==1.: self.fit_map(); return
        new_s=self.transform[0]*self.zoom/old
        w,h=self.canvas.winfo_width(),self.canvas.winfo_height()
        self.center=(px-(event.x-w/2)/new_s,py+(event.y-h/2)/new_s)
        self.resize()

    def pan_start(self,event):
        self.drag=(event.x,event.y,self.world(self.canvas.winfo_width()/2,self.canvas.winfo_height()/2))

    def pan_move(self,event):
        if self.drag is None: return
        x,y,(cx,cy)=self.drag; s=self.transform[0]
        self.center=(cx-(event.x-x)/s,cy+(event.y-y)/s); self.resize()

    def draw(self):
        c=self.canvas;c.delete('overlay')
        def line(points,color,width,dash=None):
            if len(points)>1:
                xy=[v for p in points for v in self.screen(p)]
                c.create_line(*xy,fill=color,width=width,dash=dash,capstyle='round',joinstyle='round',tags='overlay')
        def marker(p,color,r):
            x,y=self.screen(p); c.create_oval(x-r,y-r,x+r,y+r,fill=color,outline='white',width=2,tags='overlay')
        if self.network.get():
            for a,b in EDGES: line([NODES[a],NODES[b]],'#9aa6b5',1,(4,4))
        line(self.nav.trail,'#2267cc',3)
        if self.nav.position is not None:
            line([self.nav.position]+self.nav.remaining,'#007c91' if self.nav.state=='RETURNING_HOME' else '#07945e',4)
        if self.nav.home is not None:
            x,y=self.screen(self.nav.home)
            c.create_rectangle(x-8,y-8,x+8,y+8,fill='#007c91',outline='white',width=2,tags='overlay')
            c.create_text(x,y-17,text='HOME',fill='#006778',font=('TkDefaultFont',10,'bold'),tags='overlay')
        if self.show_doors.get():
            for d in DOORS.values():
                marker(d['door'],'#d62828',3); marker(d['stop'],'#07945e',3)
        if self.calibration is not None:
            r=ROOMS[self.calibration]; marker((r['x'],r['y']),'#a33bb8',6)
            for p,col in zip(self.cal_points,['#d62828','#07945e']): marker(p,col,6)
        selected=ROOM_LABELS.get(self.room.get().strip())
        if self.calibration is None and selected in DOORS:
            d=DOORS[selected]; marker(d['door'],'#d62828',5); marker(d['stop'],'#07945e',6)
        if self.nav.target is not None:
            if self.nav.target in DOORS: marker(DOORS[self.nav.target]['door'],'#d62828',5)
            r=ROOMS[self.nav.target]; marker((r['x'],r['y']),'#a33bb8',5)
            marker(self.nav.goal,'#07945e',7)
        if self.nav.position is not None: marker(self.nav.position,'#f0a025',8)

    def invalidate_camera(self):
        if self.cancel is not None: self.cancel.set()
        self.pending=None

    def refresh_task(self):
        label = next((k for k,v in ROOM_LABELS.items() if v==self.nav.target), 'HOME' if self.nav.state in ('RETURNING_HOME','AT_HOME') else '-')
        entrance = DOORS.get(self.nav.target,{}).get('shared_with')
        if DOORS.get(self.nav.target,{}).get('entrance_type')=='shared_open_access': entrance='2270 / 2271 (open access)'
        self.task.set(f'{self.nav.state} | {label}'+(f' | Entrance: {entrance}' if entrance else ''))
        labels={v:k for k,v in ROOM_LABELS.items()}
        self.queue_text.set(f'Queue ({len(self.nav.waiting)}): '+(' -> '.join(labels[i] for i in self.nav.waiting) or 'empty'))
        self.history_text.set(f'Completed ({len(self.nav.completed)}): '+(' -> '.join(labels[i] for i in self.nav.completed[-12:]) or 'none'))

    def save_home(self):
        try:
            tmp=self.home_file.with_suffix('.tmp')
            tmp.write_text(json.dumps({'home':self.nav.home}),encoding='utf-8')
            tmp.replace(self.home_file)
            return 'HOME saved.'
        except OSError as e:
            return f'HOME set for this session only: {e}'

    def set_home(self):
        if self.calibration is not None: self.status.set('Finish or cancel calibration first.'); return
        try: self.nav.set_home()
        except ValueError as e: self.status.set(str(e)); return
        self.status.set(self.save_home()); self.refresh_task(); self.draw()

    def refresh_door_status(self):
        i=ROOM_LABELS.get(self.room.get().strip())
        shared=DOORS.get(i,{}).get('shared_with')
        if DOORS.get(i,{}).get('entrance_type')=='shared_open_access': shared='2270 / 2271 (open access)'
        self.door_status.set(f'{len(DOORS)}/{len(ROOMS)} CAD goals | '+(f'Shared entrance: {shared}' if shared else 'Door detected' if i in DOORS else 'No direct route'))

    def show_report(self):
        report=json.loads((ROOT/'door_detection_report.json').read_text(encoding='utf-8'))
        missing=[k for k,v in report['rooms'].items() if k not in ROOM_LABELS or ROOM_LABELS[k] not in DOORS]
        messagebox.showinfo('Automatic door detection',f"{len(DOORS)}/{len(ROOMS)} goals automatically detected.\n\nNo direct route / unresolved:\n"+', '.join(missing)+'\n\nSimulation only. See door_detection_report.json for sources.')

    def next_room(self):
        if self.calibration is not None:
            self.status.set('Finish or cancel the current calibration first.'); return
        labels=[label for label,i in ROOM_LABELS.items() if i not in DOORS]
        if not labels:
            self.batch=False; self.status.set('All room door goals are saved.'); return
        self.room.set(labels[0]); self.calibrate(batch=True)

    def undo_point(self):
        if self.calibration is None or not self.cal_points: return
        self.cal_points.pop()
        self.status.set('1/3: Click the door opening center.' if not self.cal_points else '2/3: Click the stop outside the door.')
        self.draw()

    def calibrate(self,batch=False):
        if self.calibration is not None:
            self.status.set('Finish or cancel the current calibration first.'); return
        i=ROOM_LABELS.get(self.room.get().strip())
        if i is None: self.status.set('Select a room first.'); return
        if self.nav.state not in ('IDLE','AT_HOME','STOPPED') or self.nav.waiting:
            self.status.set('STOP before changing the map.'); return
        if self.nav.position is not None and project(self.nav.position,base=True)[3]>1e-5:
            self.status.set('Place the robot on the main corridor before changing the map.'); return
        self.calibration=i; self.cal_points=[]; self.network.set(True); self.batch=batch
        self.combo.configure(state='disabled')
        self.center=(ROOMS[i]['x'],ROOMS[i]['y']); self.zoom=3.; self.resize()
        self.status.set(f'{self.room.get()} | 1/3: Click the center of the actual door opening on the plan.')

    def cancel_calibration(self):
        self.calibration=None; self.cal_points=[]; self.batch=False
        self.combo.configure(state='normal')
        self.status.set('Door calibration cancelled.'); self.fit_map()

    def place(self,event):
        if self.calibration is not None:
            s,ox,oy=self.transform; xmin,xmax,ymin,ymax=DATA['bounds']
            p=(xmin+(event.x-ox)/s,ymax-(event.y-oy)/s)
            if not (xmin<=p[0]<=xmax and ymin<=p[1]<=ymax): return
            if len(self.cal_points)==0:
                self.cal_points.append(p); self.status.set('2/3: Click the desired robot stop outside the door, in free space.')
            elif len(self.cal_points)==1:
                self.cal_points.append(p); self.status.set('3/3: Click the nearby dashed corridor line. Ensure the straight segment to the stop is clear.')
            else:
                if project(p,base=True)[3]>1000:
                    self.status.set('Click closer to the main corridor line.'); return
                anchor=project(p,base=True)[2]
                if not messagebox.askyesno('Confirm door route','Is the straight connection from the corridor to the selected stop clear of walls, door swing and stairs?'):
                    return
                try: save_door(self.calibration,*self.cal_points,anchor)
                except (OSError,ValueError) as e: self.status.set(str(e)); return
                self.calibration=None; self.cal_points=[]; self.combo.configure(state='normal'); self.refresh_door_status()
                self.status.set(f'{self.room.get()} door saved. {len(DOORS)}/{len(ROOMS)} complete.')
                if self.batch:
                    self.next_room()
                    if self.calibration is None: self.fit_map()
                    return
            self.draw(); return
        if self.nav.state not in ('IDLE','AT_HOME','STOPPED'):
            self.status.set('Stop first to manually relocate the robot.'); return
        s,ox,oy=self.transform; xmin,xmax,ymin,ymax=DATA['bounds']
        p=(xmin+(event.x-ox)/s,ymax-(event.y-oy)/s)
        if project(p,base=True)[3]>3500:
            self.status.set('Click closer to a corridor centerline.'); return
        first=self.nav.home is None
        self.invalidate_camera(); self.nav.place(p)
        self.status.set(self.save_home() if first else 'Robot placed. Fixed HOME unchanged.')
        self.refresh_task(); self.draw()

    def send(self):
        if self.calibration is not None: self.status.set('Finish or cancel calibration first.'); return
        label=self.room.get().strip()
        if label not in ROOM_LABELS:
            self.status.set('Choose an exact room from the list (2780 needs a location).'); return
        old=self.nav.state
        try: started=self.nav.new_task(ROOM_LABELS[label])
        except ValueError as e: self.status.set(str(e)); return
        if started is None:
            self.status.set(f'Room {label} queued. Current task is unchanged.')
        else:
            self.invalidate_camera(); self.observed.set(''); self.last=time.monotonic()
            self.status.set('HOME return interrupted; routing from current position.' if old=='RETURNING_HOME' else 'A* route calculated from current position.')
        self.refresh_task(); self.draw()

    def stop(self):
        if self.calibration is not None: self.cancel_calibration()
        self.invalidate_camera(); self.nav.stop(); self.refresh_task()
        self.status.set('STOPPED. Pending tasks cleared; no automatic HOME return.'); self.draw()

    def arrived(self):
        code=verification_code(self.nav.target); self.refresh_task(); self.observed.set('')
        requested=ROOMS[self.nav.target]['code']
        location=(f'At shared entrance {code} for room {requested} (not inside the room).' if code!=requested else 'At saved door stop (not field verified).')
        if DOORS[self.nav.target].get('entrance_type')=='shared_open_access':
            location='At shared open access for 2270 / 2271. External sign not confirmed; OCR must actually see the requested code.'
        location+=f' Expected sign: {code}.'
        self.verification_mode=self.mode.get()
        if self.verification_mode=='Simulation':
            self.status.set(location+' Enter the observed room code and press Submit OCR (simulated).')
        else:
            self.pending=(self.nav.task_id,self.source.get().strip(),code)
            self.status.set(location+' Checking camera OCR; camera must face the sign.')
            self.launch_pending()

    def launch_pending(self):
        if self.pending is None or (self.worker and self.worker.is_alive()): return
        task_id,source,code=self.pending; self.pending=None
        cancel=threading.Event(); self.cancel=cancel
        def work():
            try: detected,detail=camera_verify(source,code,cancel)
            except Exception as e: detected,detail=None,f'Camera/OCR error: {e}'
            self.results.put((task_id,detected,detail))
        self.worker=threading.Thread(target=work,daemon=True); self.worker.start()

    def finish_verification(self,task_id,code,detail):
        if task_id != self.nav.task_id or self.nav.state != 'VERIFYING': return
        expected=verification_code(self.nav.target)
        requested=ROOMS[self.nav.target]['code']
        try: accepted=self.nav.verify(task_id,code)
        except ValueError as e:
            self.status.set(f'Room {expected} confirmed. {e} Retry route or STOP.')
            self.refresh_task(); self.draw(); return
        if accepted:
            prefix='Simulated verification: ' if self.verification_mode=='Simulation' else ''
            outcome={
                'MOVING':' Next queued task started.',
                'RETURNING_HOME':' Queue empty; returning HOME.',
                'AT_HOME':' Queue empty; at HOME.',
                'UNVERIFIED':' Not confirmed; stays stopped. Retry verification or STOP.'
            }[self.nav.state]
            shared_detail=f' Shared entrance {expected} verified for room {requested}.' if expected!=requested and self.nav.state!='UNVERIFIED' else ''
            self.status.set(prefix+detail+shared_detail+outcome)
            self.last=time.monotonic(); self.refresh_task(); self.draw()

    def manual_verify(self):
        if self.verification_mode!='Simulation' or self.nav.state!='VERIFYING': return
        value=self.observed.get().strip(); self.finish_verification(self.nav.task_id,value,f'Observed code: {value or "none"}.')

    def retry(self):
        if self.nav.state not in ('UNVERIFIED','BLOCKED'): return
        try: self.nav.retry()
        except ValueError as e: self.status.set(str(e)); return
        if self.nav.state=='VERIFYING': self.arrived()
        else:
            self.status.set('Route retry successful.'); self.refresh_task(); self.draw()

    def tick(self):
        now=time.monotonic(); dt=min(.2,now-self.last); self.last=now
        if self.nav.state in ('MOVING','RETURNING_HOME'):
            if self.nav.advance(self.speed.get()*dt):
                if self.nav.state=='VERIFYING': self.arrived()
                else:
                    self.refresh_task(); self.status.set('At HOME. Waiting for a new task.')
            self.draw()
        while True:
            try: result=self.results.get_nowait()
            except queue.Empty: break
            self.finish_verification(*result)
        self.launch_pending(); self.root.after(35,self.tick)

    def close(self):
        self.invalidate_camera(); self.root.destroy()


if __name__=='__main__':
    root=tk.Tk(); App(root); root.mainloop()
