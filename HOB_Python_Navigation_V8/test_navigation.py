import unittest, math
import hob_navigation as h
class TestAutoGoals(unittest.TestCase):
    def test_real_goals(self):
        self.assertEqual(len(h.DOORS),92)
        for i,d in h.DOORS.items():
            self.assertLess(abs(math.dist(d['door'],d['stop'])-900),1e-5)
            self.assertEqual(h.room_goal(i)[2],tuple(d['stop']))
    def test_all_pairs(self):
        goals=[h.room_goal(i) for i in h.DOORS]
        for a in goals:
            for b in goals:
                p=h.astar(a,b);self.assertTrue(p);self.assertEqual(p[0],a[2]);self.assertEqual(p[-1],b[2])
    def test_fifo_home(self):
        n=h.Navigator();n.place((6700,357500));home=n.home
        ids=[h.ROOM_LABELS[c] for c in ('2010','2030','2700')]
        for i in ids:n.new_task(i)
        for i in ids:
            self.assertEqual(n.target,i);n.advance(1e9);self.assertEqual(n.position,tuple(h.DOORS[i]['stop']));self.assertEqual(n.state,'VERIFYING');n.verify(n.task_id,h.ROOMS[i]['code'])
        self.assertEqual(n.state,'RETURNING_HOME');n.advance(1e9);self.assertEqual(n.state,'AT_HOME');self.assertEqual(n.position,home)
    def test_wrong_ocr(self):
        n=h.Navigator();n.place((6700,357500));n.new_task(h.ROOM_LABELS['2010']);n.new_task(h.ROOM_LABELS['2030']);n.advance(1e9);n.verify(n.task_id,'2030');self.assertEqual(n.state,'UNVERIFIED');self.assertEqual(len(n.waiting),1)
    def test_shared_goals(self):
        pairs={'2011':'2010','2031':'2030','2032':'2030','2041':'2040','2061':'2060','2231':'2230','2251':'2250'}
        for child,parent in pairs.items():
            i,j=h.ROOM_LABELS[child],h.ROOM_LABELS[parent]
            self.assertEqual(h.DOORS[i]['shared_with'],parent)
            self.assertEqual(h.room_goal(i),h.room_goal(j))
            self.assertEqual(h.verification_code(i),parent)
    def test_shared_ocr_fifo(self):
        n=h.Navigator();n.place((6700,357500));home=n.home
        ids=[h.ROOM_LABELS[c] for c in ('2031','2032','2011')]
        for i in ids:n.new_task(i)
        n.advance(1e9); pos=n.position; old=n.task_id
        n.verify(old,'2031'); self.assertEqual(n.state,'UNVERIFIED');self.assertFalse(n.completed)
        n.retry();self.assertFalse(n.verify(old,'2030'))
        n.verify(n.task_id,'2030');self.assertEqual(n.target,ids[1]);self.assertEqual(n.position,pos)
        n.advance(1e9);self.assertEqual(n.position,pos);self.assertEqual(n.state,'VERIFYING')
        n.verify(n.task_id,'2030');self.assertEqual(n.target,ids[2]);n.advance(1e9);n.verify(n.task_id,'2010')
        self.assertEqual(n.completed,ids);self.assertEqual(n.state,'RETURNING_HOME');n.advance(1e9);self.assertEqual(n.position,home)
    def test_recovered_entrances(self):
        a,b,c=[h.ROOM_LABELS[k] for k in ('2270','2271','2680')]
        self.assertEqual(h.room_goal(a),h.room_goal(b))
        self.assertEqual(h.DOORS[a]['entrance_type'],'shared_open_access')
        self.assertEqual(h.verification_code(a),'2270')
        self.assertEqual(h.verification_code(b),'2271')
        self.assertIn('2F4BB',h.DOORS[c]['source'])
        n=h.Navigator();n.place((6700,357500));home=n.home
        for i in (a,b,c): n.new_task(i)
        for i in (a,b,c):
            self.assertEqual(n.target,i);n.advance(1e9)
            self.assertEqual(n.position,tuple(h.DOORS[i]['stop']))
            n.verify(n.task_id,h.verification_code(i))
        self.assertEqual(n.completed,[a,b,c]);n.advance(1e9);self.assertEqual(n.position,home)
    def test_unresolved(self):
        n=h.Navigator();n.place((6700,357500))
        for i in range(len(h.ROOMS)):
            if i not in h.DOORS:
                with self.assertRaises(ValueError):n.new_task(i)
if __name__=='__main__':unittest.main()
