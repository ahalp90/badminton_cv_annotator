"""Small tests for the audit maths and the actual returned four-interval trace."""
import json
import unittest
from pathlib import Path

import cv2
import numpy as np

from verify_return import bilinear, footprint, inside, longest_run

ROOT = Path(__file__).resolve().parent


class MathTests(unittest.TestCase):
    def test_linear_field(self):
        y,x=np.indices((10,10)); gray=3*x+5*y+7
        coords=np.array([[[1.25,2.5],[6.75,4.125]]],dtype=np.float32)
        np.testing.assert_array_equal(bilinear(gray,coords),3*coords[...,0]+5*coords[...,1]+7)

    def test_replicate_border_and_availability_are_separate(self):
        gray=np.array([[0.,10.],[20.,30.]])
        coords=np.array([[[1.5,.5],[-.25,.5]]],dtype=np.float32)
        np.testing.assert_allclose(bilinear(gray,coords),[[20.,10.]])
        self.assertEqual(inside(coords,gray.shape).tolist(),[[True,False]])

    def test_shared_footprint_does_not_mean_same_weights(self):
        gray=np.arange(16,dtype=float).reshape(4,4)
        a=footprint(gray,np.array([1.2,1.3])); b=footprint(gray,np.array([1.8,1.7]))
        self.assertEqual([(r['x'],r['y']) for r in a],[(r['x'],r['y']) for r in b])
        self.assertNotEqual([r['weight'] for r in a],[r['weight'] for r in b])
        self.assertAlmostEqual(sum(r['weight'] for r in a),1.)

    def test_run_transition_and_gap(self):
        mask=np.zeros((24,5),bool);mask[:,2]=True
        self.assertEqual(longest_run(mask),24)
        mask[12]=False
        self.assertEqual(longest_run(mask),12)
        mask=np.zeros((24,5),bool);mask[::2,0]=True;mask[1::2,4]=True
        self.assertEqual(longest_run(mask),1)


class RealPacketTests(unittest.TestCase):
    def test_returned_four_masks_and_runs(self):
        expected={10:16,11:12,6:4,7:3}
        total=0
        for name in ['184_4123','30_33']:
            p=ROOT/'evidence'/name
            trace=json.loads((p/'pair_trace.json').read_text())
            frame=cv2.imread(str(p/'context_raw.png'))
            gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY).astype(float)
            for r in trace['original_intervals']:
                sets=[np.array(r[k]) for k in ['tested_sample_coordinates_working_px',
                                               'minus_side_coordinates_working_px','plus_side_coordinates_working_px']]
                values=[bilinear(gray,xy) for xy in sets]
                available=np.logical_and.reduce([inside(xy,gray.shape) for xy in sets])
                passed=available&(np.minimum(values[0]-values[1],values[0]-values[2])>=10)
                np.testing.assert_array_equal(passed,r['pass_by_shift'])
                self.assertEqual(longest_run(passed),expected[r['interval']])
                self.assertTrue(available.all());total+=1
        self.assertEqual(total,4)

    def test_same_pixel_footprints_in_approved_contrary_example(self):
        p=ROOT/'evidence/30_33'; t=json.loads((p/'pair_trace.json').read_text())
        a,b=t['original_intervals']
        gray=cv2.cvtColor(cv2.imread(str(p/'context_raw.png')),cv2.COLOR_BGR2GRAY).astype(float)
        for key in ['tested_sample_coordinates_working_px','minus_side_coordinates_working_px','plus_side_coordinates_working_px']:
            fa=footprint(gray,np.array(a[key][1][2]))
            fb=footprint(gray,np.array(b[key][2][1]))
            self.assertEqual([(r['x'],r['y']) for r in fa],[(r['x'],r['y']) for r in fb])
        self.assertTrue(a['pass_by_shift'][1][2]);self.assertTrue(b['pass_by_shift'][2][1])


if __name__=='__main__':
    unittest.main(verbosity=2)
