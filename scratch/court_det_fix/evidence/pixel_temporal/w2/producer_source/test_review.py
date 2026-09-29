"""Real-CSV arithmetic tests plus constructed image tests. No real pixels accessed."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import analyze_fresh as af
import build_pair_atlas as bp


def synthetic_interval(gray, identity, start, end):
    start,end=np.asarray(start,float),np.asarray(end,float)
    u=end-start;normal=np.array([-u[1],u[0]])/np.linalg.norm(u)
    stations=start+np.linspace(0,1,24)[:,None]*u
    xy=stations[:,None]+np.array([-4.,-2.,0.,2.,4.])[None,:,None]*normal
    xm,xp=xy-6*normal,xy+6*normal
    c,m,p=[bp.sample(gray,q) for q in [xy,xm,xp]]
    available=c[1]&m[1]&p[1]
    c1,c2=c[0]-m[0],c[0]-p[0]
    minimum=np.minimum(c1,c2);passed=available&(minimum>=10)
    return {'interval':identity,'visible':True,
            'tested_sample_coordinates_working_px':xy.tolist(),
            'minus_side_coordinates_working_px':xm.tolist(),
            'plus_side_coordinates_working_px':xp.tolist(),
            'contrast_centre_minus_minus_side':c1.tolist(),
            'contrast_centre_minus_plus_side':c2.tolist(),
            'minimum_contrast':minimum.tolist(),'available':available.tolist(),
            'pass_by_shift':passed.tolist(),'interval_pass':bool(passed.any(axis=1).mean()>=.4)}


class RealCSV(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows=af.read_rows(ROOT/'inputs/real_interval_probe.csv')
        cls.result=af.analyse(cls.rows)
    def test_bounds_and_missingness(self):
        v=self.result['validation']
        self.assertEqual((v['rows'],v['measured_rows'],v['unavailable_rows']),(48,41,7))
        self.assertEqual((v['unknown_offset_tests'],v['rows_with_run_bound_gap']),(44,3))
    def test_dropin_fails_am2_contrary_example(self):
        summary={r['candidate_id']:r for r in self.result['candidates']}
        self.assertEqual(summary['184:4123']['dropin_run_profile_lower']['exact'],'1/2')
        self.assertEqual(summary['30:33']['dropin_run_profile_lower']['exact'],'5/11')
        for row in summary.values():
            self.assertEqual(row['dropin_run_profile_lower'],row['dropin_run_profile_upper'])
    def test_same_five_markings(self):
        false,usable=self.result['same_five_scene19_markings']
        self.assertEqual(false['run_lower_fraction']['exact'],'7/30')
        self.assertEqual(usable['run_lower_fraction']['exact'],'11/12')
        self.assertEqual(false['intervals'],usable['intervals'])
    def test_modified_csv_blocked(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'changed.csv'
            p.write_bytes((ROOT/'inputs/real_interval_probe.csv').read_bytes()+b'\n')
            with self.assertRaisesRegex(ValueError,'checksum'):
                af.read_rows(p)


class SyntheticPixels(unittest.TestCase):
    def setUp(self):
        self.gray=np.full((540,960),40.,np.float32)
        self.gray[202,:]=200.
        self.pair=[synthetic_interval(self.gray,10,[100,200],[500,200]),
                   synthetic_interval(self.gray,11,[130,204],[470,204])]
    def test_same_ridge_passes_both_profiles(self):
        for row in self.pair:
            self.assertTrue(bp.verify_pixel_trace(self.gray,row)['passed'])
            metrics=bp.interval_probe(row)
            self.assertEqual(metrics['run_lower_stations'],24)
    def test_pixel_mismatch_blocks(self):
        modified=copy.deepcopy(self.pair[0])
        modified['contrast_centre_minus_minus_side'][0][0]+=1
        with self.assertRaisesRegex(ValueError,'mismatch'):
            bp.verify_pixel_trace(self.gray,modified)
    def test_shared_chart_does_not_align_station_indices(self):
        strip=bp.joint_strip(self.gray,self.pair)
        self.assertAlmostEqual(strip['t'][0],30.)
        self.assertAlmostEqual(strip['t'][-1],370.)
        np.testing.assert_allclose(strip['z_b'],4.)
        # One ridge remains one ridge in the shared chart.
        column=strip['intensity'][:,len(strip['t'])//2]
        self.assertAlmostEqual(strip['z'][np.nanargmax(column)],2.)
    def test_two_ridges_remain_distinct_in_raw_strip(self):
        two=np.full_like(self.gray,40.);two[200,:]=200.;two[204,:]=200.
        strip=bp.joint_strip(two,self.pair)
        column=strip['intensity'][:,len(strip['t'])//2]
        maxima=strip['z'][column==200.]
        np.testing.assert_array_equal(maxima,np.array([0.,4.]))
    def test_unknown_positions_have_explicit_mask(self):
        pair=[synthetic_interval(self.gray,6,[100,1],[500,1]),
              synthetic_interval(self.gray,7,[100,3],[500,3])]
        strip=bp.joint_strip(self.gray,pair)
        self.assertTrue((~strip['available']).any())
        self.assertTrue(np.isnan(strip['intensity'][~strip['available']]).all())
    def test_render_smoke_and_native_scaling(self):
        working=cv2.cvtColor(self.gray.astype(np.uint8),cv2.COLOR_GRAY2BGR)
        native=np.repeat(np.repeat(working,2,axis=0),2,axis=1)
        candidate={'existing_ruling':'CONSTRUCTED TEST ONLY','trace':{'intervals':self.pair}}
        target={**bp.TARGETS[0],'label':'CONSTRUCTED shared ridge; not real Am2 pixels'}
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'pair'
            result=bp.render_pair(native,candidate,target,out)
            self.assertEqual(result['native_size'],[1920,1080])
            self.assertEqual(result['native_per_working_xy'],[2.,2.])
            for file in ['context_raw.png','native_pair_raw.png','native_pair_overlay.png',
                         'shared_strip_raw.png','shared_strip_overlay.png','pair_trace.json','shared_strip.npz']:
                self.assertTrue((out/file).is_file())
            self.assertTrue(all(r['passed'] for r in result['pixel_replay_checks']))

if __name__=='__main__':
    unittest.main(verbosity=2)
