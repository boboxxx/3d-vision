"""Meaningful synthetic geometry/transport checks, no native/AP claim."""
from pathlib import Path
import hashlib,json,time,unittest
import numpy as np
from proxy import epipolar_proxy,groups,group_noise


class GeometryChecks(unittest.TestCase):
    def test_public_crop_excludes_partial_padding_blocks_at_both_match_endpoints(self):
        left=np.random.default_rng(48).normal(size=(32,16,256)).astype(np.float32)
        right=np.roll(left,-8,axis=2)
        result=epipolar_proxy(left,right,focal_baseline=100.,valid_image_shape=(15,247))
        for view in ('left','right'):
            self.assertFalse(result[view]['valid'][3,:].any())
            self.assertFalse(result[view]['valid'][:,61:].any())
            self.assertEqual(result[view]['probability'][:,3,:].sum(),0)
            self.assertEqual(result[view]['probability'][:,:,61:].sum(),0)
        with self.assertRaises(ValueError):epipolar_proxy(left,right,focal_baseline=100.,valid_image_shape=np.ones((16,256),bool))

    def test_known_stereo_disparity_both_signs(self):
        left=np.random.default_rng(40).normal(size=(32,16,256)).astype(np.float32)
        right=np.zeros_like(left);right[:,:,:-8]=left[:,:,8:]
        result=epipolar_proxy(left,right,focal_baseline=100.)
        self.assertTrue(result['left']['valid'][:,48:62].all())
        self.assertTrue(result['right']['valid'][:,:14].all())
        np.testing.assert_array_equal(np.argmax(result['left']['probability'][:,:,48:62],axis=0),np.ones((4,14),dtype=int))
        np.testing.assert_array_equal(np.argmax(result['right']['probability'][:,:,:14],axis=0),np.ones((4,14),dtype=int))
        self.assertFalse(result['left']['valid'][:,0].any())
        self.assertFalse(result['right']['valid'][:,-1].any())
        self.assertTrue(np.all(result['left']['probability'][:, :,0]==0))
        self.assertEqual(result['left']['support_count'][0,55],48)
        p=result['left']['probability'][:,0,55]
        z=np.array([100/(4*d) for d in range(1,49)])
        expected_mean=sum(float(p[i])*float(z[i]) for i in range(48))
        expected_var=sum(float(p[i])*(float(z[i])-expected_mean)**2 for i in range(48))
        self.assertAlmostEqual(result['left']['depth_mean'][0,55],expected_mean,places=12)
        self.assertAlmostEqual(result['left']['depth_variance'][0,55],expected_var,places=12)

    def test_nontexture_zero_and_target_inputs(self):
        for image in [np.zeros((32,16,256),dtype=np.float32),np.ones((32,16,256),dtype=np.float64)]:
            result=epipolar_proxy(image,image,focal_baseline=100.)
            for view in ['left','right']:
                self.assertFalse(result[view]['valid'].any());self.assertTrue(np.isnan(result[view]['entropy']).all())
                self.assertTrue(np.isnan(result[view]['depth_variance']).all())
        with self.assertRaises(ValueError):epipolar_proxy(np.ones((7,8,8)),np.ones((7,8,8)),focal_baseline=100.)
        with self.assertRaises(ValueError):epipolar_proxy(np.ones((32,16,256)),np.ones((32,16,256)),focal_baseline=np.ones(7))
        with self.assertRaises(ValueError):epipolar_proxy(np.ones((32,16,256)),np.ones((32,16,252)),focal_baseline=100.)

    def test_every_native_symbol_in_exactly_one_group(self):
        mapping=groups([(1,2,20,1248),(1,2,20,1248),(1,8,20,156)])
        self.assertEqual(len(mapping),62400);np.testing.assert_array_equal(np.unique(mapping),np.arange(32))
        self.assertTrue(np.all(np.bincount(mapping,minlength=32)>0))
        np.testing.assert_array_equal(mapping[:24960],mapping[24960:49920])
        self.assertEqual(mapping[0],0);self.assertEqual(mapping[1247],7);self.assertEqual(mapping[24959],31)
        self.assertEqual(mapping[49920],0);self.assertEqual(mapping[-1],31)
        # Four consecutive real-channel pairs in each appearance cell share one cell label.
        tail=mapping[49920:].reshape(20,156,4)
        self.assertTrue(np.all(tail==tail[:,:,:1]))

    def test_noise_mask_power_rng_and_global_isolation(self):
        mapping=groups([(1,2,20,1248),(1,2,20,1248),(1,8,20,156)])
        state=np.random.get_state();a=np.random.Generator(np.random.PCG64(2801));b=np.random.Generator(np.random.PCG64(2801))
        power=0.;degrees=0
        for _ in range(4):
            x,record=group_noise(mapping,17,rng=a);y,other=group_noise(mapping,17,rng=b)
            np.testing.assert_array_equal(x,y);self.assertEqual(record,other)
            self.assertTrue(np.all(x[mapping!=17]==0))
            self.assertEqual(record['attempted_complex_uses'],62400)
            power+=float(np.sum(x[mapping==17].astype(float)**2))/.05
            degrees+=2*int(np.count_nonzero(mapping==17))
        self.assertLess(abs(power-degrees),8*np.sqrt(2*degrees))
        self.assertTrue(all(np.array_equal(x,y) for x,y in zip(state,np.random.get_state())))


def main():
    root=Path(__file__).resolve().parents[3]
    out=root/'data/engineering/geometry-risk-local-CPU-001.json'
    if out.exists():raise RuntimeError('unique evidence required')
    paths=list(Path(__file__).parent.glob('*.py'))+[root/'experiments/geometry-risk/protocol-001.md']
    hashes={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    started=time.time();result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(GeometryChecks))
    record=dict(state='passed' if result.wasSuccessful() else 'failed',tests=result.testsRun,
                scope='synthetic_geometry_CPU_arithmetic_only_no_native_calibration_AP',started_at_unix=started,
                finished_at_unix=time.time(),numpy=np.__version__,source_sha256=hashes)
    assert hashes=={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    with out.open('x') as f:json.dump(record,f,indent=2)
    if not result.wasSuccessful():raise SystemExit(1)


if __name__=='__main__':main()
