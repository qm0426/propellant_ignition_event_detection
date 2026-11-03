"""LoG scale, intensity-unit, duplicate and background regression checks."""
import unittest
import numpy as np
from PIL import Image
from ignition_revision.detection import classical_candidates, log_nms


class LogDetectorTests(unittest.TestCase):
    def config(self, **kwargs):
        return dict(polarity='dark', intensity_scale='unit', sigmas=[2.,3.,4.,6.,8.,12.,16.,24.,32.,48.],
                    response_threshold=.02, nms_iou=.3, **kwargs)

    def image(self):
        y,x=np.mgrid[:256,:320]
        gray=np.full(x.shape,220.)
        for cx,cy,sigma in [(55,60,4),(155,65,12),(210,180,28)]:
            gray-=160*np.exp(-((x-cx)**2+(y-cy)**2)/(2*sigma*sigma))
        return Image.fromarray(np.clip(gray,0,255).astype('uint8'))

    def test_multiple_sizes_have_one_center_and_recover_large_scale(self):
        boxes=classical_candidates(self.image(),self.config(),'log')
        self.assertEqual(len(boxes),3)
        for cx,cy,sigma in [(55,60,4),(155,65,12),(210,180,28)]:
            near=[b for b in boxes if abs((b['x1']+b['x2'])/2-cx)<=2 and abs((b['y1']+b['y2'])/2-cy)<=2]
            self.assertEqual(len(near),1)
            self.assertLess(abs(np.log(near[0]['sigma']/sigma)),.4)

    def test_intensity_threshold_units_are_equivalent(self):
        unit=classical_candidates(self.image(),self.config(),'log')
        cfg=self.config();cfg.update(intensity_scale='raw_255',response_threshold=.02*255)
        raw=classical_candidates(self.image(),cfg,'log')
        self.assertEqual([(b['x1'],b['y1'],b['sigma']) for b in unit],[(b['x1'],b['y1'],b['sigma']) for b in raw])
        np.testing.assert_allclose([b['native_score']*255 for b in unit],[b['native_score'] for b in raw],rtol=1e-4)

    def test_uniform_background_has_no_candidates(self):
        self.assertEqual(classical_candidates(Image.new('L',(128,128),128),self.config(),'log'),[])

    def test_cross_scale_nms_preserves_distinct_neighbor(self):
        def box(x,size,score):
            return dict(x1=x,y1=0,x2=x+size,y2=size,native_score=score,sigma=size)
        a,b,c=box(0,20,.5),box(1,22,.4),box(60,20,.3)
        self.assertEqual(log_nms([b,c,a],.3),[a,c])

    def test_rejects_invalid_scale_units_and_threshold(self):
        for key,value in [('intensity_scale','unknown'),('response_threshold',0),('sigmas',[4,2]),('nms_iou',0)]:
            cfg=self.config();cfg[key]=value
            with self.assertRaises(ValueError):classical_candidates(self.image(),cfg,'log')


if __name__=='__main__':unittest.main()
