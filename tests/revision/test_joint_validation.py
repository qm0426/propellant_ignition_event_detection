import unittest
from types import SimpleNamespace
import numpy as np
import pandas as pd
from ignition_revision.common import load_legacy
from ignition_revision.tracking import advance_track_state
from ignition_revision.evaluation import evaluate_fixed,evaluate_events


class JointValidationTests(unittest.TestCase):
    def test_missing_frames_integrate_elapsed_time_once(self):
        mod=load_legacy('ot_kalman_tracker.py')
        kf=mod.KalmanBoxFilter(10,20,4,4);kf.x[2]=2
        tr=SimpleNamespace(start_frame=10,last_frame=10,kf=kf,age=1,misses=0)
        for frame in [11,12,13]:advance_track_state(tr,frame)
        self.assertAlmostEqual(float(kf.x[0]),16)
        self.assertEqual(tr.last_frame,10)
        self.assertEqual(tr.misses,3)
        self.assertEqual(tr.age,4)
        before=kf.x.copy();advance_track_state(tr,13)
        np.testing.assert_array_equal(kf.x,before)
        with self.assertRaises(ValueError):advance_track_state(tr,12)

    def test_skipped_calls_use_actual_elapsed_interval(self):
        kf=load_legacy('ot_kalman_tracker.py').KalmanBoxFilter(10,20,4,4);kf.x[2]=2
        tr=SimpleNamespace(start_frame=10,last_frame=10,kf=kf,age=1,misses=0)
        advance_track_state(tr,13)
        self.assertAlmostEqual(float(kf.x[0]),16)

    def test_fixed_time_miss_counts_both_unmatched_events_not_negative_alarm(self):
        labels=pd.DataFrame([dict(recording_id='r',track_id=i,ref_object_id=str(i),has_ignition=i<2,onset_frame=10 if i<2 else None) for i in range(4)])
        pred=pd.DataFrame([dict(recording_id='r',track_id=i,has_ignition=i<3,onset_frame=[13,14,20,None][i]) for i in range(4)])
        raw,_=evaluate_fixed(pred,labels)
        self.assertEqual([raw[k] for k in ['TP','FP','FN','TN']],[2,1,0,1])
        m,rows= evaluate_fixed(pred,labels,tolerance=3)
        self.assertEqual([m[k] for k in ['TP','FP','FN']],[1,2,1])
        self.assertIsNone(m['false_positive_rate'])
        self.assertEqual(m['negative_object_FAR'],.5)
        self.assertEqual(m['existence_onset_mae'],3.5)
        self.assertEqual(m['onset_mae'],3)
        self.assertEqual(sum(r['match_status']=='FN' for r in rows),1)

    def test_time_gate_precedes_one_to_one_duplicate_matching(self):
        reference=pd.DataFrame([dict(recording_id='r',ref_event_id='e',ref_object_id='a',onset_frame=10)])
        pred=pd.DataFrame([dict(recording_id='r',pred_event_id='bad',track_id=1,onset_frame=50),dict(recording_id='r',pred_event_id='good',track_id=2,onset_frame=12)])
        mapping=pd.DataFrame([dict(recording_id='r',track_id=i,ref_object_id='a') for i in [1,2]])
        m,rows=evaluate_events(pred,reference,mapping,tolerance=3)
        self.assertEqual([m[k] for k in ['TP','FP','FN']],[1,1,0])
        self.assertEqual(next(r for r in rows if r['match_status']=='TP')['pred_event_id'],'good')
        m,_=evaluate_events(pred,reference,mapping,tolerance=1)
        self.assertEqual([m[k] for k in ['TP','FP','FN']],[0,2,1])


if __name__=='__main__':unittest.main()
