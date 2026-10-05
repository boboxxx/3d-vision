import importlib.util
from pathlib import Path
import tempfile
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('audit_liga',ROOT/'scripts/audit_liga_run.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class ArtifactTests(unittest.TestCase):
    def test_camera_dimensions_and_text_precision(self):
        anno=dict(name=np.array(['Car']),alpha=np.array([.1234567]),
            bbox=np.array([[10.,20.,100.,200.]]),dimensions=np.array([[3.9,1.56,1.6]]),
            location=np.array([[1.,2.,20.]]),rotation_y=np.array([.5]),score=np.array([.98765432]))
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'000001.txt'
            line='Car -1 -1 0.1235 10 20 100 200 1.56 1.6 3.9 1 2 20 .5 .98765432\n'
            path.write_text(line);module.audit_prediction(anno,path)
            # Same positive dimensions in wrong l/h/w convention must be caught.
            path.write_text(line.replace('1.56 1.6 3.9','3.9 1.56 1.6'))
            with self.assertRaises(AssertionError):module.audit_prediction(anno,path)


if __name__=='__main__':unittest.main()
