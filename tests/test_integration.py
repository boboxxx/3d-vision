"""Checks that matter before a full detector/data experiment."""
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import yaml

ROOT = Path(__file__).resolve().parents[1]


class IntegrationTests(unittest.TestCase):
    def test_generated_config_preserves_full_detector(self):
        base = ROOT / "third_party/LIGA-Stereo/configs/stereo/kitti_models/liga.3d-and-bev.yaml"
        if not base.exists():
            self.skipTest("official checkout not fetched")
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "config.yaml"
            subprocess.run([sys.executable, str(ROOT / "scripts/make_liga_config.py"),
                            "--base", str(base), "--output", str(output)],
                           check=True, capture_output=True)
            actual = yaml.safe_load(output.read_text(encoding="utf-8"))
            original = yaml.safe_load(base.read_text(encoding="utf-8"))
            del actual["MODEL"]["BACKBONE_3D"]["SEMANTIC_LINK"]
            self.assertEqual(actual, original)
            subprocess.run([sys.executable,str(ROOT/'scripts/make_liga_config.py'),
                            '--base',str(base),'--output',str(output),'--representation','rgb','--width','40'],
                            check=True,capture_output=True)
            rgb = yaml.safe_load(output.read_text())
            del rgb['MODEL']['RGB_LINK']
            self.assertEqual(rgb,original)
            subprocess.run([sys.executable,str(ROOT/'scripts/make_liga_config.py'),
                            '--base',str(base),'--output',str(output),'--boundary','raw_cost','--student'],
                            check=True,capture_output=True)
            student = yaml.safe_load(output.read_text())
            del student['MODEL']['BACKBONE_3D']['SEMANTIC_LINK']
            del student['MODEL']['BACKBONE_3D']['STUDENT_ENCODER']
            self.assertEqual(student,original)

    def test_boundary_patch_idempotent(self):
        checkout = ROOT / "third_party/LIGA-Stereo"
        if not checkout.exists():
            self.skipTest("official checkout not fetched")
        paths = ["liga/models/backbones_3d_stereo/liga_backbone.py",
                 "liga/models/detectors_stereo/liga.py", "tools/eval_utils/eval_utils.py",
                 "liga/models/dense_heads/anchor_head_template.py"]
        originals = {}
        # Read immutable upstream sources so the test doesn't mutate the checkout.
        for path in paths:
            p = subprocess.run(["git", "-C", str(checkout), "show", "HEAD:" + path],
                               check=True, capture_output=True)
            originals[path] = p.stdout
        spec = importlib.util.spec_from_file_location("patch_liga", ROOT / "scripts/patch_liga.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for path, content in originals.items():
                (root / path).parent.mkdir(parents=True, exist_ok=True)
                (root / path).write_bytes(content)
            self.assertTrue(module.patch(root))
            first = {path: (root/path).read_bytes() for path in paths}
            self.assertFalse(module.patch(root))
            self.assertEqual(first, {path: (root/path).read_bytes() for path in paths})
            backbone = first[paths[0]].decode("utf-8")
            forward = backbone.split("    def forward(self, batch_dict):", 1)[1]
            self.assertLess(forward.index("cost0, left_sem_feat = self.semantic_link"),
                            forward.index("batch_dict['sem_features']"))
            student_spec = importlib.util.spec_from_file_location('patch_student',ROOT/'scripts/patch_student_liga.py')
            student_module = importlib.util.module_from_spec(student_spec)
            student_spec.loader.exec_module(student_module)
            self.assertTrue(student_module.patch(root))
            patched = {path:(root/path).read_bytes() for path in paths}
            self.assertFalse(student_module.patch(root))
            self.assertEqual(patched,{path:(root/path).read_bytes() for path in paths})
            sensitivity_spec = importlib.util.spec_from_file_location('patch_sensitivity',ROOT/'scripts/patch_sensitivity_liga.py')
            sensitivity_module = importlib.util.module_from_spec(sensitivity_spec)
            sensitivity_spec.loader.exec_module(sensitivity_module)
            self.assertTrue(sensitivity_module.patch(root))
            patched = {path:(root/path).read_bytes() for path in paths}
            self.assertFalse(sensitivity_module.patch(root))
            self.assertEqual(patched,{path:(root/path).read_bytes() for path in paths})

    def test_split_overlap_rejected_before_data_read(self):
        spec = importlib.util.spec_from_file_location("audit_kitti", ROOT / "scripts/audit_kitti.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/"train.txt").write_text("000001\n", encoding="utf-8")
            (root/"val.txt").write_text("000001\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "overlap"):
                module.audit(root, root/"train.txt", root/"val.txt")
