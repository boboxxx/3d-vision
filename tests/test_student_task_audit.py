import importlib.util
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('task_audit',ROOT/'scripts/audit_student_task.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class NativeLearningRateTests(unittest.TestCase):
    def test_native_before_after_endpoints_and_missing_evidence(self):
        actual={index:.0001 for index in range(4)}
        module.check_native_learning_rate(actual,3,.0001)
        for missing in (0,1,3):
            broken={key:value for key,value in actual.items() if key!=missing}
            with self.assertRaises(ValueError):
                module.check_native_learning_rate(broken,3,.0001)
        changed=dict(actual)
        changed[2]=.0002
        with self.assertRaises(ValueError):
            module.check_native_learning_rate(changed,3,.0001)


if __name__=='__main__':
    unittest.main()
