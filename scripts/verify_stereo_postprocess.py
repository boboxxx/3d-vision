#!/usr/bin/env python3
"""Compare our headless decoder against execution of the author's actual demo.

AST changes remove presentation/LiDAR visualization, supply the checkpoint,
and update tensor-holder/scalar assignment APIs. Detection/3D math is retained.
"""
import argparse
import ast
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT/'third_party/Stereo-RCNN'
sys.path[:0] = [str(UPSTREAM),str(UPSTREAM/'lib'),str(ROOT/'src')]
from geocomm.evidence import sha256,source_identity


class Headless(ast.NodeTransformer):
    def visit_Assign(self,node):
        names = {n.id for target in node.targets for n in ast.walk(target) if isinstance(n,ast.Name)}
        if names & {'im_box','im2show','pointcloud','k'}:
            return None
        if names == {'load_name'}:
            node.value = ast.Name(id='_checkpoint_path',ctx=ast.Load())
        if names == {'args'} and isinstance(node.value,ast.Call):
            node.value = ast.Name(id='_demo_args',ctx=ast.Load())
        if isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name) and node.value.func.id == 'vis_detections':
            return None
        if names == {'poses'}:
            return self.generic_visit(node)
        if any(isinstance(t,ast.Tuple) for t in node.targets) and 'poses' in names:
            node.value = ast.parse('tuple(float(v) for v in _values)',mode='eval').body
            node.value.args[0].generators[0].iter = self._scalar_values
        return self.generic_visit(node)

    def visit_Module(self,node):
        # Capture old scalar tuple values before traversal replaces them.
        for child in ast.walk(node):
            if isinstance(child,ast.Assign) and any(isinstance(t,ast.Tuple) for t in child.targets):
                if any(isinstance(n,ast.Name) and n.id == 'poses' for t in child.targets for n in ast.walk(t)):
                    self._scalar_values = child.value
        return self.generic_visit(node)

    def visit_Expr(self,node):
        if isinstance(node.value,ast.Call):
            call = node.value
            if isinstance(call.func,ast.Attribute) and isinstance(call.func.value,ast.Name) and call.func.value.id == 'cv2':
                return None
            if isinstance(call.func,ast.Attribute) and call.func.attr == 'copy_':
                # Modern .data.resize_ does not resize the tensor holder itself.
                source = ast.unparse(call)
                if source.startswith('im_left_data.data.resize_'):
                    return ast.parse('im_left_data = img_left.cuda()').body[0]
                if source.startswith('im_right_data.data.resize_'):
                    return ast.parse('im_right_data = img_right.cuda()').body[0]
                if source.startswith('im_info.data.resize_'):
                    return ast.parse('im_info = info.cuda()').body[0]
        return self.generic_visit(node)

    def visit_If(self,node):
        condition = ast.unparse(node.test)
        if condition == 'k == 27':
            return None
        if condition == 'score > vis_thresh':
            return ast.parse('''_predictions.append(dict(score=float(score),bbox=box_left.tolist(),
dimensions_w_h_l=dim.tolist(),location=xyz.tolist(),rotation_y=float(theta),
alpha=float(poses_all[solved_idx,7]),disparity=float(dis_final[solved_idx])))''').body[0]
        return self.generic_visit(node)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('preserve evidence; use a unique output')
    checkpoint,reference,output = args.checkpoint.resolve(),args.reference.resolve(),args.output.resolve()
    report = dict(evidence_type='engineering_only',state='running',
        author_demo_sha256=sha256(UPSTREAM/'demo.py'),
        checkpoint_sha256=sha256(checkpoint),reference_sha256=sha256(reference),
        runtime_source=source_identity(UPSTREAM,['lib']))
    predictions = []
    try:
        tree = ast.fix_missing_locations(Headless().visit(ast.parse((UPSTREAM/'demo.py').read_text())))
        environment = dict(__name__='__main__',__file__=str(UPSTREAM/'demo.py'),
            _checkpoint_path=str(checkpoint),_predictions=predictions,
            _demo_args=SimpleNamespace(load_dir=str(checkpoint.parent),checkepoch=12,checkpoint=6477))
        os.chdir(UPSTREAM)
        exec(compile(tree,str(UPSTREAM/'demo.py'),'exec'),environment)
        expected = json.loads(reference.read_text())['demo_3d']['predictions']
        assert len(predictions) == len(expected) and len(predictions)>0
        errors = {}
        for key in expected[0]:
            actual = np.asarray([row[key] for row in predictions])
            target = np.asarray([row[key] for row in expected])
            np.testing.assert_allclose(actual,target,rtol=1e-5,atol=1e-5)
            errors[key] = float(np.max(np.abs(actual-target)))
        report.update(state='passed',predictions=len(predictions),max_absolute_errors=errors,
            migration_note='demo presentation removed; tensor holder and scalar assignment APIs updated')
        from geocomm.stereo_baseline import kitti_line
        from model.utils.kitti_utils import read_obj_calibration,write_detection_results
        calibration = read_obj_calibration(str(UPSTREAM/'demo/calib.txt'))
        with tempfile.TemporaryDirectory() as temporary:
            for row in expected:
                write_detection_results(temporary,'000000',calibration,row['bbox'],row['location'],
                                        row['dimensions_w_h_l'],row['rotation_y'],row['score'])
            author_text = (Path(temporary)/'data/000000.txt').read_text()
        actual_text = ''.join(kitti_line(row,calibration) for row in expected)
        assert actual_text == author_text
        report['kitti_serialization'] = 'byte-identical to released author writer, camera/angle conversion retained'
    except Exception as error:
        report.update(state='failed',error=repr(error))
        raise
    finally:
        output.parent.mkdir(parents=True,exist_ok=True)
        output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
