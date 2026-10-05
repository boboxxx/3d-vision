"""Feature diagnostics; high agreement is not a substitute for detection AP."""
import torch

INTERFACES = ('left_stereo', 'right_stereo', 'left_appearance')


@torch.no_grad()
def feature_agreement(students, targets):
    if len(students) != 3 or len(targets) != 3:
        raise ValueError('three native feature interfaces required')
    result = {}
    for name, student, target in zip(INTERFACES, students, targets):
        if student.shape != target.shape:
            raise ValueError('student/teacher interface differs')
        if not torch.isfinite(student).all() or not torch.isfinite(target).all():
            raise ValueError('nonfinite feature')
        student, target = student.detach().float(), target.detach().float()
        ss, tt = student.square().mean(), target.square().mean()
        denominator = (ss * tt).sqrt()
        cosine = ((student * target).mean() / denominator.clamp_min(1e-20)
                  if denominator > 0 else denominator.new_zeros(()))
        result[name] = dict(student_rms=float(ss.sqrt()), teacher_rms=float(tt.sqrt()),
            normalized_mse=float((student-target).square().mean()/tt.clamp_min(1e-6)),
            cosine=float(cosine.clamp(-1, 1)))
    return result


def assert_original_state(model, reference):
    """Check every author state, including global_step and all BN buffers."""
    current = model.state_dict()
    changed = [name for name, tensor in reference.items()
               if name not in current or current[name].shape != tensor.shape
               or not torch.equal(current[name].detach().cpu(), tensor)]
    if changed:
        raise RuntimeError('original detector state changed: ' + repr(changed))
    return len(reference)
