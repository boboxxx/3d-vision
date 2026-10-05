"""Native-size synthetic source engineering, not training or wireless AP."""
import argparse
import hashlib
import importlib.util
import json
import sys
import time
import traceback
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('source_rates005',HERE/'model.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
from wireless import WirelessVariant


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def tensors(s):
    return {k:{'shape':list(v.shape),'dtype':str(v.dtype),
               'sha256':hashlib.sha256(v.detach().contiguous().numpy().tobytes()).hexdigest()} for k,v in s.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    result = {'state':'starting_native_synthetic_source_engineering','started_unix':time.time(),
              'rates':[],'dataset_GT_or_detector_read':False,'optimizer_used':False,
              'actual_training_wire_rate_or_AP_claimed':False,'radio10_50_compatible_claimed':False}
    try:
        lock = json.loads((HERE/'inputs.json').read_text())
        for rel,expected in lock['files'].items():
            assert sha(ROOT/rel) == expected, rel
        torch.set_num_threads(2)
        torch.manual_seed(17)
        asset = ROOT/'assets/original-native-001/spynet_sintel_final-3d2a1287.pth'
        control = WirelessVariant(module.base.SemanticVariant(asset)).eval()
        control_states = tensors(control.state_dict())
        assert len(control_states) == 768
        rng = np.random.default_rng(17)
        left,right = [torch.from_numpy(rng.random((1,3,375,1242),dtype=np.float32)) for _ in range(2)]
        masks = [torch.zeros((1,1,375,1242)) for _ in range(2)]
        masks[0][...,33:251,107:421] = 1
        masks[1][...,41:247,94:405] = 1
        with torch.inference_mode():
            control_outputs = control.semantic(left,right,masks)
        control_hashes = tensors({str(i):v for i,v in enumerate(control_outputs)})
        del control,control_outputs
        print('frozen30 full native control forward completed',file=sys.stderr,flush=True)
        for rate in [30,10,50]:
            torch.manual_seed(17)
            model = WirelessVariant(module.SemanticRateVariant(asset,rate)).eval()
            initial = tensors(model.state_dict())
            if rate == 30:
                assert initial == control_states
            else:
                assert initial.keys() == control_states.keys()
                assert any(v['shape'] != control_states[k]['shape'] for k,v in initial.items())
            g,k = module.FACTORS[rate]
            factor = np.lcm(g,k)
            hp,wp = 375+(-375)%factor,1242+(-1242)%factor
            with torch.inference_mode():
                payload = model.semantic.encode(left,right,masks)
                for v in payload['global_values']:
                    assert v.shape == (1,3,hp//g,wp//g) and bool(torch.isfinite(v).all())
                for v in payload['key_values']:
                    assert v.shape == (1,3,hp//k,wp//k) and bool(torch.isfinite(v).all())
                # Received clone contains only declared packet values/control.
                received = {'global_values':tuple(v.clone() for v in payload['global_values']),
                            'key_values':tuple(v.clone() for v in payload['key_values']),
                            'masks':tuple(v.clone() for v in payload['masks']),
                            'original_shape':payload['original_shape']}
                output = model.semantic.decode(received)
                assert all(v.shape == (1,3,375,1242) and bool(torch.isfinite(v).all()) for v in output)
                output_hashes = tensors({str(i):v for i,v in enumerate(output)})
                if rate == 30:
                    assert output_hashes == control_hashes
                packet_hashes = tensors({kind+str(i):v for kind in ['global_values','key_values'] for i,v in enumerate(payload[kind])})
            del payload,received,output
            model.zero_grad(set_to_none=True)
            packet = model.semantic.encode(left,right,masks)
            loss = torch.stack([v.square().mean() for kind in ['global_values','key_values'] for v in packet[kind]]).mean()
            loss.backward()
            expected = {n for n,p in model.named_parameters() if n.startswith(('semantic.global_encoder.','semantic.key_encoders.'))}
            actual = {n for n,p in model.named_parameters() if p.grad is not None}
            assert expected == actual and expected
            assert all(bool(torch.isfinite(p.grad).all()) for n,p in model.named_parameters() if n in actual)
            assert tensors(model.state_dict()) == initial
            result['rates'].append({'nominal_rate':rate,'global_linear_factor':g,'key_linear_factor':k,
                                   'padded_hw':[int(hp),int(wp)],'packet_hashes':packet_hashes,'output_hashes':output_hashes,
                                   'registered_state_tensors':len(initial),'parameter_values':sum(p.numel() for p in model.parameters()),
                                   'initial_final_state_sha256':hashlib.sha256(json.dumps(initial,sort_keys=True).encode()).hexdigest(),
                                   'finite_encoder_gradient_tensors':len(actual),'packet_energy_loss':float(loss.detach()),
                                   'nominal30_matches_frozen_full_states_and_output':rate == 30})
            del model,packet,loss
            print(f'rate{rate} complete native RGB/packets/source-gradient engineering passed',file=sys.stderr,flush=True)
        assert len(result['rates']) == 3
        result.update(state='passed_all3_native_synthetic_source_architectures_nominal30_exact_frozen_control',
                      inputs_sha256=sha(HERE/'inputs.json'),torch_version=torch.__version__,numpy_version=np.__version__)
    except BaseException:
        result['state'] = 'failed_retained'
        result['traceback'] = traceback.format_exc()
        raise
    finally:
        result['ended_unix'] = time.time()
        args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'state':result['state'],'rates':[v['nominal_rate'] for v in result['rates']]}))


if __name__ == '__main__':
    main()
