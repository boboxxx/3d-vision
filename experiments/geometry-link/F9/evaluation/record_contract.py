"""Independent final-inference coding record checks; no producer/model imports."""
import hashlib
import math


def validate(row, arm, channel, snr):
    assert arm in ('U', 'G', 'P', 'S') and channel in ('identity', 'awgn')
    assert float(snr) in ((10.,) if channel == 'identity' else (6., 10., 18.))
    assert row['arm'] == arm and row['channel'] == channel
    a, c = row['accounting'], row['F9_coding']
    assert c['arm'] == arm and a['snr_db'] == c['nominal_snr'] == float(snr) and a['channel'] == channel
    assert a['allocation'] == ('uniform' if arm == 'U' else 'F9_' + arm)
    assert row['channel_input_shape'] == [1, 62400, 2]
    assert a['data_complex_uses'] == a['total_complex_uses'] == c['data_complex_uses'] == 62400
    assert a['stereo_complex_uses'] == 49920 and a['appearance_complex_uses'] == 12480
    assert a['header_complex_uses'] == a['pilot_complex_uses'] == 0 and not c['receiver_side_information']
    assert c['head_parameters'] == 561 and c['group_energy_layout'] == 'left32_then_right32_pixel_center_groups'
    e, g = c['stereo_group_energy'], c['amplitude_gains']
    assert len(e) == len(g) == 64 and all(math.isfinite(x) and x >= 0 for x in e)
    assert math.isfinite(c['appearance_energy']) and c['appearance_energy'] >= 0
    assert all(math.isfinite(x) and .5 <= x <= 2 for x in g)
    assert math.isclose(sum(e) + c['appearance_energy'], c['actual_energy_float64'], rel_tol=1e-12, abs_tol=1e-8)
    assert abs(c['actual_energy_float64'] / 62400 - 1) <= 1e-5
    assert len(a['tx_energy_per_frame']) == 1
    assert math.isclose(c['actual_energy_float64'], a['tx_energy_per_frame'][0], rel_tol=1e-5, abs_tol=1e-3)
    if arm == 'U': assert g == [1.] * 64
    if arm in ('P', 'S'):
        assert len(c['matrix_records']) == 2
        assert all(m['row_sum_max_error'] <= 6e-5 and m['cross_vertical_mass'] == 0 for m in c['matrix_records'])
    else: assert c['matrix_records'] is None
    for k in ('noise_rng_before', 'noise_rng_after'):
        assert hashlib.sha256(bytes.fromhex(row[k]['state_hex'])).hexdigest() == row[k]['sha256']
    assert (row['noise_rng_before'] == row['noise_rng_after']) == (channel == 'identity')
    assert row['sequence'] == ['student', 'student', 'link_start', 'channel', 'link_done', 'build_cost']
    assert row['cost_inputs_are_received_features'] and row['appearance_input_is_received_feature']
    assert not row['autograd_enabled']
    assert set(row['sensor_input_keys']) <= {'batch_size', 'left_img', 'right_img', 'calib', 'image_shape', 'frame_id'}
