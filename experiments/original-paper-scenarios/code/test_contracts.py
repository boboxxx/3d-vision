import copy
import hashlib
import unittest

import numpy as np

from contracts import digital_resources, normalize_symbols, scenarios, transmit, zero_force
from image_codecs import HEADER, decode_pair, encode_pair


class Contracts(unittest.TestCase):
    def test_replay_and_global_rng(self):
        global_state = np.random.get_state()
        symbols = normalize_symbols(np.arange(800, dtype=np.float32).reshape(-1, 2) + 1)
        out, evidence = transmit(symbols, kind="rayleigh", snr_db=10,
                                 noise_rng=np.random.default_rng(17), fading_rng=np.random.default_rng(18))
        noise, fade = np.random.default_rng(), np.random.default_rng()
        noise.bit_generator.state = copy.deepcopy(evidence["rng_before"]["noise"])
        fade.bit_generator.state = copy.deepcopy(evidence["rng_before"]["fading"])
        n = noise.standard_normal((400, 2)) * np.sqrt(.05)
        h = fade.standard_normal((400, 2)) / np.sqrt(2)
        xc = symbols[:, 0].astype(float) + 1j * symbols[:, 1].astype(float)
        hc = h[:, 0] + 1j * h[:, 1]
        nc = n[:, 0] + 1j * n[:, 1]
        expected = ((hc * xc + nc) / hc)
        np.testing.assert_array_equal(out, np.column_stack((expected.real, expected.imag)).astype(np.float32))
        np.testing.assert_array_equal(np.random.get_state()[1], global_state[1])
        self.assertEqual(np.random.get_state()[2:], global_state[2:])
        self.assertEqual(noise.bit_generator.state, evidence["rng_after"]["noise"])
        self.assertEqual(fade.bit_generator.state, evidence["rng_after"]["fading"])

    def test_identity_no_draws_and_bad_power(self):
        x = np.tile(np.array([[1., 0.]], dtype=np.float32), (9, 1))
        out, evidence = transmit(x, kind="identity", snr_db=10,
                                 noise_rng=np.random.default_rng(7), fading_rng=np.random.default_rng(8))
        np.testing.assert_array_equal(out, x)
        self.assertEqual(evidence["rng_before"], evidence["rng_after"])
        with self.assertRaisesRegex(ValueError, "Es=1"):
            transmit(x * 2, kind="awgn", snr_db=10,
                     noise_rng=np.random.default_rng(7), fading_rng=np.random.default_rng(8))

    def test_complex_noise_and_deep_fade(self):
        x = np.tile(np.array([[1., 0.]], dtype=np.float64), (200000, 1))
        out, _ = transmit(x, kind="awgn", snr_db=10,
                          noise_rng=np.random.default_rng(1), fading_rng=np.random.default_rng(2))
        self.assertAlmostEqual(float(np.mean(np.sum((out - x) ** 2, axis=1))), .1, delta=.001)
        fade = np.random.default_rng(2).standard_normal((200000, 2)) / np.sqrt(2)
        self.assertAlmostEqual(float(np.mean(np.sum(fade ** 2, axis=1))), 1, delta=.01)
        h = np.array([1e-12 + 0j])
        z = zero_force(h + .1, h)
        self.assertGreater(z[0].real, 1e10)  # must not hide ZF's deep-fade amplification
        with self.assertRaisesRegex(ValueError, "zero channel"):
            zero_force(np.array([1 + 0j]), np.array([0 + 0j]))

    def test_actual_block_overheads(self):
        a = digital_resources(source_bits=65, n=96, k=64, qam=64,
                              reliable_uses=3, reliable_energy=6, pilot_uses=2)
        self.assertEqual((a["blocks"], a["source_padding_bits"], a["payload_uses"]), (2, 63, 32))
        self.assertEqual((a["total_uses"], a["total_energy"]), (37, 40))
        a = digital_resources(source_bits=48, n=96, k=48, qam=256,
                              reliable_uses=0, reliable_energy=0)
        self.assertEqual(a["payload_uses"], 12)
        a = digital_resources(source_bits=1, n=9, k=6, qam=64,
                              reliable_uses=0, reliable_energy=0)
        self.assertEqual((a["payload_uses"], a["QAM_padding_bits"]), (2, 3))
        with self.assertRaisesRegex(ValueError, "rate mismatch"):
            digital_resources(source_bits=48, n=96, k=64, qam=256,
                              reliable_uses=0, reliable_energy=0)

    def test_real_image_streams_and_native_geometry(self):
        image = np.random.default_rng(7).integers(0, 256, (65, 97, 3), dtype=np.uint8)
        for codec, parameter in (("jpeg", 50), ("jpeg2000", 10), ("jpeg2000", 30), ("jpeg2000", 50)):
            with self.subTest(codec=codec, parameter=parameter):
                wire, pair, evidence = encode_pair(image, image[:, ::-1].copy(), codec=codec, parameter=parameter)
                self.assertEqual(len(wire), 20 + sum(evidence["codestream_bytes"]))
                self.assertEqual(evidence["source_bits"], len(wire) * 8)
                self.assertEqual(evidence["wire_sha256"], hashlib.sha256(wire).hexdigest())
                self.assertEqual(pair[0].shape, (65, 97, 3))
                self.assertEqual(pair[1].shape, pair[0].shape)
                self.assertGreater(np.count_nonzero(pair[0] != image), 0)
                for a, b in zip(pair, decode_pair(wire)):
                    np.testing.assert_array_equal(a, b)
                altered = wire[:-1] + bytes([wire[-1] ^ 1])
                with self.assertRaisesRegex(ValueError, "CRC failure"):
                    decode_pair(altered)
                with self.assertRaisesRegex(ValueError, "length mismatch"):
                    decode_pair(wire + b"\0")
                with self.assertRaisesRegex(ValueError, "length mismatch"):
                    decode_pair(wire[:-1])
                swapped_header = bytearray(wire)
                swapped_header[5] = 3 - swapped_header[5]
                with self.assertRaisesRegex(ValueError, "codec/schema mismatch"):
                    decode_pair(bytes(swapped_header))
        self.assertEqual(HEADER.size, 20)

    def test_no_floating_image_or_geometry_substitution(self):
        image = np.zeros((65, 97, 3), dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, "uint8 RGB"):
            encode_pair(image.astype(np.float32), image, codec="jpeg", parameter=50)
        with self.assertRaisesRegex(ValueError, "matched pair"):
            encode_pair(image, image[:-1], codec="jpeg", parameter=50)

    def test_reference_grids_and_reuse(self):
        rows = scenarios()
        self.assertEqual(len(rows), 154)
        self.assertEqual(sum(len(r["tables"]) for r in rows), 167)
        self.assertEqual({t: sum(t in r["tables"] for r in rows) for t in ("source", "channel", "joint")},
                         {"source": 16, "channel": 60, "joint": 91})
        for channel, expected in (("awgn", set(range(6, 19))), ("rayleigh", {6, 8, 10, 12, 14, 16, 18})):
            self.assertEqual({r["snr_db"] for r in rows if r["channel"] == channel}, expected)
        shared = [r for r in rows if len(r["tables"]) == 2]
        self.assertEqual(len(shared), 13)
        self.assertTrue(all(r["source"] == "original_rgb" and r["transport"] == "learned" for r in shared))


if __name__ == "__main__":
    unittest.main(verbosity=2)
