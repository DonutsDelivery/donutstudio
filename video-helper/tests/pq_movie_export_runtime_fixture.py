#!/usr/bin/env python3
"""Decode real PQ and HLG Scene3D helper exports against independently specified scene pixels.

The input JSON is an actual saved helper job with a native-linear Scene3D plan.
The oracle JSON names uniform sample pixels for three roles: reference-white, peak, and
color. Each has frame, x, y, and linearSrgb values from the authored scene.
"""
import argparse
import hashlib
import json
import math
import pathlib
import shutil
import struct
import subprocess
import sys
import tempfile
import wave
from fractions import Fraction

from export_runtime_fixture import Helper, poll_export, require


def pq_inverse(value):
    m1, m2 = 2610 / 16384, 2523 / 32
    c1, c2, c3 = 3424 / 4096, 2413 / 128, 2392 / 128
    power = max(value, 0) ** (1 / m2)
    return 10000 * (max(power - c1, 0) / (c2 - c3 * power)) ** (1 / m1)


def expected_nits(rgb):
    matrix = ((0.6274040, 0.3292820, 0.0433136),
              (0.0690970, 0.9195400, 0.0113612),
              (0.0163916, 0.0880132, 0.8955950))
    return [203 * sum(row[index] * rgb[index] for index in range(3))
            for row in matrix]


def hlg_display(rgb, peak_nits):
    # Independent forward reference EOTF, ITU-R BT.2100-3 Table 5/Note 5f.
    # Encoded black has no display-dependent lift. Gamma acts on luminance.
    a, b, c = 0.17883277, 0.28466892, 0.55991073
    scene = []
    for component in rgb:
        value = max(0, component)
        scene.append(value ** 2 / 3 if value <= 0.5
                     else (math.exp((value - c) / a) + b) / 12)
    gamma = (1.2 + 0.42 * math.log10(peak_nits / 1000) if peak_nits <= 2000
             else 1.2 * 1.111 ** math.log2(peak_nits / 1000))
    luminance = sum(weight * value for weight, value in zip((0.2627, 0.6780, 0.0593), scene))
    return [peak_nits * luminance ** (gamma - 1) * value for value in scene]


def decoded_pixel(raw, width, height, frame_number, x, y, transfer='pq', peak_nits=1000):
    frame_bytes = width * height * 3
    data = memoryview(raw)[frame_number * frame_bytes:(frame_number + 1) * frame_bytes]
    luma_size = width * height * 2
    chroma_size = width * height // 2
    y_code = struct.unpack_from('<H', data, 2 * (y * width + x))[0]
    c_index = ((y // 2) * (width // 2) + x // 2) * 2
    u_code = struct.unpack_from('<H', data, luma_size + c_index)[0]
    v_code = struct.unpack_from('<H', data, luma_size + chroma_size + c_index)[0]
    luma = (y_code - 64) / 876
    cb, cr = (u_code - 512) / 896, (v_code - 512) / 896
    r = luma + 2 * (1 - 0.2627) * cr
    b = luma + 2 * (1 - 0.0593) * cb
    g = (luma - 0.2627 * r - 0.0593 * b) / (1 - 0.2627 - 0.0593)
    return (hlg_display((r, g, b), peak_nits) if transfer == 'hlg'
            else [pq_inverse(value) for value in (r, g, b)])


def command_json(command):
    result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    require(result.returncode == 0, result.stderr)
    return json.loads(result.stdout)


def export(helper_path, job):
    helper = Helper(helper_path)
    try:
        expected_backend = 'metal' if sys.platform == 'darwin' else 'opengl'
        if sys.platform == 'darwin':
            backend = helper.response(helper.send('gpu_backend_info')).get('result', {})
            require(backend.get('available') and backend.get('backend') == 'metal',
                    f'fixture requires the actual native Metal helper: {backend}')
        status, _, response = poll_export(helper, helper.send('export', job))
        if 'error' not in response:
            backend = status.get('visualTelemetry', {}).get('backend', {})
            require(backend.get('available') and backend.get('initial') == expected_backend
                    and backend.get('current') == expected_backend and backend.get('fallbackCount') == 0,
                    f'HDR export did not retain its actual {expected_backend} owner: {backend}')
        return response
    finally:
        helper.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('helper', type=pathlib.Path)
    parser.add_argument('scene_job', type=pathlib.Path)
    parser.add_argument('oracle', type=pathlib.Path)
    parser.add_argument('--expected-encoder', choices=('donut-kvazaar10', 'libx265'), required=True)
    parser.add_argument('--receipt', type=pathlib.Path, required=True)
    parser.add_argument('--movie-dir', type=pathlib.Path,
                        help='retain accepted movies; defaults to a directory next to the receipt')
    parser.add_argument('--with-audio', action='store_true',
                        help='include and independently decode a generated calibration tone')
    args = parser.parse_args()
    require(sys.platform.startswith("linux") or sys.platform == "darwin",
            "HDR movie fixture requires the native Linux OpenGL or macOS Metal helper")
    movie_dir = args.movie_dir or args.receipt.parent / (args.receipt.stem + '-movies')
    require(not args.receipt.exists(), 'receipt destination already exists')
    require(not movie_dir.exists(), 'movie evidence directory already exists')
    job = json.loads(args.scene_job.read_text())
    oracle = json.loads(args.oracle.read_text())
    require(job.get('visualLayerPlans') and job.get('segments'),
            'fixture requires a real compiled Scene3D helper job')
    require({sample['role'] for sample in oracle['samples']}
            == {'reference-white', 'peak', 'color'},
            'oracle requires reference-white, peak and color scene pixels')
    samples_by_role = {sample['role']: sample for sample in oracle['samples']}
    require(all(math.isclose(value, 1, abs_tol=0.01) for value in
                samples_by_role['reference-white']['linearSrgb']),
            'reference-white oracle must be linear white at 203 nits')
    require(max(expected_nits(samples_by_role['peak']['linearSrgb']))
            >= oracle['peakNits'] * 0.8, 'peak oracle must be near declared peak')
    require(len(set(samples_by_role['color']['linearSrgb'])) > 1,
            'color oracle must contain distinct RGB channels')
    width, height, fps = job['width'], job['height'], job['fps']
    frames = oracle['frames']
    require(oracle.get('width', width) == width and oracle.get('height', height) == height,
            'authored oracle extent differs from scene job')
    if 'frameRate' in oracle:
        require(math.isclose(float(Fraction(oracle['frameRate'])), fps, rel_tol=1e-12),
                'authored oracle frame rate differs from scene job')
    require({sample['frame'] for sample in oracle['samples']} >= {0, frames // 2, frames - 1},
            'linear oracle must sample initial, middle and final delayed frames')
    require(width % 2 == height % 2 == 0 and frames >= 17,
            'fixture needs even extents and at least 17 frames to cover delayed GOP drain')
    if args.expected_encoder == 'donut-kvazaar10':
        require(width % 8 == height % 8 == 0, 'private adapter requires exact extents divisible by eight')
    require(math.isfinite(fps) and fps > 0, 'invalid oracle cadence')
    rate = Fraction(fps).limit_denominator(100000)
    receipt = {
        'helperSha256': hashlib.sha256(args.helper.read_bytes()).hexdigest(),
        'sceneJobSha256': hashlib.sha256(args.scene_job.read_bytes()).hexdigest(),
        'linearOracleSha256': hashlib.sha256(args.oracle.read_bytes()).hexdigest(),
        'expectedEncoder': args.expected_encoder, 'frameRate': str(rate), 'rows': [],
        'staticHdrMetadata': 'not-authored; peakNits is not mastering-display or MaxCLL/MaxFALL',
    }
    for transfer, transfer_tag in (('pq', 'smpte2084'), ('hlg', 'arib-std-b67')):
        with tempfile.TemporaryDirectory(prefix=f'donutstudio-{transfer}-movie-') as directory:
            output = pathlib.Path(directory) / 'scene.mp4'
            if args.with_audio:
                audio = pathlib.Path(directory) / 'calibration.wav'
                sample_rate = 48000
                count = round(frames / fps * sample_rate)
                with wave.open(str(audio), 'wb') as wav:
                    wav.setnchannels(1)
                    wav.setsampwidth(2)
                    wav.setframerate(sample_rate)
                    wav.writeframes(b''.join(struct.pack('<h', round(4096 * math.sin(
                        2 * math.pi * 997 * index / sample_rate))) for index in range(count)))
                job['audioPath'] = str(audio)
            job.update(outPath=str(output), codec='h265', encoder='software', intraOnly=False,
                       interpolation='none', hdrImageProfile='off',
                       hdrVideoOutput={'primaries': 'rec2020-d65', 'transfer': transfer,
                                       'alpha': 'opaque', 'peakNits': oracle['peakNits']})
            response = export(args.helper, job)
            require('error' not in response, f'{transfer.upper()} Scene3D export failed: {response}')
            require(response.get('result', {}).get('encoder') == args.expected_encoder,
                    f'wrong production encoder: {response}')
            probe = command_json(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                                  '-count_frames', '-show_frames', '-show_streams',
                                  '-of', 'json', str(output)])
            stream = probe['streams'][0]
            require((stream['codec_name'], stream['pix_fmt'], stream['color_range'],
                     stream['color_space'], stream['color_transfer'], stream['color_primaries'],
                     stream['chroma_location'])
                    == ('hevc', 'yuv420p10le', 'tv', 'bt2020nc', transfer_tag, 'bt2020', 'left'),
                    f'{transfer.upper()} stream tags or depth differ: {stream}')
            require(stream.get('profile') == 'Main 10', f'HEVC profile differs: {stream}')
            require((stream['width'], stream['height']) == (width, height),
                    f'visible movie extent changed: {stream}')
            require(Fraction(stream['avg_frame_rate']) == rate,
                    f'rational movie frame rate differs: {stream}')
            require(all((frame['width'], frame['height']) == (width, height)
                        and not any(frame.get(field, 0) for field in
                                    ('crop_left', 'crop_right', 'crop_top', 'crop_bottom'))
                        for frame in probe['frames']), 'decoded extent or crop differs')
            require(int(stream['nb_read_frames']) == frames, f'{transfer.upper()} movie frame count differs')
            timestamps = [float(frame['best_effort_timestamp_time']) for frame in probe['frames']]
            require(len(timestamps) == frames and all(
                math.isclose(value, index / fps, abs_tol=0.00001)
                for index, value in enumerate(timestamps)), f'{transfer.upper()} movie cadence differs')
            packets = command_json(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                                    '-show_packets', '-of', 'json', str(output)])['packets']
            require(len(packets) == frames, 'encoder dropped or duplicated access units during drain')
            require(all('pts' in packet and 'dts' in packet and int(packet.get('duration', 0)) > 0
                        for packet in packets), 'missing real packet PTS/DTS/duration')
            require(all(int(a['dts']) < int(b['dts']) for a, b in zip(packets, packets[1:])),
                    'packet DTS does not increase in mux order')
            time_base = Fraction(stream['time_base'])
            pts = sorted(Fraction(int(packet['pts'])) * time_base for packet in packets)
            require(all(abs(float(value - Fraction(index, 1) / rate)) <= 0.00001
                        for index, value in enumerate(pts)), 'packet presentation timeline differs')
            decode = subprocess.run(['ffmpeg', '-v', 'error', '-i', str(output),
                                     '-f', 'rawvideo', '-pix_fmt', 'yuv420p10le', '-'],
                                    capture_output=True, timeout=60)
            require(decode.returncode == 0 and len(decode.stdout) == frames * width * height * 3,
                    f'{transfer.upper()} movie decode failed: {decode.stderr.decode(errors="replace")}')
            sample_receipts = []
            for sample in oracle['samples']:
                require(0 <= sample['frame'] < frames and 0 <= sample['x'] < width
                        and 0 <= sample['y'] < height, 'sample coordinate is outside movie')
                actual = decoded_pixel(decode.stdout, width, height,
                                       sample['frame'], sample['x'], sample['y'], transfer, oracle['peakNits'])
                expected = expected_nits(sample['linearSrgb'])
                require(all(abs(a - b) <= max(12, b * 0.08) for a, b in zip(actual, expected)),
                        f"{sample['role']} decoded {transfer.upper()} differs: {actual} versus {expected}")
                sample_receipts.append(dict(sample, decodedNits=actual, expectedNits=expected))
            if args.with_audio:
                audio_probe = command_json(['ffprobe', '-v', 'error', '-select_streams', 'a:0',
                                            '-show_streams', '-of', 'json', str(output)])
                require(len(audio_probe['streams']) == 1 and audio_probe['streams'][0]['codec_name'] == 'aac',
                        'HDR mux lost its AAC audio stream')
                decoded_audio = subprocess.run(['ffmpeg', '-v', 'error', '-i', str(output),
                                                '-map', '0:a:0', '-ac', '1', '-ar', '48000',
                                                '-f', 's16le', '-'], capture_output=True, timeout=60)
                require(decoded_audio.returncode == 0, 'muxed audio did not decode')
                values = struct.unpack('<' + 'h' * (len(decoded_audio.stdout) // 2), decoded_audio.stdout)
                require(abs(len(values) - round(frames / fps * 48000)) <= 2048,
                        'muxed audio duration differs from movie')
                # Independent frequency projection over the middle of the decoded
                # tone tolerates AAC delay while rejecting silence or unrelated audio.
                middle = values[len(values) // 4:len(values) * 3 // 4]
                projection = abs(sum(value * complex(math.cos(2 * math.pi * 997 * index / 48000),
                                                     math.sin(2 * math.pi * 997 * index / 48000))
                                     for index, value in enumerate(middle))) / len(middle)
                require(projection > 1500, f'muxed calibration tone differs: {projection}')
            receipt['rows'].append({
                'transfer': transfer, 'movieSha256': hashlib.sha256(output.read_bytes()).hexdigest(),
                'codec': stream['codec_name'], 'profile': stream['profile'],
                'format': stream['pix_fmt'], 'width': width, 'height': height,
                'frames': frames, 'packetCount': len(packets), 'audio': args.with_audio,
                'linearSamples': sample_receipts,
                'packetPts': [packet['pts'] for packet in packets],
                'packetDts': [packet['dts'] for packet in packets],
                'packetTimeBase': stream['time_base'],
                'moviePath': str((movie_dir / (transfer + '.mp4')).resolve()),
            })
            for wrong_transfer, image_profile in ((transfer, 'linear-rec2020'), ('linear', 'off')):
                wrong = dict(job, outPath=str(pathlib.Path(directory) / 'wrong.mp4'),
                             hdrImageProfile=image_profile,
                             hdrVideoOutput=dict(job['hdrVideoOutput'], transfer=wrong_transfer))
                require('error' in export(args.helper, wrong),
                        f'{wrong_transfer}/{image_profile} wrong route was accepted')
                require(not pathlib.Path(wrong['outPath']).exists(), 'wrong route left a movie')
            cancelled = dict(job, outPath=str(pathlib.Path(directory) / 'cancelled.mp4'),
                             durationSec=2000 / fps, endSec=2000 / fps,
                             segments=[dict(segment, outSec=2000 / fps)
                                       for segment in job['segments']])
            helper = Helper(args.helper)
            try:
                _, samples, result = poll_export(
                    helper, helper.send('export', cancelled), cancel_after_frame=1)
                require(any(sample['frame'] >= 1 for sample in samples),
                        f'{transfer.upper()} cancellation did not start after a rendered frame')
                require(result.get('error', {}).get('message') == 'cancelled',
                        f'{transfer.upper()} cancellation failed: {result}')
                require(not pathlib.Path(cancelled['outPath']).exists(),
                        f'{transfer.upper()} cancellation left a partial movie')
            finally:
                helper.close()
            movie_dir.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(output, movie_dir / (transfer + '.mp4'))
    receipt['result'] = 'PASS'
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2) + '\n')
    print('PASS: decoded Scene3D PQ/HLG pixels, Rec.2020 transfer tags, ten-bit depth, cadence, rejection and cancellation')


if __name__ == '__main__':
    main()
