#!/usr/bin/env python
"""

author: Javier Nistal
"""
import argparse
import numpy as np
from measurement import signals, ir_measurement, ir_window


def main(args):
    time = args.time
    # method = args.method
    name = args.name

    assert time > 0, \
        'Time variable must be greater than zero '
    ss = signals.SineSweep(duration=time)
    irm = ir_measurement.IRMeasurement()

    sweep = ss.create()
    sweep_response = irm.measure(sweep)
    ir = ss.get_ir(sweep, sweep_response)
    ir = ir_window.IRwindow.lundeby(ir[np.argmax(ir):])
    ir_measurement.save_file(ir, 'cukiverb/impulse-responses/'+name+'.wav')
    print '* finished'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Impulse response measurement'
    )

    parser.add_argument(
        '-t',
        dest='time',
        type=float,
        required=True,
        help='Input: measurement time. Example: \
        "-t 10"'
    )
    parser.add_argument(
        '-m',
        dest='method',
        default='logarithmic_sinesweep',
        help='Input: method for measuring the ir. \
        DEFAULT: logarithmic sine-sweep'
    )
    parser.add_argument(
        '-n',
        dest='name',
        default='default',
        help='Input: output file-name. \
        DEFAULT: default.wav'
    )

    args = parser.parse_args()

    main(args)
