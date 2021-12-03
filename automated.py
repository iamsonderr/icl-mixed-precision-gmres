#!/usr/bin/python3

import argparse
import csv
import itertools
import os
import re
import subprocess
import sys


def run_cmd(cmd):
    env = dict(os.environ);
    env['OMP_PROC_BIND'] = 'spread'
    env['OMP_PLACES'] = 'threads'

    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True, env=env)

    out = ''

    while process.poll() is None:
        l = process.stdout.readline() # This blocks until it receives a newline.
        out += l
        print(l, flush=True, end='')
    l = process.stdout.read()
    out += l
    print(l, flush=True)

    process.communicate()

    return (out, process.returncode)

summary_regex = r'''
Found solution with rel prec res norm = (\d\.?\d*e(?:\+|-)\d+|\d+\.?\d*) when k = (\d+) and i = (\d+)
  total iterations = (\d+)
  ilu took (\d\.?\d*e(?:\+|-)\d+|\d+\.?\d*)s; gmres took (\d\.?\d*e(?:\+|-)\d+|\d+\.?\d*)s
  resNorm = (\d\.?\d*e(?:\+|-)\d+|\d+\.?\d*); errNorm = (\d\.?\d*e(?:\+|-)\d+|\d+\.?\d*)
'''


def run_test(matrix, mode='mixed', rlen='100', tol='1e-6', max_restarts='1000000',
                     rtol='0', rorth='0',
                     rand='42', orth='MGS', repeated_iter=False, gpu=False,
                     prec='ilu',
                     rhs=False):

    print('test: '
          + matrix
          + ' ' + mode
          + ' ' + orth
          + ' tol = '+tol
          + ' max-restart = '+max_restarts
          + ' rlen = '+rlen
          + ' rtol = '+rtol
          + ' rorth = '+rorth
          + ' seed = '+rand
          + ' repeated-iter = '+str(repeated_iter)
          + ' gpu = '+str(gpu)
          + ' prec = '+prec
          + ' rhs = '+str(rhs)
          , flush=True)

    if prec[0:11] == 'ilu_jacobi(':
        prec_flags = ['--jacobi-steps', prec[11:-1]]
        prec = 'ilu_jacobi'
    else:
        prec_flags = []

    mat_dir = os.getenv('MTXDIR', 'mats')

    command = (['./gmres_perf_test',
                '--Apath', mat_dir+'/'+matrix+'.mtx',
                '--rand', str(rand),
                '--orth', orth,
                '--tol', tol,
                '--max-restarts', max_restarts,
                '--rtol', (rtol if rorth == '0' else rorth),
                '--rlen', rlen,
                '--mode', mode,
                '--prec', prec
               ] + (['--repeat-iter'] if repeated_iter else [])
                 + (['--orthloss'] if rorth != '0' else [])
                 + (['--gpu'] if gpu else [])
                 + (['--bpath', mat_dir+'/'+matrix+'_b.mtx'] if rhs else [])
                 + prec_flags)
    result, returncode = run_cmd(command)

    summary_match = re.search(re.compile(summary_regex), result)
    if summary_match == None:
        return {'k':'-',
                'i':'-',
                'precRes':'-',
                'total_iters':'-',
                'res':'-',
                'err':'-',
                'ilu':'-',
                'gmres':'-',
            }

    return {'k':summary_match.group(2),
            'i':summary_match.group(3),
            'precRes':summary_match.group(1),
            'total_iters':summary_match.group(4),
            'res':summary_match.group(7),
            'err':summary_match.group(8),
            'ilu':summary_match.group(5),
            'gmres':summary_match.group(6),
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Runs experiments for mixed precision gmres')
    parser.add_argument('--no-baseline', dest='skip_baseline', action='store_true')
    parser.add_argument('--no-mixed', dest='skip_mixed', action='store_true')
    parser.add_argument('--no-singleprec', dest='skip_singlePrec', action='store_true')
    parser.add_argument('--no-single', dest='skip_single', action='store_true')
    parser.add_argument('--orth', dest='orth', action='store', default='MGS')
    parser.add_argument('--rorth', action='store', default='0')
    parser.add_argument('--repeated-iter', dest='repeated_iter', action='store_true')
    parser.add_argument('--gpu', dest='gpu', action='store_true')
    parser.add_argument('--prec', dest='prec', action='store', default='ilu')
    parser.add_argument('--max-restarts', dest='max_restarts', action='store', default='1000000')
    parser.add_argument('--rhs', action='store_true')
    parser.add_argument('mat')
    parser.add_argument('rlens')
    parser.add_argument('rtols')
    parser.add_argument('tols')
    parser.add_argument('seeds', nargs='?', default='42')
    args = parser.parse_args()
    mat = args.mat
    orth = args.orth
    rlens = args.rlens.split(' ')
    rorths = args.rorth.split(' ')
    if args.rtols != '':
        rtols = args.rtols.split(' ')
    else:
        rtols = []
    tols  = args.tols.split(' ')
    max_restarts = args.max_restarts
    seeds = args.seeds.split(' ')
    gpu = args.gpu
    precs = args.prec.split(' ')
    repeated_iter = args.repeated_iter
    rhs = args.rhs

    modes = (([] if args.skip_baseline   else [('baseline', 'b')])
            +([] if args.skip_mixed      else [('mixed', 'mp')])
            +([] if args.skip_singlePrec else [('single-prec', 'p')])
            +([] if args.skip_single else [('single', 's')]))

    print("parsed args", flush=True)

    entries = [(mode_key,  rl, ('R' if repeated_iter else '')+str(rt), ro, t, gpu, prec,
                run_test(mat, rhs=rhs, mode=mode, orth=orth, gpu=gpu, prec=prec,
                         tol=t, max_restarts=max_restarts, rlen=rl, rtol=rt, rorth=ro, repeated_iter=repeated_iter))
               for (rl, rt, t, ro, (mode, mode_key), prec)
                in itertools.product(rlens, rtols, tols, rorths, modes, precs)]

    with open('history-'+mat+'.csv', 'a') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        for (typ, rl, rt, ro, t, gpu, prec, summary) in entries:
            writer.writerow([mat, typ, orth, rl, rt, ro, t, ('cuda' if gpu else 'mkl'), prec,
                             summary['i'],
                             summary['total_iters'],
                             summary['res'],
                             summary['err'],
                             summary['ilu'],
                             summary['gmres'],
                             ])
