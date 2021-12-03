#!/usr/bin/python3

import argparse
import math
from statistics import median
import sys
from utils import open_history_file, process_rows

def find_min_time(gmres_times, ilu_times):
    min_time = math.inf
    min_ilu_time = math.inf
    min_loc = None
    for loc, times in gmres_times.items():
        med_time = median(times)
        if med_time < min_time:
            min_time = med_time
            min_loc = loc
            min_ilu_time = median(ilu_times[loc])
    return min_time, min_ilu_time, min_loc

def ensure_list_present(loc, dict):
    if loc not in dict:
        dict[loc] = []

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parses history files for exploratory tests to determine the optimal configuration')
    parser.add_argument('--timing-script-format', dest='timing_script_format', action='store_true')
    parser.add_argument('--plotting-format', dest='plotting_format', action='store_true')
    parser.add_argument('--rlen', action='store', default=None)
    parser.add_argument('--rtol', action='store', default=None)
    parser.add_argument('--rorth', action='store', default=None)
    parser.add_argument('tol')
    parser.add_argument('orth')
    parser.add_argument('device', help='The device used to run the results.  I.E. cuda or mkl.')
    parser.add_argument('prec', help='The preconditioner')
    parser.add_argument('mats', nargs='+')
    args = parser.parse_args()


    tol = args.tol
    orth = args.orth
    device = args.device
    prec = args.prec
    mats = args.mats
    timing_script_format = args.timing_script_format
    plotting_format = args.plotting_format
    rlen = args.rlen
    rtol = args.rtol
    rorth = args.rorth

    if timing_script_format and plotting_format:
        print("Cannot use both timing-script and plotting formats")
        quit(1)

    for mat in mats:
        base_ilu_times = {}
        base_gmres_times = {}
        base_restart_count = {}
        base_iter_count = {}

        mixed_ilu_times = {}
        mixed_gmres_times = {}
        mixed_restart_count = {}
        mixed_iter_count = {}

        prec_ilu_times = {}
        prec_gmres_times = {}
        prec_restart_count = {}
        prec_iter_count = {}

        single_ilu_times = {}
        single_gmres_times = {}
        single_restart_count = {}
        single_iter_count = {}

        def process(ilu_times, gmres_times, restart_count, iter_count):
            def process_inner(row):
                if row['gmres'] != '-':
                    loc = (row['rlen'], row['rtol'], row['rorth'])
                    ensure_list_present(loc, ilu_times)
                    ilu_times[loc].append(float(row['ilu']))
                    ensure_list_present(loc, gmres_times)
                    gmres_times[loc].append(float(row['gmres']))
                    restart_count[loc] = int(row['i'])
                    iter_count[loc] = int(row['total_iters'])
            return process_inner

        process_rows(mat,
                     process(base_ilu_times,   base_gmres_times,   base_restart_count,   base_iter_count),
                     process(mixed_ilu_times,  mixed_gmres_times,  mixed_restart_count,  mixed_iter_count),
                     process(prec_ilu_times,   prec_gmres_times,   prec_restart_count,   prec_iter_count),
                     process(single_ilu_times, single_gmres_times, single_restart_count, single_iter_count),
                     tol=tol, orth=orth, device=device, prec=prec, rlen=rlen, rtol=rtol, rorth=rorth)

        base_gmres_min,   base_ilu_min,   base_gmres_min_loc   = find_min_time(base_gmres_times,   base_ilu_times)
        mixed_gmres_min,  mixed_ilu_min,  mixed_gmres_min_loc  = find_min_time(mixed_gmres_times,  mixed_ilu_times)
        prec_gmres_min,   prec_ilu_min,   prec_gmres_min_loc   = find_min_time(prec_gmres_times,   prec_ilu_times)
        single_gmres_min, single_ilu_min, single_gmres_min_loc = find_min_time(single_gmres_times, single_ilu_times)

        if base_gmres_min_loc:
            if plotting_format:
                def process_times (loc, gmres_times, ilu_times, restart_count, iter_count):
                    if loc:
                        times = [gmres + ilu for gmres,ilu in zip(gmres_times[loc], ilu_times[loc])]
                        return [str(min(times)), str(median(times)), str(max(times)),
                                str(restart_count[loc]), str(iter_count[loc]), loc]
                    else:
                        return ['\'-\'', '\'-\'', '\'-\'', '\'-\'', '\'-\'', ['-', '-', '-']]

                b_times  = process_times(  base_gmres_min_loc,   base_gmres_times,   base_ilu_times,   base_restart_count,   base_iter_count)
                mp_times = process_times( mixed_gmres_min_loc,  mixed_gmres_times,  mixed_ilu_times,  mixed_restart_count,  mixed_iter_count)
                p_times  = process_times(  prec_gmres_min_loc,   prec_gmres_times,   prec_ilu_times,   prec_restart_count,   prec_iter_count)
                s_times  = process_times(single_gmres_min_loc, single_gmres_times, single_ilu_times, single_restart_count, single_iter_count)

                print('\''+mat+'\': [('+b_times[0]+', '+b_times[1]+', '+b_times[2]
                                       +', '+b_times[3]+', '+b_times[4]
                                       +', \''+b_times[5][0]+'\', \''+b_times[5][1]+'\', \''+b_times[5][2]+'\''
                                   +'), ('+mp_times[0]+', '+mp_times[1]+', '+mp_times[2]
                                          +', '+mp_times[3]+', '+mp_times[4]
                                          +', \''+mp_times[5][0]+'\', \''+mp_times[5][1]+'\', \''+mp_times[5][2]+'\''
                                   +'), ('+p_times[0]+', '+p_times[1]+', '+p_times[2]
                                          +', '+p_times[3]+', '+p_times[4]
                                          +', \''+p_times[5][0]+'\', \''+p_times[5][1]+'\', \''+p_times[5][2]+'\''
                                   +'), ('+s_times[0]+', '+s_times[1]+', '+s_times[2]
                                          +', '+s_times[3]+', '+s_times[4]
                                          +', \''+s_times[5][0]+'\', \''+s_times[5][1]+'\', \''+s_times[5][2]+'\')],')

            else:
                print(mat+' '+str(base_ilu_min) +' '+str(base_gmres_min) +' ('+base_gmres_min_loc[0]+', '+base_gmres_min_loc[1]+', '+base_gmres_min_loc[2]+')'
                         +' '+str(mixed_ilu_min)+' '+str(mixed_gmres_min)+' ('+mixed_gmres_min_loc[0]+', '+mixed_gmres_min_loc[1]+', '+mixed_gmres_min_loc[2]+')')
