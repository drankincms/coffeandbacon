#!/usr/bin/env python
from __future__ import print_function, division
import json
import argparse
from functools import partial

import uproot
import numpy as np
from coffea import processor
from coffea.util import load, save

def get_norm(item):
    dataset, filename = item
    file = uproot.open(filename)
    tree = fin['Events']
    skim_sumw = None
    if 'Events' in fin:
        runs = fin['Events']
        if 'Runs' in fin:
            skim_sumw = np.sum(fin['Runs'].array("genEventCount"))
    sumw = processor.value_accumulator(int)
    sumw += skim_sumw
    return processor.dict_accumulator({
        'sumw': processor.dict_accumulator({dataset: sumw}),
    })

def make_norm(args):
    with open(args.samplejson) as fin:
        samplefiles = json.load(fin)
    samplelist = samplefiles['2018']

    filelist = []
    for sample,sampleinfo in samplelist.items():
        for dataset, files in sampleinfo.items():
            if dataset == 'JetHT' or dataset == 'SingleMuon' or dataset == 'SingleElectron':
                continue
            for file in files:
                filelist.append((dataset, file))

    final_accumulator = processor.dict_accumulator({
        'sumw': processor.dict_accumulator(),
    })
    processor.futures_executor(filelist, get_norm, final_accumulator, workers=args.workers)

    save(final_accumulator['sumw'], 'correction_files/sumw_mc_2018.coffea')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Precompute MC weight') 
    parser.add_argument('--samplejson', default='metadata/htautaufiles_hadd.json', help='JSON file containing dataset and file locations (default: %(default)s)')
    parser.add_argument('-j', '--workers', type=int, default=8, help='Number of workers to use for multi-worker executors (e.g. futures or condor) (default: %(default)s)')
    args = parser.parse_args()

    make_norm(args)
