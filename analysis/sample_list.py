import os
import math
from array import array
from optparse import OptionParser
import sys
import json

def get2018files():
    idirCristina = 'root://cmseos.fnal.gov//store/user/cmantill/pancakes/01/'
    idirJavier = 'root://cmseos.fnal.gov//store/user/jduarte1/pancakes/01/'
    idirLpchbb = 'root://cmseos.fnal.gov//store/user/lpchbb/cmantill/pancakes/01/'
    idirJeff = 'root://cmseos.fnal.gov//store/user/jkrupa/nano/'
    idirLxplus = 'root://eoscms.cern.ch//store/group/phys_exotica/dijet/dazsle/hww/'

    tfiles = {
        'wqq': {'samples': ['WJetsToQQ_HT400to600_qc19_3j_TuneCP5_13TeV-madgraphMLM-pythia8',
                            'WJetsToQQ_HT600to800_qc19_3j_TuneCP5_13TeV-madgraphMLM-pythia8',
                            'WJetsToQQ_HT-800toInf_qc19_3j_TuneCP5_13TeV-madgraphMLM-pythia8'],
                'dir': idirJeff,
                'path': 'wjets-01_RunIIAutumn18MiniAOD-102X_v15-v1',
            },
        'wlnu': {'samples': ['WJetsToLNu_HT-100To200_TuneCP5_13TeV-madgraphMLM-pythia8',
                             'WJetsToLNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8',
                             'WJetsToLNu_HT-400To600_TuneCP5_13TeV-madgraphMLM-pythia8',
                             'WJetsToLNu_HT-600To800_TuneCP5_13TeV-madgraphMLM-pythia8',
                             'WJetsToLNu_HT-800To1200_TuneCP5_13TeV-madgraphMLM-pythia8',
                             'WJetsToLNu_HT-1200To2500_TuneCP5_13TeV-madgraphMLM-pythia8',
                             'WJetsToLNu_HT-2500ToInf_TuneCP5_13TeV-madgraphMLM-pythia8',
                         ],
                 'dir': idirJeff,
                 'path': 'wjets-lnu-01_RunIIAutumn18MiniAOD-102X_v15-v1',
             },
        'zqq': {'samples': ['ZJetsToQQ_HT400to600_qc19_4j_TuneCP5_13TeV-madgraphMLM-pythia8',
                            'ZJetsToQQ_HT600to800_qc19_4j_TuneCP5_13TeV-madgraphMLM-pythia8',
                        ],
                'dir': idirJeff,
                'path': 'zjetsqq-01_RunIIAutumn18MiniAOD-102X_v15-v1',
            },
        'zqq_800': {'samples': ['ZJetsToQQ_HT-800toInf_qc19_4j_TuneCP5_13TeV-madgraphMLM-pythia8'],
                    'dir': idirJeff,
                    'path': 'zqqjets-01_RunIIAutumn18MiniAOD-102X_v15-v1',
                },
        'qcd': {'samples': ['QCD_HT500to700_TuneCP5_13TeV-madgraphMLM-pythia8',
                            'QCD_HT700to1000_TuneCP5_13TeV-madgraphMLM-pythia8',
                            'QCD_HT1000to1500_TuneCP5_13TeV-madgraphMLM-pythia8',
                            'QCD_HT1500to2000_TuneCP5_13TeV-madgraphMLM-pythia8',
                            'QCD_HT2000toInf_TuneCP5_13TeV-madgraphMLM-pythia8',
                        ],
                'dir': idirJavier,
                'path': 'pancakes-01_RunIIAutumn18MiniAOD-102X_v15-v1',
            },
        'tt': {'samples': ['TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8',
                           'TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8',
                           'TTToHadronic_TuneCP5_13TeV-powheg-pythia8',
                       ],
               'dir': idirLpchbb,
               'path': 'pancakes-01_RunIIAutumn18MiniAOD-102X_v15-v1',
           },
        'st': {'samples': ['ST_tW_antitop_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8',
                           'ST_tW_top_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8',
                           'ST_t-channel_antitop_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8',
                           'ST_s-channel_4f_hadronicDecays_TuneCP5_13TeV-madgraph-pythia8',
                       ],
               'dir': idirLpchbb,
               'path': 'pancakes-01_RunIIAutumn18MiniAOD-102X_v15-v1-ext1',
           },
        'vv': {'samples': ['WW_TuneCP5_13TeV-pythia8',
                           'WZ_TuneCP5_13TeV-pythia8',
                           'ZZ_TuneCP5_13TeV-pythia8',
                       ],
               'dir': idirLpchbb,
               'path': 'pancakes-01_RunIIAutumn18MiniAOD-102X_v15-v2',
           },
        'h125': {'samples':['hww_mc',
                            'hwwmc',
                        ],
                 'dir': idirLxplus,
                 'path': 'NanoAOD'
             },
    }

    return tfiles

def read(redirector,eosp):
    try:
        #print('xrdfs %s ls %s > tmp.txt'%(redirector,eosp))
        os.system('xrdfs %s ls %s > tmp.txt'%(redirector,eosp))
        with open("tmp.txt") as f: lineList = f.read().splitlines()
        return lineList
    except:
        #print('empty ')
        return []

def readXroot(redirector,eosp):
    expandList = []
    lineList = read(redirector,eosp)
    if any(('.root' in it and 'NanoAOD' in it) for it in lineList):
        new_f = [it.replace('/store',redirector+'/store') for it in lineList]
        expandList.extend(new_f)
    else:
        for f in lineList:
            loop = True
            if any(('.root' in it and 'NanoAOD' in it) for it in f): 
                loop = False
            while loop:
                if type(f) == type([]):
                    newf = read(redirector,f[0])
                else:
                    newf = read(redirector,f)
                if len(newf)==0: 
                    loop = False
                else: 
                    f = newf
                    if any('.root' in it for it in f):
                        loop = False
            new_f = [it.replace('/store',redirector+'/store') for it in f]
            expandList.extend(new_f)
    return expandList

def expand(path,idir,midpath):
    expandedPaths = []
    redirector = 'root://cmseos.fnal.gov/'
    if 'cern' in idir: redirector = 'root://eoscms.cern.ch/'
    eosp = idir.replace(redirector,'')+'/'+path+'/'+midpath
    new_content = readXroot(redirector,eosp)
    expandedPaths.extend(new_content)
    return expandedPaths 

def expandPath(dicts):
    rdict = {}
    for sample,sampleDict in dicts.iteritems():
        d={} 
        print(sample)
        for subSname in sampleDict['samples']:
            print(subSname,sampleDict['dir'],sampleDict['path'])
            expandedPath = expand(subSname,sampleDict['dir'],sampleDict['path']) 
            if len(expandedPath)==0:
                print "ERROR: %s has no files"%(subSname)
                print "Trying to expand path with %s"%sampleDict['path']
            d[subSname] = expandedPath 
        rdict[sample] =  d
    return rdict

def diffDict(loadedJson,finaljson):
    if loadedJson == finaljson:
        print "No changes to samplefiles.json detected" 
        return
    else:
        diffed_json = {}
        changed_subsamples = []
        for fset in  finaljson.keys():
            diffed_json[fset] = copy.deepcopy(finaljson[fset])
            #if len(finaljson[fset]) is not len(loadedJson[fset]):
            #    print "list of sample changed."
            samples       = finaljson[fset]
            if fset in loadedJson.keys():
                loadedsamples = loadedJson[fset]
            else:
                loadedsamples = samples
            for sample,subsample in samples.iteritems():
                if type(subsample)==type([]):
                    ## ignore data 
                    diffed_json[fset][sample] = []
                elif type(subsample)==type({}):
                    loadedsubsamples = loadedsamples[sample]
                    for s,paths in subsample.iteritems():
                        if s in loadedsubsamples.keys():
                            loadedpaths = loadedsubsamples[s]
                        else:
                            print s,'is different'
                            changed_subsamples.append(s)
                            diffed_json[fset][sample][s] = paths 
                        if len(loadedpaths) != len(paths):
                            print s,'is different'
                            changed_subsamples.append(s)
                            diffed_json[fset][sample][s] = paths 
                        else:
                            diffed_json[fset][sample][s] = []
        #print json.dumps(diffed_json,indent=4)
        return  diffed_json,changed_subsamples

def main(options,args):
    #outf = open("metadata/samplefiles.json","r")
    #loadedJson = json.load(outf)
    finaljson = {}
    finaljson['2018'] = expandPath(get2018files())
    # print "LoadedJson == new json: ", loadedJson == finaljson
    '''
    updateNorms = True 
    remakeAllnorms = options.remakeAllnorms
    if loadedJson != finaljson and updateNorms :
        diffed_json, changed_subsamples = diffDict(loadedJson,finaljson)
        #print "Following subsamples are changed:"
        #for s in changed_subsamples: print s
        makeNormRoot(diffed_json)
    if remakeAllnorms:
        makeNormRoot(finaljson,remakeAllnorms=True)
    '''

    #if not options.printOnly and finaljson is not {}:
    outf = open("metadata/samplefiles.json","w")
    outf.write((json.dumps(finaljson,indent=4)))
    #else:
    #    print (json.dumps(finaljson,indent=4,sort_keys=True))
    for key,tfiles in sorted(finaljson.iteritems()):
        print "list of samples used by %s =  "%key, sorted(tfiles.keys())


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('-p','--printOnly', dest='printOnly',action='store_true', default=False,help='print json to screen only', metavar='printOnly')
    parser.add_option('--remakeAllnorms', dest='remakeAllnorms',action='store_true', default=False,help='Remake all PU histograms and Nevent ', metavar='printOnly')
    (options, args) = parser.parse_args()
    main(options,args)
