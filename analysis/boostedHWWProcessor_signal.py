#!/usr/bin/env python
import pprint
import numpy as np
from coffea import hist, processor
from coffea.util import load, save
import argparse
import warnings
import uproot
import uproot_methods
import awkward

def deltaphi(a, b):
    return (a - b + np.pi)%(2*np.pi) - np.pi

def nanoObject(df, prefix):
    branches = set(k.decode('ascii') for k in df.available if k.decode('ascii').startswith(prefix))
    p4branches = [prefix + k for k in ['pt', 'eta', 'phi', 'mass']]
    branches -= set(p4branches)
    objp4 = uproot_methods.TLorentzVectorArray.from_ptetaphim(*[df[b] for b in p4branches])
    branches = {k[len(prefix):]: df[k] for k in branches}
    obj = awkward.JaggedArray.zip(p4=objp4, **branches)
    return obj

def flattenAndPad(var,dfsize, val=-1):
    try: 
        x = var.pad(1, clip=True).fillna(val).regular().flatten()
    except:
        x = np.zeros(shape=(dfsize))
    return x

class BoostedHWWProcessor(processor.ProcessorABC):
    def __init__(self, corrections, debug=False, year='2018'):
        self._corrections = corrections
        self._debug = debug
        self._year = year

        dataset_axis = hist.Cat("dataset", "Primary dataset")
        genHdecay_axis = hist.Bin("genH_decay",r"h $\rightarrow WW^*$ decay",6,0,6)

        genHpt_axis = hist.Bin("genH_pt", r"gen Higgs $p_T$",50, 100, 800)
        genElept_axis = hist.Bin("genEle_pt", r"gen Electron $p_T$",50, 50, 400)
        genMupt_axis = hist.Bin("genMu_pt", r"gen Muon $p_T$",50, 50, 400)

        elept0_axis = hist.Bin("ele0_pt", r"Electron $p_T$", 20, 0, 400)
        mupt0_axis = hist.Bin("mu0_pt", r"Muon $p_T$", 20, 0, 400)

        hists = processor.dict_accumulator()
        hist.Hist.DEFAULT_DTYPE = 'f'  # save some space by keeping float bin counts instead of double
        hists['sumw'] = processor.defaultdict_accumulator(int)
        for key in ['hadPreselection','hadTrigHad']:
            hists[key] = hist.Hist("Events / GeV",
                                   genHpt_axis,
                               )
        for key in ['elePreselection','eleTrigEle','eleTrigHad']: 
            hists[key] = hist.Hist("Events / GeV",
                                   elept0_axis,
                                   genHpt_axis,
                                   genElept_axis
                               )
        for key in ['muPreselection','muTrigMu','muTrigHad']:
            hists[key] = hist.Hist("Events / GeV",
                                   mupt0_axis,
                                   genHpt_axis,
                                   genMupt_axis
                               )

        self._accumulator = hists

    @property
    def accumulator(self):
        return self._accumulator

    def buildGenVariables(self,df):
        _gen_statusFlags = {
            0: 'isPrompt',
            1: 'isDecayedLeptonHadron',
            2: 'isTauDecayProduct',
            3: 'isPromptTauDecayProduct',
            4: 'isDirectTauDecayProduct',
            5: 'isDirectPromptTauDecayProduct',
            6: 'isDirectHadronDecayProduct',
            7: 'isHardProcess',
            8: 'fromHardProcess',
            9: 'isHardProcessTauDecayProduct',
            10: 'isDirectHardProcessTauDecayProduct',
            11: 'fromHardProcessBeforeFSR',
            12: 'isFirstCopy',
            13: 'isLastCopy',
            14: 'isLastCopyBeforeFSR'
        }        
        
        def statusmask(array, require):
            mask = sum((1<<k) for k,v in _gen_statusFlags.items() if v in require)
            return (array & mask)==mask
        
        genParticles = nanoObject(df, 'GenPart_')

        hidx_showered = (genParticles['pdgId']==25) & statusmask(genParticles['statusFlags'], {'fromHardProcess', 'isLastCopy'})
        widx = (abs(genParticles['pdgId'])==24) & statusmask(genParticles['statusFlags'], {'fromHardProcess', 'isLastCopy'})
        qidx = (abs(genParticles['pdgId'])>=1) & (abs(genParticles['pdgId'])<=5) & statusmask(genParticles['statusFlags'], {'fromHardProcess'}) & (genParticles['status']==23)
        eleidx = (abs(genParticles['pdgId'])==11) & statusmask(genParticles['statusFlags'], {'fromHardProcess','isFirstCopy'}) & (genParticles['status']==1)
        elenuidx = (abs(genParticles['pdgId'])==12) & statusmask(genParticles['statusFlags'], {'fromHardProcess','isFirstCopy'}) & (genParticles['status']==1)
        muidx = (abs(genParticles['pdgId'])==13) & statusmask(genParticles['statusFlags'], {'fromHardProcess','isFirstCopy'}) & (genParticles['status']==1)
        munuidx = (abs(genParticles['pdgId'])==14) & statusmask(genParticles['statusFlags'], {'fromHardProcess','isFirstCopy'}) & (genParticles['status']==1)
        
        genHiggs_showered = genParticles[hidx_showered]
        genW = genParticles[widx]
        genQfromW = genParticles[qidx]
        genElefromW = genParticles[eleidx]
        genEleNufromW = genParticles[elenuidx]
        genMufromW = genParticles[muidx]
        genMuNufromW = genParticles[munuidx]

        df['genH_pt'] = flattenAndPad(genHiggs_showered['p4'].pt, df.size)
        df['genMu_pt'] = flattenAndPad(genMufromW['p4'].pt, df.size)
        df['genEle_pt'] = flattenAndPad(genElefromW['p4'].pt, df.size)

        ishWW_qqelev = (genHiggs_showered.counts==1) & (genW.counts==2) & (genQfromW.counts==2) & (genElefromW.counts==1) & (genEleNufromW.counts==1)
        ishWW_qqmuv = (genHiggs_showered.counts==1) & (genW.counts==2) & (genQfromW.counts==2) & (genMufromW.counts==1) & (genMuNufromW.counts==1)
        ishWW_qqqq = (genHiggs_showered.counts==1) & (genW.counts==2) & (genQfromW.counts==4)

        df['genH_decay'] = np.zeros(shape=(df.size))
        df['genH_decay'][ishWW_qqqq] = 1
        df['genH_decay'][ishWW_qqelev] = 2 
        df['genH_decay'][ishWW_qqmuv] = 3

        self._maphid = {'0': 'other',
                        '1': r'h$\rightarrow WW^*(qqqq)$',
                        '2': r'h$\rightarrow WW^*(e\nu_{e}qq)$',
                        '3': r'h$\rightarrow WW^*(\mu\nu_{\mu}qq)$',
                    }

    def process(self, df):
        dataset = df['dataset']
        if self._debug:
            print("Processing dataframe from", dataset)

        jetsAK8 = nanoObject(df, 'CustomAK8Puppi_')
        electrons = nanoObject(df, 'Electron_')
        muons = nanoObject(df, 'Muon_')
        
        self.buildGenVariables(df)

        # construct objects
        leadingjet = jetsAK8[:, 0:1]
        leadingele = electrons[:, 0:1]
        leadingmu = muons[:, 0:1]

        good_leadingjet = ((leadingjet['p4'].eta < 2.4) & (leadingjet['p4'].pt > 300))
        good_electron = (leadingele['p4'].pt > 115)
        good_muon = (leadingmu['p4'].pt > 50)
        
        df['ele0_pt'] = flattenAndPad(leadingele['p4'].pt, df.size)
        df['mu0_pt'] = flattenAndPad(leadingmu['p4'].pt, df.size)

        # trigger
        trigger = {}
        trigger['had'] = ['HLT_PFHT1050','HLT_AK8PFJet400_TrimMass30','HLT_AK8PFJet420_TrimMass30','HLT_AK8PFHT800_TrimMass50','HLT_PFJet500','HLT_AK8PFJet500','HLT_AK8PFJet330_BTagCSV_p17']
        trigger['ele'] = ['HLT_Ele27_WPTight_Gsf','HLT_Ele40_WPTight_Gsf','HLT_Ele20_WPLoose_Gsf','HLT_Ele115_CaloIdVT_GsfTrkIdT']
        trigger['mu'] = ['HLT_Mu50','HLT_Mu55']
        trigger_selection = {}
        for key,trig in trigger.items():
            trigger_selection[key] = False
            for path in trig:
                if path in df:
                    trigger_selection[key] |= df[path]
        
        # selection
        selection = processor.PackedSelection()
        selection.add('qqqq',df['genH_decay']==1)
        selection.add('qqelev',df['genH_decay']==2)
        selection.add('qqmuv',df['genH_decay']==3)
        for key,trig in trigger_selection.items():
            selection.add('trigger_%s'%key,trig)

        selection.add('good_1jet',good_leadingjet.any())
        selection.add('good_1ele',good_electron.any())
        selection.add('good_1mu',good_muon.any())

        regions = {}
        regions['noselection'] = {}
        regions['hadPreselection'] = {'qqqq','good_1jet'}
        regions['hadTrigHad'] = {'qqqq','good_1jet','trigger_had'}
        regions['elePreselection'] = {'qqelev','good_1jet','good_1ele'}
        regions['eleTrigEle'] = {'qqelev','good_1jet','good_1ele','trigger_ele'}
        regions['eleTrigHad'] = {'qqelev','good_1jet','good_1ele','trigger_had'}
        regions['muPreselection'] = {'qqmuv','good_1jet','good_1mu'}
        regions['muTrigMu'] = {'qqmuv','good_1jet','good_1mu','trigger_mu'}
        regions['muTrigHad'] = {'qqmuv','good_1jet','good_1mu','trigger_had'}

        # weights

        hout = self.accumulator.identity()
        for histname, h in hout.items():
            if not isinstance(h, hist.Hist):
                continue
            if not all(k in df or k == 'systematic' for k in h.fields):
                # Cannot fill this histogram due to missing fields
                # is this an error, warning, or ignorable?
                if self._debug:
                    print("Missing fields %r from %r" % (set(h.fields) - set(df.keys()), h))
                continue
            fields = {k: df[k] for k in h.fields if k in df}
            region = [r for r in regions.keys() if r in histname.split('_')]

            if len(region) == 1:
                region = region[0]
                cut = selection.all(*regions[region])
                h.fill(**fields, weight=cut)
            elif len(region) > 1:
                raise ValueError("Histogram '%s' has a name matching multiple region definitions: %r" % (histname, region))
            else:
                raise ValueError("Histogram '%s' does not fall into any region definitions." % (histname, ))

        return hout

    def postprocess(self, accumulator):
        # set everything to 1/fb scale
        lumi = 1000  # [1/pb]
        
        '''
        if 'sumw_external' in self._corrections:
            normlist = self._corrections['sumw_external']
            for key in accumulator['sumw'].keys():
                accumulator['sumw'][key] = normlist[key].value

        scale = {}
        for dataset, dataset_sumw in accumulator['sumw'].items():
            if dataset in self._corrections['xsections']:
                scale[dataset] = lumi*self._corrections['xsections'][dataset]/dataset_sumw
            else:
                warnings.warn("Missing cross section for dataset %s.  Normalizing to 1 pb" % dataset, RuntimeWarning)
                scale[dataset] = lumi / dataset_sumw
            
        for h in accumulator.values():
            if isinstance(h, hist.Hist):
                h.scale(scale, axis="dataset")
        '''
        return accumulator


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Boosted Hbb processor')
    parser.add_argument('--year', choices=['2016', '2017', '2018'], default='2017', help='Which data taking year to correct MC to.')
    parser.add_argument('--debug', action='store_true', help='Enable debug printouts')
    parser.add_argument('--externalSumW', help='Path to external sum weights file (if provided, will be used in place of self-determined sumw)')
    args = parser.parse_args()

    corrections = load('corrections.coffea')

    if args.externalSumW is not None:
        corrections['sumw_external'] = load(args.externalSumW)

    processor_instance = BoostedHWWProcessor(corrections=corrections,
                                             debug=args.debug,
                                             year=args.year,
                                             )

    save(processor_instance, 'boostedHWWProcessor_signal.coffea')
