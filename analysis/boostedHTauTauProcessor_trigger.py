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

def nanoObject(df, prefix, columns):
    branches = set(prefix + k for k in columns)
    p4branches = [prefix + k for k in ['pt', 'eta', 'phi', 'mass']]
    branches -= set(p4branches)
    objp4 = uproot_methods.TLorentzVectorArray.from_ptetaphim(*[df[b] for b in p4branches])
    branches = {k[len(prefix):]: df[k] for k in branches}
    obj = awkward.JaggedArray.zip(p4=objp4, **branches)
    return obj

def flattenAndPad(var,val=-1):
    return var.pad(1, clip=True).fillna(val).regular().flatten()

def match(obj0, obj1):
    obj = obj0['p4'].cross(obj1['p4'])
    return obj.i0.delta_r(obj.i1)

class BoostedHTauTauProcessorTrigger(processor.ProcessorABC):
    def __init__(self, corrections, debug=False, year='2018'):
        self._corrections = corrections
        self._debug = debug
        self._year = year

        dataset_axis = hist.Cat("dataset", "Primary dataset")
        metpt_axis = hist.Bin("MET_pt", r"MET $p_T$", 20, 0, 100)
        genHpt_axis = hist.Bin("genH_pt", r"gen Higgs $p_T$",40, 100, 800)
        genElept_axis = hist.Bin("genEle_pt", r"gen Electron $p_T$",20, 20, 400)
        genMupt_axis = hist.Bin("genMu_pt", r"gen Muon $p_T$",20, 20, 400)

        ele0pt_axis = hist.Bin("ele0_pt", r"Electron $p_T$", 20, 20, 400)
        mu0pt_axis = hist.Bin("mu0_pt", r"Muon $p_T$", 20, 20, 400)

        jet0lsf3_axis = hist.Bin("jet0_lsf3", r"Leading Jet LSF_3", 20, 0, 1)
        jet0pt_axis = hist.Bin("jet0_pt", r"Leading Jet $p_T$", 20, 200, 1000)

        hists = processor.dict_accumulator()
        hist.Hist.DEFAULT_DTYPE = 'f'  # save some space by keeping float bin counts instead of double
        hists['sumw'] = processor.defaultdict_accumulator(int)
        for key in ['hadseljet0Trignone','hadseljet0Trighad']:
            hists[key] = hist.Hist("Events / GeV",
                                   dataset_axis,
                                   jet0pt_axis,
                                   genHpt_axis,
                                   )
        for key in ['eleseljet0Trignone','eleseljet0Trighad','eleseljet0Trigele','eleseljet0Trigvvlele']:
            hists[key] = hist.Hist("Events / GeV",
                                   dataset_axis,
                                   ele0pt_axis,
                                   genHpt_axis,
                                   genElept_axis,
                                   )

        for key in ['museljet0Trignone','museljet0Trighad','museljet0Trigmu','museljet0Trigvvlmu','museljet0Trigbtagmu','museljet0Trigmet']:
            hists[key] = hist.Hist("Events / GeV",
                                   dataset_axis,
                                   mu0pt_axis,
                                   genHpt_axis,
                                   genMupt_axis,
                                   metpt_axis,
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
        
        genParticles = nanoObject(df, 'GenPart_', ['pt','eta','phi','mass','pdgId','statusFlags','status'])

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

        df['genH_pt'] = flattenAndPad(genHiggs_showered['p4'].pt)
        df['genMu_pt'] = flattenAndPad(genMufromW['p4'].pt)
        df['genEle_pt'] = flattenAndPad(genElefromW['p4'].pt)
        
        ishWW_qqelev = (genHiggs_showered.counts==1) & (genW.counts==2) & (genQfromW.counts==2) & (genElefromW.counts==1) & (genEleNufromW.counts==1)
        ishWW_qqmuv = (genHiggs_showered.counts==1) & (genW.counts==2) & (genQfromW.counts==2) & (genMufromW.counts==1) & (genMuNufromW.counts==1)
        ishWW_qqqq = (genHiggs_showered.counts==1) & (genW.counts==2) & (genQfromW.counts==4)

        df['genH_decay'] = np.zeros(shape=(df.size))
        df['genH_decay'][ishWW_qqqq] = 1
        df['genH_decay'][ishWW_qqelev] = 2 
        df['genH_decay'][ishWW_qqmuv] = 3

    def build_leadingelectron_variables(self, df):
        leadingele = self._electrons[:, 0:1]
        df['ele0_pt'] = flattenAndPad(leadingele['p4'].pt)

    def build_leadingmuon_variables(self, df):
        leadingmu = self._muons[:, 0:1]
        df['mu0_pt'] = flattenAndPad(leadingmu['p4'].pt)

    def build_jet_variables(self, df, ij=0):
        jet = self._jetsAK8[:, ij:ij+1]
        df['jet%i_lsf3'%ij] = flattenAndPad(jet['lsf3'])
        df['jet%i_pt'%ij] = flattenAndPad(jet['p4'].pt)
        df['jet%i_msd'%ij] = flattenAndPad(jet['msoftdrop'])

    def process(self, df):
        dataset = df['dataset']
        print("Processing dataframe from", dataset)

        self._jetsAK8 = nanoObject(df, 'CustomAK8Puppi_', ['pt', 'eta', 'phi', 'mass','lsf3','msoftdrop'])
        self._electrons = nanoObject(df, 'Electron_', ['pt', 'eta', 'phi', 'mass'])
        self._muons = nanoObject(df, 'Muon_', ['pt', 'eta', 'phi', 'mass'])
        
        # construct objects
        leadingjet = self._jetsAK8[:, 0:1]
        subleadingjet = self._jetsAK8[:, 1:2]
        leadingele = self._electrons[:, 0:1]
        leadingmu = self._muons[:, 0:1]

        good_leadingjet = (leadingjet['p4'].pt > 300 & (abs(leadingjet['p4'].eta) < 2.4))
        good_electron = (leadingele['p4'].pt > 30)
        good_muon = (leadingmu['p4'].pt > 30)

        self.buildGenVariables(df)
        self.build_leadingelectron_variables(df)
        self.build_leadingmuon_variables(df)
        self.build_jet_variables(df,0)

        # trigger
        trigger = {}
        trigger['had'] = ['HLT_PFHT1050',
                          'HLT_AK8PFJet400_TrimMass30',
                          'HLT_AK8PFJet420_TrimMass30',
                          'HLT_AK8PFHT800_TrimMass50',
                          'HLT_PFJet500',
                          'HLT_AK8PFJet500']
        trigger['ele'] = ['HLT_Ele27_WPTight_Gsf','HLT_Ele40_WPTight_Gsf','HLT_Ele20_WPLoose_Gsf','HLT_Ele115_CaloIdVT_GsfTrkIdT']
        trigger['mu'] = ['HLT_Mu50','HLT_Mu55']
        trigger['mu_loose'] = ['HLT_Mu15_IsoVVVL_PFHT600','HLT_Mu50_IsoVVVL_PFHT450']
        trigger['ele_loose'] = ['HLT_Ele15_IsoVVVL_PFHT450','HLT_Ele15_IsoVVVL_PFHT600']
        trigger['met'] = ['HLT_PFMETNoMu110_PFMHTNoMu110_IDTight','HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60','HLT_PFMETNoMu120_PFMHTNoMu120_IDTight']
        trigger['btagmu'] = ['HLT_BTagMu_AK8Jet300_Mu5']

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

        # trigger selection
        for key,trig in trigger_selection.items():
            selection.add('trigger_%s'%key,trig)

        # object selection
        selection.add('good_ele',good_electron.any())
        selection.add('good_mu',good_muon.any())
        selection.add('good_jet0',good_leadingjet.any())

        regions = {}
        regions['noselection'] = {}

        regions['hadseljet0Trignone'] = {'qqqq','good_jet0'}
        regions['hadseljet0Trighad'] = {'qqqq','good_jet0','trigger_had'}

        regions['eleseljet0Trignone'] = {'qqelev','good_ele','good_jet0'}
        regions['eleseljet0Trighad'] = {'qqelev','good_ele','good_jet0','trigger_had'}
        regions['eleseljet0Trigele'] = {'qqelev','good_ele','good_jet0','trigger_ele'}
        regions['eleseljet0Trigvvlele'] = {'qqelev','good_ele','good_jet0','trigger_ele_loose'}

        regions['museljet0Trignone'] = {'qqmuv','good_mu', 'good_jet0'} 
        regions['museljet0Trighad'] = {'qqmuv','good_mu', 'good_jet0','trigger_had'}
        regions['museljet0Trigmu'] = {'qqmuv','good_mu', 'good_jet0','trigger_mu'}
        regions['museljet0Trigvvlmu'] = {'qqmuv','good_mu', 'good_jet0','trigger_mu_loose'}
        regions['museljet0Trigbtagmu'] = {'qqmuv','good_mu', 'good_jet0','trigger_btagmu'}
        regions['museljet0Trigmet'] = {'qqmuv','good_mu', 'good_jet0','trigger_met'}

        hout = self.accumulator.identity()
        for histname, h in hout.items():
            if not isinstance(h, hist.Hist):
                continue
            if not all(k in df or k == 'systematic' for k in h.fields):
                # Cannot fill this histogram due to missing fields
                # is this an error, warning, or ignorable?
                print("Missing fields %r from %r" % (set(h.fields) - set(df.keys()), h))
                continue
            fields = {k: df[k] for k in h.fields if k in df}
            region = [r for r in regions.keys() if r in histname.split('_')]

            print(region,histname.split('_'))
            if len(region) == 1:
                region = region[0]
                print(regions[region])
                cut = selection.all(*regions[region])
                print(cut.any())
                h.fill(**fields, weight=cut)
            elif len(region) > 1:
                raise ValueError("Histogram '%s' has a name matching multiple region definitions: %r" % (histname, region))
            else:
                raise ValueError("Histogram '%s' does not fall into any region definitions." % (histname, ))

        return hout

    def postprocess(self, accumulator):
        # set everything to 1/fb scale
        lumi = 1000  # [1/pb]
        
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

        return accumulator


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Boosted Htautau processor')
    parser.add_argument('--year', choices=['2016', '2017', '2018'], default='2017', help='Which data taking year to correct MC to.')
    parser.add_argument('--debug', action='store_true', help='Enable debug printouts')
    parser.add_argument('--externalSumW', help='Path to external sum weights file (if provided, will be used in place of self-determined sumw)', default='correction_files/sumw_mc_2018.coffea')
    args = parser.parse_args()

    corrections = load('corrections.coffea')

    #if args.externalSumW is not None:
    #    corrections['sumw_external'] = load(args.externalSumW)

    processor_instance = BoostedHTauTauProcessorTrigger(corrections=corrections,
                                                   debug=args.debug,
                                                   year=args.year,
                                                   )

    save(processor_instance, 'boostedHTauTauProcessor_trigger.coffea')
