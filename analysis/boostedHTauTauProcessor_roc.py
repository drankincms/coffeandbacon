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

# ROC curves for leptons that pass ID and impact parameter cuts
# leptons selection:                                                                                                                                                              
# pT > 27 (30) GeV, |η| < 2.4 (1.479) for muons (electrons)                                                                                                                    
# SIP < 4, |dz| < 0.1, |d0| < 0.05                                                                                                                                                                         
# ID: Medium (MVA90) for muons (electrons)                                                                                                                                    
# signal: match reco lepton to gen lepton

# jets selection:
# pT> 300 GeV, |η| < 2.4 
# signal match jet (either jet0 or jet1 selection) to higgs jet & reco lepton

class BoostedHTauTauProcessorROC(processor.ProcessorABC):
    def __init__(self, corrections, debug=False, year='2018'):
        self._corrections = corrections
        self._debug = debug
        self._year = year

        dataset_axis = hist.Cat("dataset", "Primary dataset")

        ele0miso_axis = hist.Bin("ele0_miso", r"Electron mini PF ISO (total)", 60, 0, 1)
        ele0iso_axis = hist.Bin("ele0_iso", r"Electron PF ISO (total)", 60, 0, 1)

        mu0miso_axis = hist.Bin("mu0_miso", r"Muon mini PF ISO (total)", 60, 0, 1)
        mu0iso_axis = hist.Bin("mu0_iso", r"Muon PF ISO (total)", 60, 0, 1)

        jet0lsf3_axis = hist.Bin("jet0_lsf3", r"Leading Jet LSF_3", 60, 0, 1)
        jet1lsf3_axis = hist.Bin("jet1_lsf3", r"Sub-Leading Jet LSF_3", 60, 0, 1)

        hists = processor.dict_accumulator()
        hist.Hist.DEFAULT_DTYPE = 'f'  # save some space by keeping float bin counts instead of double
        hists['sumw'] = processor.defaultdict_accumulator(int)
        hists['roc_eleseljet0'] = hist.Hist("Events / GeV",
                                            dataset_axis,
                                            ele0miso_axis,
                                            ele0iso_axis,
                                            jet0lsf3_axis
                                            )
        
        hists['roc_museljet0'] = hist.Hist("Events / GeV",
                                           dataset_axis,
                                           mu0miso_axis,
                                           mu0iso_axis,
                                           jet0lsf3_axis
                                           )


        hists['roc_eleseljet1'] = hist.Hist("Events / GeV",
                                            dataset_axis,
                                            ele0miso_axis,
                                            ele0iso_axis,
                                            jet1lsf3_axis
                                            )

        hists['roc_museljet1'] = hist.Hist("Events / GeV",
                                           dataset_axis,
                                           mu0miso_axis,
                                           mu0iso_axis,
                                           jet1lsf3_axis
                                           )

        self._accumulator = hists

    @property
    def accumulator(self):
        return self._accumulator

    def build_leadingelectron_variables(self, df):
        leadingele = self._electrons[:, 0:1]
        df['ele0_pt'] = flattenAndPad(leadingele['p4'].pt)
        df['ele0_miso'] = flattenAndPad(leadingele.miniPFRelIso_all)
        df['ele0_iso'] = flattenAndPad(leadingele.pfRelIso03_all)

    def build_leadingmuon_variables(self, df):
        leadingmu = self._muons[:, 0:1]
        df['mu0_pt'] = flattenAndPad(leadingmu['p4'].pt)
        df['mu0_miso'] = flattenAndPad(leadingmu.miniPFRelIso_all)
        df['mu0_iso'] = flattenAndPad(leadingmu.pfRelIso03_all)

    def build_jet_variables(self, df, ij=0):
        jet = self._jetsAK8[:, ij:ij+1]
        df['jet%i_lsf3'%ij] = flattenAndPad(jet['lsf3'])

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
        
        genParticles = nanoObject(df, 'GenPart_', ['pt', 'eta', 'phi', 'mass','statusFlags','pdgId','status'])

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

        #df['genH_pt'] = flattenAndPad(genHiggs_showered['p4'].pt)
        #df['genMu_pt'] = flattenAndPad(genMufromW['p4'].pt)
        #df['genEle_pt'] = flattenAndPad(genElefromW['p4'].pt)

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

        df['ele0_genEledR'] = flattenAndPad(match(self._electrons[:, 0:1], genElefromW[:, 0:1]))
        df['mu0_genMudR'] = flattenAndPad(match(self._muons[:, 0:1], genMufromW[:, 0:1]))

        df['jet0_genHdR'] = flattenAndPad(match(self._jetsAK8[:, 0:1], genHiggs_showered))
        df['jet1_genHdR'] = flattenAndPad(match(self._jetsAK8[:, 1:2], genHiggs_showered))

    def process(self, df):
        dataset = df['dataset']
        #if self._debug:
        print("Processing dataframe from", dataset)

        self._jetsAK8 = nanoObject(df, 'CustomAK8Puppi_', ['pt', 'eta', 'phi', 'mass','electronIdx3SJ','muonIdx3SJ','lsf3'])
        self._electrons = nanoObject(df, 'Electron_', ['pt', 'eta', 'phi', 'mass','sip3d','dxy','dz','mvaFall17V1noIso_WP90','pfRelIso03_all','miniPFRelIso_all'])
        self._muons = nanoObject(df, 'Muon_', ['pt', 'eta', 'phi', 'mass','sip3d','dxy','dz','mvaId','pfRelIso03_all','miniPFRelIso_all'])
        
        # construct objects
        leadingjet = self._jetsAK8[:, 0:1]
        subleadingjet = self._jetsAK8[:, 1:2]
        leadingele = self._electrons[:, 0:1]
        leadingmu = self._muons[:, 0:1]

        good_leadingjet = (leadingjet['p4'].pt > 300 & (abs(leadingjet['p4'].eta) < 2.4))
        good_subleadingjet = (subleadingjet['p4'].pt > 300 & (abs(subleadingjet['p4'].eta) < 2.4))
        good_electron = (leadingele['p4'].pt > 30 & (abs(leadingele['p4'].eta) < 1.479) & (leadingele['sip3d'] < 4) & (abs(leadingele['dz']) < 0.1) & (abs(leadingele['dxy']) < 0.05) & (leadingele['mvaFall17V1noIso_WP90']))
        good_muon = (leadingmu['p4'].pt > 50 & (abs(leadingmu['p4'].eta) < 2.4) & (leadingmu['sip3d'] < 4) & (abs(leadingmu['dz']) < 0.1) & (abs(leadingmu['dxy']) < 0.05) & (leadingmu['mvaId']==2))

        self.build_leadingelectron_variables(df)
        self.build_leadingmuon_variables(df)
        self.build_jet_variables(df,0)
        self.build_jet_variables(df,1)

        # selection
        selection = processor.PackedSelection()

        # match jet to reco lepton
        selection.add('good_jet0ele',(leadingjet['electronIdx3SJ']==0).any())
        selection.add('good_jet1ele',(subleadingjet['electronIdx3SJ']==0).any())
        selection.add('good_jet0mu',(leadingjet['muonIdx3SJ']==0).any())
        selection.add('good_jet1mu',(subleadingjet['muonIdx3SJ']==0).any())

        if 'htautau' in dataset:
            self.buildGenVariables(df)
            good_genEleEvt = ((df['genH_decay'] == 2) & (abs(df['ele0_genEledR']) < 0.2))
            good_genMuEvt = ((df['genH_decay'] == 3) & (abs(df['mu0_genMudR']) < 0.2))

            selection.add('good_ele',(good_electron & good_genEleEvt).any())
            selection.add('good_mu',(good_muon & good_genMuEvt).any())
            selection.add('good_jet0',(good_leadingjet & (abs(df['jet0_genHdR']) < 0.8)).any())
            selection.add('good_jet1',(good_subleadingjet & (abs(df['jet1_genHdR']) < 0.8)).any())
        else:
            selection.add('good_ele',good_electron.any())
            selection.add('good_mu',good_muon.any())
            selection.add('good_jet0',good_leadingjet.any())
            selection.add('good_jet1',good_subleadingjet.any())

        regions = {}
        regions['noselection'] = {}
        regions['eleseljet0'] = {'good_ele','good_jet0ele','good_jet0'}
        regions['eleseljet1'] = {'good_ele','good_jet1ele','good_jet1'}
        regions['museljet0'] = {'good_mu', 'good_jet0mu', 'good_jet0'} 
        regions['museljet1'] = {'good_mu', 'good_jet1mu', 'good_jet1'}

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
    parser = argparse.ArgumentParser(description='Boosted Htautau processor')
    parser.add_argument('--year', choices=['2016', '2017', '2018'], default='2017', help='Which data taking year to correct MC to.')
    parser.add_argument('--debug', action='store_true', help='Enable debug printouts')
    parser.add_argument('--externalSumW', help='Path to external sum weights file (if provided, will be used in place of self-determined sumw)')
    args = parser.parse_args()

    corrections = load('corrections.coffea')

    if args.externalSumW is not None:
        corrections['sumw_external'] = load(args.externalSumW)

    processor_instance = BoostedHTauTauProcessorROC(corrections=corrections,
                                                debug=args.debug,
                                                year=args.year,
                                                )

    save(processor_instance, 'boostedHTauTauProcessor_roc.coffea')
