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

# PreSel for hadronic and semi-leptonic HWW

# semi-lep: leptons that pass ID and impact parameter cuts
# leptons selection:                                                                                                                                                              
# pT > 27 (30) GeV, |η| < 2.4 (1.479) for muons (electrons)                                                                                                                    
# SIP < 4, |dz| < 0.1, |d0| < 0.05                                                                                                                                                                         
# ID: Medium (MVA90) for muons (electrons)                                                                                                                                    
# signal: match reco lepton to gen lepton

# jets selection:
# pT> 300 GeV, |η| < 2.4 
# signal match jet (either jet0 or jet1 selection) to higgs jet & reco lepton

class BoostedHWWProcessorPreSel(processor.ProcessorABC):
    def __init__(self, corrections, debug=False, year='2018'):
        self._corrections = corrections
        self._debug = debug
        self._year = year

        dataset_axis = hist.Cat("dataset", "Primary dataset")
        metpt_axis = hist.Bin("MET_pt", r"MET $p_T$", 20, 0, 100)
        #ht_axis = hist.Bin("ht",r"HT",20, 300, 1000) 

        ele0pt_axis = hist.Bin("ele0_pt", r"Electron $p_T$", 20, 0, 400)
        ele0miso_axis = hist.Bin("ele0_miso", r"Electron mini PF ISO (total)", 20, 0, 1)
        ele0dRjet0_axis = hist.Bin("ele0_dRjet0", r"Electron dR(Leading jet)", 20, 0, 4)
        ele0dRjet1_axis = hist.Bin("ele0_dRjet1", r"Electron dR(Sub-Leading jet)", 20, 0, 4)

        mu0pt_axis = hist.Bin("mu0_pt", r"Muon $p_T$", 20, 0, 400)
        mu0miso_axis = hist.Bin("mu0_miso", r"Muon mini PF ISO (total)", 20, 0, 1)
        mu0dRjet0_axis = hist.Bin("mu0_dRjet0", r"Muon dR(Leading jet)", 20, 0, 4)
        mu0dRjet1_axis = hist.Bin("mu0_dRjet1", r"Muon dR(Sub-Leading jet)", 20, 0, 4)

        jet0lsf3_axis = hist.Bin("jet0_lsf3", r"Leading Jet LSF_3", 20, 0, 1)
        jet0pt_axis = hist.Bin("jet0_pt", r"Leading Jet $p_T$", 20, 200, 1000)
        jet0mass_axis = hist.Bin("jet0_msd", r"Leading Jet $m_{sd}$", 20, 10, 200)
        jet0dHqqqq_axis = hist.Bin("jet0_dHqqqq",r"Leading Jet dAK8 H(4q)", 20, 0, 1)
        jet0dHqqqqMD_axis = hist.Bin("jet0_dHqqqqMD",r"Leading Jet dAK8 H(4q) MD", 20, 0, 1)
        jet1lsf3_axis = hist.Bin("jet1_lsf3", r"Sub-Leading Jet LSF_3", 20, 0, 1)
        jet1pt_axis = hist.Bin("jet1_pt", r"Sub-Leading Jet $p_T$", 20, 200, 1000)
        jet1mass_axis = hist.Bin("jet1_msd", r"Sub-Leading Jet $m_{sd}$", 20, 10, 200)

        hists = processor.dict_accumulator()
        hist.Hist.DEFAULT_DTYPE = 'f'  # save some space by keeping float bin counts instead of double
        hists['sumw'] = processor.defaultdict_accumulator(int)
        hists['presel_hadseljet0'] = hist.Hist("Events / GeV",
                                               dataset_axis,
                                               jet0pt_axis,
                                               jet0mass_axis,
                                               jet0dHqqqq_axis,
                                               jet0dHqqqqMD_axis,
                                               )

        hists['presel_eleseljet0'] = hist.Hist("Events / GeV",
                                               dataset_axis,
                                               ele0miso_axis,
                                               jet0lsf3_axis,
                                               jet0pt_axis,
                                               jet0mass_axis,
                                               ele0pt_axis,
                                               ele0dRjet0_axis,
                                               #ele0mjet0_axis,
                                               metpt_axis,
                                               )
        
        hists['presel_museljet0'] = hist.Hist("Events / GeV",
                                              dataset_axis,
                                              mu0miso_axis,
                                              jet0lsf3_axis,
                                              jet0pt_axis,
                                              jet0mass_axis,
                                              mu0pt_axis,
                                              mu0dRjet0_axis,
                                              metpt_axis,
                                              )

        
        hists['presel_eleseljet1'] = hist.Hist("Events / GeV",
                                               dataset_axis,
                                               ele0miso_axis,
                                               jet1lsf3_axis,
                                               jet1pt_axis,
                                               jet1mass_axis,
                                               ele0pt_axis,
                                               ele0dRjet1_axis,
                                               metpt_axis,
                                               )

        hists['presel_museljet1'] = hist.Hist("Events / GeV",
                                              dataset_axis,
                                              mu0miso_axis,
                                              jet1lsf3_axis,
                                              jet1pt_axis,
                                              jet1mass_axis,
                                              mu0pt_axis,
                                              mu0dRjet1_axis,
                                              metpt_axis,
                                              )

        self._accumulator = hists

    @property
    def accumulator(self):
        return self._accumulator

    def build_leadingelectron_variables(self, df):
        leadingele = self._electrons[:, 0:1]
        df['ele0_pt'] = flattenAndPad(leadingele['p4'].pt)
        df['ele0_miso'] = flattenAndPad(leadingele['miniPFRelIso_all'])

    def build_leadingmuon_variables(self, df):
        leadingmu = self._muons[:, 0:1]
        df['mu0_pt'] = flattenAndPad(leadingmu['p4'].pt)
        df['mu0_miso'] = flattenAndPad(leadingmu['miniPFRelIso_all'])

    def build_jet_variables(self, df, ij=0):
        jet = self._jetsAK8[:, ij:ij+1]
        df['jet%i_lsf3'%ij] = flattenAndPad(jet['lsf3'])
        df['jet%i_pt'%ij] = flattenAndPad(jet['p4'].pt)
        df['jet%i_mass'%ij] = flattenAndPad(jet['msoftdrop'])
        df['jet%i_dHqqqq'%ij] = flattenAndPad(jet['deepTagHqqqq'])
        df['jet%i_dHqqqqMD'%ij] = flattenAndPad(jet['deepTagMDHqqqq'])

    def build_leptonjet_variables(self, df):
        df['ele0_dRjet0'] = flattenAndPad(match(self._jetsAK8[:, 0:1], self._electrons[:, 0:1]))
        df['ele0_dRjet1'] = flattenAndPad(match(self._jetsAK8[:, 1:2], self._electrons[:, 0:1]))

        df['mu0_dRjet0'] = flattenAndPad(match(self._jetsAK8[:, 0:1], self._muons[:, 0:1]))
        df['mu0_dRjet1'] = flattenAndPad(match(self._jetsAK8[:, 1:2], self._muons[:, 0:1]))

    def process(self, df):
        dataset = df['dataset']
        print("Processing dataframe from", dataset)

        self._jetsAK8 = nanoObject(df, 'CustomAK8Puppi_', ['pt', 'eta', 'phi', 'mass','electronIdx3SJ','muonIdx3SJ','lsf3','msoftdrop','deepTagHqqqq','deepTagMDHqqqq'])
        print(self._jetsAK8)
        self._electrons = nanoObject(df, 'Electron_', ['pt', 'eta', 'phi', 'mass','sip3d','dxy','dz','mvaFall17V1noIso_WP90','miniPFRelIso_all'])
        self._muons = nanoObject(df, 'Muon_', ['pt', 'eta', 'phi', 'mass','sip3d','dxy','dz','mvaId','miniPFRelIso_all'])
        
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
        self.build_leptonjet_variables(df)

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
        trigger_selection = {}
        for key,trig in trigger.items():
            trigger_selection[key] = False
            for path in trig:
                if path in df:
                    trigger_selection[key] |= df[path]

        # selection
        selection = processor.PackedSelection()

        # trigger selection
        for key,trig in trigger_selection.items():
            selection.add('trigger_%s'%key,trig)

        # match jet to reco lepton
        selection.add('good_jet0ele',(leadingjet['electronIdx3SJ']==0).any())
        selection.add('good_jet1ele',(subleadingjet['electronIdx3SJ']==0).any())
        selection.add('good_jet0mu',(leadingjet['muonIdx3SJ']==0).any())
        selection.add('good_jet1mu',(subleadingjet['muonIdx3SJ']==0).any())

        # object selection
        selection.add('good_ele',good_electron.any())
        selection.add('good_mu',good_muon.any())
        selection.add('good_jet0',good_leadingjet.any())
        selection.add('good_jet1',good_subleadingjet.any())

        regions = {}
        regions['noselection'] = {}
        regions['hadseljet0'] = {'good_jet0','trigger_had'}
        regions['eleseljet0'] = {'good_ele','good_jet0ele','good_jet0','trigger_had'}
        regions['eleseljet1'] = {'good_ele','good_jet1ele','good_jet1','trigger_had',}
        regions['museljet0'] = {'good_mu', 'good_jet0mu', 'good_jet0','trigger_had',} 
        regions['museljet1'] = {'good_mu', 'good_jet1mu', 'good_jet1','trigger_had'}

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
    parser = argparse.ArgumentParser(description='Boosted Hbb processor')
    parser.add_argument('--year', choices=['2016', '2017', '2018'], default='2017', help='Which data taking year to correct MC to.')
    parser.add_argument('--debug', action='store_true', help='Enable debug printouts')
    parser.add_argument('--externalSumW', help='Path to external sum weights file (if provided, will be used in place of self-determined sumw)')
    args = parser.parse_args()

    corrections = load('corrections.coffea')

    if args.externalSumW is not None:
        corrections['sumw_external'] = load(args.externalSumW)

    processor_instance = BoostedHWWProcessorPreSel(corrections=corrections,
                                                   debug=args.debug,
                                                   year=args.year,
                                                   )

    save(processor_instance, 'boostedHWWProcessor_presel.coffea')
