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

def flattenAndPad(var,val=-99):
    return var.pad(1, clip=True).fillna(val).regular().flatten()

class BoostedHWWProcessor(processor.ProcessorABC):
    def __init__(self, corrections, debug=False, year='2018'):
        self._corrections = corrections
        self._debug = debug
        self._year = year

        dataset_axis = hist.Cat("dataset", "Primary dataset")
        #ht_axis = hist.Bin("ht",r"HT",20, 300, 1000)
        metpt_axis = hist.Bin("MET_pt", r"MET $p_T$", 20, 0, 100)

        ele0pt_axis = hist.Bin("ele0_pt", r"Electron $p_T$", 20, 0, 400)
        ele0miso_axis = hist.Bin("ele0_miso", r"Electron mini PF ISO (total)", 20, 0, 1)
        ele0dRjet0_axis = hist.Bin("ele0_dRjet0", r"Electron dR(Leading jet)", 20, 0, 4)
        ele0mjet0_axis = hist.Bin("ele0_mjet0", r"Leading Jet - Electron mass", 20, -99, 120)
        ele0dRjet1_axis = hist.Bin("ele0_dRjet1", r"Electron dR(Sub-Leading jet)", 20, 0, 4)
        ele0mjet1_axis = hist.Bin("ele0_mjet1",r"Sub-Leading Jet - Electron mass",20, -99, 120)

        mu0pt_axis = hist.Bin("mu0_pt", r"Muon $p_T$", 20, 0, 400)
        mu0miso_axis = hist.Bin("mu0_miso", r"Muon mini PF ISO (total)", 20, 0, 1)
        mu0dRjet0_axis = hist.Bin("mu0_dRjet0", r"Muon dR(Leading jet)", 20, 0, 4)
        mu0mjet0_axis = hist.Bin("mu0_mjet0", r"Leading Jet - Muon mass", 20, -99, 120)
        mu0dRjet1_axis = hist.Bin("mu0_dRjet1", r"Muon dR(Sub-Leading jet)", 20, 0, 4)
        mu0mjet1_axis = hist.Bin("mu0_mjet1",r"Sub-Leading Jet - Muon mass",20, -99, 120)

        jet0pt_axis = hist.Bin("jet0_pt", r"Leading Jet $p_T$", 20, 200, 1000)
        jet0mass_axis = hist.Bin("jet0_msd", r"Leading Jet $m_{sd}$", 20, 10, 200)
        jet0lsf3_axis = hist.Bin("jet0_lsf3", r"Leading Jet LSF_3", 20, 0, 1)
        jet0dRLep_axis = hist.Bin("jet0_dRLep", r"Leading Jet - Lep dR", 20, 0, 4)
        jet0met_axis =  hist.Bin("jet0_met", r"Leading Jet + MET", 20, 20, 300)

        jet1pt_axis = hist.Bin("jet1_pt", r"Sub-Leading Jet $p_T$", 20, 200, 1000)
        jet1mass_axis = hist.Bin("jet1_msd", r"Sub-Leading Jet $m_{sd}$", 20, 10, 200)
        jet1lsf3_axis = hist.Bin("jet1_lsf3", r"Sub-Leading Jet LSF_3", 20, 0, 1)
        jet1dRLep_axis = hist.Bin("jet1_dRLep", r"Sub-Leading Jet - Lep dR", 20, 0, 4)
        jet1met_axis =hist.Bin("jet1_met", r"Sub-Leading Jet + MET", 20, 20, 300)

        #hmassjet0_axis = hist.Bin("hmass_jet0", r"Leading Jet Reconstructed H mass", 20, 20, 200)

        hists = processor.dict_accumulator()
        hist.Hist.DEFAULT_DTYPE = 'f'  # save some space by keeping float bin counts instead of double
        hists['sumw'] = processor.defaultdict_accumulator(int)
        for key in ['hadPreselection','hadTrigPreselection',
                    'elePreselection','eleTrigElePreselection','eleTrigHadPreselection',
                    'muPreselection','muTrigMuPreselection','muTrigHadPreselection']:
            hists['jet0_%s'%key] = hist.Hist("Events / GeV",
                                             dataset_axis,
                                             jet0pt_axis,
                                             jet0mass_axis,
                                             jet0lsf3_axis,
                                             )
            hists['jet1_%s'%key] = hist.Hist("Events / GeV",
                                             dataset_axis,
                                             jet1pt_axis,
                                             jet1mass_axis,
                                             jet1lsf3_axis
                                             )
            hists['evt_%s'%key] = hist.Hist("Events / GeV",
                                            dataset_axis,
                                            #ht_axis,
                                            metpt_axis,
                                            )
            '''
            if 'ele' in key:
                hists['ele0_%s'%key] = hist.Hist("Events / GeV",
                                                 dataset_axis,
                                                 ele0pt_axis,
                                                 ele0miso_axis,
                                                 ele0dRjet0_axis,
                                                 ele0mjet0_axis,
                                                 ele0dRjet1_axis,
                                                 ele0mjet1_axis,
                                                 )
            if 'mu' in key:
                hists['mu0_%s'%key] = hist.Hist("Events / GeV",
                                                dataset_axis,
                                                mu0pt_axis,
                                                mu0miso_axis,
                                                mu0dRjet0_axis,
                                                mu0mjet0_axis,
                                                mu0dRjet1_axis,
                                                mu0mjet1_axis,
            )
            '''
        self._accumulator = hists

    @property
    def accumulator(self):
        return self._accumulator
    
    def jetPMET(self, df, jet):
        vjet = uproot_methods.TLorentzVectorArray.from_ptetaphim(df['%s_pt'%jet], df['%s_eta'%jet], df['%s_phi'%jet], df['%s_msd'%jet])
        #vjet = vjet['p4']
        vmet = uproot_methods.TLorentzVectorArray.from_ptetaphim(df['MET_pt'], df['%s_eta'%jet], df['MET_phi'], 0)
        return vmet+vjet

    def build_jet_variables(self, df, ij=0):
        jet = self._jetsAK8[:, ij:ij+1]
        df['jet%i_pt'%ij] = flattenAndPad(jet['p4'].pt)
        df['jet%i_eta'%ij] = flattenAndPad(jet['p4'].eta)
        df['jet%i_phi'%ij] = flattenAndPad(jet['p4'].phi)
        df['jet%i_mass'%ij] = flattenAndPad(jet['p4'].mass)
        df['jet%i_msd'%ij] = flattenAndPad(jet['msoftdrop'])
        df['jet%i_met'%ij] = self.jetPMET(df, 'jet%i'%ij)
        df['jet%i_lsf3'%ij] = flattenAndPad(jet['lsf3'])
        df['jet%i_dRLep'%ij] = flattenAndPad(jet['dRLep'])

    def build_leadingelectron_variables(self, df):
        leadingele = self._electrons[:, 0:1]
        df['ele0_pt'] = flattenAndPad(leadingele['p4'].pt)
        df['ele0_miso'] = flattenAndPad(leadingele['miniPFRelIso_all'])

    def build_leadingmuon_variables(self, df):
        leadingmu = self._muons[:, 0:1]
        df['mu0_pt'] = flattenAndPad(leadingmu['p4'].pt)
        df['mu0_miso'] = flattenAndPad(leadingmu['miniPFRelIso_all'])

    def build_leptonjet_variables(self, df, jet, jtag, lep, ltag):
        obj = jet['p4'].cross(lep['p4'], nested=False)
        df['%s_dR%s'%(ltag,jtag)] = flattenAndPad(obj.i0.delta_r(obj.i1))
        #vjet = uproot_methods.TLorentzVectorArray.from_ptetaphim(df['%s_pt'%jtag], df['%s_eta'%jtag], df['%s_phi'%jtag], df['%s_msd'%jtag])
        #vlep = uproot_methods.TLorentzVectorArray.from_ptetaphim(df['%s_pt'%ltag], df['%s_eta'%ltag], df['%s_phi'%ltag], df['%s_msd'%ltag])
        df['%s_m%s'%(ltag,jtag)] = flattenAndPad((obj.i0 - obj.i1).mass)
        
    def process(self, df):
        dataset = df['dataset']
        isRealData = dataset in ["JetHT", "SingleMuon", "SingleElectron"]
        if self._debug:
            print("Processing dataframe from", dataset)

        self._jetsAK8 = nanoObject(df, 'CustomAK8Puppi_')
        self._electrons = nanoObject(df, 'Electron_')
        self._muons = nanoObject(df, 'Muon_')
        
        self.build_jet_variables(df, 0)
        self.build_jet_variables(df, 1)
        self.build_leadingelectron_variables(df)
        self.build_leadingmuon_variables(df)
        for ij in range(0, 2):
            self.build_leptonjet_variables(df, self._jetsAK8[:, ij:ij+1], 'jet%i'%ij, self._electrons[:, 0:1], 'ele0')
            self.build_leptonjet_variables(df, self._jetsAK8[:, ij:ij+1], 'jet%i'%ij, self._muons[:, 0:1], 'mu0')

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
        for key,trig in trigger_selection.items():
            selection.add('trigger_%s'%key,trig)

        selection.add('good_1jet',(df['jet0_pt']>300) & (df['jet0_eta']<2.4))
        selection.add('good_1ele',(df['ele0_pt']>115))
        selection.add('good_1mu',(df['mu0_pt']>50))


        regions = {}
        regions['hadPreselection'] = {'good_1jet'}
        regions['hadTrigPreselection'] = {'good_1jet','trigger_had'}
        regions['elePreselection'] = {'good_1jet','good_1ele'}
        regions['eleTrigElePreselection'] = {'good_1jet','good_1ele','trigger_ele'}
        regions['eleTrigHadPreselection'] = {'good_1jet','good_1ele','trigger_had'}
        regions['muPreselection'] = {'good_1jet','good_1mu'}
        regions['muTrigMuPreselection'] = {'good_1jet','good_1mu','trigger_mu'}
        regions['muTrigHadPreselection'] = {'good_1jet','good_1mu','trigger_had'}

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

        if not isRealData:
            if 'skim_sumw' in df:
                # hacky way to only accumulate file-level information once
                if df['skim_sumw'] is not None:
                    hout['sumw'][dataset] += df['skim_sumw']
            else:
                raise ValueError("No skim sumw")

        return hout

    def postprocess(self, accumulator):
        # set everything to 1/fb scale
        lumi = 1000  # [1/pb]
        

        scale = {}
        print(accumulator['sumw'])
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

    save(processor_instance, 'boostedHWWProcessor.coffea')
