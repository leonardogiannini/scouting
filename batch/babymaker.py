import os
import sys

import ROOT as r
from tqdm import tqdm

import array
import glob
import math
import time
import argparse
import os
import pickle
import gzip

fast = False
if fast:
    print("NOTE: fast option is True, so we will skip some crucial things")

def xrootdify(fname):
    if "/hadoop/cms/store/user/namin/" in fname:
        fname = "root://redirector.t2.ucsd.edu/" + fname.replace("/hadoop/cms","")
    if fname.startswith("/store"):
        fname = "root://cmsxrootd.fnal.gov/" + fname
    return fname

def argsort(seq, key=lambda x:x):
    if not seq:
        return [], []
    # return two lists: indices that sort `seq`, and the sorted `seq`, according to `key` function
    indices, sorted_seq = zip(*sorted(enumerate(seq), key=lambda y:key(y[1])))
    return indices, sorted_seq

def sortwitharg(seq, arg):
    # reorder `seq` according to t
    return [seq[i] for i in arg]

class Looper(object):

    def __init__(self,fnames=[], output="output.root", nevents=-1, expected=-1, skim1cm=False, allevents=False, treename="Events"):
        if any("*" in x for x in fnames):
            fnames = sum(map(glob.glob,fnames),[])
        self.fnames = map(xrootdify,sum(map(lambda x:x.split(","),fnames),[]))
        # print(self.fnames)
        self.nevents = nevents
        self.do_skimreco = not allevents
        self.do_skim1cm = skim1cm
        self.do_jets = True
        self.do_tracks = False
        self.do_trigger = True
        self.is_mc = False
        self.has_gen_info = False
        self.expected = expected
        self.branches = {}
        self.treename = treename
        self.fname_out = output
        self.ch = None
        self.outtree = None
        self.outfile = None

        # beamspot stuff - index is [year][is_mc]
        self.bs_data = { 
                2017: {False: {}, True: {}},
                2018: {False: {}, True: {}},
                }

        self.loaded_pixel_code = False

        self.init_tree()
        self.init_branches()


    def init_tree(self):

        self.ch = r.TChain(self.treename)

        # alias
        ch = self.ch

        for fname in self.fnames:
            ch.Add(fname)

        branchnames = [b.GetName() for b in ch.GetListOfBranches()]
        self.has_gen_info = any("genParticles" in name for name in branchnames)
        self.is_mc = self.has_gen_info
        self.has_trigger_info = any("triggerMaker" in name for name in branchnames)
        self.has_hit_info = any("hitMaker" in name for name in branchnames)

        if not self.has_trigger_info:
            print("[!] Didn't find trigger branches. Saving dummy trigger information.")
            self.do_trigger = False

        ch.SetBranchStatus("*",0)
        ch.SetBranchStatus("*hltScoutingMuonPackerCalo*",1)
        ch.SetBranchStatus("*hltScoutingCaloPacker*",1)
        ch.SetBranchStatus("*hltScoutingPrimaryVertexPacker*",1)
        ch.SetBranchStatus("*EventAuxiliary*",1)
        if self.do_tracks:
            ch.SetBranchStatus("*hltScoutingTrackPacker*",1)
        if self.has_trigger_info:
            ch.SetBranchStatus("*triggerMaker*",1)
        if self.has_hit_info:
            ch.SetBranchStatus("*hitMaker*",1)
        if self.has_gen_info:
            ch.SetBranchStatus("*genParticles*",1)

        self.outfile = r.TFile(self.fname_out, "recreate")
        self.outtree = r.TTree(self.treename,"")

        cachesize = 30000000
        ch.SetCacheSize(cachesize)
        ch.SetCacheLearnEntries(500)

    def make_branch(self, name, tstr="vi"):
        extra = []
        if tstr == "vvi": obj = r.vector("vector<int>")()
        if tstr == "vvf": obj = r.vector("vector<float>")()
        if tstr == "vvb": obj = r.vector("vector<bool>")()
        if tstr == "vi": obj = r.vector("int")()
        if tstr == "vf": obj = r.vector("float")()
        if tstr == "vb": obj = r.vector("bool")()
        if tstr == "f":
            obj = array.array("f",[999])
            extra.append("{}/f".format(name))
        if tstr == "b":
            obj = array.array("b",[0])
            extra.append("{}/O".format(name))
        if tstr == "i":
            obj = array.array("I",[999])
            extra.append("{}/I".format(name))
        if tstr == "l":
            obj = array.array("L",[999])
            extra.append("{}/L".format(name))
        self.branches[name] = obj
        self.outtree.Branch(name,obj,*extra)

    def clear_branches(self):
        for v in self.branches.values():
            if hasattr(v,"clear"):
                v.clear()

    def load_bs_data(self, year):
        if self.is_mc:
            # Events->Scan("recoBeamSpot_offlineBeamSpot__RECO.obj.x0()") in miniaod (and y0). 
            # From 2017 MC with global tag of 94X_mc2017_realistic_v14
            self.bs_data[2017][self.is_mc] = { (0,0): [-0.024793, 0.0692861, 0.789895] }
            # From 2018 MC with global tag of 102X_upgrade2018_realistic_v11
            self.bs_data[2018][self.is_mc] = { (0,0): [0.0107796, 0.041893, 0.0248755] }
        else:
            print("Loading beamspot data for year={}".format(year))
            t0 = time.time()
            data = []
            with gzip.open("data/beamspots_{}.pkl.gz".format(year),"r") as fh:
                data = pickle.load(fh)
            for run,lumi,x,y,z in data:
                self.bs_data[year][self.is_mc][(run,lumi)] = [x,y,z]
            t1 = time.time()
            print("Finished loading {} rows in {:.1f} seconds".format(len(data),t1-t0))

    def get_bs(self, run, lumi, year=2018):
        if fast: return 0., 0., 0.
        if self.is_mc: run,lumi = 0,0
        if not self.bs_data[year][self.is_mc]:
            self.load_bs_data(year=year)
        data = self.bs_data[year][self.is_mc]
        xyz = data.get((run,lumi),None)
        if xyz is None:
            xyz = data.get((0,0),[0,0,0])
            print("WARNING: Couldn't find (run={},lumi={},is_mc={},year={}) in beamspot lookup data. Falling back to the total mean: {}".format(run,lumi,self.is_mc,year,xyz))
        return xyz

    def in_pixel_rectangles_rough(self,x,y,z):
        if fast: return False
        rho = math.hypot(x,y)
        if (rho < 2.0): return False
        boxes = [
            [[2.5,3.5],[-23,23]],
            [[6.3,7.4],[-27,27]],
            [[10.3,11.5],[-27,27]],
            [[15.5,16.5],[-27,27]],
            [[9,16.5],[29,33]],
            [[9,16.5],[-33,-29]],
            [[9,16.5],[36,39.5]],
            [[9,16.5],[-39.5,-36]],
        ]
        for (rholow,rhohigh),(zlow,zhigh) in boxes:
            if (rholow < rho < rhohigh) and (zlow < z < zhigh):
                return True
        return False

    def load_pixel_code(self):
        if not self.loaded_pixel_code:
            print("Loading pixel utilities and lookup tables")
            t0 = time.time()
            r.gROOT.ProcessLine(".L data/calculate_pixel.cc")
            self.loaded_pixel_code = True
            t1 = time.time()
            print("Finished loading in {:.1f} seconds".format(t1-t0))

    def in_pixel_rectangles(self,px,py,pz):
        if fast: return False
        self.load_pixel_code()
        rho = math.hypot(px,py)
        if (rho < 2.0): return False
        return r.is_point_in_any_module(px, py, pz)

    def calculate_module_crosses(self, vx, vy, vz, px, py, pz):
        if fast: return False
        self.load_pixel_code()
        return r.calculate_module_crosses(vx, vy, vz, px, py, pz)


    def init_branches(self):

        make_branch = self.make_branch

        make_branch("run", "l")
        make_branch("luminosityBlock", "l")
        make_branch("event", "l")

        make_branch("pass_skim", "b")
        make_branch("pass_l1", "b")
        make_branch("pass_fiducialgen", "b")
        make_branch("pass_fiducialgen_norho", "b")

        make_branch("MET_pt", "f")
        make_branch("MET_phi", "f")
        make_branch("MET_pt_muonCorr", "f")
        make_branch("MET_phi_muonCorr", "f")
        make_branch("rho", "f")
        # make_branch("LeadingPair_mass", "f")
        # make_branch("LeadingPair_sameVtx", "b")
        # make_branch("LeadingPair_isOS", "b")

        make_branch("nDV", "i")
        make_branch("nDV_good", "i")
        make_branch("DV_x","vf")
        make_branch("DV_y","vf")
        make_branch("DV_z","vf")
        make_branch("DV_xError","vf")
        make_branch("DV_yError","vf")
        make_branch("DV_zError","vf")
        make_branch("DV_tracksSize","vi")
        make_branch("DV_chi2","vf")
        make_branch("DV_ndof","vi")
        make_branch("DV_isValidVtx","vb")
        make_branch("DV_good","vb")
        make_branch("DV_rho", "vf")
        make_branch("DV_rhoCorr", "vf")
        make_branch("DV_inPixelRectangles", "vb")
        # make_branch("DV_inPixelRectanglesRough", "vb")

        if self.do_jets:
            make_branch("nJet", "i")
            make_branch("Jet_pt", "vf")
            make_branch("Jet_eta", "vf")
            make_branch("Jet_phi", "vf")
            make_branch("Jet_m", "vf")
            make_branch("Jet_jetArea", "vf")
            make_branch("Jet_maxEInEmTowers", "vf")
            make_branch("Jet_maxEInHadTowers", "vf")
            make_branch("Jet_hadEnergyInHB", "vf")
            make_branch("Jet_hadEnergyInHE", "vf")
            make_branch("Jet_hadEnergyInHF", "vf")
            make_branch("Jet_emEnergyInEB", "vf")
            make_branch("Jet_emEnergyInEE", "vf")
            make_branch("Jet_emEnergyInHF", "vf")
            make_branch("Jet_towersArea", "vf")
            make_branch("Jet_mvaDiscriminator", "vf")
            make_branch("Jet_btagDiscriminator", "vf")

        make_branch("nPV", "i")
        make_branch("PV_x", "vf")
        make_branch("PV_y", "vf")
        make_branch("PV_z", "vf")
        make_branch("PV_tracksSize", "vi")
        make_branch("PV_chi2", "vf")
        make_branch("PV_ndof", "vi")
        make_branch("PV_isValidVtx", "vb")

        make_branch("nPVM", "i")
        make_branch("PVM_x", "vf")
        make_branch("PVM_y", "vf")
        make_branch("PVM_z", "vf")
        make_branch("PVM_tracksSize", "vi")
        make_branch("PVM_chi2", "vf")
        make_branch("PVM_ndof", "vi")
        make_branch("PVM_isValidVtx", "vb")

        if self.do_tracks:
            make_branch("Track_nValidPixelHits", "vi")
            make_branch("Track_nTrackerLayersWithMeasurement", "vi")
            make_branch("Track_nValidStripHits", "vi")
            make_branch("Track_pt", "vf")
            make_branch("Track_phi", "vf")
            make_branch("Track_eta", "vf")
            make_branch("Track_chi2", "vf")
            make_branch("Track_ndof", "vf")
            make_branch("Track_charge", "vi")
            make_branch("Track_dxy", "vf")
            make_branch("Track_dz", "vf")

        make_branch("nMuon", "i")
        make_branch("nMuon_good", "i")
        make_branch("Muon_pt", "vf")
        make_branch("Muon_eta", "vf")
        make_branch("Muon_phi", "vf")
        make_branch("Muon_m", "vf")
        make_branch("Muon_trackIso", "vf")
        make_branch("Muon_chi2", "vf")
        make_branch("Muon_ndof", "vf")
        make_branch("Muon_charge", "vi")
        make_branch("Muon_dxy", "vf")
        make_branch("Muon_dz", "vf")
        make_branch("Muon_nValidMuonHits", "vi")
        make_branch("Muon_nValidPixelHits", "vi")
        make_branch("Muon_nMatchedStations", "vi")
        make_branch("Muon_nTrackerLayersWithMeasurement", "vi")
        make_branch("Muon_nValidStripHits", "vi")
        make_branch("Muon_trk_qoverp", "vf")
        make_branch("Muon_trk_lambda", "vf")
        make_branch("Muon_trk_pt", "vf")
        make_branch("Muon_trk_phi", "vf")
        make_branch("Muon_trk_eta", "vf")
        make_branch("Muon_dxyError", "vf")
        make_branch("Muon_dzError", "vf")
        make_branch("Muon_trk_qoverpError", "vf")
        make_branch("Muon_trk_lambdaError", "vf")
        make_branch("Muon_trk_phiError", "vf")
        make_branch("Muon_trk_dsz", "vf")
        make_branch("Muon_trk_dszError", "vf")
        make_branch("Muon_vtxNum","vi")
        make_branch("Muon_vtxIdx1","vi")
        make_branch("Muon_vx", "vf")
        make_branch("Muon_vy", "vf")
        make_branch("Muon_vz", "vf")
        make_branch("Muon_dxyCorr", "vf")
        make_branch("Muon_nExpectedPixelHits", "vi")
        make_branch("Muon_nExpectedPixelHitsCrappy", "vi")
        make_branch("Muon_jetIdx1", "vi")
        make_branch("Muon_jetIdx2", "vi")
        make_branch("Muon_drjet", "vf")
        make_branch("Muon_good", "vb")

        make_branch("Muon_hit_x", "vvf")
        make_branch("Muon_hit_y", "vvf")
        make_branch("Muon_hit_z", "vvf")
        make_branch("Muon_hit_active", "vvb")
        make_branch("Muon_hit_barrel", "vvb")
        make_branch("Muon_hit_ndet", "vvi")
        make_branch("Muon_hit_layer", "vvi")

        make_branch("nGenPart", "i")
        make_branch("GenPart_pt", "vf")
        make_branch("GenPart_eta", "vf")
        make_branch("GenPart_phi", "vf")
        make_branch("GenPart_m", "vf")
        make_branch("GenPart_vx", "vf")
        make_branch("GenPart_vy", "vf")
        make_branch("GenPart_vz", "vf")
        make_branch("GenPart_status", "vi")
        make_branch("GenPart_pdgId", "vi")
        make_branch("GenPart_motherId", "vi")

        make_branch("nGenMuon", "i")
        make_branch("GenMuon_pt", "vf")
        make_branch("GenMuon_eta", "vf")
        make_branch("GenMuon_phi", "vf")
        make_branch("GenMuon_m", "vf")
        make_branch("GenMuon_vx", "vf")
        make_branch("GenMuon_vy", "vf")
        make_branch("GenMuon_vz", "vf")
        make_branch("GenMuon_status", "vi")
        make_branch("GenMuon_pdgId", "vi")
        make_branch("GenMuon_motherId", "vi")

        make_branch("BS_x", "f")
        make_branch("BS_y", "f")
        make_branch("BS_z", "f")


        self.outtree.SetBasketSize("*",int(512*1024))

    def run(self):

        made_trigger_branches = False
        ch = self.ch
        branches = self.branches
        make_branch = self.make_branch

        ievt = 0
        nevents_in = ch.GetEntries()
        print(">>> Started slimming/skimming tree with {} events".format(nevents_in))
        t0 = time.time()
        tprev = time.time()
        nprev = 0
        for evt in ch:
            # if (ievt-1) % 1000 == 0:
            #     ch.GetTree().PrintCacheStats()
            if (ievt-1) % 1000 == 0:
                nnow = ievt
                tnow = time.time()
                print(">>> [currevt={}] Last {} events in {:.2f} seconds @ {:.1f}Hz".format(nnow,nnow-nprev,(tnow-tprev),(nnow-nprev)/(tnow-tprev)))
                tprev = tnow
                nprev = nnow
            if (self.nevents > 0) and (ievt > self.nevents): break

            ievt += 1

            dvs = evt.ScoutingVertexs_hltScoutingMuonPackerCalo_displacedVtx_HLT.product()
            if not dvs: dvs = []

            muons = evt.ScoutingMuons_hltScoutingMuonPackerCalo__HLT.product()

            pass_skim = (len(dvs) >= 1) and (len(muons) >= 2)
            if self.do_skimreco and not pass_skim: continue
            if self.do_skim1cm and len(dvs) >= 1:
                if not any(math.hypot(dv.x(),dv.y())>1. for dv in dvs): continue
            branches["pass_skim"][0] = pass_skim
            branches["pass_l1"][0] = False

            pvs = evt.ScoutingVertexs_hltScoutingPrimaryVertexPacker_primaryVtx_HLT.product()
            pvms = evt.ScoutingVertexs_hltScoutingPrimaryVertexPackerCaloMuon_primaryVtx_HLT.product()

            # sort muons in descending pT (turns out a nontrivial amount are not sorted already, like 5%-10% I think)
            # muons = sorted(muons, key=lambda x:-x.pt())
            muon_sort_indices, muons = argsort(muons, key=lambda x:-x.pt())

            if self.do_trigger:
                hltresults = map(bool,evt.bools_triggerMaker_hltresult_SLIM.product())
                hltnames = list(evt.Strings_triggerMaker_hltname_SLIM.product())
                l1results = map(bool,evt.bools_triggerMaker_l1result_SLIM.product())
                l1names = list(evt.Strings_triggerMaker_l1name_SLIM.product())
                l1prescales = list(evt.ints_triggerMaker_l1prescale_SLIM.product())
                if not made_trigger_branches:
                    for name in hltnames:
                        make_branch("HLT_{}".format(name), "b")
                    for name in l1names:
                        make_branch("{}".format(name), "b")
                        make_branch("{}_prescale".format(name), "i")
                    made_trigger_branches = True

            self.clear_branches()

            if self.do_trigger:
                for bit,name in zip(hltresults,hltnames):
                    branches["HLT_{}".format(name)][0] = bit
                for bit,name,prescale in zip(l1results,l1names,l1prescales):
                    branches["{}".format(name)][0] = bit
                    branches["{}_prescale".format(name)][0] = max(prescale,0)
                d = dict(zip(l1names,l1results))
                branches["pass_l1"][0] = d["L1_DoubleMu4p5_SQ_OS_dR_Max1p2"] or d["L1_DoubleMu0er1p4_SQ_OS_dR_Max1p4"] or d["L1_DoubleMu_15_7"]
            else:
                branches["pass_l1"][0] = True


            run = int(evt.EventAuxiliary.run())
            lumi = int(evt.EventAuxiliary.luminosityBlock())
            eventnum = int(evt.EventAuxiliary.event())
            branches["run"][0] = run
            branches["luminosityBlock"][0] = lumi
            branches["event"][0] = eventnum

            metpt = evt.double_hltScoutingCaloPacker_caloMetPt_HLT.product()[0]
            metphi = evt.double_hltScoutingCaloPacker_caloMetPhi_HLT.product()[0]
            branches["MET_pt"][0] = metpt
            branches["MET_phi"][0] = metphi
            branches["rho"][0] = evt.double_hltScoutingCaloPacker_rho_HLT.product()[0]

            bsx,bsy,bsz = self.get_bs(run=run,lumi=lumi,year=2018)
            branches["BS_x"][0] = bsx
            branches["BS_y"][0] = bsy
            branches["BS_z"][0] = bsz

            ngooddv = 0
            for dv in dvs:
                vx = dv.x()
                vy = dv.y()
                vz = dv.z()
                rho = (vx**2 + vy**2)**0.5
                rhoCorr = ((vx-bsx)**2 + (vy-bsy)**2)**0.5
                branches["DV_x"].push_back(vx)
                branches["DV_y"].push_back(vy)
                branches["DV_z"].push_back(vz)
                branches["DV_xError"].push_back(dv.xError())
                branches["DV_yError"].push_back(dv.yError())
                branches["DV_zError"].push_back(dv.zError())
                branches["DV_tracksSize"].push_back(dv.tracksSize())
                branches["DV_chi2"].push_back(dv.chi2())
                branches["DV_ndof"].push_back(dv.ndof())
                branches["DV_isValidVtx"].push_back(dv.isValidVtx())
                branches["DV_rho"].push_back(rho)
                branches["DV_rhoCorr"].push_back(rhoCorr)
                branches["DV_inPixelRectangles"].push_back(self.in_pixel_rectangles(vx,vy,vz))
                # branches["DV_inPixelRectanglesRough"].push_back(self.in_pixel_rectangles_rough(vx,vy,vz))
                gooddv = ((dv.xError() < 0.05) and (dv.yError() < 0.05) and (dv.zError() < 0.10))
                branches["DV_good"].push_back(gooddv)
                ngooddv += gooddv
            branches["nDV"][0] = len(dvs)
            branches["nDV_good"][0] = ngooddv

            # I think these only get calculated/written out when mass>10gev (event also fell into PFscouting stream)
            # so most of the pv collections are size 0.
            # However, PVM below is the set of PVs with an associated muon. That always has at least one entry.
            # We'll store the full PVM stuff, but just the multiplicity of PV for now (PU reweighting if mass>10?). Cuts down on filesize.
            # for pv in pvs:
            #     branches["PV_x"].push_back(pv.x())
            #     branches["PV_y"].push_back(pv.y())
            #     branches["PV_z"].push_back(pv.z())
            #     branches["PV_tracksSize"].push_back(pv.tracksSize())
            #     branches["PV_chi2"].push_back(pv.chi2())
            #     branches["PV_ndof"].push_back(pv.ndof())
            #     branches["PV_isValidVtx"].push_back(pv.isValidVtx())
            branches["nPV"][0] = len(pvs)

            for pvm in pvms:
                branches["PVM_x"].push_back(pvm.x())
                branches["PVM_y"].push_back(pvm.y())
                branches["PVM_z"].push_back(pvm.z())
                branches["PVM_tracksSize"].push_back(pvm.tracksSize())
                branches["PVM_chi2"].push_back(pvm.chi2())
                branches["PVM_ndof"].push_back(pvm.ndof())
                branches["PVM_isValidVtx"].push_back(pvm.isValidVtx())
            branches["nPVM"][0] = len(pvms)

            if self.do_jets:
                jets = evt.ScoutingCaloJets_hltScoutingCaloPacker__HLT.product()
                jets = sorted(jets, key=lambda x:-x.pt())
                jet_etaphis = []
                for jet in jets:
                    branches["Jet_pt"].push_back(jet.pt())
                    branches["Jet_eta"].push_back(jet.eta())
                    branches["Jet_phi"].push_back(jet.phi())
                    branches["Jet_m"].push_back(jet.m())
                    branches["Jet_jetArea"].push_back(jet.jetArea())
                    branches["Jet_maxEInEmTowers"].push_back(jet.maxEInEmTowers())
                    branches["Jet_maxEInHadTowers"].push_back(jet.maxEInHadTowers())
                    branches["Jet_hadEnergyInHB"].push_back(jet.hadEnergyInHB())
                    branches["Jet_hadEnergyInHE"].push_back(jet.hadEnergyInHE())
                    branches["Jet_hadEnergyInHF"].push_back(jet.hadEnergyInHF())
                    branches["Jet_emEnergyInEB"].push_back(jet.emEnergyInEB())
                    branches["Jet_emEnergyInEE"].push_back(jet.emEnergyInEE())
                    branches["Jet_emEnergyInHF"].push_back(jet.emEnergyInHF())
                    branches["Jet_towersArea"].push_back(jet.towersArea())
                    branches["Jet_mvaDiscriminator"].push_back(jet.mvaDiscriminator())
                    branches["Jet_btagDiscriminator"].push_back(jet.btagDiscriminator())
                    jet_etaphis.append((jet.eta(),jet.phi()))
                branches["nJet"][0] = len(jets)

            if self.has_gen_info:
                try:
                    genparts = list(evt.recoGenParticles_genParticles__HLT.product()) # rawsim
                except:
                    genparts = []
            else:
                genparts = []
            nGenPart = 0
            nGenMuon = 0
            nFiducialMuon = 0
            nFiducialMuon_norho = 0
            for genpart in genparts:
                pdgid = genpart.pdgId()
                if abs(pdgid) not in [13,23,25]: continue
                motheridx = genpart.motherRef().index()
                mother = genparts[motheridx]
                motherid = mother.pdgId()
                branches["GenPart_pt"].push_back(genpart.pt())
                branches["GenPart_eta"].push_back(genpart.eta())
                branches["GenPart_phi"].push_back(genpart.phi())
                branches["GenPart_m"].push_back(genpart.mass())
                branches["GenPart_vx"].push_back(genpart.vx())
                branches["GenPart_vy"].push_back(genpart.vy())
                branches["GenPart_vz"].push_back(genpart.vz())
                branches["GenPart_status"].push_back(genpart.status())
                branches["GenPart_pdgId"].push_back(pdgid)
                branches["GenPart_motherId"].push_back(motherid)
                nGenPart += 1
                # For the useful muons, ALSO store them in separate GenMuon branches to avoid reading a lot of extra junk
                if (motherid == 23) and (abs(pdgid)==13): 
                    branches["GenMuon_pt"].push_back(genpart.pt())
                    branches["GenMuon_eta"].push_back(genpart.eta())
                    branches["GenMuon_phi"].push_back(genpart.phi())
                    branches["GenMuon_m"].push_back(genpart.mass())
                    branches["GenMuon_vx"].push_back(genpart.vx())
                    branches["GenMuon_vy"].push_back(genpart.vy())
                    branches["GenMuon_vz"].push_back(genpart.vz())
                    branches["GenMuon_status"].push_back(genpart.status())
                    branches["GenMuon_pdgId"].push_back(pdgid)
                    branches["GenMuon_motherId"].push_back(motherid)
                    if (genpart.pt() > 4.) and (abs(genpart.eta()) < 2.4):
                        nFiducialMuon_norho += 1
                        if (math.hypot(genpart.vx(),genpart.vy())<11.):
                            nFiducialMuon += 1
                    nGenMuon += 1
            branches["nGenPart"][0] = nGenPart
            branches["nGenMuon"][0] = nGenMuon
            branches["pass_fiducialgen"][0] = (nFiducialMuon >= 2) or (not self.is_mc)
            branches["pass_fiducialgen_norho"][0] = (nFiducialMuon_norho >= 2) or (not self.is_mc)

            if self.do_tracks:
                tracks = evt.ScoutingTracks_hltScoutingTrackPacker__HLT.product()
                for track in tracks:
                    branches["Track_pt"].push_back(track.tk_pt())
                    branches["Track_eta"].push_back(track.tk_eta())
                    branches["Track_phi"].push_back(track.tk_phi())
                    branches["Track_chi2"].push_back(track.tk_chi2())
                    branches["Track_ndof"].push_back(track.tk_ndof())
                    branches["Track_charge"].push_back(track.tk_charge())
                    branches["Track_dxy"].push_back(track.tk_dxy())
                    branches["Track_dz"].push_back(track.tk_dz())
                    branches["Track_nValidPixelHits"].push_back(track.tk_nValidPixelHits())
                    branches["Track_nTrackerLayersWithMeasurement"].push_back(track.tk_nTrackerLayersWithMeasurement())
                    branches["Track_nValidStripHits"].push_back(track.tk_nValidStripHits())

            metx_muoncorr = metpt*math.cos(metphi)
            mety_muoncorr = metpt*math.sin(metphi)
            ngoodmuon = 0

            if self.has_hit_info:
                # NOTE, need to sort these to maintain same order as muons since we have sorted them by pT earlier
                # FIXME if we somehow embedded these into the muons, that would be nicer. or use classes properly in this script...
                muon_hit_x = sortwitharg(evt.floatss_hitMaker_x_SLIM.product(), muon_sort_indices)
                muon_hit_y = sortwitharg(evt.floatss_hitMaker_y_SLIM.product(), muon_sort_indices)
                muon_hit_z = sortwitharg(evt.floatss_hitMaker_z_SLIM.product(), muon_sort_indices)
                muon_hit_layer = sortwitharg(evt.intss_hitMaker_layernum_SLIM.product(), muon_sort_indices)
                muon_hit_ndet = sortwitharg(evt.intss_hitMaker_ndet_SLIM.product(), muon_sort_indices)
                muon_hit_barrel = sortwitharg(evt.boolss_hitMaker_isbarrel_SLIM.product(), muon_sort_indices)
                muon_hit_active = sortwitharg(evt.boolss_hitMaker_isactive_SLIM.product(), muon_sort_indices)
                muon_hit_expectedhits = sortwitharg(evt.ints_hitMaker_nexpectedhits_SLIM.product(), muon_sort_indices)
                for i in range(len(muon_hit_expectedhits)):
                    branches["Muon_hit_x"].push_back(muon_hit_x[i])
                    branches["Muon_hit_y"].push_back(muon_hit_y[i])
                    branches["Muon_hit_z"].push_back(muon_hit_z[i])
                    branches["Muon_hit_active"].push_back(muon_hit_active[i])
                    branches["Muon_hit_barrel"].push_back(muon_hit_barrel[i])
                    branches["Muon_hit_ndet"].push_back(muon_hit_ndet[i])
                    branches["Muon_hit_layer"].push_back(muon_hit_layer[i])
                    branches["Muon_nExpectedPixelHits"].push_back(muon_hit_expectedhits[i])

            for imuon,muon in enumerate(muons):
                pt = muon.pt()
                eta = muon.eta()
                phi = muon.phi()
                branches["Muon_pt"].push_back(pt)
                branches["Muon_eta"].push_back(eta)
                branches["Muon_phi"].push_back(phi)
                branches["Muon_m"].push_back(0.10566) # hardcode since otherwise we get 0.
                branches["Muon_trackIso"].push_back(muon.trackIso())
                branches["Muon_chi2"].push_back(muon.chi2())
                branches["Muon_ndof"].push_back(muon.ndof())
                branches["Muon_charge"].push_back(muon.charge())
                branches["Muon_dxy"].push_back(muon.dxy())
                branches["Muon_dz"].push_back(muon.dz())
                branches["Muon_nValidMuonHits"].push_back(muon.nValidMuonHits())
                branches["Muon_nValidPixelHits"].push_back(muon.nValidPixelHits())
                branches["Muon_nMatchedStations"].push_back(muon.nMatchedStations())
                branches["Muon_nTrackerLayersWithMeasurement"].push_back(muon.nTrackerLayersWithMeasurement())
                branches["Muon_nValidStripHits"].push_back(muon.nValidStripHits())
                branches["Muon_trk_qoverp"].push_back(muon.trk_qoverp())
                branches["Muon_trk_lambda"].push_back(muon.trk_lambda())
                branches["Muon_trk_pt"].push_back(muon.trk_pt())
                branches["Muon_trk_phi"].push_back(muon.trk_phi())
                branches["Muon_trk_eta"].push_back(muon.trk_eta())
                branches["Muon_dxyError"].push_back(muon.dxyError())
                branches["Muon_dzError"].push_back(muon.dzError())
                branches["Muon_trk_qoverpError"].push_back(muon.trk_qoverpError())
                branches["Muon_trk_lambdaError"].push_back(muon.trk_lambdaError())
                branches["Muon_trk_phiError"].push_back(muon.trk_phiError())
                branches["Muon_trk_dsz"].push_back(muon.trk_dsz())
                branches["Muon_trk_dszError"].push_back(muon.trk_dszError())

                jetIdx1 = -1
                jetIdx2 = -1
                drjet = -1
                if self.do_jets:
                    # find index of jet that is closest to this muon. `sorted_etaphis`: [(index, (eta,phi)), ...]
                    sorted_etaphis = sorted(enumerate(jet_etaphis), key=lambda x: math.hypot(eta-x[1][0], phi-x[1][1]))
                    if len(sorted_etaphis) > 0: jetIdx1 = sorted_etaphis[0][0]
                    if len(sorted_etaphis) > 1: jetIdx2 = sorted_etaphis[1][0]
                    if jetIdx1 >= 0:
                        jeteta, jetphi = sorted_etaphis[0][1]
                        drjet = math.hypot(eta-jeteta,phi-jetphi)
                branches["Muon_drjet"].push_back(drjet)
                branches["Muon_jetIdx1"].push_back(jetIdx1)
                branches["Muon_jetIdx2"].push_back(jetIdx2)

                metx_muoncorr -= pt*math.cos(phi)
                mety_muoncorr -= pt*math.sin(phi)

                indices = muon.vtxIndx()
                num = len(indices)
                # branches["Muon_vtxIndx"].push_back(indices)
                branches["Muon_vtxNum"].push_back(num)
                if num > 0: branches["Muon_vtxIdx1"].push_back(indices[0])
                else: branches["Muon_vtxIdx1"].push_back(-1)

                dxy = muon.dxy()

                if len(dvs) > 0:
                    # get index into `dvs` for this muon, falling back to first one in collection
                    idx = 0
                    if num > 0 and (0 <= indices[0] < len(dvs)):
                        idx = indices[0]
                    dv = dvs[idx]
                    vx = dv.x()
                    vy = dv.y()
                    vz = dv.z()
                    phi = muon.phi()
                    # # https://github.com/cms-sw/cmssw/blob/master/DataFormats/TrackReco/interface/TrackBase.h#L24
                    dxyCorr = -(vx-bsx)*math.sin(phi) + (vy-bsy)*math.cos(phi)
                else:
                    # fill muon vertex branches with dummy values since there are no DVs to even look at
                    vx, vy, vz = 0, 0, 0
                    dxyCorr = dxy
                branches["Muon_vx"].push_back(vx)
                branches["Muon_vy"].push_back(vy)
                branches["Muon_vz"].push_back(vz)
                branches["Muon_dxyCorr"].push_back(dxyCorr)
                vmu = r.TLorentzVector()
                vmu.SetPtEtaPhiM(muon.pt(), muon.eta(), muon.phi(), 0.10566)
                branches["Muon_nExpectedPixelHitsCrappy"].push_back(self.calculate_module_crosses(vx,vy,vz,vmu.Px(),vmu.Py(),vmu.Pz()))
                if not self.has_hit_info:
                    branches["Muon_nExpectedPixelHits"].push_back(0)

                goodmuon = (
                        (muon.trackIso() < 0.1) and
                        (muon.chi2()/muon.ndof() < 3.0) and
                        (muon.nValidMuonHits() > 0) and
                        ((drjet < 0) or (drjet > 0.3))
                        )
                ngoodmuon += goodmuon
                branches["Muon_good"].push_back(goodmuon)
            branches["nMuon"][0] = len(muons)
            branches["nMuon_good"][0] = ngoodmuon

            metpt_muoncorr = math.hypot(metx_muoncorr, mety_muoncorr)
            metphi_muoncorr = math.atan2(mety_muoncorr, metx_muoncorr)
            branches["MET_pt_muonCorr"][0] = metpt_muoncorr
            branches["MET_phi_muonCorr"][0] = metphi_muoncorr



            # if len(muons) >= 2:
            #     v1 = r.TLorentzVector()
            #     v2 = r.TLorentzVector()
            #     v1.SetPtEtaPhiM(muons[0].pt(), muons[0].eta(), muons[0].phi(), muons[0].m())
            #     v2.SetPtEtaPhiM(muons[1].pt(), muons[1].eta(), muons[1].phi(), muons[1].m())
            #     branches["LeadingPair_mass"][0] = (v1+v2).M()
            #     branches["LeadingPair_sameVtx"][0] = (branches["Muon_vtxIdx1"][0]>=0) and (branches["Muon_vtxIdx1"][0] == branches["Muon_vtxIdx1"][1])
            #     branches["LeadingPair_isOS"][0] = (branches["Muon_charge"][0] == -branches["Muon_charge"][1])
            # else:
            #     branches["LeadingPair_mass"][0] = 0.
            #     branches["LeadingPair_sameVtx"][0] = False
            #     branches["LeadingPair_isOS"][0] = False

            self.outtree.Fill()

        t1 = time.time()

        neventsout = self.outtree.GetEntries()
        self.outtree.Write()
        self.outfile.Close()

        print(">>> Finished slim/skim of {} events in {:.2f} seconds @ {:.1f}Hz".format(ievt,(t1-t0),ievt/(t1-t0)))
        print(">>> Output tree has size {:.1f}MB and {} events".format(os.stat(self.fname_out).st_size/1e6,neventsout))

        if (ievt != nevents_in):
            print(">>> Looped over {} entries instead of {}. Raising exit code=2.".format(ievt,nevents_in))
            sys.exit(2)
        if (self.expected > 0) and (int(self.expected) != ievt):
            print(">>> Expected {} events but ran on {}. Raising exit code=2.".format(self.expected,ievt))
            sys.exit(2)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("fnames", help="input file(s)", nargs="*")
    parser.add_argument("-o", "--output", help="output file name", default="output.root", type=str)
    parser.add_argument("-n", "--nevents", help="max number of events to process (-1 = all)", default=-1, type=int)
    parser.add_argument("-e", "--expected", help="expected number of events", default=-1, type=int)
    parser.add_argument(      "--skim1cm", help="require at least one DV with rho>1cm", action="store_true")
    parser.add_argument("-a", "--allevents", help="don't skim nDV>=1 && nMuon>=2", action="store_true")
    args = parser.parse_args()

    looper = Looper(
            fnames=args.fnames,
            output=args.output,
            nevents=args.nevents,
            expected=args.expected,
            skim1cm=args.skim1cm,
            allevents=args.allevents,
    )
    looper.run()
