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

import socket
import gzip

MUON_MASS = 0.10566

fast = False
if fast:
    print(">>> [!] NOTE: fast option is True, so we will skip some crucial things")

isuaf = any(x in socket.gethostname() for x in ["uaf-","cabinet-","sdsc-"])
def xrootdify(fname):
    if ("/hadoop/cms/store/user/namin/" in fname) and not isuaf:
        fname = "root://redirector.t2.ucsd.edu/" + fname.replace("/hadoop/cms","")
    if fname.startswith("/store"):
        fname = "root://cmsxrootd.fnal.gov/" + fname
    return fname

def argsort(seq, key=lambda x:x):
    if len(seq) == 0:
        return [], []
    # return two lists: indices that sort `seq`, and the sorted `seq`, according to `key` function
    indices, sorted_seq = zip(*sorted(enumerate(seq), key=lambda y:key(y[1])))
    return indices, sorted_seq

def sortwitharg(seq, arg):
    # reorder `seq` according to t
    return [seq[i] for i in arg]

def get_track_reference_point(muon, dvx,dvy,dvz):
    pt = muon.pt()
    eta = muon.eta()
    phi = muon.phi()

    dsz = muon.trk_dsz()
    dz = muon.dz()
    lmb = muon.trk_lambda()
    dxy = muon.dxyCorr

    sinphi = math.sin(phi)
    cosphi = math.cos(phi)
    sinlmb = math.sin(lmb)
    tanlmb = sinlmb/math.cos(lmb)
    refz = 1.0*dz
    refx = -sinphi*dxy - (cosphi/sinlmb)*dsz + (cosphi/tanlmb)*refz
    refy =  cosphi*dxy - (sinphi/sinlmb)*dsz + (sinphi/tanlmb)*refz
    return refx, refy, refz

class Looper(object):

    def __init__(self,fnames=[], output="output.root", nevents=-1, expected=-1, skim1cm=False, allevents=False, treename="Events", year=2018):
        if any("*" in x for x in fnames):
            fnames = sum(map(glob.glob,fnames),[])
        self.fnames = map(xrootdify,sum(map(lambda x:x.split(","),fnames),[]))
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
        self.year = year
        self.ch = None
        self.outtree = None
        self.outfile = None

        if "Run2017" in self.fnames[0]:
            self.year = 2017
            print(">>> Autodetected year and overrided it to {}".format(self.year))
        if "Run2018" in self.fnames[0]:
            self.year = 2018
            print(">>> Autodetected year and overrided it to {}".format(self.year))

        # beamspot stuff - index is [year][is_mc]
        self.bs_data = { 
                2017: {False: {}, True: {}},
                2018: {False: {}, True: {}},
                }
        self.goldenjson_data = {}

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
        self.has_bs_info = any("beamSpotMaker" in name for name in branchnames)

        if not self.has_trigger_info:
            print(">>> [!] Didn't find trigger branches. Saving dummy trigger information.")
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
            ch.SetBranchStatus("*hitMaker*nexpectedhitsmultiple*",1)
        if self.has_gen_info:
            ch.SetBranchStatus("*genParticles*",1)
        if self.has_bs_info:
            ch.SetBranchStatus("*beamSpotMaker*",1)

        self.outfile = r.TFile(self.fname_out, "recreate")
        # self.outfile.SetCompressionSettings(int(404)) # https://root.cern.ch/doc/master/Compression_8h_source.html
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

    def load_goldenjson_data(self):
        print(">>> Loading goldenjson data")
        t0 = time.time()
        d = {}
        with open("data/goldenjson_run2_snt.csv","r") as fh:
            fh.readline() # skip header
            for iline,line in enumerate(fh):
                run, lumilow, lumihigh = map(int,line.split(","))
                if run < 294927: continue # skip everything before beginning of 2017
                if run not in d: d[run] = []
                d[run].append((lumilow,lumihigh))
        self.goldenjson_data = d
        t1 = time.time()
        print(">>> Finished loading in {:.1f} seconds".format(t1-t0))

    def is_in_goldenjson(self, run, lumi):
        if self.is_mc: return True
        if not self.goldenjson_data:
            self.load_goldenjson_data()
        return any(low <= lumi <= high for low, high in self.goldenjson_data.get(run, [(0,0)]))

    def load_bs_data(self, year):
        if self.is_mc:
            # Events->Scan("recoBeamSpot_offlineBeamSpot__RECO.obj.x0()") in miniaod (and y0). 
            # From 2017 MC with global tag of 94X_mc2017_realistic_v14
            self.bs_data[2017][self.is_mc] = { (0,0): [-0.024793, 0.0692861, 0.789895] }
            # From 2018 MC with global tag of 102X_upgrade2018_realistic_v11
            self.bs_data[2018][self.is_mc] = { (0,0): [0.0107796, 0.041893, 0.0248755] }
        else:
            print(">>> Loading beamspot data for year={}".format(year))
            t0 = time.time()
            data = []
            # with gzip.open("data/beamspots_{}.pkl.gz".format(year),"r") as fh:
            with open("data/beamspots_{}.pkl".format(year),"r") as fh:
                data = pickle.load(fh)
            for run,lumi,x,y,z in data:
                self.bs_data[year][self.is_mc][(run,lumi)] = [x,y,z]
            t1 = time.time()
            print(">>> Finished loading {} rows in {:.1f} seconds".format(len(data),t1-t0))

    def get_bs(self, run, lumi, year=2018):
        if fast: return 0., 0., 0.
        if self.is_mc: run,lumi = 0,0
        if not self.bs_data[year][self.is_mc]:
            self.load_bs_data(year=year)
        data = self.bs_data[year][self.is_mc]
        xyz = data.get((run,lumi),None)
        if xyz is None:
            xyz = data.get((0,0),[0,0,0])
            print(">>> WARNING: Couldn't find (run={},lumi={},is_mc={},year={}) in beamspot lookup data. Falling back to the total mean: {}".format(run,lumi,self.is_mc,year,xyz))
        return xyz

    def load_pixel_code(self):
        if not self.loaded_pixel_code:
            print(">>> Loading pixel utilities and lookup tables")
            t0 = time.time()
            r.gROOT.ProcessLine(".L data/calculate_pixel.cc")
            self.loaded_pixel_code = True
            t1 = time.time()
            print(">>> Finished loading in {:.1f} seconds".format(t1-t0))

    def in_pixel_rectangles(self,px,py,pz):
        if fast: return False
        self.load_pixel_code()
        rho = math.hypot(px,py)
        if (0.0 < rho < 2.4): return False
        if (3.7 < rho < 5.7): return False
        return r.is_point_in_any_module(px, py, pz)

    def init_branches(self):

        make_branch = self.make_branch

        make_branch("run", "l")
        make_branch("luminosityBlock", "l")
        make_branch("event", "l")

        make_branch("pass_skim", "b")
        make_branch("pass_l1", "b")
        make_branch("pass_json", "b")
        make_branch("pass_fiducialgen", "b")
        make_branch("pass_excesshits", "b")
        make_branch("pass_fiducialgen_norho", "b")

        # more event level
        make_branch("dimuon_isos", "b")
        make_branch("dimuon_pt", "f")
        make_branch("dimuon_eta", "f")
        make_branch("dimuon_phi", "f")
        make_branch("dimuon_mass", "f")
        make_branch("absdphimumu", "f")
        make_branch("absdphimudv", "f")
        make_branch("minabsdxy", "f")
        make_branch("logabsetaphi", "f")
        make_branch("lxy", "f")
        make_branch("cosphi", "f")
        make_branch("pass_baseline", "b")
        make_branch("pass_baseline_iso", "b")

        make_branch("MET_pt", "f")
        make_branch("MET_phi", "f")
        make_branch("rho", "f")

        make_branch("nDV", "i")
        make_branch("nDV_passid", "i")
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
        make_branch("DV_passid","vb")
        make_branch("DV_rho", "vf")
        make_branch("DV_rhoCorr", "vf")
        make_branch("DV_inPixelRectangles", "vb")


        if self.do_jets:
            make_branch("nJet", "i")
            make_branch("Jet_pt", "vf")
            make_branch("Jet_eta", "vf")
            make_branch("Jet_phi", "vf")
            make_branch("Jet_m", "vf")

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
        make_branch("nMuon_passid", "i")
        make_branch("nMuon_passiso", "i")
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
        # make_branch("Muon_trk_refx", "vf")
        # make_branch("Muon_trk_refy", "vf")
        # make_branch("Muon_trk_refz", "vf")
        make_branch("Muon_dxyCorr", "vf")
        make_branch("Muon_nExpectedPixelHits", "vi")
        make_branch("Muon_jetIdx1", "vi")
        make_branch("Muon_drjet", "vf")
        make_branch("Muon_passid", "vb")
        make_branch("Muon_passiso", "vb")

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
        make_branch("BS_dxdz", "f")
        make_branch("BS_dydz", "f")

        make_branch("L1_DoubleMu4p5_SQ_OS_dR_Max1p2", "b")
        make_branch("L1_DoubleMu4_SQ_OS_dR_Max1p2", "b")
        make_branch("L1_DoubleMu0er1p4_SQ_OS_dR_Max1p4", "b")
        make_branch("L1_DoubleMu_15_7", "b")

        if self.year == 2018:
            self.seeds_to_OR =   ["L1_DoubleMu4p5_SQ_OS_dR_Max1p2","L1_DoubleMu0er1p4_SQ_OS_dR_Max1p4","L1_DoubleMu_15_7"]
        else:
            self.seeds_to_OR =   ["L1_DoubleMu4_SQ_OS_dR_Max1p2",  "L1_DoubleMu0er1p4_SQ_OS_dR_Max1p4","L1_DoubleMu_15_7"]
        self.seeds_to_save = ["L1_DoubleMu4_SQ_OS_dR_Max1p2", "L1_DoubleMu4p5_SQ_OS_dR_Max1p2","L1_DoubleMu0er1p4_SQ_OS_dR_Max1p4","L1_DoubleMu_15_7"]

        self.outtree.SetBasketSize("*",int(1*1024*1024))

    def run(self):

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
            if self.do_skimreco and (len(dvs)<1): continue

            muons = evt.ScoutingMuons_hltScoutingMuonPackerCalo__HLT.product()
            if self.do_skimreco and (len(muons)<2): continue

            if self.do_skim1cm and len(dvs) >= 1:
                if not any(math.hypot(dv.x(),dv.y())>1. for dv in dvs): continue

            self.clear_branches()

            branches["pass_skim"][0] = (len(dvs) >= 1) and (len(muons) >= 2)
            branches["pass_l1"][0] = False

            # sort muons in descending pT (turns out a nontrivial amount are not sorted already, like 5%-10% I think)
            # muons = sorted(muons, key=lambda x:-x.pt())
            muon_sort_indices, muons = argsort(muons, key=lambda x:-x.pt())

            if self.do_trigger:
                theOR = False
                l1results = map(bool,evt.bools_triggerMaker_l1result_SLIM.product())
                l1names = list(evt.Strings_triggerMaker_l1name_SLIM.product())
                for bit,name in zip(l1results,l1names):
                    if name not in self.seeds_to_save: continue
                    branches[name][0] = bit
                    if name not in self.seeds_to_OR: continue
                    theOR = bit or theOR
                branches["pass_l1"][0] = theOR
            else:
                branches["pass_l1"][0] = True


            run = int(evt.EventAuxiliary.run())
            lumi = int(evt.EventAuxiliary.luminosityBlock())
            eventnum = int(evt.EventAuxiliary.event())
            branches["run"][0] = run
            branches["luminosityBlock"][0] = lumi
            branches["event"][0] = eventnum

            # branches["pass_json"][0] = self.is_in_goldenjson(run, lumi)
            branches["pass_json"][0] = True # started requiring golden upstream

            metpt = evt.double_hltScoutingCaloPacker_caloMetPt_HLT.product()[0]
            metphi = evt.double_hltScoutingCaloPacker_caloMetPhi_HLT.product()[0]
            branches["MET_pt"][0] = metpt
            branches["MET_phi"][0] = metphi
            branches["rho"][0] = evt.double_hltScoutingCaloPacker_rho_HLT.product()[0]

            if self.has_bs_info:
                bsx = float(evt.float_beamSpotMaker_x_SLIM.product()[0])
                bsy = float(evt.float_beamSpotMaker_y_SLIM.product()[0])
                bsz = float(evt.float_beamSpotMaker_z_SLIM.product()[0])
                bsdxdz = float(evt.float_beamSpotMaker_dxdz_SLIM.product()[0])
                bsdydz = float(evt.float_beamSpotMaker_dydz_SLIM.product()[0])
            else:
                bsx,bsy,bsz = self.get_bs(run=run,lumi=lumi,year=self.year)
                bsdxdz, bsdydz = 0., 0.
            branches["BS_x"][0] = bsx
            branches["BS_y"][0] = bsy
            branches["BS_z"][0] = bsz
            branches["BS_dxdz"][0] = bsdxdz
            branches["BS_dydz"][0] = bsdydz

            # NOTE, we correct x-y quantities using first PV from this collection
            pvms = evt.ScoutingVertexs_hltScoutingPrimaryVertexPackerCaloMuon_primaryVtx_HLT.product()
            for pvm in pvms:
                branches["PVM_x"].push_back(pvm.x())
                branches["PVM_y"].push_back(pvm.y())
                branches["PVM_z"].push_back(pvm.z())
                branches["PVM_tracksSize"].push_back(pvm.tracksSize())
                branches["PVM_chi2"].push_back(pvm.chi2())
                branches["PVM_ndof"].push_back(pvm.ndof())
                branches["PVM_isValidVtx"].push_back(pvm.isValidVtx())
            branches["nPVM"][0] = len(pvms)
            pvmx = bsx if not len(pvms) else pvms[0].x()
            pvmy = bsy if not len(pvms) else pvms[0].y()

            ndv_passid = 0
            for dv in dvs:
                vx = dv.x()
                vy = dv.y()
                vz = dv.z()
                rho = (vx**2 + vy**2)**0.5
                rhoCorr = ((vx-pvmx)**2 + (vy-pvmy)**2)**0.5
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
                gooddv = (
                        (dv.xError() < 0.05) 
                        and (dv.yError() < 0.05) 
                        and (dv.zError() < 0.10)
                        and (dv.chi2()/dv.ndof() < 5)
                        )
                ndv_passid += gooddv
                branches["DV_passid"].push_back(gooddv)
            branches["nDV"][0] = len(dvs)
            branches["nDV_passid"][0] = ndv_passid

            pvs = evt.ScoutingVertexs_hltScoutingPrimaryVertexPacker_primaryVtx_HLT.product()
            branches["nPV"][0] = len(pvs)


            if self.do_jets:
                jets = evt.ScoutingCaloJets_hltScoutingCaloPacker__HLT.product()
                jets = sorted(jets, key=lambda x:-x.pt())
                jet_etaphis = []
                for jet in jets:
                    branches["Jet_pt"].push_back(jet.pt())
                    branches["Jet_eta"].push_back(jet.eta())
                    branches["Jet_phi"].push_back(jet.phi())
                    branches["Jet_m"].push_back(jet.m())
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
                if abs(pdgid) not in [13,23,25,6000211]: continue
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
                if ((motherid == 23) or (motherid == 6000211)) and (abs(pdgid)==13): 
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

            if self.has_hit_info:
                # NOTE, need to sort these to maintain same order as muons since we have sorted them by pT earlier
                muon_hit_expectedhitsmultiple = sortwitharg(evt.ints_hitMaker_nexpectedhitsmultiple_SLIM.product(), muon_sort_indices)
                for i in range(len(muon_hit_expectedhitsmultiple)):
                    branches["Muon_nExpectedPixelHits"].push_back(muon_hit_expectedhitsmultiple[i])

            nmuon_passid = 0
            nmuon_passiso = 0
            for imuon,muon in enumerate(muons):
                pt = muon.pt()
                eta = muon.eta()
                phi = muon.phi()
                branches["Muon_pt"].push_back(pt)
                branches["Muon_eta"].push_back(eta)
                branches["Muon_phi"].push_back(phi)
                branches["Muon_m"].push_back(MUON_MASS) # hardcode since otherwise we get 0.
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
                drjet = -1
                if self.do_jets:
                    # find index of jet that is closest to this muon. `sorted_etaphis`: [(index, (eta,phi)), ...]
                    sorted_etaphis = sorted(enumerate(jet_etaphis), key=lambda x: math.hypot(eta-x[1][0], phi-x[1][1]))
                    if len(sorted_etaphis) > 0: jetIdx1 = sorted_etaphis[0][0]
                    if jetIdx1 >= 0:
                        jeteta, jetphi = sorted_etaphis[0][1]
                        drjet = math.hypot(eta-jeteta,phi-jetphi)
                branches["Muon_drjet"].push_back(drjet)
                branches["Muon_jetIdx1"].push_back(jetIdx1)

                indices = muon.vtxIndx()
                num = len(indices)
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
                    vx, vy, vz = dv.x(), dv.y(), dv.z()
                    phi = muon.phi()
                    # https://github.com/cms-sw/cmssw/blob/master/DataFormats/TrackReco/interface/TrackBase.h#L24
                    muon.dxyCorr = -(vx-pvmx)*math.sin(phi) + (vy-pvmy)*math.cos(phi)
                else:
                    # fill muon vertex branches with dummy values since there are no DVs to even look at
                    vx, vy, vz = 0, 0, 0
                    muon.dxyCorr = dxy
                branches["Muon_vx"].push_back(vx)
                branches["Muon_vy"].push_back(vy)
                branches["Muon_vz"].push_back(vz)
                branches["Muon_dxyCorr"].push_back(muon.dxyCorr)
                # refx,refy,refz = get_track_reference_point(muon, vx,vy,vz)
                # branches["Muon_trk_refx"].push_back(refx)
                # branches["Muon_trk_refy"].push_back(refy)
                # branches["Muon_trk_refz"].push_back(refz)
                if not self.has_hit_info:
                    branches["Muon_nExpectedPixelHits"].push_back(0)

                # nMatchedStations hardcoded to 0 in 2017 HLT code:
                # https://github.com/cms-sw/cmssw/blob/CMSSW_9_2_10/HLTrigger/Muon/src/HLTScoutingMuonProducer.cc#L161
                # and nValidMuonHits also 0, so we scrap this cut, even for 2018, for simplicity
                muon_passid = (
                        (muon.chi2()/muon.ndof() < 3.0) and
                        # (muon.nValidMuonHits() > 0) and
                        (muon.nTrackerLayersWithMeasurement() > 5)
                        )
                muon_passiso = (
                        (muon.trackIso() < 0.1) 
                        and ((drjet < 0) or (drjet > 0.3))
                        )
                nmuon_passid += muon_passid
                nmuon_passiso += muon_passiso
                branches["Muon_passid"].push_back(muon_passid)
                branches["Muon_passiso"].push_back(muon_passiso)
            branches["nMuon"][0] = len(muons)
            branches["nMuon_passid"][0] = nmuon_passid
            branches["nMuon_passiso"][0] = nmuon_passiso

            # some event level things if we pass a pre-selection of at least 2 muons and at least 1 dv
            # we take the leading 2 muons and leading DV
            if len(muons) >= 2 and len(dvs) >= 1:
                mu1 = muons[0]
                mu2 = muons[1]
                dv1 = dvs[0]
                mu1p4 = r.TLorentzVector()
                mu2p4 = r.TLorentzVector()
                mu1p4.SetPtEtaPhiM(mu1.pt(), mu1.eta(), mu1.phi(), MUON_MASS)
                mu2p4.SetPtEtaPhiM(mu2.pt(), mu2.eta(), mu2.phi(), MUON_MASS)
                dimuon = (mu1p4+mu2p4)
                vecdv2d = r.TVector2(dv1.x()-pvmx, dv1.y()-pvmy)
                vecdimuon2d = r.TVector2(dimuon.Px(),dimuon.Py())
                cosphi = (vecdv2d.Px()*vecdimuon2d.Px() + vecdv2d.Py()*vecdimuon2d.Py()) / (vecdv2d.Mod()*vecdimuon2d.Mod())

                branches["dimuon_isos"][0] = mu1.charge()*mu2.charge() < 0
                branches["dimuon_pt"][0] = dimuon.Pt()
                branches["dimuon_eta"][0] = dimuon.Eta()
                branches["dimuon_phi"][0] = dimuon.Phi()
                branches["dimuon_mass"][0] = dimuon.M()
                branches["absdphimumu"][0] = abs(mu1p4.DeltaPhi(mu2p4))
                branches["absdphimudv"][0] = abs(vecdimuon2d.DeltaPhi(vecdv2d))
                branches["minabsdxy"][0] = min(abs(branches["Muon_dxyCorr"][0]),abs(branches["Muon_dxyCorr"][1]))
                branches["logabsetaphi"][0] = math.log10(max(abs(mu1p4.Eta()-mu2p4.Eta()),1e-6)/max(abs(mu1p4.DeltaPhi(mu2p4)),1e-6))
                branches["cosphi"][0] = cosphi
                # definition on s2 of https://indico.cern.ch/event/846681/contributions/3557724/attachments/1907377/3150380/Displaced_Scouting_Status_Update.pdf
                # branches["Lxy"][0] = vecdv2d_pvm.Mod() * cosphi_pvm
                # rutgers lxy is lowercase lxy from that set of slides, which does not have the cosine term
                # branches["lxy"][0] = vecdv2d_pvm.Mod()
                branches["lxy"][0] = vecdv2d.Mod()

                # both muons need to have no excess hits if the displacement is >3.5cm (otherwise we're within the 1st bpix layer and extra hits don't make sense to calculate)
                branches["pass_excesshits"][0] = (
                        (branches["DV_rhoCorr"][0] < 3.5) or 
                            ( (branches["Muon_nValidPixelHits"][0] - branches["Muon_nExpectedPixelHits"][0] <= 0) and
                              (branches["Muon_nValidPixelHits"][1] - branches["Muon_nExpectedPixelHits"][1] <= 0)
                            )
                        )

                # now our baseline selection is
                # *exactly* 2 muons, 1 DV, and both muons and DV pass at least ID
                # with some kinematic cuts
                pass_baseline = (
                        True
                        and (len(muons) == 2)
                        and (len(dvs) == 1)
                        and (nmuon_passid == 2)
                        and (ndv_passid == 1)
                        and (branches["cosphi"][0] > 0)
                        and (branches["absdphimumu"][0] < 2.8)
                        and (branches["absdphimudv"][0] < 0.02)
                        and (branches["dimuon_isos"][0])
                        and (branches["pass_l1"][0])
                        and (branches["pass_json"][0])
                        and (branches["DV_rhoCorr"][0] < 11.)
                        # and (branches["pass_fiducialgen"][0])
                        # and (branches["pass_excesshits"][0])
                        )
                pass_baseline_iso = (nmuon_passiso == 2) and pass_baseline
                branches["pass_baseline"][0] = pass_baseline
                branches["pass_baseline_iso"][0] = pass_baseline_iso


            self.outtree.Fill()

        t1 = time.time()

        neventsout = self.outtree.GetEntries()
        self.outtree.Write()

        
        # number of events in the input chain
        r.TParameter(int)("nevents_input",nevents_in).Write()
        # number of events we actually looped over
        r.TParameter(int)("nevents_processed",ievt).Write()
        # number of events in the output tree
        r.TParameter(int)("nevents_output",self.outtree.GetEntries()).Write()

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
    parser.add_argument("-y", "--year", help="year (2017 or 2018)", default=2018, type=int)
    args = parser.parse_args()

    looper = Looper(
            fnames=args.fnames,
            output=args.output,
            nevents=args.nevents,
            expected=args.expected,
            skim1cm=args.skim1cm,
            allevents=args.allevents,
            year=args.year,
    )
    looper.run()
