from __future__ import print_function

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

# from https://github.com/root-project/root/blob/7e47c421a88fdd75b7dce058ece9b4f31135b41b/bindings/pyroot/ROOT.py#L230
def get_iter(ch, entrystart=None, entrystop=None):
   i = 0
   if entrystart is not None: i += entrystart
   bytes_read = ch.GetEntry(i)
   while (bytes_read > 0) and ((entrystop is None) or (i < entrystop-1)):
      yield i, ch
      i += 1
      bytes_read = ch.GetEntry(i)
   if bytes_read == -1:
      raise RuntimeError( "TTree I/O error" )


def delta_phi(phi1,phi2):
    return (phi1 - phi2 + math.pi) % (2*math.pi) - math.pi

def delta_r(eta1,eta2,phi1,phi2):
    return math.hypot(eta1-eta2, delta_phi(phi1,phi2))


def get_track_reference_point(muon):
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

def does_dv_pass_id(dv):
    if dv.xError() > 0.05: return False
    if dv.yError() > 0.05: return False
    if dv.zError() > 0.10: return False
    if dv.chi2()/dv.ndof() > 5: return False
    if dv.rhoCorr > 11.: return False
    return True

def pick_best_objects(dvs, muons, run, is_mc):
    """
    Given a list of DVs, list of muons, run number, is_mc flag,
    return a dict of information about the index of the best DV, associated muons.

    Algorithm selects the best DV (with minimum max(xError,yError)) out
    of all the DVs that pass analysis ID requirements. Then the (exactly) two muons
    that are associated with that DV (based on the vtxIndx() vectors for all the
    muons), are selected as well. Several pathological cases where a DV has number of 
    associated muons != 2 are rejected.

    The index is wrt the original CMSSW collections.
    If >=2 DVs and >=4 muons, store additional information about subleading DV if its 2 muons
    are distinct from the leading DV's muons.

    Might return something like
    {'bestidv': 0, 'bestimu2': 1, 'bestimu1': 0, 'secondary': {'bestidv': 1, 'bestimu2': 2, 'bestimu1': 3}}
    (most likely without the "secondary" key)

    Note for 2017 data (not MC), might have vtxIndx for 5 muons/3 DVs that looks like
        [0]
        [0, 1]
        [0, 1, 0, 2]
        [0, 1, 0, 2, 2]
        [0, 1, 0, 2, 2, 1]
    Due to a missing vector clear() statement:
    https://github.com/cms-sw/cmssw/commit/6296892159b21e3add611cb43b485560e8b06075 
    Only consider the new elements on the right to get:
        [0]
        [1]
        [0, 2]
        [2]
        [1]
    This issue exists in 2017 data for all runs before Run 305405 (which happens in the middle of 2017F)
    """
    ret = {}
    ret["bestimu1"] = -1
    ret["bestimu2"] = -1
    ret["bestidv"] = -1

    bugged_branches = ((100000 < run < 305405) and not is_mc)

    # Make some mappings from muon to DV indices and vice versa
    d_imuon_to_idv = {}
    d_imuon_to_pt = {}
    d_idv_to_imuon = {}
    curr_length = 0
    for imuon,muon in enumerate(muons):
        indices = list(muon.vtxIndx())
        if bugged_branches:
            indices = indices[curr_length:]
            curr_length += len(indices)
        d_imuon_to_idv[imuon] = indices
        d_imuon_to_pt[imuon] = muon.pt()
        for idx in indices:
            if idx not in d_idv_to_imuon: d_idv_to_imuon[idx] = []
            d_idv_to_imuon[idx].append(imuon)

    # Make a mapping of *good* dv indices to max(xError, yError)
    d_goodidv_to_xyerrormax = {}
    for idv, dv in enumerate(dvs):
        xerror = dv.xError()
        yerror = dv.yError()
        zerror = dv.zError()
        if not dv.passid: continue
        d_goodidv_to_xyerrormax[idv] = max(xerror, yerror)
    sortedidvs = sorted(d_goodidv_to_xyerrormax.keys(), key=d_goodidv_to_xyerrormax.__getitem__)

    # Best DV is one with lowest max(xError, yError)
    if len(sortedidvs) > 0: ret["bestidv"] = sortedidvs[0]

    # If no DV, return
    if ret["bestidv"] < 0:
        return ret

    # It could (rarely) be the case that NO muon is associated to a DV, then
    # d_idv_to_imuon will not have a key/value for that DV index
    if ret["bestidv"] not in d_idv_to_imuon:
        print("There are NO muons associated to this DV, which has tracksSize={}".format(dv.tracksSize()))
        return ret

    # Need exactly 2 associated muons (no more, no less). This is satisfied by 99.9+% of DVs.
    num_associated_muons = len(d_idv_to_imuon[ret["bestidv"]])
    if num_associated_muons != 2:
        print("There's {} muon(s) associated to this DV, which has tracksSize={}".format(num_associated_muons, dv.tracksSize()))
    else:
        # get indices of two muons for that DV. make sure first one has higher pT.
        ret["bestimu1"], ret["bestimu2"] = d_idv_to_imuon[ret["bestidv"]]
        if d_imuon_to_pt[ret["bestimu2"]] > d_imuon_to_pt[ret["bestimu1"]]:
            ret["bestimu1"], ret["bestimu2"] = ret["bestimu2"], ret["bestimu1"]
        # if >=2 DVs and >=4 muons, store information about subleading DV if its 2 muons
        # are distinct from the leading DV's muons
        if len(sortedidvs) >= 2 and len(muons) >= 4:
            idv2 = sortedidvs[1]
            imus = d_idv_to_imuon[idv2]
            if (len(imus) == 2 and (
                    (imus[0] != ret["bestimu1"]) and
                    (imus[0] != ret["bestimu2"]) and
                    (imus[1] != ret["bestimu1"]) and
                    (imus[1] != ret["bestimu2"])
                    )):
                imu1, imu2 = imus
                if d_imuon_to_pt[imu2] > d_imuon_to_pt[imu1]:
                    imu1, imu2 = imu2, imu1
                ret["secondary"] = dict(bestidv=idv2, bestimu1=imu1, bestimu2=imu2)
    return ret

class PrintableMixin(object):
    def __repr__(self):
        my_attrs = self.__dict__.keys()
        strs = []
        for x in dir(self):
            if x.startswith("_"): continue
            val = getattr(self,x)
            if x not in my_attrs: val = val()
            if "vector" in type(val).__name__: val = list(val)
            strs.append("{}={}".format(x,val))
        varstr = " ".join(strs)
        return "<{} at {}: {}>".format(self.__class__.__name__, hex(id(self)), varstr)

class VertexWrapper(PrintableMixin, r.ScoutingVertex):
    pass

class MuonWrapper(PrintableMixin, r.ScoutingMuon):
    pass

def write_metadata_to_tfile(tfile, metadata):
    # Given TFile and dict of metadata, write some TObjStrings to the file
    for k, v in metadata.items():
        obj = r.TString(str(v))
        tfile.WriteObject(obj, k)

class Looper(object):

    def __init__(self,fnames=[], output="output.root", nevents=-1, expected=-1, treename="Events", year=2018):
        if any("*" in x for x in fnames):
            fnames = sum(map(glob.glob,fnames),[])
        self.fnames = map(xrootdify,sum(map(lambda x:x.split(","),fnames),[]))
        self.nevents = nevents
        self.do_tracks = False
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

        self.loaded_pixel_code = False
        self.loaded_prop_code = False

        self.init_tree()
        self.init_branches()


    def init_tree(self):

        self.ch = r.TChain(self.treename)

        # alias
        ch = self.ch

        print(">>> Started making TChain with {} files".format(len(self.fnames)))

        for fname in self.fnames:
            ch.Add(fname)

        print(">>> Done making TChain")

        branchnames = [b.GetName() for b in ch.GetListOfBranches()]
        self.has_gen_info = any("genParticles" in name for name in branchnames)
        self.is_mc = self.has_gen_info
        self.has_trigger_info = any("triggerMaker" in name for name in branchnames)
        self.has_hit_info = any("hitMaker" in name for name in branchnames)
        self.has_bs_info = any("beamSpotMaker" in name for name in branchnames)

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
        # Python: https://docs.python.org/2/library/array.html
        # ROOT: https://root.cern.ch/doc/v612/classTTree.html
        extra = []
        if tstr == "vvi": obj = r.vector("vector<int>")()
        if tstr == "vvf": obj = r.vector("vector<float>")()
        if tstr == "vvb": obj = r.vector("vector<bool>")()
        if tstr == "vi": obj = r.vector("int")()
        if tstr == "vf": obj = r.vector("float")()
        if tstr == "vb": obj = r.vector("bool")()
        if tstr == "f":
            obj = array.array("f",[999]) # float
            extra.append("{}/F".format(name)) # Float_t
        if tstr == "b":
            obj = array.array("b",[0]) # signed char
            extra.append("{}/O".format(name)) # Bool_t
        if tstr == "i":
            obj = array.array("i",[999]) # signed int
            extra.append("{}/I".format(name)) # Int_t
        if tstr == "l":
            obj = array.array("L",[999]) # unsigned long
            extra.append("{}/L".format(name)) # Long64_t
        self.branches[name] = obj
        self.outtree.Branch(name,obj,*extra)

    def clear_branches(self):
        for v in self.branches.values():
            if hasattr(v,"clear"):
                v.clear()
            elif v.typecode in ["f", "i", "L"]:
                v[0] = 999
            elif v.typecode in ["b"]:
                v[0] = 0


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

    def load_prop_code(self):
        if not self.loaded_prop_code:
            print(">>> Loading propagation utilities")
            t0 = time.time()
            r.gROOT.ProcessLine(".L data/propagation_utils.cc")
            self.loaded_prop_code = True
            t1 = time.time()
            print(">>> Finished loading in {:.1f} seconds".format(t1-t0))

    def get_pixel_rectangle_info(self,px,py,pz):
        if fast: return dict()
        self.load_pixel_code()
        rho = math.hypot(px,py)
        if (0.0 < rho < 2.4): return dict()
        if (3.7 < rho < 5.7): return dict()
        imodule = r.point_in_which_module(px, py, pz)
        return dict(
                imodule = imodule,
                planedist = r.dist_to_imodule_plane(px, py, pz, imodule),
                layernum = r.imodule_to_layernum(imodule),
                )

    def get_corrected_phi(self,muon,dv):
        if fast: return muon.phi()
        self.load_prop_code()
        refx, refy, refz = get_track_reference_point(muon)
        vec = r.TLorentzVector()
        vec.SetPtEtaPhiM(muon.pt(), muon.eta(), muon.phi(), MUON_MASS)
        newphi = r.recalculate_phi_at_DV(
            refx, refy, refz, vec.Px(), vec.Py(), vec.Pz(),
            muon.charge(),
            dv.x(), dv.y(),
            )
        return newphi

    def init_branches(self):

        make_branch = self.make_branch

        make_branch("run", "l")
        make_branch("luminosityBlock", "l")
        make_branch("event", "l")

        make_branch("year", "i")

        make_branch("pass_l1", "b")
        # make_branch("pass_fiducialgen", "b")
        make_branch("pass_excesshits", "b")
        make_branch("pass_materialveto", "b")
        make_branch("pass_dxyscaled", "b")
        make_branch("pass_dxysig", "b")
        # make_branch("pass_fiducialgen_norho", "b")
        make_branch("pass_genmatch", "b")

        # more event level
        make_branch("dimuon_isos", "b")
        make_branch("dimuon_pt", "f")
        make_branch("dimuon_eta", "f")
        make_branch("dimuon_phi", "f")
        make_branch("dimuon_mass", "f")
        make_branch("mass", "f")
        make_branch("dimuon_massRaw", "f")
        make_branch("absdphimumu", "f")
        make_branch("absdphimudv", "f")
        make_branch("minabsdxy", "f")
        make_branch("ctau", "f")
        make_branch("logabsetaphi", "f")
        make_branch("lxy", "f")
        make_branch("lxyError", "f")
        make_branch("cosphi", "f")
        make_branch("pass_baseline", "b")
        make_branch("pass_baseline_iso", "b")
        make_branch("pass_baseline_isohalf", "b")
        make_branch("pass_baseline_extra", "b")
        make_branch("pass_baseline_extra_iso", "b")
        make_branch("pass_baseline_extra_isohalf", "b")

        make_branch("sublead_dimuon_isos", "b")
        make_branch("sublead_dimuon_mass", "f")
        make_branch("sublead_logabsetaphi", "f")
        make_branch("sublead_lxy", "f")

        make_branch("MET_pt", "f")
        make_branch("MET_phi", "f")
        make_branch("rho", "f")

        make_branch("nDV_raw", "i")
        make_branch("nDV", "i")
        for pfx in ["DV_", "sublead_DV_"]:
            make_branch(pfx+"x","f")
            make_branch(pfx+"y","f")
            make_branch(pfx+"z","f")
            make_branch(pfx+"xError","f")
            make_branch(pfx+"yError","f")
            make_branch(pfx+"zError","f")
            make_branch(pfx+"chi2","f")
            make_branch(pfx+"ndof","i")
            make_branch(pfx+"rho", "f")
            make_branch(pfx+"rhoCorr", "f")
            make_branch(pfx+"inPixel", "b")
            make_branch(pfx+"distPixel", "f")
            make_branch(pfx+"layerPixel", "i")
            make_branch(pfx+"passid","b")

        make_branch("nPVM_raw", "i")
        make_branch("PVM_x", "f")
        make_branch("PVM_y", "f")
        make_branch("PVM_z", "f")
        make_branch("PVM_chi2", "f")
        make_branch("PVM_ndof", "i")

        make_branch("nJet", "i")
        make_branch("Jet_pt", "vf")
        make_branch("Jet_eta", "vf")
        make_branch("Jet_phi", "vf")
        make_branch("Jet_m", "vf")
        make_branch("Jet_mvaDiscriminator", "vf")
        make_branch("Jet_btagDiscriminator", "vf")

        make_branch("nMuon_raw", "i")
        make_branch("nMuon", "i")
        for pfx in ["Muon1_", "Muon2_", "sublead_Muon1_", "sublead_Muon2_"]:
            make_branch(pfx+"pt", "f")
            make_branch(pfx+"eta", "f")
            make_branch(pfx+"phi", "f")
            make_branch(pfx+"m", "f")
            make_branch(pfx+"trackIso", "f")
            make_branch(pfx+"chi2", "f")
            make_branch(pfx+"ndof", "f")
            make_branch(pfx+"charge", "i")
            make_branch(pfx+"dxy", "f")
            make_branch(pfx+"dz", "f")
            make_branch(pfx+"nValidMuonHits", "i")
            make_branch(pfx+"nValidPixelHits", "i")
            make_branch(pfx+"nMatchedStations", "i")
            make_branch(pfx+"nTrackerLayersWithMeasurement", "i")
            make_branch(pfx+"nValidStripHits", "i")
            make_branch(pfx+"dxyError", "f")
            make_branch(pfx+"dzError", "f")
            make_branch(pfx+"dxyCorr", "f")
            make_branch(pfx+"nExpectedPixelHits", "i")
            make_branch(pfx+"drjet", "f")
            make_branch(pfx+"passid", "b")
            make_branch(pfx+"passiso", "b")
            make_branch(pfx+"genMatch_dr", "f") # genMatch branches are for matched gen muons
            make_branch(pfx+"genMatch_pt", "f")
            make_branch(pfx+"genMatch_eta", "f")
            make_branch(pfx+"genMatch_phi", "f")
            make_branch(pfx+"genMatch_m", "f")
            make_branch(pfx+"genMatch_vx", "f")
            make_branch(pfx+"genMatch_vy", "f")
            make_branch(pfx+"genMatch_vz", "f")
            make_branch(pfx+"genMatch_lxy", "f")
            make_branch(pfx+"genMatch_status", "i")
            make_branch(pfx+"genMatch_pdgId", "i")
            make_branch(pfx+"genMatch_motherId", "i") # genMatch_mother for mothers of matched genmuons
            make_branch(pfx+"genMatch_mothervx", "f")
            make_branch(pfx+"genMatch_mothervy", "f")
            make_branch(pfx+"genMatch_mothervz", "f")
            make_branch(pfx+"genMatch_motherct", "f")
            make_branch(pfx+"trk_refx", "f")
            make_branch(pfx+"trk_refy", "f")
            make_branch(pfx+"trk_refz", "f")
            make_branch(pfx+"phiCorr", "f")

            make_branch(pfx+"trk_qoverp", "f")
            make_branch(pfx+"trk_lambda", "f")
            make_branch(pfx+"trk_qoverpError", "f")
            make_branch(pfx+"trk_lambdaError", "f")
            make_branch(pfx+"trk_phiError", "f")
            make_branch(pfx+"trk_dsz", "f")
            make_branch(pfx+"trk_dszError", "f")

        make_branch("GenOther_pt", "vf")
        make_branch("GenOther_eta", "vf")
        make_branch("GenOther_phi", "vf")
        make_branch("GenOther_m", "vf")
        make_branch("GenOther_vx", "vf")
        make_branch("GenOther_vy", "vf")
        make_branch("GenOther_vz", "vf")
        make_branch("GenOther_lxy", "vf")
        make_branch("GenOther_status", "vi")
        make_branch("GenOther_pdgId", "vi")
        make_branch("GenOther_motherId", "vi")

        make_branch("nGenMuon", "i")
        make_branch("GenMuon_pt", "vf")
        make_branch("GenMuon_eta", "vf")
        make_branch("GenMuon_phi", "vf")
        make_branch("GenMuon_m", "vf")
        make_branch("GenMuon_vx", "vf")
        make_branch("GenMuon_vy", "vf")
        make_branch("GenMuon_vz", "vf")
        make_branch("GenMuon_lxy", "vf")
        make_branch("GenMuon_status", "vi")
        make_branch("GenMuon_pdgId", "vi")
        make_branch("GenMuon_motherId", "vi")

        make_branch("BS_x", "f")
        make_branch("BS_y", "f")
        make_branch("BS_z", "f")

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

        l1names = []

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
            muons = evt.ScoutingMuons_hltScoutingMuonPackerCalo__HLT.product()
            if not dvs: dvs = []

            # NOTE, this guarantees HLT bit. if we have a DV in the collection, our HLT trigger of interest
            # must have (and did) fire
            if len(dvs) < 1: continue
            if len(muons) < 2: continue
            # To verify, one could check bool(evt.bools_triggerMaker_hltresult_SLIM.product()[0])
            # for every event, but I already did that.

            self.clear_branches()

            branches["pass_l1"][0] = False

            run = int(evt.EventAuxiliary.run())
            lumi = int(evt.EventAuxiliary.luminosityBlock())
            eventnum = int(evt.EventAuxiliary.event())
            branches["run"][0] = run
            branches["luminosityBlock"][0] = lumi
            branches["event"][0] = eventnum
            branches["year"][0] = self.year

            # NOTE, we correct x-y quantities using first PV from this collection
            pvms = evt.ScoutingVertexs_hltScoutingPrimaryVertexPackerCaloMuon_primaryVtx_HLT.product()
            branches["nPVM_raw"][0] = len(pvms)
            pvmx, pvmy = 0., 0.
            if len(pvms):
                pvmx = pvms[0].x()
                pvmy = pvms[0].y()
                pvmz = pvms[0].z()
                branches["PVM_x"][0] = pvmx
                branches["PVM_y"][0] = pvmy
                branches["PVM_z"][0] = pvmz

            # wrap muons and DVs in custom class to embed information like expected hits
            muons = map(MuonWrapper, muons)
            dvs = map(VertexWrapper, dvs)
            if self.has_hit_info:
                expectedhits = evt.ints_hitMaker_nexpectedhitsmultiple_SLIM.product()
                for i,n in enumerate(expectedhits):
                    muons[i].nExpectedPixelHits = n
            for i in range(len(dvs)):
                vx = dvs[i].x()
                vy = dvs[i].y()
                vz = dvs[i].z()
                dvs[i].pvmx = pvmx
                dvs[i].pvmy = pvmy
                dvs[i].rho = (vx**2 + vy**2)**0.5
                dvs[i].rhoCorr = ((vx-pvmx)**2 + (vy-pvmy)**2)**0.5
                pixinfo = self.get_pixel_rectangle_info(vx,vy,vz)
                dvs[i].inPixel = pixinfo.get("imodule",-1)>=0
                dvs[i].distPixel = pixinfo.get("planedist",999.)
                dvs[i].layerPixel = pixinfo.get("layernum",-1)
                dvs[i].passid = does_dv_pass_id(dvs[i])


            # Get information about best DV and matching 2 muons (in the form of indices)
            info = pick_best_objects(dvs, muons, run, self.is_mc)

            # If we don't have a good DV with 2 matched muons, skip the event
            if info["bestidv"] < 0: continue
            if info["bestimu1"] < 0: continue
            if info["bestimu2"] < 0: continue

            selected_dvs = [dvs[info["bestidv"]]]
            selected_muons = [muons[info["bestimu1"]], muons[info["bestimu2"]]]
            if "secondary" in info:
                selected_dvs.append(dvs[info["secondary"]["bestidv"]])
                selected_muons.append(muons[info["secondary"]["bestimu1"]])
                selected_muons.append(muons[info["secondary"]["bestimu2"]])

            # Start filling things

            ########################################
            ########## # Fill L1 branches ##########
            ########################################
            pass_l1 = False
            l1results = evt.bools_triggerMaker_l1result_SLIM.product()
            # For slight speedup, first (and only once) make a mapping 
            # from seeds of interest -> indices in names vector 
            if not len(l1names):
                l1names = list(evt.Strings_triggerMaker_l1name_SLIM.product())
                l1indices = { name:l1names.index(name) for name in self.seeds_to_save }
            for name,idx in l1indices.items():
                bit = bool(l1results[idx])
                branches[name][0] = bit
                if name in self.seeds_to_OR:
                    pass_l1 = bit or pass_l1
            branches["pass_l1"][0] = pass_l1

            ########################################
            # # Fill MET, energy density branches ##
            ########################################
            metpt = evt.double_hltScoutingCaloPacker_caloMetPt_HLT.product()[0]
            metphi = evt.double_hltScoutingCaloPacker_caloMetPhi_HLT.product()[0]
            branches["MET_pt"][0] = metpt
            branches["MET_phi"][0] = metphi
            branches["rho"][0] = evt.double_hltScoutingCaloPacker_rho_HLT.product()[0]

            ########################################
            ######### # Fill beamspot info #########
            ########################################
            if self.has_bs_info:
                bsx = float(evt.float_beamSpotMaker_x_SLIM.product()[0])
                bsy = float(evt.float_beamSpotMaker_y_SLIM.product()[0])
                bsz = float(evt.float_beamSpotMaker_z_SLIM.product()[0])
            else:
                bsx,bsy,bsz = self.get_bs(run=run,lumi=lumi,year=self.year)
            branches["BS_x"][0] = bsx
            branches["BS_y"][0] = bsy
            branches["BS_z"][0] = bsz

            ########################################
            ########### # Fill jet info ############
            ########################################
            jets = evt.ScoutingCaloJets_hltScoutingCaloPacker__HLT.product()
            branches["nJet"][0] = len(jets)
            jet_etaphis = []
            for jet in jets:
                jet_etaphis.append((jet.eta(),jet.phi()))
                branches["Jet_pt"].push_back(jet.pt())
                branches["Jet_eta"].push_back(jet.eta())
                branches["Jet_phi"].push_back(jet.phi())
                branches["Jet_m"].push_back(jet.m())
                branches["Jet_mvaDiscriminator"].push_back(jet.mvaDiscriminator())
                branches["Jet_btagDiscriminator"].push_back(jet.btagDiscriminator())

            ########################################
            ########## # Fill DV branches ##########
            ########################################
            branches["nDV_raw"][0] = len(dvs)
            branches["nDV"][0] = len(selected_dvs)
            for idv, dv in enumerate(selected_dvs):
                pfx = "DV_" if idv == 0 else "sublead_DV_"
                branches[pfx+"x"][0] = dv.x()
                branches[pfx+"y"][0] = dv.y()
                branches[pfx+"z"][0] = dv.z()
                branches[pfx+"xError"][0] = dv.xError()
                branches[pfx+"yError"][0] = dv.yError()
                branches[pfx+"zError"][0] = dv.zError()
                branches[pfx+"chi2"][0] = dv.chi2()
                branches[pfx+"ndof"][0] = dv.ndof()
                branches[pfx+"rho"][0] = dv.rho
                branches[pfx+"rhoCorr"][0] = dv.rhoCorr
                branches[pfx+"inPixel"][0] = dv.inPixel
                branches[pfx+"distPixel"][0] = dv.distPixel
                branches[pfx+"layerPixel"][0] = dv.layerPixel
                branches[pfx+"passid"][0] = dv.passid

            ########################################
            ####### # Fill Gen muon branches #######
            ########################################
            genparts = []
            if self.has_gen_info:
                try:
                    genparts = list(evt.recoGenParticles_genParticles__HLT.product()) # rawsim
                except:
                    pass
            nGenMuon = 0
            # nFiducialMuon = 0
            # nFiducialMuon_norho = 0
            genmuons = []
            for genpart in genparts:
                pdgid = genpart.pdgId()
                if abs(pdgid) not in [13,23,25,6000211,3000022,999999,1999999]: continue
                motheridx = genpart.motherRef().index()
                mother = genparts[motheridx]
                motherid = mother.pdgId()
                if abs(pdgid) in [23,25,6000211,3000022,999999,1999999]: # "exotic" except muons
                    branches["GenOther_pt"].push_back(genpart.pt())
                    branches["GenOther_eta"].push_back(genpart.eta())
                    branches["GenOther_phi"].push_back(genpart.phi())
                    branches["GenOther_m"].push_back(genpart.mass())
                    branches["GenOther_vx"].push_back(genpart.vx())
                    branches["GenOther_vy"].push_back(genpart.vy())
                    branches["GenOther_vz"].push_back(genpart.vz())
                    branches["GenOther_lxy"].push_back(math.hypot(genpart.vx(), genpart.vy()))
                    branches["GenOther_status"].push_back(genpart.status())
                    branches["GenOther_pdgId"].push_back(pdgid)
                    branches["GenOther_motherId"].push_back(motherid)
                # For the useful muons, store them in GenMuon branches to avoid reading a lot of extra junk
                if (motherid in [23, 6000211, 999999, 1999999, 3000022]) and (abs(pdgid)==13): 
                    branches["GenMuon_pt"].push_back(genpart.pt())
                    branches["GenMuon_eta"].push_back(genpart.eta())
                    branches["GenMuon_phi"].push_back(genpart.phi())
                    branches["GenMuon_m"].push_back(genpart.mass())
                    branches["GenMuon_vx"].push_back(genpart.vx())
                    branches["GenMuon_vy"].push_back(genpart.vy())
                    branches["GenMuon_vz"].push_back(genpart.vz())
                    branches["GenMuon_lxy"].push_back(math.hypot(genpart.vx(), genpart.vy()))
                    branches["GenMuon_status"].push_back(genpart.status())
                    branches["GenMuon_pdgId"].push_back(pdgid)
                    branches["GenMuon_motherId"].push_back(motherid)
                    # if (genpart.pt() > 3.) and (abs(genpart.eta()) < 2.4):
                    #     nFiducialMuon_norho += 1
                    #     if (math.hypot(genpart.vx(),genpart.vy())<11.):
                    #         nFiducialMuon += 1
                    nGenMuon += 1
                    genmuons.append(genpart)
            branches["nGenMuon"][0] = nGenMuon
            # branches["pass_fiducialgen"][0] = (nFiducialMuon >= 2) or (not self.is_mc)
            # branches["pass_fiducialgen_norho"][0] = (nFiducialMuon_norho >= 2) or (not self.is_mc)


            ########################################
            ######### # Fill Muon branches #########
            ########################################
            branches["nMuon_raw"][0] = len(muons)
            branches["nMuon"][0] = len(selected_muons)
            for imuon,(pfx,muon) in enumerate(zip(["Muon1_", "Muon2_", "sublead_Muon1_", "sublead_Muon2_"], selected_muons)):
                pt = muon.pt()
                eta = muon.eta()
                phi = muon.phi()
                branches[pfx+"pt"][0] = pt
                branches[pfx+"eta"][0] = eta
                branches[pfx+"phi"][0] = phi
                branches[pfx+"m"][0] = MUON_MASS
                branches[pfx+"trackIso"][0] = muon.trackIso()
                branches[pfx+"chi2"][0] = muon.chi2()
                branches[pfx+"ndof"][0] = muon.ndof()
                branches[pfx+"charge"][0] = muon.charge()
                branches[pfx+"dxy"][0] = muon.dxy()
                branches[pfx+"dz"][0] = muon.dz()
                branches[pfx+"nValidMuonHits"][0] = muon.nValidMuonHits()
                branches[pfx+"nValidPixelHits"][0] = muon.nValidPixelHits()
                branches[pfx+"nMatchedStations"][0] = muon.nMatchedStations()
                branches[pfx+"nTrackerLayersWithMeasurement"][0] = muon.nTrackerLayersWithMeasurement()
                branches[pfx+"nValidStripHits"][0] = muon.nValidStripHits()
                branches[pfx+"dxyError"][0] = muon.dxyError()
                branches[pfx+"dzError"][0] = muon.dzError()
                branches[pfx+"nExpectedPixelHits"][0] = muon.nExpectedPixelHits

                branches[pfx+"trk_qoverp"][0] = muon.trk_qoverp()
                branches[pfx+"trk_lambda"][0] = muon.trk_lambda()
                branches[pfx+"trk_qoverpError"][0] = muon.trk_qoverpError()
                branches[pfx+"trk_lambdaError"][0] = muon.trk_lambdaError()
                branches[pfx+"trk_phiError"][0] = muon.trk_phiError()
                branches[pfx+"trk_dsz"][0] = muon.trk_dsz()
                branches[pfx+"trk_dszError"][0] = muon.trk_dszError()

                # find index of jet that is closest to this muon. `sorted_etaphis`: [(index, (eta,phi)), ...]
                jetIdx1, drjet = -1, 999.
                sorted_etaphis = sorted(enumerate(jet_etaphis), key=lambda x: math.hypot(eta-x[1][0], delta_phi(phi,x[1][1])))
                if len(sorted_etaphis) > 0: jetIdx1 = sorted_etaphis[0][0]
                if jetIdx1 >= 0:
                    jeteta, jetphi = sorted_etaphis[0][1]
                    drjet = math.hypot(eta-jeteta,delta_phi(phi,jetphi))
                branches[pfx+"drjet"][0] = drjet


                # get appropriate DV for this muon (order matches selected_dvs, with 2 muons per dv)
                dv = selected_dvs[1 if imuon >= 2 else 0]
                # https://github.com/cms-sw/cmssw/blob/master/DataFormats/TrackReco/interface/TrackBase.h#L24
                muon.dxyCorr = -(dv.x()-pvmx)*math.sin(muon.phi()) + (dv.y()-pvmy)*math.cos(muon.phi())
                branches[pfx+"dxyCorr"][0] = muon.dxyCorr

                refx, refy, refz = get_track_reference_point(muon)
                branches[pfx+"trk_refx"][0] = refx
                branches[pfx+"trk_refy"][0] = refy
                branches[pfx+"trk_refz"][0] = refz

                muon.phi_corr = self.get_corrected_phi(muon,dv)
                branches[pfx+"phiCorr"][0] = muon.phi_corr


                # nMatchedStations hardcoded to 0 in 2017 HLT code:
                # https://github.com/cms-sw/cmssw/blob/CMSSW_9_2_10/HLTrigger/Muon/src/HLTScoutingMuonProducer.cc#L161
                # and nValidMuonHits also 0, so we scrap this cut, even for 2018, for simplicity
                muon.passid = (
                        (muon.chi2()/muon.ndof() < 3.0) and
                        # (muon.nValidMuonHits() > 0) and
                        (muon.nTrackerLayersWithMeasurement() > 5)
                        # (muon.dxyError() < 0.01)
                        )
                muon.passiso = ((muon.trackIso() < 0.1) and  (drjet > 0.3))
                branches[pfx+"passid"][0] = muon.passid
                branches[pfx+"passiso"][0] = muon.passiso

                # Find closest GenMuon by DeltaR and also embed the info into the muon branches for convenience
                matched_genmu = None
                calc_dr = lambda x: math.hypot(eta-x.eta(), delta_phi(phi,x.phi()))
                sorted_genmuons = sorted(genmuons, key=calc_dr)
                if len(sorted_genmuons) > 0:
                    matched_genmu = sorted_genmuons[0]
                muon.genMatch_dr = 999.
                if matched_genmu is not None:
                    muon.genMatch_dr = calc_dr(matched_genmu)
                    branches[pfx+"genMatch_dr"][0] = muon.genMatch_dr
                    branches[pfx+"genMatch_pt"][0] = matched_genmu.pt()
                    branches[pfx+"genMatch_eta"][0] = matched_genmu.eta()
                    branches[pfx+"genMatch_phi"][0] = matched_genmu.phi()
                    branches[pfx+"genMatch_m"][0] = matched_genmu.mass()
                    branches[pfx+"genMatch_vx"][0] = matched_genmu.vx()
                    branches[pfx+"genMatch_vy"][0] = matched_genmu.vy()
                    branches[pfx+"genMatch_vz"][0] = matched_genmu.vz()
                    branches[pfx+"genMatch_lxy"][0] = math.hypot(matched_genmu.vx(), matched_genmu.vy())
                    branches[pfx+"genMatch_status"][0] = matched_genmu.status()
                    branches[pfx+"genMatch_pdgId"][0] = matched_genmu.pdgId()
                    # find non-muon mother recursively 
                    mother = matched_genmu
                    found = False
                    for _ in range(10):
                        motheridx = mother.motherRef().index()
                        mother = genparts[motheridx]
                        motherid = mother.pdgId()
                        # stop recursing if we found a non-muon
                        if abs(motherid) != 13:
                            found = True
                            break
                    if found:
                        branches[pfx+"genMatch_motherId"][0] = mother.pdgId()
                        branches[pfx+"genMatch_mothervx"][0] = mother.vx()
                        branches[pfx+"genMatch_mothervy"][0] = mother.vy()
                        branches[pfx+"genMatch_mothervz"][0] = mother.vz()
                        # https://arxiv.org/pdf/1710.08949.pdf eq 1
                        ct = (matched_genmu.vx()*mother.px()+matched_genmu.vy()*mother.py())*mother.mass()/mother.pt()**2.
                        branches[pfx+"genMatch_motherct"][0] = ct
                    else:
                        branches[pfx+"genMatch_motherId"][0] = 0
                        branches[pfx+"genMatch_mothervx"][0] = 0.
                        branches[pfx+"genMatch_mothervy"][0] = 0.
                        branches[pfx+"genMatch_mothervz"][0] = 0.
                        branches[pfx+"genMatch_motherct"][0] = 0.


            ########################################
            ### # Fill more event level branches ###
            ########################################

            if len(selected_dvs) >= 1 and len(selected_muons) >= 2:
                mu1 = selected_muons[0]
                mu2 = selected_muons[1]
                dv = selected_dvs[0]
                mu1p4 = r.TLorentzVector()
                mu2p4 = r.TLorentzVector()
                mu1p4.SetPtEtaPhiM(mu1.pt(), mu1.eta(), mu1.phi(), MUON_MASS)
                mu2p4.SetPtEtaPhiM(mu2.pt(), mu2.eta(), mu2.phi(), MUON_MASS)
                dimuon = (mu1p4+mu2p4)
                vecdv2d = r.TVector2(dv.x()-pvmx, dv.y()-pvmy)
                vecdimuon2d = r.TVector2(dimuon.Px(),dimuon.Py())
                cosphi = (vecdv2d.Px()*vecdimuon2d.Px() + vecdv2d.Py()*vecdimuon2d.Py()) / (vecdv2d.Mod()*vecdimuon2d.Mod())
                # definition on s2 of https://indico.cern.ch/event/846681/contributions/3557724/attachments/1907377/3150380/Displaced_Scouting_Status_Update.pdf
                # rutgers lxy is lowercase lxy from that set of slides, which does not have the cosine term
                lxy = vecdv2d.Mod()
                logabsetaphi = math.log10(max(abs(mu1p4.Eta()-mu2p4.Eta()),1e-6)/max(abs(mu1p4.DeltaPhi(mu2p4)),1e-6))

                # corrected p4s with phi calculated at DV instead of track reference point.
                mu1p4_corr = r.TLorentzVector()
                mu2p4_corr = r.TLorentzVector()
                mu1p4_corr.SetPtEtaPhiM(mu1.pt(), mu1.eta(), mu1.phi_corr, MUON_MASS)
                mu2p4_corr.SetPtEtaPhiM(mu2.pt(), mu2.eta(), mu2.phi_corr, MUON_MASS)
                dimuon_corr = (mu1p4_corr+mu2p4_corr)

                ctau = lxy*cosphi*dimuon_corr.M()/dimuon_corr.Pt()

                absdphimumu = abs(mu1p4.DeltaPhi(mu2p4))
                absdphimudv = abs(vecdimuon2d.DeltaPhi(vecdv2d))
                dimuon_isos = mu1.charge()*mu2.charge() < 0
                branches["dimuon_isos"][0] = dimuon_isos
                branches["dimuon_pt"][0] = dimuon_corr.Pt()
                branches["dimuon_eta"][0] = dimuon_corr.Eta()
                branches["dimuon_phi"][0] = dimuon_corr.Phi()
                branches["dimuon_mass"][0] = dimuon_corr.M()
                branches["mass"][0] = dimuon_corr.M()
                branches["dimuon_massRaw"][0] = dimuon.M()
                branches["absdphimumu"][0] = absdphimumu
                branches["absdphimudv"][0] = absdphimudv
                branches["minabsdxy"][0] = min(abs(mu1.dxyCorr),abs(mu2.dxyCorr))
                branches["ctau"][0] = ctau
                branches["logabsetaphi"][0] = logabsetaphi
                branches["cosphi"][0] = cosphi
                branches["lxy"][0] = lxy
                branches["lxyError"][0] = ((dv.xError()*(dv.x()-pvmx))**2 + (dv.yError()*(dv.y()-pvmy))**2)**0.5 / lxy

                # both muons need to have no excess hits if the displacement is >3.5cm (otherwise we're within the 1st bpix layer and extra hits don't make sense to calculate)
                pass_excesshits = (
                        (lxy < 3.5) or 
                            ( (mu1.nValidPixelHits() - mu1.nExpectedPixelHits <= 0) and
                              (mu2.nValidPixelHits() - mu2.nExpectedPixelHits <= 0)
                            )
                        )
                branches["pass_excesshits"][0] = pass_excesshits

                branches["pass_genmatch"][0] = ((mu1.genMatch_dr < 0.1) and (mu2.genMatch_dr < 0.1)) or (not self.is_mc)

                pass_materialveto = (dv.distPixel > 0.05)
                branches["pass_materialveto"][0] = pass_materialveto

                pass_dxyscaled = (
                        (abs(mu1.dxyCorr/(lxy*dimuon_corr.M()/dimuon_corr.Pt())) > 0.1) and
                        (abs(mu2.dxyCorr/(lxy*dimuon_corr.M()/dimuon_corr.Pt())) > 0.1)
                        )
                branches["pass_dxyscaled"][0] = pass_dxyscaled
                pass_dxysig = (
                        (abs(mu1.dxyCorr/mu1.dxyError()) > 2) and
                        (abs(mu2.dxyCorr/mu2.dxyError()) > 2)
                        )
                branches["pass_dxysig"][0] = pass_dxysig

                # Baseline selection
                pass_baseline = (
                        True
                        and mu1.passid
                        and mu2.passid
                        # and (cosphi > 0) # redundant with absdphimudv cut
                        and (absdphimumu < 2.8)
                        and (absdphimudv < 0.02)
                        and dimuon_isos
                        and pass_l1
                        and (lxy < 11.)
                        )
                pass_baseline_iso = pass_baseline and (mu1.passiso and mu2.passiso)
                pass_baseline_isohalf = pass_baseline and (mu1.passiso ^ mu2.passiso)

                branches["pass_baseline"][0] = pass_baseline
                branches["pass_baseline_iso"][0] = pass_baseline_iso
                branches["pass_baseline_isohalf"][0] = pass_baseline_isohalf

                # baseline+"extra" is baseline with pixel requirements and logabsetaphi<1.25
                pass_extra = pass_excesshits and pass_materialveto and (logabsetaphi < 1.25)
                branches["pass_baseline_extra"][0] = pass_baseline and pass_extra
                branches["pass_baseline_extra_iso"][0] = pass_baseline_iso and pass_extra
                branches["pass_baseline_extra_isohalf"][0] = pass_baseline_isohalf and pass_extra

            # Fill some branches for subleading DV and associated muons, if they exist
            if len(selected_dvs) >= 2 and len(selected_muons) >= 4:
                mu1 = selected_muons[2]
                mu2 = selected_muons[3]
                dv = selected_dvs[1]
                mu1p4 = r.TLorentzVector()
                mu2p4 = r.TLorentzVector()
                mu1p4.SetPtEtaPhiM(mu1.pt(), mu1.eta(), mu1.phi(), MUON_MASS)
                mu2p4.SetPtEtaPhiM(mu2.pt(), mu2.eta(), mu2.phi(), MUON_MASS)
                dimuon = (mu1p4+mu2p4)
                vecdv2d = r.TVector2(dv.x()-pvmx, dv.y()-pvmy)
                lxy = vecdv2d.Mod()
                dimuon_isos = mu1.charge()*mu2.charge() < 0
                branches["sublead_dimuon_isos"][0] = dimuon_isos
                branches["sublead_dimuon_mass"][0] = dimuon.M()
                branches["sublead_logabsetaphi"][0] = math.log10(max(abs(mu1p4.Eta()-mu2p4.Eta()),1e-6)/max(abs(mu1p4.DeltaPhi(mu2p4)),1e-6))
                branches["sublead_lxy"][0] = lxy

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

        # # Embed babymaking code inside root file
        # metadata = dict(
        #         # babymaker code
        #         babymaker = open(__file__,"r").read(),
        #         # number of events in the input chain
        #         nevents_input = nevents_in,
        #         # number of events we actually looped over
        #         nevents_processed = ievt,
        #         # number of events in the output tree
        #         nevents_output = self.outtree.GetEntries(),
        #         input_fnames = ",".join(self.fnames),
        #         )
        # write_metadata_to_tfile(self.outfile, metadata)

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
    parser.add_argument("-y", "--year", help="year (2017 or 2018)", default=2018, type=int)
    args = parser.parse_args()

    looper = Looper(
            fnames=args.fnames,
            output=args.output,
            nevents=args.nevents,
            expected=args.expected,
            year=args.year,
    )
    looper.run()
