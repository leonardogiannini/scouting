
import time
import ROOT as r
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('inputs', help='comma-separated file names')
parser.add_argument("-o", '--output', default="output.root", help="output file name")
args = parser.parse_args()
inputs = args.inputs.split(",")
output = args.output

# inputs = ["/hadoop/cms/store/user/namin/ProjectMetis/ScoutingCaloMuon_Run2018skim_2018C_v13_RAW_v25/output_119.root"]
# output = "output.root"
# inputs = ["/hadoop/cms/store/user/namin/ProjectMetis/ScoutingCaloMuon_Run2017skim_2017C_v13_RAW_v25/output_42.root"]
# output = "output.root"

ch = r.TChain("Events")
for fname in inputs:
    fname = fname.strip()
    ch.Add(fname)

ch.SetBranchStatus("*", 0)
branches = [
        "pass_*",
        "lxy",
        "dimuon_pt",
        "Muon1_dxyCorr",
        "Muon2_dxyCorr",
        "Muon1_dxyError",
        "Muon2_dxyError",
        "Muon2_pt",
        "Muon2_eta",
        "Muon2_trackIso",
        "Muon2_drjet",
        "Muon2_chi2",
        "Muon2_ndof",
        "DV_passid",
        "DV_x",
        "DV_y",
        "DV_z",
        "DV_rhoCorr",
        "logabsetaphi",
        "absdphimudv",
        "Muon2_nValidPixelHits",
        "Muon2_nExpectedPixelHits",
        "absdphimumu",
        "Muon2_nTrackerLayersWithMeasurement",
        "Muon*_passid",
        "Muon*_passiso",
        "pass_materialveto",
        "mass",
        "pass_excesshits",
        "pass_dxysig",
        "pass_dxyscaled",
        ]
for b in branches:
    ch.SetBranchStatus(b, 1)

fout = r.TFile(output, "recreate")
hists = dict()

for smass in ["allmass","lowmass","highmass"]:
    hists["dimupt_full_lxy0to11_{}".format(smass)] = r.TH1D("dimupt_full_lxy0to11_{}".format(smass), "", 100, 0, 100)

    hists["mu2pt_trig_lxy0to0p2_{}".format(smass)] = r.TH1D("mu2pt_trig_lxy0to0p2_{}".format(smass), "", 20, 0, 20)
    hists["mu2pt_trig_lxy0p2to1_{}".format(smass)] = r.TH1D("mu2pt_trig_lxy0p2to1_{}".format(smass), "", 20, 0, 20)
    hists["mu2pt_trig_lxy1to11_{}".format(smass)] = r.TH1D("mu2pt_trig_lxy1to11_{}".format(smass), "", 20, 0, 20)

    hists["mu2eta_trig_lxy0to0p2_{}".format(smass)] = r.TH1D("mu2eta_trig_lxy0to0p2_{}".format(smass), "", 24, -3, 3)
    hists["mu2eta_trig_lxy0p2to1_{}".format(smass)] = r.TH1D("mu2eta_trig_lxy0p2to1_{}".format(smass), "", 24, -3, 3)
    hists["mu2eta_trig_lxy1to11_{}".format(smass)] = r.TH1D("mu2eta_trig_lxy1to11_{}".format(smass), "", 24, -3, 3)

    hists["mu2chi2ndof_trig_lxy0to11_{}".format(smass)] = r.TH1D("mu2chi2ndof_trig_lxy0to11_{}".format(smass), "", 50, 0, 5)
    hists["mu2trkmeas_trig_lxy0to11_{}".format(smass)] = r.TH1D("mu2trkmeas_trig_lxy0to11_{}".format(smass), "", 20, 0, 20)

    hists["absdphimudv_passid_lxy0to0p2_{}".format(smass)] = r.TH1D("absdphimudv_passid_lxy0to0p2_{}".format(smass), "", 20, 0, 0.2)
    hists["absdphimudv_passid_lxy0p2to1_{}".format(smass)] = r.TH1D("absdphimudv_passid_lxy0p2to1_{}".format(smass), "", 20, 0, 0.2)
    hists["absdphimudv_passid_lxy1to11_{}".format(smass)] = r.TH1D("absdphimudv_passid_lxy1to11_{}".format(smass), "", 20, 0, 0.2)
    hists["absdphimumu_passid_lxy0to0p2_{}".format(smass)] = r.TH1D("absdphimumu_passid_lxy0to0p2_{}".format(smass), "", 35, 0, 3.5)
    hists["absdphimumu_passid_lxy0p2to1_{}".format(smass)] = r.TH1D("absdphimumu_passid_lxy0p2to1_{}".format(smass), "", 35, 0, 3.5)
    hists["absdphimumu_passid_lxy1to11_{}".format(smass)] = r.TH1D("absdphimumu_passid_lxy1to11_{}".format(smass), "", 35, 0, 3.5)

    hists["mu2trackiso_passkin_lxy0to0p2_{}".format(smass)] = r.TH1D("mu2trackiso_passkin_lxy0to0p2_{}".format(smass), "", 20, 0, 1)
    hists["mu2trackiso_passkin_lxy0p2to1_{}".format(smass)] = r.TH1D("mu2trackiso_passkin_lxy0p2to1_{}".format(smass), "", 20, 0, 1)
    hists["mu2trackiso_passkin_lxy1to11_{}".format(smass)] = r.TH1D("mu2trackiso_passkin_lxy1to11_{}".format(smass), "", 20, 0, 1)
    hists["mu2drjet_passkin_lxy0to0p2_{}".format(smass)] = r.TH1D("mu2drjet_passkin_lxy0to0p2_{}".format(smass), "", 50, 0, 5)
    hists["mu2drjet_passkin_lxy0p2to1_{}".format(smass)] = r.TH1D("mu2drjet_passkin_lxy0p2to1_{}".format(smass), "", 50, 0, 5)
    hists["mu2drjet_passkin_lxy1to11_{}".format(smass)] = r.TH1D("mu2drjet_passkin_lxy1to11_{}".format(smass), "", 50, 0, 5)

    hists["mu2excesshits_baseline_lxy0to0p2_{}".format(smass)] = r.TH1D("mu2excesshits_baseline_lxy0to0p2_{}".format(smass), "", 50, -5, 5)
    hists["mu2excesshits_baseline_lxy1to11_{}".format(smass)] = r.TH1D("mu2excesshits_baseline_lxy1to11_{}".format(smass), "", 50, -5, 5)
    hists["mu2excesshits_baseline_lxy3p1to7_{}".format(smass)] = r.TH1D("mu2excesshits_baseline_lxy3p1to7_{}".format(smass), "", 50, -5, 5)

    hists["logabsetaphi_baseline_lxy0to0p2_{}".format(smass)] = r.TH1D("logabsetaphi_baseline_lxy0to0p2_{}".format(smass), "", 20, -5, 5)
    hists["logabsetaphi_baseline_lxy0p2to1_{}".format(smass)] = r.TH1D("logabsetaphi_baseline_lxy0p2to1_{}".format(smass), "", 20, -5, 5)
    hists["logabsetaphi_baseline_lxy1to11_{}".format(smass)] = r.TH1D("logabsetaphi_baseline_lxy1to11_{}".format(smass), "", 20, -5, 5)

    hists["mindxy_extraiso_lxy0to0p2_{}".format(smass)] = r.TH1D("mindxy_extraiso_lxy0to0p2_{}".format(smass), "", 50, 0, 1)
    hists["mindxy_extraiso_lxy0p2to1_{}".format(smass)] = r.TH1D("mindxy_extraiso_lxy0p2to1_{}".format(smass), "", 50, 0, 1)
    hists["mindxy_extraiso_lxy1to11_{}".format(smass)] = r.TH1D("mindxy_extraiso_lxy1to11_{}".format(smass), "", 50, 0, 1)

    hists["mindxysig_extraiso_lxy0to0p2_{}".format(smass)] = r.TH1D("mindxysig_extraiso_lxy0to0p2_{}".format(smass), "", 40, 0, 20)
    hists["mindxysig_extraiso_lxy0p2to1_{}".format(smass)] = r.TH1D("mindxysig_extraiso_lxy0p2to1_{}".format(smass), "", 40, 0, 20)
    hists["mindxysig_extraiso_lxy1to11_{}".format(smass)] = r.TH1D("mindxysig_extraiso_lxy1to11_{}".format(smass), "", 40, 0, 20)

    hists["mindxyscaled_extraiso_lxy0to0p2_{}".format(smass)] = r.TH1D("mindxyscaled_extraiso_lxy0to0p2_{}".format(smass), "", 40, 0, 1)
    hists["mindxyscaled_extraiso_lxy0p2to1_{}".format(smass)] = r.TH1D("mindxyscaled_extraiso_lxy0p2to1_{}".format(smass), "", 40, 0, 1)
    hists["mindxyscaled_extraiso_lxy1to11_{}".format(smass)] = r.TH1D("mindxyscaled_extraiso_lxy1to11_{}".format(smass), "", 40, 0, 1)


    # INCLUSIVE

    hists["mu2pt_incl_lxy0to0p2_{}".format(smass)] = r.TH1D("mu2pt_incl_lxy0to0p2_{}".format(smass), "", 20, 0, 20)
    hists["mu2pt_incl_lxy0p2to1_{}".format(smass)] = r.TH1D("mu2pt_incl_lxy0p2to1_{}".format(smass), "", 20, 0, 20)
    hists["mu2pt_incl_lxy1to11_{}".format(smass)] = r.TH1D("mu2pt_incl_lxy1to11_{}".format(smass), "", 20, 0, 20)

    hists["mu2eta_incl_lxy0to0p2_{}".format(smass)] = r.TH1D("mu2eta_incl_lxy0to0p2_{}".format(smass), "", 24, -3, 3)
    hists["mu2eta_incl_lxy0p2to1_{}".format(smass)] = r.TH1D("mu2eta_incl_lxy0p2to1_{}".format(smass), "", 24, -3, 3)
    hists["mu2eta_incl_lxy1to11_{}".format(smass)] = r.TH1D("mu2eta_incl_lxy1to11_{}".format(smass), "", 24, -3, 3)

    hists["mu2chi2ndof_incl_lxy0to11_{}".format(smass)] = r.TH1D("mu2chi2ndof_incl_lxy0to11_{}".format(smass), "", 50, 0, 5)
    hists["mu2trkmeas_incl_lxy0to11_{}".format(smass)] = r.TH1D("mu2trkmeas_incl_lxy0to11_{}".format(smass), "", 20, 0, 20)

    hists["absdphimudv_incl_lxy0to0p2_{}".format(smass)] = r.TH1D("absdphimudv_incl_lxy0to0p2_{}".format(smass), "", 20, 0, 0.2)
    hists["absdphimudv_incl_lxy0p2to1_{}".format(smass)] = r.TH1D("absdphimudv_incl_lxy0p2to1_{}".format(smass), "", 20, 0, 0.2)
    hists["absdphimudv_incl_lxy1to11_{}".format(smass)] = r.TH1D("absdphimudv_incl_lxy1to11_{}".format(smass), "", 20, 0, 0.2)
    hists["absdphimumu_incl_lxy0to0p2_{}".format(smass)] = r.TH1D("absdphimumu_incl_lxy0to0p2_{}".format(smass), "", 35, 0, 3.5)
    hists["absdphimumu_incl_lxy0p2to1_{}".format(smass)] = r.TH1D("absdphimumu_incl_lxy0p2to1_{}".format(smass), "", 35, 0, 3.5)
    hists["absdphimumu_incl_lxy1to11_{}".format(smass)] = r.TH1D("absdphimumu_incl_lxy1to11_{}".format(smass), "", 35, 0, 3.5)

    hists["mu2trackiso_incl_lxy0to0p2_{}".format(smass)] = r.TH1D("mu2trackiso_incl_lxy0to0p2_{}".format(smass), "", 20, 0, 1)
    hists["mu2trackiso_incl_lxy0p2to1_{}".format(smass)] = r.TH1D("mu2trackiso_incl_lxy0p2to1_{}".format(smass), "", 20, 0, 1)
    hists["mu2trackiso_incl_lxy1to11_{}".format(smass)] = r.TH1D("mu2trackiso_incl_lxy1to11_{}".format(smass), "", 20, 0, 1)
    hists["mu2drjet_incl_lxy0to0p2_{}".format(smass)] = r.TH1D("mu2drjet_incl_lxy0to0p2_{}".format(smass), "", 50, 0, 5)
    hists["mu2drjet_incl_lxy0p2to1_{}".format(smass)] = r.TH1D("mu2drjet_incl_lxy0p2to1_{}".format(smass), "", 50, 0, 5)
    hists["mu2drjet_incl_lxy1to11_{}".format(smass)] = r.TH1D("mu2drjet_incl_lxy1to11_{}".format(smass), "", 50, 0, 5)

    hists["mu2excesshits_incl_lxy0to0p2_{}".format(smass)] = r.TH1D("mu2excesshits_incl_lxy0to0p2_{}".format(smass), "", 50, -5, 5)
    hists["mu2excesshits_incl_lxy1to11_{}".format(smass)] = r.TH1D("mu2excesshits_incl_lxy1to11_{}".format(smass), "", 50, -5, 5)
    hists["mu2excesshits_incl_lxy3p1to7_{}".format(smass)] = r.TH1D("mu2excesshits_incl_lxy3p1to7_{}".format(smass), "", 50, -5, 5)

    hists["logabsetaphi_incl_lxy0to0p2_{}".format(smass)] = r.TH1D("logabsetaphi_incl_lxy0to0p2_{}".format(smass), "", 20, -5, 5)
    hists["logabsetaphi_incl_lxy0p2to1_{}".format(smass)] = r.TH1D("logabsetaphi_incl_lxy0p2to1_{}".format(smass), "", 20, -5, 5)
    hists["logabsetaphi_incl_lxy1to11_{}".format(smass)] = r.TH1D("logabsetaphi_incl_lxy1to11_{}".format(smass), "", 20, -5, 5)

    hists["mindxy_incl_lxy0to0p2_{}".format(smass)] = r.TH1D("mindxy_incl_lxy0to0p2_{}".format(smass), "", 50, 0, 1)
    hists["mindxy_incl_lxy0p2to1_{}".format(smass)] = r.TH1D("mindxy_incl_lxy0p2to1_{}".format(smass), "", 50, 0, 1)
    hists["mindxy_incl_lxy1to11_{}".format(smass)] = r.TH1D("mindxy_incl_lxy1to11_{}".format(smass), "", 50, 0, 1)

    hists["mindxysig_incl_lxy0to0p2_{}".format(smass)] = r.TH1D("mindxysig_incl_lxy0to0p2_{}".format(smass), "", 40, 0, 20)
    hists["mindxysig_incl_lxy0p2to1_{}".format(smass)] = r.TH1D("mindxysig_incl_lxy0p2to1_{}".format(smass), "", 40, 0, 20)
    hists["mindxysig_incl_lxy1to11_{}".format(smass)] = r.TH1D("mindxysig_incl_lxy1to11_{}".format(smass), "", 40, 0, 20)

    hists["mindxyscaled_incl_lxy0to0p2_{}".format(smass)] = r.TH1D("mindxyscaled_incl_lxy0to0p2_{}".format(smass), "", 40, 0, 1)
    hists["mindxyscaled_incl_lxy0p2to1_{}".format(smass)] = r.TH1D("mindxyscaled_incl_lxy0p2to1_{}".format(smass), "", 40, 0, 1)
    hists["mindxyscaled_incl_lxy1to11_{}".format(smass)] = r.TH1D("mindxyscaled_incl_lxy1to11_{}".format(smass), "", 40, 0, 1)

cutflow_hnames = [
    "cutflow_masslt5_lxylt1",
    "cutflow_masslt5_lxygt1",
    "cutflow_massgt5_lxylt1",
    "cutflow_massgt5_lxygt1",
    ]
for x in cutflow_hnames:
    hists[x] = r.TH1D(x, "", 100, -0.5, 99.5)

def fill_cutflow(val, mass, lxy):
    if mass < 5:
        if lxy < 1:
            hists["cutflow_masslt5_lxylt1"].Fill(val)
        else:
            hists["cutflow_masslt5_lxygt1"].Fill(val)
    else:
        if lxy < 1:
            hists["cutflow_massgt5_lxylt1"].Fill(val)
        else:
            hists["cutflow_massgt5_lxygt1"].Fill(val)


hists["DV_rho_vs_z_tot"] = r.TH2D("DV_rho_vs_z_tot", "", 150, -40, 40, 150, 0, 11)
hists["DV_y_vs_x_tot"] = r.TH2D("DV_y_vs_x_tot", "", 300, -11, 11, 300, -11, 11)
hists["DV_rho_tot"] = r.TH1D("DV_rho_tot", "", 110, 0, 11)
hists["DV_rho_vs_z_matveto"] = r.TH2D("DV_rho_vs_z_matveto", "", 150, -40, 40, 150, 0, 11)
hists["DV_y_vs_x_matveto"] = r.TH2D("DV_y_vs_x_matveto", "", 300, -11, 11, 300, -11, 11)
hists["DV_rho_matveto"] = r.TH1D("DV_rho_matveto", "", 110, 0, 11)

N = ch.GetEntries()
print("Looping on TChain with {} entries".format(N))
t0 = time.time()
for i, evt in enumerate(ch):

    if (i == N-1) or (i % 100000 == 0):
        print("Reached entry {} in {:.2f}s".format(i, time.time()-t0))

    # if (i == N-1) or (i % 10000 == 0):
    #     print("Reached entry {} in {:.2f}s".format(i, time.time()-t0))
    # if i > 100000: break

    if not bool(evt.pass_l1): continue

    mu2pt = evt.Muon2_pt
    mu2eta = evt.Muon2_eta
    mass = evt.mass
    lxy = evt.lxy

    incl = evt.DV_passid and evt.Muon1_passid and evt.Muon2_passid

    if bool(evt.DV_passid):
        rho = (evt.DV_x**2+evt.DV_y**2)**0.5
        hists["DV_rho_vs_z_tot"].Fill(evt.DV_z, rho)
        hists["DV_y_vs_x_tot"].Fill(evt.DV_x, evt.DV_y)
        hists["DV_rho_tot"].Fill(evt.DV_rhoCorr)
        if bool(evt.pass_materialveto):
            hists["DV_rho_vs_z_matveto"].Fill(evt.DV_z, rho)
            hists["DV_y_vs_x_matveto"].Fill(evt.DV_x, evt.DV_y)
            hists["DV_rho_matveto"].Fill(evt.DV_rhoCorr)

    if evt.pass_l1 and evt.DV_passid:
        fill_cutflow(0, mass, lxy)
        if bool(evt.Muon1_passid) and bool(evt.Muon2_passid):
            fill_cutflow(1, mass, lxy)
            if bool(evt.Muon1_passiso) and bool(evt.Muon2_passiso):
                fill_cutflow(2, mass, lxy)
                if (evt.absdphimumu < 2.8):
                    fill_cutflow(3, mass, lxy)
                    if (evt.absdphimudv < 0.02):
                        fill_cutflow(4, mass, lxy)
                        if evt.pass_excesshits:
                            fill_cutflow(5, mass, lxy)
                            if evt.pass_materialveto:
                                fill_cutflow(6, mass, lxy)
                                if (evt.logabsetaphi < 1.25):
                                    fill_cutflow(7, mass, lxy)
                                    if evt.pass_dxysig:
                                        fill_cutflow(8, mass, lxy)
                                        if evt.pass_dxyscaled:
                                            fill_cutflow(9, mass, lxy)

    for smass in ["allmass","lowmass","highmass"]:

        if (smass == "lowmass") and (mass > 5.): continue
        if (smass == "highmass") and (mass < 5.): continue

        if 0. < lxy < 0.2:
            hists["mu2pt_trig_lxy0to0p2_{}".format(smass)].Fill(mu2pt)
            hists["mu2eta_trig_lxy0to0p2_{}".format(smass)].Fill(mu2eta)
        elif 0.2 < lxy < 1.0:
            hists["mu2pt_trig_lxy0p2to1_{}".format(smass)].Fill(mu2pt)
            hists["mu2eta_trig_lxy0p2to1_{}".format(smass)].Fill(mu2eta)
        elif 1.0 < lxy < 11:
            hists["mu2pt_trig_lxy1to11_{}".format(smass)].Fill(mu2pt)
            hists["mu2eta_trig_lxy1to11_{}".format(smass)].Fill(mu2eta)

        if 0.0 < lxy < 11.:
            hists["mu2chi2ndof_trig_lxy0to11_{}".format(smass)].Fill(evt.Muon2_chi2/evt.Muon2_ndof)
            hists["mu2trkmeas_trig_lxy0to11_{}".format(smass)].Fill(evt.Muon2_nTrackerLayersWithMeasurement)

        if bool(evt.DV_passid) and bool(evt.Muon1_passid) and bool(evt.Muon2_passid):
            if 0. < lxy < 0.2:
                hists["absdphimudv_passid_lxy0to0p2_{}".format(smass)].Fill(evt.absdphimudv)
                hists["absdphimumu_passid_lxy0to0p2_{}".format(smass)].Fill(evt.absdphimumu)
            elif 0.2 < lxy < 1.0:
                hists["absdphimudv_passid_lxy0p2to1_{}".format(smass)].Fill(evt.absdphimudv)
                hists["absdphimumu_passid_lxy0p2to1_{}".format(smass)].Fill(evt.absdphimumu)
            elif 1.0 < lxy < 11:
                hists["absdphimudv_passid_lxy1to11_{}".format(smass)].Fill(evt.absdphimudv)
                hists["absdphimumu_passid_lxy1to11_{}".format(smass)].Fill(evt.absdphimumu)

        if bool(evt.DV_passid) and bool(evt.Muon1_passid) and bool(evt.Muon2_passid) and (evt.absdphimumu < 2.8) and (evt.absdphimudv < 0.02):
            if 0. < lxy < 0.2:
                hists["mu2trackiso_passkin_lxy0to0p2_{}".format(smass)].Fill(evt.Muon2_trackIso)
                hists["mu2drjet_passkin_lxy0to0p2_{}".format(smass)].Fill(evt.Muon2_drjet)
            elif 0.2 < lxy < 1.0:
                hists["mu2trackiso_passkin_lxy0p2to1_{}".format(smass)].Fill(evt.Muon2_trackIso)
                hists["mu2drjet_passkin_lxy0p2to1_{}".format(smass)].Fill(evt.Muon2_drjet)
            elif 1.0 < lxy < 11:
                hists["mu2trackiso_passkin_lxy1to11_{}".format(smass)].Fill(evt.Muon2_trackIso)
                hists["mu2drjet_passkin_lxy1to11_{}".format(smass)].Fill(evt.Muon2_drjet)

        if bool(evt.pass_baseline):
            mu2excesshits = evt.Muon2_nValidPixelHits-evt.Muon2_nExpectedPixelHits
            if 0. < lxy < 0.2:
                hists["mu2excesshits_baseline_lxy0to0p2_{}".format(smass)].Fill(mu2excesshits)
            if 1 < lxy < 11:
                hists["mu2excesshits_baseline_lxy1to11_{}".format(smass)].Fill(mu2excesshits)
            if 3.1 < lxy < 7:
                hists["mu2excesshits_baseline_lxy3p1to7_{}".format(smass)].Fill(mu2excesshits)

            if 0. < lxy < 0.2:
                hists["logabsetaphi_baseline_lxy0to0p2_{}".format(smass)].Fill(evt.logabsetaphi)
            elif 0.2 < lxy < 1:
                hists["logabsetaphi_baseline_lxy0p2to1_{}".format(smass)].Fill(evt.logabsetaphi)
            elif 1 < lxy < 11:
                hists["logabsetaphi_baseline_lxy1to11_{}".format(smass)].Fill(evt.logabsetaphi)

        if bool(evt.pass_baseline_extra_iso):
            mindxy = min(abs(evt.Muon1_dxyCorr), abs(evt.Muon2_dxyCorr))
            mindxysig = min(abs(evt.Muon1_dxyCorr/evt.Muon1_dxyError), abs(evt.Muon2_dxyCorr/evt.Muon2_dxyError))
            mindxyscaled = min(
                    abs(evt.Muon1_dxyCorr/(lxy*mass/evt.dimuon_pt)),
                    abs(evt.Muon2_dxyCorr/(lxy*mass/evt.dimuon_pt)),
                    )
            if 0. < lxy < 0.2:
                hists["mindxy_extraiso_lxy0to0p2_{}".format(smass)].Fill(mindxy)
                hists["mindxysig_extraiso_lxy0to0p2_{}".format(smass)].Fill(mindxysig)
                hists["mindxyscaled_extraiso_lxy0to0p2_{}".format(smass)].Fill(mindxyscaled)
            elif 0.2 < lxy < 1:
                hists["mindxy_extraiso_lxy0p2to1_{}".format(smass)].Fill(mindxy)
                hists["mindxysig_extraiso_lxy0p2to1_{}".format(smass)].Fill(mindxysig)
                hists["mindxyscaled_extraiso_lxy0p2to1_{}".format(smass)].Fill(mindxyscaled)
            elif 1 < lxy < 11:
                hists["mindxy_extraiso_lxy1to11_{}".format(smass)].Fill(mindxy)
                hists["mindxysig_extraiso_lxy1to11_{}".format(smass)].Fill(mindxysig)
                hists["mindxyscaled_extraiso_lxy1to11_{}".format(smass)].Fill(mindxyscaled)

        if bool(evt.pass_all):
            if 0.0 < lxy < 11.:
                hists["dimupt_full_lxy0to11_{}".format(smass)].Fill(evt.dimuon_pt)

        if incl:

            if 0. < lxy < 0.2:
                hists["mu2pt_incl_lxy0to0p2_{}".format(smass)].Fill(mu2pt)
                hists["mu2eta_incl_lxy0to0p2_{}".format(smass)].Fill(mu2eta)
            elif 0.2 < lxy < 1.0:
                hists["mu2pt_incl_lxy0p2to1_{}".format(smass)].Fill(mu2pt)
                hists["mu2eta_incl_lxy0p2to1_{}".format(smass)].Fill(mu2eta)
            elif 1.0 < lxy < 11:
                hists["mu2pt_incl_lxy1to11_{}".format(smass)].Fill(mu2pt)
                hists["mu2eta_incl_lxy1to11_{}".format(smass)].Fill(mu2eta)

            if 0.0 < lxy < 11.:
                hists["mu2chi2ndof_incl_lxy0to11_{}".format(smass)].Fill(evt.Muon2_chi2/evt.Muon2_ndof)
                hists["mu2trkmeas_incl_lxy0to11_{}".format(smass)].Fill(evt.Muon2_nTrackerLayersWithMeasurement)

            if 0. < lxy < 0.2:
                hists["absdphimudv_incl_lxy0to0p2_{}".format(smass)].Fill(evt.absdphimudv)
                hists["absdphimumu_incl_lxy0to0p2_{}".format(smass)].Fill(evt.absdphimumu)
            elif 0.2 < lxy < 1.0:
                hists["absdphimudv_incl_lxy0p2to1_{}".format(smass)].Fill(evt.absdphimudv)
                hists["absdphimumu_incl_lxy0p2to1_{}".format(smass)].Fill(evt.absdphimumu)
            elif 1.0 < lxy < 11:
                hists["absdphimudv_incl_lxy1to11_{}".format(smass)].Fill(evt.absdphimudv)
                hists["absdphimumu_incl_lxy1to11_{}".format(smass)].Fill(evt.absdphimumu)

            if 0. < lxy < 0.2:
                hists["mu2trackiso_incl_lxy0to0p2_{}".format(smass)].Fill(evt.Muon2_trackIso)
                hists["mu2drjet_incl_lxy0to0p2_{}".format(smass)].Fill(evt.Muon2_drjet)
            elif 0.2 < lxy < 1.0:
                hists["mu2trackiso_incl_lxy0p2to1_{}".format(smass)].Fill(evt.Muon2_trackIso)
                hists["mu2drjet_incl_lxy0p2to1_{}".format(smass)].Fill(evt.Muon2_drjet)
            elif 1.0 < lxy < 11:
                hists["mu2trackiso_incl_lxy1to11_{}".format(smass)].Fill(evt.Muon2_trackIso)
                hists["mu2drjet_incl_lxy1to11_{}".format(smass)].Fill(evt.Muon2_drjet)

            mu2excesshits = evt.Muon2_nValidPixelHits-evt.Muon2_nExpectedPixelHits
            if 0. < lxy < 0.2:
                hists["mu2excesshits_incl_lxy0to0p2_{}".format(smass)].Fill(mu2excesshits)
            if 1 < lxy < 11:
                hists["mu2excesshits_incl_lxy1to11_{}".format(smass)].Fill(mu2excesshits)
            if 3.1 < lxy < 7:
                hists["mu2excesshits_incl_lxy3p1to7_{}".format(smass)].Fill(mu2excesshits)

            if 0. < lxy < 0.2:
                hists["logabsetaphi_incl_lxy0to0p2_{}".format(smass)].Fill(evt.logabsetaphi)
            elif 0.2 < lxy < 1:
                hists["logabsetaphi_incl_lxy0p2to1_{}".format(smass)].Fill(evt.logabsetaphi)
            elif 1 < lxy < 11:
                hists["logabsetaphi_incl_lxy1to11_{}".format(smass)].Fill(evt.logabsetaphi)

            mindxy = min(abs(evt.Muon1_dxyCorr), abs(evt.Muon2_dxyCorr))
            mindxysig = min(abs(evt.Muon1_dxyCorr/evt.Muon1_dxyError), abs(evt.Muon2_dxyCorr/evt.Muon2_dxyError))
            mindxyscaled = min(
                    abs(evt.Muon1_dxyCorr/(lxy*mass/evt.dimuon_pt)),
                    abs(evt.Muon2_dxyCorr/(lxy*mass/evt.dimuon_pt)),
                    )
            if 0. < lxy < 0.2:
                hists["mindxy_incl_lxy0to0p2_{}".format(smass)].Fill(mindxy)
                hists["mindxysig_incl_lxy0to0p2_{}".format(smass)].Fill(mindxysig)
                hists["mindxyscaled_incl_lxy0to0p2_{}".format(smass)].Fill(mindxyscaled)
            elif 0.2 < lxy < 1:
                hists["mindxy_incl_lxy0p2to1_{}".format(smass)].Fill(mindxy)
                hists["mindxysig_incl_lxy0p2to1_{}".format(smass)].Fill(mindxysig)
                hists["mindxyscaled_incl_lxy0p2to1_{}".format(smass)].Fill(mindxyscaled)
            elif 1 < lxy < 11:
                hists["mindxy_incl_lxy1to11_{}".format(smass)].Fill(mindxy)
                hists["mindxysig_incl_lxy1to11_{}".format(smass)].Fill(mindxysig)
                hists["mindxyscaled_incl_lxy1to11_{}".format(smass)].Fill(mindxyscaled)

for h in hists.values():
    h.Write()
fout.Close()
