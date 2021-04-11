
import time
import ROOT as r
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('inputs', help='comma-separated file names')
parser.add_argument("-o", '--output', default="output.root", help="output file name")
args = parser.parse_args()
inputs = args.inputs.split(",")
output = args.output

# inputs = ["/hadoop/cms/store/user/namin/ProjectMetis/ScoutingCaloMuon_Run2018skim_2018D_v13_RAW_dvnm1v2/output_565.root"]
# output = "output.root"

ch = r.TChain("Events")
for fname in inputs:
    fname = fname.strip()
    ch.Add(fname)

ch.SetBranchStatus("*", 0)
branches = [
        "DV_rhoCorr", 
        "DV_x","DV_y","DV_z", 
        "DV_xError", "DV_yError", "DV_zError", 
        "DV_chi2", 
        "DV_ndof", 
        "DV_passid", 
        "pass_materialveto", "pass_l1", 
        "nDV_raw", "nMuon_raw","run"]
for b in branches:
    ch.SetBranchStatus(b, 1)

first_run = 297046
last_run = 325175
run_bins = int((last_run-first_run)/5)

fout = r.TFile(output, "recreate")
hists = dict()

hists["lxy_inc"] = r.TH1D("lxy_inc", "", 1000, 0, 50)
hists["xError_inc"] = r.TH1D("xError_inc", "", 500, 0, 0.5)
hists["yError_inc"] = r.TH1D("yError_inc", "", 500, 0, 0.5)
hists["zError_inc"] = r.TH1D("zError_inc", "", 500, 0, 0.5)
hists["chi2ndof_inc"] = r.TH1D("chi2ndof_inc", "", 500, 0, 10)

hists["lxy_nm1"] = r.TH1D("lxy_nm1", "", 1000, 0, 50)
hists["xError_nm1"] = r.TH1D("xError_nm1", "", 500, 0, 0.5)
hists["yError_nm1"] = r.TH1D("yError_nm1", "", 500, 0, 0.5)
hists["zError_nm1"] = r.TH1D("zError_nm1", "", 500, 0, 0.5)
hists["chi2ndof_nm1"] = r.TH1D("chi2ndof_nm1", "", 500, 0, 10)

hists["nDV_vs_run"] = r.TH2D("nDV_vs_run", "", run_bins, first_run, last_run, 20, 0, 20)
hists["nMuon_vs_run"] = r.TH2D("nMuon_vs_run", "", run_bins, first_run, last_run, 20, 0, 20)

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

    if not bool(evt.pass_l1): continue

    hists["lxy_inc"].Fill(evt.DV_rhoCorr)
    hists["xError_inc"].Fill(evt.DV_xError)
    hists["yError_inc"].Fill(evt.DV_yError)
    hists["zError_inc"].Fill(evt.DV_zError)
    hists["chi2ndof_inc"].Fill(evt.DV_chi2/evt.DV_ndof)
    if evt.DV_xError<0.05 and evt.DV_yError<0.05 and evt.DV_zError<0.10 and evt.DV_chi2/evt.DV_ndof<5                       : hists["lxy_nm1"].Fill(evt.DV_rhoCorr)
    if                        evt.DV_yError<0.05 and evt.DV_zError<0.10 and evt.DV_chi2/evt.DV_ndof<5 and evt.DV_rhoCorr<11.: hists["xError_nm1"].Fill(evt.DV_xError)
    if evt.DV_xError<0.05 and                        evt.DV_zError<0.10 and evt.DV_chi2/evt.DV_ndof<5 and evt.DV_rhoCorr<11.: hists["yError_nm1"].Fill(evt.DV_yError)
    if evt.DV_xError<0.05 and evt.DV_yError<0.05                        and evt.DV_chi2/evt.DV_ndof<5 and evt.DV_rhoCorr<11.: hists["zError_nm1"].Fill(evt.DV_zError)
    if evt.DV_xError<0.05 and evt.DV_yError<0.05 and evt.DV_zError<0.10                               and evt.DV_rhoCorr<11.: hists["chi2ndof_nm1"].Fill(evt.DV_chi2/evt.DV_ndof)

    try:
        hists["nDV_vs_run"].Fill(evt.run, evt.nDV_raw)
        hists["nMuon_vs_run"].Fill(evt.run, evt.nMuon_raw)
    except:
        pass

    if bool(evt.DV_passid):
        rho = (evt.DV_x**2+evt.DV_y**2)**0.5
        hists["DV_rho_vs_z_tot"].Fill(evt.DV_z, rho)
        hists["DV_y_vs_x_tot"].Fill(evt.DV_x, evt.DV_y)
        hists["DV_rho_tot"].Fill(evt.DV_rhoCorr)
        if bool(evt.pass_materialveto):
            hists["DV_rho_vs_z_matveto"].Fill(evt.DV_z, rho)
            hists["DV_y_vs_x_matveto"].Fill(evt.DV_x, evt.DV_y)
            hists["DV_rho_matveto"].Fill(evt.DV_rhoCorr)

for h in hists.values():
    h.Write()
fout.Close()
