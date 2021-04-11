import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
from matplotlib.colors import LogNorm
from tqdm.auto import tqdm
import time
import re
import subprocess
import json
import requests
import uproot4

from yahist import Hist1D, Hist2D

def set_plotting_style():
    from matplotlib import rcParams
    rcParams["font.family"] = "sans-serif"
    rcParams["font.sans-serif"] = ["Helvetica", "Arial", "Liberation Sans", "Bitstream Vera Sans", "DejaVu Sans"]
    rcParams['legend.fontsize'] = 11
    rcParams['legend.labelspacing'] = 0.2
    rcParams['hatch.linewidth'] = 0.5  # https://stackoverflow.com/questions/29549530/how-to-change-the-linewidth-of-hatch-in-matplotlib
    rcParams['axes.xmargin'] = 0.0 # rootlike, no extra padding within x axis
    rcParams['axes.labelsize'] = 'x-large'
    rcParams['axes.formatter.use_mathtext'] = True
    rcParams['legend.framealpha'] = 0.65
    rcParams['axes.labelsize'] = 'x-large'
    rcParams['axes.titlesize'] = 'large'
    rcParams['xtick.labelsize'] = 'large'
    rcParams['ytick.labelsize'] = 'large'
    rcParams['figure.subplot.hspace'] = 0.1
    rcParams['figure.subplot.wspace'] = 0.1
    rcParams['figure.subplot.right'] = 0.97
    rcParams['figure.subplot.top'] = 0.92
    rcParams['figure.max_open_warning'] = 0
    rcParams['figure.dpi'] = 100
    rcParams["axes.formatter.limits"] = [-5,4] # scientific notation if log(y) outside this

def add_cms_info_1d(ax, typ="Preliminary", lumi="101", xtype=0.105):
    ax.text(0.0, 1.01,"CMS", horizontalalignment='left', verticalalignment='bottom', transform = ax.transAxes, name="Arial", weight="bold", size=15)
    ax.text(xtype, 1.01,typ, horizontalalignment='left', verticalalignment='bottom', transform = ax.transAxes, name="Arial", style="italic", size=14)
    if lumi is not None:
        ax.text(0.99, 1.01,"%s fb${}^\mathregular{-1}$ (13 TeV)" % (lumi), horizontalalignment='right', verticalalignment='bottom', transform = ax.transAxes, size=13)
    else:
        ax.text(0.99, 1.01,"(13 TeV)", horizontalalignment='right', verticalalignment='bottom', transform = ax.transAxes, size=13)

def add_cms_info_2d(ax, typ="Preliminary", lumi="101", xtype=0.15):
    ax.text(0.0, 1.01,"CMS", horizontalalignment='left', verticalalignment='bottom', transform = ax.transAxes, name="Arial", weight="bold", size=14)
    ax.text(xtype, 1.01,"Preliminary", horizontalalignment='left', verticalalignment='bottom', transform = ax.transAxes, name="Arial", style="italic", size=13)
    ax.text(0.99, 1.01,"%s fb${}^\mathregular{-1}$ (13 TeV)" % (lumi), horizontalalignment='right', verticalalignment='bottom', transform = ax.transAxes, size=12)
#     ax.text(0.99, 1.01,"(13 TeV)", horizontalalignment='right', verticalalignment='bottom', transform = ax.transAxes, size=12)

def to_yahist(h, overflow=False):
    if "TH1" in str(type(h)):
        c, e = h.to_numpy(flow=overflow)
        if overflow:
            c[1] += c[0]
            c[-2] += c[-1]
            c = c[1:-1]
            e = e[1:-1]
        h = Hist1D.from_bincounts(c, e)
    else:
        c, ex, ey = h.to_numpy(flow=False)
        h = Hist2D.from_bincounts(c.T, (ex, ey))
    return h

set_plotting_style()
# model_info = {
#     ("bphi",0.5,1): dict(label=r"B$\rightarrow\phi$ (0.5GeV,c$\tau$=1mm)", color=[0.98,0.85,0.29], fname="output_BToPhi_mphi0p5_ctau1mm.root"),
#     ("bphi",2,10): dict(label=r"B$\rightarrow\phi$ (2GeV,c$\tau$=10mm)", color=[0.94,0.58,0.21], fname="output_BToPhi_mphi2_ctau10mm.root"),
#     ("bphi",4,100):  dict(label=r"B$\rightarrow\phi$ (4GeV,c$\tau$=100mm)", color=[0.92,0.28,0.15], fname="output_BToPhi_mphi4_ctau100mm.root"),
#     ("hzd",2,100): dict(label=r"H$\rightarrow \mathrm{Z_d Z_d}$ (2GeV,c$\tau$=100mm)", color=[0.46,0.98,0.73], fname="output_HToZdZdTo2Mu2X_mzd2_ctau100mm.root"),
#     ("hzd",8,10): dict(label=r"H$\rightarrow \mathrm{Z_d Z_d}$ (8GeV,c$\tau$=10mm)", color=[0.33,0.73,0.98], fname="output_HToZdZdTo2Mu2X_mzd8_ctau10mm.root"),
#     ("hzd",15,1): dict(label=r"H$\rightarrow \mathrm{Z_d Z_d}$ (15GeV,c$\tau$=1mm)", color=[0.53,0.10,0.96], fname="output_HToZdZdTo2Mu2X_mzd15_ctau1mm.root"),
# }
model_info = {
    ("bphi",0.5,1): dict(label=r"B$\rightarrow\phi$ (0.5GeV,c$\tau$=1mm)", color="C0", fname="output_BToPhi_mphi0p5_ctau1mm.root"),
    ("bphi",2,10): dict(label=r"B$\rightarrow\phi$ (2GeV,c$\tau$=10mm)", color="C1", fname="output_BToPhi_mphi2_ctau10mm.root"),
    ("bphi",4,100):  dict(label=r"B$\rightarrow\phi$ (4GeV,c$\tau$=100mm)", color="C2", fname="output_BToPhi_mphi4_ctau100mm.root"),
    ("hzd",2,100): dict(label=r"H$\rightarrow \mathrm{Z_d Z_d}$ (2GeV,c$\tau$=100mm)", color="C4", fname="output_HToZdZdTo2Mu2X_mzd2_ctau100mm.root"),
    ("hzd",8,10): dict(label=r"H$\rightarrow \mathrm{Z_d Z_d}$ (8GeV,c$\tau$=10mm)", color="C3", fname="output_HToZdZdTo2Mu2X_mzd8_ctau10mm.root"),
    ("hzd",15,1): dict(label=r"H$\rightarrow \mathrm{Z_d Z_d}$ (15GeV,c$\tau$=1mm)", color="C5", fname="output_HToZdZdTo2Mu2X_mzd15_ctau1mm.root"),
}
os.system("mkdir -p plots_selection")

def plot_1():

    with uproot4.open("mcoutputs/main/output_HToZdZdTo2Mu2X_mzd8_ctau10mm.root") as f:

        fig, ax = plt.subplots()

        label = model_info[("hzd",8,10)]["label"]

        h1 = to_yahist(f["DV_rho_tot"], overflow=False).rebin(2)
        h1.plot(ax=ax, label=f"{label}, before veto", color="k", lw=2.0)

        h2 = to_yahist(f["DV_rho_matveto"], overflow=False).rebin(2)
        eff = h2.integral/h1.integral * 100.
        h2.plot(ax=ax, label=f"{label}, after veto (eff. = {eff:.1f}%)", color="C3", lw=1.0)

        add_cms_info_1d(ax, lumi=None, typ="Simulation")
        ax.set_ylim(bottom=0.)
        ax.set_ylabel("Unweighted events", ha="right", y=1.)
        ax.set_xlabel(r"$l_\mathrm{xy}$ (cm)", ha="right", x=1., labelpad=-1.0)

        fname = f"plots_selection/signal_passL1_lxy_materialveto.pdf"
        print(fname)
        fig.savefig(fname)

        os.system(f"ic {fname}")

def plot_2():

    with uproot4.open("dataoutputs/main/output.root") as f:

        fig, ax = plt.subplots()

        label = r"Data"

        h1 = to_yahist(f["DV_rho_tot"], overflow=False)
        h1.plot(ax=ax, label=f"{label}, before veto", color="k", lw=2.0)

        h2 = to_yahist(f["DV_rho_matveto"], overflow=False)
        eff = h2.integral/h1.integral * 100.
        h2.plot(ax=ax, label=f"{label}, after veto", color="C3", lw=1.0)

        add_cms_info_1d(ax)
        ax.set_yscale("log")
        ax.set_ylabel("Events", ha="right", y=1.)
        ax.set_xlabel(r"$l_\mathrm{xy}$ (cm)", ha="right", x=1., labelpad=-1.0)

        fname = f"plots_selection/data_passL1_lxy_materialveto.pdf"
        print(fname)
        fig.savefig(fname)

        os.system(f"ic {fname}")

def plot_3():

    with uproot4.open("dataoutputs/main/output.root") as f:

        for saxis in ["xy", "rhoz"]:
            for which in ["all", "pass"]:
                fig, ax = plt.subplots()
                hname = None
                if saxis == "xy":
                    if which == "all": hname = "DV_y_vs_x_tot"
                    if which == "pass": hname = "DV_y_vs_x_matveto"
                    xlabel = "DV x (cm)"
                    ylabel = "DV y (cm)"
                if saxis == "rhoz":
                    if which == "all": hname = "DV_rho_vs_z_tot"
                    if which == "pass": hname = "DV_rho_vs_z_matveto"
                    xlabel = "DV z (cm)"
                    ylabel = r"DV $\rho$ (cm)"

                h = to_yahist(f[hname])
                h.plot(ax=ax, logz=True, cmap="viridis")

                add_cms_info_2d(ax)
                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)
                ax.set_aspect(1.0 if saxis == "xy" else 2.5)

                fname = f"plots_selection/passL1_DV_{saxis}_{which}.pdf"
                print(fname)
                fig.savefig(fname)
                os.system(f"ic {fname}")

def plot_4():

    with uproot4.open("dataoutputs/nm1/output.root") as f:

        for which in ["nDV", "nMuon"]:
            fig, ax = plt.subplots()

            hname = f"{which}_vs_run"
            xlabel = "run number"
            ylabel = f"average reco. {which}"

            h = to_yahist(f[hname])
            h = h.restrict(300000,None)
            h = h.rebin(2,1)
            h = h.profile("x")
            h.plot(ax=ax, show_errors=True, ms=2., color="k")

            add_cms_info_1d(ax)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            if which == "nDV":
                ax.set_ylim(1.4,1.6)
            if which == "nMuon":
                ax.set_ylim(2.4,2.6)

            fname = f"plots_selection/passL1_{which}_vs_run.pdf"
            print(fname)
            fig.savefig(fname)
            os.system(f"ic {fname}")

def plot_5():
    f_data = uproot4.open("dataoutputs/nm1/output.root")
    f_mc = uproot4.open("mcoutputs/nm1/output_HToZdZdTo2Mu2X_mzd8_ctau10mm.root")

    for which, xlabel in [
            ("xError", "DV x Error (cm)"),
            ("yError", "DV y Error (cm)"),
            ("zError", "DV z Error (cm)"),
            ("chi2ndof", "DV chi2/ndof"),
            ("lxy", "$l_\mathrm{xy}$ (cm)"),
            ]:
        fig, ax = plt.subplots()

        hname = f"{which}_inc"
        ylabel = "Events"


        h1 = to_yahist(f_data[hname])
        h1.plot(ax=ax, color="k", label="Data")

        label = model_info[("hzd",8,10)]["label"]
        h2 = to_yahist(f_mc[hname])
        h2 *= h1.integral/h2.integral
        h2.plot(ax=ax, color="C3", label=label)

        add_cms_info_1d(ax)
        ax.set_xlabel(xlabel, ha="right", x=1., labelpad=-1.0)
        ax.set_ylabel("Events", ha="right", y=1.)
        ax.set_yscale("log")

        fname = f"plots_selection/passL1_DV_{which}.pdf"
        print(fname)
        fig.savefig(fname)
        os.system(f"ic {fname}")

    f_data.close()
    f_mc.close()

def plot_6():

    f_data = uproot4.open("dataoutputs/main/output.root")
    hists_data = dict()
    for k,v in f_data.items():
        if "_lxy" not in k: continue
        k = str(k).rsplit(";",1)[0]
        hists_data[k] = to_yahist(v)
    f_data.close()

    hists_mc = dict()
    for mk,model in model_info.items():
        print(mk, model)
        hists_mc[mk] = dict()
        fname = model["fname"]
        f_mc = uproot4.open(f"mcoutputs/main/{fname}")
        for k,v in f_mc.items():
            if "_lxy" not in k: continue
            k = str(k).rsplit(";",1)[0]
            h = to_yahist(v)
            h = Hist1D(h, label=model["label"], color=model["color"])
            hists_mc[mk][k] = h
        f_mc.close()

    for which, xlabel, log, line in [

            ("dimupt_full", r"dimuon $p_\mathrm{T}$", False, 25.),

            ("mu2pt_trig", r"trailing muon $p_\mathrm{T}$", False, None),
            ("mu2eta_trig", r"trailing muon $\eta$", False, None),
            ("mu2chi2ndof_trig", r"Trailing muon $\chi^2/\mathrm{ndof}$", False, 3.),
            ("mu2trkmeas_trig", r"Trailing muon tracker layers with meas.", False, 6.),
            ("absdphimudv_passid", r"|$\Delta\phi(\mu,\vec{\mathrm{DV}})$|", True, 0.02),
            ("absdphimumu_passid", r"|$\Delta\phi(\mu_1,\mu_2)$|", False, 2.8),
            ("mu2trackiso_passkin", r"Trailing muon relative track isolation", True, 0.1),
            ("mu2drjet_passkin", r"$\Delta R(\mu_2,\mathrm{jet})$", True, 0.3),
            ("mu2excesshits_baseline", r"Trailing muon n(valid-expected) pixel hits", False, 0.5),
            ("logabsetaphi_baseline", r"$\mathrm{log_{10}abs}(\Delta\eta_{\mu\mu}/\Delta\phi_{\mu\mu})$", False, 1.25),
            ("mindxy_extraiso", r"minimum $|d_\mathrm{xy}|$", True, None),
            ("mindxysig_extraiso", r"minimum $d_\mathrm{xy}$ significance", True, 2.),
            ("mindxyscaled_extraiso", r"minimum lifetime-scaled |$d_\mathrm{xy}$|", True, 0.1),

            ("mu2pt_incl", r"trailing muon $p_\mathrm{T}$", False, None),
            ("mu2eta_incl", r"trailing muon $\eta$", False, None),
            ("mu2chi2ndof_incl", r"Trailing muon $\chi^2/\mathrm{ndof}$", False, 3.),
            ("mu2trkmeas_incl", r"Trailing muon tracker layers with meas.", False, 6.),
            ("absdphimudv_incl", r"|$\Delta\phi(\mu\mu,\vec{\mathrm{DV}})$|", True, 0.02),
            ("absdphimumu_incl", r"|$\Delta\phi(\mu_1,\mu_2)$|", False, 2.8),
            ("mu2trackiso_incl", r"Trailing muon relative track isolation", True, 0.1),
            ("mu2drjet_incl", r"$\Delta R(\mu_2,\mathrm{jet})$", True, 0.3),
            ("mu2excesshits_incl", r"Trailing muon n(valid-expected) pixel hits", False, 0.5),
            ("logabsetaphi_incl", r"$\mathrm{log_{10}abs}(\Delta\eta_{\mu\mu}/\Delta\phi_{\mu\mu})$", False, 1.25),
            ("mindxy_incl", r"minimum $|d_\mathrm{xy}|$", True, None),
            ("mindxysig_incl", r"minimum $d_\mathrm{xy}$ significance", True, 2.),
            ("mindxyscaled_incl", r"minimum lifetime-scaled |$d_\mathrm{xy}$|", True, 0.1),

            ]:

        hnames = set([k.rsplit("_",1)[0] for k in hists_data.keys() if k.startswith(which)])
        for basehname in hnames:
            lxystr = basehname.split("_lxy",1)[1].split("_")[0]
            lxylow, lxyhigh = list(map(float, lxystr.replace("p",".").split("to")))

            fig, ax = plt.subplots()

            h = hists_data[f"{basehname}_lowmass"]
            N = h.integral
            h = h.normalize()
            label = "Data (mass < 5 GeV)"
            if which in ["dimupt_full"]:
                label += f" [N = {int(N):,}]"
            h.plot(ax=ax, show_errors=True, color="k", label=label, ms=3.5)

            h = hists_data[f"{basehname}_highmass"]
            N = h.integral
            h = h.normalize()
            label = "Data (mass > 5 GeV)"
            if which in ["dimupt_full"]:
                label += f" [N = {int(N):,}]"
            h.plot(ax=ax, show_errors=True, color="b", label=label, ms=3.5)

            for mk in hists_mc.keys():
                h = hists_mc[mk][f"{basehname}_allmass"]
                h = h.normalize()
                h.plot(ax=ax, histtype="step")

            if line is not None:
                ax.axvline(line,color="red",linestyle="--")

            add_cms_info_1d(ax)
            ax.set_xlabel(xlabel, ha="right", x=1., labelpad=-1.0)
            ax.set_ylabel("Fraction of events", ha="right", y=1.)

            ax.set_title(rf"{lxylow} cm < $l_\mathrm{{xy}}$ < {lxyhigh} cm", color=(0.2,0.2,0.2))

            if log:
                ax.set_yscale("log")

            fname = f"plots_selection/{basehname}.pdf"
            print(fname)
            fig.savefig(fname)
            # os.system(f"ic {fname}")

if __name__ == "__main__":
    pass


    # # plot_1()
    # plot_2()
    # plot_3()
    # # plot_4()
    # plot_5()
    plot_6()

