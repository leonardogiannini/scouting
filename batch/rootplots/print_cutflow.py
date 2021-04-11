import ROOT


N = 10

fname = "dataoutputs/main/output.root"
f = ROOT.TFile(fname)
h = f.Get("cutflow_masslt5_lxylt1")
vals_data_masslt5_lxylt1 = [int(h.GetBinContent(ibin)) for ibin in range(1,N+1)]
h = f.Get("cutflow_masslt5_lxygt1")
vals_data_masslt5_lxygt1 = [int(h.GetBinContent(ibin)) for ibin in range(1,N+1)]
h = f.Get("cutflow_massgt5_lxylt1")
vals_data_massgt5_lxylt1 = [int(h.GetBinContent(ibin)) for ibin in range(1,N+1)]
h = f.Get("cutflow_massgt5_lxygt1")
vals_data_massgt5_lxygt1 = [int(h.GetBinContent(ibin)) for ibin in range(1,N+1)]


fname = "mcoutputs/main/output_HToZdZdTo2Mu2X_mzd2_ctau1mm.root"
f = ROOT.TFile(fname)
h1 = f.Get("cutflow_masslt5_lxylt1")
h2 = f.Get("cutflow_massgt5_lxylt1")
vals_hzd_mzd2_ctau1mm_lxylt1 = [int(h1.GetBinContent(ibin))+int(h2.GetBinContent(ibin)) for ibin in range(1,N+1)]
h1 = f.Get("cutflow_masslt5_lxygt1")
h2 = f.Get("cutflow_massgt5_lxygt1")
vals_hzd_mzd2_ctau1mm_lxygt1 = [int(h1.GetBinContent(ibin))+int(h2.GetBinContent(ibin)) for ibin in range(1,N+1)]

fname = "mcoutputs/main/output_HToZdZdTo2Mu2X_mzd2_ctau10mm.root"
f = ROOT.TFile(fname)
h1 = f.Get("cutflow_masslt5_lxylt1")
h2 = f.Get("cutflow_massgt5_lxylt1")
vals_hzd_mzd2_ctau10mm_lxylt1 = [int(h1.GetBinContent(ibin))+int(h2.GetBinContent(ibin)) for ibin in range(1,N+1)]
h1 = f.Get("cutflow_masslt5_lxygt1")
h2 = f.Get("cutflow_massgt5_lxygt1")
vals_hzd_mzd2_ctau10mm_lxygt1 = [int(h1.GetBinContent(ibin))+int(h2.GetBinContent(ibin)) for ibin in range(1,N+1)]

fname = "mcoutputs/main/output_HToZdZdTo2Mu2X_mzd8_ctau100mm.root"
f = ROOT.TFile(fname)
h1 = f.Get("cutflow_masslt5_lxylt1")
h2 = f.Get("cutflow_massgt5_lxylt1")
vals_hzd_mzd8_ctau100mm_lxylt1 = [int(h1.GetBinContent(ibin))+int(h2.GetBinContent(ibin)) for ibin in range(1,N+1)]
h1 = f.Get("cutflow_masslt5_lxygt1")
h2 = f.Get("cutflow_massgt5_lxygt1")
vals_hzd_mzd8_ctau100mm_lxygt1 = [int(h1.GetBinContent(ibin))+int(h2.GetBinContent(ibin)) for ibin in range(1,N+1)]

def format_val(v):
    return "{:13,}".format(v)

print("# right justify the values in latex")
for rawrow in zip(
        vals_data_masslt5_lxylt1,
        vals_data_masslt5_lxygt1,
        vals_data_massgt5_lxylt1,
        vals_data_massgt5_lxygt1,
        vals_hzd_mzd2_ctau1mm_lxylt1,
        vals_hzd_mzd2_ctau1mm_lxygt1,
        vals_hzd_mzd2_ctau10mm_lxylt1,
        vals_hzd_mzd2_ctau10mm_lxygt1,
        vals_hzd_mzd8_ctau100mm_lxylt1,
        vals_hzd_mzd8_ctau100mm_lxygt1,
        ):
    row = " & ".join([format_val(v) for v in rawrow])
    print(row)


