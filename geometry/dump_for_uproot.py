import ROOT as r
import array

branches = {}
def main():

    # f = r.TFile("cmsRecoGeom-2017.root")
    # f = r.TFile("cmsRecoGeom-data2018.root")
    # gtag = "102X_dataRun2_v11"
    # f = r.TFile("cmsRecoGeom-{}.root".format(gtag))
    # f = r.TFile("cmsRecoGeom-data2018.root")
    f = r.TFile("cmsRecoGeom-data2017.root")
    t = f.Get("idToGeo")

    # fname_out = "tracker_geometry_data2018_{}.root".format(gtag)
    # fname_out = "tracker_geometry_data2018.root"
    fname_out = "tracker_geometry_data2017.root"
    newfile = r.TFile(fname_out, "recreate")
    newtree = r.TTree("idToGeo","")

    def make_branch(name, tstr="vi"):
        global branches
        extra = []
        if tstr == "vvi": obj = r.vector("vector<int>")()
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
        branches[name] = obj
        newtree.Branch(name,obj,*extra)

    def clear_branches():
        global branches
        for v in branches.values():
            if hasattr(v,"clear"):
                v.clear()

    make_branch("id", "i")
    make_branch("points", "vf")
    make_branch("topology", "vf")
    make_branch("shape", "vf")
    make_branch("translation", "vf")
    make_branch("matrix", "vf")

    for e in t:

        clear_branches()
        for x in e.points: branches["points"].push_back(x)
        for x in e.topology: branches["topology"].push_back(x)
        for x in e.shape: branches["shape"].push_back(x)
        for x in e.translation: branches["translation"].push_back(x)
        for x in e.matrix: branches["matrix"].push_back(x)
        branches["id"][0] = e.id

        newtree.Fill()

    print(newtree.GetEntries())
    newtree.Write()
    newfile.Close()

if __name__ == "__main__":
    main()
