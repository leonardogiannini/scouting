#!/usr/bin/env sh

mkdir -p dataoutputs/{main,nm1}/

hadd -k -f dataoutputs/main/output.root /hadoop/cms/store/user/namin/ProjectMetis/Scouting_main_HISTS_histsv3/*.root
hadd -k -f dataoutputs/nm1/output.root /hadoop/cms/store/user/namin/ProjectMetis/Scouting_nm1_HISTS_histsv1/*.root
