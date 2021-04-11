#!/usr/bin/env sh

function get_csv() {
    echo $@ | sed -s 's/ /,/g'
}

mkdir -p mcoutputs/{main,nm1}/

python looper_main.py $(get_csv /hadoop/cms/store/user/namin/ProjectMetis/BToPhi_params_year201{7,8}_mphi0p5_ctau1mm_BABY_v25/*.root) -o mcoutputs/main/output_BToPhi_mphi0p5_ctau1mm.root
python looper_main.py $(get_csv /hadoop/cms/store/user/namin/ProjectMetis/BToPhi_params_year201{7,8}_mphi2_ctau10mm_BABY_v25/*.root) -o mcoutputs/main/output_BToPhi_mphi2_ctau10mm.root
python looper_main.py $(get_csv /hadoop/cms/store/user/namin/ProjectMetis/BToPhi_params_year201{7,8}_mphi4_ctau100mm_BABY_v25/*.root) -o mcoutputs/main/output_BToPhi_mphi4_ctau100mm.root
python looper_main.py $(get_csv /hadoop/cms/store/user/namin/ProjectMetis/HToZdZdTo2Mu2X_params_year201{7,8}_mzd2_ctau100mm_BABY_v25/*.root) -o mcoutputs/main/output_HToZdZdTo2Mu2X_mzd2_ctau100mm.root
python looper_main.py $(get_csv /hadoop/cms/store/user/namin/ProjectMetis/HToZdZdTo2Mu2X_params_year201{7,8}_mzd8_ctau10mm_BABY_v25/*.root) -o mcoutputs/main/output_HToZdZdTo2Mu2X_mzd8_ctau10mm.root
python looper_main.py $(get_csv /hadoop/cms/store/user/namin/ProjectMetis/HToZdZdTo2Mu2X_params_year201{7,8}_mzd15_ctau1mm_BABY_v25/*.root) -o mcoutputs/main/output_HToZdZdTo2Mu2X_mzd15_ctau1mm.root
# # python looper_nm1.py $(get_csv /home/users/namin/2019/scouting/repo/batch/dvnm1/output_mzd8_ctau10mm.root) -o mcoutputs/nm1/output_HToZdZdTo2Mu2X_mzd8_ctau10mm.root

python looper_main.py $(get_csv /hadoop/cms/store/user/namin/ProjectMetis/HToZdZdTo2Mu2X_params_year201{7,8}_mzd2_ctau1mm_BABY_v25/*.root) -o mcoutputs/main/output_HToZdZdTo2Mu2X_mzd2_ctau1mm.root
python looper_main.py $(get_csv /hadoop/cms/store/user/namin/ProjectMetis/HToZdZdTo2Mu2X_params_year201{7,8}_mzd2_ctau10mm_BABY_v25/*.root) -o mcoutputs/main/output_HToZdZdTo2Mu2X_mzd2_ctau10mm.root
python looper_main.py $(get_csv /hadoop/cms/store/user/namin/ProjectMetis/HToZdZdTo2Mu2X_params_year201{7,8}_mzd8_ctau100mm_BABY_v25/*.root) -o mcoutputs/main/output_HToZdZdTo2Mu2X_mzd8_ctau100mm.root
