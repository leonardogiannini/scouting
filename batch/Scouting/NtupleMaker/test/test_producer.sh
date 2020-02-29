
cmsRun producer.py inputs=/hadoop/cms/store/user/namin/ProjectMetis/BToPhi_params_mphi2_ctau20mm_RAWSIM_v0/rawsim/output_215.root era=2017 data=False nevents=500 output=output_mc.root >& log_data.txt &
cmsRun producer.py inputs=/hadoop/cms/store/user/namin/nanoaod/ScoutingCaloMuon__Run2018C-v1/6A94C331-F38D-E811-B4D7-FA163E146D61.root era=2018C data=True nevents=500 output=output_data.root >& log_mc.txt &
