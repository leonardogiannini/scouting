import os
import time

from metis.CondorTask import CondorTask
from metis.Sample import DBSSample, DirectorySample
from metis.StatsParser import StatsParser

# if not os.path.exists("inputs_nm1.tar.gz"):
#     os.system("tar cvzf inputs_nm1.tar.gz looper_nm1.py")
# if not os.path.exists("inputs_main.tar.gz"):
#     os.system("tar cvzf inputs_main.tar.gz looper_main.py")
os.system("tar cvzf inputs_nm1.tar.gz looper_nm1.py")
os.system("tar cvzf inputs_main.tar.gz looper_main.py")

for i in range(100):
    total_summary = {}
    tasks = []

    # tag = "histsv1"
    # task = CondorTask(
    #         sample = DirectorySample(
    #             location="/hadoop/cms/store/user/namin/ProjectMetis/ScoutingCaloMuon_Run201*_v13_RAW_dvnm1v2/",
    #             dataset="/Scouting/nm1/HISTS",
    #             ),
    #         files_per_output = 15,
    #         output_name = "output.root",
    #         tag = tag,
    #         cmssw_version = "CMSSW_10_2_5",
    #         scram_arch = "slc6_amd64_gcc700",
    #         tarfile = "inputs_nm1.tar.gz",
    #         executable = "condor_exe.sh",
    #         condor_submit_params = {
    #             "sites":"T2_US_UCSD",
    #             },
    #         )
    # tasks.append(task)

    tag = "histsv3"
    task = CondorTask(
            sample = DirectorySample(
                location="/hadoop/cms/store/user/namin/ProjectMetis/ScoutingCaloMuon_Run201*_v13_RAW_v25/",
                dataset="/Scouting/main/HISTS",
                ),
            files_per_output = 2,
            output_name = "output.root",
            tag = tag,
            cmssw_version = "CMSSW_10_2_5",
            scram_arch = "slc6_amd64_gcc700",
            tarfile = "inputs_main.tar.gz",
            executable = "condor_exe.sh",
            condor_submit_params = {
                "sites":"T2_US_UCSD",
                },
            )
    tasks.append(task)

    for task in tasks:
        task.process()
        total_summary[task.get_sample().get_datasetname()] = task.get_task_summary()


    StatsParser(data=total_summary, webdir="~/public_html/dump/metis_nanotest/").do()
    time.sleep(30*60)
