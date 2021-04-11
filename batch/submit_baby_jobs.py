from metis.CMSSWTask import CMSSWTask
from metis.CondorTask import CondorTask
from metis.Sample import DirectorySample, DBSSample
from metis.StatsParser import StatsParser
from metis.Optimizer import Optimizer
import time
from pprint import pprint
import glob
import sys

extra_requirements = "True"
blacklisted_machines = [
         ]
if blacklisted_machines:
    extra_requirements = " && ".join(map(lambda x: '(TARGET.Machine != "{0}")'.format(x),blacklisted_machines))

def get_tasks(infos):
    tasks = []
    for info in infos:
        location = info["location"]
        dataset = info["dataset"]
        isdata = info["isdata"]
        open_dataset = info.get("open_dataset",False)
        tag = info["tag"]
        extra_args = info.get("extra_args","")
        kwargs = {}
        if isdata:
            kwargs["MB_per_output"] = (4000 if "skim1cm" in extra_args else 1000)
            batchname = dataset.split("_",1)[-1].split("/")[0]+"_"+tag
        else:
            kwargs["files_per_output"] = 200
            batchname = "_".join(dataset.split("params_",1)[1].split("/",1)[0].split("_")[:2] + [tag])
        task = CondorTask(
                sample = DirectorySample(location=location,dataset=dataset),
                open_dataset = open_dataset,
                flush = True,
                output_name = "output.root",
                executable = "executables/scouting_exe.sh",
                tarfile = "package.tar.gz",
                condor_submit_params = {
                    "container": "/cvmfs/singularity.opensciencegrid.org/cmssw/cms:rhel6-m202006",
                    "sites":"T2_US_UCSD",
                    "classads": [
                        ["metis_extraargs",extra_args],
                        ["JobBatchName",batchname],
                        ],
                    "requirements_line": 'Requirements = ((HAS_SINGULARITY=?=True) && {extra_requirements})'.format(extra_requirements=extra_requirements),
                    },
                cmssw_version = "CMSSW_10_2_5",
                scram_arch = "slc6_amd64_gcc700",
                tag = tag,
                **kwargs
                )
        tasks.append(task)
    return tasks


if __name__ == "__main__":

    print("Did you do `./make_tar.sh`? Sleeping for 3s for you to quit if not.")
    time.sleep(3)

    # each task needs 
    # - a location (path to input root files)
    # - an output dataset name (just for bookkeping/output folder naming)
    # - a tag (for bookkeeping/output folder naming/versioning)
    # - isdata=[True|False]
    # extra parameters to the babymaker can be passed with extra_args.
    infos = []


    # tag_data = "v25"
    # infos.extend([
    #         dict(location="/hadoop/cms/store/user/namin/ScoutingCaloMuon/crab_skim_2018A_v13/*/*/", dataset="/ScoutingCaloMuon/Run2018skim_2018A_v13/RAW", tag=tag_data, isdata=True, extra_args="--year 2018"),
    #         dict(location="/hadoop/cms/store/user/namin/ScoutingCaloMuon/crab_skim_2018B_v13/*/*/", dataset="/ScoutingCaloMuon/Run2018skim_2018B_v13/RAW", tag=tag_data, isdata=True, extra_args="--year 2018"),
    #         dict(location="/hadoop/cms/store/user/namin/ScoutingCaloMuon/crab_skim_2018C_v13/*/*/", dataset="/ScoutingCaloMuon/Run2018skim_2018C_v13/RAW", tag=tag_data, isdata=True, extra_args="--year 2018"),
    #         dict(location="/hadoop/cms/store/user/namin/ScoutingCaloMuon/crab_skim_2018D_v13/*/*/", dataset="/ScoutingCaloMuon/Run2018skim_2018D_v13/RAW", tag=tag_data, isdata=True, extra_args="--year 2018"),
    #         dict(location="/hadoop/cms/store/user/namin/ScoutingCaloMuon/crab_skim_2017C_v13/*/*/", dataset="/ScoutingCaloMuon/Run2017skim_2017C_v13/RAW", tag=tag_data, isdata=True, extra_args="--year 2017"),
    #         dict(location="/hadoop/cms/store/user/namin/ScoutingCaloMuon/crab_skim_2017D_v13/*/*/", dataset="/ScoutingCaloMuon/Run2017skim_2017D_v13/RAW", tag=tag_data, isdata=True, extra_args="--year 2017"),
    #         dict(location="/hadoop/cms/store/user/namin/ScoutingCaloMuon/crab_skim_2017E_v13/*/*/", dataset="/ScoutingCaloMuon/Run2017skim_2017E_v13/RAW", tag=tag_data, isdata=True, extra_args="--year 2017"),
    #         dict(location="/hadoop/cms/store/user/namin/ScoutingCaloMuon/crab_skim_2017F_v13/*/*/", dataset="/ScoutingCaloMuon/Run2017skim_2017F_v13/RAW", tag=tag_data, isdata=True, extra_args="--year 2017"),
    #         ])

    tag_mc = "v25"
    # MC
    locations = []
    # locations += glob.glob("/hadoop/cms/store/user/namin/ProjectMetis/HToZdZdTo2Mu2X_params_m*_ctau*mm_RAWSIM_vtestfine2/")
    # locations += glob.glob("/hadoop/cms/store/user/namin/ProjectMetis/BToPhi_params_m*_ctau*mm_RAWSIM_vtestfine2/")
    locations += glob.glob("/hadoop/cms/store/user/namin/scoutingmc/HToZdZdTo2Mu2X_params_year*_m*_ctau*mm_RAWSIM_v1/")
    locations += glob.glob("/hadoop/cms/store/user/namin/scoutingmc/BToPhi_params_year*_m*_ctau*mm_RAWSIM_v1/")
    locations += glob.glob("/hadoop/cms/store/user/namin/scoutingmc/ggPhi_params_year*_m*_ctau*mm_RAWSIM_v1/")
    for location in locations:
        taskname = location.rstrip("/").rsplit("/")[-1]
        dataset = "/{}/{}/BABY".format(
                taskname.split("_",1)[0],
                taskname.split("_",1)[1].split("_RAWSIM")[0],
                )
        year = 2018
        if "year2017" in dataset:
            year = 2017
        infos.append(dict(location=location, dataset=dataset, isdata=False, tag=tag_mc, extra_args="--year {}".format(year)))

    # # test 3/4 body
    # locations = []
    # locations += glob.glob("/hadoop/cms/store/user/namin/testscoutingmc/BToPhi_params_year*_m*_ctau*mm_RAWSIM_v*/")
    # for location in locations:
    #     tag_mc = location.rsplit("_",1)[-1].replace("/","")
    #     taskname = location.rstrip("/").rsplit("/")[-1]
    #     dataset = "/{}/{}/BABY".format(
    #             taskname.split("_",1)[0],
    #             taskname.split("_",1)[1].split("_RAWSIM")[0],
    #             )
    #     year = 2018
    #     if "year2017" in dataset:
    #         year = 2017
    #     infos.append(dict(location=location, dataset=dataset, isdata=False, tag=tag_mc, extra_args="--year {}".format(year)))

    tasks = get_tasks(infos)

    for _ in range(500):
        total_summary = {}
        for task in tasks:
            task.process()
            total_summary[task.get_sample().get_datasetname()] = task.get_task_summary()
        StatsParser(data=total_summary, webdir="~/public_html/dump/scouting/").do()
        # time.sleep(30*60)
        time.sleep(4*60*60)
