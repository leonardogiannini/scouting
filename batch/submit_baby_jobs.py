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
                    "requirements_line": 'Requirements = ((HAS_SINGULARITY=?=True) && (HAS_CVMFS_cms_cern_ch =?= true) && {extra_requirements})'.format(extra_requirements=extra_requirements),
                    },
                cmssw_version = "CMSSW_10_2_5",
                scram_arch = "slc6_amd64_gcc700",
                tag = tag,
                **kwargs
                )
        tasks.append(task)
    return tasks


def get_bg_tasks(tag,extra_args=""):
    samples = [
            DBSSample(dataset="/JpsiToMuMu_JpsiPt8_TuneCP5_13TeV-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v1/AODSIM"),
            DBSSample(dataset="/BuToKJpsi_ToMuMu_MuFilter_SoftQCDnonD_TuneCP5_13TeV-pythia8-evtgen/RunIIAutumn18DR-PUPoissonAve20_102X_upgrade2018_realistic_v15-v2/AODSIM"),
            ]
    tasks = []
    for sample in samples:
        task = CondorTask(
                sample = sample,
                output_name = "output.root",
                executable = "executables/scouting_exe.sh",
                tarfile = "package.tar.gz",
                events_per_output = 500e3,
                condor_submit_params = {
                    "sites":"T2_US_UCSD",
                    "classads": [
                        ["metis_extraargs",extra_args],
                        ["JobBatchName",sample.get_datasetname().split("/")[1].split("_")[0]+"_"+tag],
                        ],
                    "requirements_line": 'Requirements = ((HAS_SINGULARITY=?=True) && (HAS_CVMFS_cms_cern_ch =?= true) && {extra_requirements})'.format(extra_requirements=extra_requirements),
                    },
                cmssw_version = "CMSSW_10_2_5",
                scram_arch = "slc6_amd64_gcc700",
                tag = tag,
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


    # DATA
    tag_data, tag_mc = "fourmuv24", "vtestfine2"
    infos.extend([
            dict(location="/hadoop/cms/store/user/namin/ScoutingCaloMuon/crab_skim_2018A_v12_unblindsubset/*/*/", dataset="/ScoutingCaloMuon/Run2018skim_2018A_v12/RAW", tag=tag_data, isdata=True, extra_args="--year 2018"),
            dict(location="/hadoop/cms/store/user/namin/ScoutingCaloMuon/crab_skim_2018B_v12_unblindsubset/*/*/", dataset="/ScoutingCaloMuon/Run2018skim_2018B_v12/RAW", tag=tag_data, isdata=True, extra_args="--year 2018"),
            dict(location="/hadoop/cms/store/user/namin/ScoutingCaloMuon/crab_skim_2018C_v12_unblindsubset/*/*/", dataset="/ScoutingCaloMuon/Run2018skim_2018C_v12/RAW", tag=tag_data, isdata=True, extra_args="--year 2018"),
            dict(location="/hadoop/cms/store/user/namin/ScoutingCaloMuon/crab_skim_2018D_v12_unblindsubset/*/*/", dataset="/ScoutingCaloMuon/Run2018skim_2018D_v12/RAW", tag=tag_data, isdata=True, extra_args="--year 2018"),
            dict(location="/hadoop/cms/store/user/namin/ScoutingCaloMuon/crab_skim_2017C_v12_unblindsubset/*/*/", dataset="/ScoutingCaloMuon/Run2017skim_2017C_v12/RAW", tag=tag_data, isdata=True, extra_args="--year 2017"),
            dict(location="/hadoop/cms/store/user/namin/ScoutingCaloMuon/crab_skim_2017D_v12_unblindsubset/*/*/", dataset="/ScoutingCaloMuon/Run2017skim_2017D_v12/RAW", tag=tag_data, isdata=True, extra_args="--year 2017"),
            dict(location="/hadoop/cms/store/user/namin/ScoutingCaloMuon/crab_skim_2017E_v12_unblindsubset/*/*/", dataset="/ScoutingCaloMuon/Run2017skim_2017E_v12/RAW", tag=tag_data, isdata=True, extra_args="--year 2017"),
            dict(location="/hadoop/cms/store/user/namin/ScoutingCaloMuon/crab_skim_2017F_v12_unblindsubset/*/*/", dataset="/ScoutingCaloMuon/Run2017skim_2017F_v12/RAW", tag=tag_data, isdata=True, extra_args="--year 2017"),
            ])

    # tag_data, tag_mc = "v25", "vtestfine2"
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


    # # MC
    # locations = []
    # locations += glob.glob("/hadoop/cms/store/user/namin/ProjectMetis/HToZdZdTo2Mu2X_params_m*_ctau*mm_RAWSIM_vtestfine2/")
    # locations += glob.glob("/hadoop/cms/store/user/namin/ProjectMetis/BToPhi_params_m*_ctau*mm_RAWSIM_vtestfine2/")
    # for location in locations:
    #     taskname = location.rstrip("/").rsplit("/")[-1]
    #     dataset = "/{}/{}/BABY".format(
    #             taskname.split("_",1)[0],
    #             taskname.split("_",1)[1].split("_RAWSIM")[0],
    #             )
    #     infos.append(dict(location=location, dataset=dataset, isdata=False, tag=tag_mc, extra_args="--year 2018"))

    # MC
    locations = []
    tag_mc = "fourmuv24"
    locations += glob.glob("/hadoop/cms/store/user/namin/aodsim4mu/HToZdZdTo4Mu_params_mzd*_ctau1mm_SKIM_v1")
    for location in locations:
        taskname = location.rstrip("/").rsplit("/")[-1]
        dataset = "/{}/{}/BABY".format(
                taskname.split("_",1)[0],
                taskname.split("_",1)[1].split("_SKIM")[0],
                )
        infos.append(dict(location=location, dataset=dataset, isdata=False, tag=tag_mc, extra_args="--year 2018"))

    tasks = get_tasks(infos)

    for _ in range(500):
        total_summary = {}
        for task in tasks:
            task.process()
            total_summary[task.get_sample().get_datasetname()] = task.get_task_summary()
        StatsParser(data=total_summary, webdir="~/public_html/dump/scouting/").do()
        # time.sleep(30*60)
        time.sleep(4*60*60)
