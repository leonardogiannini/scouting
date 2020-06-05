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
            kwargs["files_per_output"] = 500
            batchname = "_".join(dataset.split("params_",1)[1].split("/",1)[0].split("_")[:2] + [tag])
        task = CondorTask(
                sample = DirectorySample(location=location,dataset=dataset),
                open_dataset = open_dataset,
                flush = True,
                output_name = "output.root",
                executable = "executables/scouting_exe.sh",
                tarfile = "package.tar.gz",
                condor_submit_params = {
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

    # print("Did you do `./make_tar.sh`? Sleeping for 5s for you to quit if not.")
    # time.sleep(5)

    # each task needs 
    # - a location (path to input root files)
    # - an output dataset name (just for bookkeping/output folder naming)
    # - a tag (for bookkeeping/output folder naming/versioning)
    # - isdata=[True|False]
    # extra parameters to the babymaker can be passed with extra_args.
    infos = []

    # tag = "v22" # MC and all data
    tag = "v23" # MC and unblinded data

    # DATA
    infos.extend([

            dict(location="/hadoop/cms/store/user/namin/ScoutingCaloMuon/crab_skim_2018A_v11_unblind/*/*/", dataset="/ScoutingCaloMuon/Run2018skim_2018A_v11_unblind/RAW", tag=tag, isdata=True, extra_args="--year 2018"),
            dict(location="/hadoop/cms/store/user/namin/ScoutingCaloMuon/crab_skim_2018B_v11_unblind/*/*/", dataset="/ScoutingCaloMuon/Run2018skim_2018B_v11_unblind/RAW", tag=tag, isdata=True, extra_args="--year 2018"),
            dict(location="/hadoop/cms/store/user/namin/ScoutingCaloMuon/crab_skim_2018C_v11_unblind/*/*/", dataset="/ScoutingCaloMuon/Run2018skim_2018C_v11_unblind/RAW", tag=tag, isdata=True, extra_args="--year 2018"),
            dict(location="/hadoop/cms/store/user/namin/ScoutingCaloMuon/crab_skim_2018D_v11_unblind/*/*/", dataset="/ScoutingCaloMuon/Run2018skim_2018D_v11_unblind/RAW", tag=tag, isdata=True, extra_args="--year 2018"),
            dict(location="/hadoop/cms/store/user/namin/ScoutingCaloMuon/crab_skim_2017C_v11_unblind/*/*/", dataset="/ScoutingCaloMuon/Run2017skim_2017C_v11_unblind/RAW", tag=tag, isdata=True, extra_args="--year 2017"),
            dict(location="/hadoop/cms/store/user/namin/ScoutingCaloMuon/crab_skim_2017D_v11_unblind/*/*/", dataset="/ScoutingCaloMuon/Run2017skim_2017D_v11_unblind/RAW", tag=tag, isdata=True, extra_args="--year 2017"),
            dict(location="/hadoop/cms/store/user/namin/ScoutingCaloMuon/crab_skim_2017E_v11_unblind/*/*/", dataset="/ScoutingCaloMuon/Run2017skim_2017E_v11_unblind/RAW", tag=tag, isdata=True, extra_args="--year 2017"),
            dict(location="/hadoop/cms/store/user/namin/ScoutingCaloMuon/crab_skim_2017F_v11_unblind/*/*/", dataset="/ScoutingCaloMuon/Run2017skim_2017F_v11_unblind/RAW", tag=tag, isdata=True, extra_args="--year 2017"),

            # dict(location="/hadoop/cms/store/user/namin/ScoutingCaloMuon/crab_skim_2018A_v11/*/*/", dataset="/ScoutingCaloMuon/Run2018skim_2018A_v11/RAW", tag=tag, isdata=True, extra_args="--year 2018"),
            # dict(location="/hadoop/cms/store/user/namin/ScoutingCaloMuon/crab_skim_2018B_v11/*/*/", dataset="/ScoutingCaloMuon/Run2018skim_2018B_v11/RAW", tag=tag, isdata=True, extra_args="--year 2018"),
            # dict(location="/hadoop/cms/store/user/namin/ScoutingCaloMuon/crab_skim_2018C_v11/*/*/", dataset="/ScoutingCaloMuon/Run2018skim_2018C_v11/RAW", tag=tag, isdata=True, extra_args="--year 2018"),
            # dict(location="/hadoop/cms/store/user/namin/ScoutingCaloMuon/crab_skim_2018D_v11/*/*/", dataset="/ScoutingCaloMuon/Run2018skim_2018D_v11/RAW", tag=tag, isdata=True, extra_args="--year 2018"),
            # dict(location="/hadoop/cms/store/user/namin/ScoutingCaloMuon/crab_skim_2017C_v11/*/*/", dataset="/ScoutingCaloMuon/Run2017skim_2017C_v11/RAW", tag=tag, isdata=True, extra_args="--year 2017"),
            # dict(location="/hadoop/cms/store/user/namin/ScoutingCaloMuon/crab_skim_2017D_v11/*/*/", dataset="/ScoutingCaloMuon/Run2017skim_2017D_v11/RAW", tag=tag, isdata=True, extra_args="--year 2017"),
            # dict(location="/hadoop/cms/store/user/namin/ScoutingCaloMuon/crab_skim_2017E_v11/*/*/", dataset="/ScoutingCaloMuon/Run2017skim_2017E_v11/RAW", tag=tag, isdata=True, extra_args="--year 2017"),
            # dict(location="/hadoop/cms/store/user/namin/ScoutingCaloMuon/crab_skim_2017F_v11/*/*/", dataset="/ScoutingCaloMuon/Run2017skim_2017F_v11/RAW", tag=tag, isdata=True, extra_args="--year 2017"),

            ])

    # MC
    locations = glob.glob("/hadoop/cms/store/user/namin/ProjectMetis/HToZdZdTo2Mu2X_params_mzd*_ctau*mm_RAWSIM_v10/")
    for location in locations:
        taskname = location.rstrip("/").rsplit("/")[-1]
        dataset = "/{}/{}/BABY".format(
                taskname.split("_",1)[0],
                taskname.split("_",1)[1].split("_RAWSIM")[0],
                )
        infos.append(dict(location=location, dataset=dataset, isdata=False, tag=tag, extra_args="--year 2018"))

    infos.append(dict(
        location="/hadoop/cms/store/user/namin/DisplacedMuons/2017/DirectGluonFusion_PhiToMuMu/ggPhimumu_Phimass2_Phictau0_5_part1/",
        dataset="/GGPhiToMuMu/params_mphi2_ctau0p5mm/BABY",
        isdata=False,
        tag=tag,
        extra_args="--year 2017",
        ))

    infos.append(dict(
        location="/hadoop/cms/store/user/namin/ProjectMetis/BToPhi_params_mphi2_ctau20mm_RAWSIM_v0/",
        dataset="/BToPhi/params_mphi2_ctau20mm/BABY",
        isdata=False,
        tag=tag,
        extra_args="--year 2017",
        ))
        
    # pprint(infos,width=200)
    # sys.exit()

    tasks = get_tasks(infos)

    # tasks.extend(get_bg_tasks(
    #         tag="v7",
    #         ))

    for _ in range(500):
        total_summary = {}
        for task in tasks:
            task.process()
            total_summary[task.get_sample().get_datasetname()] = task.get_task_summary()
        StatsParser(data=total_summary, webdir="~/public_html/dump/scouting/").do()
        # time.sleep(30*60)
        time.sleep(4*60*60)
