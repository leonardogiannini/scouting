from metis.CMSSWTask import CMSSWTask
from metis.CondorTask import CondorTask
from metis.Sample import DirectorySample, DBSSample, DummySample
from metis.StatsParser import StatsParser
from metis.Optimizer import Optimizer
import time

import itertools


def get_tasks(which):

    extra_requirements = "true"
    slc7ggphi = [2,4,5,10,12,25,40,62.5]

    if which == "hzdzd":

        # tag = "v1"
        # events_per_point = 50000
        # events_per_job = 500
        # masses = [0.5, 0.75, 1, 1.5, 2, 3, 4, 5, 8, 10, 15, 25]
        # ctaus = [1, 10, 100]
        # years = [2017, 2018]
        # sname = "zd"
        # pdname = "HToZdZdTo2Mu2X"

        tag = "v1"
        events_per_point = 50000
        events_per_job = 500
        masses = [35,50]
        ctaus = [1, 10, 100, 1000]
        # years = [2017, 2018]
        years = [2018]
        sname = "zd"
        pdname = "HToZdZdTo2Mu2X"

    elif which == "btophi":

        tag = "v1"
        events_per_point = 1250000 # x25 due to filter eff.
        events_per_job = 12500 # x25 due to filter eff.
        masses = [0.3, 0.5, 0.6, 0.75, 1, 1.25, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
        ctaus = [1, 10, 100]
        years = [2017, 2018]
        sname = "phi"
        pdname = "BToPhi"

    elif which == "ggphi":


        # tag = "v1"
        # events_per_point = 50000
        # events_per_job = 500
        # masses = [2, 3, 4, 5, 8, 10, 12, 15, 25]
        # ctaus = [1, 10, 100]
        # years = [2017, 2018]
        # sname = "phi"
        # pdname = "ggPhi"

        tag = "v1"
        events_per_point = 50000
        events_per_job = 500
        # masses = [40, 62.5]
        masses = [62.5]
        ctaus = [1, 10, 100]
        # years = [2017, 2018]
        years = [2018]
        sname = "phi"
        pdname = "ggPhi"

    else:
        raise Exception()

    tasks = []
    for year,mass,ctau in itertools.product(years,masses,ctaus):

        fmass = float(mass)
        mass = str(mass).replace(".","p")

        epp = int(events_per_point)

        if which == "hzdzd":
            if 1 <= fmass <= 2:
                epp *= 2
            elif fmass < 1:
                epp *= 3
        elif which == "ggphi":
            if fmass < 2:
                epp *= 2
        elif which == "btophi":
            if 0.75 <= fmass <= 1.25:
                epp *= 2
            elif fmass < 0.75:
                epp *= 3

        reqname = "y{}_m{}{}_ctau{}_{}".format(year,sname,mass,ctau,tag)
        njobs = epp//events_per_job
        extra_classads = []

        if fmass in slc7ggphi:
            extra_classads.append(["SingularityImage","/cvmfs/singularity.opensciencegrid.org/cmssw/cms:rhel7-m202006"])
        else:
            extra_classads.append(["SingularityImage","/cvmfs/singularity.opensciencegrid.org/cmssw/cms:rhel6-m202006"])

        task = CondorTask(
                sample = DummySample(dataset="/{}/params_year{}_m{}{}_ctau{}mm/RAWSIM".format(pdname,year,sname,mass,ctau),N=njobs,nevents=epp),
                output_name = "output.root",
                executable = "executables/condor_executable_{}.sh".format(which),
                tarfile = "package.tar.gz",
                open_dataset = True,
                special_dir = "scoutingmc",
                files_per_output = 1,
                # recopy_inputs = True, # FIXME
                condor_submit_params = {
                    # "sites":"T2_US_UCSD,T2_US_MIT,T2_US_Caltech", # FIXME
                    "classads": [
                        ["param_year",year],
                        ["param_mass",mass],
                        ["param_ctau",ctau],
                        ["param_nevents",events_per_job],
                        ["metis_extraargs",""],
                        ["JobBatchName",reqname],
                        ]+extra_classads,
                    "requirements_line": 'Requirements = ((HAS_SINGULARITY=?=True) && {extra_requirements})'.format(extra_requirements=extra_requirements),
                    },
                tag = tag,
                )
        tasks.append(task)
    return tasks


if __name__ == "__main__":

    for i in range(500):

        tasks = []
        # tasks.extend(get_tasks("hzdzd"))
        # tasks.extend(get_tasks("btophi"))
        tasks.extend(get_tasks("ggphi"))

        total_summary = {}
        for task in tasks:
            task.process()
            total_summary[task.get_sample().get_datasetname()] = task.get_task_summary()
        StatsParser(data=total_summary, webdir="~/public_html/dump/scouting/").do()

        time.sleep(6*60*60)
