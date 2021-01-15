from metis.CMSSWTask import CMSSWTask
from metis.CondorTask import CondorTask
from metis.Sample import DirectorySample, DBSSample, DummySample
from metis.StatsParser import StatsParser
from metis.Optimizer import Optimizer
import time

import itertools

def submit(which):
    total_summary = {}

    extra_requirements = "true"

    if which == "hzdzd":
        tag = "vtestfine2"
        events_per_point = 30000
        events_per_job = 500
        masses = [0.25,0.3,0.4,0.5,0.6,0.75,1,1.25,1.5,2,2.5,3,4,5,6,8,10,12,15,18,21,25]
        ctaus = [1,10,100,1000]
        sname = "zd"
        pdname = "HToZdZdTo2Mu2X"

        # # FIXME
        # ctaus = [10]
        # masses = [8]
        # events_per_point *= 10

    elif which == "btophi":

        # tag = "vtestfine2"
        # events_per_point = 1000000 # 2000000
        # events_per_job = 5000 # filter efficiency around 4%
        # masses = [0.25,0.3,0.4,0.5,0.6,0.75,1,1.25,1.5,2,2.5,3,3.5,4,4.5]
        # ctaus = [1,10,100,1000]
        # sname = "phi"
        # pdname = "BToPhi"

        tag = "vtestfine2"
        events_per_point = 1000000 # 2000000
        events_per_job = 5000 # filter efficiency around 4%
        masses = [4.5, 5.0]
        ctaus = [1,10,100,1000]
        sname = "phi"
        pdname = "BToPhi"

    else:
        raise Exception()

    for mass,ctau in itertools.product(masses,ctaus):

        fmass = float(mass)
        mass = str(mass).replace(".","p")

        epp = int(events_per_point)

        if which == "hzdzd":
            if fmass <= 2:
                epp *= 12

        reqname = "m{}{}_ctau{}_{}".format(sname,mass,ctau,tag)
        njobs = epp//events_per_job
        task = CondorTask(
                sample = DummySample(dataset="/{}/params_m{}{}_ctau{}mm/RAWSIM".format(pdname,sname,mass,ctau),N=njobs,nevents=epp),
                output_name = "output.root",
                executable = "executables/condor_executable_{}.sh".format(which),
                tarfile = "package.tar.gz",
                open_dataset = True,
                files_per_output = 1,
                condor_submit_params = {
                    "classads": [
                        ["param_mass",mass],
                        ["param_ctau",ctau],
                        ["param_nevents",events_per_job],
                        ["metis_extraargs",""],
                        ["JobBatchName",reqname],
                        ],
                    "requirements_line": 'Requirements = ((HAS_SINGULARITY=?=True) && (HAS_CVMFS_cms_cern_ch =?= true) && {extra_requirements})'.format(extra_requirements=extra_requirements),
                    },
                tag = tag,
                )

        task.process()
        total_summary[task.get_sample().get_datasetname()] = task.get_task_summary()

    StatsParser(data=total_summary, webdir="~/public_html/dump/scouting/").do()

if __name__ == "__main__":

    for i in range(500):
        # submit("hzdzd")
        submit("btophi")
        time.sleep(2*60*60)

