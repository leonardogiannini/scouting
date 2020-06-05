from metis.CMSSWTask import CMSSWTask
from metis.CondorTask import CondorTask
from metis.Sample import DirectorySample, DBSSample, DummySample
from metis.StatsParser import StatsParser
from metis.Optimizer import Optimizer
import time

import itertools

def submit():
    total_summary = {}

    extra_requirements = "true"

    tag = "v0"
    events_per_job = 3000 # ~7-8% gen efficiency, so ~200-250 events from each job
    events_per_point = events_per_job * 2000
    masses = [2]
    ctaus = [20]

    for mass,ctau in itertools.product(masses,ctaus):

        reqname = "mphi{}_ctau{}_{}".format(mass,ctau,tag)
        njobs = int(events_per_point)//events_per_job
        dsname = "/BToPhi/params_mphi{}_ctau{}mm/RAWSIM".format(mass,ctau)
        task = CondorTask(
                sample = DummySample(dataset=dsname,N=njobs,nevents=int(events_per_point)),
                output_name = "output.root",
                executable = "condor_executable.sh",
                tarfile = "package.tar.gz",
                open_dataset = True,
                files_per_output = 1,
                condor_submit_params = {
                    "sites":"T2_US_UCSD",
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
        total_summary[dsname] = task.get_task_summary()

if __name__ == "__main__":

    for i in range(500):
        submit()
        break
        time.sleep(60*60)

