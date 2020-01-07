from metis.CMSSWTask import CMSSWTask
from metis.CondorTask import CondorTask
from metis.Sample import DirectorySample, DBSSample, DummySample
from metis.StatsParser import StatsParser
from metis.Optimizer import Optimizer
import time

import itertools

def submit():
    total_summary = {}

    extra_requirements = "True"
    blacklisted_machines = [
             "cabinet-4-4-29.t2.ucsd.edu",
             "cabinet-7-7-36.t2.ucsd.edu",
             "cabinet-8-8-1.t2.ucsd.edu",
             "sdsc-37.t2.ucsd.edu",
             ]
    if blacklisted_machines:
        extra_requirements = " && ".join(map(lambda x: '(TARGET.Machine != "{0}")'.format(x),blacklisted_machines))

    # Edit these parameters.
    # in total, there will be `len(masses)*len(ctaus)*events_per_point` events

    tag = "v9"

    events_per_point = 50000
    events_per_job = 500
    masses = [2,5,8,10,15,20]
    ctaus = [1,5,10,25,50]

    # # FIXME
    # events_per_point = 50000
    # events_per_job = 500
    # masses = [20]
    # ctaus = [50]

    for mass,ctau in itertools.product(masses,ctaus):

        # 4x the events for this particular point
        epp = int(events_per_point)
        if mass == 20 and ctau == 50:
            epp = int(4*events_per_point)

        reqname = "mzd{}_ctau{}_{}".format(mass,ctau,tag)
        njobs = epp//events_per_job
        task = CondorTask(
                sample = DummySample(dataset="/HToZdZdTo2Mu2X/params_mzd{}_ctau{}mm/RAWSIM".format(mass,ctau),N=njobs,nevents=epp),
                output_name = "output.root",
                executable = "executables/condor_executable.sh",
                tarfile = "package.tar.gz",
                # open_dataset = True,
                files_per_output = 1,
                condor_submit_params = {
                    # "sites":"T2_US_UCSD",
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
        submit()
        time.sleep(60*60)

