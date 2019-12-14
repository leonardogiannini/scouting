from CRABClient.UserUtilities import config, getUsernameFromSiteDB

# https://twiki.cern.ch/twiki/bin/view/CMSPublic/CRAB3ConfigurationFile
config = config()

# data_era = "D"
# ntuple_version = "v4"
# do_2018C_1fb_unblind = False

# data_era = "C"
# ntuple_version = "v9"
# do_2018C_1fb_unblind = True

data_era = "C"
ntuple_version = "v9"
do_2018C_1fb_unblind = False

config.General.requestName = 'skim_2018{}_{}{}'.format(
        data_era,
        ntuple_version,
        ("_unblind1fb" if do_2018C_1fb_unblind else "")
        )
config.Data.inputDataset = '/ScoutingCaloMuon/Run2018{}-v1/RAW'.format(data_era)

config.General.workArea = 'crab'
config.General.transferLogs = True

config.JobType.pluginName = 'Analysis'
config.JobType.psetName = 'Scouting/NtupleMaker/test/dataproducer.py'

config.Data.splitting = 'EventAwareLumiBased'
config.Data.unitsPerJob = int(10e6)

if do_2018C_1fb_unblind:
    config.Data.lumiMask = "data/unblind_2018C_1fb_JSON.txt"

config.Data.publication = False
config.Site.storageSite = "T2_US_UCSD"
