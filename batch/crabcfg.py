from CRABClient.UserUtilities import config, getUsernameFromSiteDB

# https://twiki.cern.ch/twiki/bin/view/CMSPublic/CRAB3ConfigurationFile
config = config()

# era = "2018C"
# ntuple_version = "v9"
# do_2018C_1fb_unblind = True

era = "2017D"
ntuple_version = "v9"
do_2018C_1fb_unblind = False

config.General.requestName = 'skim_{}_{}{}'.format(
        era,
        ntuple_version,
        ("_unblind1fb" if do_2018C_1fb_unblind else "")
        )
config.Data.inputDataset = '/ScoutingCaloMuon/Run{}-v1/RAW'.format(era)

config.General.workArea = 'crab'
config.General.transferLogs = True

config.JobType.pluginName = 'Analysis'
config.JobType.psetName = 'Scouting/NtupleMaker/test/dataproducer.py'

config.JobType.pyCfgParams=["era={}".format(era),]

config.Data.splitting = 'EventAwareLumiBased'
config.Data.unitsPerJob = int(10e6)

if "2018" in era:
    config.Data.lumiMask = "data/Cert_314472-325175_13TeV_PromptReco_Collisions18_JSON.txt"
if "2017" in era:
    config.Data.lumiMask = "data/Cert_294927-306462_13TeV_EOY2017ReReco_Collisions17_JSON.txt"
if do_2018C_1fb_unblind:
    config.Data.lumiMask = "data/unblind_2018C_1fb_JSON.txt"


config.Data.publication = False
config.Site.storageSite = "T2_US_UCSD"
