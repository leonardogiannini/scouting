import FWCore.ParameterSet.Config as cms

import FWCore.ParameterSet.VarParsing as VarParsing
opts = VarParsing.VarParsing('python')
vpbool = VarParsing.VarParsing.varType.bool
vpint = VarParsing.VarParsing.varType.int
vpstring = VarParsing.VarParsing.varType.string
opts.register('data'    , True  , mytype=vpbool)
opts.register('era'    , "2018C"  , mytype=vpstring)
opts.register('output'    , "output.root"  , mytype=vpstring)
opts.register('inputs'    , ""  , mytype=vpstring) # comma separated list of input files
opts.register('nevents'    , -1  , mytype=vpint)
opts.parseArguments()

inputs = []
def convert_fname(fname):
    if fname.startswith("/store") or fname.startswith("root://"): return fname
    return "file:" + fname.replace("file:","",1)
if not opts.inputs:
    if opts.data:
        inputs = ['file:/hadoop/cms/store/user/namin/nanoaod/ScoutingCaloMuon__Run2018C-v1/6A94C331-F38D-E811-B4D7-FA163E146D61.root']
        opts.era = "2018C"
    else:
        inputs = ['file:/hadoop/cms/store/user/namin/ProjectMetis/BToPhi_params_mphi2_ctau20mm_RAWSIM_v0/rawsim/output_215.root']
        opts.era = "2017"
else:
    inputs = map(convert_fname,opts.inputs.split(","))

print("""
Running with options:
    data = {}
    era = {}
    output = {}
    inputs = {}
    nevents = {}
""".format(
    opts.data,
    opts.era,
    opts.output,
    inputs,
    opts.nevents
    ))

process = cms.Process('SLIM')

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')

process.load('PhysicsTools.PatAlgos.producersLayer1.patCandidates_cff')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.Reconstruction_cff')
process.load('Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff')


## ----------------- Global Tag ------------------
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_condDBv2_cff')

gtag = None
if opts.data:
    gtags = {
            "2017C": "92X_dataRun2_HLT_v4",
            "2017D": "92X_dataRun2_HLT_v7",
            "2017E": "92X_dataRun2_HLT_v7",
            "2017F": "92X_dataRun2_HLT_v7",
            "2018A": "101X_dataRun2_HLT_v7",
            "2018B": "101X_dataRun2_HLT_v7",
            "2018C": "101X_dataRun2_HLT_v7",
            "2018D": "101X_dataRun2_HLT_v7",
            }
    if opts.era not in gtags.keys():
        raise RuntimeError("Invalid era. Must be one of: {}".format(sorted(gtag.keys())))
    gtag = gtags[opts.era]
else:
    if "2017" in opts.era:
        # gtag = "94X_mc2017_realistic_v10"
        gtag = "94X_mc2017_realistic_v11" # FIXME how to automatically determine (without using hltInfo <edmfile>)
    else:
        gtag = "102X_upgrade2018_realistic_v15"
print("Using {}".format(gtag))
process.GlobalTag.globaltag = gtag

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(opts.nevents)
)
process.MessageLogger.cerr.FwkReport.reportEvery = 5000
process.MessageLogger.suppressWarning = cms.untracked.vstring(["MeasurementTrackerEvent"])

# Input source
process.source = cms.Source("PoolSource",
    dropDescendantsOfDroppedBranches = cms.untracked.bool(True),
    fileNames = cms.untracked.vstring(),
    inputCommands = cms.untracked.vstring(
        'keep *', 
        'drop *_hltScoutingTrackPacker_*_*', 
    ),
)
process.source.fileNames = inputs

if not opts.data:
    process.source.duplicateCheckMode = cms.untracked.string('noDuplicateCheck')

# skim data, but keep all events for MC acceptance calculations
do_skim = opts.data

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('file:output.root'),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('skimpath'),
        ),
    outputCommands = cms.untracked.vstring(
        "drop *",
        "keep *_hltScoutingMuonPackerCalo_*_*",
        "keep *_hltScoutingCaloPacker_*_*",
        "keep *_hltScoutingPrimaryVertexPacker_*_*",
        "keep *_hltScoutingPrimaryVertexPackerCaloMuon_*_*",
        "keep *_triggerMaker_*_*",
        "keep *_hitMaker_*_*",
        "keep *_genParticles_*_HLT",
        ),
     basketSize = cms.untracked.int32(128*1024), # 128kb basket size instead of ~30kb default
)

if len(opts.output):
    process.out.fileName = "file:" + str(opts.output).strip().replace("file:","")

process.outpath = cms.EndPath(process.out)

process.Timing = cms.Service("Timing",
        summaryOnly = cms.untracked.bool(True)
        )

process.countmu = cms.EDFilter("ScoutingMuonCountFilter",
    src = cms.InputTag("hltScoutingMuonPackerCalo"),
    minNumber = cms.uint32(2)
)

process.countvtx = cms.EDFilter("ScoutingVertexCountFilter",
    src = cms.InputTag("hltScoutingMuonPackerCalo","displacedVtx"),
    minNumber = cms.uint32(1)
)

L1Info = ["L1_DoubleMu0", "L1_DoubleMu0_Mass_Min1", "L1_DoubleMu0_OQ",
 "L1_DoubleMu0_SQ", "L1_DoubleMu0_SQ_OS", "L1_DoubleMu0er1p4_SQ_OS_dR_Max1p4",
 "L1_DoubleMu0er1p5_SQ", "L1_DoubleMu0er1p5_SQ_OS",
 "L1_DoubleMu0er1p5_SQ_OS_dR_Max1p4", "L1_DoubleMu0er1p5_SQ_dR_Max1p4",
 "L1_DoubleMu0er2p0_SQ_OS_dR_Max1p4", "L1_DoubleMu0er2p0_SQ_dR_Max1p4",
 "L1_DoubleMu10_SQ", "L1_DoubleMu18er2p1", "L1_DoubleMu4_SQ_OS",
 "L1_DoubleMu4_SQ_OS_dR_Max1p2", "L1_DoubleMu4p5_SQ_OS",
 "L1_DoubleMu4p5_SQ_OS_dR_Max1p2", "L1_DoubleMu4p5er2p0_SQ_OS",
 "L1_DoubleMu4p5er2p0_SQ_OS_Mass7to18", "L1_DoubleMu9_SQ", "L1_DoubleMu_12_5",
 "L1_DoubleMu_15_5_SQ", "L1_DoubleMu_15_7", "L1_DoubleMu_15_7_Mass_Min1",
 "L1_DoubleMu_15_7_SQ",
 ]

HLTInfo = [
           ['DoubleMu3_noVtx',                   'DST_DoubleMu3_noVtx_CaloScouting_v*'],
           ['DoubleMu3_noVtx_Monitoring',        'DST_DoubleMu3_noVtx_CaloScouting_Monitoring_v*'],
           ]

do_trigger_objects = not opts.data
print(do_trigger_objects)

process.triggerMaker = cms.EDProducer("TriggerMaker",
        triggerAlias = cms.vstring(zip(*HLTInfo)[0]),
        triggerSelection = cms.vstring(zip(*HLTInfo)[1]),
        triggerConfiguration = cms.PSet(
            hltResults            = cms.InputTag('TriggerResults','','HLT'),
            l1tResults            = cms.InputTag(''),
            daqPartitions         = cms.uint32(1),
            l1tIgnoreMaskAndPrescale = cms.bool(False),
            throw                 = cms.bool(True),
            ),
        doL1 = cms.bool(True),
        doTriggerObjects = cms.bool(do_trigger_objects),
        AlgInputTag = cms.InputTag("gtStage2Digis"),
        l1tAlgBlkInputTag = cms.InputTag("gtStage2Digis"),
        l1tExtBlkInputTag = cms.InputTag("gtStage2Digis"),
        ReadPrescalesFromFile = cms.bool(False),
        l1Seeds = cms.vstring(L1Info),
        )

process.hitMaker = cms.EDProducer("HitMaker",
        muonInputTag = cms.InputTag("hltScoutingMuonPackerCalo"),
        dvInputTag = cms.InputTag("hltScoutingMuonPackerCalo:displacedVtx"),
        measurementTrackerEventInputTag = cms.InputTag("MeasurementTrackerEvent"),
        )

process.beamSpotMaker = cms.EDProducer("BeamSpotMaker")

from RecoTracker.MeasurementDet.measurementTrackerEventDefault_cfi import measurementTrackerEventDefault as _measurementTrackerEventDefault
process.MeasurementTrackerEvent = _measurementTrackerEventDefault.clone()

process.load("EventFilter.L1TRawToDigi.gtStage2Digis_cfi")
process.gtStage2Digis.InputLabel = cms.InputTag( "hltFEDSelectorL1" )

process.offlineBeamSpot = cms.EDProducer("BeamSpotProducer")

# trigger object stuff

process.load("PhysicsTools.PatAlgos.triggerLayer1.triggerProducer_cfi")
process.patTrigger.triggerResults = cms.InputTag("TriggerResults","","HLT")
process.patTrigger.triggerEvent = cms.InputTag("hltTriggerSummaryAOD","","HLT")

parts = []
if do_skim: parts.extend([ process.countmu, process.countvtx, ])
parts.extend([ process.gtStage2Digis, ])
if do_trigger_objects: parts.extend([ process.patTrigger ])
parts.extend([ process.triggerMaker, process.offlineBeamSpot, process.beamSpotMaker, process.MeasurementTrackerEvent, process.hitMaker, ])
process.skimpath = cms.Path(reduce(lambda x,y: x*y, parts))
