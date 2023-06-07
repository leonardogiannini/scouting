import FWCore.ParameterSet.VarParsing as VarParsing

opts     = VarParsing.VarParsing('python')
vpbool   = VarParsing.VarParsing.varType.bool
vpint    = VarParsing.VarParsing.varType.int
vpstring = VarParsing.VarParsing.varType.string

opts.register('data',    True ,         mytype = vpbool)
opts.register('era',     "2018C",       mytype = vpstring)
opts.register('output',  "output.root", mytype = vpstring)
opts.register('inputs',  "",            mytype = vpstring) # comma separated list of input files
opts.register('nevents', -1,            mytype = vpint)
opts.parseArguments()

def convert_fname(fname):
    if fname.startswith("/store") or fname.startswith("root://"): return fname
    return "file:" + fname.replace("file:","",1)

inputs = []
if opts.inputs:
    inputs = map(convert_fname,opts.inputs.split(","))

lin=list(inputs)

print(""" Running with options:data = {} era = {} output = {} inputs = {} nevents = {} """.format(opts.data,opts.era,opts.output,lin,opts.nevents))

import FWCore.ParameterSet.Config as cms
process = cms.Process('SLIM')

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('PhysicsTools.PatAlgos.producersLayer1.patCandidates_cff')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.Reconstruction_cff')
process.load('Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

gtag="130X_dataRun3_HLT_frozen_v2" #"130X_dataRun3_v2",
gtag="124X_dataRun3_HLT_v7" #from https://twiki.cern.ch/twiki/bin/view/CMSPublic/SWGuideFrontierConditions#Global_Tags_for_2022_data_taking
process.GlobalTag.globaltag = gtag
print("GTAG",process.GlobalTag.globaltag)
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(opts.nevents))

if opts.data:
    process.MessageLogger.cerr.FwkReport.reportEvery = 1
else:
    process.MessageLogger.cerr.FwkReport.reportEvery = 1


#process.source = cms.Source("PoolSource",
#    dropDescendantsOfDroppedBranches = cms.untracked.bool(True),
#    inputCommands = cms.untracked.vstring(
#        'keep *',
#        'drop *_hltScoutingTrackPacker_*_*',
#    ),
#    # replace 'myfile.root' with the source file you want to use
#    fileNames = cms.untracked.vstring(
#            'file:/home/users/legianni/scou/CMSSW_13_0_6/src/f3fa2c90-1fa6-45e1-a25b-b053c018b2f3.root'
#    )
#)

print("--")
print(lin)

process.source = cms.Source("PoolSource",
    dropDescendantsOfDroppedBranches = cms.untracked.bool(True),
    fileNames = cms.untracked.vstring(),
    inputCommands = cms.untracked.vstring(
        'keep *',
        'drop *_hltScoutingTrackPacker_*_*',
    ),
)
process.source.fileNames = lin

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
        "keep *_hltScoutingMuonPacker_*_*",
        "keep *_hltScoutingPFPacker_*_*",
        "keep *_hltScoutingPrimaryVertexPacker_*_*",
        #"keep *_hltScoutingPrimaryVertexPackerCaloMuon_*_*",
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


#FEDRawDataCollection              "hltFEDSelectorL1"          ""               "HLT"     
#double                            "hltScoutingPFPacker"       "pfMetPhi"       "HLT"     
#double                            "hltScoutingPFPacker"       "pfMetPt"        "HLT"     
#double                            "hltScoutingPFPacker"       "rho"            "HLT"     
#edm::TriggerResults               "TriggerResults"            ""               "HLT"     
#vector<Run3ScoutingElectron>      "hltScoutingEgammaPacker"   ""               "HLT"     
#vector<Run3ScoutingMuon>          "hltScoutingMuonPacker"     ""               "HLT"     
#vector<Run3ScoutingPFJet>         "hltScoutingPFPacker"       ""               "HLT"     
#vector<Run3ScoutingParticle>      "hltScoutingPFPacker"       ""               "HLT"     
#vector<Run3ScoutingPhoton>        "hltScoutingEgammaPacker"   ""               "HLT"     
#vector<Run3ScoutingTrack>         "hltScoutingTrackPacker"    ""               "HLT"     
#vector<Run3ScoutingVertex>        "hltScoutingMuonPacker"     "displacedVtx"   "HLT"     
#vector<Run3ScoutingVertex>        "hltScoutingPrimaryVertexPacker"   "primaryVtx"     "HLT"   

process.countmu = cms.EDFilter("ScoutingMuonCountFilter",
    src = cms.InputTag("hltScoutingMuonPacker"),
    minNumber = cms.uint32(2)
)

process.countvtx = cms.EDFilter("ScoutingVertexCountFilter",
    src = cms.InputTag("hltScoutingMuonPacker","displacedVtx"),
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

L1Info =[
    "L1_DoubleMu_12_5","L1_DoubleMu_15_7","L1_DoubleMu4p5er2p0_SQ_OS_Mass_Min7","L1_DoubleMu4p5er2p0_SQ_OS_Mass_7to18",
"L1_DoubleMu4_SQ_OS_dR_Max1p2","L1_DoubleMu4p5_SQ_OS_dR_Max1p2","L1_HTT200er","L1_HTT255er","L1_HTT280er","L1_HTT320er",
"L1_HTT360er","L1_HTT400er","L1_HTT450er","L1_ETT2000","L1_SingleJet180","L1_SingleJet200","L1_DoubleJet30er2p5_Mass_Min300_dEta_Max1p5",
"L1_DoubleJet30er2p5_Mass_Min330_dEta_Max1p5","L1_DoubleJet30er2p5_Mass_Min360_dEta_Max1p5","L1_SingleLooseIsoEG28er2p1",
"L1_SingleLooseIsoEG28er1p5","L1_SingleLooseIsoEG30er1p5","L1_SingleIsoEG28er2p1","L1_SingleIsoEG30er2p1",
"L1_SingleIsoEG32er2p1","L1_DoubleEG_LooseIso16_LooseIso12_er1p5","L1_DoubleEG_LooseIso18_LooseIso12_er1p5",
"L1_DoubleEG_LooseIso20_LooseIso12_er1p5","L1_DoubleEG_LooseIso22_LooseIso12_er1p5"
    ]
#L1Info =[
#    "L1_DoubleMu_12_5", "L1_DoubleMu_15_7"
#    ]

L1Info = ["L1_DoubleMu_12_5","L1_DoubleMu_15_7","L1_DoubleMu4p5er2p0_SQ_OS_Mass_Min7","L1_DoubleMu4p5er2p0_SQ_OS_Mass_7to18","L1_DoubleMu4_SQ_OS_dR_Max1p2","L1_DoubleMu4p5_SQ_OS_dR_Max1p2"]

HLTInfo = [
           ['DoubleMu3_noVtx',                   'DST_DoubleMu3_noVtx_CaloScouting_v*'],
           ['DoubleMu3_noVtx_Monitoring',        'DST_DoubleMu3_noVtx_CaloScouting_Monitoring_v*'],
           ]

HLTInfo=[[ 'DoubleMu3_noVtx', 'DST_Run3_PFScoutingPixelTracking_*']]

#DST_Run3_PFScoutingPixelTracking_v16
ii="vb"
print(list(zip(*HLTInfo)))
print(ii,list(zip(*HLTInfo))[0])
print(ii,list(zip(*HLTInfo))[1])
#quit()
do_trigger_objects = not opts.data

process.triggerMaker = cms.EDProducer("TriggerMaker",
        triggerAlias = cms.vstring(list(zip(*HLTInfo))[0]),
        triggerSelection = cms.vstring(list(zip(*HLTInfo))[1]),
        triggerConfiguration = cms.PSet(
            hltResults            = cms.InputTag('TriggerResults','','HLT'),
            l1tResults            = cms.InputTag(''),
            daqPartitions         = cms.uint32(1),
            l1tIgnoreMaskAndPrescale = cms.bool(False),
            throw                 = cms.bool(True),
            usePathStatus = cms.bool( True ),
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
        muonInputTag = cms.InputTag("hltScoutingMuonPacker"),
        dvInputTag = cms.InputTag("hltScoutingMuonPacker:displacedVtx"),
        measurementTrackerEventInputTag = cms.InputTag("MeasurementTrackerEvent"),
        )

process.beamSpotMaker = cms.EDProducer("BeamSpotMaker")

from RecoTracker.MeasurementDet.measurementTrackerEventDefault_cfi import measurementTrackerEventDefault as _measurementTrackerEventDefault
process.MeasurementTrackerEvent = _measurementTrackerEventDefault.clone()

process.load("EventFilter.L1TRawToDigi.gtStage2Digis_cfi")
process.gtStage2Digis.InputLabel = cms.InputTag( "hltFEDSelectorL1" )

process.offlineBeamSpot = cms.EDProducer("BeamSpotProducer")
#process.options = cms.untracked.PSet(
#    SkipEvent = cms.untracked.vstring('ProductNotFound')
#)

# trigger object stuff

process.load("PhysicsTools.PatAlgos.triggerLayer1.triggerProducer_cfi")
process.patTrigger.triggerResults = cms.InputTag("TriggerResults","","HLT")
process.patTrigger.triggerEvent = cms.InputTag("hltTriggerSummaryAOD","","HLT")

parts = []
if do_skim: parts.extend([ process.countmu, process.countvtx, ])
parts.extend([ process.gtStage2Digis, ])
if do_trigger_objects: parts.extend([ process.patTrigger ])
#parts.extend([ process.triggerMaker, process.offlineBeamSpot, process.beamSpotMaker, process.MeasurementTrackerEvent, process.hitMaker, ])
parts.extend([ process.offlineBeamSpot, process.beamSpotMaker, process.MeasurementTrackerEvent, process.hitMaker, ])
import functools
print(functools.reduce(lambda x,y: x*y, parts))
#what we are going to run
#countmu+countvtx+gtStage2Digis+triggerMaker+offlineBeamSpot+beamSpotMaker+MeasurementTrackerEvent+hitMaker
#process.skimpath = cms.Path(functools.reduce(lambda x,y: x*y, parts))
#+process.triggerMaker
process.skimpath = cms.Path(process.countmu+process.countvtx+process.gtStage2Digis+process.triggerMaker+process.offlineBeamSpot+process.beamSpotMaker+process.MeasurementTrackerEvent+process.hitMaker)
