# Auto generated configuration file
# using: 
# Revision: 1.19 
# Source: /local/reps/CMSSW/CMSSW/Configuration/Applications/python/ConfigBuilder.py,v 
# with command line options: Configuration/GenProduction/python/TOP-RunIIFall17wmLHEGS-00069-fragment.py --filein file:events.lhe --fileout file:output_gensim.root --mc --eventcontent RAWSIM,LHE --datatier GEN-SIM,LHE --conditions 93X_mc2017_realistic_v3 --beamspot Realistic25ns13TeVEarly2017Collision --step LHE,GEN,SIM --nThreads 1 --geometry DB:Extended --era Run2_2017 --python_filename pset_gensim.py --no_exec --customise Configuration/DataProcessing/Utils.addMonitoring --customise_commands process.RandomNumberGeneratorService.externalLHEProducer.initialSeed=int(1555458438%100) -n 20
import FWCore.ParameterSet.Config as cms
import os
import sys
import random

import FWCore.ParameterSet.VarParsing as VarParsing
opts = VarParsing.VarParsing('python')
vpbool = VarParsing.VarParsing.varType.bool
vpint = VarParsing.VarParsing.varType.int
vpstring = VarParsing.VarParsing.varType.string
opts.register('output'    , "output.root"  , mytype=vpstring)
opts.register('inputs'    , ""  , mytype=vpstring) # comma separated list of input files
opts.register('nevents'    , 1000  , mytype=vpint)
opts.parseArguments()

from Configuration.StandardSequences.Eras import eras

process = cms.Process('SIM',eras.Run2_2017)

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.GeometrySimDB_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.Generator_cff')
process.load('IOMC.EventVertexGenerators.VtxSmearedRealistic25ns13TeVEarly2017Collision_cfi')
process.load('GeneratorInterface.Core.genFilterSummary_cff')
process.load('Configuration.StandardSequences.SimIdeal_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.maxEvents = cms.untracked.PSet(
        input = cms.untracked.int32(opts.nevents)
)

# Input source
process.source = cms.Source("EmptySource")
process.options = cms.untracked.PSet(
)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    annotation = cms.untracked.string('Configuration/GenProduction/python/TOP-RunIIFall17wmLHEGS-00069-fragment.py nevts:20'),
    name = cms.untracked.string('Applications'),
    version = cms.untracked.string('$Revision: 1.19 $')
)

# Output definition

process.RAWSIMoutput = cms.OutputModule("PoolOutputModule",
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('generation_step')
    ),
    compressionAlgorithm = cms.untracked.string('LZMA'),
    compressionLevel = cms.untracked.int32(9),
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('GEN-SIM'),
        filterName = cms.untracked.string('')
    ),
    eventAutoFlushCompressedSize = cms.untracked.int32(20971520),
    fileName = cms.untracked.string('file:output_gensim.root'),
    outputCommands = process.RAWSIMEventContent.outputCommands,
    splitLevel = cms.untracked.int32(0)
)
if len(opts.output):
    process.RAWSIMoutput.fileName = "file:" + str(opts.output).strip().replace("file:","")

# Additional output definition

# Other statements
process.XMLFromDBSource.label = cms.string("Extended")
process.genstepfilter.triggerConditions=cms.vstring("generation_step")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, '93X_mc2017_realistic_v3', '')


process.RandomNumberGeneratorService.generator.initialSeed      = int(random.randint(1,90000))
process.RandomNumberGeneratorService.VtxSmeared.initialSeed     = 2+ 1+int(random.randint(1,90000))
process.RandomNumberGeneratorService.g4SimHits.initialSeed      =  5+int(random.randint(1,90000))
process.RandomNumberGeneratorService.mix.initialSeed            =  7+int(random.randint(1,90000))


process.generator = cms.EDFilter("Pythia8GeneratorFilter",
    PythiaParameters = cms.PSet(
        parameterSets = cms.vstring('pythia8CommonSettings',
            'pythia8CP5Settings',
            'processParameters'),

        processParameters = cms.vstring(
                'HardQCD:hardbbbar  = on',
                # 'PhaseSpace:pTHatMin = 4',  
                # '999999:all = GeneralResonance void 0 0 0 2.0 0.001 0.0 0.0 50.0',
                # '999999:addChannel = 1 1.0 101 13 -13',
                # '521:oneChannel = 1 1.0 0 999999 321', 
                # '511:oneChannel = 1 1.0 0 999999 311'  
                'PhaseSpace:pTHatMin = 4',  
                '6000211:all = GeneralResonance void 0 0 0 2.0 0.001 0.0 0.0 20.0',
                '6000211:addChannel = 1 1.0 101 13 -13',
                '521:oneChannel = 1 1.0 0 6000211 321', 
                '511:oneChannel = 1 1.0 0 6000211 311'  
        ),
       


        pythia8CP5Settings = cms.vstring('Tune:pp 14',
            'Tune:ee 7',
            'MultipartonInteractions:ecmPow=0.03344',
            'PDF:pSet=20',
            'MultipartonInteractions:bProfile=2',
            'MultipartonInteractions:pT0Ref=1.41',
            'MultipartonInteractions:coreRadius=0.7634',
            'MultipartonInteractions:coreFraction=0.63',
            'ColourReconnection:range=5.176',
            'SigmaTotal:zeroAXB=off',
            'SpaceShower:alphaSorder=2',
            'SpaceShower:alphaSvalue=0.118',
            'SigmaProcess:alphaSvalue=0.118',
            'SigmaProcess:alphaSorder=2',
            'MultipartonInteractions:alphaSvalue=0.118',
            'MultipartonInteractions:alphaSorder=2',
            'TimeShower:alphaSorder=2',
            'TimeShower:alphaSvalue=0.118'),

        pythia8CommonSettings = cms.vstring('Tune:preferLHAPDF = 2',
            'Main:timesAllowErrors = 10000',
            'Check:epTolErr = 0.01',
            'Beams:setProductionScalesFromLHEF = off',
            'SLHA:keepSM = on',
            'SLHA:minMassSM = 1000.',
            'ParticleDecays:limitTau0 = on',
            'ParticleDecays:tau0Max = 1000',
            'ParticleDecays:allowPhotonRadiation = on')
    ),
    comEnergy = cms.double(13000),
    crossSection = cms.untracked.double(1),
    filterEfficiency = cms.untracked.double(-1),
    maxEventsToPrint = cms.untracked.int32(0),
    pythiaHepMCVerbosity = cms.untracked.bool(False),
    pythiaPylistVerbosity = cms.untracked.int32(0)
)

process.mugenfilter = cms.EDFilter("PythiaDauFilter",
                           MinPt = cms.untracked.double(3.0),
                           MinEta = cms.untracked.double(-2.4),
                           MaxEta = cms.untracked.double(2.4),
                           ParticleID = cms.untracked.int32(6000211),
                           DaughterIDs = cms.untracked.vint32(-13,13),
                           NumberDaughters = cms.untracked.int32(2),
)



# Path and EndPath definitions
process.generation_step = cms.Path(process.pgen)
process.simulation_step = cms.Path(process.psim)
process.genfiltersummary_step = cms.EndPath(process.genFilterSummary)
process.endjob_step = cms.EndPath(process.endOfProcess)
process.RAWSIMoutput_step = cms.EndPath(process.RAWSIMoutput)

process.options.numberOfStreams=cms.untracked.uint32(0)

# Schedule definition
process.schedule = cms.Schedule(process.generation_step,process.genfiltersummary_step,process.simulation_step,process.endjob_step,process.RAWSIMoutput_step)
from PhysicsTools.PatAlgos.tools.helpers import associatePatAlgosToolsTask
associatePatAlgosToolsTask(process)
# filter all path with the production filter sequence
for path in process.paths:
    getattr(process,path)._seq = process.generator * process.mugenfilter * getattr(process,path)._seq 

# customisation of the process.

# Automatic addition of the customisation function from Configuration.DataProcessing.Utils
from Configuration.DataProcessing.Utils import addMonitoring 

#call to customisation function addMonitoring imported from Configuration.DataProcessing.Utils
process = addMonitoring(process)

# End of customisation functions

# Customisation from command line

# Add early deletion of temporary data products to reduce peak memory need
from Configuration.StandardSequences.earlyDeleteSettings_cff import customiseEarlyDelete
process = customiseEarlyDelete(process)
# End adding early deletion
