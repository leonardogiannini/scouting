#!/bin/bash

OUTPUTDIR=$1
OUTPUTNAME=$2
INPUTFILENAMES=$3
IFILE=$4
CMSSWVERSION=$5
SCRAMARCH=$6

# Make sure OUTPUTNAME doesn't have .root since we add it manually
OUTPUTNAME=$(echo $OUTPUTNAME | sed 's/\.root//')

export SCRAM_ARCH=${SCRAMARCH}

function getjobad {
    grep -i "^$1" "$_CONDOR_JOB_AD" | cut -d= -f2- | xargs echo
}

function setup_chirp {
    if [ -e ./condor_chirp ]; then
    # Note, in the home directory
        mkdir chirpdir
        mv condor_chirp chirpdir/
        export PATH="$PATH:$(pwd)/chirpdir"
        echo "[chirp] Found and put condor_chirp into $(pwd)/chirpdir"
    elif [ -e /usr/libexec/condor/condor_chirp ]; then
        export PATH="$PATH:/usr/libexec/condor"
        echo "[chirp] Found condor_chirp in /usr/libexec/condor"
    else
        echo "[chirp] No condor_chirp :("
    fi
}

function chirp {
    # Note, $1 (the classad name) must start with Chirp
    condor_chirp set_job_attr_delayed $1 $2
    ret=$?
    echo "[chirp] Chirped $1 => $2 with exit code $ret"
}

function stageout {
    COPY_SRC=$1
    COPY_DEST=$2
    retries=0
    COPY_STATUS=1
    until [ $retries -ge 3 ]
    do
        echo "Stageout attempt $((retries+1)): env -i X509_USER_PROXY=${X509_USER_PROXY} gfal-copy -p -f -t 7200 --verbose --checksum ADLER32 ${COPY_SRC} ${COPY_DEST}"
        env -i X509_USER_PROXY=${X509_USER_PROXY} gfal-copy -p -f -t 7200 --verbose --checksum ADLER32 ${COPY_SRC} ${COPY_DEST}
        COPY_STATUS=$?
        if [ $COPY_STATUS -ne 0 ]; then
            echo "Failed stageout attempt $((retries+1))"
        else
            echo "Successful stageout with $retries retries"
            break
        fi
        retries=$[$retries+1]
        echo "Sleeping for 30m"
        sleep 30m
    done
    if [ $COPY_STATUS -ne 0 ]; then
        echo "Removing output file because gfal-copy crashed with code $COPY_STATUS"
        env -i X509_USER_PROXY=${X509_USER_PROXY} gfal-rm --verbose ${COPY_DEST}
        REMOVE_STATUS=$?
        if [ $REMOVE_STATUS -ne 0 ]; then
            echo "Uhh, gfal-copy crashed and then the gfal-rm also crashed with code $REMOVE_STATUS"
            echo "You probably have a corrupt file sitting on hadoop now."
            exit 1
        fi
    fi
}

function setup_environment {
    if [ -r "$OSGVO_CMSSW_Path"/cmsset_default.sh ]; then
        echo "sourcing environment: source $OSGVO_CMSSW_Path/cmsset_default.sh"
        source "$OSGVO_CMSSW_Path"/cmsset_default.sh
    elif [ -r "$OSG_APP"/cmssoft/cms/cmsset_default.sh ]; then
        echo "sourcing environment: source $OSG_APP/cmssoft/cms/cmsset_default.sh"
        source "$OSG_APP"/cmssoft/cms/cmsset_default.sh
    elif [ -r /cvmfs/cms.cern.ch/cmsset_default.sh ]; then
        echo "sourcing environment: source /cvmfs/cms.cern.ch/cmsset_default.sh"
        source /cvmfs/cms.cern.ch/cmsset_default.sh
    else
        echo "ERROR! Couldn't find $OSGVO_CMSSW_Path/cmsset_default.sh or /cvmfs/cms.cern.ch/cmsset_default.sh or $OSG_APP/cmssoft/cms/cmsset_default.sh"
        exit 1
    fi
}

function setup_cmssw {
  CMSSW=$1
  export SCRAM_ARCH=$2
  scram p CMSSW $CMSSW
  cd $CMSSW
  eval $(scramv1 runtime -sh)
  cd -

  # # For Purdue, NotreDame, and maybe Nebraska
  # if [ ! -e $CMS_PATH/SITECONF/local/JobConfig/site-local-config.xml ] ; then
  #     echo "$CMS_PATH/SITECONF/local/JobConfig/site-local-config.xml does not exist!"
  #     if [ -e $CMS_PATH/SITECONF/$GLIDEIN_CMSSite/JobConfig/site-local-config.xml ] ; then
  #         echo "But $CMS_PATH/SITECONF/$GLIDEIN_CMSSite/JobConfig/site-local-config.xml does exist. Copying it locally."
  #         mkdir -p ${CMSSW_BASE}/test/SITECONF/local/JobConfig
  #         cp $CMS_PATH/SITECONF/$GLIDEIN_CMSSite/JobConfig/site-local-config.xml ${CMSSW_BASE}/test/SITECONF/local/JobConfig/site-local-config.xml
  #         export CMS_PATH=${CMSSW_BASE}/test
  #     fi
  # fi

}

function edit_gridpack {
    # extract gridpack in temp dir
    # overwrite 2 parameters
    # copy back the new one, overwriting the original
    filename="$1"
    mass="$2" # GeV
    mass=$(echo $mass | sed 's/p/./')
    ctau="$3" # mm
    bname=$(basename $filename)
    pushd .
    tempdir=$(mktemp -d)
    echo $tempdir
    cp $filename $tempdir
    cd $tempdir
    ls -lrth
    tar xf $bname
    sed -i 's/MZprime=\w\+/MZprime='"$mass"'/' JHUGen.input
    sed -i 's/ctauVprime=\w\+/ctauVprime='"$ctau"'/' JHUGen.input
    rm $bname
    tar czf $bname *
    popd
    cp $tempdir/$bname $filename
}


function edit_psets {
    gridpack="$1"
    seed=$2
    nevents=$3
    absgridpack=$(readlink -f "$gridpack")

    # gensim
    echo "process.RandomNumberGeneratorService.externalLHEProducer.initialSeed = $seed" >> $gensimcfg
    echo "process.externalLHEProducer.args = [\"$absgridpack\"]" >> $gensimcfg
    echo "process.externalLHEProducer.nEvents = $nevents" >> $gensimcfg
    echo "process.maxEvents.input = $nevents" >> $gensimcfg
    echo "process.source.firstLuminosityBlock = cms.untracked.uint32($seed)" >> $gensimcfg
    echo "process.RAWSIMoutput.fileName = \"file:output_gensim.root\"" >> $gensimcfg

    # rawsim
    echo "process.maxEvents.input = $nevents" >> $rawsimcfg
    echo "process.source.fileNames = [\"file:output_gensim.root\"]" >> $rawsimcfg
    echo "process.PREMIXRAWoutput.fileName = \"file:output_rawsim.root\"" >> $rawsimcfg

    # aodsim
    echo "process.maxEvents.input = $nevents" >> $aodsimcfg
    echo "process.source.fileNames = [\"file:output_rawsim.root\"]" >> $aodsimcfg
    echo "process.AODSIMoutput.fileName = \"file:output_aodsim.root\"" >> $aodsimcfg

    # slimmer
    echo "process.maxEvents.input = $nevents" >> $slimmercfg
    echo "process.source.fileNames = [\"file:output_rawsim.root\"]" >> $slimmercfg
    echo "process.out.fileName = \"file:output.root\"" >> $slimmercfg

}

function setup_slimmer {
    pushd .
    cp -rp Scouting/ $CMSSW_BASE/src
    cd $CMSSW_BASE/src
    scram b -j1
    popd
}



echo -e "\n--- begin header output ---\n" #                     <----- section division
echo "OUTPUTDIR: $OUTPUTDIR"
echo "OUTPUTNAME: $OUTPUTNAME"
echo "INPUTFILENAMES: $INPUTFILENAMES"
echo "IFILE: $IFILE"
echo "CMSSWVERSION: $CMSSWVERSION"
echo "SCRAMARCH: $SCRAMARCH"

echo "GLIDEIN_CMSSite: $GLIDEIN_CMSSite"
echo "hostname: $(hostname)"
echo "uname -a: $(uname -a)"
echo "time: $(date +%s)"
echo "args: $@"
echo "tag: $(getjobad tag)"
echo "taskname: $(getjobad taskname)"

YEAR=$(getjobad param_year)
MASS=$(getjobad param_mass)
CTAU=$(getjobad param_ctau)
NEVENTS=$(getjobad param_nevents)
echo "MASS: $MASS"
echo "CTAU: $CTAU"
echo "YEAR: $YEAR"

echo -e "\n--- end header output ---\n" #                       <----- section division


gridpack="gridpacks/gridpack.tar.gz"
gensimcfg="psets/$YEAR/gensim_hzdzd_cfg.py"
rawsimcfg="psets/$YEAR/rawsim_cfg.py"
aodsimcfg="psets/$YEAR/aodsim_cfg.py"
slimmercfg="psets/$YEAR/slimmer_cfg.py"

setup_chirp
setup_environment

# Make temporary directory to keep original dir clean
# Go inside and extract the package tarball
mkdir temp
cd temp
cp ../*.gz .
tar xf *.gz

edit_gridpack $gridpack $MASS $CTAU
edit_psets $gridpack $IFILE $NEVENTS

echo "before running: ls -lrth"
ls -lrth

echo -e "\n--- begin running ---\n" #                           <----- section division

chirp ChirpMetisStatus "before_cmsRun"

if [[ "$YEAR" == "2017" ]]; then
    CMSSW1=CMSSW_9_3_18
    CMSSW2=CMSSW_9_4_7
    SCRAM_ARCH=slc6_amd64_gcc630 
elif [[ "$YEAR" == "2018" ]]; then
    CMSSW1=CMSSW_10_2_3
    CMSSW2=CMSSW_10_2_5
    SCRAM_ARCH=slc6_amd64_gcc700 
else
    echo "YEAR NOT SPECIFIED"
fi

echo $CMSSW1
echo $CMSSW2
echo $SCRAM_ARCH

setup_cmssw $CMSSW1 $SCRAM_ARCH
cmsRun $gensimcfg
setup_cmssw $CMSSW2 $SCRAM_ARCH 
cmsRun $rawsimcfg
cmsRun $aodsimcfg
setup_slimmer
cmsRun $slimmercfg
CMSRUN_STATUS=$?

chirp ChirpMetisStatus "after_cmsRun"

echo "after running: ls -lrth"
ls -lrth

if [[ $CMSRUN_STATUS != 0 ]]; then
    echo "Removing output file because cmsRun crashed with exit code $?"
    rm ${OUTPUTNAME}.root
    exit 1
fi

echo -e "\n--- end running ---\n" #                             <----- section division

echo -e "\n--- begin copying output ---\n" #                    <----- section division

echo "Sending output file $OUTPUTNAME.root"

if [ ! -e "$OUTPUTNAME.root" ]; then
    echo "ERROR! Output $OUTPUTNAME.root doesn't exist"
    exit 1
fi

echo "time before copy: $(date +%s)"
chirp ChirpMetisStatus "before_copy"


COPY_SRC="file://`pwd`/output_aodsim.root"
OUTPUTDIRSTORE=$(echo $OUTPUTDIR | sed "s#^/hadoop/cms/store#/store#")
COPY_DEST="davs://redirector.t2.ucsd.edu:1094${OUTPUTDIRSTORE}/aodsim/output_${IFILE}.root"
stageout $COPY_SRC $COPY_DEST

COPY_SRC="file://`pwd`/${OUTPUTNAME}.root"
OUTPUTDIRSTORE=$(echo $OUTPUTDIR | sed "s#^/hadoop/cms/store#/store#")
COPY_DEST="davs://redirector.t2.ucsd.edu:1094${OUTPUTDIRSTORE}/${OUTPUTNAME}_${IFILE}.root"
stageout $COPY_SRC $COPY_DEST

echo -e "\n--- end copying output ---\n" #                      <----- section division

echo "time at end: $(date +%s)"

chirp ChirpMetisStatus "done"

