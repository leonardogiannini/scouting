import os
import sys
from tqdm import tqdm
import glob
import multiprocessing

def make_chunks(v, n):
    """Yield successive n-sized chunks from v."""
    for i in range(0, len(v), n):
        yield v[i:i + n]

def run(info):
    # cmd = "python babymaker.py {fnames} --year {year} -o {output}".format(
    cmd = "python gen_babymaker.py {fnames} --year {year} -o {output}".format(
            fnames=" ".join(info["fnames"]),
            year=info["year"],
            output=info["output"],
            )
    os.system(cmd)

# fpatt = "/hadoop/cms/store/user/namin/ProjectMetis/BToPhi_params_mphi2_ctau20mm_RAWSIM_v0/*.root"
fpatt = "/hadoop/cms/store/user/namin/ProjectMetis/HToZdZdTo2Mu2X_params_mzd8_ctau10mm_RAWSIM_v10/output_*.root"
year = 2018
output_name = "output_combined.root"

chunks = list(make_chunks(glob.glob(fpatt), 20))
pool = multiprocessing.Pool(12)

infos = []
os.system("mkdir -p paralleloutputs")
os.system("rm paralleloutputs/*.root")
for i,chunk in enumerate(chunks):
    infos.append(dict(
            fnames=chunk,
            output="paralleloutputs/output_{}.root".format(i),
            year=year,
        ))
    # print(chunk)
    # if i>3:
    #     break

for _ in tqdm(pool.imap_unordered(run, infos),total=len(infos)):
    pass
os.system("hadd -k -f {output} paralleloutputs/*.root".format(output=output_name))
