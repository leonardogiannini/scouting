import numba
import numpy as np
import functools
import pandas as pd
import uproot_methods
import uproot

def set_plotting_style():
    from matplotlib import rcParams
    rcParams["font.family"] = "sans-serif"
    rcParams["font.sans-serif"] = ["Helvetica", "Arial", "Liberation Sans", "Bitstream Vera Sans", "DejaVu Sans"]
    rcParams['legend.fontsize'] = 11
    rcParams['legend.labelspacing'] = 0.2
    rcParams['hatch.linewidth'] = 0.5  # https://stackoverflow.com/questions/29549530/how-to-change-the-linewidth-of-hatch-in-matplotlib
    rcParams['axes.xmargin'] = 0.0 # rootlike, no extra padding within x axis
    rcParams['axes.labelsize'] = 'x-large'
    rcParams['axes.formatter.use_mathtext'] = True
    rcParams['legend.framealpha'] = 0.65
    rcParams['axes.labelsize'] = 'x-large'
    rcParams['axes.titlesize'] = 'large'
    rcParams['xtick.labelsize'] = 'large'
    rcParams['ytick.labelsize'] = 'large'
    rcParams['figure.subplot.hspace'] = 0.1
    rcParams['figure.subplot.wspace'] = 0.1
    rcParams['figure.subplot.right'] = 0.96
    rcParams['figure.max_open_warning'] = 0
    rcParams['figure.dpi'] = 100
    rcParams["axes.formatter.limits"] = [-5,4] # scientific notation if log(y) outside this
    
@numba.njit()
def compute_bin_1d_uniform(x, bins, overflow=False):
    n = bins.shape[0] - 1
    b_min = bins[0]
    b_max = bins[-1]
    if overflow:
        if x > b_max: return n-1
        elif x < b_min: return 0
    ibin = int(n * (x - b_min) / (b_max - b_min))
    if x < b_min or x > b_max:
        return -1
    else:
        return ibin
    
@numba.njit()
def numba_histogram(a, bins, weights=None,overflow=False):
    db = np.ediff1d(bins)
    is_uniform_binning = np.all(db-db[0]<1e-6)
    hist = np.zeros((len(bins)-1,), dtype=np.float64)
    a = a.flat
    b_min = bins[0]
    b_max = bins[-1]
    n = bins.shape[0] - 1
    if weights is None:
        weights = np.ones(len(a),dtype=np.float64)
    if is_uniform_binning:
        for i in range(len(a)):
            ibin = compute_bin_1d_uniform(a[i], bins, overflow=overflow)
            if ibin >= 0:
                hist[ibin] += weights[i]
    else:
        ibins = np.searchsorted(bins, a, side='left')
        for i in range(len(a)):
            ibin = ibins[i]
            if overflow:
                if ibin == n+1: ibin = n
                elif ibin == 0: ibin = 1
            if ibin >= 1 and ibin <= n:
                hist[ibin-1] += weights[i]
        pass
    return hist, bins

@numba.njit()
def numba_histogram2d(ax,ay, bins_x, bins_y, weights=None,overflow=False):
    db_x = np.ediff1d(bins_x)
    db_y = np.ediff1d(bins_y)
    is_uniform_binning_x = np.all(db_x-db_x[0]<1e-6)
    is_uniform_binning_y = np.all(db_y-db_y[0]<1e-6)
    hist = np.zeros((len(bins_x)-1,len(bins_y)-1), dtype=np.float64)
    ax = ax.flat
    ay = ay.flat
    b_min_x = bins_x[0]
    b_max_x = bins_x[-1]
    n_x = bins_x.shape[0] - 1
    b_min_y = bins_y[0]
    b_max_y = bins_y[-1]
    n_y = bins_y.shape[0] - 1
    if weights is None:
        weights = np.ones(len(ax),dtype=np.float64)
    if is_uniform_binning_x and is_uniform_binning_y:
        for i in range(len(ax)):
            ibin_x = compute_bin_1d_uniform(ax[i], bins_x, overflow=overflow)
            ibin_y = compute_bin_1d_uniform(ay[i], bins_y, overflow=overflow)
            if ibin_x >= 0 and ibin_y >= 0:
                hist[ibin_x,ibin_y] += weights[i]
    else:
        ibins_x = np.searchsorted(bins_x, ax, side='left')
        ibins_y = np.searchsorted(bins_y, ay, side='left')
        for i in range(len(ax)):
            ibin_x = ibins_x[i]
            ibin_y = ibins_y[i]
            if overflow:
                if ibin_x == n_x+1: ibin_x = n_x
                elif ibin_x == 0: ibin_x = 1
                if ibin_y == n_y+1: ibin_y = n_y
                elif ibin_y == 0: ibin_y = 1
            if ibin_x >= 1 and ibin_y >= 1 and ibin_x <= n_x and ibin_y <= n_y:
                hist[ibin_x-1,ibin_y-1] += weights[i]
    return hist, bins_x, bins_y

def make_profile(tobin,toreduce,edges=None,errors=True):
    from scipy.stats import binned_statistic
    yvals = binned_statistic(tobin,toreduce, 'mean', bins=edges).statistic
    yerr = yvals*0.
    if errors:
        yerr = binned_statistic(tobin,toreduce, 'std', bins=edges).statistic/binned_statistic(tobin,toreduce, 'count', bins=edges).statistic**0.5
    from yahist import Hist1D
    h = Hist1D()
    h._counts = yvals
    h._errors = yerr
    h._edges = edges
    return h


@functools.lru_cache(maxsize=256)
def get_chunking(filelist, chunksize, treename="Events", workers=12, skip_bad_files=False, xrootd=False, client=None, use_dask=False):
    """
    Return 2-tuple of
    - chunks: triplets of (filename,entrystart,entrystop) calculated with input `chunksize` and `filelist`
    - total_nevents: total event count over `filelist`
    """
    import uproot
    from tqdm.auto import tqdm
    import concurrent.futures

    if xrootd:
        temp = []
        for fname in filelist:
            if fname.startswith("/hadoop/cms"):
                temp.append(fname.replace("/hadoop/cms","root://redirector.t2.ucsd.edu/"))
            else:
                temp.append(fname.replace("/store/","root://cmsxrootd.fnal.gov//store/"))
        filelist = temp

    chunksize = int(chunksize)
    chunks = []
    nevents = 0

    if use_dask:
        if not client:
            from dask.distributed import get_client
            client = get_client()
        def numentries(fname):
            import uproot
            try:
                return (fname,uproot.numentries(fname,treename))
            except:
                return (fname,-1)
        info = client.gather(client.map(numentries, filelist))
        for fn, nentries in info:
            if nentries < 0:
                if skip_bad_files:
                    print("Skipping bad file: {}".format(fn))
                    continue
                else: raise RuntimeError("Bad file: {}".format(fn))
            nevents += nentries
            for index in range(nentries // chunksize + 1):
                chunks.append((fn, chunksize*index, min(chunksize*(index+1), nentries)))
    else:
        if skip_bad_files:
            # slightly slower (serial loop), but can skip bad files
            for fname in tqdm(filelist):
                try:
                    items = uproot.numentries(fname, treename, total=False).items()
                except (IndexError, ValueError) as e:
                    print("Skipping bad file", fname)
                    continue
                for fn, nentries in items:
                    nevents += nentries
                    for index in range(nentries // chunksize + 1):
                        chunks.append((fn, chunksize*index, min(chunksize*(index+1), nentries)))
        else:
            executor = None if len(filelist) < 5 else concurrent.futures.ThreadPoolExecutor(min(workers, len(filelist)))
            for fn, nentries in uproot.numentries(filelist, treename, total=False, executor=executor).items():
                nevents += nentries
                for index in range(nentries // chunksize + 1):
                    chunks.append((fn, chunksize*index, min(chunksize*(index+1), nentries)))

    return chunks, nevents

def hist2d_dask(df, x, y, bins, method=1):
    """
    np.histogram2d from dask dataframe.
    3 methods to test:
        1 - map over delayed objects from dataframe
        2 - map over partitions from dataframe
        3 - map over partitions from dataframe using numba

    Examples
    --------
    >>> bins = [np.linspace(-15,15,200),np.linspace(-15,15,200)]
    >>> hist2d_dask(df, x="DV_x", y="DV_y", bins=bins).compute()
    """
    if method == 1:
        from dask import delayed
        @delayed
        def f(df, bins):
            return np.histogram2d(df.eval(x),df.eval(y), bins=bins)[0]
        bins = delayed(bins)
        agg = delayed(sum)
        return agg(f(obj, bins) for obj in df.to_delayed())
    if method == 2:
        def f(df, bins):
            return np.histogram2d(df.eval(x), df.eval(y), bins=bins)[0].flatten()[np.newaxis]
        # FIXME, probably better way to do this
        # can specify a (x, y bins) or just 1d bins to be used for x and y
        # need to reshape differently depending on case
        if len(bins) == 2: 
            shape = [bins[0].shape[0]-1, bins[1].shape[0]-1]
        else:
            shape = [bins.shape[0]-1, bins.shape[0]-1]
        return df.map_partitions(f, bins).sum(axis=0).reshape(shape)
    if method == 3:
        def f(df, bins):
            if len(bins) == 2:
                binsx, binsy = bins
            else:
                binsx, binsy = bins, bins
            return numba_histogram2d(df.eval(x).values, df.eval(y).values, binsx, binsy, None, False)[0].flatten()[np.newaxis]
        # FIXME, probably better way to do this
        # can specify a (x, y bins) or just 1d bins to be used for x and y
        # need to reshape differently depending on case
        if len(bins) == 2: 
            shape = [bins[0].shape[0]-1, bins[1].shape[0]-1]
        else:
            shape = [bins.shape[0]-1, bins.shape[0]-1]
        return df.map_partitions(f, bins).sum(axis=0).reshape(shape)

def get_geometry_df(fname):
    """
    Get pixel geometry from inputs made in `geometry/`
    """
    import uproot
    import numpy as np
    f = uproot.open(fname)
    t = f["idToGeo"]
    df = t.pandas.df(branches=["shape","translation","matrix"],flatten=False)
    df["translation_x"] = df["translation"].str[0]
    df["translation_y"] = df["translation"].str[1]
    df["translation_z"] = df["translation"].str[2]
    df["translation_rho"] = np.hypot(df["translation_x"],df["translation_y"])
    df = df[df["shape"].apply(lambda x:x[0])==2.]
    df["endcap"] = df.eval("abs(translation_z)>25")
    # layer 1-4 for barrel, 5,7,8 for z disks
    layers = []
    for rho, z in df[["translation_rho","translation_z"]].values:
        ilayer = -1
        if abs(z)<25:
            if 0 < rho < 5: ilayer = 1
            if 5 < rho < 9: ilayer = 2
            if 9 < rho < 14: ilayer = 3
            if 14 < rho < 25: ilayer = 4
        else:
            if 25 < abs(z) < 35: ilayer = 5
            if 35 < abs(z) < 45: ilayer = 6
            if 45 < abs(z) < 70: ilayer = 7
        layers.append(ilayer)
    df["layer"] = np.array(layers)
    df = df.query("translation_rho<18") # 4 pixel layers
    return df


def plot_overlay_bpix(ax,**kwargs):
    """
    Given an axes object, overlays 2D lines for the transverse projection of the first 3 bpix layers
    Note the hardcoded geometry path
    """
    import numpy as np
    color = kwargs.pop("color","k")
    binary_triplets = np.unpackbits(np.arange(8,dtype=np.uint8)[:,np.newaxis],1)[:,-3:].astype(int)
    step_directions = binary_triplets*2-1
    geometryfile = kwargs.pop("geometryfile","/home/users/namin/2019/scouting/repo/geometry/tracker_geometry_data2018.root")
    gdf = get_geometry_df(geometryfile)
    expand_l = kwargs.pop("expand_l",0.00)
    expand_w = kwargs.pop("expand_w",0.00)
    expand_h = kwargs.pop("expand_h",0.05)
    do_expand = (expand_h > 0) or (expand_w > 0) or (expand_h)
    for irow,entry in gdf.query("0 < translation_z < 8 and translation_rho<{}".format(kwargs.pop("maxrho",14))).iterrows():
        shape = entry["shape"][1:-1].T
        if do_expand:
            newshape = np.array(shape)
            newshape[0] += expand_l
            newshape[1] += expand_w
            newshape[2] += expand_h
            shape = newshape
        translation = entry["translation"]
        matrix = entry["matrix"].reshape(3,3)
        points = shape * step_directions
        points = np.array([np.dot(matrix,point)+translation for point in points])
        points = points[np.array([6,2,1,5,6])]
        ax.plot(points[:,0],points[:,1],color=color,**kwargs)
    return ax

def plot_overlay_pixel_rhoz(ax,**kwargs):
    """
    Given an axes object, overlays 2D lines for the rho-z projection of the pixel layers
    """
    import numpy as np
    color = kwargs.pop("color","k")
    lw = kwargs.pop("lw",1.0)
    alpha = kwargs.pop("alpha",0.015)
    binary_triplets = np.unpackbits(np.arange(8,dtype=np.uint8)[:,np.newaxis],1)[:,-3:].astype(int)
    step_directions = binary_triplets*2-1
    geometryfile = kwargs.pop("geometryfile","/home/users/namin/2019/scouting/repo/geometry/tracker_geometry_data2018.root")
    gdf = get_geometry_df(geometryfile)
    expand_l = kwargs.pop("expand_l",0.00)
    expand_w = kwargs.pop("expand_w",0.00)
    expand_h = kwargs.pop("expand_h",0.00)
    do_expand = (expand_h > 0) or (expand_w > 0) or (expand_h)
#     for irow,entry in gdf.query("0 < translation_z < 8 and translation_rho<14").iterrows():
    for irow,entry in gdf.iterrows():
        shape = entry["shape"][1:-1].T
        if do_expand:
            newshape = np.array(shape)
            newshape[0] += expand_l
            newshape[1] += expand_w
            newshape[2] += expand_h
            shape = newshape
        translation = entry["translation"]
        matrix = entry["matrix"].reshape(3,3)
        points = shape * step_directions
        points = np.array([np.dot(matrix,point)+translation for point in points])
        points = points[np.array([6,2,1,5,6])]
        rho = np.hypot(points[:,0],points[:,1])
        p1,p2=points[:,2],rho
        ax.plot(p1,p2,color=color,alpha=alpha,lw=lw,**kwargs)
    return ax

def futures_widget(futures):
    """
    Takes a list of futures and returns a jupyter widget object of squares,
    one per future, which turn green or red on success or failure
    """
    import ipywidgets
    import toolz
    def make_button(future):
        button = ipywidgets.Button(
            description='',
            disabled=False,
            button_style='', # 'success', 'info', 'warning', 'danger' or ''
            tooltip='',
            layout=ipywidgets.Layout(width='20px', height='20px'),
        )
        def callback(f):
            if f.exception():
                button.button_style = "danger"
            else:
                button.button_style = "success"
        future.add_done_callback(callback)
        return button

    items = [make_button(future) for future in futures]

    box = ipywidgets.VBox([ipywidgets.HBox(row) for row in toolz.itertoolz.partition(30, items)])
    return box

@pd.api.extensions.register_dataframe_accessor("tree")
class TreeLikeAccessor:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def draw(self, varexp, sel, bins=None, overflow=True, fast=True, label=None):
        try:
            from yahist import Hist1D
        except:
            raise Exception("Need Hist1D object from the yahist package")

        df = self._obj

        if fast:
            return Hist1D(hacky_query_eval(df, varexp, sel), bins=bins, label=label)

        weights = df.eval(sel)
        mask = np.zeros(len(df), dtype=bool)
        extra = dict()
        extra["overflow"] = overflow
        if (weights.dtype in [int, np.int32]):
            mask = weights != 0
            extra["weights"] = weights[mask]
        if (weights.dtype == bool):
            mask = weights > 0.5
            # no weights for bools
        if (weights.dtype == float):
            mask = weights != 0.
            extra["weights"] = weights[mask]
        vals = df[mask].eval(varexp)
        if bins is not None:
            # if type(bins) in [str]:
            #     raise NotImplementedError()
            extra["bins"] = bins
        return Hist1D(vals, **extra)

@pd.api.extensions.register_dataframe_accessor("vec")
class LorentzVectorAccessor:
    def __init__(self, pandas_obj):
        self._validate(pandas_obj)
        self._obj = pandas_obj

    @staticmethod
    def _validate(obj):
        missing_columns = set(["Muon1_pt","Muon1_eta","Muon1_phi","Muon2_pt","Muon2_eta","Muon2_phi"])-set(obj.columns)
        if len(missing_columns):
            raise AttributeError("Missing columns: {}".format(missing_columns))

    @property
    def mu1(self):
        LV = uproot_methods.TLorentzVectorArray.from_ptetaphim(
            self._obj["Muon1_pt"],self._obj["Muon1_eta"],self._obj["Muon1_phi"],0.10566,
        )
        return LV

    @property
    def mu2(self):
        LV = uproot_methods.TLorentzVectorArray.from_ptetaphim(
            self._obj["Muon2_pt"],self._obj["Muon2_eta"],self._obj["Muon2_phi"],0.10566,
        )
        return LV

    @property
    def dimu(self):
        return self.mu1 + self.mu2

smaller_dtypes = [
    ["dimuon_mass","float32"],
    ["dimuon_pt","float32"],
    ["DV_chi2prob","float32"],
    ["DV_ndof","int8"],
    ["DV_redchi2","float32"],
    ["DV_layerPixel","int8"],
    ["Muon1_charge","int8"],
    ["Muon1_excesshits","int8"],
    ["Muon1_m","float32"],
    ["Muon1_mass","float32"],
    ["Muon1_nExcessPixelHits","int8"],
    ["Muon1_nExpectedPixelHits","int8"],
    ["Muon1_nMatchedStations","int8"],
    ["Muon1_nTrackerLayersWithMeasurement","int8"],
    ["Muon1_nValidMuonHits","int8"],
    ["Muon1_nValidPixelHits","int8"],
    ["Muon1_nValidStripHits","int8"],
    ["Muon2_charge","int8"],
    ["Muon2_excesshits","int8"],
    ["Muon2_m","float32"],
    ["Muon2_mass","float32"],
    ["Muon2_nExcessPixelHits","int8"],
    ["Muon2_nExpectedPixelHits","int8"],
    ["Muon2_nMatchedStations","int8"],
    ["Muon2_nTrackerLayersWithMeasurement","int8"],
    ["Muon2_nValidMuonHits","int8"],
    ["Muon2_nValidPixelHits","int8"],
    ["Muon2_nValidStripHits","int8"],
    ["categ","int8"],
    ["luminosityBlock","int32"],
    ["nDV","int8"],
    ["nDV_good","int8"],
    ["nDV_raw","int8"],
    ["nGenMuon","int8"],
    ["nGenPart","int16"],
    ["nJet","int8"],
    ["nMuon","int8"],
    ["nMuon_good","int8"],
    ["nMuon_raw","int8"],
    ["nPV","int8"],
    ["nPVM","int8"],
    ["run","int32"],
]


def make_df(
    path,
    branches = ["dimuon_mass", "pass_*"],
    cut = "pass_baseline_iso",
    chunksize = 500e3,
    xrootd = False,
    persist = True,
    client = None,
    func = None,
    partition_size = None,
    npartitions = None,
    use_dask = False,
    skip_bad_files = False,
    nthreads = 6,
    progress = True,
):
    """
    Returns dataframe from input ROOT files containing given branches
    for events passing a given cut. If `use_dask=True`, returns dask dataframe.

    path: file path(s) or glob string(s)
    branches: list of branches/glob strings/regex for branches to read
    cut: selection string input to `df.query()`
    chunksize: events per task
    xrootd: use xrootd for input files
    persist: whether to return persisted dask dataframe or not
    func: override reading function (must read a chunk and return a DataFrame)
    partition_size: if not None, passed into df.repartition() before persisting - NOTE might be duplicating the reading. use `npartitions`.
    npartitions: if not None, passed into df.repartition() before persisting
    skip_bad_files: whether to skip bad files according to failure of uproot.numentries
    nthreads: number of ThreadPoolExecutor threads when not using dask (default 6)
    progress: show progress bar
    """
    import dask.dataframe as dd
    from dask import delayed
    import uproot
    import pandas as pd
    from tqdm.auto import tqdm
    import concurrent.futures
    if isinstance(path, (str, bytes)):
        paths = uproot.tree._filename_explode(path)
    else:
        paths = [y for x in path for y in uproot.tree._filename_explode(x)]

    if not func:
        def func(fname, entrystart = None, entrystop = None):
            t = uproot.open(fname)["Events"]
            arrs = t.arrays(
                branches,
                outputtype = dict,
                namedecode = "ascii",
                entrystart = entrystart,
                entrystop = entrystop,
            )
            sel = slice(None,None)
            df = pd.DataFrame()
            for k in arrs.keys():
                v = arrs[k][sel]
                is_jagged = ("Jagged" in str(type(v)))
                if is_jagged:
                    if k.startswith("DV_"):
                        df[k] = v[:,0]
                    elif k.startswith("PVM_"):
                        df[k] = v[:,0]
                    elif k.startswith("Muon_"):
                        df[k.replace("Muon_","Muon1_")] = v[:,0]
                        df[k.replace("Muon_","Muon2_")] = v[:,1]
                    elif k.startswith("Jet_"):
                        if "ht" not in df.columns:
                            df["ht"] = v.sum()
                        v = v.pad(2).fillna(-1)
                        df[k.replace("Jet_","Jet1_")] = v[:,0]
                        df[k.replace("Jet_","Jet2_")] = v[:,1]
                    else:
                        df[k] = v
                else:
                    df[k] = v
            for name,dtype in smaller_dtypes:
                if name not in df.columns: continue
                df[name] = df[name].astype(dtype, copy=False)
            df = df.query(cut)
            return df

    chunks, total_events = get_chunking(tuple(paths), chunksize, client=client, xrootd=xrootd, use_dask=use_dask, skip_bad_files=skip_bad_files)

    smallchunk_nevents = int(chunks[0][1] + (chunks[0][2]-chunks[0][1])//10)
    smallchunk = (chunks[0][0], chunks[0][1], smallchunk_nevents)
    meta = func(*smallchunk)

    if not use_dask:
        smallchunk_mb = meta.memory_usage().sum()/1e6
        estimated_mb = smallchunk_mb * total_events / smallchunk_nevents
        if estimated_mb > 15e3:
            raise RuntimeError("This dataframe would take approx. {:.1f}GB of RAM. Reduce the input size.".format(estimated_mb*1e-3))

        executor = concurrent.futures.ThreadPoolExecutor(nthreads)
        futures = [executor.submit(func, *chunk) for chunk in chunks]
        def wrapper(x):
            if progress:
                return tqdm(x)
            return x
        ddf = pd.concat((future.result() for future in wrapper(futures)), sort=True, ignore_index=True, copy=False)
        del executor
    else:
        delayed_func = delayed(func)
        ddf = dd.from_delayed((delayed_func(*chunk) for chunk in chunks), meta=meta).reset_index(drop=True)
        if partition_size:
            ddf = ddf.repartition(partition_size=partition_size)
        if npartitions:
            ddf = ddf.repartition(npartitions=npartitions)
        if persist:
            ddf = ddf.persist()

    return ddf

def dataframe_to_ttree(df, filename, treename="t", chunksize=1e6, compression=uproot.LZ4(1), progress=True):
    """
    Writes ROOT file containing one TTree with the input pandas DataFrame.

    filename: name of output file
    treename: name of output TTree
    chunksize: number of rows per basket
    compression: uproot compression object (LZ4, ZLIB, LZMA, or None)
    progress: show tqdm progress bar?
    """
    t = uproot.newtree(df.dtypes)
    with uproot.recreate(filename, compression=compression) as f:
        f[treename] = t
        chunksize = int(chunksize)
        iterable = range(0, len(df), chunksize)
        if progress:
            from tqdm.auto import tqdm
            iterable = tqdm(iterable)
        for i in iterable:
            chunk = df.iloc[i:i+chunksize]
            f[treename].extend({ k:chunk[k].values for k in chunk.columns })

def ttree_to_dataframe(filename, treename="t", branches=None, progress=True, **kwargs):
    """
    Read ROOT file containing one TTree into pandas DataFrame.
    Thin wrapper around `uproot.iterate`.

    filename: filename(s)/file pattern(s)
    treename: name of input TTree
    progress: show tqdm progress bar?
    branches: list of branches to read (default of `None` reads all)
    **kwargs: extra kwargs to pass to `uproot.iterate`
    """
    # entrysteps of None iterates by basket to match `dataframe_to_ttree`
    iterable = uproot.iterate(filename, treename, branches,
                              entrysteps=kwargs.pop("entrysteps",None), outputtype=pd.DataFrame,
                              namedecode="ascii", **kwargs
                             )
    if progress:
        from tqdm.auto import tqdm
        iterable = tqdm(iterable)
    # df = pd.concat(iterable, ignore_index=True, sort=True, copy=False)
    df = pd.concat(iterable, ignore_index=True, sort=True)
    return df

def hacky_query_eval(df, varstr, selstr="", verbose=False):
    """
    Please don't read/use. This is dangerous and stupid, kind of like 
    integrating a function by printing out a plot, coloring the area under it in red,
    faxing it to yourself, then counting red pixels to get the area.

    Basically I wanted some way to convert

        df.query("dimuon_mass > 5 and pass_baseline_iso").eval("dimuon_mass").mean()

    into

        df["dimuon_mass"][ (df["dimuon_mass"] > 5) & (df["pass_baseline_iso"]) ].mean()

    because the latter doesn't make an intermediate copy of all the columns with query(),
    and it also doesn't do jitting with numexpr. In principle, this is much faster to execute.

    Usage:

        arr = hacky_query_eval(
            df_data,
            varstr = "dimuon_mass",
            selstr = "pass_baseline_iso and 0<logabsetaphi<1.25",
        )
        print(arr.mean())
    """
    from pandas.core.computation.expr import Expr
    from pandas.core.computation.scope import Scope
    env = Scope(
        1,
        global_dict=globals(),
        local_dict=locals(),
        resolvers=[df],
        target=None,
    )
    def inject_df(s):
        """
        convert expression string like (a > 1) to (df["a"] > 1)
        so that it can be eval'd later
        """
        expr = Expr(s, env=env, parser="pandas")
        self = expr._visitor
        def visit_Name_hack(node, **kwargs):
            result = self.term_type(node.id, self.env, **kwargs)
            result._name = f'df["{result._name}"]'
            return result
        def _maybe_downcast_constants_hack(left, right):
            return left, right
        expr._visitor.visit_Name = visit_Name_hack
        expr._visitor._maybe_downcast_constants = _maybe_downcast_constants_hack
        expr.terms = expr.parse()
        return str(expr)
    varexpr = inject_df(varstr)
    toeval = f"({varexpr})"
    if selstr:
        selexpr = str(inject_df(selstr))
        toeval += f"[{selexpr}]"
    if verbose: print(f"Evaluating string: {toeval}")
    result = eval(toeval)
    return result

def query_dis(query, typ="basic", return_raw=False):
    import requests
    endpoint_url = "http://uaf-7.t2.ucsd.edu:50010/dis/serve"
    short = "short=true" if typ in ["basic", "sites"] else ""
    url = f"{endpoint_url}?type={typ}&query={query}&{short}"
    js = requests.get(url).json()
    if not return_raw:
        js = js["payload"]
    return js
