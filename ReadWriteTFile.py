import ROOT
import pandas as pd
import numpy as np

def force_extension(file_name, extension=".root"):
    # Ensure the file_name has the correct extension
    ext_index = file_name.find(".")
    if ext_index == -1:  # If there is no extension, add it
        file_name += extension
    elif file_name[ext_index:] != extension:  # If the extension is wrong, remove
        file_name = file_name[0:ext_index + 1] + extension
    return file_name


def open_tfile_as_dataframe(
        file_name_raw="/home/kanoa/Documents/UCA/Internship LHCb2026/datasets_signalMC2018_selected/signalMC2018down_Kpi.root",
        tree_name_raw="hBtohhpi0R_Tuple",
        branches=None):
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    Open the raw data TFile and stores relevant variables in a dataframe.
    The TFile must contain a TTree to be legible with this method.
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    # Open the TTree directly from the .root file into an RDataFrame

    if branches is None:
        branches = ["hB_M", "hB_PE", "hB_P", "hB_PX", "hB_PY", "hB_PZ",
                    "h1_M", "h1_PE", "h1_P", "h1_PX", "h1_PY", "h1_PZ",
                    "h2_M", "h2_PE", "h2_P", "h2_PX", "h2_PY", "h2_PZ",
                    "Pi0_M", "Pi0_PE", "Pi0_P", "Pi0_PX", "Pi0_PY", "Pi0_PZ"]

    rdf = ROOT.RDataFrame(tree_name_raw, force_extension(file_name_raw))

    # Specify desired branches from the TTree and store those into a pandas dataframe
    df_raw = pd.DataFrame({b: rdf.AsNumpy([b])[b] for b in branches})
    return df_raw


def save_dataframe_as_tfile(df: pd.DataFrame, file_name, tree_name, file_location="", tree_title=None, keys=None,
                            branch_name_suffix=""):
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    Store the desired Series of a provided DataFrame (as indicated by keys) as branches of a TTree.
    The TTree then gets saved as a TFile.

    If file_name has an invalid or missing file extension, it will be correctly replaced by ".root"
    If file_location is changed from default, the file will be written to that folder
    If tree_title=None the title will be the same as the tree name
    If keys=None then all Series of df will be written into branches of the TTree

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    # Mapping from numpy dtype to ROOT leaf type descriptor
    np_to_root_type = {
        np.dtype('float32'): 'F',
        np.dtype('float64'): 'D',
        np.dtype('int32'): 'I',
        np.dtype('int64'): 'L',
        np.dtype('uint32'): 'i',
        np.dtype('uint64'): 'l',
        np.dtype('bool'): 'O',
    }

    # If no tree title provided, make the title the same as the tree name
    if tree_title is None:
        tree_title = tree_name

    # If no list of keys is specified, get all the keys
    if keys is None:
        keys = list(df.keys())  # Cast the pandas index object into a list of strings

    # Instantiate TTree and TFile
    ttree = ROOT.TTree(tree_name, tree_title)
    tfile = ROOT.TFile(file_location + force_extension(file_name), "RECREATE")

    # Fill branches of TTree with columns of df

    buffers = {}       # First, create all branches. Otherwise the first branch gets corrupted somehow
    for k in keys:
        dtype = df[k].dtype    # Get the type of data stored in current column of df
        root_type = np_to_root_type.get(dtype, 'D')  # .Branch often mistakes the dtype, so manually determine it
        buf = np.zeros(1, dtype=dtype)  # Make buf the same datatype as corresponding df column
        # Create the k branch of ttree
        ttree.Branch(k, buf, f"{k}{branch_name_suffix}/{root_type}")    # Note manual use of latex /{root_type}
        buffers[k] = buf

    # Then fill all branches simultaneously, one row at a time
    n_events = len(df[keys[0]])   # DataFrames must have equal length columns so can use length of 0th for all
    for i in range(n_events):
        for k in keys:
            buffers[k][0] = df[k].iloc[i]
        ttree.Fill()  # One Fill() per event, filling each branch simultaneously

    ttree.Write()
    tfile.Close()
