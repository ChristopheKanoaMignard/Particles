#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 12 08:43:26 2026

@author: kanoa
"""

"""
    NEW USERS: should ctrl-f search for the 
"""

import ROOT
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from uncertainties import ufloat, umath
from uncertainties import unumpy as unp


# Control where images are saved to
loc_img = "/home/kanoa/Documents/UCA/Internship LHCb2026/Temp/"

# Cut df_master_data and _mc on defined window to get subsets
B_MASS = 5279
# Signal window is 5279 +- 100MeV
THRESH_SIG_L = B_MASS - 100 
THRESH_SIG_U = B_MASS + 100
WIDTH_SIG = THRESH_SIG_U-THRESH_SIG_L
# Lower sideband window is 4750-5150
THRESH_LSB_L = 4750
THRESH_LSB_U = 5150
WIDTH_LSB = THRESH_LSB_U-THRESH_LSB_L
# Upper sideband window is 5600-6500
THRESH_USB_L = 5600
THRESH_USB_U = 6500
WIDTH_USB = THRESH_USB_U-THRESH_USB_L


""" --------------------------------------------------------------------------
    Define functions used for various calculations 
---------------------------------------------------------------------------"""

def boost_to_B_rest_frame(df, debug=False):
    """
    * Boost hB, Pi0, h1, h2 into the rest frame of the B meson.
    * Returns the dataframe with new columns for the boosted 4-momenta,
      plus a list of the boost vectors beta_x,y,z and gamma.
      Seen recommendation df should be returned to avoid memory issues, 
      but it is technically unnecessarys as the new df columns are filled in-function.
    * If debug=True, then print statements will indicate the following:
      NaN inputs, NaN outputs, or an unphysical (|beta|>=1) boost.
      
    * Note the boost uses the pdg mass for the Lorentz factors not the reconstructed mass.
      This is necessary to prevent entering the rest frame of the center of mass.
    """

    results = {
        "Pi0": {"E": [], "PX": [], "PY": [], "PZ": []},
        "h1":  {"E": [], "PX": [], "PY": [], "PZ": []},
        "h2":  {"E": [], "PX": [], "PY": [], "PZ": []},
        "hB":  {"E": [], "PX": [], "PY": [], "PZ": []},
    }

    # Can change these to use keys for MC true variables
    input_cols = [
        "hB_pi0constPVconst_PX", "hB_pi0constPVconst_PY", "hB_pi0constPVconst_PZ",
        "hB_Pi0_pi0constPVconst_PX", "hB_Pi0_pi0constPVconst_PY",
        "hB_Pi0_pi0constPVconst_PZ", "hB_Pi0_pi0constPVconst_E",
        "hB_h1_pi0constPVconst_PX", "hB_h1_pi0constPVconst_PY",
        "hB_h1_pi0constPVconst_PZ", "hB_h1_pi0constPVconst_E",
        "hB_h2_pi0constPVconst_PX", "hB_h2_pi0constPVconst_PY",
        "hB_h2_pi0constPVconst_PZ", "hB_h2_pi0constPVconst_E",
    ]

    boost_vectors = []
    nan_input_events = []
    unphysical_boost_events = []
    nan_output_events = []

    for idx, row in df.iterrows():
        if debug:
            # Check 1: NaN already present in the inputs for this row
            if row[input_cols].isna().any():
                bad_cols = row[input_cols].index[row[input_cols].isna()].tolist()
                nan_input_events.append((idx, bad_cols))
                # fill with NaN placeholders and skip this event's calculation
                for label in results:
                    for comp in results[label]:
                        results[label][comp].append(np.nan)
                boost_vectors.append([np.nan, np.nan, np.nan, np.nan])
                continue

        # Set B meson 4-vector (defines the boost), using PDG mass
        px = row["hB_pi0constPVconst_PX"]
        py = row["hB_pi0constPVconst_PY"]
        pz = row["hB_pi0constPVconst_PZ"]
        E_pdg = np.sqrt(B_MASS**2 + px**2 + py**2 + pz**2)
        hB = ROOT.TLorentzVector() # Create Lorentz variable
        hB.SetPxPyPzE(px, py, pz, E_pdg) # Fill Lorentz variable

        # Check 2: is the resulting boost physical?
        beta_mag = hB.Beta()  # |p|/E, should be < 1
        if not np.isfinite(beta_mag) or beta_mag >= 1.0:
            unphysical_boost_events.append((idx, beta_mag))
            for label in results:
                for comp in results[label]:
                    results[label][comp].append(np.nan)
            boost_vectors.append([np.nan, np.nan, np.nan, np.nan])
            continue

        Pi0 = ROOT.TLorentzVector()
        Pi0.SetPxPyPzE(
            row["hB_Pi0_pi0constPVconst_PX"],
            row["hB_Pi0_pi0constPVconst_PY"],
            row["hB_Pi0_pi0constPVconst_PZ"],
            row["hB_Pi0_pi0constPVconst_E"],
        )
        h1 = ROOT.TLorentzVector()
        h1.SetPxPyPzE(
            row["hB_h1_pi0constPVconst_PX"],
            row["hB_h1_pi0constPVconst_PY"],
            row["hB_h1_pi0constPVconst_PZ"],
            row["hB_h1_pi0constPVconst_E"],
        )
        h2 = ROOT.TLorentzVector()
        h2.SetPxPyPzE(
            row["hB_h2_pi0constPVconst_PX"],
            row["hB_h2_pi0constPVconst_PY"],
            row["hB_h2_pi0constPVconst_PZ"],
            row["hB_h2_pi0constPVconst_E"],
        )

        boost_vec = -hB.BoostVector()
        gamma = hB.Gamma()
        boost_vectors.append([boost_vec.X(), boost_vec.Y(), boost_vec.Z(), gamma])

        Pi0.Boost(boost_vec)
        h1.Boost(boost_vec)
        h2.Boost(boost_vec)
        hB.Boost(boost_vec)
        if True: # This is sort of a debug, but it is also a vital component
            # Check 3: did the boost produce NaN/inf in the outputs?
            row_is_nan = False
            for label, vec in [("Pi0", Pi0), ("h1", h1), ("h2", h2), ("hB", hB)]:
                vals = [vec.E(), vec.Px(), vec.Py(), vec.Pz()]
                if not all(np.isfinite(v) for v in vals):
                    row_is_nan = True
                results[label]["E"].append(vec.E())
                results[label]["PX"].append(vec.Px())
                results[label]["PY"].append(vec.Py())
                results[label]["PZ"].append(vec.Pz())
            if row_is_nan:
                nan_output_events.append(idx)

    # Attach new columns to dataframe
    for label in ["Pi0", "h1", "h2"]:
        prefix = f"hB_{label}_pi0constPVconst"
        df[f"{prefix}_E_Brest"]  = results[label]["E"]
        df[f"{prefix}_PX_Brest"] = results[label]["PX"]
        df[f"{prefix}_PY_Brest"] = results[label]["PY"]
        df[f"{prefix}_PZ_Brest"] = results[label]["PZ"]
    prefix = "hB_pi0constPVconst"
    df[f"{prefix}_E_Brest"]  = results["hB"]["E"]
    df[f"{prefix}_PX_Brest"] = results["hB"]["PX"]
    df[f"{prefix}_PY_Brest"] = results["hB"]["PY"]
    df[f"{prefix}_PZ_Brest"] = results["hB"]["PZ"]

    if debug:
        # debug reporting
        print(f"Total events: {len(df)}")
        print(f"Events with NaN in inputs: {len(nan_input_events)}")
        if nan_input_events:
            print("  First few:", nan_input_events[:5])
        print(f"Events with |beta|>=1 (unphysical boost): {len(unphysical_boost_events)}")
        if unphysical_boost_events:
            print("  First few (idx, beta):", unphysical_boost_events[:5])
        print(f"Events with NaN/inf in boosted output (other cause): {len(nan_output_events)}")
        if nan_output_events:
            print("  First few:", nan_output_events[:5])

    return df, boost_vectors


def calculate_coplanarity_angle(df, constrained=True, Brest=False, true_momentum=False):
    """
    df : A dataframe with branches for the 3 momentum of 3 particles
    constrained : whether or not to use the DTF adjusted momenta
    Brest: whether or not to use Brest frame or lab frame
    true_momentum: whether or not to use monte carlo true momenta

    Returns angle between Pi0 and the plane containing K and Pi for each event
    
    The three optional booleans make this code messy and inflexible regarding df keys for momentum.
    For improved flexibility, use calculate_coplanarity_angle_manual.
    
    cos(phi) = p3 dot (p1 cross p2)
    p3_theta = phi - pi/2 = arcsin( p3 dot (p1 cross p2) )
    phi is angle from norm, but we are intereted in p3_theta, the angle from plane
    """
    # Reshape dataframe
    # This is very messy, and should either be rewritten, or simply replaced with calculate_coplanarity_angle_manual
    if constrained:
        if Brest:
            if true_momentum:
                df_p1 = df[["h1_Pi0_TRUE_PX_Brest", "h1_Pi0_TRUE_PY_Brest", "h1_Pi0_TRUE_PZ_Brest"]]
                df_p2 = df[["h2_Pi0_TRUE_PX_Brest", "h2_Pi0_TRUE_PY_Brest", "h2_Pi0_TRUE_PZ_Brest"]]
                df_p3 = df[["Pi0_Pi0_TRUE_PX_Brest", "Pi0_Pi0_TRUE_PY_Brest", "Pi0_Pi0_TRUE_PZ_Brest"]]
            else:
                df_p1 = df[["hB_h1_pi0constPVconst_PX_Brest", "hB_h1_pi0constPVconst_PY_Brest", "hB_h1_pi0constPVconst_PZ_Brest"]]
                df_p2 = df[["hB_h2_pi0constPVconst_PX_Brest", "hB_h2_pi0constPVconst_PY_Brest", "hB_h2_pi0constPVconst_PZ_Brest"]]
                df_p3 = df[["hB_Pi0_pi0constPVconst_PX_Brest", "hB_Pi0_pi0constPVconst_PY_Brest", "hB_Pi0_pi0constPVconst_PZ_Brest"]]
        else:
            if true_momentum:
                df_p1 = df[["h1_Pi0_TRUE_PX", "h1_Pi0_TRUE_PY", "h1_Pi0_TRUE_PZ"]]
                df_p2 = df[["h2_Pi0_TRUE_PX", "h2_Pi0_TRUE_PY", "h2_Pi0_TRUE_PZ"]]
                df_p3 = df[["Pi0_Pi0_TRUE_PX", "Pi0_Pi0_TRUE_PY", "Pi0_Pi0_TRUE_PZ"]]
            else:
                df_p1 = df[["hB_h1_pi0constPVconst_PX", "hB_h1_pi0constPVconst_PY", "hB_h1_pi0constPVconst_PZ"]]
                df_p2 = df[["hB_h2_pi0constPVconst_PX", "hB_h2_pi0constPVconst_PY", "hB_h2_pi0constPVconst_PZ"]]
                df_p3 = df[["hB_Pi0_pi0constPVconst_PX", "hB_Pi0_pi0constPVconst_PY", "hB_Pi0_pi0constPVconst_PZ"]]
    else:
        if Brest:  
            df_p1 = df[["h1_PX_Brest", "h1_PY_Brest", "h1_PZ_Brest"]]
            df_p2 = df[["h2_PX_Brest", "h2_PY_Brest", "h2_PZ_Brest"]]
            df_p3 = df[["Pi0_PX_Brest", "Pi0_PY_Brest", "Pi0_PZ_Brest"]]
        else:
            df_p1 = df[["h1_PX", "h1_PY", "h1_PZ"]]
            df_p2 = df[["h2_PX", "h2_PY", "h2_PZ"]]
            df_p3 = df[["Pi0_PX", "Pi0_PY", "Pi0_PZ"]]
            
    # Get magnitude of all momentas
    norms1 = np.linalg.norm(df_p1.values, axis=1, keepdims=True)
    norms2 = np.linalg.norm(df_p2.values, axis=1, keepdims=True)
    norms3 = np.linalg.norm(df_p3.values, axis=1, keepdims=True)
    # Calculate unit vectors of all momentas
    p1_unitvec = df_p1.values / norms1
    p2_unitvec = df_p2.values / norms2
    p3_unitvec = df_p3.values / norms3
    
    # Calculate unit vector of the normal of the plane containing particles 1 and 2
    plane_normal = np.cross(p1_unitvec, p2_unitvec)                              # Cross product for all events at once
    plane_normal = plane_normal / np.linalg.norm(plane_normal, axis=1, keepdims=True)  # Normalize all at once

    # Determine angle from dot product between particle 3 momentum and normal of particle 1 and 2's plane
    p3_theta = np.arcsin((p3_unitvec * plane_normal).sum(axis=1))
    
    return p3_theta


def calculate_coplanarity_angle_manual(df, keys_p1, keys_p2, keys_p3):
    """
    Same as calculate_coplanarity_angle, except the keys must manually be provided.
    Each list of keys should point to (px, py, pz) for particles 1, 2, 3;
    for example keys_p1=["h1_PX", "h1_PY", "h1_PZ"]
    """
    df_p1 = df[keys_p1]
    df_p2 = df[keys_p2]
    df_p3 = df[keys_p3]

    # Get magnitude of all momentas
    norms1 = np.linalg.norm(df_p1.values, axis=1, keepdims=True)
    norms2 = np.linalg.norm(df_p2.values, axis=1, keepdims=True)
    norms3 = np.linalg.norm(df_p3.values, axis=1, keepdims=True)
    # Calculate unit vectors of all momentas
    p1_unitvec = df_p1.values / norms1
    p2_unitvec = df_p2.values / norms2
    p3_unitvec = df_p3.values / norms3
    
    # Calculate unit vector of the normal of the plane containing particles 1 and 2
    plane_normal = np.cross(p1_unitvec, p2_unitvec)       # Cross product for all events at once
    plane_normal = plane_normal / np.linalg.norm(plane_normal, axis=1, keepdims=True)  # Normalize all at once

    # Determine angle from dot product between particle 3 momentum and normal of particle 1 and 2's plane
    p3_theta = np.arcsin((p3_unitvec * plane_normal).sum(axis=1))
    
    return p3_theta

def calculate_FOM_angle(df_signal, df_background, key_coplanarity, cuts, 
                        bkg_width, sig_width=WIDTH_SIG, debug=False):
    """
        Calculates and returns puirity and significance with their uncertainties from two dataframes 
        * df_signal contains data relating to signal window 
        * df_background contains data relating to a sideband
        * key_coplanarity indicates the keys of df_signal and df_background
          that index the coplanarity angle. Cahnge this key to choose lab or Brest frame.
        * cuts is an array of angles from large to small values where for each iteration,
          any event with coplanarity angle greater than current cut value is rejected.
        * bkg_width should be either WIDTH_USB or WIDTH_LSB and scales events in
          the sidebands to match the width of the signal window.
        * sig_width should probably never be overwritten.
        
        Purity = (N-B) / N ; Significiance = (N-B) / sqrt(N)
    """
    # Initialize all needed arrays prior to looping
    l = len(cuts)
    x = ufloat(np.nan, np.nan)  # Dummy array used to turn all numpy arrays below into arrays containing ufloat objects
    ufloat_N = x*np.ones(l, dtype=object)
    ufloat_B = x*np.ones(l, dtype=object)
    ufloat_Bscaled = x*np.ones(l, dtype=object)
    ufloat_purity = x*np.ones(l, dtype=object)
    ufloat_significance = x*np.ones(l, dtype=object)
      
    for i, angle in enumerate(cuts):
        # Caluclate N, B and B scaled to the relative size of the signal window
        N = len( df_signal[ np.abs(df_signal[key_coplanarity]) <= angle ].copy().reset_index(drop=True) ) # reset_index means that if event n is rejected, then n+1 becomes the new n
        if N==0:
            print("Calculate FOM early termination")
            return ufloat_purity, ufloat_significance 
        B = len( df_background[ np.abs(df_background[key_coplanarity]) <= angle ].copy().reset_index(drop=True) )     
        Bscaled = B * sig_width / bkg_width # Background events scaled to signal window
        
        # Calculate uncertenties of N, B, B scaled
        ufloat_N[i] = ufloat(N, np.sqrt(N))
        ufloat_B[i] = ufloat(B, np.sqrt(B))
        ufloat_Bscaled[i] = ufloat(Bscaled, sig_width / bkg_width * np.sqrt(B)) 
        
        # Calculate FoM and propgate uncertenties
        ufloat_purity[i] = (ufloat_N[i] - ufloat_Bscaled[i]) / ufloat_N[i]
        ufloat_significance[i] = (ufloat_N[i] - ufloat_Bscaled[i]) / umath.sqrt(ufloat_N[i]) 
        if debug:
            print(f"N={N}, B={B}, Bscaled={Bscaled}")
            print(f"N={ufloat_N}[i]")
            print(f"B={ufloat_B}[i]")
            print(f"Bscaled={ufloat_Bscaled}[i]")
            print(f"P={ufloat_purity}[i]")
            print(f"N={N}, B'={Bscaled}, P={ufloat_purity[i]}")
    return ufloat_purity, ufloat_significance 

def plot_FOM(cuts, P, S, list_df_indexes, key_coplanarity,
             P_ylim, S_ylim, PS_ylim,
             title, file_name, 
             bkg_width, sig_width=WIDTH_SIG, normalize=False):
    """
        * Plots 3 vertically stacked graph with all x axes being a choice of coplanarity angle cuts:
            plot1 graphs purity and signficiance,
            plot2 graphs purity*significance,
            plot3 graphs MC signal, data signal window and data sideband histogram
        
        * cuts is an array of angles from large to small values where for each iteration,
          any event with coplanarity angle greater than current cut value is rejected.
        * P is a list of purity values
        * S is a list of significance values
        * list_df_indexes a very stupid way to keep colors of graphs consistant throughout code.
          should be [0,1] for upper sideband or [0,2] for LSB
        * key_coplanarity indicates the keys of df_signal and df_background
          that index the coplanarity angle. Cahnge this key to choose lab or Brest frame. 
        * P_ylim, S_ylim, PS_ylim control the y limit of the corresponding axes
        * title: title of the figure
        * file_name: name of the file; save location is loc_img
        
    """
    PS = P*S # Purity*Significance is determining FoM
    optimal_angle = cuts[np.argmax(PS)] # Optimal angle to cut on is where P*S is maximized
    
    fig, [ax1, ax3, ax4] = plt.subplots(3, figsize=(12,4*3))
    # Graph purity on first subplot
    color = "tab:red"
    ax1.set_ylabel("Purity", color=color)
    ax1.errorbar(cuts, unp.nominal_values(P), color=color,
                 yerr=unp.std_devs(P), fmt="-", linewidth=linewidth)
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.set_ylim(P_ylim)
    ax1.set_xlim(np.max(cuts), np.min(cuts))
    
    # Graph significance on same graph but different y axis as purity
    color = "tab:blue"
    ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis
    ax2.set_ylabel("Significance", color=color)  # we already handled the x-label with ax1
    ax2.errorbar(cuts, unp.nominal_values(S), color=color,
                 yerr=unp.std_devs(S), fmt="-", linewidth=linewidth)
    ax2.tick_params(axis="y", labelcolor=color)
    ax2.set_ylim(S_ylim)

    # Graph P*S on second subplot
    color = "tab:purple"
    ax3.set_ylabel("Purity*Significance", color=color)  # we already handled the x-label with ax1
    ax3.errorbar(cuts, unp.nominal_values(PS), color=color,
                 yerr=unp.std_devs(PS), fmt="o", linewidth=linewidth)
    ax3.tick_params(axis="y", labelcolor=color)
    ax3.set_ylim(PS_ylim)
    ax3.set_xlim(np.max(cuts), np.min(cuts))
    ax3.set_xlabel("cuts on angle (radians)")
    
    # Graph 
    bins = np.linspace(np.min(cuts), np.max(cuts), len(cuts), False) # each histogram bin should correspond to a cut
    '''
    # Only the MC signal and data sideband are significant, so stop showing the data signal window
    # Histogram of the signal window
    ax4.hist(list_df[list_df_indexes[0]][key_coplanarity], bins = bins, density=True,
             histtype="step", label=list_df_labels[list_df_indexes[0]], 
             color=list_df_colors[list_df_indexes[0]], linewidth=linewidth)
    '''
    # Histogram of the background in sideband
    ax4.hist(list_df[list_df_indexes[1]][key_coplanarity] , bins = bins, density=True,
             histtype="step", label=list_df_labels[list_df_indexes[1]], 
             color=list_df_colors[list_df_indexes[1]], linewidth=linewidth)
    # Histogram of the MC signal
    ax4.hist(list_df[3][key_coplanarity] , bins = bins, density=True,
             histtype="step", label=list_df_labels[3], 
             color=list_df_colors[3], linewidth=linewidth)
    ax4.set_title(title)
    ax4.set_xlabel(r"$\pi ^0$ absolute angle from plane (rad)")
    ax4.set_ylabel(r"Arbitrary units")
    ax4.set_xlim(np.max(cuts), np.min(cuts))
    ax4.legend()
    
    # Draw verticle line on all subplots for the optimal angle cut
    ax1.axvline(x=optimal_angle, color="black", linestyle="--", linewidth=1)
    ax3.axvline(x=optimal_angle, color="black", linestyle="--", linewidth=1)
    ax4.axvline(x=optimal_angle, color="black", linestyle="--", linewidth=1)

    #fig.text(0,0,f"The product is maximal at {cuts[optimal_index]:.4f} rad where P={P[optimal_index]:.4f} and S={S[optimal_index]:.4f}")
    fig.suptitle(title)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    fig.savefig(loc_img+file_name, bbox_inches="tight") # bbox prevents subtext from getting clipped
    plt.show()
    return optimal_angle

def force_extension(file_name, extension=".root"):
    # Ensure the file_name has the correct extension. .root by default
    ext_index = file_name.find(".")
    if ext_index == -1:  # If there is no extension, add it
        file_name += extension
    elif file_name[ext_index:] != extension:  # If the extension is wrong, replace it
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
    rdf = ROOT.RDataFrame(tree_name_raw, force_extension(file_name_raw))
    
    if branches is None:   # Get all branches if not specified
        branches = [str(b) for b in rdf.GetColumnNames()] # Must cast explicitly to string because GetColumnNames returns a std::vector<std::string>
        
    # Specify desired branches from the TTree and store those into a pandas dataframe
    df_raw = pd.DataFrame({b: rdf.AsNumpy([b])[b] for b in branches})
    return df_raw




""" --------------------------------------------------------------------------
    Open TFiles containing MC and Data. 
    TFiles tuples should have all particle 4 momenta.
---------------------------------------------------------------------------"""
'''
# File location and names of the Monte Carlo Simulation

file_loc_mc = "/home/kanoa/Documents/UCA/Internship LHCb2026/monte_carlo_analysis/" 
file_names_mc = ["invariant_mass_squared_pi0constPVconst_down_Kpi_mc", 
                  "invariant_mass_squared_pi0constPVconst_up_Kpi_mc", 
                  "invariant_mass_squared_pi0constPVconst_down_piK_mc", 
                  "invariant_mass_squared_pi0constPVconst_up_piK_mc"]

# File location and names of the LHCb Run 2 Dataset
file_loc_data = "/home/kanoa/Documents/UCA/Internship LHCb2026/blinded_data_analysis/" 
file_names_data =  ["invariant_mass_squared_pi0constPVconst_down_Kpi_data", 
                    "invariant_mass_squared_pi0constPVconst_up_Kpi_data", 
                    "invariant_mass_squared_pi0constPVconst_down_piK_data", 
                    "invariant_mass_squared_pi0constPVconst_up_piK_data"]
tree_name = "invariant_masses"
branches_data = None # This will get all branches
branches_mc = None
'''
# File location and names of the Monte Carlo Simulation
file_loc_mc = "/home/kanoa/Documents/UCA/Internship LHCb2026/datasets_signalMC2018_selected/" 
file_names_mc = ["signalMC2018down_Kpi", 
                  "signalMC2018up_Kpi", 
                  "signalMC2018down_piK", 
                  "signalMC2018up_piK"]

# File location and names of the LHCb Run 2 Dataset
file_loc_data = "/home/kanoa/Documents/UCA/Internship LHCb2026/blinded_data2018/" 
file_names_data =  ["data2018down_Kpi", 
                    "data2018up_Kpi", 
                    "data2018down_piK", 
                    "data2018up_piK"]
tree_name = "hBtohhpi0R_Tuple"

# What branches of Tuple to put into dataframes
branches_data = ['Pi0_M','Pi0_P', 'Pi0_PE', 'Pi0_PX', 'Pi0_PY', 'Pi0_PZ',
                 'h1_M', 'h1_P','h1_PE', 'h1_PX', 'h1_PY', 'h1_PZ', 
                 'h2_M', 'h2_P', 'h2_PE', 'h2_PX', 'h2_PY', 'h2_PZ', 
                 'hB_M','hB_P', 'hB_PE', 'hB_PX', 'hB_PY', 'hB_PZ',
                 'hB_Pi0_pi0constPVconst_E', 'hB_Pi0_pi0constPVconst_M', 'hB_Pi0_pi0constPVconst_PX', 'hB_Pi0_pi0constPVconst_PY', 'hB_Pi0_pi0constPVconst_PZ', 
                 'hB_h1_pi0constPVconst_E', 'hB_h1_pi0constPVconst_M', 'hB_h1_pi0constPVconst_PX', 'hB_h1_pi0constPVconst_PY', 'hB_h1_pi0constPVconst_PZ',
                 'hB_h2_pi0constPVconst_E', 'hB_h2_pi0constPVconst_M', 'hB_h2_pi0constPVconst_PX', 'hB_h2_pi0constPVconst_PY', 'hB_h2_pi0constPVconst_PZ',
                 'hB_pi0constPVconst_E', 'hB_pi0constPVconst_M', 'hB_pi0constPVconst_P', 'hB_pi0constPVconst_PX', 'hB_pi0constPVconst_PY', 'hB_pi0constPVconst_PZ',]
branches_mc = branches_data + ['hB_TRUEP_X', 'hB_TRUEP_Y', 'hB_TRUEP_Z', 'hB_TRUEP_E',
                               'h1_TRUEP_X', 'h1_TRUEP_Y', 'h1_TRUEP_Z', 'h1_TRUEP_E',
                               'h2_TRUEP_X', 'h2_TRUEP_Y', 'h2_TRUEP_Z', 'h2_TRUEP_E',
                               'Pi0_TRUEP_X', 'Pi0_TRUEP_Y', 'Pi0_TRUEP_Z', 'Pi0_TRUEP_E',
                               'hB_BKGCAT',]

file_index = 0 # 0 for B0, 2 for B0bar. Dalitz labels will update automatically

### Open TFiles in master dataframes that contain all 4 momentum, masses, and invariant masses.
# Data B0. Merge both magnet orientations using file_index for down and file_index+1 for up
file_active_data = file_loc_data + file_names_data[file_index]
df_master_data = open_tfile_as_dataframe(file_active_data, tree_name, branches_data)
file_up_data = file_loc_data + file_names_data[file_index+1]
df_master_data = pd.concat([df_master_data, open_tfile_as_dataframe(file_up_data, tree_name, branches_data)])
if branches_data==None:
    branches_data = list(df_master_data.keys())

# Monte Carlo B0. Merge both magnet orientations using file_index for down and file_index+1 for up
file_active_mc = file_loc_mc + file_names_mc[file_index]
df_master_mc = open_tfile_as_dataframe(file_active_mc, tree_name, branches_mc)
file_up_mc = file_loc_mc + file_names_mc[file_index+1]
df_master_mc = pd.concat([df_master_mc, open_tfile_as_dataframe(file_up_mc, tree_name, branches_mc)])
if branches_mc==None:
    branches_mc = list(df_master_mc.keys())

### Divide master dataframes into subframes based on window of interest
# Initialize empty dataframes
df_signal_window_data = pd.DataFrame()
df_upperSB_data = pd.DataFrame()
df_lowerSB_data = pd.DataFrame()
df_signal_mc = pd.DataFrame()
list_df = [df_signal_window_data, df_upperSB_data, df_lowerSB_data, df_signal_mc]

# Apply window cuts to master dataframes to form dataframes used in the rest of code
# Apply window cuts and truth matching on df_master_data.
str_bmass = "hB_pi0constPVconst_M" # Key for the tuple that indexes reconstructed B mass to select windows
for b in branches_data: # Would probably be faster to remove loop since I am now getting all branches but leave it in in case I later change my mind and return to getting only select branches
    df_signal_window_data[b] = df_master_data[( (df_master_data[str_bmass]>THRESH_SIG_L) & (df_master_data[str_bmass]<THRESH_SIG_U))][b].copy().reset_index(drop=True)
    df_upperSB_data[b] = df_master_data[(       (df_master_data[str_bmass]>THRESH_USB_L) & (df_master_data[str_bmass]<THRESH_USB_U))][b].copy().reset_index(drop=True)
    df_lowerSB_data[b] = df_master_data[(       (df_master_data[str_bmass]>THRESH_LSB_L) & (df_master_data[str_bmass]<THRESH_LSB_U))][b].copy().reset_index(drop=True)

# Apply window cuts and truth matching on df_master_mc.
for b in branches_mc: # Would probably be faster to remove loop since I am now getting all branches but leave it in in case I later change my mind and return to getting only select branches
    df_signal_mc[b] = df_master_mc[(            (df_master_mc[str_bmass]>THRESH_SIG_L)   & (df_master_mc[str_bmass]<THRESH_SIG_U) & (df_master_mc["hB_BKGCAT"]==0))][b].copy().reset_index(drop=True)

# Ensure consistency of color and labels between graphs by indexing these lists
list_df_colors = ["C0", "C1", "C2", "C3"]
list_df_labels = ["Data Signal Window", "Data Upper SB", "Data Lower SB", "MC True Signal"]

# Add Lorentz boosted 4-momentum of all particles in B rest frame
"""
FUTURE IMPROVEMENT Now that I am boosting df_master_data it would probably be better to do that once for df_master_data then mask along selection windows like I do for all other branches
"""
df_signal_window_data, lorentz_sw = boost_to_B_rest_frame(df_signal_window_data)
df_upperSB_data, lorentz_usb      = boost_to_B_rest_frame(df_upperSB_data)
df_lowerSB_data, lorentz_lsb      = boost_to_B_rest_frame(df_lowerSB_data)
df_signal_mc, lorentz_mc          = boost_to_B_rest_frame(df_signal_mc)

# Calculate and add coplanarity angle in both lab and B rest frame to dataframes
constrained = True # Use constrained 4 momentum
Brest = False      # Use Lab frame
key_coplanarity = "coplanarity_angle_lab"
for df in list_df:
    df[key_coplanarity] = calculate_coplanarity_angle(df, constrained=constrained, Brest=Brest)
    df[key_coplanarity+"_abs"] = np.abs(df[key_coplanarity])
Brest = True      # Use Lab frame
key_coplanarity = "coplanarity_angle_Brest"
for df in list_df:
    df[key_coplanarity] = calculate_coplanarity_angle(df, constrained=constrained, Brest=Brest)
    df[key_coplanarity+"_abs"] = np.abs(df[key_coplanarity])



""" --------------------------------------------------------------------------
    Graph coplanarity angle for a variety of windows and reference frames
---------------------------------------------------------------------------"""

### COPLANARITY IN MC IN BOTH FRAMES
linewidth = 3 # Control width of histogram lines
normalize = True # Control if histograms should be normalized
txt_size = 20 # Control text size for titles and axis labels
nbins = 50

### ABS VALUE OF COPLANARITY ALL WINDOWS IN LAB FRAME
# Get max angle for 
combined_max_lab = max(df_signal_mc["coplanarity_angle_lab_abs"].max(), df_signal_window_data["coplanarity_angle_lab_abs"].max(), df_upperSB_data["coplanarity_angle_lab_abs"].max(), df_lowerSB_data["coplanarity_angle_lab_abs"].max())
bin_edges = np.linspace(0, combined_max_lab, nbins)

fig, ax = plt.subplots(1, figsize=(8,8))
ax.hist(df_signal_mc["coplanarity_angle_lab_abs"], bins=bin_edges, histtype="step", 
        label="MC signal", linewidth=linewidth, density=normalize, color=list_df_colors[3])
ax.hist(df_signal_window_data["coplanarity_angle_lab_abs"], bins=bin_edges, histtype="step", 
        label="Data Signal Window", linewidth=linewidth, density=normalize, color=list_df_colors[0])
ax.hist(df_upperSB_data["coplanarity_angle_lab_abs"], bins=bin_edges, histtype="step", 
        label="Data USB", linewidth=linewidth, density=normalize, color=list_df_colors[1])
ax.hist(df_lowerSB_data["coplanarity_angle_lab_abs"], bins=bin_edges, histtype="step", 
        label="Data LSB", linewidth=linewidth, density=normalize, color=list_df_colors[2])
ax.set_yscale('log')
ax.set_title(r"Coplanarity angle in lab frame for all windows", )
ax.set_xlabel("Angle (rad)")
ax.set_ylabel("Arbitrary units")
ax.legend()

### ABS VALUE OF COPLANARITY ALL WINDOWS IN B REST FRAME
combined_max_Brest = max(df_signal_mc["coplanarity_angle_Brest_abs"].max(), df_signal_window_data["coplanarity_angle_Brest_abs"].max(), df_upperSB_data["coplanarity_angle_Brest_abs"].max(), df_lowerSB_data["coplanarity_angle_Brest_abs"].max())
bin_edges = np.linspace(0, combined_max_Brest, nbins)

fig, ax = plt.subplots(1, figsize=(8,8))
ax.hist(df_signal_mc["coplanarity_angle_Brest_abs"], bins=bin_edges, histtype="step", 
        label="MC signal", linewidth=linewidth, density=normalize, color="red")
ax.hist(df_signal_window_data["coplanarity_angle_Brest_abs"], bins=bin_edges, histtype="step", 
        label="Data Signal Window", linewidth=linewidth, density=normalize, color="blue")
ax.hist(df_upperSB_data["coplanarity_angle_Brest_abs"], bins=bin_edges, histtype="step", 
        label="Data USB", linewidth=linewidth, density=normalize, color="orange")
ax.hist(df_lowerSB_data["coplanarity_angle_Brest_abs"], bins=bin_edges, histtype="step", 
        label="Data LSB", linewidth=linewidth, density=normalize, color="green")
ax.set_yscale('log')
ax.set_title(r"Coplanarity angle in B rest frame for all windows", )
ax.set_xlabel("Angle (rad)")
ax.set_ylabel("Arbitrary units")
ax.legend()



""" --------------------------------------------------------------------------
    Calculate and graph figures of merit
---------------------------------------------------------------------------"""

### CALCULATE PURITY AND SIGNIFICANCE FOR ELIMINATING BACKGROUND IN UPPER SIDEBAND OF BREST
nbins = 100
# Make num_bins of selections between maximum angle in all windows and 0 radians
cuts_coplanarity_Brest = np.linspace(combined_max_Brest, 0, nbins, False)

lim_lab = np.max(df_upperSB_data["coplanarity_angle_lab_abs"])
cuts_coplanarity_lab = np.linspace(combined_max_lab, 0, nbins, False)

purity_data_sig_usb_Brest, signif_data_sig_usb_Brest = calculate_FOM_angle(df_signal_window_data, 
                                                                           df_upperSB_data, 
                                                                           "coplanarity_angle_Brest",
                                                                           cuts_coplanarity_Brest,
                                                                           WIDTH_USB)
### GRAPH PURITY AND SIGNIFINCE AND PURITY*SIGNFICANCE AND DEPICT AMOUNT OF DATA CUT
P_ylim = [0.8, 1]
S_ylim = [0 ,40]
PS_ylim = [0, 40]
opt_angle_usb = plot_FOM(cuts_coplanarity_Brest, purity_data_sig_usb_Brest, signif_data_sig_usb_Brest, 
                         list_df_indexes=[0,1], key_coplanarity="coplanarity_angle_Brest_abs", 
                         P_ylim=P_ylim, S_ylim=S_ylim, PS_ylim=PS_ylim, 
                         title="FoM: Data Signal Window and Upper SB in Brest Frame", 
                         file_name="FoM_dataSigUppersb_angle_Brest.png",
                         bkg_width=WIDTH_USB)

lost_comb_usb = len(df_upperSB_data[df_upperSB_data["coplanarity_angle_Brest_abs"]>=opt_angle_usb] )
tot_comb_usb = len(df_upperSB_data["coplanarity_angle_Brest_abs"])
perc_comb_usb = lost_comb_usb/tot_comb_usb * 100
lost_even_usb = len(df_signal_window_data[df_signal_window_data["coplanarity_angle_Brest_abs"]>=opt_angle_usb] )
tot_even_usb = len(df_signal_window_data["coplanarity_angle_Brest_abs"])
perc_even_usb = lost_even_usb/tot_even_usb * 100
lost_sign_usb = len(df_signal_mc[df_signal_mc["coplanarity_angle_Brest_abs"]>=opt_angle_usb] )
tot_sign_usb = len(df_signal_mc["coplanarity_angle_Brest_abs"])
perc_sign_usb = lost_sign_usb/tot_sign_usb * 100
print(f"Cutting USB in Brest at {opt_angle_usb:.4f}% rad results in rejection of {perc_comb_usb:.1f}% background; {perc_even_usb:.1f}% event; {perc_sign_usb:.1f}% signal")


### REPEAT FOR ELIMINATING BACKGROUND IN LOWER SIDEBAND BREST
purity_data_sig_lsb_Brest, signif_data_sig_lsb_Brest = calculate_FOM_angle(df_signal_window_data, 
                                                                           df_lowerSB_data, 
                                                                           "coplanarity_angle_Brest",
                                                                           cuts_coplanarity_Brest, 
                                                                           bkg_width=WIDTH_LSB)

opt_angle_lsb = plot_FOM(cuts_coplanarity_Brest, purity_data_sig_usb_Brest, signif_data_sig_lsb_Brest, 
                         list_df_indexes=[0,2], key_coplanarity="coplanarity_angle_Brest_abs", 
                         P_ylim=P_ylim, S_ylim=S_ylim, PS_ylim=PS_ylim, 
                         title="FoM: Data Signal Window and Lower SB in Brest Frame", 
                         file_name="FoM_dataSigLowersb_angle_Brest.png",
                         bkg_width=WIDTH_LSB)

lost_comb_lsb = len(df_lowerSB_data[df_lowerSB_data["coplanarity_angle_Brest_abs"]>=opt_angle_lsb] )
tot_comb_lsb = len(df_lowerSB_data["coplanarity_angle_Brest_abs"])
perc_comb_lsb = lost_comb_lsb/tot_comb_lsb * 100
lost_even_lsb = len(df_signal_window_data[df_signal_window_data["coplanarity_angle_Brest_abs"]>=opt_angle_lsb] )
tot_even_lsb = len(df_signal_window_data["coplanarity_angle_Brest_abs"])
perc_even_lsb = lost_even_lsb/tot_even_lsb * 100
lost_sign_lsb = len(df_signal_mc[df_signal_mc["coplanarity_angle_Brest_abs"]>=opt_angle_lsb] )
tot_sign_lsb = len(df_signal_mc["coplanarity_angle_Brest_abs"])
perc_sign_lsb = lost_sign_lsb/tot_sign_lsb * 100
print(f"Cutting LSB in Brest at {opt_angle_lsb:.4f}% rad results in rejection of {perc_comb_lsb:.1f}% background; {perc_even_lsb:.1f}% event; {perc_sign_lsb:.1f}% signal")
print("Note that for LSB, while combinatorial background is uniform, PR is not, so the above estimates are very rough approximations of what PR background will actually be detected in data signal window.\n")

'''
### REPEAT BOTH GRAPHS BUT NOW FOR LAB FRAME

purity_data_sig_usb_lab, signif_data_sig_usb_lab = calculate_FOM_angle(df_signal_window_data, 
                                                                       df_upperSB_data, 
                                                                       "coplanarity_angle_lab",
                                                                       cuts_coplanarity_lab,
                                                                       bkg_width=WIDTH_USB)
P_ylim = [0.8, 1]
S_ylim = [30 ,40]
PS_ylim = [30, 40]
opt_angle_usb2 = plot_FOM(cuts_coplanarity_lab, purity_data_sig_usb_lab, signif_data_sig_usb_lab, 
                         list_df_indexes=[0,1], key_coplanarity="coplanarity_angle_lab_abs", 
                         P_ylim=P_ylim, S_ylim=S_ylim, PS_ylim=PS_ylim, 
                         title="FoM: Data Signal Window and Upper SB in lab Frame", 
                         file_name="FoM_dataSigUppersb_angle_lab.png",
                         bkg_width=WIDTH_USB)

purity_data_sig_lsb_lab, signif_data_sig_lsb_lab = calculate_FOM_angle(df_signal_window_data, 
                                                                       df_upperSB_data, 
                                                                       "coplanarity_angle_lab",
                                                                       cuts_coplanarity_lab,
                                                                       bkg_width=WIDTH_LSB)

opt_angle_lsb2 = plot_FOM(cuts_coplanarity_lab, purity_data_sig_lsb_lab, signif_data_sig_lsb_lab, 
                         list_df_indexes=[0,2], key_coplanarity="coplanarity_angle_lab_abs", 
                         P_ylim=P_ylim, S_ylim=S_ylim, PS_ylim=PS_ylim, 
                         title="FoM: Data Signal Window and Lower SB in lab Frame", 
                         file_name="FoM_dataSigLowersb_angle_lab.png",
                         bkg_width=WIDTH_LSB)
'''

# Graph zoomed-in version of optimal cut
cuts = cuts_coplanarity_Brest
key_coplanarity = "coplanarity_angle_Brest_abs"
bins = np.linspace(np.min(cuts), np.max(cuts), len(cuts), False) # each histogram bin should correspond to a cut
normalize = True
fig, [ax1, ax2] = plt.subplots(2, figsize=(8,16))
# Histogram of the data upper sideband
ax1.hist(df_upperSB_data[key_coplanarity] , bins = bins, density=normalize,
         histtype="step", label=list_df_labels[1], 
         color=list_df_colors[1], linewidth=linewidth)
# Histogram of the MC signal
ax1.hist(list_df[3][key_coplanarity] , bins = bins, density=normalize,
         histtype="step", label=list_df_labels[3], 
         color=list_df_colors[3], linewidth=linewidth)

ax1.set_xlabel(r"$\pi ^0$ absolute angle from plane (rad)")
ax1.set_ylabel(r"Arbitrary units")
ax1.set_xlim(0.3, 0)
ax1.legend()
# Draw verticle line on all subplots for the optimal angle cut
ax1.axvline(x=opt_angle_usb, color="black", linestyle="--", linewidth=1)

# Histogram of the data lower sideband
ax2.hist(df_lowerSB_data[key_coplanarity] , bins = bins, density=normalize,
         histtype="step", label=list_df_labels[2], 
         color=list_df_colors[2], linewidth=linewidth)
# Histogram of the MC signal
ax2.hist(list_df[3][key_coplanarity] , bins = bins, density=normalize,
         histtype="step", label=list_df_labels[3], 
         color=list_df_colors[3], linewidth=linewidth)

ax2.set_xlabel(r"$\pi ^0$ absolute angle from plane (rad)")
ax2.set_ylabel(r"Arbitrary units")
ax2.set_xlim(0.3, 0)
ax2.legend()
# Draw verticle line on all subplots for the optimal angle cut
ax2.axvline(x=opt_angle_lsb, color="black", linestyle="--", linewidth=1)

fig.suptitle("Rejection for optimal cuts")
fig.tight_layout()  # otherwise the right y-label is slightly clipped
fig.savefig(loc_img+"OptimalCuts_USBvsLSB", bbox_inches="tight") # bbox prevents subtext from getting clipped
plt.show()

"""
    Compare distribution of beauty invariant mass before and after optimal cuts
"""
# keys to make code more concise
k_theta = "coplanarity_angle_Brest_abs"
k_E = "hB_pi0constPVconst_E"
k_M = "hB_pi0constPVconst_M"


# Boost all data events into B rest frame, then get coplanarity angle in B rest frame
df_master_data, lorentz_master = boost_to_B_rest_frame(df_master_data)
df_master_data["coplanarity_angle_Brest"] = calculate_coplanarity_angle(df_master_data, Brest=True)
df_master_data[k_theta] = np.abs(df_master_data["coplanarity_angle_Brest"])

df = df_master_data # Make pointer variable to df_master_data to make code concise
cut_angle = opt_angle_usb
df_cut = df[df[k_theta] < cut_angle].copy().reset_index(drop=True)

mass_uncut = df[k_M] # B mass distribution without coplanarity cuts
mass_cut = df_cut[k_M] #B mass distribution with coplanarity cuts

nbins = 40
bins = np.linspace(4000, 6000, nbins)
fig, ax = plt.subplots(1, figsize=(8,8))

hist_uncut = ax.hist(mass_uncut, bins=bins, label="uncut", density=False, histtype="step")
hist_cut = ax.hist(mass_cut, bins=bins, label=f"cut {cut_angle:.3f} rad", density=False, histtype="step")
ax.legend()
ax.set_title("Invariant mass of B descirmination with USB optimal cut")
ax.set_xlabel("B mass")
ax.set_ylabel("Events")

# These prints are just sanity checks to get a rough estimate of how many signal (maximally filled bins) are lost compared to average
print(f"---Cutting B distribution at USB's optimal angle: {cut_angle:.4f}")
avg_loss = np.average(hist_uncut[0] - hist_cut[0])
index_peak = np.argmax(hist_cut[0])
sig_loss = hist_uncut[0][index_peak] - hist_cut[0][index_peak]
print(f"lost {sig_loss} signal compared to an average of {avg_loss:.1f}")
print(f"Losses per bin: {hist_uncut[0] - hist_cut[0]}---\n")

# Repeat but now for optimal cut determined for lower sideband
cut_angle = opt_angle_lsb
df_cut = df[df[k_theta] < cut_angle].copy().reset_index(drop=True)
mass_cut = df_cut[k_M] #B mass distribution with coplanarity cuts

fig, ax = plt.subplots(1, figsize=(8,8))

hist_uncut = ax.hist(mass_uncut, bins=bins, label="uncut", density=False, histtype="step")
hist_cut = ax.hist(mass_cut, bins=bins, label=f"cut {cut_angle:.3f} rad", density=False, histtype="step")
ax.legend()
ax.set_title("Invariant mass of B descirmination with LSB optimal cut")
ax.set_xlabel("B mass")
ax.set_ylabel("Events")

# These prints are just sanity checks to get a rough estimate of how many signal (maximally filled bins) are lost compared to average
print(f"---Cutting B distribution at LSB's optimal angle: {cut_angle:.4f}")
avg_loss = np.average(hist_uncut[0] - hist_cut[0])
index_peak = np.argmax(hist_cut[0])
sig_loss = hist_uncut[0][index_peak] - hist_cut[0][index_peak]
print(f"lost {sig_loss} signal compared to an average of {avg_loss:.1f}")
print(f"Losses per bin: {hist_uncut[0] - hist_cut[0]}---\n")




