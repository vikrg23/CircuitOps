import os, sys
sys.path.append("/home/vgopal18/Circuitops/CircuitOps/src/python/")
from circuitops_api import *

def filter_pin_arcs(pin_pin_df, arc_type):
    """
    Return a pin to pin df after filtering the arc_type

    :param pin_pin_df: Required. Original pin to pin df to be filtered
    :type pin_pin_df: Pandas datframe of type pin_pin_df 
    :param arc_type: Required. Can be either cell or net
    :type arc_type: str
    :return: filtered pin to pin df
    :rtype: Pandas df 

    Example:
        cell_arcs_df = filter_pin_arcs(pin_pin_df,"cell")
    """
    if (arc_type == "cell"):
        return pin_pin_df[pin_pin_df['is_net']==0]
    elif (arc_type == "net"):
        return pin_pin_df[pin_pin_df['is_net']==1]
    else:
        return None

def calculate_load_cap(output_pins, pin_pin_df):
    """
    Return a pin to pin df with output_cap column

    :param output_pins: Required. List of pins for which load cap has to be calculated
    :type output_pins: List
    :param pin_pin_df: Required. Original pin to pin df to be processed
    :type pin_pin_df: Pandas datframe of type pin_pin_df 
    :return: pin to pin df with output_cap column
    :rtype: Pandas df

    Example:
        cell_arcs_df = calculate_load_cap(output_pins, pin_pin_df)
    """
    pin_pin_df["output_cap"] = -1
    for pin_id in output_pins:
        sink_pins = pin_pin_df[(pin_pin_df["src_id"]==pin_id) & (pin_pin_df["is_net"]==1)]["tar_id"]
        if (len(sink_pins) == 0):
            print("Warn: pin id ",pin_id," doesn't have sinks")
            continue
        total_cap = pin_df[pin_df['id'].isin(sink_pins)]['input_pin_cap']
        pin_pin_df.loc[pin_pin_df['tar_id']==pin_id,'output_cap'] = total_cap.sum()
    return pin_pin_df

def merge_tran_cell(pin_pin_df, pin_df, cell_df):
    """
    Return a pin to pin df after merging tran and cell type columns
    cell_types are coded as int to make it ML friendly.

    :param pin_pin_df: Required. Original pin to pin df to be processed
    :type pin_pin_df: Pandas datframe of type pin_pin_df 
    :param pin_df: Required. Original pin properties df to be processed
    :type pin_df: Pandas datframe of type pin_df
    :param cell_df: Required. Original cell properties df to be processed
    :type cell_df: Pandas datframe of type cell_df
    :return: pin to pin df with pin_tran, cell_type, cell_type_coded columns
    :rtype: Pandas df

    Example:
        cell_arcs_df = filter_pin_arcs(pin_pin_df,"cell")
    """
    pin_pin_df = pin_pin_df.merge(pin_df[['cell_name','pin_tran','id']].rename(columns={"id":"src_id"}), on="src_id", how="left")
    pin_pin_df = pin_pin_df.merge(cell_df[['cell_name','libcell_name']], on="cell_name", how="left")
    pin_pin_df["cell_type"] = pin_pin_df["libcell_name"].str.split('_').str[0]
    letter_to_int = {letter: idx for idx, letter in enumerate(sorted(pin_pin_df['cell_type'].unique()))}
    pin_pin_df['cell_type_coded'] = pin_pin_df['cell_type'].map(letter_to_int)
    return pin_pin_df
