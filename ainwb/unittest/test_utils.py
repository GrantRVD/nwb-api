
import sys
import traceback
import inspect
import h5py
import numpy as np
import nwb
from nwb.nwbco import *


def print_error(context, err_string):
    func = traceback.extract_stack()[-3][2]
    print("----------------------------------------")
    print("**** Failed unit test %s" % inspect.stack()[0][1])
    print("**** Error in function '%s'" % func)
    print("Context: " + context)
    print("Error: " + err_string)
    print("Stack:")
    traceback.print_stack()
    print("----------------------------------------")


def error(context, err_string):
    print_error(context, err_string)


def exc_error(context, exc):
    print_error(context, str(exc))


def search_for_string(h5_str, value):
    match = False
    if h5_str is not None:
        if isinstance(h5_str, (str, np.string_)):
            if h5_str == value:
                match = True
        elif isinstance(h5_str, (list, np.ndarray)):
            match = False
            for i in range(len(h5_str)):
                if h5_str[i] == value or h5_str[i] == np.bytes_(value):
                    match = True
                    break
    return match


def search_for_substring(h5_str, value):
    match = False
    if h5_str is not None:
        if isinstance(h5_str, (str, np.string_)):
            if str(h5_str).find(value) >= 0:
                match = True
        elif isinstance(h5_str, (list, np.ndarray)):
            match = False
            for i in range(len(h5_str)):
                if str(h5_str[i]).find(value) >= 0:
                    match = True
                    break
#    if not match and not isinstance(value, (np.bytes_)):
#        return search_for_substring(h5_str, np.bytes_(value))
    return match


def verify_timeseries(hfile, name, location, ts_type):
    """
    Verify that a time series is valid

    makes sure that the entity with this name at the specified path
    has the minimum required fields for being a time series,
    that it is labeled as one, and that its ancestry is correct

    Arguments:
        hfile (text) name of nwb file (include path)

        name (text) name of time series

        location (text) path in HDF5 file

        ts_type (text) class name of time series to check for
        (eg, AnnotationSeries)

    Returns:
        *nothing*
    """
    try:
        f = h5py.File(hfile, 'r')
    except IOError as e:
        exc_error("Opening file", e)
    try:
        g = f[location]
    except Exception as e:
        exc_error("Opening group", e)
    try:
        ts = g[name]
    except Exception as e:
        exc_error("Fetching time series", e)
    try:
        nd_type = ts.attrs["neurodata_type"]
    except Exception as e:
        exc_error("reading neurodata_type", e)
    if nd_type != b"TimeSeries" and nd_type != "TimeSeries":
        error("checking neurodata type", "Unexpectedly found type %s, expected 'TimeSeries'" % nd_type)
    try:
        anc = ts.attrs["ancestry"]
    except Exception as e:
        exc_error("Reading ancestry", e)
    if not search_for_string(anc, ts_type):
        print("ts_type is " + ts_type)
        error("Checking ancestry", "Time series is not of type " + ts_type)
    missing = None
    if "missing_fields" in ts.attrs:
        missing = ts.attrs["missing_fields"]
    try:
        samp = ts["num_samples"].value
    except Exception as e:
        if not search_for_substring(missing, "num_samples"):
            error("Reading number of samples", e)
    try:
        samp = ts["data"].value
    except Exception as e:
        if not search_for_substring(missing, "data"):
            exc_error("Reading data", e)
    try:
        samp = ts["timestamps"].value
    except Exception as e:
        if "starting_time" not in ts:
            if not search_for_substring(missing, "timestamps"):
                error("Reading timestamps", str(e))
    f.close()


def verify_present(hfile, group, field):
    """
    Verify that a field is present and returns its contents
    """
    try:
        f = h5py.File(hfile, 'r')
    except IOError as e:
        exc_error("Opening file", e)
    try:
        g = f[group]
    except Exception as e:
        exc_error("Opening group", e)
    if field not in g:
        error("Verifying presence of '"+field+"'", "Field absent")
    obj = g[field]
    if type(obj).__name__ == "Group":
        val = None
    else:
        val = obj.value
    f.close()
    return val


def verify_attribute_present(hfile, obj, field):
    """
    Verify that an attribute is present and returns its contents
    """
    try:
        f = h5py.File(hfile, 'r')
    except IOError as e:
        exc_error("Opening file", e)
    try:
        g = f[obj]
    except Exception as e:
        exc_error("Fetching object", e)
    if field not in g.attrs:
        error("Verifying presence of attribute '"+field+"'", "Field absent")
    val = g.attrs[field]
    f.close()
    return val


def verify_absent(hfile, group, field):
    """
    Verify that a field is not present
    """
    try:
        f = h5py.File(hfile, 'r')
    except IOError as e:
        exc_error("Opening file", e)
    try:
        g = f[group]
    except Exception as e:
        exc_error("Opening group", e)
    if field in g:
        error("Verifying absence of '"+field+"'", "Field exists")
    f.close()

def create_new_file(fname, identifier):
    settings = {}
    settings["filename"] = fname
    settings["identifier"] = nwb.create_identifier(identifier)
    settings["overwrite"] = True
    settings["description"] = "softlink test"
    return nwb.NWB(**settings)


def create_annotation_series(fname, name, target, desc="annotation ",
                             newfile=True):
    """
    Helper function to help tests create example annotations
    """
    settings = {}
    settings["filename"] = fname
    if newfile:
        settings["identifier"] = nwb.create_identifier(desc+"example")
        settings["overwrite"] = True
        settings["start_time"] = "Sat Jul 04 2015 3:14:16"
        settings["description"] = "Test " + desc + "file"
    else:
        settings["modify"] = True
    neurodata = nwb.NWB(**settings)
    #
    annot = neurodata.create_timeseries("AnnotationSeries", name, target)
    annot.set_description("This is an AnnotationSeries '%s' with sample data" % name)
    annot.set_comment("The comment and description fields can store arbitrary human-readable data")
    annot.set_source("Observation of Dr. J Doe")
    #
    annot.add_annotation("Rat in bed, beginning sleep 1", 15.0)
    annot.add_annotation("Rat placed in enclosure, start run 1", 933.0)
    annot.add_annotation("Rat taken out of enclosure, end run 1", 1456.0)
    annot.add_annotation("Rat in bed, start sleep 2", 1461.0)
    annot.add_annotation("Rat placed in enclosure, start run 2", 2401.0)
    annot.add_annotation("Rat taken out of enclosure, end run 2", 3210.0)
    annot.add_annotation("Rat in bed, start sleep 3", 3218.0)
    annot.add_annotation("End sleep 3", 4193.0)
    #
    annot.finalize()
    neurodata.close()

def create_general_extra(fname):
    """
    Create the test file for test_general_extra
    """
    settings = {}
    settings["filename"] = fname
    settings["identifier"] = nwb.create_identifier("general extracellular test")
    settings["overwrite"] = True
    settings["description"] = "test elements in /general/extracellular_ephys"
    neurodata = nwb.NWB(**settings)

    neurodata.set_metadata(EXTRA_ELECTRODE_MAP, [[1, 1, 1], [1, 2, 3]])
    neurodata.set_metadata(EXTRA_ELECTRODE_GROUP, ["p1", "p2"])
    neurodata.set_metadata(EXTRA_IMPEDANCE, [1.0e6, 2.0e6])
    neurodata.set_metadata(EXTRA_FILTERING, "EXTRA_FILTERING")
    neurodata.set_metadata(EXTRA_CUSTOM("EXTRA_CUSTOM"), "EXTRA_CUSTOM")

    neurodata.set_metadata(EXTRA_SHANK_DESCRIPTION("p1"), "DESCRIPTION")
    neurodata.set_metadata(EXTRA_SHANK_LOCATION("p1"), "LOCATION")
    neurodata.set_metadata(EXTRA_SHANK_DEVICE("p1"), "DEVICE")
    neurodata.set_metadata(EXTRA_SHANK_CUSTOM("p1", "extra_shank_custom"), "EXTRA_SHANK_CUSTOM")

    neurodata.set_metadata(EXTRA_SHANK_DESCRIPTION("p2"), "DESCRIPTION")
    neurodata.set_metadata(EXTRA_SHANK_LOCATION("p2"), "LOCATION")
    neurodata.set_metadata(EXTRA_SHANK_DEVICE("p2"), "DEVICE")
    neurodata.set_metadata(EXTRA_SHANK_CUSTOM("p2", "extra_shank_custom"), "EXTRA_SHANK_CUSTOM")

    neurodata.close()


def strcmp(s1, s2):
    if s1 == s2 or s1 == np.bytes_(s2):
        return True
    return False
