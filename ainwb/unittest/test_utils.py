
import sys
import os.path
from tempfile import gettempdir
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


def create_test_filename(fname="test.nwb"):
    """
    Creates a filename with the given fname(string) in a temporary directory
    """
    return os.path.join(gettempdir(), fname)


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


def substring_is_present(h5_str, value):
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
#        return substring_is_present(h5_str, np.bytes_(value))
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
        if not substring_is_present(missing, "num_samples"):
            error("Reading number of samples", e)
    try:
        samp = ts["data"].value
    except Exception as e:
        if not substring_is_present(missing, "data"):
            exc_error("Reading data", e)
    try:
        samp = ts["timestamps"].value
    except Exception as e:
        if "starting_time" not in ts:
            if not substring_is_present(missing, "timestamps"):
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


def verify_field(fname, name, topdir="general/", subdir=''):
    """
    Helper function to check presence of a field in file
    """
    val = verify_present(fname, topdir+subdir+'/', name.lower())
    assert val == name or val == np.bytes_(name)


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


def create_general_optophys(fname):
    """
    Create the test file for test_general_optophys
    """
    settings = {}
    settings["filename"] = fname
    settings["identifier"] = nwb.create_identifier("general optophysiology test")
    settings["overwrite"] = True
    settings["description"] = "test elements in /general/optophysiology"
    neurodata = nwb.NWB(**settings)

    neurodata.set_metadata(IMAGE_CUSTOM("image_custom"), "IMAGE_CUSTOM")
    neurodata.set_metadata(IMAGE_SITE_DESCRIPTION("p1"), "DESCRIPTION")
    neurodata.set_metadata(IMAGE_SITE_MANIFOLD("p1"), [[[1,2,3],[2,3,4]],[[3,4,5],[4,5,6]]])
    neurodata.set_metadata(IMAGE_SITE_INDICATOR("p1"), "INDICATOR")
    neurodata.set_metadata(IMAGE_SITE_EXCITATION_LAMBDA("p1"), "EXCITATION_LAMBDA")
    neurodata.set_metadata(IMAGE_SITE_CHANNEL_LAMBDA("p1", "red"), "CHANNEL_LAMBDA")
    neurodata.set_metadata(IMAGE_SITE_CHANNEL_DESCRIPTION("p1", "red"), "DESCRIPTION")
    neurodata.set_metadata(IMAGE_SITE_CHANNEL_LAMBDA("p1", "green"), "CHANNEL_LAMBDA")
    neurodata.set_metadata(IMAGE_SITE_CHANNEL_DESCRIPTION("p1", "green"), "DESCRIPTION")
    neurodata.set_metadata(IMAGE_SITE_IMAGING_RATE("p1"), "IMAGING_RATE")
    neurodata.set_metadata(IMAGE_SITE_LOCATION("p1"), "LOCATION")
    neurodata.set_metadata(IMAGE_SITE_DEVICE("p1"), "DEVICE")
    neurodata.set_metadata(IMAGE_SITE_CUSTOM("p1", "image_site_custom"), "IMAGE_SITE_CUSTOM")

    neurodata.close()


def create_general_optogen(fname):
    """
    Create the test file for test_general_optogen
    """
    settings = {}
    settings["filename"] = fname
    settings["identifier"] = nwb.create_identifier("metadata optogenetic test")
    settings["overwrite"] = True
    settings["description"] = "test elements in /general/optogentics"
    neurodata = nwb.NWB(**settings)

    neurodata.set_metadata(OPTOGEN_CUSTOM("optogen_custom"), "OPTOGEN_CUSTOM")

    neurodata.set_metadata(OPTOGEN_SITE_DESCRIPTION("p1"), "DESCRIPTION")
    neurodata.set_metadata(OPTOGEN_SITE_DEVICE("p1"), "DEVICE")
    neurodata.set_metadata(OPTOGEN_SITE_LAMBDA("p1"), "LAMBDA")
    neurodata.set_metadata(OPTOGEN_SITE_LOCATION("p1"), "LOCATION")
    neurodata.set_metadata(OPTOGEN_SITE_CUSTOM("p1", "optogen_site_custom"), "OPTOGEN_SITE_CUSTOM")
    #
    neurodata.close()


def create_general_intra(fname):
    """
    Create test file for test_general_intra
    """
    settings = {}
    settings["filename"] = fname
    settings["identifier"] = nwb.create_identifier("general intracellular test")
    settings["overwrite"] = True
    settings["description"] = "test elements in /general/intracellular_ephys"
    neurodata = nwb.NWB(**settings)
    #
    neurodata.set_metadata(INTRA_CUSTOM("intra_custom"), "INTRA_CUSTOM")
    #
    neurodata.set_metadata(INTRA_ELECTRODE_DESCRIPTION("p1"), "DESCRIPTION")
    neurodata.set_metadata(INTRA_ELECTRODE_FILTERING("p1"), "FILTERING")
    neurodata.set_metadata(INTRA_ELECTRODE_DEVICE("p1"), "DEVICE")
    neurodata.set_metadata(INTRA_ELECTRODE_LOCATION("p1"), "LOCATION")
    neurodata.set_metadata(INTRA_ELECTRODE_RESISTANCE("p1"), "RESISTANCE")
    neurodata.set_metadata(INTRA_ELECTRODE_SEAL("p1"), "SEAL")
    neurodata.set_metadata(INTRA_ELECTRODE_SLICE("p1"), "SLICE")
    neurodata.set_metadata(INTRA_ELECTRODE_INIT_ACCESS_RESISTANCE("p1"), "INITIAL_ACCESS_RESISTANCE")
    neurodata.set_metadata(INTRA_ELECTRODE_CUSTOM("p1", "intra_electrode_custom"), "INTRA_ELECTRODE_CUSTOM")
    #
    neurodata.set_metadata(INTRA_ELECTRODE_DESCRIPTION("e2"), "DESCRIPTION")
    neurodata.set_metadata(INTRA_ELECTRODE_FILTERING("e2"), "FILTERING")
    neurodata.set_metadata(INTRA_ELECTRODE_DEVICE("e2"), "DEVICE")
    neurodata.set_metadata(INTRA_ELECTRODE_LOCATION("e2"), "LOCATION")
    neurodata.set_metadata(INTRA_ELECTRODE_RESISTANCE("e2"), "RESISTANCE")
    neurodata.set_metadata(INTRA_ELECTRODE_SEAL("e2"), "SEAL")
    neurodata.set_metadata(INTRA_ELECTRODE_SLICE("e2"), "SLICE")
    neurodata.set_metadata(INTRA_ELECTRODE_INIT_ACCESS_RESISTANCE("e2"), "INITIAL_ACCESS_RESISTANCE")
    neurodata.set_metadata(INTRA_ELECTRODE_CUSTOM("e2", "intra_electrode_custom"), "INTRA_ELECTRODE_CUSTOM")
    #
    neurodata.close()


def create_general_subject(fname):
    """
    Create test file for test_general_subject
    """
    settings = {}
    settings["filename"] = fname
    settings["identifier"] = nwb.create_identifier("general top test")
    settings["overwrite"] = True
    settings["description"] = "test top-level elements in /general"
    neurodata = nwb.NWB(**settings)
    #
    neurodata.set_metadata(SUBJECT, "SUBJECT")
    neurodata.set_metadata(SUBJECT_ID, "SUBJECT_ID")
    neurodata.set_metadata(SPECIES, "SPECIES")
    neurodata.set_metadata(GENOTYPE, "GENOTYPE")
    neurodata.set_metadata(SEX, "SEX")
    neurodata.set_metadata(AGE, "AGE")
    neurodata.set_metadata(WEIGHT, "WEIGHT")
    #
    neurodata.close()


def create_general_top(fname):
    settings = {}
    settings["filename"] = fname
    settings["identifier"] = nwb.create_identifier("general top test")
    settings["overwrite"] = True
    settings["description"] = "test top-level elements in /general"
    neurodata = nwb.NWB(**settings)

    neurodata.set_metadata(DATA_COLLECTION, "DATA_COLLECTION")
    neurodata.set_metadata(EXPERIMENT_DESCRIPTION, "EXPERIMENT_DESCRIPTION")
    neurodata.set_metadata(EXPERIMENTER, "EXPERIMENTER")
    neurodata.set_metadata(INSTITUTION, "INSTITUTION")
    neurodata.set_metadata(LAB, "LAB")
    neurodata.set_metadata(NOTES, "NOTES")
    neurodata.set_metadata(PROTOCOL, "PROTOCOL")
    neurodata.set_metadata(PHARMACOLOGY, "PHARMACOLOGY")
    neurodata.set_metadata(RELATED_PUBLICATIONS, "RELATED_PUBLICATIONS")
    neurodata.set_metadata(SESSION_ID, "SESSION_ID")
    neurodata.set_metadata(SLICES, "SLICES")
    neurodata.set_metadata(STIMULUS, "STIMULUS")
    neurodata.set_metadata(SURGERY, "SURGERY")
    neurodata.set_metadata(VIRUS, "VIRUS")

    neurodata.set_metadata_from_file("source_script", __file__)
    neurodata.close()


def create_iface_series(fname, newfile):
    """
    Create test file for test_add_ts
    """
    settings = {}
    settings["filename"] = fname
    if newfile:
        settings["identifier"] = nwb.create_identifier("interface timeseries example")
        settings["overwrite"] = True
        settings["start_time"] = "Sat Jul 04 2015 3:14:16"
        settings["description"] = "Test interface timeseries file"
    else:
        settings["modify"] = True
    neurodata = nwb.NWB(**settings)

    mod = neurodata.create_module("test module")
    iface = mod.create_interface("BehavioralEvents")
    ts = neurodata.create_timeseries("TimeSeries", "Ones")
    ts.set_data(np.ones(10), unit="Event", conversion=1.0, resolution=float('nan'))
    ts.set_value("num_samples", 10)
    ts.set_time(np.arange(10))
    iface.add_timeseries(ts)
    iface.finalize()
    mod.finalize()

    neurodata.close()


def create_isi_iface(fname, name):
    """
    Create test file for test_isi_iface
    """
    settings = {}
    settings["filename"] = fname
    settings["identifier"] = nwb.create_identifier("reference image test")
    settings["overwrite"] = True
    settings["description"] = "reference image test"
    neurodata = nwb.NWB(**settings)
    module = neurodata.create_module(name)
    iface = module.create_interface("ImagingRetinotopy")
    iface.add_axis_1_phase_map([[1.0, 1.1, 1.2],[2.0,2.1,2.2]], "altitude", .1, .1)
    iface.add_axis_2_phase_map([[3.0, 3.1, 3.2],[4.0,4.1,4.2]], "azimuth", .1, .1, unit="degrees")
    iface.add_axis_1_power_map([[0.1, 0.2, 0.3],[0.4, 0.5, 0.6]], .1, .1)
    iface.add_sign_map([[-.1, .2, -.3],[.4,-.5,.6]])
    iface.add_vasculature_image([[1,0,129],[2,144,0]], height=.22, width=.35)
    iface.add_focal_depth_image([[1,0,129],[2,144,0]], bpp=8)
    iface.finalize()
    module.finalize()
    neurodata.close()


def verify_axis(fname, iname, num):
    val = verify_present(fname, iname, "axis_"+num+"_phase_map")
    assert len(val) == 2
    assert len(val[0]) == 3

    if num == "1":
        assert val[0][0] == 1.0
    elif num == "2":
        assert val[0][0] == 3.0

    val = verify_attribute_present(fname, iname+"/axis_"+num+"_phase_map", "unit")
    assert strcmp(val, "degrees")

    val = verify_attribute_present(fname, iname+"/axis_"+num+"_phase_map", "dimension")
    assert val[0] == 2
    assert val[1] == 3

    val = verify_attribute_present(fname, iname+"/axis_"+num+"_phase_map", "field_of_view")
    assert val[0] == .1
    assert val[1] == .1

    # Check power map, which only exists for axis 1
    if num == "1":
        val = verify_present(fname, iname, "axis_"+num+"_power_map")
        assert len(val) == 2
        assert len(val[0]) == 3

        val = verify_attribute_present(fname, iname+"/axis_"+num+"_power_map", "dimension")
        assert val[0] == 2
        assert val[1] == 3

        val = verify_attribute_present(fname, iname+"/axis_"+num+"_power_map", "field_of_view")
        assert val[0] == .1
        assert val[1] == .1


def verify_image(fname, iname, img):
    """
    Helper function for test_isi_iface
    """
    val = verify_present(fname, iname, img)
    assert len(val) == 2
    assert len(val[0]) == 3
    assert val[1][1] == 144

    val = verify_attribute_present(fname, iname+"/"+img, "format")
    assert strcmp(val, "raw")

    val = verify_attribute_present(fname, iname+"/"+img, "dimension")
    assert len(val) == 2
    assert val[0] == 2
    assert val[1] == 3

    val = verify_attribute_present(fname, iname+"/"+img, "bits_per_pixel")
    assert val == 8


def verify_sign_map(fname, iname):
    """
    Helper function for test_isi_iface
    """
    val = verify_present(fname, iname, "sign_map")
    assert len(val) == 2
    assert len(val[0]) == 3
    assert val[1][1] == -.5

    val = verify_attribute_present(fname, iname+"/sign_map", "dimension")
    assert len(val) == 2
    assert val[0] == 2
    assert val[1] == 3


def create_nodata_series(fname, name, target):
    """
    Creates test file for test_nodata_series
    """
    settings = {}
    settings["filename"] = fname
    settings["identifier"] = nwb.create_identifier("nodata example")
    settings["overwrite"] = True
    settings["description"] = "time series no data test"
    settings["start_time"] = "Sat Jul 04 2015 3:14:16"
    neurodata = nwb.NWB(**settings)

    nodata = neurodata.create_timeseries("TimeSeries", name, target)
    nodata.ignore_data()
    nodata.set_time([0])

    nodata.finalize()
    neurodata.close()


def create_notime_series(fname, name, target):
    """
    Create test file for test_notime_series
    """
    settings = {}
    settings["filename"] = fname
    settings["identifier"] = nwb.create_identifier("notime example")
    settings["overwrite"] = True
    settings["start_time"] = "Sat Jul 04 2015 3:14:16"
    settings["description"] = "Test no time"
    neurodata = nwb.NWB(**settings)

    notime = neurodata.create_timeseries("TimeSeries", name, target)
    notime.ignore_time()
    notime.set_data([0], unit="n/a", conversion=1, resolution=1)

    notime.finalize()
    neurodata.close()


def create_refimage(fname, name):
    """
    Create test file for test_refimage_series and test_
    """
    settings = {}
    settings["filename"] = fname
    settings["identifier"] = nwb.create_identifier("reference image test")
    settings["overwrite"] = True
    settings["description"] = "reference image test"
    settings["start_time"] = "xyz"
    neurodata = nwb.NWB(**settings)
    neurodata.create_reference_image([1, 2, 3, 4, 5], name, "raw", "test")
    neurodata.close()


def create_softlink_reader(fname, name, src_fname, src_name, target):
    """
    Create test file (with softlinked data) for test_softlink
    """
    settings = {}
    settings["filename"] = fname
    settings["identifier"] = nwb.create_identifier("softlink reader")
    settings["overwrite"] = True
    settings["description"] = "softlink test"
    neurodata = nwb.NWB(**settings)
    source = neurodata.create_timeseries("TimeSeries", name, target)
    source.set_data_as_remote_link(src_fname, "acquisition/timeseries/"+src_name+"/data")
    source.set_time([345])
    source.finalize()
    neurodata.close()


def create_softlink_source(fname, name, target):
    """
    Create softlinked data source for test_softlink
    """
    settings = {}
    settings["filename"] = fname
    settings["identifier"] = nwb.create_identifier("softlink source")
    settings["overwrite"] = True
    settings["description"] = "time series no data test"
    settings["start_time"] = "Sat Jul 04 2015 3:14:16"
    neurodata = nwb.NWB(**settings)
    source = neurodata.create_timeseries("TimeSeries", name, target)
    source.set_data([234], unit="parsec", conversion=1, resolution=1e-3)
    source.set_time([123])
    source.finalize()
    neurodata.close()


def create_startingtime_series(fname, name, target):
    """
    Create test file for test_starting_time
    """
    settings = {}
    settings["filename"] = fname
    settings["identifier"] = nwb.create_identifier("starting time test")
    settings["overwrite"] = True
    settings["description"] = "time series starting time test"
    settings["start_time"] = "Sat Jul 04 2015 3:14:16"
    neurodata = nwb.NWB(**settings)

    stime = neurodata.create_timeseries("TimeSeries", name, target)
    stime.set_data([0, 1, 2, 3], unit="n/a", conversion=1, resolution=1)
    stime.set_value("num_samples", 4)
    stime.set_time_by_rate(0.125, 2)

    stime.finalize()
    neurodata.close()

def create_linked_series(fname, root):
    """
    Create test file for test_ts_link
    """
    settings = {}
    settings["filename"] = fname
    settings["identifier"] = nwb.create_identifier("link test")
    settings["overwrite"] = True
    settings["description"] = "time series link test"
    settings["start_time"] = "Sat Jul 04 2015 3:14:16"
    neurodata = nwb.NWB(**settings)

    first = neurodata.create_timeseries("TimeSeries", root+"1", "template")
    first.ignore_time()
    first.set_value("num_samples", 1)
    first.set_data([1], unit="parsec", conversion=1, resolution=1e-12)
    first.finalize()

    second = neurodata.create_timeseries("TimeSeries", root+"2", "stimulus")
    second.set_time([2])
    second.set_value("num_samples", 1)
    second.set_data_as_link(first)
    second.finalize()

    third = neurodata.create_timeseries("TimeSeries", root+"3", "acquisition")
    third.set_time_as_link(second)
    third.set_value("num_samples", 1)
    third.set_data([3], unit="parsec", conversion=1, resolution=1e-9)
    third.finalize()

    neurodata.close()

def create_spikes():
    spikes = []
    spikes.append([1.3, 1.4, 1.9, 2.1, 2.2, 2.3])
    spikes.append([2.2, 3.0])
    spikes.append([0.3, 0.4, 1.0, 1.1, 1.45, 1.8, 1.81, 2.2])
    return spikes


def create_spike_data(fname):
    settings = {}
    settings["filename"] = fname
    settings["identifier"] = nwb.create_identifier("UnitTimes example")
    settings["overwrite"] = True
    settings["start_time"] = "Sat Jul 04 2015 3:14:16"
    settings["description"] = "Test file with spike times in processing module"
    neurodata = nwb.NWB(**settings)
    return neurodata


def strcmp(s1, s2):
    """
    Python 3 handles all strings as unicode by default, so we perform a more
    rigorous check here to make the module Python 2- and 3-compatible.
    """
    return s1 == s2 or s1 == np.bytes_(s2)
