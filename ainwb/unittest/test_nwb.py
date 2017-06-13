#!/usr/bin/python
import pytest
import sys
import h5py
import numpy as np
import nwb
from nwb import nwbco
import test_utils as ut


def test_annotation_series():
    """
    Test creation of AnnotationSeries and ancestry of TimeSeries
    """
    fname = ut.create_test_filename("x_annotation.nwb")
    name = "annot"
    acquisition_loc = "acquisition/timeseries"
    stimulus_loc = "stimulus/presentation"

    ut.create_annotation_series(fname, name, "acquisition")
    ut.verify_timeseries(fname, name, acquisition_loc, "TimeSeries")
    ut.verify_timeseries(fname, name, acquisition_loc, "AnnotationSeries")
    ut.create_annotation_series(fname, name, "stimulus")
    ut.verify_timeseries(fname, name, stimulus_loc, "TimeSeries")
    ut.verify_timeseries(fname, name, stimulus_loc, "AnnotationSeries")


def test_append():
    """
    Test opening file in append mode
    Test modification of existing file
    Test creation of modification_time field
    Test appending of TimeSeries to existing file
    Test preservation of (previous) TimeSeries when modified
    """
    fname = ut.create_test_filename("x_append.nwb")
    name1 = "annot1"
    name2 = "annot2"
    acquisition_loc = "acquisition/timeseries"
    ut.create_annotation_series(fname, name1, "acquisition", "append", True)
    ut.create_annotation_series(fname, name2, "acquisition", "append", False)
    ut.verify_timeseries(fname, name1, acquisition_loc, "TimeSeries")
    ut.verify_timeseries(fname, name1, acquisition_loc, "AnnotationSeries")
    ut.verify_timeseries(fname, name2, acquisition_loc, "TimeSeries")
    ut.verify_timeseries(fname, name2, acquisition_loc, "`AnnotationSeries")
    ut.verify_attribute_present(fname, "file_create_date", "modification_time")


def test_epoch_tag():
    """
    Test creation of two epochs with different tags
    Test creation of main folder with all unique tags & tags assigned to epochs
    """
    fname = ut.create_test_filename("x_epoch_tag.nwb")
    borg = ut.create_new_file(fname, "Epoch tags")
    tags = [b"tag-a", b"tag-b", b"tag-c"]

    # Create two epochs, each with one unique tag and one common tag
    e1_tags_write, e2_tags_write = tags[1:], tags[:-1]
    epoch1 = borg.create_epoch("epoch-1", 0, 3)
    epoch2 = borg.create_epoch("epoch-2", 1, 4)

    for t1, t2 in zip(e1_tags_write, e2_tags_write):
        epoch1.add_tag(t1)
        epoch2.add_tag(t2)

    borg.close()

    # Load the tags from the file and confirm they are unchanged.
    e1_tags_read = ut.verify_attribute_present(fname, "epochs/epoch-1", "tags")
    e2_tags_read = ut.verify_attribute_present(fname, "epochs/epoch-2", "tags")
    epoch_tags_read = ut.verify_attribute_present(fname, "epochs", "tags")

    assert set(e1_tags_read) == set(e1_tags_write)  # Compare sets
    assert set(e2_tags_read) == set(e2_tags_write)  # Compare sets
    if sys.version_info[0] < 3:  # Version determines if tags are bytes or not
        assert sorted(epoch_tags_read) == sorted(str(tags))  # Compare strings
    else:
        assert sorted(epoch_tags_read) == sorted(tags)  # Compare strings


def test_general_extra():
    """
    Test the implementation of extracellular metadata fields
    """

    fname = ut.create_test_filename("x_general_ephys.nwb")
    gen_ephys_dir = "general/extracellular_ephys/"
    ut.create_general_extra(fname)

    # Verify electrode map
    val = ut.verify_present(fname, gen_ephys_dir, "electrode_map")
    assert len(val) == 2
    assert len(val[0]) == 3

    # Verify electrode group
    val = ut.verify_present(fname, gen_ephys_dir, "electrode_group")
    assert len(val) == 2
    assert val[0] == "p1" or val[0] == b"p1"
    assert val[1] == "p2" or val[1] == b"p2"

    # Verify extracellular impedance, filtering, and custom fields
    val = ut.verify_present(fname, gen_ephys_dir, "impedance")
    assert len(val) == 2
    val = ut.verify_present(fname, gen_ephys_dir, "filtering")
    assert val == "EXTRA_FILTERING" or val == b"EXTRA_FILTERING"
    val = ut.verify_present(fname, gen_ephys_dir, "EXTRA_CUSTOM")
    assert val == "EXTRA_CUSTOM" or val == b"EXTRA_CUSTOM"

    # Verify miscellaneous fields
    ut.verify_field(fname, "DESCRIPTION", gen_ephys_dir, "p1")
    ut.verify_field(fname, "LOCATION", gen_ephys_dir, "p1")
    ut.verify_field(fname, "DEVICE", gen_ephys_dir, "p1")
    ut.verify_field(fname, "EXTRA_SHANK_CUSTOM", gen_ephys_dir, "p1")
    ut.verify_field(fname, "DESCRIPTION", gen_ephys_dir, "p2")
    ut.verify_field(fname, "LOCATION", gen_ephys_dir, "p2")
    ut.verify_field(fname, "DEVICE", gen_ephys_dir, "p2")
    ut.verify_field(fname, "EXTRA_SHANK_CUSTOM", gen_ephys_dir, "p2")


def test_general_optophys():
    """
    Test implementation of intracellular optophysiology metadata fields
    """
    fname = ut.create_test_filename("x_general_image.nwb")
    gen_optophys_dir = "general/optophysiology/"
    ut.create_general_optophys(fname)

    val = ut.verify_present(fname, gen_optophys_dir, "image_custom")
    assert ut.strcmp(val, "IMAGE_CUSTOM")

    # Verify field presence
    ut.verify_field(fname, "DESCRIPTION", gen_optophys_dir, "p1")
    ut.verify_field(fname, "DEVICE", gen_optophys_dir, "p1")
    ut.verify_field(fname, "EXCITATION_LAMBDA", gen_optophys_dir, "p1")
    ut.verify_field(fname, "IMAGE_SITE_CUSTOM", gen_optophys_dir, "p1")
    ut.verify_field(fname, "IMAGING_RATE", gen_optophys_dir, "p1")
    ut.verify_field(fname, "INDICATOR", gen_optophys_dir, "p1")
    ut.verify_field(fname, "LOCATION", gen_optophys_dir, "p1")

    # Verify field values
    val = ut.verify_present(fname, gen_optophys_dir+"p1/", "manifold")
    assert len(val) == 2
    assert len(val[0]) == 2
    assert len(val[0][0]) == 3
    val = ut.verify_present(fname, gen_optophys_dir+"p1/red/", "description")
    assert ut.strcmp(val, "DESCRIPTION")
    val = ut.verify_present(fname, gen_optophys_dir+"p1/green/", "description")
    assert ut.strcmp(val, "DESCRIPTION")
    val = ut.verify_present(fname, gen_optophys_dir+"p1/red/", "emission_lambda")
    assert ut.strcmp(val, "CHANNEL_LAMBDA")
    val = ut.verify_present(fname, gen_optophys_dir+"p1/green/", "emission_lambda")
    assert ut.strcmp(val, "CHANNEL_LAMBDA")


def test_general_optogen():
    """
    Test implementation of optogenetics metadata fields
    """
    fname = ut.create_test_filename("x_general_opto.nwb")
    gen_optogen_dir = "general/optogenetics/"
    ut.create_general_optogen(fname)

    val = ut.verify_present(fname, gen_optogen_dir, "optogen_custom")
    assert ut.strcmp(val, "OPTOGEN_CUSTOM")

    ut.verify_field(fname, "DESCRIPTION", gen_optogen_dir, "p1")
    ut.verify_field(fname, "DEVICE", gen_optogen_dir, "p1")
    ut.verify_field(fname, "LAMBDA", gen_optogen_dir, "p1")
    ut.verify_field(fname, "LOCATION", gen_optogen_dir, "p1")
    val = ut.verify_present(fname, gen_optogen_dir+"p1/", "optogen_site_custom")
    assert ut.strcmp(val, "OPTOGEN_SITE_CUSTOM")


def test_general_intra():
    fname = ut.create_test_filename("x_general_intra.nwb")
    gen_intra_dir = "general/intracellular_ephys/"
    ut.create_general_intra(fname)

    val = ut.verify_present(fname, gen_intra_dir, "intra_custom")
    assert ut.strcmp(val, "INTRA_CUSTOM")

    ut.verify_field(fname, "DESCRIPTION", gen_intra_dir, "p1")
    ut.verify_field(fname, "FILTERING", gen_intra_dir, "p1")
    ut.verify_field(fname, "DEVICE", gen_intra_dir, "p1")
    ut.verify_field(fname, "LOCATION", gen_intra_dir, "p1")
    ut.verify_field(fname, "RESISTANCE", gen_intra_dir, "p1")
    ut.verify_field(fname, "SLICE", gen_intra_dir, "p1")
    ut.verify_field(fname, "SEAL", gen_intra_dir, "p1")
    ut.verify_field(fname, "INITIAL_ACCESS_RESISTANCE", gen_intra_dir, "p1")
    ut.verify_field(fname, "INTRA_ELECTRODE_CUSTOM", gen_intra_dir, "p1")

    ut.verify_field(fname, "DESCRIPTION", gen_intra_dir, "e2")
    ut.verify_field(fname, "FILTERING", gen_intra_dir, "e2")
    ut.verify_field(fname, "DEVICE", gen_intra_dir, "e2")
    ut.verify_field(fname, "LOCATION", gen_intra_dir, "e2")
    ut.verify_field(fname, "RESISTANCE", gen_intra_dir, "e2")
    ut.verify_field(fname, "SLICE", gen_intra_dir, "e2")
    ut.verify_field(fname, "SEAL", gen_intra_dir, "e2")
    ut.verify_field(fname, "INITIAL_ACCESS_RESISTANCE", gen_intra_dir, "e2")
    ut.verify_field(fname, "INTRA_ELECTRODE_CUSTOM", gen_intra_dir, "e2")


def test_general_subject():
    """
    Test implementation of subject record fields
    """
    fname = ut.create_test_filename("x_general_species.nwb")
    gen_subj_dir = "general/subject/"
    ut.create_general_subject(fname)
    val = ut.verify_present(fname, gen_subj_dir, "description")

    assert ut.strcmp(val, "SUBJECT")

    ut.verify_field(fname, "SUBJECT_ID", gen_subj_dir)
    ut.verify_field(fname, "SPECIES", gen_subj_dir)
    ut.verify_field(fname, "GENOTYPE", gen_subj_dir)
    ut.verify_field(fname, "SEX", gen_subj_dir)
    ut.verify_field(fname, "AGE", gen_subj_dir)
    ut.verify_field(fname, "WEIGHT", gen_subj_dir)


def test_general_top():
    """
    Test implementation of top-level metadata
    """
    fname = ut.create_test_filename("x_general_top.nwb")
    ut.create_general_top(fname)

    ut.verify_field(fname, "DATA_COLLECTION")
    ut.verify_field(fname, "EXPERIMENT_DESCRIPTION")
    ut.verify_field(fname, "EXPERIMENTER")
    ut.verify_field(fname, "INSTITUTION")
    ut.verify_field(fname, "LAB")
    ut.verify_field(fname, "NOTES")
    ut.verify_field(fname, "PROTOCOL")
    ut.verify_field(fname, "PHARMACOLOGY")
    ut.verify_field(fname, "RELATED_PUBLICATIONS")
    ut.verify_field(fname, "SESSION_ID")
    ut.verify_field(fname, "SLICES")
    ut.verify_field(fname, "STIMULUS")
    ut.verify_field(fname, "SURGERY")
    ut.verify_field(fname, "VIRUS")

    val = ut.verify_present(fname, "general/", "source_script")
    assert len(val) >= 1000
    val = ut.verify_attribute_present(fname, "general/source_script", "neurodata_type")
    assert ut.strcmp(val, "Custom")


def test_add_ts():
    """
    Test opening file in append mode
    Test creation of module and interface
    Test adding timeseries to interface
    """
    fname = ut.create_test_filename("x_if_add_ts.nwb")
    ut.create_iface_series(fname, True)
    name1 = "Ones"
    ut.verify_timeseries(fname, name1, "processing/test module/BehavioralEvents", "TimeSeries")


def test_isi_iface():
    """
    Test storage of retinotopic imaging data
    """
    fname = ut.create_test_filename("x_if_isi.nwb")
    name = "test_module"
    iname = "processing/" + name + "/ImagingRetinotopy"
    ut.create_isi_iface(fname, name)

    ut.verify_axis(fname, iname, "1")
    ut.verify_axis(fname, iname, "2")

    val = ut.verify_present(fname, iname, "axis_descriptions")
    assert len(val) == 2
    assert ut.strcmp(val[0], "altitude") or not ut.strcmp(val[1], "azimuth")

    ut.verify_image(fname, iname, "vasculature_image")
    ut.verify_image(fname, iname, "focal_depth_image")
    ut.verify_sign_map(fname, iname)


def test_file_modification():
    """
    Test file modification and logging of multiple modification datetimes
    """
    fname = ut.create_test_filename("x_modification_time.nwb")

    # File creation
    settings = {}
    settings["filename"] = fname
    settings["identifier"] = nwb.create_identifier("Modification example")
    settings["overwrite"] = True
    settings["description"] = "Modified empty file"
    settings["start_time"] = "Sat Jul 04 2015 3:14:16"
    neurodata = nwb.NWB(**settings)
    neurodata.close()

    # First modification
    settings = {}
    settings["filename"] = fname
    settings["overwrite"] = False
    settings["modify"] = True
    neurodata = nwb.NWB(**settings)
    neurodata.set_metadata(nwbco.INSTITUTION, "Allen Institute for Brain Science")
    neurodata.close()

    # Second modification
    settings = {}
    settings["filename"] = fname
    settings["overwrite"] = False
    settings["modify"] = True
    attrs = {}
    attrs["phrasebook"] = "Sir William, I cannot wait until lunchtime"
    neurodata.set_metadata(nwbco.EXPERIMENT_DESCRIPTION, "My hovercraft is full of eels", **attrs)
    neurodata = nwb.NWB(**settings)
    neurodata.close()

    # Check the number of modification dates
    with h5py.File(fname) as f:
        dates = f["file_create_date"]
        assert len(dates) == 3
        assert dates[0] <= dates[1]
        assert dates[1] <= dates[2]


def test_nodata_series():
    """
    Test implementation of NWB.TimeSeries.ignore_data
    Test acknowledgment of missing 'data' attribute
    """
    fname = ut.create_test_filename("x_no_data.nwb")
    name = "nodata"
    ut.create_nodata_series(fname, name, "acquisition")
    ut.verify_timeseries(fname, name, "acquisition/timeseries", "TimeSeries")
    ut.verify_absent(fname, "acquisition/timeseries/"+name, "data")


def test_notime_series():
    """
    Test time series placement in acquisition, template, and stimulus attributes
    Test NWB.TimeSeries.ignore_time
    """
    fname = ut.create_test_filename("x_no_time.nwb")
    name = "notime"
    ut.create_notime_series(fname, name, "acquisition")
    ut.verify_timeseries(fname, name, "acquisition/timeseries", "TimeSeries")
    ut.verify_absent(fname, "acquisition/timeseries/"+name, "timestamps")
    ut.verify_absent(fname, "acquisition/timeseries/"+name, "starting_time")

    ut.create_notime_series(fname, name, "stimulus")
    ut.verify_timeseries(fname, name, "stimulus/presentation", "TimeSeries")
    ut.create_notime_series(fname, name, "template")
    ut.verify_timeseries(fname, name, "stimulus/templates", "TimeSeries")


def test_refimage_series():
    """
    Test implementation of reference image storage
    """
    fname = ut.create_test_filename("x_ref_image.nwb")
    name = "refimage"
    ut.create_refimage(fname, name)
    val = ut.verify_present(fname, "acquisition/images/", name)
    assert len(val) == 5
    val = ut.verify_attribute_present(fname, "acquisition/images/"+name, "format")
    assert ut.strcmp(val, "raw")
    val = ut.verify_attribute_present(fname, "acquisition/images/"+name, "description")
    assert ut.strcmp(val, "test")

    val = ut.verify_present(fname, "/", "identifier")
    assert ut.strcmp(val, nwb.create_identifier("reference image test"))
    ut.verify_present(fname, "/", "file_create_date")
    val = ut.verify_present(fname, "/", "session_start_time")
    assert ut.strcmp(val, "xyz")
    val = ut.verify_present(fname, "/", "session_description")
    assert ut.strcmp(val, "reference image test")


def test_softlink():
    """
    Test implementation of time series with softlinked data source
    """
    fname1 = ut.create_test_filename("x_softlink1.nwb")
    fname2 = ut.create_test_filename("x_softlink2.nwb")
    name1 = "softlink_source"
    name2 = "softlink_reader"
    ut.create_softlink_source(fname1, name1, "acquisition")
    ut.create_softlink_reader(fname2, name2, fname1, name1, "acquisition")

    ut.verify_timeseries(fname1, name1, "acquisition/timeseries", "TimeSeries")
    ut.verify_timeseries(fname2, name2, "acquisition/timeseries", "TimeSeries")

    val = ut.verify_present(fname2, "acquisition/timeseries/"+name2, "data")


def test_starting_time():
    """
    Test implementation of NWB.TimeSeries.starting_time attribute
    """
    fname = ut.create_test_filename("x_starting_time.nwb")
    name = "starting_time"
    ut.create_startingtime_series(fname, name, "acquisition")
    ut.verify_timeseries(fname, name, "acquisition/timeseries", "TimeSeries")
    ut.verify_absent(fname, "acquisition/timeseries/"+name, "timestamps")

    val = ut.verify_present(fname, "acquisition/timeseries/"+name, "starting_time")
    assert val - 0.125 < 1e6  # An == check for floating point values
    val = ut.verify_attribute_present(fname, "acquisition/timeseries/starting_time/"+name, "rate")
    assert val == 2


def test_ts_link():
    """
    Test implementation of time series linking and hardlinking for data and timestamps
    """
    fname = ut.create_test_filename("x_timeseries_link.nwb")
    root = "root"
    ut.create_linked_series(fname, root)

    # Verify the three time series are present
    ut.verify_timeseries(fname, root+"1", "stimulus/templates", "TimeSeries")
    ut.verify_timeseries(fname, root+"2", "stimulus/presentation", "TimeSeries")
    ut.verify_timeseries(fname, root+"3", "acquisition/timeseries", "TimeSeries")

    # Verify that the second time series data is properly linked to the first
    val = ut.verify_present(fname, "stimulus/presentation/root2", "data")
    assert val[0] == 1
    val = ut.verify_attribute_present(fname, "stimulus/presentation/root2", "data_link")
    assert ut.substring_is_present(val, "root1")
    assert ut.substring_is_present(val, "root2")
    val = ut.verify_attribute_present(fname, "stimulus/templates/root1", "data_link")
    assert ut.substring_is_present(val, "root1")
    assert ut.substring_is_present(val, "root2")

    # Verify the timestamps of the third time series are properly linked to
    # the second
    val = ut.verify_present(fname, "acquisition/timeseries/root3", "timestamps")
    assert val[0] == 2
    val = ut.verify_attribute_present(fname, "stimulus/presentation/root2", "timestamp_link")
    assert ut.substring_is_present(val, "root2")
    assert ut.substring_is_present(val, "root3")
    val = ut.verify_attribute_present(fname, "acquisition/timeseries/root3", "timestamp_link")
    assert ut.substring_is_present(val, "root2")
    assert ut.substring_is_present(val, "root3")


def test_unit_times():
    """
    Test implementation of UnitTimes interface
    """
    fname = ut.create_test_filename("x_unittimes.nwb")

    # Create test file containing a spike data module with UnitTimes interface
    neurodata = ut.create_spike_data(fname)
    mod = neurodata.create_module("my spike times")
    iface = mod.create_interface("UnitTimes")

    # Add synthetic spike data
    spikes = ut.create_spikes()
    for i, spk in enumerate(spikes):
        iface.add_unit(unit_name="unit-%d" % i,
                       unit_times=spk,
                       description="<description of unit>",
                       source="Data spike-sorted by B. Bunny")

    # Finalize interface and module, then close file
    iface.finalize()
    mod.finalize()
    neurodata.close()

    # Verify that synthetic data was stored correctly.
    h5 = h5py.File(fname)
    times = h5["processing/my spike times/UnitTimes/unit-0/times"].value
    assert len(times) == len(spikes[0]), "Spike count for unit-0 wrong"
    assert abs(times[1] - spikes[0][1]) < 0.001, "Wrong time found in file"
    h5.close()

########################### TESTS NOT YET IMPLEMENTED #########################
# Prepend "test_" to the name of a function to have pytest run it             #
###############################################################################


def append_timeseries():
    """
    create time series in existing file and link 'data' to existing series
    Test TimeSeries link annotation in append mode
    Test preservance of existing links (hard/soft/both)
    Test appending new links (hard/soft/both)
    Test creation of new links (hard/soft/both)
    """
    # TODO


def epoch_link_overlay():
    """
    create epoch and add time series, examining overlap
    Test boundary checking (start_idx, count) for epoch.add_timeseries()
    """
    # TODO


def epoch_link():
    """
    create epoch and add time series
    Test epoch.add_timeseries()
    """
    # TODO


def epoch():
    """
    create epoch in file
    TESTS epoch creation
    """
    # TODO


def ts_hardsoft_link():
    """
    (No text)
    """
    # TODO


if __name__ == "__main__":
    pytest.main()
