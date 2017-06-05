#!/usr/bin/python
import pytest
import sys
import numpy as np
import nwb
import test_utils as ut


def test_annotation_series():
    """
    Test creation of AnnotationSeries and ancestry of TimeSeries
    """
    fname = "x_annotation.nwb"
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
    fname = "x_append.nwb"
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
    fname = "x_epoch_tag.nwb"
    borg = ut.create_new_file(fname, "Epoch tags")
    tags = ["tag-a", "tag-b", "tag-c"]

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
    assert sorted(epoch_tags_read) == sorted(str(tags))  # Compare strings


def test_general_extra():
    """
    Test the implementation of extra metadata fields
    """

    def test_field(fname, name, subdir):
        """
        Helper function to check presence of a field in file
        """
        file_dir = "general/extracellular_ephys/"+subdir+"/"
        val = ut.verify_present(fname, file_dir, name.lower())
        assert val == name or val == np.bytes_(name)

    fname = "x_general_ephys.nwb"
    ut.create_general_extra(fname)

    # Verify electrode map
    val = ut.verify_present(fname, "general/extracellular_ephys", "electrode_map")
    assert len(val) == 2
    assert len(val[0]) == 3

    # Verify electrode group
    val = ut.verify_present(fname, "general/extracellular_ephys", "electrode_group")
    assert len(val) == 2
    assert val[0] == "p1" or val[0] == b"p1"
    assert val[1] == "p2" or val[1] == b"p2"

    # Verify impedance, filtering, and custom fields
    val = ut.verify_present(fname, "general/extracellular_ephys", "impedance")
    assert len(val) == 2
    val = ut.verify_present(fname, "general/extracellular_ephys/", "filtering")
    assert val == "EXTRA_FILTERING" or val == b"EXTRA_FILTERING"
    val = ut.verify_present(fname, "general/extracellular_ephys/", "EXTRA_CUSTOM")
    assert val == "EXTRA_CUSTOM" or val == b"EXTRA_CUSTOM"

    test_field(fname, "DESCRIPTION", "p1")
    test_field(fname, "LOCATION", "p1")
    test_field(fname, "DEVICE", "p1")
    test_field(fname, "EXTRA_SHANK_CUSTOM", "p1")
    test_field(fname, "DESCRIPTION", "p2")
    test_field(fname, "LOCATION", "p2")
    test_field(fname, "DEVICE", "p2")
    test_field(fname, "EXTRA_SHANK_CUSTOM", "p2")




if __name__ == "__main__":
    pytest.main()
