#!/usr/bin/python
import nwb
import test_utils as ut

# create multiple time series and link data and timestamps to between them
# TESTS linking of TimeSeries.data
# TESTS annotation of TimeSeries.data link
# TESTS linking of TimeSeries.timestamps
# TESTS annotation of TimeSeries.timestamps link


def test_ts_link():
    #fname = "x_nodata_series_acq.nwb"
    fname = "x" + __file__[3:-3] + ".nwb"
    root = "root"
    create_linked_series(fname, root)
    ut.verify_timeseries(fname, root+"1", "stimulus/templates", "TimeSeries")
    ut.verify_timeseries(fname, root+"2", "stimulus/presentation", "TimeSeries")
    ut.verify_timeseries(fname, root+"3", "acquisition/timeseries", "TimeSeries")
    ##################################################
    # make sure data is present in ts using link
    ut.verify_present(fname, "stimulus/presentation/root2", "data")
    # make sure link is documented
    val = ut.verify_attribute_present(fname, "stimulus/presentation/root2", "data_link")
    if ut.search_for_string(val, "root1") < 0:
        ut.error("Checking attribute data_link", "Name missing")
    if ut.search_for_string(val, "root2") < 0:
        ut.error("Checking attribute data_link", "Name missing")
    val = ut.verify_attribute_present(fname, "stimulus/templates/root1", "data_link")
    if ut.search_for_string(val, "root1") < 0:
        ut.error("Checking attribute data_link", "Name missing")
    if ut.search_for_string(val, "root2") < 0:
        ut.error("Checking attribute data_link", "Name missing")
    ##################################################
    # make sure timestamps is present in ts using link
    ut.verify_present(fname, "acquisition/timeseries/root3", "timestamps")
    # make sure link is documented
    val = ut.verify_attribute_present(fname, "stimulus/presentation/root2", "timestamp_link")
    if ut.search_for_string(val, "root2") < 0:
        ut.error("Checking attribute timestamp_link", "Name missing")
    if ut.search_for_string(val, "root3") < 0:
        ut.error("Checking attribute timestamp_link", "Name missing")
    val = ut.verify_attribute_present(fname, "acquisition/timeseries/root3", "timestamp_link")
    if ut.search_for_string(val, "root2") < 0:
        ut.error("Checking attribute timestamp_link", "Name missing")
    if ut.search_for_string(val, "root3") < 0:
        ut.error("Checking attribute timestamp_link", "Name missing")

def create_linked_series(fname, root):
    settings = {}
    settings["filename"] = fname
    settings["identifier"] = nwb.create_identifier("nodata example")
    settings["overwrite"] = True
    settings["description"] = "time series no data test"
    settings["start_time"] = "Sat Jul 04 2015 3:14:16"
    neurodata = nwb.NWB(**settings)
    #
    first = neurodata.create_timeseries("TimeSeries", root+"1", "template")
    first.ignore_time()
    first.set_value("num_samples", 1)
    first.set_data([0])
    first.finalize()
    #
    second = neurodata.create_timeseries("TimeSeries", root+"2", "stimulus")
    second.set_time([0])
    second.set_value("num_samples", 1)
    second.set_data_as_link(first)
    second.finalize()
    #
    third = neurodata.create_timeseries("TimeSeries", root+"3", "acquisition")
    third.set_time_as_link(second)
    third.set_value("num_samples", 1)
    third.set_data([0])
    third.finalize()
    #
    neurodata.close()

test_ts_link()
print "%s PASSED" % __file__
