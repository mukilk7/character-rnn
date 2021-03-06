#!/usr/bin/env python

"""
This should be the main and only entry point
to running all functionality implemented in
this project.
"""

__author__ = "Mukil Kesavan"

import sys

#Setup project-wide logging
#NOTE: This should *ONLY* be done here!
from customlogging import log
logger = log.setup_logging()

from optparse import OptionParser
from charrnn import master
from charrnn.data_processor import LocalDataProcessor
from charrnn.configs import ModelParams
from charrnn.model import CharacterRNN
import common.utils as utils

_CMD_CHOICES = [
    "train",
    "generate",
    "anomaly-detect",
]


def init_cmdline_args():
    """
    Initialize command line arguments.
    """
    parser = OptionParser(usage="Usage %prog [options]")

    parser.add_option("--command", "-c", help="command to run %s" % _CMD_CHOICES,
                      type="choice", choices=_CMD_CHOICES, action="store", dest="cmd")
    parser.add_option("--num-epochs", help="train model for specified epochs",
                      action="store", dest="num_epochs")
    parser.add_option("--num-chars",
                      help="generate specified num of chars using trained model",
                      action="store", dest="num_chars_to_gen")
    parser.add_option("--input-file",
                      help="input file for train or test",
                      action="store", dest="input_file")
    parser.add_option("--input-url",
                      help="input data url for train or test (can be http or gs)",
                      action="store", dest="input_url")
    parser.add_option("--anomaly-threshold",
                      help="the percentile threshold for anomaly detection (0-100)",
                      action="store", dest="anomaly_threshold")
    return parser


def cmd_router(cmd, opts):
    """
    Call appropriate module/function based on
    command requested.
    """
    utils.reset_tensorboard_logs()
    if cmd == "train":
        default_config = ModelParams()
        data_processor = LocalDataProcessor(default_config,
                                            input_url=opts.input_url,
                                            local_filename=opts.input_file)
        num_epochs = int(opts.num_epochs)
        logger.info("Training epochs set to %s", num_epochs)
        model = CharacterRNN(data_processor.num_classes, default_config).build_computation_graph()
        master.train(data_processor, model, num_epochs)
        logger.info("Training complete. Model saved!")
    elif cmd == "generate":
        num_chars = int(opts.num_chars_to_gen)
        output = master.generate_text_from_model(num_chars)
        logger.info("Generated Text\n---\n%s\n---\n", output)
    elif cmd == "anomaly-detect":
        logger.info("Average Line Loss is an Experimental Feature")
        anomaly_threshold = 95
        if opts.anomaly_threshold is not None:
            anomaly_threshold = int(opts.anomaly_threshold)
        anomaly_lines = master.compute_average_line_loss(local_filename=opts.input_file,
                                                         anomaly_percentile=anomaly_threshold,
                                                         input_url=opts.input_url)
        logger.info("*** Anomaly Lines***\n----\n%s----", "".join(anomaly_lines))


def main():
    parser = init_cmdline_args()
    (opts, args) = parser.parse_args()
    if len(sys.argv[1:]) <= 0:
        parser.print_help()
        sys.exit(0)
    cmd_router(opts.cmd.lower(), opts)


if __name__ == '__main__':
    main()
