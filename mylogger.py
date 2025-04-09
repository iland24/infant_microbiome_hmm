import logging
import os

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# set up logger to directory of choice
outdir='output/'
if not os.path.exists(outdir) and not os.path.isdir(outdir):
    os.mkdir(outdir)

logging.basicConfig(
    filename = outdir+"output.log",  # Log file
    level = logging.INFO,  # Logging level
    format = "%(asctime)s: %(message)s",
    datefmt = "%b/%d %H:%M"
)

logger = logging.getLogger(__name__)