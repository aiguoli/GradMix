import logging
from datetime import datetime


def get_logger(name: str="HOGMask"):
    logging.basicConfig(
        filename="logs/{}_{}_.log".format(datetime.now().strftime("%Y-%m-%d_%H-%M"), name),
        format="%(asctime)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.DEBUG,
    )
    return logging.getLogger(name)


if __name__ == "__main__":
    get_logger().debug("Hello World")
