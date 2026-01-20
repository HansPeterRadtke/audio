"""
Simple Mumble client for connecting to the mermer-server.
- Uses pymumble to connect to a fixed IP.
- Prints incoming text messages and sends an initial greeting.
- Configurable via environment variable MERMER_SERVER_IP or CLI args.

Install dependency: pip install pymumble
"""

import os
import time
import argparse
import logging
import socket
from pymumble_py3 import Mumble
from pymumble_py3.constants import PYMUMBLE_CLBK_TEXTMESSAGERECEIVED

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_PORT = 64738
DEFAULT_USER = "mermer-client"

class MumbleClient:
    def __init__(self, server, port=DEFAULT_PORT, user=DEFAULT_USER, password=None, channel=None):
        self.server = server
        self.port = port
        self.user = user
        self.password = password if password is not None else ""
        self.channel = channel
        self.mumble = None

    def _on_text_received(self, text_message):
        # text_message is a dictionary-like object in pymumble
        try:
            actor = text_message.actor
            message = text_message.message
        except Exception:
            # older/newer pymumble shapes may differ; attempt mapping
            actor = getattr(text_message, 'actor', None) or text_message.get('actor')
            message = getattr(text_message, 'message', None) or text_message.get('message')

        logger.info("Text message from %s: %s", actor, message)
        print(f"[TEXT] from {actor}: {message}")

    def connect(self, timeout=10.0):
        logger.info("Connecting to mumble server %s:%s as %s", self.server, self.port, self.user)
        # Direct TCP check before pymumble
        try:
            with socket.create_connection((self.server, self.port), timeout=5) as sock:
                logger.info("TCP connection to %s:%s succeeded", self.server, self.port)
        except Exception as sock_err:
            logger.error("TCP connection to %s:%s failed: %s", self.server, self.port, sock_err)
            raise RuntimeError(f"TCP connection to {self.server}:{self.port} failed: {sock_err}")
        # Now try pymumble
        try:
            self.mumble = Mumble(self.server, self.user, port=self.port, password=self.password)
            self.mumble.set_receive_sound(True)  # Enable audio
            self.mumble.start()
            # is_ready() is a blocking call that waits until connection is complete
            self.mumble.is_ready()
        except Exception as e:
            logger.error("Exception during pymumble connect: %s", e, exc_info=True)
            raise

        # register text callback
        try:
            self.mumble.callbacks.set_callback(PYMUMBLE_CLBK_TEXTMESSAGERECEIVED, self._on_text_received)
        except Exception as e:
            # some pymumble variants expose callbacks differently
            try:
                self.mumble.set_callback(PYMUMBLE_CLBK_TEXTMESSAGERECEIVED, self._on_text_received)
            except Exception:
                logger.warning("Could not register text message callback: %s", e)

        # join channel if requested
        if self.channel:
            try:
                chan = self.mumble.channels.find_by_name(self.channel)
                if chan:
                    chan.move_in()
                    logger.info("Moved to channel %s", self.channel)
                else:
                    logger.warning("Channel '%s' not found; staying in default channel", self.channel)
            except Exception as e:
                logger.warning("Failed to move to channel '%s': %s", self.channel, e)

        # send a greeting
        try:
            self.mumble.users.myself.send_text_message(f"Hello from {self.user}")
        except Exception:
            try:
                self.mumble.send_message(f"Hello from {self.user}")
            except Exception:
                logger.debug("Unable to send initial text message; continuing")

        logger.info("Connected and ready")

    def disconnect(self):
        if self.mumble:
            try:
                self.mumble.stop()
            except Exception:
                pass
            self.mumble = None
            logger.info("Disconnected from mumble server")


def parse_args():
    parser = argparse.ArgumentParser(description="Simple Mumble client for mermer-server")
    parser.add_argument("--server", default=os.environ.get("MERMER_SERVER_IP", "127.0.0.1"), help="Mumble server IP or hostname (env MERMER_SERVER_IP)")
    parser.add_argument("--port", type=int, default=int(os.environ.get("MERMER_SERVER_PORT", DEFAULT_PORT)), help="Mumble port")
    parser.add_argument("--user", default=os.environ.get("MERMER_CLIENT_USER", DEFAULT_USER), help="Username")
    parser.add_argument("--password", default=os.environ.get("MERMER_SERVER_PASSWORD", None), help="Server password (optional)")
    parser.add_argument("--channel", default=os.environ.get("MERMER_CHANNEL", None), help="Channel name to move into after connect (optional)")
    return parser.parse_args()


def main():
    args = parse_args()
    client = MumbleClient(server=args.server, port=args.port, user=args.user, password=args.password, channel=args.channel)
    try:
        client.connect()
        logger.info("Press Ctrl-C to exit")
        # keep running to receive messages
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        logger.info("Interrupted by user, shutting down")
    except Exception as e:
        logger.exception("Unexpected error: %s", e)
    finally:
        client.disconnect()


if __name__ == "__main__":
    main()
