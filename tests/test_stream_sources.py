"""Per-source streaming plumbing (streaming/sources.py) and the processor's
handling of sources whose blocks must not be cut by an idle timeout."""

import json
import unittest

from streaming.config import TOPIC_LOGS_RAW
from streaming.sources import IDLE_TIMEOUTS, STREAM_SOURCES, line_parser, message_key, topic_for
from tests.test_source_parsers import APACHE_CLIENT, CLOUDTRAIL_RECORD, DJANGO_LINES, OPENSTACK_INSTANCE, OPENSTACK_NO_BLOCK


class TopicTest(unittest.TestCase):
    def test_hdfs_keeps_its_topic_and_others_get_their_own(self):
        self.assertEqual(topic_for("hdfs"), TOPIC_LOGS_RAW)
        topics = {topic_for(source) for source in STREAM_SOURCES}
        self.assertEqual(len(topics), len(STREAM_SOURCES))
        self.assertEqual(topic_for("django"), f"{TOPIC_LOGS_RAW}.django")

    def test_unknown_source_is_rejected(self):
        with self.assertRaises(ValueError):
            topic_for("syslog-ng")

    def test_every_source_has_an_idle_timeout_entry(self):
        self.assertEqual(set(IDLE_TIMEOUTS), set(STREAM_SOURCES))
        self.assertIsNone(IDLE_TIMEOUTS["django"])  # a user spans the whole period


class MessageKeyTest(unittest.TestCase):
    """The producer keys each message by the block the batch parser assigns,
    so a block's events land on one partition in order."""

    def test_keys_match_the_batch_parsers_block_ids(self):
        cases = {
            "openstack": OPENSTACK_INSTANCE,
            "apache": APACHE_CLIENT,
            "django": DJANGO_LINES[0],
            "cloudtrail": json.dumps(CLOUDTRAIL_RECORD, sort_keys=True),
        }
        for source, message in cases.items():
            parser = line_parser(source)
            expected = parser.parse_line(message)["block_id"]
            self.assertTrue(expected, source)
            self.assertEqual(message_key(parser, message), str(expected).encode(), source)

    def test_a_line_without_a_block_is_sent_unkeyed(self):
        self.assertIsNone(message_key(line_parser("openstack"), OPENSTACK_NO_BLOCK))

    def test_cloudtrail_adapter_tolerates_garbage(self):
        parser = line_parser("cloudtrail")
        self.assertEqual(parser.parse_line("not json"), {})
        self.assertIsNone(message_key(parser, "[1, 2]"))


class NoTimeoutTest(unittest.TestCase):
    def test_expire_never_fires_without_a_timeout(self):
        from streaming.processor import StreamProcessor

        class FakeScorer:
            early = {20: object()}

        processor = StreamProcessor(FakeScorer(), ks=[20], idle_timeout_seconds=None,
                                    encoder=_NullEncoder(), source="django", parser=line_parser("django"))
        processor.watermark = 10**9
        processor.blocks["user-1"] = object()
        self.assertEqual(processor.expire(), ([], []))
        self.assertIn("user-1", processor.blocks)


class _NullEncoder:
    unseen = {}

    def event_type_names(self):
        return []



class SyslogTest(unittest.TestCase):
    def test_rfc5424_and_rfc3164_messages_are_extracted(self):
        from streaming.syslog_receiver import extract_message

        line = "081109 203615 148 INFO dfs.DataNode$PacketResponder: PacketResponder 1 for block blk_38865049064139660 terminating"
        rfc5424 = f"<134>1 2026-11-13T10:00:00Z host hdfs 148 - - {line}".encode()
        rfc5424_sd = f'<134>1 2026-11-13T10:00:00Z host hdfs 148 ID47 [exampleSDID@32473 iut="3"] {line}'.encode()
        rfc3164 = f"<134>Nov 13 10:00:00 host hdfs[148]: {line}".encode()
        for datagram in (rfc5424, rfc5424_sd, rfc3164):
            self.assertEqual(extract_message(datagram), line)

    def test_non_syslog_is_rejected(self):
        from streaming.syslog_receiver import extract_message

        self.assertIsNone(extract_message(b"just a line"))
        self.assertIsNone(extract_message(b"<134>1 2026-11-13T10:00:00Z host app - - -"))


if __name__ == "__main__":
    unittest.main()
