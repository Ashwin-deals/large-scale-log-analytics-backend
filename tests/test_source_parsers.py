import gzip
import importlib.util
import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

from multi_source import labels as label_rules
from multi_source import upload as multi_source_upload
from parser import cloudtrail_parser
from parser.apache_parser import ApacheParser, window_block_id
from parser.base_parser import COMMON_LOG_SCHEMA
from parser.cloudtrail_parser import CloudTrailParser, identity_of, iter_records
from parser.django_parser import DjangoParser
from parser.openstack_parser import OpenStackParser
from parser.templates import normalize_template

# Real lines, copied verbatim from data/raw/.
OPENSTACK_INSTANCE = ("nova-compute.log.1.2017-05-17_12:02:35 2017-05-16 15:15:54.960 2931 INFO nova.compute.manager "
                      "[req-7a738b84-d574-43c6-a6c4-68c164365101 e887c6de57b5411cb33a5943be2d3c1a 54fadb412c4e40cdbaed9335e4c35a9e - - -] "
                      "[instance: 0f079bdd-4117-4f6a-8b49-f3fb720b483c] Took 0.54 seconds to deallocate network for instance.")
OPENSTACK_REQUEST = ("nova-api.log.1.2017-05-16_13:53:08 2017-05-16 00:00:00.008 25746 INFO nova.osapi_compute.wsgi.server "
                     "[req-38101a0b-2096-447d-96ea-a692162415ae 113d3a99c3da401fbd62cc2caa5b96d2 54fadb412c4e40cdbaed9335e4c35a9e - - -] "
                     "10.11.10.1 \"GET /v2/54fadb412c4e40cdbaed9335e4c35a9e/servers/detail HTTP/1.1\" status: 200 len: 1893 time: 0.2477829")
OPENSTACK_NO_CONTEXT_INSTANCE = ("nova-compute.log.1.2017-05-16_13:55:31 2017-05-16 00:00:10.302 2931 INFO nova.virt.libvirt.driver [-] "
                                 "[instance: b9000564-fe1a-409b-b8cc-1e88b294cd1d] Instance spawned successfully.")
OPENSTACK_NO_BLOCK = ("nova-api.log.1.2017-05-16_13:53:08 2017-05-16 00:00:16.806 25783 INFO nova.metadata.wsgi.server [-] "
                      "10.11.21.122,10.11.10.1 \"GET /openstack/2013-10-17 HTTP/1.1\" status: 200 len: 157 time: 0.0008290")

APACHE_NOTICE = "[Thu Jun 09 06:07:04 2005] [notice] LDAP: Built with OpenLDAP LDAP SDK"
APACHE_CLIENT = "[Fri Jun 10 03:37:39 2005] [error] [client 194.116.250.2] File does not exist: /var/www/html/sumthin"
APACHE_JK = "[Sun Dec 04 04:47:44 2005] [notice] jk2_init() Found child 30770 in scoreboard slot 9"
APACHE_CONTINUATION = "script not found or unable to stat"

CLOUDTRAIL_RECORD = {
    "eventTime": "2019-08-21T11:31:47Z", "eventSource": "ec2.amazonaws.com", "eventName": "RunInstances",
    "sourceIPAddress": "5.205.62.253", "errorCode": "Client.RequestLimitExceeded", "eventID": "3038ebd2-c98a-4c65-9b6e-e22506292313",
    "userIdentity": {"type": "IAMUser", "principalId": "AIDA9BO36HFBHKGJAO9C1", "arn": "arn:aws:iam::811596193553:user/backup",
                     "accountId": "811596193553", "accessKeyId": "ASIARF55FBMFZBXLKDFW", "userName": "backup"},
    "awsRegion": "us-west-1",
}

DJANGO_LINES = [
    '[2026-03-02 00:00:37,903] INFO django.request user=260 ip=10.239.185.48 "POST /api/cart HTTP/1.1" 200 438 41ms',
    '[2026-03-02 00:18:58,818] ERROR django.request user=817 ip=10.178.111.22 "GET /api/products/3885 HTTP/1.1" 500 89 1284ms Internal Server Error: /api/products/3885',
    '[2026-03-02 01:02:03,004] WARNING app.performance user=88 ip=10.1.9.4 "GET /api/reports/12 HTTP/1.1" 200 20431 5230ms Slow request',
]


class TemplateTest(unittest.TestCase):
    def test_variable_parts_become_placeholders(self):
        self.assertEqual(normalize_template("File does not exist: /var/www/html/sumthin"), "File does not exist: <path>")
        self.assertEqual(normalize_template("jk2_init() Found child 30770 in scoreboard slot 9"),
                         "jk2_init() Found child <n> in scoreboard slot <n>")
        self.assertEqual(normalize_template("Took 0.54 seconds to deallocate network for instance."),
                         "Took <n>.<n> seconds to deallocate network for instance.")
        self.assertIn("<uuid>", normalize_template("instance 0f079bdd-4117-4f6a-8b49-f3fb720b483c gone"))
        self.assertIn("<ip>", normalize_template("connect from 10.11.10.1:8774"))

    def test_lines_differing_only_in_variables_share_a_template(self):
        a = normalize_template("File does not exist: /var/www/html/a/b.php")
        b = normalize_template("File does not exist: /srv/x")
        self.assertEqual(a, b)

    def test_empty_message(self):
        self.assertEqual(normalize_template(""), "<empty>")


class OpenStackParserTest(unittest.TestCase):
    def setUp(self):
        self.parser = OpenStackParser()

    def test_instance_line_uses_the_instance_uuid(self):
        row = self.parser.parse_line(OPENSTACK_INSTANCE)
        self.assertEqual(set(row), set(COMMON_LOG_SCHEMA))
        self.assertEqual(row["block_id"], "0f079bdd-4117-4f6a-8b49-f3fb720b483c")
        self.assertEqual((row["date"], row["time"], row["pid"]), ("2017-05-16", "15:15:54", 2931))  # ms truncated
        self.assertEqual((row["log_level"], row["component"]), ("INFO", "nova.compute.manager"))
        self.assertEqual(row["event_type"], "Took <n>.<n> seconds to deallocate network for instance.")

    def test_line_without_an_instance_has_no_block_by_default(self):
        row = self.parser.parse_line(OPENSTACK_REQUEST)
        self.assertIsNone(row["block_id"])  # block = instance session only
        self.assertEqual((row["source_ip"], row["block_size"]), ("10.11.10.1", 1893))

    def test_request_id_fallback_is_opt_in(self):
        row = OpenStackParser(fallback_to_request_id=True).parse_line(OPENSTACK_REQUEST)
        self.assertEqual(row["block_id"], "req-38101a0b-2096-447d-96ea-a692162415ae")

    def test_instance_wins_even_without_a_request_context(self):
        self.assertEqual(self.parser.parse_line(OPENSTACK_NO_CONTEXT_INSTANCE)["block_id"], "b9000564-fe1a-409b-b8cc-1e88b294cd1d")

    def test_no_instance_and_no_request_means_no_block(self):
        row = OpenStackParser(fallback_to_request_id=True).parse_line(OPENSTACK_NO_BLOCK)
        self.assertIsNone(row["block_id"])
        self.assertEqual(row["date"], "2017-05-16")

    def test_unmatched_line(self):
        row = self.parser.parse_line("garbage")
        self.assertIsNone(row["date"])
        self.assertEqual(row["event_type"], "OTHER")

    def test_parse_file(self):
        with tempfile.NamedTemporaryFile("w", suffix=".log", delete=False) as handle:
            handle.write("\n".join([OPENSTACK_INSTANCE, OPENSTACK_REQUEST, "garbage"]) + "\n")
        frame = self.parser.parse(handle.name)
        os.unlink(handle.name)
        self.assertEqual(list(frame.columns), COMMON_LOG_SCHEMA)
        self.assertEqual(len(frame), 3)


class ApacheParserTest(unittest.TestCase):
    def test_client_error_line(self):
        row = ApacheParser().parse_line(APACHE_CLIENT)
        self.assertEqual((row["date"], row["time"], row["log_level"]), ("2005-06-10", "03:37:39", "ERROR"))
        self.assertEqual((row["component"], row["source_ip"]), ("client", "194.116.250.2"))
        self.assertEqual(row["event_type"], "File does not exist: <path>")
        self.assertEqual(row["block_id"], "win-20050610T033700")
        self.assertEqual(row["pid"], 0)

    def test_module_components(self):
        self.assertEqual(ApacheParser().parse_line(APACHE_NOTICE)["component"], "LDAP")
        jk = ApacheParser().parse_line(APACHE_JK)
        self.assertEqual((jk["component"], jk["log_level"]), ("jk2_init", "NOTICE"))
        self.assertEqual(jk["event_type"], "jk2_init() Found child <n> in scoreboard slot <n>")

    def test_window_size_is_configurable(self):
        self.assertEqual(ApacheParser(300).parse_line(APACHE_CLIENT)["block_id"], "win-20050610T033500")
        self.assertEqual(ApacheParser(3600).parse_line(APACHE_CLIENT)["block_id"], "win-20050610T030000")
        with self.assertRaises(ValueError):
            ApacheParser(0)

    def test_lines_in_one_window_share_a_block(self):
        a = ApacheParser().parse_line("[Fri Jun 10 03:37:01 2005] [error] [client 1.2.3.4] x")
        b = ApacheParser().parse_line("[Fri Jun 10 03:37:59 2005] [error] [client 1.2.3.4] y")
        c = ApacheParser().parse_line("[Fri Jun 10 03:38:00 2005] [error] [client 1.2.3.4] z")
        self.assertEqual(a["block_id"], b["block_id"])
        self.assertNotEqual(b["block_id"], c["block_id"])

    def test_continuation_line_is_unparsed(self):
        self.assertIsNone(ApacheParser().parse_line(APACHE_CONTINUATION)["date"])

    def test_single_digit_day_with_double_space(self):
        row = ApacheParser().parse_line("[Sun Dec  4 04:47:44 2005] [notice] jk2_init() Found child 1 in scoreboard slot 2")
        self.assertEqual(row["date"], "2005-12-04")


class CloudTrailParserTest(unittest.TestCase):
    def test_record_mapping(self):
        row = CloudTrailParser().record_to_row(CLOUDTRAIL_RECORD)
        self.assertEqual(set(row), set(COMMON_LOG_SCHEMA))
        self.assertEqual((row["component"], row["event_type"], row["source_ip"]), ("ec2.amazonaws.com", "RunInstances", "5.205.62.253"))
        self.assertEqual(row["log_level"], "ERROR")
        self.assertEqual(row["block_id"], "arn:aws:iam::811596193553:user/backup|20190821T11")
        self.assertIn("errorCode=Client.RequestLimitExceeded", row["raw_message"])

    def test_no_error_code_is_info(self):
        record = {k: v for k, v in CLOUDTRAIL_RECORD.items() if k != "errorCode"}
        self.assertEqual(CloudTrailParser().record_to_row(record)["log_level"], "INFO")

    def test_identity_fallbacks(self):
        self.assertEqual(identity_of({"userIdentity": {"type": "AWSAccount", "accountId": "1"}, "sourceIPAddress": "9.9.9.9"}),
                         "anonymous@9.9.9.9")
        self.assertEqual(identity_of({"userIdentity": {"type": "IAMUser", "principalId": "AIDA1"}}), "AIDA1")
        self.assertEqual(identity_of({"userIdentity": {"type": "AWSService"}}), "AWSService:unknown")

    def test_streaming_reader_handles_records_split_across_reads(self):
        records = [dict(CLOUDTRAIL_RECORD, eventID=f"id-{i}", requestParameters={"note": "has ] and , and { inside"}) for i in range(50)]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "t.json.gz"
            with gzip.open(path, "wt", encoding="utf-8") as handle:
                json.dump({"Records": records}, handle)
            with mock.patch.object(cloudtrail_parser, "READ_CHUNK", 64):  # force many partial reads
                streamed = list(iter_records(path))
            self.assertEqual([r["eventID"] for r in streamed], [f"id-{i}" for i in range(50)])
            plain = Path(tmp) / "t.json"
            plain.write_text(json.dumps({"Records": records[:3]}))
            self.assertEqual(len(list(iter_records(plain))), 3)  # not gzipped works too
            empty = Path(tmp) / "e.json"
            empty.write_text('{"Records": []}')
            self.assertEqual(list(iter_records(empty)), [])

    def test_real_file_first_record(self):
        path = Path("data/raw/cloudtrail/flaws_cloudtrail19.json.gz")
        if not path.exists():
            self.skipTest("CloudTrail data not present")
        first = next(iter(CloudTrailParser().iter_rows(path)))
        self.assertTrue(first["block_id"].endswith("|20200921T22"))


class DjangoParserTest(unittest.TestCase):
    def test_request_error_and_slow_lines(self):
        parser = DjangoParser()
        ok, error, slow = (parser.parse_line(line) for line in DJANGO_LINES)
        self.assertEqual((ok["block_id"], ok["event_type"], ok["block_size"], ok["log_level"]), ("user-260", "POST /api/cart 200", 438, "INFO"))
        self.assertEqual((error["event_type"], error["log_level"], error["component"]), ("GET /api/products/<n> 500", "ERROR", "django.request"))
        self.assertEqual((slow["event_type"], slow["component"]), ("GET /api/reports/<n> 200 SLOW", "app.performance"))
        self.assertEqual((ok["date"], ok["time"], ok["source_ip"]), ("2026-03-02", "00:00:37", "10.239.185.48"))

    def test_unmatched_line(self):
        self.assertIsNone(DjangoParser().parse_line("not django")["date"])


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, Path("scripts") / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class GeneratorTest(unittest.TestCase):
    def test_django_generator_is_deterministic_and_labels_every_user(self):
        generator = load_script("generate_django_logs")
        lines_a, labels_a = generator.generate(users=50, days=1, fault_rate=0.1, seed=3)
        lines_b, labels_b = generator.generate(users=50, days=1, fault_rate=0.1, seed=3)
        self.assertEqual((lines_a, labels_a), (lines_b, labels_b))
        self.assertNotEqual(lines_a, generator.generate(users=50, days=1, fault_rate=0.1, seed=4)[0])
        self.assertEqual(len(labels_a), 50)
        self.assertEqual(sum(1 for _, label, _ in labels_a if label == "Anomaly"), 5)
        parsed = [DjangoParser().parse_line(line) for line in lines_a]
        self.assertTrue(all(row["date"] for row in parsed))  # the parser reads everything the generator writes


class LabelRulesTest(unittest.TestCase):
    def test_instances_in_a_file(self):
        with tempfile.NamedTemporaryFile("w", suffix=".log", delete=False) as handle:
            handle.write(OPENSTACK_INSTANCE + "\n" + OPENSTACK_REQUEST + "\n" + OPENSTACK_NO_CONTEXT_INSTANCE + "\n")
        self.assertEqual(label_rules.instances_in(handle.name),
                         {"0f079bdd-4117-4f6a-8b49-f3fb720b483c", "b9000564-fe1a-409b-b8cc-1e88b294cd1d"})
        os.unlink(handle.name)

    def test_openstack_label_file_with_header_and_crlf(self):
        with tempfile.NamedTemporaryFile("wb", delete=False) as handle:
            handle.write(b"The following VM instances have injected anomalies.\r\n\r\n544fd51c-4edc-4780-baae-ba1d80a0acfc\r\n"
                         b"ae651dff-c7ad-43d6-ac96-bbcd820ccca8\r\n")
        self.assertEqual(label_rules.openstack_anomalous_instances(handle.name),
                         {"544fd51c-4edc-4780-baae-ba1d80a0acfc", "ae651dff-c7ad-43d6-ac96-bbcd820ccca8"})
        os.unlink(handle.name)

    def cleaned(self, rows):
        return pd.DataFrame(rows, columns=["datetime", "block_id", "component", "event_type", "source_ip", "raw_message"])

    def test_weak_rule_r1_auth_error_burst(self):
        labeler = label_rules.CloudTrailWeakLabeler()
        burst = [("2019-08-21 11:00:%02d" % i, "u|20190821T11", "ec2.amazonaws.com", "DescribeInstances", "1.1.1.1",
                  "eventID=x errorCode=Client.UnauthorizedOperation") for i in range(label_rules.AUTH_ERROR_BURST)]
        few = [("2019-08-21 11:00:00", "v|20190821T11", "ec2.amazonaws.com", "DescribeInstances", "1.1.1.1",
                "eventID=y errorCode=AccessDenied")] * (label_rules.AUTH_ERROR_BURST - 1)
        labeler.update(self.cleaned(burst + few))
        table = labeler.labels().set_index("block_id")
        self.assertEqual(table.loc["u|20190821T11", "label"], "Anomaly")
        self.assertTrue(table.loc["u|20190821T11", "rule_r1_auth_error_burst"])
        self.assertEqual(table.loc["v|20190821T11", "label"], "Normal")

    def test_weak_rule_r2_only_for_a_never_seen_ip_in_its_first_hour(self):
        labeler = label_rules.CloudTrailWeakLabeler()
        labeler.update(self.cleaned([
            ("2019-08-21 10:00:00", "a|20190821T10", "s3.amazonaws.com", "ListBuckets", "2.2.2.2", "errorCode=-"),
            ("2019-08-21 11:00:00", "a|20190821T11", "iam.amazonaws.com", "AttachUserPolicy", "2.2.2.2", "errorCode=-"),  # IP seen at 10h
            ("2019-08-21 11:05:00", "b|20190821T11", "iam.amazonaws.com", "CreateAccessKey", "3.3.3.3", "errorCode=-"),  # new IP
            ("2019-08-21 11:06:00", "c|20190821T11", "iam.amazonaws.com", "ListUsers", "4.4.4.4", "errorCode=-"),  # new IP, not sensitive
        ]))
        table = labeler.labels().set_index("block_id")["label"]
        self.assertEqual(table.to_dict(), {"a|20190821T10": "Normal", "a|20190821T11": "Normal",
                                           "b|20190821T11": "Anomaly", "c|20190821T11": "Normal"})

    def test_first_seen_persists_across_chunks(self):
        labeler = label_rules.CloudTrailWeakLabeler()
        labeler.update(self.cleaned([("2019-08-21 10:00:00", "a|20190821T10", "s3.amazonaws.com", "ListBuckets", "5.5.5.5", "errorCode=-")]))
        labeler.update(self.cleaned([("2019-08-21 12:00:00", "a|20190821T12", "s3.amazonaws.com", "PutBucketPolicy", "5.5.5.5", "errorCode=-")]))
        self.assertEqual(labeler.labels().set_index("block_id").loc["a|20190821T12", "label"], "Normal")

    def test_sensitive_actions(self):
        self.assertTrue(label_rules.is_sensitive("iam.amazonaws.com", "CreateUser"))
        self.assertFalse(label_rules.is_sensitive("iam.amazonaws.com", "ListUsers"))
        self.assertTrue(label_rules.is_sensitive("s3.amazonaws.com", "PutBucketPolicy"))
        self.assertFalse(label_rules.is_sensitive("ec2.amazonaws.com", "CreateVpc"))


class UploadDispatchTest(unittest.TestCase):
    def setUp(self):
        import jwt
        from flask import Flask

        import sources_api

        self.api = sources_api
        self.inserted = []

        class FakeUploads:
            def insert_one(inner, doc):
                self.inserted.append(doc)
                return mock.Mock(inserted_id="fake-id")

        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        for patch in (mock.patch.object(sources_api, "uploads", FakeUploads()),
                      mock.patch.object(sources_api, "UPLOAD_DIR", Path(self.tmp.name)),
                      mock.patch.object(sources_api.threading, "Thread"),
                      mock.patch.dict(os.environ, {"JWT_SECRET": "test-secret"})):
            patch.start()
            self.addCleanup(patch.stop)
        app = Flask(__name__)
        app.register_blueprint(sources_api.sources_bp)
        self.client = app.test_client()
        token = jwt.encode({"sub": "t", "email": "t@example.com", "exp": int(time.time()) + 600}, "test-secret", algorithm="HS256")
        self.headers = {"Authorization": f"Bearer {token}"}

    def post(self, source=None, name="a.log"):
        data = {"file": (__import__("io").BytesIO(APACHE_CLIENT.encode()), name)}
        if source is not None:
            data["source"] = source
        return self.client.post("/api/sources/upload", data=data, headers=self.headers, content_type="multipart/form-data")

    def test_unknown_source_is_rejected(self):
        response = self.post("syslog")
        self.assertEqual(response.status_code, 400)
        self.assertIn("Unknown source", response.get_json()["error"])
        self.assertEqual(self.inserted, [])

    def test_source_without_a_trained_model_is_rejected(self):
        with mock.patch.object(multi_source_upload, "deployed_model", return_value=None):
            response = self.post("apache")
        self.assertEqual(response.status_code, 400)
        self.assertIn("run_source_pipeline.py --source apache", response.get_json()["error"])

    def test_source_is_recorded_and_defaults_to_hdfs(self):
        self.assertEqual(self.post().status_code, 201)
        with mock.patch.object(multi_source_upload, "deployed_model", return_value={"version": 2, "model_path": "m.pkl"}):
            self.assertEqual(self.post("Apache").status_code, 201)
        self.assertEqual([doc["log_source"] for doc in self.inserted], ["hdfs", "apache"])

    def test_gzipped_cloudtrail_upload_is_accepted(self):
        with mock.patch.object(multi_source_upload, "deployed_model", return_value={"version": 1, "model_path": "m.pkl"}):
            response = self.post("cloudtrail", name="trail.json.gz")
        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.get_json()["log_source"], "cloudtrail")

    def test_recognized_mask_needs_a_date_and_a_block(self):
        frame = pd.DataFrame({"date": ["2005-06-10", None, "2005-06-10"], "block_id": ["win-1", "win-1", None]})
        self.assertEqual(multi_source_upload.recognized_mask(frame).tolist(), [True, False, False])


if __name__ == "__main__":
    unittest.main()
