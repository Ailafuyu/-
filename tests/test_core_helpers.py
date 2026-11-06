import io
import unittest

from scores_info import _normalize_score_result, _parse_model_json
from tools_info import concat_files, parse_salary


class ScoringHelperTests(unittest.TestCase):
    def test_parse_json_from_markdown_fence(self):
        result = _parse_model_json(' ```json\n{"ability_match": 30}\n``` ')
        self.assertEqual(result["ability_match"], 30)

    def test_parse_json_embedded_in_prose(self):
        result = _parse_model_json('结果如下： {"ok": true} 请查收')
        self.assertTrue(result["ok"])

    def test_normalize_score_clamps_and_recomputes_total(self):
        result = _normalize_score_result({
            "ability_match": 40,
            "responsibility_match": 20,
            "career_match": 15,
            "background_match": 8,
            "development_value": 9,
            "match_score": 999,
        })
        self.assertEqual(result["ability_match"], 35)
        self.assertEqual(result["match_score"], 87)

    def test_normalize_score_rejects_missing_or_invalid_values(self):
        with self.assertRaises(ValueError):
            _normalize_score_result({"ability_match": "not-a-number"})


class JobDataTests(unittest.TestCase):
    def test_salary_parser(self):
        self.assertEqual(parse_salary("10-20K"), (10000.0, 20000.0))
        low, high = parse_salary("150元/天")
        self.assertAlmostEqual(low, 3262.5)
        self.assertAlmostEqual(high, 3262.5)

    def test_concat_rejects_empty_data(self):
        with self.assertRaises(ValueError):
            concat_files([io.StringIO("[]")], "不开启 HR 活跃度筛选")

    def test_concat_deduplicates_jobs(self):
        payload = (
            '[{"公司名称":"甲","职位名称":"工程师","薪资":"10K"},'
            '{"公司名称":"甲","职位名称":"工程师","薪资":"10K"}]'
        )
        result, summary = concat_files(
            [io.StringIO(payload)], "不开启 HR 活跃度筛选"
        )
        self.assertEqual(len(result), 1)
        self.assertIn("去重后剩余岗位1个", summary)


if __name__ == "__main__":
    unittest.main()
