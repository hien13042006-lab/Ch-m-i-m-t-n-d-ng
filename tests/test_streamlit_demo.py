"""Integration checks using the repository's actual result files."""
from pathlib import Path
import unittest

import pandas as pd
from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[1]


class DemoTests(unittest.TestCase):
    def app(self, page):
        app = AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=45).run()
        app.sidebar.radio[0].set_value(page).run()
        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        return app

    def test_overview_matches_saved_metrics(self):
        app = self.app("Tổng quan")
        metrics = pd.read_csv(ROOT / "eval_results/metrics_auc_bootstrap.csv")
        row = metrics.loc[metrics.group == "full_population"].iloc[0]
        self.assertEqual(app.metric[2].value, f"{row.auc_b:.4f}")
        self.assertEqual(app.metric[3].value, f"{row.pr_auc_b:.4f}")

    def test_customer_filters_use_current_scores(self):
        app = self.app("Hồ sơ khách hàng")
        for case in app.selectbox[0].options:
            app.selectbox[0].set_value(case).run()
            self.assertFalse(app.exception)
            self.assertFalse(app.error)
            if len(app.selectbox) > 1:
                customer_id = app.selectbox[1].value
                scores = pd.read_csv(ROOT / "scored_customers.csv")
                row = scores.loc[scores.SK_ID_CURR == customer_id].iloc[0]
                self.assertEqual(app.metric[1].value, f"{row.score_b:.2f}")
                self.assertEqual(app.metric[2].value, f"{row.score_diff:+.2f}")

    def test_grade_matrices_match_summary_counts(self):
        app = self.app("Điểm và hạng")
        for t in [2, 3, 4]:
            app.selectbox[0].set_value(t).run()
            self.assertFalse(app.exception)
            self.assertFalse(app.error)
            summary = app.dataframe[1].value
            matrix = pd.read_csv(ROOT / f"eval_results/grade_transition_matrix_t{t}.csv", index_col=0)
            self.assertEqual(int(summary.n_customers.sum()), int(matrix.to_numpy().sum()))



if __name__ == "__main__":
    unittest.main()
