import unittest

import numpy as np

from src.models.market_residual import (
    logit_market_blend,
    opening_market_btts_proxy,
)


class MarketResidualTests(unittest.TestCase):
    def test_opening_proxy_responds_to_total_goals_market(self):
        low = opening_market_btts_proxy(0.4, 0.25, 0.35, 0.35)
        high = opening_market_btts_proxy(0.4, 0.25, 0.35, 0.70)

        self.assertGreater(high, low)

    def test_logit_blend_endpoints_equal_inputs(self):
        model = np.array([0.2, 0.7])
        market = np.array([0.4, 0.6])

        self.assertTrue(np.allclose(logit_market_blend(model, market, 1), model))
        self.assertTrue(np.allclose(logit_market_blend(model, market, 0), market))

    def test_missing_market_probability_falls_back_to_model(self):
        blended = logit_market_blend([0.3], [np.nan], 0.5)

        self.assertAlmostEqual(float(blended[0]), 0.3)


if __name__ == "__main__":
    unittest.main()
