import unittest

from src.models.btts_market_benchmark import (
    direct_btts_market_probability,
    implied_btts_from_closing_markets,
)


class BttsMarketBenchmarkTests(unittest.TestCase):
    def test_direct_market_probability_removes_two_way_margin(self):
        self.assertAlmostEqual(direct_btts_market_probability(1.90, 1.90), 0.5)

    def test_balanced_high_total_market_implies_more_btts(self):
        low = implied_btts_from_closing_markets(2.8, 3.0, 2.8, 2.8, 1.45)
        high = implied_btts_from_closing_markets(2.8, 3.0, 2.8, 1.45, 2.8)

        self.assertGreater(high, low)
        self.assertGreaterEqual(low, 0)
        self.assertLessEqual(high, 1)


if __name__ == "__main__":
    unittest.main()
