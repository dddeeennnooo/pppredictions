import unittest

from src.importers.open_btts_data_importer import _normalise_team


class OpenBttsImporterTests(unittest.TestCase):
    def test_normalises_provider_team_aliases(self):
        self.assertEqual(_normalise_team("Manchester United"), "man united")
        self.assertEqual(_normalise_team("Internazionale"), "inter")


if __name__ == "__main__":
    unittest.main()
