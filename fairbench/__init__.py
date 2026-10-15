"""fairbench - a vendor-neutral AI benchmark for laptop CPU / GPU / NPU.

The suite is built around three rules that make Intel and AMD numbers
comparable:

1. **Identical work.** Every device executes byte-identical ONNX graphs with
   byte-identical inputs. Models and inputs are generated deterministically
   and their SHA-256 hashes are recorded in every result file.
2. **Identical measurement.** Same warm-up policy, same iteration policy,
   same aggregation (median-first), same thread policy defaults.
3. **Full disclosure.** Thread counts, execution providers, precision, driver
   versions, power source and thermal state are written into the result file,
   so an unfair run is visible rather than hidden.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
