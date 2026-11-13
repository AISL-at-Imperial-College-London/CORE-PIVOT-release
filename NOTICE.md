# Attribution and release status

The accompanying manuscript is in preparation, with preprint release planned
soon. Its link and citation metadata will be added when available. This repository
is a public reproduction package; no manuscript publication or acceptance is claimed.

Original code and accompanying repository materials are licensed under the
[MIT License](LICENSE), copyright (c) 2026 Autonomous Industrial Systems Laboratory,
unless identified otherwise. Third-party components retain their own licenses.
The repository license does not license the accompanying manuscript or override
separate terms attached to externally distributed artifacts.

Software author and maintainer: Christos Evripidou. This attribution does not
replace the authorship or licences of third-party implementations, or the
authorship of the accompanying research paper.

The CORL-derived Python files in `algorithms/corl_vendor/` and the adapted loss in
`ablations/cql_backup.py` are derived from CORL, revision
`6afec90484bbf47dee05fdf525e26a3ebe028e9b`, and remain under Apache-2.0, not the
root MIT license. The [Apache-2.0 license](algorithms/corl_vendor/LICENSE),
upstream file hashes and extraction/adaptation notes are retained alongside
the vendored code. The surrounding wrappers are research-specific adaptations,
not claims of unmodified upstream benchmark results.

Upstream source attribution is also retained here: CORL's pinned IQL
implementation credits [IQL-PyTorch](https://github.com/gwthomas/IQL-PyTorch),
whose [MIT notice](algorithms/corl_vendor/third_party/IQL_PyTorch_LICENSE.txt)
credits Garrett Thomas (2021). CORL's CQL implementation credits
[Xinyang Geng's CQL implementation](https://github.com/young-geng/CQL/tree/934b0e8354ca431d6c083c4e3a29df88d4b0a24d);
its [MIT notice](algorithms/corl_vendor/third_party/CQL_LICENSE.txt) is retained too.
The TD3+BC implementation credit includes
[Scott Fujimoto's TD3_BC](https://github.com/sfujim/TD3_BC); its
[MIT copyright and permission notice](algorithms/corl_vendor/third_party/TD3_BC_LICENSE.txt)
is distributed with this repository. These upstream notices are additional to,
and are not replaced by, the CORL or repository-wide notices.

Credit the original algorithm papers as well as the implementation library:
TD3 and TD3+BC (Fujimoto and colleagues), IQL (Kostrikov and colleagues), CQL
(Kumar and colleagues), one-step offline RL/EasyBCQ (Brandfonbrener and
colleagues), AWR (Peng and colleagues), the policy-extraction investigation
(Park and colleagues), and HOFLON (Durkin and colleagues). HOFLON-EasyBCQ is
the radius-constrained hybrid studied here, not a reproduction of the original
HOFLON autoencoder-regularised controller.

Implementation sources: [CORL](https://github.com/corl-team/CORL) and
[TD3+BC](https://github.com/sfujim/TD3_BC). Implementation versions and verification
records are listed in `provenance/`; data/figure manifests identify the archived
results separately from newly generated runs.
