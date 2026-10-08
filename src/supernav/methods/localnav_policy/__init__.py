"""Learned-policy backends for the localnav service.

Import discipline: this package's ``backends`` module (and the debug backend)
must stay torch-free so the service can run plumbing smoke tests on hosts
without a GPU stack. torch is imported only by ``scheduler`` / ``model`` /
``preprocess`` / ``nomad_backend``, which ``build_backend`` loads lazily.
"""
