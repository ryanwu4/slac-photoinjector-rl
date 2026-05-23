from impact import Impact, tools
from impact.evaluate import default_impact_merit
from distgen import Generator

from h5py import File
import numpy as np
import os


def custom_evaluate_impact_with_distgen(
    settings,
    distgen_input_file=None,
    impact_config=None,
    workdir=None,
    archive_path=None,
    merit_f=None,
    verbose=False,
    reuse_archive=True,
):
    """
    Run Impact-T with a distgen-generated initial distribution, archive
    initial + terminal particles only (no intermediate write_beam markers).

    If ``reuse_archive`` is True and ``archive_path/<fingerprint>.h5`` already
    exists for the configured settings, the cached archive is loaded and its
    merit dict is recomputed -- Impact-T is NOT re-run. This makes large
    sweeps resumable across crashes/kills with no overhead on cache hits.
    The cache key matches the fingerprint scheme used by the original
    evaluator, so existing archives are picked up automatically.
    """

    if isinstance(impact_config, str):
        I = Impact.from_yaml(impact_config)
    else:
        I = Impact(**impact_config)

    if workdir:
        I._workdir = workdir  # TODO: fix in LUME-Base
        I.configure()

    I.verbose = verbose
    G = Generator(distgen_input_file)
    G.verbose = verbose

    if settings:
        for key in settings:
            val = settings[key]
            if key.startswith("distgen:"):
                key = key[len("distgen:"):]
                if verbose:
                    print(f"Setting distgen {key} = {val}")
                G[key] = val
            else:
                if verbose:
                    print(f"Setting impact {key} = {val}")
                I[key] = val

    G.run()
    P = G.particles

    I.initial_particles = P
    I.distgen_input = G.input

    # Trigger the distgen->header bookkeeping (Bcurr, Np, Temission, Tini)
    # without running Impact-T, so the fingerprint matches the post-run scheme
    # that archived files are keyed by.
    I.write_input()

    G_for_fp = Generator(I.distgen_input)
    fingerprint = fingerprint_impact_with_distgen(I, G_for_fp)

    # Cache lookup: if an archive already exists for this fingerprint,
    # reload it and recompute merit instead of running Impact-T again.
    if reuse_archive and archive_path:
        cache_dir = tools.full_path(archive_path)
        cached_file = os.path.join(cache_dir, fingerprint + ".h5")
        if os.path.exists(cached_file):
            try:
                I_cached = Impact()
                with File(cached_file, "r") as h5:
                    I_cached.load_archive(h5["impact"])
                output = merit_f(I_cached) if merit_f else default_impact_merit(I_cached)
                if not output.get("error", False):
                    output["fingerprint"] = fingerprint
                    output["archive"] = cached_file
                    output["cached"] = True
                    if verbose:
                        print(f"[cache-hit] {fingerprint}")
                    return output
                # cached run recorded an error -- fall through and re-run
            except (OSError, KeyError) as e:
                # corrupt or schema-mismatched archive; fall through to re-run
                if verbose:
                    print(f"[cache-miss-corrupt] {fingerprint}: {e}")

    I.run()

    if merit_f:
        output = merit_f(I)
    else:
        output = default_impact_merit(I)

    if "error" in output and output["error"]:
        raise ValueError("run_impact_with_distgen returned error in output")

    output["fingerprint"] = fingerprint
    output["cached"] = False

    if archive_path:
        path = tools.full_path(archive_path)
        assert os.path.exists(path), f"archive path does not exist: {path}"
        archive_file = os.path.join(path, fingerprint + ".h5")
        output["archive"] = archive_file
        archive_impact_with_distgen(
            I, G_for_fp, archive_file=archive_file, settings=settings
        )

    return output


def fingerprint_impact_with_distgen(impact_object, distgen_object):
    f1 = impact_object.fingerprint()
    f2 = distgen_object.fingerprint()
    return tools.fingerprint({"f1": f1, "f2": f2})


def archive_impact_with_distgen(
    impact_object,
    distgen_object,
    archive_file=None,
    impact_group="impact",
    distgen_group="distgen",
    settings=None,
):
    h5 = File(archive_file, "w")
    g = h5.create_group(distgen_group)
    distgen_object.archive(g)
    g = h5.create_group(impact_group)
    impact_object.archive(g)
    if settings:
        sg = h5.create_group("settings")
        for k, v in settings.items():
            sg.attrs[k] = v
    h5.close()
