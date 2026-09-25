project(
  'gravtracer',
  'c', 'fortran',
  version: '0.3.0',
  meson_version: '>=1.1.0',
  default_options: ['buildtype=release', 'warning_level=1'],
)

py = import('python').find_installation(pure: false)
py_dep = py.dependency()
omp = dependency('openmp')

incdir_numpy = run_command(
  py, ['-c', 'import numpy; print(numpy.get_include())'],
  check: true).stdout().strip()
incdir_f2py = run_command(
  py, ['-c', 'import numpy.f2py; print(numpy.f2py.get_include())'],
  check: true).stdout().strip()
# Passed as -I flags rather than include_directories(): with an in-tree .venv
# the numpy headers live under the source dir, which include_directories() rejects.
inc_np_args = ['-I' + incdir_numpy, '-I' + incdir_f2py]

fortran_sources = files(
  'src/kind_params.f90',
  'src/kerr_metric.f90',
  'src/q_metric.f90',
  'src/tabulated_metric.f90',
  'src/spacetime.f90',
  'src/geodesic_eom.f90',
  'src/rk_integrators.f90',
  'src/disk_model.f90',
  'src/raytracer.f90',
)

# f2py wrapper generation (same routine list as the Makefile dev flow)
core_module = custom_target(
  '_coremodule',
  input: fortran_sources,
  output: ['_coremodule.c', '_core-f2pywrappers2.f90'],
  command: [
    py, '-m', 'numpy.f2py', '@INPUT@', '-m', '_core', '--lower',
    '--build-dir', '@OUTDIR@',
    '--f2cmap', meson.current_source_dir() / '.f2py_f2cmap',
    'only:', 'metric_cov', 'metric_contra', 'get_horizon', 'get_isco',
    'camera_init', 'trace_ray', 'render_image', 'trace_geodesic',
    'flux_profile', 'init_flux_table', 'get_disk_omega', 'get_redshift',
    'trace_to_plane', 'trace_bundle_to_plane', 'set_metric_table', 'render_geometry', ':',
  ],
)

py.extension_module(
  '_core',
  fortran_sources + [core_module, incdir_f2py / 'fortranobject.c'],
  c_args: inc_np_args,
  dependencies: [py_dep, omp],
  install: true,
  subdir: 'grayt',
)

py.install_sources(
  [
    'python/grayt/__init__.py',
    'python/grayt/__main__.py',
    'python/grayt/api.py',
    'python/grayt/_runtime.py',
    'python/grayt/metrics.py',
    'python/grayt/emission.py',
    'python/grayt/sky.py',
    'python/grayt/scene.py',
    'python/grayt/animation.py',
    'python/grayt/volume.py',

    'python/grayt/cli.py',
    'python/grayt/geometry.py',
    'python/grayt/instruments.py',
    'python/grayt/matter.py',
    'python/grayt/plotting.py',
    'python/grayt/results.py',
    'python/grayt/spacetime.py',
    'python/grayt/system.py',
    'python/grayt/viewer.py',
    'python/grayt/desktop_scene.py',
  ],
  subdir: 'grayt',
)

py.install_sources(
  [
    'python/grayt/web/metrics.js',
    'python/grayt/web/geodesics.js',
    'python/grayt/web/worker.js',
    'python/grayt/web/observer.js',
    'python/grayt/web/observer.css',
  ],
  subdir: 'grayt/web',
)

py.install_sources(
  ['python/grayt/assets/nasa_starmap_4k.jpg'],
  subdir: 'grayt/assets',
)

py.install_sources(
  [
    'python/grayt/gpu/__init__.py',
    'python/grayt/gpu/engine.py',
    'python/grayt/gpu/kernel.cl',
  ],
  subdir: 'grayt/gpu',
)
