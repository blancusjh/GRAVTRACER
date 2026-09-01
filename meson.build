project(
  'gravtracer',
  'c', 'fortran',
  version: '0.2.0',
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
inc_np = include_directories(incdir_numpy, incdir_f2py)

fortran_sources = files(
  'src/kind_params.f90',
  'src/kerr_metric.f90',
  'src/q_metric.f90',
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
    'trace_to_plane', 'trace_bundle_to_plane', ':',
  ],
)

py.extension_module(
  '_core',
  fortran_sources + [core_module, incdir_f2py / 'fortranobject.c'],
  include_directories: inc_np,
  dependencies: [py_dep, omp],
  install: true,
  subdir: 'grayt',
)

py.install_sources(
  [
    'python/grayt/__init__.py',
    'python/grayt/__main__.py',
    'python/grayt/api.py',
    'python/grayt/cli.py',
    'python/grayt/geometry.py',
    'python/grayt/instruments.py',
    'python/grayt/matter.py',
    'python/grayt/plotting.py',
    'python/grayt/results.py',
    'python/grayt/spacetime.py',
    'python/grayt/system.py',
  ],
  subdir: 'grayt',
)
