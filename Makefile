PYTHON ?= python3
F2PY = $(PYTHON) -m numpy.f2py
SRCS = src/kind_params.f90 src/kerr_metric.f90 src/q_metric.f90 \
       src/spacetime.f90 src/geodesic_eom.f90 src/rk_integrators.f90 \
       src/disk_model.f90 src/raytracer.f90

.PHONY: all core test clean

all: core

core:
	FFLAGS="-O3 -fopenmp" $(F2PY) -c $(SRCS) -m _core --dep openmp \
	    only: metric_cov metric_contra get_horizon get_isco camera_init \
	          trace_ray render_image trace_geodesic flux_profile \
	          init_flux_table get_disk_omega get_redshift \
	          trace_to_plane trace_bundle_to_plane :
	mv _core.*.so python/grayt/

test:
	$(PYTHON) -m pytest tests -q

clean:
	rm -f python/grayt/_core.*.so
	rm -rf src/*.mod
