# Reproducible Dieharder environment.
#
# Windows has no native dieharder build, so this is the path of least
# resistance for validating from a Windows host: Docker Desktop, or any Linux
# box with a container runtime.
#
#   docker build -t radiarandom-dieharder validation/
#
# Test a file you already generated:
#   docker run --rm -v "$PWD:/work" -w /work radiarandom-dieharder \
#       dieharder -a -g 201 -f data/output.bin
#
# Or stream straight from the generator on the host (the container never needs
# USB access this way, which keeps it simple):
#   radiarandom gen --stream -f bin -o - --quiet \
#     | docker run --rm -i radiarandom-dieharder dieharder -a -g 200
#
# To run the generator *inside* the container instead, pass the device through
# with --device=/dev/bus/usb and install the udev rule on the host first.

FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        dieharder \
        python3 \
        python3-pip \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# The SP 800-90B entropy estimators are pure Python and have no dependencies
# beyond the standard library, so the image stays small.
WORKDIR /work
COPY sp800_90b.py localtests.py report.py /opt/radiarandom-validation/
ENV PATH="/opt/radiarandom-validation:${PATH}"

RUN dieharder -l | head -40

CMD ["dieharder", "-h"]
