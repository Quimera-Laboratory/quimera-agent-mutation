CARGO_TARGET_DIR ?= /tmp/quimera-target
TMPDIR ?= /tmp
RUSTFLAGS ?= -C target-cpu=native -C target-feature=+avx2,+fma

export CARGO_TARGET_DIR
export TMPDIR
export RUSTFLAGS

.PHONY: build test release check clean

build:
	cargo build --release

test:
	cargo test --release -- --nocapture

check:
	cargo check --all-targets

clean:
	cargo clean
