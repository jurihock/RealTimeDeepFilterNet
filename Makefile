.PHONY: help build clean run

CONFIG = Release
INPUT  = cpp
OUTPUT = build

help:
	@echo build
	@echo clean
	@echo run

build:
	@cmake -DCMAKE_BUILD_TYPE=$(CONFIG) -S $(INPUT) -B $(OUTPUT)
	@cmake --build $(OUTPUT)

clean:
	@rm -rf $(OUTPUT)

run:
	build/debug_cpp
