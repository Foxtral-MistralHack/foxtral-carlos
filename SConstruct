#!/usr/bin/env python

import os
import sys

env = SConscript("godot-cpp/SConstruct")

env.Append(CPPPATH=["src/"])

# -- voxtral.cpp (GGUF-based speech-to-text, uses ggml) -----------------------

voxtral_cpp_root = "voxtral.cpp"
voxtral_cpp_build = os.path.join(voxtral_cpp_root, "build")

env.Append(CPPPATH=[
    os.path.join(voxtral_cpp_root, "include"),
    os.path.join(voxtral_cpp_root, "ggml", "include"),
])

env.Append(LIBPATH=[
    voxtral_cpp_build,
    os.path.join(voxtral_cpp_build, "ggml", "src"),
    os.path.join(voxtral_cpp_build, "ggml", "src", "ggml-metal"),
    os.path.join(voxtral_cpp_build, "ggml", "src", "ggml-blas"),
])

env.Append(LIBS=[
    "voxtral_lib",
    "ggml",
    "ggml-base",
    "ggml-cpu",
    "ggml-metal",
    "ggml-blas",
])

# Set RPATH so the extension finds voxtral.cpp dylibs at runtime
if env["platform"] == "macos":
    env.Append(LINKFLAGS=[
        "-Wl,-rpath,@loader_path",
        "-Wl,-rpath,@loader_path/../../../voxtral.cpp/build",
        "-Wl,-rpath,@loader_path/../../../voxtral.cpp/build/ggml/src",
        "-Wl,-rpath,@loader_path/../../../voxtral.cpp/build/ggml/src/ggml-metal",
        "-Wl,-rpath,@loader_path/../../../voxtral.cpp/build/ggml/src/ggml-blas",
    ])

# -- llama.cpp (text generation) -----------------------------------------------

llama_root = "llama.cpp"
llama_build = os.path.join(llama_root, "build")

env.Append(CPPPATH=[
    os.path.join(llama_root, "include"),
    os.path.join(llama_root, "common"),
])

env.Append(LIBPATH=[
    os.path.join(llama_build, "src"),
    os.path.join(llama_build, "common"),
])

# Only link libllama and common — ggml symbols come from voxtral.cpp's ggml
env.Append(LIBS=[
    "llama",
    "common",
])

# -- macOS frameworks ----------------------------------------------------------

if env["platform"] == "macos":
    env.Append(LINKFLAGS=[
        "-framework", "Metal",
        "-framework", "MetalPerformanceShaders",
        "-framework", "MetalPerformanceShadersGraph",
        "-framework", "Foundation",
        "-framework", "Accelerate",
        "-framework", "AudioToolbox",
        "-framework", "CoreFoundation",
    ])

# -- GDExtension sources (C++) ------------------------------------------------

sources = Glob("src/*.cpp")

# -- Build the shared library -------------------------------------------------

libname = "foxtral"
suffix = env["suffix"]
prefix = env.subst("$SHLIBPREFIX")
shsuffix = env.subst("$SHLIBSUFFIX")
lib_filename = "{}{}{}{}".format(prefix, libname, suffix, shsuffix)

library = env.SharedLibrary(
    target="bin/{}/{}".format(env["platform"], lib_filename),
    source=sources,
)

Default(library)
