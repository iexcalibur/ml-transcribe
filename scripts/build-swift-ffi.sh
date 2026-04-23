#!/usr/bin/env bash
# Build the universal Swift FFI xcframework from the sidecar Rust crate.
#
# Output: swift/artifacts/TranscribeFFI.xcframework
#
# Rust targets built:
#   aarch64-apple-darwin    macOS Apple Silicon
#   x86_64-apple-darwin     macOS Intel
#   aarch64-apple-ios       iOS / iPadOS physical devices
#   aarch64-apple-ios-sim   iOS Simulator on Apple Silicon
#   x86_64-apple-ios        iOS Simulator on Intel (target name quirk:
#                           there is no `x86_64-apple-ios-sim` because
#                           iOS devices were never x86_64)
#
# We fuse arch-pair .a files with `lipo` into universal binaries, then
# assemble the xcframework with 3 slices:
#   macos-arm64_x86_64                     (universal)
#   ios-arm64                              (device, single arch)
#   ios-arm64_x86_64-simulator             (universal simulator)
#
# An xcframework slice may contain multiple architectures as long as they
# share the same platform + variant (device vs simulator). Device and
# simulator MUST be separate slices even on the same arch — they have
# different SDKs.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CRATE_DIR="${REPO_ROOT}/src/native-swift"
HEADERS_DIR="${CRATE_DIR}/include"
OUTPUT_DIR="${REPO_ROOT}/swift/artifacts"
XCFRAMEWORK_PATH="${OUTPUT_DIR}/TranscribeFFI.xcframework"
UNIVERSAL_DIR="${CRATE_DIR}/target/universal"
LIB_NAME="libml_transcribe_ffi.a"

TARGETS=(
    "aarch64-apple-darwin"
    "x86_64-apple-darwin"
    "aarch64-apple-ios"
    "aarch64-apple-ios-sim"
    "x86_64-apple-ios"
)

cd "${CRATE_DIR}"
for target in "${TARGETS[@]}"; do
    echo "==> Building staticlib for ${target}"
    cargo build --release --target "${target}"
done

# Resolve per-target .a paths.
macos_arm64="${CRATE_DIR}/target/aarch64-apple-darwin/release/${LIB_NAME}"
macos_x86_64="${CRATE_DIR}/target/x86_64-apple-darwin/release/${LIB_NAME}"
ios_device_arm64="${CRATE_DIR}/target/aarch64-apple-ios/release/${LIB_NAME}"
ios_sim_arm64="${CRATE_DIR}/target/aarch64-apple-ios-sim/release/${LIB_NAME}"
ios_sim_x86_64="${CRATE_DIR}/target/x86_64-apple-ios/release/${LIB_NAME}"

for lib in "${macos_arm64}" "${macos_x86_64}" "${ios_device_arm64}" "${ios_sim_arm64}" "${ios_sim_x86_64}"; do
    if [[ ! -f "${lib}" ]]; then
        echo "error: expected staticlib at ${lib}" >&2
        exit 1
    fi
done

echo "==> lipo-fusing macOS universal slice (arm64 + x86_64)"
mkdir -p "${UNIVERSAL_DIR}/macos"
lipo -create "${macos_arm64}" "${macos_x86_64}" \
    -output "${UNIVERSAL_DIR}/macos/${LIB_NAME}"
lipo -info "${UNIVERSAL_DIR}/macos/${LIB_NAME}"

echo "==> lipo-fusing iOS simulator universal slice (arm64 + x86_64)"
mkdir -p "${UNIVERSAL_DIR}/ios-simulator"
lipo -create "${ios_sim_arm64}" "${ios_sim_x86_64}" \
    -output "${UNIVERSAL_DIR}/ios-simulator/${LIB_NAME}"
lipo -info "${UNIVERSAL_DIR}/ios-simulator/${LIB_NAME}"

echo "==> Creating xcframework at ${XCFRAMEWORK_PATH}"
mkdir -p "${OUTPUT_DIR}"
rm -rf "${XCFRAMEWORK_PATH}"
xcodebuild -create-xcframework \
    -library "${UNIVERSAL_DIR}/macos/${LIB_NAME}" \
    -headers "${HEADERS_DIR}" \
    -library "${ios_device_arm64}" \
    -headers "${HEADERS_DIR}" \
    -library "${UNIVERSAL_DIR}/ios-simulator/${LIB_NAME}" \
    -headers "${HEADERS_DIR}" \
    -output "${XCFRAMEWORK_PATH}"

echo "==> Done."
echo "    macOS:         cd swift && swift test"
echo "    iOS Simulator: cd swift && \\"
echo "                   SIM_ID=\$(xcrun simctl list devices available | grep -oE 'iPhone 1[5-9] \\([A-F0-9-]+\\)' | head -1 | grep -oE '[A-F0-9-]{36}') && \\"
echo "                   xcodebuild test -scheme Transcribe -destination \"id=\${SIM_ID}\""
