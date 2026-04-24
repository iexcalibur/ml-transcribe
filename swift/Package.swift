// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "Transcribe",
    platforms: [
        .macOS(.v13),
        .iOS(.v16),
    ],
    products: [
        .library(name: "Transcribe", targets: ["Transcribe"]),
        .executable(name: "inspect-model", targets: ["InspectModel"]),
    ],
    targets: [
        // Binary xcframework wrapping the Rust staticlib.
        // Built by: scripts/build-swift-ffi.sh
        .binaryTarget(
            name: "TranscribeFFIC",
            path: "artifacts/TranscribeFFI.xcframework"
        ),
        // Swift-idiomatic wrapper over the raw C symbols.
        .target(
            name: "Transcribe",
            dependencies: ["TranscribeFFIC"],
            path: "Sources/Transcribe"
        ),
        // CLI tool for inspecting a real HuggingFace safetensors model:
        //   swift run inspect-model /path/to/model.safetensors
        //   swift run inspect-model /path/to/model-dir   (sharded)
        .executableTarget(
            name: "InspectModel",
            dependencies: ["Transcribe"],
            path: "Sources/InspectModel"
        ),
        .testTarget(
            name: "TranscribeTests",
            dependencies: ["Transcribe"],
            path: "Tests/TranscribeTests"
        ),
    ]
)
